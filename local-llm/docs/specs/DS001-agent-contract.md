---
id: DS001
title: Agent Contract
status: accepted
owner: local-llm
summary: Manifest, process model, tools, authorization, the chat endpoint, and drain.
---

# DS001 Agent Contract

## Introduction

This specification defines how `local-llm` presents itself to Ploinky: the manifest, the processes inside the container, the MCP tools, who may call them, the OpenAI-compatible chat endpoint, and how the agent stops.

## Core Content

### Manifest

| Field | Value | Why |
| --- | --- | --- |
| `container` | `localhost/local-llm:dev` during development; a pinned digest once published | Built from `container-image-builds/images/local-llm` |
| `agent` | `exec node /code/src/main.mjs` | The controller is the container's main process |
| `readiness` | `{ "protocol": "mcp" }` | Ready when AgentServer answers MCP |
| `volumes` | `{ ".data/local-llm": "/data" }` | Weights, state and logs survive restarts |
| `llmRuntime.runtimePolicy.devices` | `[{ "type": "cdi", "value": "ploinky.local/gpu=all" }]` | The only device request; admitted only with an active Box GPU grant for this agent (DS003) |
| `routerAccess.agentPorts` | `false` | Closes the Router's agent-port relay, so runner ports are never reachable from a browser session |
| `ideSettings` | key `local-llm-settings`, scope `workspace`, plugin `local-llm/local-llm-settings`, `adminOnly: true` | Settings → Agents → Local LLMs |
| `endpoints.chatCompletions` | `node /code/src/chatResponder.mjs`, `supportsStream: true` | The model's only consumer-facing surface |

The manifest declares no `containerSecurity`, no published ports and no `llmRuntime.enabled`.

### Processes

```
main.mjs (controller, PID 1 under AgentEntrypoint.sh)
  ├─ control socket /dev/shm/local-llm/controller.sock (0700 directory, 0600 socket)
  ├─ AgentServer.mjs (MCP on 7000; spawns one short-lived process per tool call)
  └─ runner: llama-server or ollama serve, bound to 127.0.0.1 only
```

`main.mjs` owns all state. Tool processes and the chat responder are stateless clients of the control socket (`src/controlSocket.mjs`, one JSON request and one JSON reply per connection).

### Tools

Every tool is declared in `mcp-config.json` with `command: "node"`, `args: ["/code/tools/local_llm_tool.mjs"]`, `cwd: "/code"`, `tags: ["admin"]`, `env: { TOOL_NAME }` and a strict input schema (`additionalProperties: false`).

| Tool | Controller operation |
| --- | --- |
| `local_llm_overview` | `overview`: hardware, runners, models with per-runner download state and admission, current deployment |
| `local_llm_status` | `status`: deployment phase, progress and log lines after `sinceSeq` |
| `local_llm_run` | `run`: `requestId`, `modelId`, `runnerId`, `params`, `replace` |
| `local_llm_stop` | `stop` |
| `local_llm_download_cancel` | `cancelDownload`: keeps the partial file |
| `local_llm_weights_delete` | `deleteWeights` |
| `local_llm_model_add` / `_update` / `_remove` | registry edits; seed entries are read-only |
| `local_llm_test_prompt` | a smoke chat through AchillesAgentLib and the local Soul Gateway (below) |

### Authorization

Each tool call verifies the invocation grant with `authInfoFromInvocation` from `/Agent/lib/invocation-auth.mjs`. The caller must hold the `admin` role and must not be a guest. Anything else, including a missing grant, fails with `admin_required` before the controller is contacted. The `admin` tag also hides the tools from non-admin `tools/list` answers.

### Chat endpoint

AgentServer runs `src/chatResponder.mjs` for each `POST /v1/chat/completions` routed to the agent. The responder asks the controller for the ready deployment (`chatTarget`), keeps only OpenAI chat fields from the request, replaces `model` with the deployed model, and forwards to the runner on `127.0.0.1` with the runner's per-start API key. With no ready model it answers 503 `not_ready`. Streaming requests are piped through unchanged.

The Router lists the agent at `/api/router/openai-agent-discovery` because the manifest declares `endpoints.chatCompletions`. The workspace-local Soul Gateway turns it into a provider and model, and AchillesAgentLib callers reach it as `soul_gateway/<model>`. `local_llm_test_prompt` uses exactly that path, so request-time inference goes through AchillesAgentLib as the workspace requires.

### Drain

On SIGTERM, SIGINT or SIGHUP the controller:

1. stops accepting new commands and aborts the active job, which checkpoints a download as `paused` and keeps the `.partial` file and its identity sidecar;
2. stops the runner (SIGTERM, then SIGKILL after the grace period) and reaps it;
3. closes the control socket and stops AgentServer (SIGTERM, SIGKILL after 15 s);
4. exits 0, or exits 1 if the whole drain exceeds 30 s.

If AgentServer exits on its own, the controller stops the runner and exits non-zero, so Ploinky restarts the agent.

## Decisions & Questions

### Question #1: Why does the controller spawn AgentServer instead of the reverse?

Response: The controller must outlive tool calls and must be the process that receives SIGTERM, so it can checkpoint downloads and reap the runner before Ploinky's drain deadline. AgentServer spawns a fresh process per tool call and has no place for long-lived state.

### Question #2: Why is the runner API key passed on the command line?

Response: `llama-server` b11125 reads the key from `--api-key`, `LLAMA_API_KEY` or `--api-key-file`. The controller passes it as an argument and redacts it from the runner log. The key changes on every start, and only processes inside the container, which all run as the same user, can read another process's arguments or environment, so an environment variable or a key file would not narrow who can read it.

## Conclusion

The agent exposes admin-only tools backed by one serialized controller, publishes no ports, serves chat only through the Router and Soul Gateway, and drains within Ploinky's restart window.
