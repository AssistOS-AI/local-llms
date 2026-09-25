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
| `container` | `docker.io/assistos/local-llm@sha256:b6c79af2aad08b542339346f4a985c84d5cbcacc31e8c6df45f951da7ba0fbf7`, pinned by digest; linux/amd64 only | Built and proven by `publish-local-llm-image.yml` from `container-image-builds/images/local-llm` at `95ba10a` (run 36131512550): llama.cpp b11159, Ollama 0.34.4, ik_llama.cpp `20f7a72` built for sm_86 and sm_89, the on-demand runner lock (vLLM 0.30.0 with gpt-oss's pinned `o200k_base.tiktoken`, TabbyAPI f07131c, and LM Studio's llmster 0.0.25-1, proprietary, which CI validates but never downloads) with uv 0.12.18 (DS004), and LM Studio's MIT SDK `@lmstudio/sdk` 2.0.0 in `/opt/local-llm/lmstudio-sdk`. The workflow proves the image contains no LM Studio file. A local build tagged `localhost/local-llm:dev` can stand in during development. |
| `agent` | `exec node /code/src/main.mjs` | The controller is the container's main process |
| `readiness` | `{ "protocol": "mcp" }` | Ready when AgentServer answers MCP |
| `volumes` | `{ ".data/local-llm": "/data" }` | Weights, state and logs survive restarts |
| `containerSecurity.shmSize` | `"8g"` | The agent's own `/dev/shm` (Ploinky gives every agent its own IPC namespace). vLLM keeps its inter-process sockets there, because the container's `/tmp` (fuse-overlayfs) cannot hold a usable socket, and PyTorch runners exchange tensors through it. It is a tmpfs limit, not memory set aside. |
| `containerSecurity.gpu` | `true` | Declares GPU access (Ploinky D14): Ploinky attaches the single CDI device `ploinky.local/gpu=all` when the Box's GPU wiring names this agent and is active; otherwise the agent starts without it and reports why (DS003) |
| `routerAccess.agentPorts` | `false` | Closes the Router's agent-port relay, so runner ports are never reachable from a browser session |
| `ideSettings` | key `local-llm-settings`, scope `workspace`, plugin `local-llm/local-llm-settings`, `adminOnly: true` | Settings → Agents → Local LLMs. The entry is a launcher: it opens the dashboard below and closes itself. |
| IDE plugin `local-llm-tool-button` (found in `IDE-plugins/`, not a manifest field) | `file-exp:toolbar`, `locationOrder` 295, `adminOnly: true`, `toolbarModal` `{mode: "component", component: "local-llm-dashboard"}` | The toolbar button, placed after Soul Gateway's. It opens the WebSkel dashboard `local-llm-dashboard` in Explorer's full-screen panel, which has cards for GPU, RAM, disk and the running model, and tabs for Models, Playground and Logs. The dashboard uses Explorer's tokens only, so it follows Explorer's light and dark themes. |
| `endpoints.chatCompletions` | `node /code/src/chatResponder.mjs`, `supportsStream: true` | The model's only consumer-facing surface |

The manifest declares no other `containerSecurity` field (and needs a Ploinky that knows `shmSize`, from ploinky 9512fd30), no `llmRuntime` block (so no `runtimePolicy` device entry and no `llmRuntime.enabled`), and no published ports.

### Processes

```
main.mjs (controller, PID 1 under AgentEntrypoint.sh)
  ├─ control socket: a Linux abstract socket @local-llm-<random>, new on every start, and a per-start token
  ├─ AgentServer.mjs (MCP on 7000; spawns one short-lived process per tool call)
  └─ runner: llama.cpp's or ik_llama.cpp's llama-server, or ollama serve, bound to 127.0.0.1 only, in its own process group
```

`main.mjs` owns all state. Tool processes and the chat responder are stateless clients of the control socket (`src/controlSocket.mjs`, one JSON request and one JSON reply per connection).

The control socket has two locks (runners plan, I5). Other agents in the Box share its `/dev/shm` and its uid 1000, so the socket is not a file there. It is an abstract socket, which exists only in this agent's own network namespace. On each start `main.mjs` also creates a random 32-byte token. It hands the token only to AgentServer, through the environment (`LOCAL_LLM_CONTROL_TOKEN`, next to `LOCAL_LLM_SOCKET`), so only the tools and the chat responder receive it. The controller refuses any request without the token (`unauthorized`) before an operation runs. The token is never written to disk or to a log. Runner processes get a minimal environment without it, and a token already in `main.mjs`'s own environment is dropped, never used. The token still holds when an agent runs with host networking. The two locks keep out other agents. They do not keep out processes inside this container. A runner runs as the same user in the same PID namespace, so it could read AgentServer's `/proc/<pid>/environ`. A runner could already reach the old socket, so this adds no exposure.

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
| `local_llm_test_prompt` | an admin smoke chat against the active runner on loopback; the one inference-routing exception (Question #3) |
| `local_llm_runner_install` | `installRunner`: `runnerId`, `acceptLicence`; installs an on-demand runner from the image's runner lock (DS004). Who accepted a licence comes from the Router-signed invocation, never from tool input. |
| `local_llm_runner_uninstall` | `uninstallRunner`: `runnerId`; refused while that runner runs a model |

### Authorization

AgentServer verifies the Router-signed invocation grant of every tool call (`requireVerifiedInvocation`) before the tool process starts. The tool then reads the caller's identity from that verified grant with `authInfoFromInvocation` from `/Agent/lib/invocation-auth.mjs`, which only normalizes it. The caller must hold the `admin` role and must not be a guest. Anything else, including a missing grant, fails with `admin_required` before the controller is contacted. The `admin` tag also hides the tools from non-admin `tools/list` answers.

### Chat endpoint

AgentServer runs `src/chatResponder.mjs` for each `POST /v1/chat/completions` routed to the agent. The responder asks the controller for the ready deployment (`chatTarget`), keeps only OpenAI chat fields from the request, replaces `model` with the deployed model, and forwards to the runner on `127.0.0.1`: llama.cpp with its per-start API key, Ollama without a key (it has none; it listens only on loopback inside the container and the agent-port relay is closed). With no ready model it answers 503 `not_ready`. Streaming requests are piped through unchanged. A request may ask for one choice only (`n` 1, otherwise 400); `max_tokens` and `max_completion_tokens` must be positive integers and are clamped to 8,192, and a request that sets neither is sent with `max_tokens` 8,192; the runner call is aborted after 570 s, inside the endpoint's 600 s command limit, with 504 `runner_timeout`.

The Router lists the agent at `/api/router/openai-agent-discovery` because the manifest declares `endpoints.chatCompletions`. The workspace-local Soul Gateway turns it into the model `local-llms/local-llm/default` (AgentServer's fallback `/v1/models` id), and AchillesAgentLib callers in other agents reach it as `soul_gateway/local-llms/local-llm/default`. That is the only way other agents use the model. The manifest's `capabilities.tags: ["local-llm"]` keeps the model out of the gateway's shared `generic-agent` group.

Soul Gateway refuses a call whose caller is the agent that the target model fronts ("Agent cannot call its own discovered Soul Gateway model"). The guard stays as it is. The admin test prompt therefore uses the exception in Question #3.

### Drain

On SIGTERM, SIGINT or SIGHUP the controller:

1. stops accepting new commands, waits up to 1 s for a command already in progress (a Run that reaches its start then refuses with `shutting_down`), and aborts the active job, which checkpoints a download as `paused` and keeps the `.partial` file and its identity sidecar;
2. stops the runner's process group (SIGTERM, then SIGKILL after 3 s) and reaps it;
3. closes the control socket and stops AgentServer (SIGTERM; its own shutdown waits up to 20 s for in-flight tool calls, and it is killed after 21 s);
4. exits 0, or exits 1 if the whole drain exceeds 30 s. The worst case of steps 1–3 is 25 s, 5 s under that deadline and 10 s under Ploinky's 35 s restart window (`src/drainBudget.mjs`).

If AgentServer exits on its own, the controller stops the runner and exits non-zero, so Ploinky restarts the agent.

## Decisions & Questions

### Question #1: Why does the controller spawn AgentServer instead of the reverse?

Response: The controller must outlive tool calls and must be the process that receives SIGTERM, so it can checkpoint downloads and reap the runner before Ploinky's drain deadline. AgentServer spawns a fresh process per tool call and has no place for long-lived state.

### Question #2: Why is the runner API key passed on the command line?

Response: `llama-server` b11159 reads the key from `--api-key`, `LLAMA_API_KEY` or `--api-key-file`. The controller passes it as an argument and redacts it from the runner log. The key changes on every start, and only processes inside the container, which all run as the same user, can read another process's arguments or environment, so an environment variable or a key file would not narrow who can read it.

### Question #3: How does the admin test prompt reach the model?

Response: Through a narrow exception to the workspace rule that request-time inference goes through AchillesAgentLib. It was approved by the user in the implementation session's question dialog on 2026-09-23, with its bounds set in a follow-up message in that session the same day, after Soul Gateway's self-call guard was observed to refuse the planned path (AchillesAgentLib from inside local-llm to `soul_gateway/local-llms/local-llm/default`). The exception covers `local_llm_test_prompt` only (`src/testPrompt.mjs`), with these bounds:

| Bound | How it is enforced |
| --- | --- |
| Admin only | The tool is tagged `admin` and checks for the `admin` role without `guest` before contacting the controller. |
| Only the active local runner | The target comes from the controller's `chatTarget`: the ready deployment's runner on `127.0.0.1` and its per-start key. The input selects no URL, model or key, and a non-loopback target is refused. |
| Bounded input and output | `prompt` 1–4,000 characters and `maxTokens` 1–1,024, in the tool schema and again in code; the returned text is capped at 16,000 characters. |
| Timeout | The runner call is aborted after 240 s, inside the tool's 300 s limit. |
| Same request shape | The request goes through the chat responder's field allowlist (`buildRunnerRequest`). |

The exception adds no network exposure: the runner stays on loopback behind `routerAccess.agentPorts: false`. Soul Gateway's self-call guard is unchanged, and other agents keep using the model only through AchillesAgentLib and Soul Gateway.

## Conclusion

The agent exposes admin-only tools backed by one serialized controller, publishes no ports, serves other agents only through the Router and Soul Gateway, makes one bounded admin-only loopback exception for its test prompt, and drains within Ploinky's restart window.
