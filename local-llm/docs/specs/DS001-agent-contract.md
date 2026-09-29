---
id: DS001
title: Agent Contract
status: accepted
owner: local-llm
summary: Manifest, process model, tools, authorization, the chat endpoint, drain, and LM Studio's operator switch and limits.
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
| `profiles.default.env` | `HF_TOKEN`, `LOCAL_LLM_LMSTUDIO`, `LOCAL_LLM_MAX_COMPLETION_TOKENS`, `LOCAL_LLM_RUNNER_TIMEOUT_MS` and `LOCAL_LLM_VLLM_UNIFIED`, all `required: false` with no default | `HF_TOKEN` authenticates Hugging Face downloads (DS002). `LOCAL_LLM_LMSTUDIO=internal-use` is the operator's switch for LM Studio (below); unset, LM Studio is off. `LOCAL_LLM_VLLM_UNIFIED=experimental` is the operator's switch for vLLM on unified memory (below); unset, vLLM is off there. |

The manifest declares no other `containerSecurity` field (and needs a Ploinky that knows `shmSize`, from ploinky 9512fd30), no `llmRuntime` block (so no `runtimePolicy` device entry and no `llmRuntime.enabled`), and no published ports.

### Processes

```
main.mjs (controller, PID 1 under AgentEntrypoint.sh)
  ├─ control socket: a Linux abstract socket @local-llm-<random>, new on every start, and a per-start token
  ├─ AgentServer.mjs (MCP on 7000; spawns one short-lived process per tool call)
  └─ runner: llama.cpp's or ik_llama.cpp's llama-server, ollama serve, vLLM, TabbyAPI, or LM Studio's llmster
             (with its engine, a llama-server, and its workers), bound to 127.0.0.1 only, in its own process group
```

`main.mjs` owns all state. Tool processes and the chat responder are stateless clients of the control socket (`src/controlSocket.mjs`, one JSON request and one JSON reply per connection).

The control socket has two locks (runners plan, I5). Other agents in the Box share its `/dev/shm` and its uid 1000, so the socket is not a file there. It is an abstract socket, which exists only in this agent's own network namespace. On each start `main.mjs` also creates a random 32-byte token. It hands the token only to AgentServer, through the environment (`LOCAL_LLM_CONTROL_TOKEN`, next to `LOCAL_LLM_SOCKET`), so only the tools and the chat responder receive it. The controller refuses any request without the token (`unauthorized`) before an operation runs. The token is never written to disk or to a log. Runner processes get a minimal environment without it, and a token already in `main.mjs`'s own environment is dropped, never used. The token still holds when an agent runs with host networking. The two locks keep out other agents. They do not keep out processes inside this container. A runner runs as the same user in the same PID namespace, so it could read AgentServer's `/proc/<pid>/environ`. A runner could already reach the old socket, so this adds no exposure.

### Tools

Every tool is declared in `mcp-config.json` with `command: "node"`, `args: ["/code/tools/local_llm_tool.mjs"]`, `cwd: "/code"`, `tags: ["admin"]`, `env: { TOOL_NAME }` and a strict input schema (`additionalProperties: false`).

| Tool | Controller operation |
| --- | --- |
| `local_llm_overview` | `overview`: the hardware profile (DS005), hardware, runners (with `supported: false` and the reason for a runner this image does not include), the models offered in the profile with per-runner download state and admission, current deployment |
| `local_llm_status` | `status`: deployment phase, progress and log lines after `sinceSeq`, the hardware profile and, on unified memory, the memory guard's lowest MemAvailable, highest pressure and sample count |
| `local_llm_run` | `run`: `requestId`, `modelId`, `runnerId`, `params` (validated against the runner's schema for the profile), `replace`; a retry of an accepted `requestId` is answered as a duplicate. Every id the pattern allows, `__proto__`, `constructor` and `toString` included, is an ordinary key of the accepted-request records: fresh until accepted, recorded and persisted as data, never read from or written through a prototype. While no snapshot has shown a usable GPU, so no hardware profile is decided, it is refused with `admission_incompatible` and records nothing, so the same request can be sent again (DS005); a runner the image does not include is refused with `runner_unavailable` before anything downloads |
| `local_llm_stop` | `stop`; also cancels every Run submitted before it that has not started its job (below) |
| `local_llm_download_cancel` | `cancelDownload`: keeps the partial file; also cancels every Run submitted before it that has not started its job (below), and then succeeds; `not_downloading` when nothing is transferring and no Run was cancelled |
| `local_llm_weights_delete` | `deleteWeights` |
| `local_llm_model_add` / `_update` / `_remove` | registry edits; seed entries are read-only; an entry that carries a unified envelope (`unified`) or any `validated` label is refused with `invalid_model`, because only the trusted seed catalog may certify measurements (DS005) |
| `local_llm_test_prompt` | an admin smoke chat against the active runner on loopback; the one inference-routing exception (Question #3) |
| `local_llm_runner_install` | `installRunner`: `runnerId`, `acceptLicence`; installs an on-demand runner from the image's runner lock (DS004). Who accepted a licence comes from the Router-signed invocation, never from tool input. |
| `local_llm_runner_uninstall` | `uninstallRunner`: `runnerId`; refused while that runner runs a model |

**Stopping Runs that have not started.** Commands run one at a time, in the order they are submitted. A Run is pending from the moment it is submitted until its job starts: it may wait in the command queue behind other commands, and its own command then reads its hardware snapshots (the profile's while none is committed, and admission's) and looks for its files (DS002). A Stop or Cancel invalidates every Run submitted before it, both the one whose command is running and those still waiting in the queue. A Run submitted after the Stop or Cancel is new intent and goes ahead.

- **The running Run's step ends at once.** The snapshot or lookup in progress stops. A snapshot answers without waiting for its nvidia-smi query, which is killed, or for its free-disk read, which cannot be cancelled and whose late answer is dropped.
- **A queued Run reads and records nothing.** When its turn comes it fails at once.
- **Outcome.**
  - Every invalidated Run fails with `cancelled`, or with `shutting_down` when the drain met it in a snapshot.
  - It records nothing (no deployment, request id or saved parameters) and starts no download or runner, so the same request can be sent again.
  - A Stop succeeds. A Cancel that invalidated at least one Run of new work (a request id not already accepted) succeeds even when nothing is downloading. A queued retry of an accepted request is not new work: it is answered as a duplicate, never cancelled, and does not make a Cancel succeed.
  - A second Stop or Cancel right after finds nothing left of those Runs.
- **Answers that come first.** A request id that was already accepted is answered as a duplicate, as always, even if its retry was queued before a Stop. Busy and replace keep their meaning for Runs that go ahead.
- **Checks.** A Run checks for a Stop or Cancel after every step that waits, and last just before it records anything; from there to its job's start nothing waits.
- **The drain** stops the running Run the same way. A Run still queued at a drain gets its turn and refuses with `shutting_down` at its first snapshot, which the drain has already stopped, or at its last check, unless an earlier check refuses it first; it records nothing and starts nothing. A Run submitted after the drain began is refused at once with `shutting_down`.

### Authorization

AgentServer verifies the Router-signed invocation grant of every tool call (`requireVerifiedInvocation`) before the tool process starts. The tool then reads the caller's identity from that verified grant with `authInfoFromInvocation` from `/Agent/lib/invocation-auth.mjs`, which only normalizes it. The caller must hold the `admin` role and must not be a guest. Anything else, including a missing grant, fails with `admin_required` before the controller is contacted. The `admin` tag also hides the tools from non-admin `tools/list` answers.

### Chat endpoint

AgentServer runs `src/chatResponder.mjs` for each `POST /v1/chat/completions` routed to the agent. The responder asks the controller for the ready deployment (`chatTarget`), keeps only OpenAI chat fields from the request, replaces `model` with the deployed model, and forwards to the runner on `127.0.0.1`: llama.cpp with its per-start API key, Ollama and LM Studio without a key (Ollama has none, and LM Studio's authentication can only be turned on in its desktop app; both listen only on loopback inside the container, and the agent-port relay is closed). With no ready model it answers 503 `not_ready`. Streaming requests are piped through unchanged. A request may ask for one choice only (`n` 1, otherwise 400); `max_tokens` and `max_completion_tokens` must be positive integers and are clamped to the completion budget, and a request that sets neither is sent with `max_tokens` equal to it. The budget is 8,192 tokens unless the operator sets `LOCAL_LLM_MAX_COMPLETION_TOKENS` (1–32,768); the runner deadline is 570 s unless `LOCAL_LLM_RUNNER_TIMEOUT_MS` sets a shorter one (10–570 s), so the call always ends inside the endpoint's 600 s command limit. A value out of bounds is ignored. The caller's own deadline still applies (Soul Gateway: 120 s by default), so a longer budget is useful only with that deadline raised; 32,000-token answers are not a requirement of this release (decision 9). The forwarded fields are `messages`, `stream`, `stream_options`, the token limits, `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `n`, `tools`, `tool_choice`, `parallel_tool_calls`, `response_format`, `reasoning_effort`, `chat_template_kwargs`, `logprobs` and `top_logprobs`; a streaming request asks the runner for `usage` in its last chunk (`stream_options.include_usage`) unless the caller set it; the runner call is aborted after 570 s, inside the endpoint's 600 s command limit, with 504 `runner_timeout`.

The Router lists the agent at `/api/router/openai-agent-discovery` because the manifest declares `endpoints.chatCompletions`. The workspace-local Soul Gateway turns it into the model `local-llms/local-llm/default` (AgentServer's fallback `/v1/models` id), and AchillesAgentLib callers in other agents reach it as `soul_gateway/local-llms/local-llm/default`. That is the only way other agents use the model. The manifest's `capabilities.tags: ["local-llm"]` keeps the model out of the gateway's shared `generic-agent` group.

Soul Gateway refuses a call whose caller is the agent that the target model fronts ("Agent cannot call its own discovered Soul Gateway model"). The guard stays as it is. The admin test prompt therefore uses the exception in Question #3.

### Drain

On SIGTERM, SIGINT or SIGHUP the controller:

1. stops accepting new commands, aborts the active job, which checkpoints a download as `paused` and keeps the `.partial` file and its identity sidecar, and stops every hardware query in flight: the job's, a command's or the overview's. A snapshot runs up to three nvidia-smi queries in turn, each allowed 10 s, so none is waited out. The query in progress is killed (SIGKILL), and the snapshot answers at once, also while only its free-disk read is pending. The drain waits up to 1 s from its start for every killed query to exit (`HARDWARE_QUERY_REAP_MS`); one that has not exited by then (a hung driver) is named in the deployment log. A stopped snapshot is never used, so nothing is admitted or launched from it. The drain then waits up to 1 s for a command already in progress (a Run that reaches its start then refuses with `shutting_down`), and the query wait runs during that same second;
2. stops the runner's process group (SIGTERM, then SIGKILL after 3 s) and reaps it;
3. closes the control socket and stops AgentServer (SIGTERM; its own shutdown waits up to 20 s for in-flight tool calls, and it is killed after 21 s);
4. exits 0, or exits 1 if the whole drain exceeds 30 s. The worst case of steps 1–3 is 25 s, 5 s under that deadline and 10 s under Ploinky's 35 s restart window (`src/drainBudget.mjs`). A hardware snapshot in progress adds nothing to it: before, a SIGTERM during a Run's snapshot waited up to 30 s of nvidia-smi queries before AgentServer was even told to stop.

If AgentServer exits on its own, the controller stops the runner and exits non-zero, so Ploinky restarts the agent.

### LM Studio

LM Studio (DS000, DS004) is for internal use only, and off unless the operator sets `LOCAL_LLM_LMSTUDIO=internal-use`. While it is off, `local_llm_runner_install` and `local_llm_run` refuse it with `runner_disabled`, and `local_llm_overview` reports `enabled: false` with the reason. Changing the switch needs a restart of local-llm, because the environment is read when the container starts. On this platform an Explorer admin can set it too, through the admin-only WebTTY Box shell, so the switch prevents accidental use; it does not keep admins out.

### vLLM on unified memory

vLLM on a GPU that shares system memory (DS005) is experimental and off unless the operator sets `LOCAL_LLM_VLLM_UNIFIED=experimental` (`ploinky var LOCAL_LLM_VLLM_UNIFIED experimental`, then a restart of local-llm). While it is off there, `local_llm_runner_install` and `local_llm_run` refuse vLLM with `runner_disabled`, and `local_llm_overview` reports `enabled: false` with the reason; a dedicated GPU is not affected. No model entry, parameter or saved value can turn it on. Like the LM Studio switch it prevents accidental use; admins who can reach the Box shell can set it.

| Topic | Rule |
| --- | --- |
| Processes | The controller starts llmster from the runnable copy in its own process group, with `HOME` inside the copy (container-local, never `/data`). llmster starts its engine and workers in that group, so a stop, a drain or an exit ends them all. Before each start and after each exit, the controller also kills any process whose executable lies under LM Studio's copy. |
| Helper steps | Each runs as a short process in its own process group, with a minimal environment: `lms import --symbolic-link`; the SDK helper (`src/runners/lmStudioLoad.mjs`, the image's `/opt/local-llm/lmstudio-sdk`), which only loads the model, because LM Studio refuses an SDK client permission to start its server; and `lms server start --bind 127.0.0.1`, `lms ps --json` and `lms unload`, since `lms` is a privileged client. |
| Only our model | JIT loading cannot be turned off headlessly: a request to LM Studio's port that names another indexed model loads it. Two things keep LM Studio to our model. The chat responder replaces `model` and forwards only `/v1/chat/completions`, and LM Studio's port is reachable only inside the container. The controller also unloads anything but its own model: at load, and every 30 s while the model is ready (a watchdog; each unload is logged). |
| Calls home | Every llmster start makes 2–3 short HTTPS connections to lmstudio.ai: version checks for the daemon and `lms`, the feed, the backend list, the extension-pack check and LM Link status. None downloads anything (observed 2026-09-25). They cannot be turned off through a published setting and are accepted. The controller refuses to start LM Studio when llmster has staged an update in its home, and refuses any engine other than the pinned 2.41.0. |
| Logs | llmster's stdout echoes every request and response. A line filter keeps only lifecycle lines, so no prompt or response reaches the runner log. LM Studio's own server log, in its container-local home, is deleted before each start and after each exit. |

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
