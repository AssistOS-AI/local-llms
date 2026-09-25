---
id: DS000
title: Vision and Scope
status: accepted
owner: local-llm
summary: What local-llm is, how it differs from the legacy agents, its runners (LM Studio for internal use only), and what it deliberately leaves out.
---

# DS000 Vision and Scope

## Introduction

`local-llm` runs open-weight language models on the workspace's own NVIDIA GPU. A workspace administrator opens Local LLMs from Explorer's toolbar (or Settings → Agents → Local LLMs), picks a model and a runner, presses Run, and the agent downloads the weights, starts the runner on the GPU, and exposes the running model to other agents through the workspace-local Soul Gateway.

## Core Content

### Relation to the legacy agents

The repository's other twelve agents share the CPU image `assistos/local-llms`, contain no application source and start one fixed model each (repository DS001, DS002). `local-llm` is a separate agent with its own image (`container-image-builds/images/local-llm`), its own Node.js source under `local-llm/src/`, and its own tests under `local-llm/tests/`. The legacy agents, their image and their dispatcher are unchanged. The repository-level statement that agent directories hold no source code describes the legacy agents only.

### Goals

| Goal | Where |
| --- | --- |
| Weights are downloaded only after an explicit Run, never at enable, restart or when a model is added. | DS002 |
| The GPU is used only through the named-agent Box GPU grant; the agent asks for no privilege, capability or host device of its own. | DS003 |
| Every control operation is admin-only and goes through one serialized controller. | DS001 |
| Other agents use the running model through AchillesAgentLib and the local Soul Gateway, never through a runner port. | DS001 |
| Admission explains, before anything is downloaded, whether a model and its parameters fit this machine. | DS003 |

### Runners

| Runner | State in this release |
| --- | --- |
| llama.cpp `b11159` (CUDA 12.8 build) | Supported. GGUF from Hugging Face, pinned by commit, size and sha256. |
| ik_llama.cpp, commit `20f7a72` (built into the image with CUDA 12.8 for sm_86 and sm_89) | Supported. A llama.cpp fork with faster hybrid CPU/GPU inference for mixture-of-experts models. It reads the same GGUF files as llama.cpp, so the two share one download. |
| Ollama `0.34.4` | Supported. Library tags pulled by the Ollama daemon, verified by manifest digest when the catalog pins one. |
| vLLM `0.30.0` | Supported once an admin installs it from the image's runner lock (DS004). It reads Hugging Face snapshots (`hf`, DS002), such as the seed Qwen3-4B-AWQ, runs eager by default for a faster start, and takes only a model whose weights and KV cache fit the GPU unless the admin offloads weights to RAM (DS003). gpt-oss's tokenizer vocabulary is pinned in the lock and read from the runnable copy, not downloaded at run time (DS004). |
| TabbyAPI `f07131c` with ExLlamaV3 1.5.1 | Supported once an admin installs it from the image's runner lock after accepting TabbyAPI's AGPL-3.0 notice (DS004). It reads EXL3 snapshots (`exl3`, DS002), such as the seed Qwen3-8B EXL3 4.0 bpw, runs from its source directory in the runnable copy on loopback, and requires its per-start key. |
| LM Studio's llmster `0.0.25-1` (bundled CUDA 12 engine 2.41.0, llama.cpp b11026) | Internal use only. Off unless the deployment's operator turns it on (below). Once on, an admin installs it from the image's runner lock after accepting LM Studio's Terms (DS004). It reads the same GGUF files as llama.cpp. |

Every runner is an adapter in `src/runners/`: its identity, the weight format it reads, its loopback port and per-start key, its parameter schema and basic form fields, detection, its start-up pipeline (launch and readiness), the model name for chat requests, its admission policy and its log parser. The controller, the tools and the dashboard have no per-runner branches, so a new runner is an adapter plus data.

### LM Studio (internal use only)

LM Studio was first left out (runners plan, decision R5 = b). On 2026-09-25 the user asked for it to be installed on demand (decision I9, Phase R7). Its headless daemon, llmster, is proprietary. LM Studio's Terms (version August 23, 2026) allow use "solely for Your personal and / or internal business purposes" and forbid distributing it or using it "as an application service provider, or a software-as-a-service". So:

| Rule | How |
| --- | --- |
| Never in an image | The image's runner lock pins the llmster tarball by URL, size and sha256, and marks its licence proprietary. The publish workflow never downloads it and checks the image contains no LM Studio file. |
| Off by default | The runner is off unless the deployment's operator sets `LOCAL_LLM_LMSTUDIO=internal-use` (`ploinky var LOCAL_LLM_LMSTUDIO internal-use`, then a restart of local-llm). Otherwise Install and Run are refused with `runner_disabled`, and the dashboard says why. Deployments offered to other people keep it off. The switch prevents accidental use; it is not a boundary against admins. On this platform an Explorer admin can open the admin-only WebTTY Box shell and run the same command, and such an admin already has a shell in the Box. |
| Terms accepted first | Installing downloads llmster, and downloading counts as accepting the Terms. The Install dialog shows the Terms link, the version date, the internal-use, no-service and published-interfaces clauses, and a note that LM Studio's supported systems list Ubuntu while this agent is a Debian container. The acceptance is recorded with who and when (DS004). |
| Published interfaces only | The agent controls LM Studio through its `lms` CLI (import), its MIT SDK `@lmstudio/sdk` 2.0.0, which is pinned in the image (load, server), and its OpenAI-compatible HTTP API (chat). It also relies on a few things beyond those. llmster runs straight from the verified tarball rather than through LM Studio's `install.sh`. `HOME` points at a container-local directory. The SDK's experimental `llamaCppArgumentsOverride` sets the exact MoE layer count and threads. The controller reads the engine's command line to check its flags. And it deletes LM Studio's server log and its own imports. |
| Same engine, measured | Underneath is the same llama-server, an older build (b11026). On the RTX 3060 Laptop it ran gpt-oss-20b at the llama.cpp runner's settings with the same 4,776 MiB of GPU memory, at 35.4 tok/s against 38.4. |

The agent uses none of what LM Studio adds on top of llama.cpp: its native REST API, `/v1/responses`, `/v1/messages`, MCP and plugins stay unused, and only `/v1/chat/completions` is proxied (DS001). Just-in-time loading cannot be turned off headlessly, so the controller unloads any model but its own (DS003).

### Out of scope

A CPU fallback and a shared host model cache. Each needs its own decision.

## Decisions & Questions

### Question #1: Why a separate agent instead of extending the shared image?

Response: The shared image is CPU-only and arm64-first, starts one fixed model per agent, and its dispatcher is shell. GPU runners, on-demand weights, admission and a Settings UI need a long-running controller with state; keeping the legacy agents unchanged avoids regressions for their users.

## Conclusion

`local-llm` is an opt-in, per-workspace, GPU-backed model host that downloads nothing until an administrator runs a model, and serves other agents only through the local Soul Gateway.
