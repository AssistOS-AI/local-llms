---
id: DS000
title: Vision and Scope
status: accepted
owner: local-llm
summary: What local-llm is, the retired shared-image agents and the services that went with them, its runners (LM Studio for internal use only), and what it deliberately leaves out.
---

# DS000 Vision and Scope

## Introduction

`local-llm` runs open-weight language models on the workspace's own hardware: on its NVIDIA GPU when one is attached and usable, and otherwise on the CPU (the `cpu` profile, DS005). A workspace administrator opens Local LLMs from Explorer's toolbar (or Settings → Agents → Local LLMs), picks a model and a runner, and presses Run. The agent then downloads the weights, starts the runner on the GPU or the CPU, and exposes the running model to other agents through the workspace-local Soul Gateway.

## Core Content

### The retired shared-image agents

Until 2026-09-28 the repository also held twelve legacy agents that shared the CPU image `assistos/local-llms`, contained no application source and started one fixed model each: `local-llms-manager` (model registration and profile lifecycle) and eleven model agents, `language-translation`, `relevance`, `function-selection`, `function-invocation`, `tool-composition-local`, `base-local`, `local`, `planning-local`, `validated-planning-local`, `adaptive-local` and `coding-local`. They were retired with their image definition (`Dockerfile`), `scripts/`, `catalog/`, `tests/validate.mjs`, their specifications (repository `docs/specs` DS000–DS011) and their HTML documentation (owner's decision 10 of the multi-arch plan). Their last revision is commit [`03697765`](https://github.com/AssistOS-AI/local-llms/tree/03697765a35d92adf4c7db1be9616c0b5a64cb63), where the code and specifications remain readable. local-llm replaces none of them with a compatibility layer.

What disappears with them:

| Service | Was | Replacement |
| --- | --- | --- |
| Translation | `language-translation`: a transformers seq2seq service (facebook/m2m100_418M) with its own translation API | none |
| Relevance scoring and reranking | `relevance`: a cross-encoder / sentence-transformers scoring API (Qwen/Qwen3-Reranker-0.6B) | none |
| CPU-only chat models | the ten role agents, each an Ollama or llama.cpp model on the CPU in the shared image | local-llm's `cpu` profile (DS005): models run on the CPU automatically when no NVIDIA GPU is usable, any GGUF model that fits the machine's memory; no role agent is recreated |
| Model and profile management | `local-llms-manager`'s registration, startup-script and profile tools | local-llm's own catalog, registry and tools (DS001, DS002) |

A workspace that enabled a legacy agent loses it at its next repository update: Ploinky can no longer find the agent's manifest. No stored data is deleted by this: models the legacy agents downloaded stay under the workspace's `.ploinky/data/local-llms/`, and local-llm's own models and state under its `/data` are untouched. The `assistos/local-llms` image is no longer built from this repository; already published tags are not removed.

### Goals

| Goal | Where |
| --- | --- |
| Weights are downloaded only after an explicit Run, never at enable, restart or when a model is added. | DS002 |
| The GPU is used only through the named-agent Box GPU grant; the agent asks for no privilege, capability or host device of its own. | DS003 |
| A machine without a usable NVIDIA GPU runs models on the CPU, with no operator step. | DS005 |
| Every control operation is admin-only and goes through one serialized controller. | DS001 |
| Other agents use the running model through AchillesAgentLib and the local Soul Gateway, never through a runner port. | DS001 |
| Admission explains, before anything is downloaded, whether a model and its parameters fit this machine. | DS003 |

### Runners

| Runner | State in this release |
| --- | --- |
| llama.cpp `b11159` (amd64: CUDA 12.8 build; arm64: ggml-org's CUDA 13.4 build with native sm_121a code for NVIDIA GB10) | Supported. GGUF from Hugging Face, pinned by commit, size and sha256. The default runner on a GPU that shares system memory (DGX Spark), for the listed models and models added at run time, sized by a labelled estimate or a trusted envelope (DS005); no per-model benchmark is required. On a machine without a usable NVIDIA GPU it runs on the CPU (`cpu` profile, DS005). The arm64 image's build carries CPU backends beside its CUDA backend; the amd64 build is made with the same release flags. |
| ik_llama.cpp, commit `20f7a72` (built into the image with CUDA 12.8 for sm_86 and sm_89) | Supported. A llama.cpp fork with faster hybrid CPU/GPU inference for mixture-of-experts models. It reads the same GGUF files as llama.cpp, so the two share one download. |
| Ollama `0.34.4` | Supported. Library tags pulled by the Ollama daemon, verified by manifest digest when the catalog pins one. |
| vLLM `0.30.0` | Supported once an admin installs it from the image's runner lock (DS004). It reads Hugging Face snapshots (`hf`, DS002), such as the seed Qwen3-4B-AWQ, runs eager by default for a faster start, and takes only a model whose weights and KV cache fit the GPU unless the admin offloads weights to RAM (DS003). gpt-oss's tokenizer vocabulary is pinned in the lock and read from the runnable copy, not downloaded at run time (DS004). On DGX Spark it is experimental: off unless the operator sets `LOCAL_LLM_VLLM_UNIFIED=experimental` (DS001, DS005), and sized as a share of the shared pool. |
| TabbyAPI `f07131c` with ExLlamaV3 1.5.1 | Supported once an admin installs it from the image's runner lock after accepting TabbyAPI's AGPL-3.0 notice (DS004). It reads EXL3 snapshots (`exl3`, DS002), such as the seed Qwen3-8B EXL3 4.0 bpw, runs from its source directory in the runnable copy on loopback, and requires its per-start key. |
| LM Studio's llmster `0.0.25-1` (bundled CUDA 12 engine 2.41.0, llama.cpp b11026) | Internal use only. Off unless the deployment's operator turns it on (below). Once on, an admin installs it from the image's runner lock after accepting LM Studio's Terms (DS004). It reads the same GGUF files as llama.cpp. |

The image decides which runners exist (DS005): the amd64 image has all of them; the arm64 image, for NVIDIA GB10 in DGX Spark, has llama.cpp and a runner lock with vLLM (its aarch64 wheels, installed on demand and experimental there), so ik_llama.cpp, Ollama, TabbyAPI and LM Studio are reported "not available on this platform" and refused before anything downloads. In the `cpu` profile only the runners with a CPU policy run (llama.cpp); every other runner is refused before anything downloads (ik_llama.cpp links the CUDA driver library directly).

Every runner is an adapter in `src/runners/`: its identity, the weight format it reads, its loopback port and per-start key, its parameter schema and basic form fields, detection, its start-up pipeline (launch and readiness), the model name for chat requests, its admission policy and its log parser. The controller, the tools and the dashboard have no per-runner branches, so a new runner is an adapter plus data.

### LM Studio (internal use only)

LM Studio was first left out (runners plan, decision R5 = b). On 2026-09-25 the user asked for it to be installed on demand (decision I9, Phase R7). Its headless daemon, llmster, is proprietary. LM Studio's Terms (version August 23, 2026) allow use "solely for Your personal and / or internal business purposes" and forbid distributing it or using it "as an application service provider, or a software-as-a-service". So:

| Rule | How |
| --- | --- |
| Never in an image | The image's runner lock pins the llmster tarball by URL, size and sha256, and marks its licence proprietary. The publish workflow never downloads it and checks the image contains no LM Studio file. |
| Off by default | The runner is off unless the deployment's operator sets `LOCAL_LLM_LMSTUDIO=internal-use` (`ploinky var LOCAL_LLM_LMSTUDIO internal-use`, then a restart of local-llm). Otherwise Install and Run are refused with `runner_disabled`, and the dashboard says why. Deployments offered to other people keep it off. The switch prevents accidental use; it is not a boundary against admins. On this platform an Explorer admin can open the admin-only WebTTY Box shell and run the same command, and such an admin already has a shell in the Box. |
| Terms accepted first | Installing downloads llmster, and downloading counts as accepting the Terms. The Install dialog shows the Terms link, the version date, the internal-use, no-service and published-interfaces clauses, and a note that LM Studio's supported systems list Ubuntu while this agent is a Debian container. The acceptance is recorded with who and when (DS004). |
| Published interfaces only | The agent controls LM Studio through its `lms` CLI (import, starting the server on 127.0.0.1, listing and unloading models), its MIT SDK `@lmstudio/sdk` 2.0.0, which is pinned in the image (the load with every memory flag; an SDK client may load but not start the server), and its OpenAI-compatible HTTP API (chat). It also relies on a few things beyond those. llmster runs straight from the verified tarball rather than through LM Studio's `install.sh`. `HOME` points at a container-local directory. The SDK's experimental `llamaCppArgumentsOverride` sets the exact MoE layer count and threads. The controller reads the engine's command line to check its flags. And it deletes LM Studio's server log and its own imports. |
| Same engine, measured | Underneath is the same llama-server, an older build (b11026). On the RTX 3060 Laptop it ran gpt-oss-20b at the llama.cpp runner's settings with the same 4,776 MiB of GPU memory, at 35.4 tok/s against 38.4. |

The agent uses none of what LM Studio adds on top of llama.cpp: its native REST API, `/v1/responses`, `/v1/messages`, MCP and plugins stay unused, and only `/v1/chat/completions` is proxied (DS001). Just-in-time loading cannot be turned off headlessly, so the controller unloads any model but its own (DS003).

### Out of scope

Translation and reranking services (retired above), a shared host model cache, and Apple GPU acceleration through Podman libkrun/Vulkan (deferred by the user 2026-09-30, to be added later). Each needs its own decision.

## Decisions & Questions

### Question #1: Why a separate agent instead of extending the shared image?

Response: The shared image was CPU-only and arm64-first, started one fixed model per agent, and its dispatcher was shell. GPU runners, on-demand weights, admission and a Settings UI need a long-running controller with state, so local-llm began as a separate agent beside the legacy ones, and the legacy agents were later retired (above).

### Question #2: Why does local-llm run models on the CPU when no NVIDIA GPU is usable?

Response: Plan decision D7 left CPU inference out. That made the agent unusable on a Mac (through its Podman machine) and on hosts without a GPU, although Explorer enables it by default. On 2026-09-30 the user decided that an NVIDIA GPU may not be a prerequisite. The CPU path is automatic, with no operator switch, and is sized from the machine's RAM (DS005). Revoking the GPU grant therefore no longer stops local-llm from serving models: it moves them to the CPU.

## Conclusion

`local-llm` is an opt-in, per-workspace model host. It uses the NVIDIA GPU when one is attached and the CPU otherwise, downloads nothing until an administrator runs a model, and serves other agents only through the local Soul Gateway.
