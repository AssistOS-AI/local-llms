---
id: DS003
title: GPU and Resources
status: accepted
owner: local-llm
summary: Manifest-declared GPU access and the operator override, admission, runner isolation, and memory estimates.
---

# DS003 GPU and Resources

## Introduction

This specification defines how `local-llm` reaches the GPU, how it decides whether a model fits before downloading anything, and how the runner processes are isolated. It replaces the operator-flag approach of repository DS010 with a grant that Ploinky enforces.

## Core Content

### GPU access through the manifest, with the operator override

The manifest declares `containerSecurity.gpu: true` (Ploinky decision D14), so the agent gets the GPU in any workspace where its repo is installed and the host has a usable NVIDIA GPU, with no host step. When the host prepares the Box (`ploinky start`, `restart`, `update`, `gpu grant`, `gpu revoke`) and the host has a usable NVIDIA GPU, it recreates that workspace's Box with the NVIDIA device nodes (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`), read-only binds of the driver libraries under `/usr/local/nvidia/lib64` and of `nvidia-smi` under `/usr/local/nvidia/bin`, a hookless CDI spec `ploinky.local/gpu=all`, and a grant marker naming the agent. The operator overrides the manifest on the host: `ploinky gpu revoke --agent local-llms/local-llm` denies it persistently, `ploinky gpu grant --agent local-llms/local-llm` restores it, `ploinky gpu revoke` without `--agent` turns manifest-declared GPU access off for the whole workspace, and `ploinky gpu status` shows each GPU agent with its source (Ploinky `ploinky-box/gpuGrant.mjs`).

The declaration asks for that CDI device and nothing else. Ploinky attaches it only while the Box's marker names the agent and is active, and checks again before every launch. Otherwise the agent still starts, without the device, and Ploinky passes `PLOINKY_GPU_STATUS=unavailable` and `PLOINKY_GPU_REASON`, which says what to run on the host: `ploinky gpu grant --agent local-llms/local-llm` after a revoke, `ploinky start` when the repo was installed after the Box was last prepared, or `ploinky gpu status` on a host without a usable GPU. The hardware snapshot reports that reason, so Settings, `local_llm_overview` and `local_llm_status` show it and every Run is refused as `incompatible` with it; there is no CPU fallback in this release (plan decision D7). The agent therefore never fails to start on a machine without a GPU, which matters because Explorer enables it by default. If the GPU disappears while the agent runs, the hardware snapshot reports it unavailable and every Run is refused as `incompatible`.

### Hardware snapshot

`src/controller/hardware.mjs` reads:

| Value | Source |
| --- | --- |
| GPU name, total, used and free memory, driver | `nvidia-smi --query-gpu` (path `/usr/local/nvidia/bin/nvidia-smi`, overridable by `LOCAL_LLM_NVIDIA_SMI` for tests) |
| Other GPU users | `nvidia-smi --query-compute-apps` |
| Total and available RAM | `/proc/meminfo` |
| Free disk under `/data` | `statfs` |

### Admission

`admit()` in `src/controller/admission.mjs` returns one of three answers, each with a reason and an estimate labelled `isEstimate: true`:

| Answer | Meaning | Example |
| --- | --- | --- |
| `ok` | The deployment fits now. A warning is attached when it needs more than half of the RAM available now. | gpt-oss-20b × llama.cpp at 16k context with 17 expert layers in RAM on a 6 GB GPU |
| `incompatible` | This machine can never run these parameters. The reason names the limit and the parameter to change. | 128k context with every expert on the GPU |
| `insufficient-now` | The hardware is enough, but memory or disk is held by something else at the moment. The reason names the other GPU users. | Ollama holding 4.5 GB of VRAM |

Estimates for llama.cpp come from the model's measured memory profile when the catalog has one (non-expert bytes, expert bytes per layer, KV bytes per token, fixed KV bytes), and from the file size otherwise. The gpt-oss-20b profile was measured on an RTX 3060 Laptop GPU (6 GB); for the default parameters the estimate is 4,851 MiB of GPU memory against 4,817 MiB measured. A 256 MiB GPU margin and a 1 GiB RAM margin are kept free. Admission runs when the overview is built, before a download starts, and again before the runner starts.

`admit()` makes the checks every runner shares (the runner is supported, the model has a source in the runner's format, a GPU is available) and then calls the runner adapter's own policy: `admitLlamaServer` for llama.cpp and ik_llama.cpp (the same memory layout for the same GGUF file), `admitLmStudio` for LM Studio, `admitOllama` for Ollama, `admitVllm` for vLLM, and `admitTabbyApi` for TabbyAPI. A runner the release lists but cannot run is `incompatible` with the reason its adapter gives. A runner this deployment's operator left off (LM Studio without its switch, DS001) is `incompatible` with the switch's reason, before any estimate.

`admitLmStudio` is the llama-server estimate plus 768 MiB of RAM for LM Studio's daemon and workers (about 0.6 GB measured at idle). It can reuse the llama-server model because LM Studio runs the same llama-server and the adapter sets every flag that affects memory. Through LM Studio's SDK it sets the context, parallel slots, unified KV, batch and micro-batch sizes, K and V cache types, flash attention on or off, mmap without mlock, and no context checkpoints. Through the SDK's argument override it sets the exact GPU layers, CPU MoE layers and threads. After each load the controller reads the engine's command line, and any flag that differs (the last value of a repeated flag counts), or an engine other than the pinned 2.41.0, fails the start with `runner_flags`. On the RTX 3060 Laptop, gpt-oss-20b at the llama.cpp runner's settings used 4,776 MiB of GPU memory on LM Studio, the same as on llama.cpp.

`admitVllm` keeps all weights on the GPU unless the admin sets `cpuOffloadGb`, which moves that many GiB of weights to system RAM with a warning that generation becomes much slower. It sizes the KV cache for `maxModelLen` from the model's `kvBytesPerToken` (halved for an fp8 KV cache) and adds 768 MiB for activations, the CUDA context and allocator slack, measured with Qwen3-4B-AWQ. CUDA reports about 94 % of nvidia-smi's total as usable. When weights, KV cache and overhead exceed 90 % of that, the model is `incompatible`; the reason names `maxModelLen`, a smaller model, or the least `cpuOffloadGb` that would fit. The GPU share vLLM gets (`--gpu-memory-utilization`) is the admin's value, or else what is free now less two margins, at most 0.9. If that share or the need exceeds what is free now, the answer is `insufficient-now`, naming the other GPU users. With `cpuOffloadGb` set, the system RAM estimate is (1.81 × the offload + 3.0 GiB) × 1.15. It comes from two runs of gpt-oss-20b with 10 GiB offloaded on the RTX 3060 machine (31 GiB of RAM, its 8 GiB swap already full). On 8080, available memory fell from 23.2 to 2.2 GiB, and `free` showed shared memory grow by about 18.1 GiB (1.81 × the offload: vLLM's pinned copies, more than any single process's RssShmem shows). The engine and API server held about 3.0 GiB more. In the fixture, available memory fell by at least 20.1 GiB before a guard stopped the run. An offload run must leave max(4 GiB, 10 % of RAM) available. When the estimate exceeds the available memory less that floor, the answer is `insufficient-now`; the reason names the estimate, what is available and the floor. When it exceeds total RAM less the floor, `incompatible`. The speed warning stays. On the 31 GiB machine, gpt-oss-20b (at least 9 GiB of offload on 6 GB) therefore needs 26.2 GiB available, so it is refused there in practice. The admission records the floor (`ramFloorBytes`), and the controller keeps a memory guard as the backstop. While such a runner loads, it samples MemAvailable every 2 s, and every 10 s once the runner is ready. If available memory falls below the floor, it stops the runner at once (SIGTERM, then SIGKILL within the drain's 3 s grace), and the deployment ends in `error` with "stopped: host memory below the floor (X GiB available, Y GiB required)". The first estimate (offload + 3 GiB, 13.0 GiB for 10 GiB) admitted a run that took the host down to about 2.2 GiB available with swap full. Without offload, the estimate stays 3 GiB with a 1 GiB margin, and there is no guard.

`admitTabbyApi` keeps all EXL3 weights on the GPU and sizes the KV cache for `cacheSize` tokens (the max sequence length unless set) at the `cacheMode` precision (Q8 a half, Q6 three eighths, Q4 a quarter of FP16), plus an estimated 640 MiB for the CUDA context and one prompt chunk's activations. ExLlamaV3 keeps the model's input embedding (`memory.embeddingBytes`) in system RAM, so that counts toward RAM, not the GPU. It is `incompatible` above the usable share of the GPU or the machine's RAM, and `insufficient-now` above what is free now.

### Runner isolation

| Rule | How |
| --- | --- |
| Loopback only | llama.cpp and ik_llama.cpp get `--host 127.0.0.1`; Ollama gets `OLLAMA_HOST=127.0.0.1:<port>` |
| Fixed ports | Each adapter declares its port; `defaultPorts()` refuses two runners on one port. llama.cpp 18080, ik_llama.cpp 18081, vLLM 18082, TabbyAPI 18083, LM Studio 18084 (its daemon also listens on 127.0.0.1:41343, and its engine on a random loopback port), Ollama 18434, all inside the agent's network namespace |
| Not reachable through the Router | `routerAccess.agentPorts: false` closes the agent-port relay for this agent |
| Per-start credentials | llama.cpp and ik_llama.cpp get a fresh random `--api-key` on every start, and vLLM and TabbyAPI a key of their own; the chat responder reads it from the controller. LM Studio has none headless: its server relies on loopback, the agent's own network namespace and the closed relay, and the controller keeps only its own model loaded (DS001) |
| Minimal environment | Runners get `PATH`, `HOME=/data/home`, `LANG` and their own variables only; no agent secrets |
| Reaped on stop and drain | Each runner leads its own process group. Stop sends SIGTERM to the group, then SIGKILL after the grace period; when the runner exits, anything left in its group is killed, so helper processes a runner starts never outlive it |

### Reuse from repository DS010

| DS010 rule | local-llm |
| --- | --- |
| RAM floor checked at startup from a catalog band | Replaced by per-deployment admission against measured or estimated needs |
| GPU passthrough is an operator runtime flag; the manifest cannot express it | Replaced by the manifest declaration `containerSecurity.gpu` with the operator's grant and revoke as override (DS010 Question #1, option 3: manifest field plus operator control) |
| `nvidia-smi` and `/dev/nvidia0` as GPU signals | Kept as the hardware snapshot source, read from the granted paths |
| One loaded model at a time to limit contention | Kept: one deployment at a time; Ollama runs with `OLLAMA_MAX_LOADED_MODELS=1` |

## Decisions & Questions

### Question #1: Why is the memory estimate profile-based instead of computed from GGUF metadata?

Response: The GGUF header gives tensor sizes but not the CUDA compute buffers or the runtime's allocation behaviour, which differ by build and flash-attention mode. A profile measured on real hardware and stored with the catalog entry is accurate for the seed model; the file-size heuristic is the fallback for user models and is labelled as such.

## Conclusion

The agent reaches the GPU through its manifest declaration, which the operator can revoke, refuses before downloading when a model cannot fit, keeps runners on loopback behind a per-start key, and never exposes a runner port through the Router.
