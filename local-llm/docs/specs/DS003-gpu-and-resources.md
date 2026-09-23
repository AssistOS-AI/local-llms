---
id: DS003
title: GPU and Resources
status: accepted
owner: local-llm
summary: The Box GPU grant, admission, runner isolation, and memory estimates.
---

# DS003 GPU and Resources

## Introduction

This specification defines how `local-llm` reaches the GPU, how it decides whether a model fits before downloading anything, and how the runner processes are isolated. It replaces the operator-flag approach of repository DS010 with a grant that Ploinky enforces.

## Core Content

### GPU access through the Box grant

An operator grants the GPU to this agent on one workspace with `ploinky gpu grant nvidia --agent local-llms/local-llm` (Ploinky `ploinky-box/gpuGrant.mjs`; `ploinky gpu status` shows the grant). The grant recreates that workspace's Box with the NVIDIA device nodes (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`), read-only binds of the driver libraries under `/usr/local/nvidia/lib64` and of `nvidia-smi` under `/usr/local/nvidia/bin`, a hookless CDI spec `ploinky.local/gpu=all`, and a grant marker naming the agent.

The manifest asks for that CDI device and nothing else. Ploinky admits the request only for the named agent while the marker is valid, and checks again before every launch. Without an active grant naming the agent, or with a stale grant, Ploinky refuses to start the agent container and names the grant command; there is no CPU fallback in this release (plan decision D7). If the GPU disappears while the agent runs, the hardware snapshot reports it unavailable and every Run is refused as `incompatible`.

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

vLLM and LM Studio are always `incompatible` in this release, with a stated reason.

### Runner isolation

| Rule | How |
| --- | --- |
| Loopback only | llama.cpp gets `--host 127.0.0.1`; Ollama gets `OLLAMA_HOST=127.0.0.1:<port>` |
| Fixed ports | llama.cpp 18080, Ollama 18434, both inside the agent's network namespace |
| Not reachable through the Router | `routerAccess.agentPorts: false` closes the agent-port relay for this agent |
| Per-start credentials | llama.cpp gets a fresh random `--api-key` on every start; the chat responder reads it from the controller |
| Minimal environment | Runners get `PATH`, `HOME=/data/home`, `LANG` and their own variables only; no agent secrets |
| Reaped on stop and drain | SIGTERM, then SIGKILL after the grace period |

### Reuse from repository DS010

| DS010 rule | local-llm |
| --- | --- |
| RAM floor checked at startup from a catalog band | Replaced by per-deployment admission against measured or estimated needs |
| GPU passthrough is an operator runtime flag; the manifest cannot express it | Replaced by the Box GPU grant and the CDI device request (DS010 Question #1, option 3: manifest field plus operator grant) |
| `nvidia-smi` and `/dev/nvidia0` as GPU signals | Kept as the hardware snapshot source, read from the granted paths |
| One loaded model at a time to limit contention | Kept: one deployment at a time; Ollama runs with `OLLAMA_MAX_LOADED_MODELS=1` |

## Decisions & Questions

### Question #1: Why is the memory estimate profile-based instead of computed from GGUF metadata?

Response: The GGUF header gives tensor sizes but not the CUDA compute buffers or the runtime's allocation behaviour, which differ by build and flash-attention mode. A profile measured on real hardware and stored with the catalog entry is accurate for the seed model; the file-size heuristic is the fallback for user models and is labelled as such.

## Conclusion

The agent reaches the GPU only through an operator's named grant, refuses before downloading when a model cannot fit, keeps runners on loopback behind a per-start key, and never exposes a runner port through the Router.
