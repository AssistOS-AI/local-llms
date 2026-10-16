---
title: DS005-hardware-profiles
summary: The hardware profiles the agent runs (NVIDIA dedicated and unified, CPU, discrete AMD Vulkan, experimental shared-memory APU, Apple Venus), how the container's limits narrow them, the memory guard, the GPU share and its vLLM qualification, and the dashboard; the engine's detection, admission and estimates are in the pinned pworker specification.
---

# DS005 Hardware Profiles

## Introduction

One agent, `local-llms/local-llm`, chooses a hardware profile from the attached GPU and runtime contract. NVIDIA GPUs keep the dedicated or unified CUDA policy. A discrete AMD Radeon GPU uses Vulkan RADV. An AMD integrated GPU can use the experimental APU policy behind an operator switch. An Apple GPU uses Vulkan Venus inside a libkrun Podman machine. A host with no usable GPU uses the CPU. Each policy has its own memory accounting and safety rules. The JavaScript is the same on every architecture; only the image's binaries and runner lock differ. The profiles themselves (detection, the choice of profile, each policy's parameters, estimates and admission, and the memory guard's engine) belong to the engine in the Ploinky-Worker repository (pworker), which this repository pins (DS001), and are specified in [pworker DS005](https://github.com/AssistOS-AI/Ploinky-Worker/blob/64aa8a0c0e4e918c05f3f0264b45438652f67a6c/docs/specs/DS005-hardware-profiles.md) at commit `64aa8a0c0e4e918c05f3f0264b45438652f67a6c`. This specification keeps what is the agent's: the profile table, what the container's limits change in each profile, the memory guard as the agent runs it, the Ploinky GPU share and its vLLM qualification, the operator switches and the dashboard.

## Core Content

### Profiles

| Profile | GPU memory | Memory the models compete for |
| --- | --- | --- |
| `dedicated` | nvidia-smi reports total, used and free memory in MiB | GPU memory for the offloaded weights, KV cache and buffers; system RAM for the rest (DS003) |
| `unified` | the GPU shares system memory | one pool: MemTotal and MemAvailable from `/proc/meminfo` |
| `cpu` | none is used: no NVIDIA GPU is attached, or the attached one cannot be used | one pool: MemTotal and MemAvailable from `/proc/meminfo`, capped by a container memory limit when one is set |
| `vulkan` | RADV discrete Radeon, with numeric device memory | Dedicated GPU estimates plus system RAM; unit-tested and unverified on AMD hardware |
| `venus` | Apple GPU reached through Venus, outside the VM's memory accounting | Provisional device cap plus a conservative VM staging pool |
| `apu` | Integrated RADV, `memoryModel: shared`, default off and unqualified | One physical RAM pool for device and host bytes, with separate mapping ceilings and no finite or unknown container budget |

The dedicated profile keeps the engine's admission and defaults (pworker DS003 and DS005), narrowed by the container's limits (DS003 here). The `cpu` profile is chosen automatically; no operator switch, catalog entry or parameter turns it on or off.

### Where the engine specifies the profiles

| Topic | Specification |
| --- | --- |
| Detecting the memory model, Vulkan and Venus detection, the unified-memory table and the states of an unavailable GPU | pworker DS005 and DS003 |
| Choosing the profile with `decideProfile`, the 60 s wait for an unreadable GPU, the committed profile | pworker DS005 |
| llama.cpp, Ollama and vLLM on unified memory, on the CPU and on Vulkan; CPU, Venus and unified admission with their reserves, floors and constants | pworker DS005 |
| The APU policy, its constants, the constraints C1, C2 and C4, its offload proof, and the activity and guard observations | pworker DS005 |
| CPU threads and performance cores | pworker DS005 (the quota cap is below) |
| Venus measurement status and the CPU fallback explanation | pworker DS005 |

The engine's text is the one this specification used to hold, with the container's limits taken out. Where it says "the host's limits", this repository's plug-in is meant (DS003).

### Operator switches and wording

The switches are read from the controller's environment when they matter, and no model entry, parameter or saved value can set one (DS001): `LOCAL_LLM_AMD_APU=experimental` enables the APU profile (an integrated GPU otherwise selects `cpu`, cause `unusable`), `LOCAL_LLM_VLLM_UNIFIED=experimental` enables vLLM on unified memory (`ploinky var LOCAL_LLM_VLLM_UNIFIED experimental`, then a restart of local-llm; DS001, DS004), `LOCAL_LLM_LMSTUDIO=internal-use` enables LM Studio (DS000), and `LOCAL_LLM_APU_GUARD_FLOOR_BYTES` raises the APU guard's runtime floor. The last is an optional raise-only test setting: a decimal safe integer at least as large as the admission floor sets the runtime floor to their maximum, malformed, unsafe or lower values are ignored with a warning in the overview and log, and it never changes admission, `estimate.floorBytes` or the dashboard floor. A floor above MemTotal exercises the runtime kill without making the Run incompatible at admission. The sentences that tell an operator what to run (`ploinky var ...`, `ploinky gpu grant --agent local-llms/local-llm`, a restart of local-llm) are this agent's wording, in `src/ploinkyMessages.mjs`; the engine's own defaults are neutral, and `tests/ploinky-wording.test.mjs` holds every key to the text it replaced.

### Runner availability and the image's GPUs

The image says what it contains: `/opt/local-llm/source.contract` (key=value lines). When it exists, a runner is available only if its executable is in the image (llama.cpp, ik_llama.cpp, Ollama) or the image's runner lock lists it (vLLM, TabbyAPI, LM Studio; DS004). An unavailable runner is `supported: false` in the overview with "not available on this platform: this image does not include it", and `local_llm_run` refuses it with `runner_unavailable` before anything downloads. Without a contract (tests, development) every runner counts as available. The arm64 image's contract names `gpu_compute_capabilities=12.1`, the GPUs its CUDA runners were built for; a GPU that reports another compute capability decides the `cpu` profile, and the Compute card and every CPU admission say why. This changes today's refusal on arm64 hosts with other NVIDIA GPUs, for example Thor or GH200. A GPU that does not report its capability counts as unreadable (Choosing the profile). The amd64 image names none and nothing is checked there. Phase 3: on the `cpu` profile the agent's runner lock (DS004) may also make a runner available, Ollama on arm64 and llama.cpp's CPU build on both architectures, when the runner has a CPU policy and the image does not already hold its executable. On the dedicated and unified profiles it makes none, so those profiles show the runners they always did, and a runner that names the profiles it serves (llama.cpp's CPU build serves `cpu`) is not listed on any other.

### What the container's limits change

The engine applies no limit. These are the effects of the limits plug-in (`src/hardwareLimits.mjs`, DS003) on the profiles, using the engine's hook names (pworker DS007).

| Profile | Effect |
| --- | --- |
| `cpu` | The pool (MemTotal and MemAvailable from `/proc/meminfo`) is at most the container's memory limit (cgroup `memory.max`), and what is available now is at most the limit less `memory.current` (`memoryPool`), with the warning that a container memory limit applies. Admission refuses, temporarily, a finite limit whose current use cannot be read. The default thread count is capped by the cgroup CPU quota (`capCores`): on the 4-CPU Podman machine of a 16 GiB Mac the default is 3 (4 minus 1) because no CPU quota is set there. |
| `unified` | The default threads are capped by the CPU quota. Absolute RAM admission remains capped by a finite container budget, while the experimental vLLM utilization keeps the physical MemTotal as its denominator. |
| `venus` | The VM staging reserve applies the CPU pool's host reserve and floor and any finite container limit. |
| `vulkan` | A configured GPU share is refused with `gpu_share_unsupported`, and Vulkan runners receive no MPS environment. |
| `apu` | A finite container budget is not proven to constrain shared GPU pages, so it is refused (below). |

The APU policy's constraint C3 is the agent's. A finite container memory limit is a permanent refusal (`refuseProfile`, constraint C3: not proven to constrain shared GPU pages), and an unknown budget gives the temporary `budget_unreadable`. At discovery, a finite or unknown cgroup memory budget replaces the GPU with an unusable one, so the profile falls back to the CPU before launch (`decorateSnapshot`); the same C3 refusal repeats at admission, and the pooled guard stops a running APU runner whose budget becomes finite or unknown (`guardBreach`, below). Container limits have not been proven to charge GTT allocations.

### Memory guard

On unified memory and on the CPU every runner is watched from its start to its exit: effective available RAM is sampled every 250 ms: host MemAvailable, additionally bounded by effective total and max(0, memory.max - memory.current) under a known finite container limit. Known zero means no headroom; unreadable required usage is a separate failure. Admission rejects a temporary shortage before model launch and can succeed after memory is released. Under a known finite limit, two consecutive unreadable `memory.current` samples (at most 500 ms) kill the runner's process group with `budget_unreadable`, and a new Run is refused on the first unreadable admission snapshot. A finite-budget guard also treats missing or malformed physical RAM readings as unreadable instead of deriving them from the cgroup cap. Existing immediate MemAvailable failures still stop pooled runners immediately; other required finite-budget read failures stop an already guarded runner after two consecutive samples with `budget_unreadable`. Sampling continues whether the runner is loading, ready or stopping (a graceful Stop or Replace keeps it watched until the process has exited). Below the floor (8 GiB on unified memory; on the CPU, the floor its admission recorded), or when `/proc/meminfo` cannot be read, the runner's whole process group is killed at once (SIGKILL; a runner starved of memory has nothing to save), and the deployment ends in `error` with the reason. PSI memory `full avg10` stops a runner only at 50 % or more together with MemAvailable below twice the floor: page-cache reclaim alone reached 14–28 % with 45 GiB or more available. `local_llm_status` reports the guard's lowest MemAvailable, highest pressure and sample count.

The guard is a backstop, not protection. A load commits memory in bursts of 7–10 GiB per 250 ms, faster than any sampling can follow, and other users of the machine allocate from the same pool at any time. Admission, sized to the whole known allocation, is the control; the kernel OOM killer, which picks the runner first (`oom_score_adj` 1000, DS003), is the last resort. Memory cgroups do not bound GPU allocations here. On eligible dedicated GPUs, Ploinky's MPS share is a best-effort per-CUDA-process GPU budget, not kernel VRAM isolation. Admission compares absolute required bytes with the effective share/free budget. Runner utilization values use their verified physical capacity denominator, avoiding a second multiplication by the share. GPU sharing is refused for unified-memory or unknown-memory-model GPUs in v1. On the CPU the runner's memory is visible to the kernel. With mmap its weights are page cache, which MemAvailable counts as available, so the guard reacts to anonymous memory and to other processes. If the weights are evicted, generation slows down; the guard does not stop the runner for that. Under a container memory limit, the kernel's cgroup OOM killer also picks the runner first.

### Ploinky GPU share and qualification

GPU budgets are applied only to the dedicated profile. CPU runners receive no MPS environment. The unified profile retains physical system MemTotal for its existing experimental vLLM utilization denominator while its absolute RAM admission remains capped by the finite container budget. Its operator switch does not enable vLLM under MPS.

A dedicated vLLM share is not available until the exact selected lock/driver/PCI-device/compute-capability/physical-memory tuple has approved calibration and subsequent normal model readiness/text evidence. No tuple is packaged yet, so an MPS vLLM Run refuses with `vllm_mps_unqualified`. Other adapters retain their own admission policies and reserves. Ordinary unlimited vLLM and the experimental unified path remain unchanged.

### Dashboard

On unified memory the Local LLMs dashboard shows one "Unified memory available" card (MemAvailable of MemTotal, the GPU's users and the page cache, with a meter of what is in use) and the disk card, instead of the GPU and RAM cards, and the run form shows one shared-memory estimate against the pool. No missing value reaches a card. A runner the image lacks, or one without parameters for the profile, cannot be run from the form, which shows why.

In the `cpu` profile the dashboard shows three cards:

- a Compute card: the CPU, its cores, and why no GPU is used, with the shared decision explanation;
- a Memory card: what is available of the pool, with a meter and the floor;
- the disk card.

The run form shows one memory estimate against what is available now (under a container memory limit, what the limit leaves). A runner with no CPU policy gets no Models column and no Install, and the Runners tab says why; its intro promises installs only while some runner can be installed under the profile, and the run form's parameter preview gives the same reason. The dedicated and unified profiles add no such refusal. A runner that serves only some profiles (llama.cpp's CPU build) is not listed at all on the others: not in the Runners tab, not as a Models column, and its Install and Run are refused before anything downloads.

The dashboard renders Venus measurement history as escaped text, without activating evidence references as links. The APU view names the integrated APU, experimental status, shared physical memory, the admission floor, the observed Vulkan0 model-byte ratio and ignored guard-floor warnings. The Venus view shows unknown headroom, the provisional cap and VM memory, and RADV shows its unverified hardware status. A CPU fallback decision with `gpuBackend=vulkan` names the Vulkan GPU; CPU decisions without a known backend use generic GPU wording.

### Rationale

#### Container limits are not engine logic

A cgroup budget and an MPS share are properties of the container the agent runs in. The engine reads the physical machine and asks the limits plug-in at the places where a limit could matter, so a host without limits sees the physical figures and this agent sees them narrowed, with the same key order, warnings and refusal texts as before the engine moved. The 60 s grace for an unreadable GPU, the reserves and the floors are the engine's and are explained in its specification.

#### Trusted unified envelopes

Where an envelope applies it replaces the whole need (weights, buffers, transient), and `validated` tells an admin what was measured. An envelope sent through `local_llm_model_add` or `_update` carries no evidence of where its numbers came from, so it could shrink any configuration's entire need, weights included, on a shared host with only the guard behind it. Certification therefore stays with inputs whose provenance the operator controls outside the model-management API: the shipped catalog and `LOCAL_LLM_CATALOG_FILE` (DS002). The rule, and what happens to entries stored before it, are in pworker DS005.

#### One committed profile

The agent normalizes, admits, launches and guards each Run under one committed hardware profile. NVIDIA keeps the dedicated and unified policies. Vulkan serves discrete RADV, Venus and default-off experimental APU. Venus has a provisional device cap and a separate guarded VM staging pool; APU counts one shared physical pool and refuses finite or unknown budgets. An unusable or absent supported GPU falls back to the CPU at profile selection, while an unreadable query waits for the existing grace period. Every launch repeats admission against a fresh snapshot.
