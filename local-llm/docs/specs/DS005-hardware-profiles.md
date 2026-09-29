---
id: DS005
title: Hardware Profiles
status: accepted
owner: local-llm
summary: The dedicated and unified hardware profiles, how the GPU's memory model is detected, and what changes on a GPU that shares system memory (NVIDIA GB10 in DGX Spark).
---

# DS005 Hardware Profiles

## Introduction

One agent, `local-llms/local-llm`, runs on hosts whose GPU has its own memory (the x86 hosts it served first) and on NVIDIA DGX Spark, whose GB10 GPU shares the machine's 119.6 GiB of system memory. The two need different memory accounting, defaults and safety rules. This specification defines the two hardware profiles and how the agent detects which one applies. The agent's JavaScript is the same on every architecture; only the image's binaries and runner lock differ (DS000, DS004).

## Core Content

### Profiles

| Profile | GPU memory | Memory the models compete for |
| --- | --- | --- |
| `dedicated` | nvidia-smi reports total, used and free memory in MiB | GPU memory for the offloaded weights, KV cache and buffers; system RAM for the rest (DS003) |
| `unified` | the GPU shares system memory | one pool: MemTotal and MemAvailable from `/proc/meminfo` |

The dedicated profile keeps DS003's admission and defaults. Some changes made for the unified profile apply to both (catalog v3, `loadMode`, the runner's OOM priority and JIT cache location, runner availability from the image); they are marked where they are described.

### Detecting the memory model

The hardware snapshot (`src/controller/hardware.mjs`) keeps its first nvidia-smi query unchanged (`name,memory.total,memory.used,memory.free,driver_version`). A second, optional query reads `pci.device_id,compute_cap,addressing_mode,utilization.gpu,power.draw,temperature.gpu`; when it fails (an older driver, an unknown field) the GPU keeps its first-query values and only these facts are missing. Utilization, power and temperature are telemetry: a value of `[N/A]` (laptop GPUs report `power.draw` so) becomes null and never disables the GPU.

| Memory figures | Device facts | Memory model |
| --- | --- | --- |
| any | PCI device id on the known-unified list (`0x2E1210DE`, NVIDIA GB10) | `unified`, also for a future driver that reports numbers |
| all three numeric | not on the list | `dedicated` (GH200 reports ATS addressing with numeric HBM and is dedicated) |
| any not numeric (`[N/A]`, `Not Supported`, empty) | addressing mode `ATS` or `HMM` | `unified` |
| any not numeric | anything else, or the second query failed | unknown: the GPU is reported unavailable, naming the values it gave, and every Run is refused |

The memory model is never inferred from the CPU architecture. On a unified GPU the snapshot's `totalBytes`, `usedBytes` and `freeBytes` are null; the pool is `memory.totalBytes` and `memory.availableBytes` from `/proc/meminfo`, which also gives `freeBytes` (MemFree) and `cachedBytes` (Cached) for display. Nothing sizes from `cudaMemGetInfo`: on GB10 it reports MemFree, not MemAvailable (48.2 GiB "free" while 106.3 GiB was available, measured 2026-09-28), and CUDA allocations reclaim clean page cache. Per-process GPU use (`--query-compute-apps`) stays numeric on GB10 and is shown as before.

Measured on DGX Spark (driver 580.159.03): `NVIDIA GB10, [N/A], [N/A], [N/A], 580.159.03` and `0x2E1210DE, 12.1, ATS, 0, 4.52, 47`.


### CPU threads

On a CPU whose cores differ in capacity (sysfs `cpu_capacity`), `performanceCoreCount()` counts the allowed CPUs within 10 % of the highest capacity: 10 on DGX Spark (Cortex-X925 at 997–1024, Cortex-A725 at 718–731). When capacities are equal or not reported it returns null. The dedicated default thread count (physical cores minus 2, DS003) is unchanged.

### Runner availability and the image's GPUs

The image says what it contains: `/opt/local-llm/source.contract` (key=value lines). When it exists, a runner is available only if its executable is in the image (llama.cpp, ik_llama.cpp, Ollama) or the image's runner lock lists it (vLLM, TabbyAPI, LM Studio; DS004). An unavailable runner is `supported: false` in the overview with "not available on this platform: this image does not include it", and `local_llm_run` refuses it with `runner_unavailable` before anything downloads. Without a contract (tests, development) every runner counts as available. The arm64 image's contract names `gpu_compute_capabilities=12.1`, the GPUs its CUDA runners were built for; a GPU of any other compute capability, or one that does not report it, is refused before downloading. The amd64 image names none and nothing is checked there.

### Choosing the profile

The controller decides the profile from the first hardware snapshot that shows a usable GPU, logs it (`hardware profile: unified (NVIDIA GB10)`) and keeps it for the container's lifetime; the overview and `local_llm_status` report it. A transient failure, such as a cold nvidia-smi or a snapshot that cannot be read, leaves it undecided. The overview decides it from the one snapshot it reads and normalizes, describes and admits every row against that same snapshot; while it is undecided the rows use the dedicated defaults for display and none is admitted (a snapshot without a usable GPU is `incompatible` with the GPU's reason).

A Run is normalized, admitted, launched and guarded under one committed profile, which the deployment records (`deployment.profile`). While no snapshot has shown a usable GPU, `local_llm_run` reads a fresh one; if that one does not show a usable GPU either, the Run is refused with `admission_incompatible` ("… the hardware profile is not decided; run again once the GPU can be read", after the GPU's own reason when it gave one) before any model lookup, and nothing is recorded, saved, downloaded or started, so the same request can be sent again. A profile that became usable in between is committed by that Run and used for all of it: parameters are never normalized for the dedicated defaults and then admitted, launched or guarded by the unified rules, or the reverse. A later snapshot whose memory model disagrees is not acted on: admission refuses every Run as `incompatible`, saying the agent must be restarted. The catalog's `profiles` (DS002) decide which models are offered; `recommended` and `validated` are read for the current profile.

### llama.cpp on unified memory

llama.cpp has the unified-memory policy of this release, and vLLM an experimental one behind the operator's switch (below); every other runner is `incompatible` there. llama.cpp's unified parameters (`paramSchemaFor('unified')`) are `ctxSize` (up to 262,144 and the model's training context per slot), `parallel`, `loadMode` (`dio`, the default, or `none`), `mtp` (`--spec-type draft-mtp --spec-draft-n-max 3`), `threads` and `chatTemplateKwargs` (`reasoning_effort`, `preserve_thinking`). Every flag that changes memory is fixed: `-ngl 999 -fa on`, f16 K and V, `-b 2048 -ub 2048`, `--kv-unified` above one slot, and `--cache-ram 8192`. Threads default to the high-performance cores (10 on DGX Spark, capped by a cgroup CPU quota), else physical cores minus 2.

**MTP is a capability of the weights.** A model entry declares `mtp: true` when its weights carry a multi-token-prediction head (DS002); a user entry may declare it too, since it is a capability, not a measurement. `mtp: true` is refused (a parameter error on `mtp`) for a model that does not declare the head, whatever the source of the value: an explicit value wins over a default but cannot make an unsupported configuration valid. A catalog `recommended` MTP default on a model without the head is refused when the catalog loads. The Qwen3.6-35B-A3B MTP entry declares the head and recommends `mtp: true` on unified memory, so omitting `mtp` turns it on and an explicit `false` turns it off; the overview's parameters, the preview, the launch flags and the estimate all use the same normalized value. gpt-oss and every other entry leave it off.

`dio` loaded a cold gpt-oss-20b in 4.1–4.5 s with +0.8 GiB of page cache; `none` took 8.7 s and added 12.1 GiB (gpt-oss-120b: 16 s vs 51 s, +0.9 vs +37 GiB). After a controller download is verified, the file is dropped from the page cache (`dd iflag=nocache count=0`, which calls `posix_fadvise(DONTNEED)`; best effort).

### Unified admission

No benchmark, measured envelope or pre-calibrated configuration is a prerequisite for a Run (owner, 2026-09-29): bundled models and models added at run time are admitted from their data and a fresh snapshot, with the same reserve, floor and guard. The need comes from one of two sources.

**A trusted envelope rectangle, where one covers the parameters.** `unified.envelope` (DS002) lists rectangles of total context and parallel slots, each with its load mode (`loadMode`, `dio` or `none`) and MTP allowed or not. A rectangle applies only to its own load mode: no rectangle stands in for a mode it was not measured with. The need is the corner's figures:

need = `bufferBytes` (the model, KV, recurrent-state, output and compute buffers llama.cpp logs at `-lv 4`) + `transientBytes` (memory beyond those buffers during load, prefill and parallel requests, times 1.25) + 1.5 GiB (the CUDA context and the runner process) + the prompt cache's bound (`--cache-ram`, 8192 MiB).

**An estimate, everywhere else** (`estimateUnifiedLlamaServer`): no envelope, parameters outside every rectangle, or another load mode. It is built only from the model entry and the existing formulas, never from a measurement on this machine:

need = the pinned weight size (every shard; all layers resident) + f16 KV for the whole `ctxSize` (`memory.kvBytesPerToken` × `ctxSize` + `memory.fixedKvBytes`; `--kv-unified` makes `ctxSize` the pool, not a per-slot size) + compute buffers (the flash-attention formula of DS003 at the fixed micro-batch of 2048) + with MTP, a draft context: one more layer of KV (`kvBytes / memory.layers`) and one more set of compute buffers + 1.5 GiB + the prompt cache's bound.

An entry without `memory.kvBytesPerToken` uses DS003's default of 64 KiB per token, and without `memory.layers` the default of 48 layers; the estimate names every defaulted input (`defaulted`, with `contextLength` when it is missing), and a warning says how to add them (Settings, or `local_llm_model_update`): the actionable path for missing metadata is data, not a benchmark. The estimate carries `envelope: null` and `measured: false`, its `basis` says it is an estimate and where its per-token figure came from (the catalog, the entry added at run time, or a default), and a warning always says the run is unmeasured, that the figure is an estimate, and that the guard stops the runner below the floor. Parameters outside a trusted rectangle add a warning naming the rectangles. User-supplied `memory` values feed estimates only; they are never envelopes. An entry without a pinned size is `incompatible`.

llama-server's prompt cache lives in host memory, is bounded by `--cache-ram` (it evicts before it allocates, and skips a single state larger than the bound), and is allocated in one burst when a new task starts. `--cache-ram 0` is not an option: it also turns off the clearing of idle slots, so with a unified KV cache an idle slot keeps its cells and new requests fail (measured: 0 of 4). 8192 MiB, llama.cpp's default, is the measured baseline; smaller bounds wait for their measurements.

| Verdict | When |
| --- | --- |
| `incompatible` | need above MemTotal less a 16 GiB host reserve; `/proc/meminfo` unreadable; no pinned weight size |
| `insufficient-now` | need above MemAvailable less the 8 GiB floor (other processes on the machine hold the rest), or the download does not fit the disk |
| `ok` | otherwise, with the estimate's warnings |

No CPU offload (`nCpuMoe`) in this profile. The reserve, floor and runtime constant are provisional; the benchmark phases that would calibrate them are deferred by the owner, and none of them is a calibrated envelope.

### vLLM on unified memory (experimental)

vLLM runs on unified memory only as an explicit, experimental operator choice: `LOCAL_LLM_VLLM_UNIFIED=experimental` (`ploinky var LOCAL_LLM_VLLM_UNIFIED experimental`, then a restart of local-llm; DS001, DS004). Without it `local_llm_run` and `local_llm_runner_install` refuse vLLM with `runner_disabled` before anything downloads, and the overview shows why; no catalog entry, parameter, saved value or runner choice turns it on. The switch concerns unified memory only: on a dedicated GPU vLLM is not gated. Before the profile is decided the switch fails closed; installing decides the profile from a snapshot first. The image must also carry vLLM in its runner lock (arm64: DS004), and an admin installs it on demand.

Its unified parameters are the dedicated ones without `cpuOffloadGb` (one pool has nothing to offload to). Admission (`admitUnifiedVllm`) treats `--gpu-memory-utilization` as a share of the whole pool: on an integrated GPU, CUDA's total is MemTotal. An empty share is computed from the snapshot: at most 0.9, within the host reserve, and leaving the floor and the runner's own RAM free. The need is that share of MemTotal plus the runner's RAM beyond it (3 GiB, measured on a dedicated GPU and unvalidated here). The weights and a KV cache for `maxModelLen` must fit in the share. An admin's share that cannot hold them, or whose need exceeds MemTotal less the reserve, is `incompatible`; a share that does not fit what is available now, or a computed share below 0.1, is `insufficient-now`. The estimate carries `experimental: true`, `measured: false` and `envelope: null`, and a warning says vLLM is experimental and unmeasured here. The fresh admission immediately before launch recomputes the share, which is the value vLLM is started with, and the 250 ms guard watches it like every runner. FlashInfer, caches and the runner's offline settings are unchanged (DS004); whether vLLM 0.30 contacts any host on GB10 is an open native check.

Admission runs again from a fresh snapshot after the download, and last immediately before the runner starts, after the runner is prepared and the weights are checked for the last time (DS003). Memory taken meanwhile refuses the start as `insufficient-now`: the guard's floor is far below a run's need, so it would not stop a runner that starts into too little memory.

**Who may certify an envelope.** `unified.envelope` and the `validated` labels come only from the trusted seed: the catalog shipped with the agent, or the operator's `LOCAL_LLM_CATALOG_FILE`, whose provenance is controlled outside the model-management API. `local_llm_model_add` and `_update` refuse an entry that carries an envelope or any `validated` label (`invalid_model`); the empty forms a stored entry carries (`unified: null`, `validated: {}`) are accepted. A registry entry stored with either before this rule is kept in the state file as it is, reported in the overview's `unsupportedModels` and the log, never offered, and can be removed. A user entry that names a seed's own file does not inherit the seed's envelope: an envelope belongs to the catalog entry, not to the artifact, so the alias is sized by estimate. No envelope is shipped in this release, and this rule approves none. An envelope is no longer a prerequisite for a Run; it only replaces the estimate where it applies.

### Memory guard

On unified memory every runner is watched from its start to its exit: MemAvailable is sampled every 250 ms whether the runner is loading, ready or stopping (a graceful Stop or Replace keeps it watched until the process has exited). Below the 8 GiB floor, or when `/proc/meminfo` cannot be read, the runner's whole process group is killed at once (SIGKILL; a runner starved of memory has nothing to save), and the deployment ends in `error` with the reason. PSI memory `full avg10` stops a runner only at 50 % or more together with MemAvailable below twice the floor: page-cache reclaim alone reached 14–28 % with 45 GiB or more available. `local_llm_status` reports the guard's lowest MemAvailable, highest pressure and sample count.

The guard is a backstop, not protection. A load commits memory in bursts of 7–10 GiB per 250 ms, faster than any sampling can follow, and other users of the machine allocate from the same pool at any time. Admission, sized to the whole known allocation, is the control; the kernel OOM killer, which picks the runner first (`oom_score_adj` 1000, DS003), is the last resort. Memory cgroups do not bound GPU allocations here.

### Dashboard

On unified memory the Local LLMs dashboard shows one "Unified memory available" card (MemAvailable of MemTotal, the GPU's users and the page cache, with a meter of what is in use) and the disk card, instead of the GPU and RAM cards, and the run form shows one shared-memory estimate against the pool. No missing value reaches a card. A runner the image lacks, or one without parameters for the profile, cannot be run from the form, which shows why.

## Decisions & Questions

### Question #1: Why a known-unified device list beside the addressing mode?

Response: The addressing mode alone is not a unified-memory signal (GH200 reports ATS with dedicated HBM), and `[N/A]` memory is how this driver reports GB10 today. A future driver that reports numbers for GB10 would otherwise turn it into a dedicated GPU with a pool the size of system memory. The device id keeps GB10 unified whatever the driver reports; the addressing mode covers other GPUs that share memory, which the image then accepts or refuses by compute capability (DS000).

### Question #2: Why can a user entry not carry a unified envelope?

Response: Where an envelope applies it replaces the estimate, and `validated` tells an admin what was measured. An envelope sent through `local_llm_model_add` or `_update` carries no evidence of where its numbers came from, so it could shrink any configuration's need below the estimate on a shared host with only the guard behind it. Certification therefore stays with inputs whose provenance the operator controls outside the model-management API: the shipped catalog and `LOCAL_LLM_CATALOG_FILE`. Admin-measured envelopes would need a recorded owner decision and a provenance field; this release has neither. Stored entries are reported rather than migrated or deleted, because they are the admin's data.

### Question #3: Why is the load mode part of an envelope rectangle?

Response: The two unified load modes use memory very differently: `none` added 12.1 GiB of page cache for gpt-oss-20b and 37 GiB for gpt-oss-120b, against 0.8–0.9 GiB for `dio`. A rectangle measured with one mode says nothing about the other, so each names the mode it was measured with and applies only to that mode. There is no inference from an older envelope; a mode no rectangle covers is sized by estimate.

### Question #4: Why is no benchmark a prerequisite for a Run?

Response: The owner decided on 2026-09-29 that users must be able to add and run models at run time without per-model benchmarks, and deferred the benchmark phases; the earlier measurements were disturbed by another user's fine-tuning and stay historical evidence, not calibration. So every run is sized by a labelled estimate from the model's data (or by a trusted envelope where one applies) and checked against a fresh snapshot with the host reserve and the floor, and the guard remains the backstop. An estimate can be wrong; it says so, names its defaults, and never presents itself as measured.

### Question #5: Why does vLLM on unified memory need an operator switch when llama.cpp does not?

Response: The owner chose experimental, opt-in vLLM on DGX Spark. vLLM preallocates its whole share of the pool for weights and KV cache, and its behaviour on GB10 (the share it can actually take, its host-side memory, any network access) has not been checked on the hardware. An explicit operator switch, outside the model-management API, keeps that choice with whoever runs the deployment; a model entry or a Run parameter cannot make it.

### Question #6: Why is MTP a declared capability rather than a parameter any model may set?

Response: MTP speculative decoding needs a multi-token-prediction head in the weights; llama.cpp cannot use it on a model without one. Honouring an explicit value must not turn an unsupported configuration into a Run that fails at load or silently misbehaves, so the entry declares the capability and the parameter is refused elsewhere. Declaring a capability is not certifying memory figures, so user entries may declare it.

## Conclusion

The agent picks the dedicated or unified profile from what nvidia-smi reports, refuses a GPU whose memory model it cannot tell, normalizes, admits, launches and guards each Run under one committed profile, runs only what the image contains, and admits unified runs, bundled or added at run time, from a trusted envelope where one applies and otherwise from a labelled estimate, checked against a fresh snapshot with a host reserve and a floor, rechecked immediately before the runner starts, with a 250 ms guard as a backstop. vLLM joins it only as an experimental operator choice.
