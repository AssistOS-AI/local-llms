---
id: DS005
title: Hardware Profiles
status: accepted
owner: local-llm
summary: The dedicated, unified and cpu hardware profiles, how the GPU's memory model and its absence are detected, what changes on a GPU that shares system memory (NVIDIA GB10 in DGX Spark), and how models run on the CPU when no NVIDIA GPU is usable.
---

# DS005 Hardware Profiles

## Introduction

One agent, `local-llms/local-llm`, runs on three kinds of host. The first are hosts whose GPU has its own memory (the x86 hosts it served first). The second is NVIDIA DGX Spark, whose GB10 GPU shares the machine's 119.6 GiB of system memory. The third are hosts where no NVIDIA GPU is usable, where it runs models on the CPU: a Mac through its Podman machine, a server without a GPU, or a host whose GPU is not granted to it. The three need different memory accounting, defaults and safety rules. This specification defines the three hardware profiles and how the agent detects which one applies. The agent's JavaScript is the same on every architecture; only the image's binaries and runner lock differ (DS000, DS004).

## Core Content

### Profiles

| Profile | GPU memory | Memory the models compete for |
| --- | --- | --- |
| `dedicated` | nvidia-smi reports total, used and free memory in MiB | GPU memory for the offloaded weights, KV cache and buffers; system RAM for the rest (DS003) |
| `unified` | the GPU shares system memory | one pool: MemTotal and MemAvailable from `/proc/meminfo` |
| `cpu` | none is used: no NVIDIA GPU is attached, or the attached one cannot be used | one pool: MemTotal and MemAvailable from `/proc/meminfo`, capped by a container memory limit when one is set |

The dedicated profile keeps DS003's admission and defaults. Some changes made for the unified profile apply to both (catalog v3, `loadMode`, the runner's OOM priority and JIT cache location, runner availability from the image); they are marked where they are described. The `cpu` profile is chosen automatically; no operator switch, catalog entry or parameter turns it on or off.

### Detecting the memory model

The hardware snapshot (`src/controller/hardware.mjs`) keeps its first nvidia-smi query unchanged (`name,memory.total,memory.used,memory.free,driver_version`). A second, optional query reads `pci.device_id,compute_cap,addressing_mode,utilization.gpu,power.draw,temperature.gpu`; when it fails (an older driver, an unknown field) the GPU keeps its first-query values and only these facts are missing. Utilization, power and temperature are telemetry: a value of `[N/A]` (laptop GPUs report `power.draw` so) becomes null and never disables the GPU.

| Memory figures | Device facts | Memory model |
| --- | --- | --- |
| any | PCI device id on the known-unified list (`0x2E1210DE`, NVIDIA GB10) | `unified`, also for a future driver that reports numbers |
| all three numeric | not on the list | `dedicated` (GH200 reports ATS addressing with numeric HBM and is dedicated) |
| any not numeric (`[N/A]`, `Not Supported`, empty) | addressing mode `ATS` or `HMM` | `unified` |
| any not numeric | the second query answered, but with no `ATS` or `HMM` addressing and not on the known-unified list | unusable: models run on the CPU (`cpu` profile) |
| any not numeric | the second query failed | unreadable: the profile waits (Choosing the profile) |

Every unavailable GPU also says why (`gpu.state`):

| State | When |
| --- | --- |
| `absent` | nvidia-smi is not in the container (Ploinky attached no GPU; its `PLOINKY_GPU_REASON` is kept) |
| `unusable` | the row above |
| `unreadable` | nvidia-smi fails or times out, reports no GPU, or its device query fails |

A usable GPU carries no state. A snapshot without a state is treated as unreadable.

The memory model is never inferred from the CPU architecture. On a unified GPU the snapshot's `totalBytes`, `usedBytes` and `freeBytes` are null; the pool is `memory.totalBytes` and `memory.availableBytes` from `/proc/meminfo`, which also gives `freeBytes` (MemFree) and `cachedBytes` (Cached) for display. Nothing sizes from `cudaMemGetInfo`: on GB10 it reports MemFree, not MemAvailable (48.2 GiB "free" while 106.3 GiB was available, measured 2026-09-28), and CUDA allocations reclaim clean page cache. Per-process GPU use (`--query-compute-apps`) stays numeric on GB10 and is shown as before.

Measured on DGX Spark (driver 580.159.03): `NVIDIA GB10, [N/A], [N/A], [N/A], 580.159.03` and `0x2E1210DE, 12.1, ATS, 0, 4.52, 47`.


### CPU threads

On a CPU whose cores differ in capacity (sysfs `cpu_capacity`), `performanceCoreCount()` counts the allowed CPUs within 10 % of the highest capacity: 10 on DGX Spark (Cortex-X925 at 997–1024, Cortex-A725 at 718–731). When capacities are equal or not reported it returns null. The dedicated default thread count (physical cores minus 2, DS003) is unchanged. On the CPU profile the default is the high-performance cores where cores differ. Otherwise it is the physical cores minus 1 on hosts with four cores or fewer, and minus 2 above that (`cpuDefaultThreads`). A cgroup CPU quota caps it. On the 4-CPU Podman machine of a 16 GiB Mac this gives 3 (4 − 1; no CPU quota is set there).

### Runner availability and the image's GPUs

The image says what it contains: `/opt/local-llm/source.contract` (key=value lines). When it exists, a runner is available only if its executable is in the image (llama.cpp, ik_llama.cpp, Ollama) or the image's runner lock lists it (vLLM, TabbyAPI, LM Studio; DS004). An unavailable runner is `supported: false` in the overview with "not available on this platform: this image does not include it", and `local_llm_run` refuses it with `runner_unavailable` before anything downloads. Without a contract (tests, development) every runner counts as available. The arm64 image's contract names `gpu_compute_capabilities=12.1`, the GPUs its CUDA runners were built for; a GPU that reports another compute capability decides the `cpu` profile, and the Compute card and every CPU admission say why. This changes today's refusal on arm64 hosts with other NVIDIA GPUs, for example Thor or GH200. A GPU that does not report its capability counts as unreadable (Choosing the profile). The amd64 image names none and nothing is checked there. Phase 3: on the `cpu` profile the agent's runner lock (DS004) may also make a runner available, Ollama on arm64 and llama.cpp's CPU build on both architectures, when the runner has a CPU policy and the image does not already hold its executable. On the dedicated and unified profiles it makes none, so those profiles show the runners they always did, and a runner that names the profiles it serves (llama.cpp's CPU build serves `cpu`) is not listed on any other.

### Choosing the profile

The controller decides the profile with `decideProfile` (`src/controller/profiles.mjs`) from every snapshot it takes while the profile is undecided: the overview's, `local_llm_status`'s (one snapshot), a Run's, and an install's. It logs the decision (`hardware profile: cpu (absent: …)`) and keeps it for the container's lifetime. The overview and status report it as `profile` and `profileDecision` (cause and reason).

| Snapshot | Profile |
| --- | --- |
| A usable GPU that the image supports | `dedicated` or `unified` |
| A usable GPU that reports a compute capability the image's CUDA runners were not built for | `cpu` (cause `mismatch`) |
| `absent` or `unusable` | `cpu`, at once |
| `unreadable`, or a usable GPU that does not report its capability on an image that lists capabilities | undecided; `cpu` (cause `unreadable-timeout`) once 60 s have passed since the first such snapshot |

A readable snapshot resets that wait. A cold nvidia-smi therefore never locks a GPU host into the CPU profile, and a GPU that stays unreadable does not stop the agent from running. A stop is not a missing GPU: a snapshot cut short because the agent is draining, or because a Run was stopped, counts as no snapshot at all. It neither advances the wait nor decides the profile, so a status or an install that meets a shutdown leaves the profile as it was.

While the profile is undecided:

- the overview's rows use the dedicated defaults for display, and none is admitted;
- `local_llm_run` refuses with `admission_incompatible`: "… the hardware profile is not decided; run again once the GPU can be read. If the GPU is still unreadable 60 s after the first failed read, the next Run uses the CPU."

Nothing is recorded, so the same request can be sent again.

A Run is normalized, admitted, launched and guarded under one committed profile, which the deployment records (`deployment.profile`). A later snapshot that disagrees is not acted on:

- Under `dedicated` or `unified`, a GPU that is no longer usable refuses every Run as `incompatible`. When the snapshot shows positive evidence (no nvidia-smi at all, or a GPU that cannot be used) the refusal says to restart local-llm, which then decides `cpu`. When the GPU is merely unreadable the refusal gives the GPU's own reason and no hint: the failure may be transient, and a restart during it could lock a GPU host into `cpu`.
- Under `cpu`, when the cause was `unreadable-timeout` and the GPU has since become usable, every admission warns that a restart would use it. Runs continue on the CPU.

The catalog's `profiles` (DS002) decide which models are offered; `recommended` and `validated` are read for the current profile.

### llama.cpp on unified memory

llama.cpp has the unified-memory policy of this release, and vLLM an experimental one behind the operator's switch (below); every other runner is `incompatible` there. llama.cpp's unified parameters (`paramSchemaFor('unified')`) are `ctxSize` (up to 262,144 and the model's training context per slot; the 32,768 default is lowered to that context when it is smaller), `parallel`, `loadMode` (`dio`, the default, or `none`), `mtp` (`--spec-type draft-mtp --spec-draft-n-max 3`), `threads` and `chatTemplateKwargs` (`reasoning_effort`, `preserve_thinking`). Every flag that changes memory is fixed: `-ngl 999 -fa on`, f16 K and V, `-b 2048 -ub 2048`, `--kv-unified` above one slot, and `--cache-ram 8192`. Threads default to the high-performance cores (10 on DGX Spark, capped by a cgroup CPU quota), else physical cores minus 2.

**MTP is a capability of the weights.** A model entry declares `mtp: true` when its weights carry a multi-token-prediction head (DS002); a user entry may declare it too, since it is a capability, not a measurement. `mtp: true` is refused (a parameter error on `mtp`) for a model that does not declare the head, whatever the source of the value: an explicit value wins over a default but cannot make an unsupported configuration valid. A catalog `recommended` MTP default on a model without the head is refused when the catalog loads. The Qwen3.6-35B-A3B MTP entry declares the head and recommends `mtp: true` on unified memory, so omitting `mtp` turns it on and an explicit `false` turns it off; the overview's parameters, the preview, the launch flags and the estimate all use the same normalized value. gpt-oss and every other entry leave it off.

`dio` loaded a cold gpt-oss-20b in 4.1–4.5 s with +0.8 GiB of page cache; `none` took 8.7 s and added 12.1 GiB (gpt-oss-120b: 16 s vs 51 s, +0.9 vs +37 GiB). After a controller download is verified, the file is dropped from the page cache (`dd iflag=nocache count=0`, which calls `posix_fadvise(DONTNEED)`; best effort).

### llama.cpp on the CPU

llama.cpp has a CPU policy, and so do Ollama (below) and llama.cpp's CPU build, `llama.cpp-cpu`: ggml-org's own CPU release (`b11295`, its `ubuntu-arm64` or `ubuntu-x64` asset), installed on demand from the agent's runner lock (DS004) and run from its runnable copy on port 18085. It is one adapter with llama.cpp's, created with `profiles: ['cpu']`: it gets the CPU schema, launch and `admitCpu` below, no schema and no policy for the GPU profiles, and the controller leaves it out of the overview there. A runner without a CPU policy is `incompatible` in the `cpu` profile ("… needs an NVIDIA GPU in this release"), and `local_llm_runner_install` refuses it with `runner_unavailable`. llama.cpp loads the best CPU backend for the CPU by itself (`libggml-cpu-armv8.2_2.so` on the M1's Podman machine, as `llama-server -lv 4 --list-devices` logs; a server run's log names only the device, `- CPU : CPU (…)` under `device_info:`, which is what the status reports). Its CUDA backend is skipped when the driver library is absent.

llama.cpp's CPU parameters (`paramSchemaFor('cpu')`):

| Parameter | Values |
| --- | --- |
| `ctxSize` | 512 to 131,072, at most the model's training context per slot; default 4,096 |
| `parallel` | 1 to 4 |
| `loadMode` | `mmap` (the default) or `none` |
| `threads` | an integer, or empty for the default above |
| `chatTemplateKwargs` | as in the other profiles |

Every flag that changes memory or the device is fixed:

- `--device none --n-gpu-layers 0 --fit off --flash-attn on`
- f16 K and V
- `-b 512 -ub 512`
- `--kv-unified` above one slot
- `--cache-ram 256`

The launch environment has no `LD_LIBRARY_PATH` and an empty `CUDA_VISIBLE_DEVICES`. `mlock` and `dio` are not offered: the container's locked-memory limit is 8 MiB, and direct I/O on a Mac's shared folder is unmeasured.

### CPU admission

No benchmark is a prerequisite. `estimateCpuLlamaServer` builds the need from the model entry and fixed terms, never from a measurement on this machine. The need is the sum of:

- the pinned weight size (every shard; all layers resident);
- f16 KV for the whole `ctxSize`, `memory.kvBytesPerToken` × `ctxSize` + `memory.fixedKvBytes` (DS003's 64 KiB per token when the entry lacks it);
- compute buffers (DS003's flash-attention formula at the fixed micro-batch of 512);
- 512 MiB for the runner process;
- the prompt cache's bound (256 MiB).

The estimate reports the need as `ramBytes`, names every defaulted input and carries `measured: false`. The pool is MemTotal and MemAvailable. When a container memory limit is set (cgroup `memory.max`), the pool is at most that limit, and what is available now is at most the limit less `memory.current`.

| Verdict | When |
| --- | --- |
| `incompatible` | the need is above the pool less the host reserve, min(16 GiB, max(1.5 GiB, 25 % of the pool)); or memory cannot be read; or the weight size is not pinned |
| `insufficient-now` | the need is above what is available less the floor, min(8 GiB, max(512 MiB, 10 % of the pool)); or the download does not fit the disk |
| `ok` | otherwise, with warnings: the run is on the CPU and slower than on a GPU (with the decision's reason); the figure is an estimate; the guard stops the runner below the floor |

Reserve and floor round up to whole bytes. On the 6 GiB Podman machine of a 16 GiB Mac (MemTotal 5.76 GiB) the reserve is 1.5 GiB and the floor 0.58 GiB. The constants are provisional, like the unified ones.

### Ollama on the CPU

Ollama's CPU parameters (`paramSchemaFor('cpu')`; the dedicated schema is unchanged, and unified memory has none):

| Parameter | Values |
| --- | --- |
| `numCtx` | 512 to 131,072; default 4,096 |
| `numThread` | an integer from 1 to 256, or empty for Ollama's default |
| `keepAlive` | as in the dedicated schema |

`numGpu` is fixed at 0, flash attention is Ollama's default, and the KV cache type is fixed at f16 (`OLLAMA_KV_CACHE_TYPE`). The CPU launch has no `LD_LIBRARY_PATH` and an empty `CUDA_VISIBLE_DEVICES`, and the load request carries `num_gpu: 0` (`requestOptions` with the profile). The chat target carries the same options, but the chat responder does not forward them, because Ollama's OpenAI-compatible endpoint takes no `options`: what keeps a chat request off a GPU is the hidden device, not a request option. The adapter starts `<runnable copy>/bin/ollama` when the agent's lock installed it (the arm64 image has no Ollama; the archive unpacks with `strip: 0`) and the image's `/opt/ollama/bin/ollama` otherwise, and detection asks the image's binary first and the installer second.

`admitCpuOllama` sizes a run from the tag's pinned size, never from a measurement. The need is the sum of:

- the pinned `size` (config plus layers);
- f16 KV for `numCtx`, `memory.kvBytesPerToken` × `numCtx` + `memory.fixedKvBytes` (64 KiB per token when the entry lacks it, named in the warnings);
- compute buffers at llama.cpp's flash-attention formula and Ollama's default batch of 512 (216.92 MiB);
- 768 MiB for the Ollama server and its runner process (provisional).

It is checked against the same pool, reserve and floor as llama.cpp, and the memory guard stops the runner below that floor. A tag whose size is not pinned, such as an entry stored before tags were pinned or added on another profile, is `incompatible` with "The tag's size is not pinned; update the model entry so it is pinned." An Update that names the bare tag pins it (DS002). The size is also what the pull's disk check counts.

That `num_gpu: 0` and the hidden device keep `size_vram` at 0 in Ollama 0.34.4 is an assumption until a live run shows `sizeVramBytes: 0` in the state file.

### Unified admission

No benchmark, measured envelope or pre-calibrated configuration is a prerequisite for a Run (owner, 2026-09-29): bundled models and models added at run time are admitted from their data and a fresh snapshot, with the same reserve, floor and guard. The need comes from one of two sources.

**A trusted envelope rectangle, where one covers the parameters.** `unified.envelope` (DS002) lists rectangles of total context and parallel slots, each with its load mode (`loadMode`, `dio` or `none`) and MTP allowed or not. A rectangle applies only to its own load mode: no rectangle stands in for a mode it was not measured with. The need is the corner's figures, labelled as catalog envelope data (`measured: false`) with a warning that they are not validated calibration for this host:

need = `bufferBytes` (the model, KV, recurrent-state, output and compute buffers llama.cpp logs at `-lv 4`) + `transientBytes` (memory beyond those buffers during load, prefill and parallel requests, times 1.25) + 1.5 GiB (the CUDA context and the runner process) + the prompt cache's bound (`--cache-ram`, 8192 MiB).

**An estimate, everywhere else** (`estimateUnifiedLlamaServer`): no envelope, parameters outside every rectangle, or another load mode. It is built only from the model entry and the existing formulas, never from a measurement on this machine:

need = the pinned weight size (every shard; all layers resident) + f16 KV for the whole `ctxSize` (`memory.kvBytesPerToken` × `ctxSize` + `memory.fixedKvBytes`; `--kv-unified` makes `ctxSize` the pool, not a per-slot size) + compute buffers (the flash-attention formula of DS003 at the fixed micro-batch of 2048) + with MTP, a draft context: one more layer of KV (`kvBytes / memory.layers`) and one more set of compute buffers + 1.5 GiB + the prompt cache's bound.

An entry without `memory.kvBytesPerToken` uses DS003's default of 64 KiB per token, and without `memory.layers` the default of 48 layers; the estimate names every defaulted input (`defaulted`, with `contextLength` when it is missing), and a warning says what each default can miss (the per-token default can understate the KV cache several times for large dense models; without `contextLength` the context is not capped at the training context) and how to add them (Settings, or `local_llm_model_update`): the actionable path for missing metadata is data, not a benchmark. The estimate carries `envelope: null` and `measured: false`, its `basis` says it is an estimate and where its per-token figure came from (the catalog, the entry added at run time, or a default), and a warning always says the run is unmeasured, that the figure is an estimate, and that the guard stops the runner below the floor. Parameters outside a trusted rectangle add a warning naming the rectangles. An entry without a pinned size is `incompatible`.

**What sizing from an entry added at run time can change.** Its `memory.kvBytesPerToken`, `memory.fixedKvBytes` and, with MTP, `memory.layers` feed only the KV and MTP terms. The weights term is the pinned size, bound to the pinned sha256 that download and adoption enforce (DS002), and the compute, runtime and prompt-cache terms are fixed. So an understated value can lower only the KV and MTP terms; the estimate lists the fields it used (`userSizing`), and a warning says nothing checks them against the weights and that the memory guard is then the only backstop. An envelope, by contrast, replaces the whole need, which is why envelopes come only from the operator's trusted catalogs (below, Question #2).

llama-server's prompt cache lives in host memory, is bounded by `--cache-ram` (it evicts before it allocates, and skips a single state larger than the bound), and is allocated in one burst when a new task starts. `--cache-ram 0` is not an option: it also turns off the clearing of idle slots, so with a unified KV cache an idle slot keeps its cells and new requests fail (measured: 0 of 4). 8192 MiB, llama.cpp's default, is the measured baseline; smaller bounds wait for their measurements.

| Verdict | When |
| --- | --- |
| `incompatible` | need above MemTotal less a 16 GiB host reserve; `/proc/meminfo` unreadable; no pinned weight size |
| `insufficient-now` | need above MemAvailable less the 8 GiB floor (other processes on the machine hold the rest), or the download does not fit the disk |
| `ok` | otherwise, with the estimate's warnings |

No CPU offload (`nCpuMoe`) in this profile. The reserve, floor and runtime constant are provisional; the benchmark phases that would calibrate them are deferred by the owner, and none of them is a calibrated envelope.

### vLLM on unified memory (experimental)

vLLM runs on unified memory only as an explicit, experimental operator choice: `LOCAL_LLM_VLLM_UNIFIED=experimental` (`ploinky var LOCAL_LLM_VLLM_UNIFIED experimental`, then a restart of local-llm; DS001, DS004). Without it `local_llm_run` and `local_llm_runner_install` refuse vLLM with `runner_disabled` before anything downloads, and the overview shows why; no catalog entry, parameter, saved value or runner choice turns it on. The switch concerns unified memory only: on a dedicated GPU vLLM is not gated. While the profile is not decided (the GPU cannot be read), an image that can never run on unified memory counts as dedicated, but only on positive evidence that it is the amd64 image: a `source.contract` that does not say `architecture=arm64` and either names `ik_llama_cpp` (built only into the amd64 image) or runs on an x64 CPU. That image keeps vLLM enabled and installable as before. Every other case, meaning `architecture=arm64`, an empty, malformed or architecture-less contract on arm64, or no contract in development, keeps the switch closed. The reason then says the GPU could not be read, so it is not known whether it shares system memory, and names the switch, with no retry that cannot succeed; there, installing first tries to decide the profile from a snapshot. A caller of the switch that names no profile gets the same closed answer. The image must also carry vLLM in its runner lock (arm64: DS004), and an admin installs it on demand.

Its unified parameters are the dedicated ones without `cpuOffloadGb` (one pool has nothing to offload to). Admission (`admitUnifiedVllm`) treats `--gpu-memory-utilization` as a share of the whole pool: on an integrated GPU, CUDA's total is MemTotal. An empty share is computed from the snapshot: at most 0.9, within the host reserve, and leaving the floor and the runner's own RAM free. The need is that share of MemTotal plus the runner's RAM beyond it (3 GiB, measured on a dedicated GPU and unvalidated here). The weights, a KV cache for `maxModelLen` and vLLM's overhead (768 MiB for activations, CUDA context and allocator slack, as on a dedicated GPU) must fit in the share; no usable-share discount applies, since CUDA's total is the pool itself. An entry added at run time that supplies `memory.kvBytesPerToken` gets the user-sizing warning above. An admin's share that cannot hold them, or whose need exceeds MemTotal less the reserve, is `incompatible`; a share that does not fit what is available now, or a computed share below 0.1, is `insufficient-now`. The estimate carries `experimental: true`, `measured: false` and `envelope: null`, and a warning says vLLM is experimental and unmeasured here. The fresh admission immediately before launch recomputes the share, which is the value vLLM is started with, and the 250 ms guard watches it like every runner. FlashInfer, caches and the runner's offline settings are unchanged (DS004); whether vLLM 0.30 contacts any host on GB10 is an open native check.

Admission runs again from a fresh snapshot after the download, and last immediately before the runner starts, after the runner is prepared and the weights are checked for the last time (DS003). Memory taken meanwhile refuses the start as `insufficient-now`: the guard's floor is far below a run's need, so it would not stop a runner that starts into too little memory.

**Who may certify an envelope.** `unified.envelope` and the `validated` labels come only from the trusted seed: the catalog shipped with the agent, or the operator's `LOCAL_LLM_CATALOG_FILE`, whose provenance is controlled outside the model-management API. `local_llm_model_add` and `_update` refuse an entry that carries an envelope or any `validated` label (`invalid_model`); the empty forms a stored entry carries (`unified: null`, `validated: {}`) are accepted. A registry entry stored with either before this rule is kept in the state file as it is, reported in the overview's `unsupportedModels` and the log, never offered, and can be removed. A user entry that names a seed's own file does not inherit the seed's envelope: an envelope belongs to the catalog entry, not to the artifact, so the alias is sized by estimate. No envelope is shipped in this release, and this rule approves none. An envelope is no longer a prerequisite for a Run; it only replaces the estimate where it applies.

### Memory guard

On unified memory and on the CPU every runner is watched from its start to its exit: MemAvailable is sampled every 250 ms whether the runner is loading, ready or stopping (a graceful Stop or Replace keeps it watched until the process has exited). Below the floor (8 GiB on unified memory; on the CPU, the floor its admission recorded), or when `/proc/meminfo` cannot be read, the runner's whole process group is killed at once (SIGKILL; a runner starved of memory has nothing to save), and the deployment ends in `error` with the reason. PSI memory `full avg10` stops a runner only at 50 % or more together with MemAvailable below twice the floor: page-cache reclaim alone reached 14–28 % with 45 GiB or more available. `local_llm_status` reports the guard's lowest MemAvailable, highest pressure and sample count.

The guard is a backstop, not protection. A load commits memory in bursts of 7–10 GiB per 250 ms, faster than any sampling can follow, and other users of the machine allocate from the same pool at any time. Admission, sized to the whole known allocation, is the control; the kernel OOM killer, which picks the runner first (`oom_score_adj` 1000, DS003), is the last resort. Memory cgroups do not bound GPU allocations here. On the CPU the runner's memory is visible to the kernel. With mmap its weights are page cache, which MemAvailable counts as available, so the guard reacts to anonymous memory and to other processes. If the weights are evicted, generation slows down; the guard does not stop the runner for that. Under a container memory limit, the kernel's cgroup OOM killer also picks the runner first.

### Dashboard

On unified memory the Local LLMs dashboard shows one "Unified memory available" card (MemAvailable of MemTotal, the GPU's users and the page cache, with a meter of what is in use) and the disk card, instead of the GPU and RAM cards, and the run form shows one shared-memory estimate against the pool. No missing value reaches a card. A runner the image lacks, or one without parameters for the profile, cannot be run from the form, which shows why.

In the `cpu` profile the dashboard shows three cards:

- a Compute card: the CPU, its cores, and why no NVIDIA GPU is used, led by the decision's cause;
- a Memory card: what is available of the pool, with a meter and the floor;
- the disk card.

The run form shows one memory estimate against what is available now (under a container memory limit, what the limit leaves). A runner with no CPU policy gets no Models column and no Install, and the Runners tab says why; its intro promises installs only while some runner can be installed under the profile, and the run form's parameter preview gives the same reason. The dedicated and unified profiles add no such refusal. A runner that serves only some profiles (llama.cpp's CPU build) is not listed at all on the others: not in the Runners tab, not as a Models column, and its Install and Run are refused before anything downloads.

## Decisions & Questions

### Question #1: Why a known-unified device list beside the addressing mode?

Response: The addressing mode alone is not a unified-memory signal (GH200 reports ATS with dedicated HBM), and `[N/A]` memory is how this driver reports GB10 today. A future driver that reports numbers for GB10 would otherwise turn it into a dedicated GPU with a pool the size of system memory. The device id keeps GB10 unified whatever the driver reports; the addressing mode covers other GPUs that share memory, which the image then accepts or refuses by compute capability (DS000).

### Question #2: Why can a user entry not carry a unified envelope?

Response: Where an envelope applies it replaces the whole need (weights, buffers, transient), and `validated` tells an admin what was measured. An envelope sent through `local_llm_model_add` or `_update` carries no evidence of where its numbers came from, so it could shrink any configuration's entire need, weights included, on a shared host with only the guard behind it. A user entry's sizing fields are narrower: they feed only the KV and MTP terms of an estimate, never the pinned weights or the fixed terms, and every run they lower carries a warning that only the guard stands behind an understated value. Certification therefore stays with inputs whose provenance the operator controls outside the model-management API: the shipped catalog and `LOCAL_LLM_CATALOG_FILE`. Admin-measured envelopes would need a recorded owner decision and a provenance field; this release has neither. Stored entries are reported rather than migrated or deleted, because they are the admin's data.

### Question #3: Why is the load mode part of an envelope rectangle?

Response: The two unified load modes use memory very differently: `none` added 12.1 GiB of page cache for gpt-oss-20b and 37 GiB for gpt-oss-120b, against 0.8–0.9 GiB for `dio`. A rectangle measured with one mode says nothing about the other, so each names the mode it was measured with and applies only to that mode. There is no inference from an older envelope; a mode no rectangle covers is sized by estimate.

### Question #4: Why is no benchmark a prerequisite for a Run?

Response: The owner decided on 2026-09-29 that users must be able to add and run models at run time without per-model benchmarks, and deferred the benchmark phases; the earlier measurements were disturbed by another user's fine-tuning and stay historical evidence, not calibration. So every run is sized by a labelled estimate from the model's data (or by a trusted envelope where one applies) and checked against a fresh snapshot with the host reserve and the floor, and the guard remains the backstop. An estimate can be wrong; it says so, names its defaults, and never presents itself as measured.

### Question #5: Why does vLLM on unified memory need an operator switch when llama.cpp does not?

Response: The owner chose experimental, opt-in vLLM on DGX Spark. vLLM preallocates its whole share of the pool for weights and KV cache, and its behaviour on GB10 (the share it can actually take, its host-side memory, any network access) has not been checked on the hardware. An explicit operator switch, outside the model-management API, keeps that choice with whoever runs the deployment; a model entry or a Run parameter cannot make it.

### Question #6: Why is MTP a declared capability rather than a parameter any model may set?

Response: MTP speculative decoding needs a multi-token-prediction head in the weights; llama.cpp cannot use it on a model without one. Honouring an explicit value must not turn an unsupported configuration into a Run that fails at load or silently misbehaves, so the entry declares the capability and the parameter is refused elsewhere. Declaring a capability is not certifying memory figures, so user entries may declare it.

### Question #7: Why is `cpu` decided at once when no GPU is attached, but only after 60 s when the GPU cannot be read?

Response: A container cannot gain a GPU without being recreated, so committing then is safe. A failing nvidia-smi or device query can be transient; committing on the first failure would lock DGX Spark or a GPU host into the CPU profile.

### Question #8: Why a smaller reserve, floor and prompt cache on the CPU?

Response: The first CPU host has 5.76 GiB, where the unified values (16 GiB reserve, 8 GiB floor) would refuse everything. The proportional values converge to the unified ones on large hosts.

### Question #9: Why is llama.cpp's CPU build hidden on the GPU profiles by its own declaration?

Response: `profileUnsupportedReason` is set only on the `cpu` profile, on purpose: the dedicated and unified overviews must stay what they were, so a runner cannot be hidden there by a reason that only the new profile produces. A runner that serves one profile says so itself (`profiles`), and the overview, the table columns, the per-model rows, Run, preview and Install all read that one declaration. A test compares both GPU profiles' overviews with a golden file made before the runner existed.

## Conclusion

The agent picks the dedicated, unified or cpu profile from what nvidia-smi reports or from its absence. It runs models on the CPU whenever no NVIDIA GPU is usable, and it waits only while a GPU is present but unreadable. It normalizes, admits, launches and guards each Run under one committed profile. It admits unified and CPU runs, bundled or added at run time, from a labelled estimate (or, on unified memory, a trusted envelope), checked against a fresh snapshot with a host reserve and a floor and rechecked immediately before the runner starts, with a 250 ms guard as the backstop. vLLM joins only as an experimental operator choice on unified memory.
