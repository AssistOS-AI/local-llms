---
title: DS003-gpu-and-resources
summary: Manifest-declared GPU access and the operator override, the container's hardware limits (memory, CPU quota, GPU share) and the plug-in that applies them to the engine's admission, and runner isolation; the engine's snapshot, admission and estimates are in the pinned pworker specification.
---

# DS003 GPU and Resources

## Introduction

This specification defines how `local-llm` reaches the GPU, how the container's hardware limits narrow what a model may use, how it decides whether a model fits before downloading anything, and how the runner processes are isolated. The hardware snapshot, admission and the runner adapters belong to the engine in the Ploinky-Worker repository (pworker), which this repository pins (DS001); its behavior is specified in [pworker DS003](https://github.com/AssistOS-AI/Ploinky-Worker/blob/4668a3f0e2cae0c85463945def068741dc8c2b68/docs/specs/DS003-gpu-and-resources.md) at commit `4668a3f0e2cae0c85463945def068741dc8c2b68`, and the hooks it offers to a host in [pworker DS007](https://github.com/AssistOS-AI/Ploinky-Worker/blob/4668a3f0e2cae0c85463945def068741dc8c2b68/docs/specs/DS007-host-limits-and-messages.md). The engine applies no limit of its own. Every hardware limit lives in this repository and reaches the engine through the limits plug-in below. This specification replaces the operator-flag approach of the retired repository DS010 ([at `03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS010-gpu-resource-policy.md)) with a grant that Ploinky enforces.

## Core Content

### GPU access through the manifest, with the operator override

The manifest declares `containerSecurity.gpu: true` (Ploinky decision D14), so the agent gets the GPU in any workspace where its repo is installed and the host has a usable NVIDIA GPU, with no host step. When the host prepares the Box (`ploinky start`, `restart`, `update`, `gpu grant`, `gpu revoke`) and the host has a usable NVIDIA GPU, it recreates that workspace's Box with the NVIDIA device nodes (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`), read-only binds of the driver libraries under `/usr/local/nvidia/lib64` and of `nvidia-smi` under `/usr/local/nvidia/bin`, a hookless CDI spec `ploinky.local/gpu=all`, and a grant marker naming the agent. The operator overrides the manifest on the host: `ploinky gpu revoke --agent local-llms/local-llm` denies it persistently, `ploinky gpu grant --agent local-llms/local-llm` restores it, `ploinky gpu revoke` without `--agent` turns manifest-declared GPU access off for the whole workspace, and `ploinky gpu status` shows each GPU agent with its source (Ploinky `ploinky-box/gpuGrant.mjs`).

The declaration asks for that CDI device and nothing else. Ploinky attaches it only while the Box's marker names the agent and is active, and checks again before every launch. Otherwise the agent still starts, without the device, and Ploinky passes `PLOINKY_GPU_STATUS=unavailable` and `PLOINKY_GPU_REASON`, which says what to run on the host: `ploinky gpu grant --agent local-llms/local-llm` after a revoke, `ploinky start` when the repo was installed after the Box was last prepared, or `ploinky gpu status` on a host without a usable GPU. The hardware snapshot reports that reason, so Settings, `local_llm_overview` and `local_llm_status` show it, and the agent runs models on the CPU instead (the `cpu` profile, DS005; the user reversed plan decision D7 on 2026-09-30). The agent therefore never fails to start, and never refuses to run only because no NVIDIA GPU is usable, which matters because Explorer enables it by default. A revoke therefore moves models to the CPU instead of stopping them. If the GPU disappears while the agent runs under a GPU profile, the snapshot reports it unavailable and every Run is refused as `incompatible` until local-llm restarts, which then decides the `cpu` profile.

### Hardware snapshot

After NVIDIA is positively absent, Ploinky may attach exactly one Vulkan render node through the same named-agent CDI grant. The node must be a character device under `/dev/dri`, readable and writable by every user. A 0660 render node is refused with a host udev fix message; the Box receives no supplementary group or new privilege. RADV uses an amdgpu render node on Linux. Venus uses the libkrun machine's virtio-gpu render node. `/dev/kfd` and display card nodes are not granted.

The snapshot reader, the Vulkan probe, the APU admission counting and the fall back to the CPU profile are the engine's (pworker DS003 and DS005). Two inputs are this agent's: Ploinky's grant decides which render node is attached, and the image's Vulkan contract (`/opt/local-llm/source.contract`, with the Vulkan server, `vulkaninfo` and ICD directory that `src/agentHost.mjs` names) decides whether that node is usable. A finite or unknown cgroup memory budget replaces an enabled APU with an unusable GPU (`decorateSnapshot`, below), so profile selection falls back to the CPU before launch.

Vulkan probes run with a fresh environment containing only PATH, HOME, LANG, the explicit ICD files, disabled loader layers and disabled Mesa shader caching. Runner launches use one selected `VK_DRIVER_FILES`, `VK_LOADER_LAYERS_DISABLE=~all~`, and `MESA_SHADER_CACHE_DIR=/opt/runners/.mesa-cache`, alongside PATH, HOME, LANG and CUDA_CACHE_PATH. The Mesa cache stays outside `/data`. Vulkan launches receive no LD_LIBRARY_PATH, GGML_VK_VISIBLE_DEVICES, MPS variables or agent secrets. llama.cpp's Vulkan build uses loopback port 18086.

### What a snapshot reads

The engine's `hardware.mjs` reads the physical machine. The limits plug-in adds the container's budget to the same snapshot through `observeSnapshot` and `decorateSnapshot`, and caps the core counts through `capCores`.

| Value | Source | Read by |
| --- | --- | --- |
| GPU name, total, used and free memory, driver (and, when it cannot be used, its `state`: `absent`, `unusable` or `unreadable`; DS005) | `nvidia-smi --query-gpu` (path `/usr/local/nvidia/bin/nvidia-smi`; `LOCAL_LLM_NVIDIA_SMI` overrides it for tests, and a bare command name is looked up in `PATH`) | engine |
| PCI device id, compute capability, addressing mode, utilization, power, temperature | a second, optional `nvidia-smi --query-gpu` (DS005) | engine |
| Other GPU users | `nvidia-smi --query-compute-apps` | engine |
| Total and available RAM | `/proc/meminfo` | engine |
| Physical CPU cores | `physicalCoreCount()` (`cores`; DS005), then capped by the CPU quota | engine, capped by the limits' `capCores` |
| The container's memory limit and use, when a limit is set | cgroup v2 `memory.max` and `memory.current` (`cgroupMemory`; null for `max` or when unreadable; DS005). Admission reads the limit separately: an absent `memory.max` is no limit, while one that exists but cannot be read or is malformed is unknown, and a new model is refused with `budget_unreadable` rather than run as unlimited | limits (`observeSnapshot`, `decorateSnapshot`) |
| The container's CPU quota | cgroup v2 `cpu.max`, in exact fractional CPUs | limits (`capCores`; `decorateSnapshot` keeps it for the overview) |
| Ploinky GPU budget, when present | Validated `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and device 0 of `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`; configured best-effort per-process budget, separate from physical device capacity (DS005). | limits (`decorateSnapshot`) |
| Free disk under `/data` | `statfs` | engine |

### Admission

`admit()` and the per-runner policies are the engine's: its three answers (`ok`, `incompatible`, `insufficient-now`), the labelled estimates, the disk term, the load modes and the runner policies are specified in pworker DS003 and DS005. This agent's admission is the engine's with the limits plug-in bound, so what follows is what the container's limits change.


Admission keeps physical capacity separate from the effective Ploinky budget. Dedicated and unified policies use RAM constrained by the container limit and current headroom; the CPU policy retains its existing cap provenance. GPU policies use the effective share/free budget while preserving the physical denominator required by runner utilization parameters. Known zero headroom is exhausted memory, not an unknown value. Under a known finite container limit, missing or malformed raw MemTotal or MemAvailable also returns temporary `budget_unreadable` before every adapter dispatch. A cgroup cap cannot substitute for a missing physical reading. Fresh launch-time admission repeats this guard; recovery of valid raw readings can admit the same fixed policy.


### The limits plug-in

`src/hardwareLimits.mjs` implements the engine's `limits` plug-in (pworker DS007). `createAgentLimits({ fsApi, readBudget, qualificationDataProvider })` returns the frozen limits of one engine instance through the engine's `defineLimits`, and `AGENT_LIMITS` is the value this process uses. The cgroup readers and parsers are in `src/controller/ploinkyBudget.mjs`, and the vLLM MPS qualification is in `src/controller/vllmMpsQualification.mjs`. Every key order, warning and refusal text is the one the agent had before the engine moved, so the overview and the admission verdicts are unchanged byte for byte. The engine holds no limit state: the finite memory limit this process has seen belongs to one limits value, keyed by the file system reader, and the snapshot hooks and the memory guard read and update that one map, so a later failure to read `memory.max` never turns an established limit into unlimited for either. `readBudget` and `qualificationDataProvider` are trusted test seams, and production uses neither. The hooks, in the engine's names:

| Hook | What this agent does |
| --- | --- |
| `capCores` | Caps the physical and performance core counts at `ceil(quota / period)` of `cpu.max` (at least 1, never above the count). The count sets llama-server's default `--threads`. |
| `observeSnapshot` | Reads `memory.max` and `memory.current` before the engine reads memory, so a failing memory read still leaves an established limit recorded. |
| `decorateSnapshot` | Adds `cgroupMemory` to the snapshot, keeps the budget facts (memory observation, CPU quota, GPU share) as non-enumerable properties for the other hooks, and replaces a shared-memory (APU) GPU with an unusable one when the limit is finite or unknown. |
| `memoryPool` | Caps the CPU, Venus and APU pool at the container limit, and what is available now at the limit less `memory.current`, with the warning "A container memory limit of X applies." A known limit of zero is a limit. |
| `refuseProfile` | Refuses the APU profile as `incompatible` with constraint C3 under a known finite limit, at admission and again in the APU policy (DS005). |
| `admit` | Before any policy, on every profile including the CPU: a finite limit whose current use or whose raw RAM readings cannot be read is a temporary `budget_unreadable` refusal, and an incomplete GPU budget is `gpu_budget_invalid`. On a Vulkan profile a configured GPU share is `gpu_share_unsupported`. On a dedicated or unified GPU a share needs a verified dedicated-memory GPU (else `gpu_share_unsupported`) and readable total and free memory (else `gpu_budget_unreadable`), and it hands the policies the share-capped GPU view. |
| `effective` | The memory the policies size against: RAM constrained by the container limit and its current headroom. |
| `refuseResourceRequest` | Ollama's pinned GPU layers must fit the free Ploinky GPU budget (`insufficient-now`); vLLM's `gpuMemoryUtilization` times the physical denominator must fit the share (`incompatible`). |
| `overview` | The `budget` section of the overview: exact fractional CPUs, the finite memory limit and the GPU share, only when one is set. |
| `runnerEnvironment` | Removes the three MPS variables from the adapter's environment and adds the validated ones on every profile except the CPU, Vulkan, Venus and APU ones, which get none, and an incomplete budget refuses a GPU start with `gpu_budget_invalid`. |
| `guardSample`, `guardBreach` | The memory guard (DS005). Under a finite limit the guard also samples its headroom; two consecutive unreadable samples stop the runner with `budget_unreadable`, and the APU guard stops it when the limit is finite or unknown. |
| `vllmQualification` | Always bound: vLLM under a GPU share is admitted only for a reviewed tuple (below). The engine's own default, with no function bound, would refuse as `share_unqualified`; this agent keeps the refusal `vllm_mps_unqualified`. |

The engine's module-level readers whose results depend on limits are bound to `AGENT_LIMITS` in the same file, with the names and signatures they had when the engine was in this repository: `readSnapshot`, `physicalCoreCount`, `performanceCoreCount`, `cpuPool`, `admit`, `admitOllama`, `admitVllm`, `admitCpuLlamaServer`, `admitVenusLlamaServer`, `admitApuLlamaServer` and `admitCpuOllama`, plus `readCgroupMemory`, `readCgroupMemoryObservation`, `captureCgroupMemory` and `readCpuQuota`. The tests that call them import these versions. `createAgentController()` in `src/engineOptions.mjs` composes the same limits into the controller (DS001), so production wires every hook without a test-injected seam.


### Runner isolation

| Rule | How |
| --- | --- |
| Loopback only | llama.cpp and ik_llama.cpp get `--host 127.0.0.1`; Ollama gets `OLLAMA_HOST=127.0.0.1:<port>` |
| Fixed ports | Each adapter declares its port; `defaultPorts()` refuses two runners on one port. llama.cpp 18080, ik_llama.cpp 18081, vLLM 18082, TabbyAPI 18083, LM Studio 18084 (its daemon also listens on 127.0.0.1:41343, and its engine on a random loopback port), Ollama 18434, llama.cpp's CPU build 18085 and its Vulkan build 18086, all inside the agent's network namespace |
| Not reachable through the Router | `routerAccess.agentPorts: false` closes the agent-port relay for this agent |
| Per-start credentials | llama.cpp and ik_llama.cpp get a fresh random `--api-key` on every start, and vLLM and TabbyAPI a key of their own; the chat responder reads it from the controller. LM Studio has none headless: its server relies on loopback, the agent's own network namespace and the closed relay, and the controller keeps only its own model loaded (DS001) |
| Minimal environment | Runners get `PATH`, `HOME=/data/home`, `LANG`, `CUDA_CACHE_PATH` and their own variables only; no agent secrets. `CUDA_CACHE_PATH` (`/opt/runners/.cuda-cache`) keeps the CUDA driver's JIT cache in the container's own filesystem, never in `/data`, which other workspace processes can write (DS004). In the `cpu` profile a runner gets no `LD_LIBRARY_PATH` and an empty `CUDA_VISIBLE_DEVICES`, so the CUDA driver library is not loaded even where a GPU is attached but unusable. |
| OOM priority | Every runner's `oom_score_adj` is set to 1000 right after spawn, before it can load anything; its children inherit it. On a GPU that shares system memory, GPU allocations are invisible to RSS, the OOM score and memory cgroups (DS005), so this is what makes an out-of-memory kill pick the runner. A runner that cannot be marked is killed; one that already exited is ignored. On the CPU the runner's memory is ordinary process memory; the mark still makes it the first victim, which matters on small hosts without swap (the Podman machine on a Mac has none). |
| Reaped on stop and drain | Each runner leads its own process group. Stop sends SIGTERM to the group, then SIGKILL after the grace period; when the runner exits, anything left in its group is killed, so helper processes a runner starts never outlive it |


The engine specification carries the same isolation table with the host's fields in place of these values (`dataDir`, `runRoot`).


### Reuse from repository DS010

The retired repository specification is kept at commit [`03697765`](https://github.com/AssistOS-AI/local-llms/blob/03697765a35d92adf4c7db1be9616c0b5a64cb63/docs/specs/DS010-gpu-resource-policy.md); the table records what local-llm took from it.

| DS010 rule | local-llm |
| --- | --- |
| RAM floor checked at startup from a catalog band | Replaced by per-deployment admission against measured or estimated needs |
| GPU passthrough is an operator runtime flag; the manifest cannot express it | Replaced by the manifest declaration `containerSecurity.gpu` with the operator's grant and revoke as override (DS010 Question #1, option 3: manifest field plus operator control) |
| `nvidia-smi` and `/dev/nvidia0` as GPU signals | Kept as the hardware snapshot source, read from the granted paths |
| One loaded model at a time to limit contention | Kept: one deployment at a time; Ollama runs with `OLLAMA_MAX_LOADED_MODELS=1` |


### Ploinky GPU budgets

The SM percentage and device-memory share are parsed together with the managed MPS pipe. A missing value, unsupported device selector, malformed number, overflowing byte count or unsupported pipe produces one bounded `gpu_budget_invalid` reason before runner admission. Unified-memory or unknown-memory-model GPUs refuse a configured share. Raw hardware telemetry remains unchanged; the effective dedicated GPU view is capped to the configured share, while `deviceTotalBytes` retains physical capacity.

GPU runners additionally receive the validated `CUDA_MPS_PIPE_DIRECTORY`, `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` values. CPU-profile runners receive none. No other agent environment or credential is forwarded. The controller adds those three values through the existing minimal runner environment for every GPU adapter.

Dedicated vLLM calculates utilization against physical device memory. A fraction exceeding the effective share is permanently incompatible; a fraction that fits but cannot be allocated against current effective free memory is insufficient-now. Defaults reserve the existing margins and do not multiply the share twice. Ollama explicit layers must fit the effective capacity and current free budget. Automatic layers spill to RAM according to effective free bytes.

vLLM under MPS requires an exact reviewed tuple of canonical selected runner-lock entry digest, driver version, PCI device id, compute capability and physical device bytes. The selected lock entry contains pinned files and checks, and the installer verifies its runnable copy. The packaged data holds one candidate entry, the RTX 3060 Laptop tuple of vLLM 0.30.0 from a stage-one calibration, whose stage-two qualification (public admission, model readiness, a text response, resource observation and cleanup for that exact tuple) is pending; the resolver matches it for exactly that tuple, and no other is packaged. Missing or mismatched facts yield `vllm_mps_unqualified` before launch, with a fix to use a qualified runner or clear the GPU share. There is no environment, model parameter or public API override. Normal vLLM without MPS is unchanged. Only approved calibration followed by normal model readiness, text-response and cleanup evidence for the exact tuple can add a production qualification; the engine's vLLM adapter has no qualification method of its own, so the limits' `vllmQualification` hook is the only place the tuple is checked. Unit-test providers do not qualify hardware.

The calibration is `tools/vllm_mps_calibration.mjs`, run inside the agent container by the live runner and never by the controller. `prerequisites` checks the image's runner lock (a vLLM entry for linux/amd64 with CUDA wheels, the interpreter's wheel tags, the driver against the wheels' CUDA runtime, the toolchain, free disk, the operator's pins) and reports a blocker with its evidence for anything missing. `calibrate` launches no model: it reads the tuple with production's own functions (`vllmMpsTuple` and `vllmRunnerLockDigest` from `src/controller/vllmMpsQualification.mjs`, and `readGpu` bound to this agent in `src/engineOptions.mjs`), asks the installed torch and the driver for their total memory under the saved limit and under a tighter one (torch's `torch.cuda.mem_get_info`, its device properties and `torch.accelerator.get_memory_info(0)`, the call vLLM 0.30.0 reads its denominator through, each recorded raw as `(free, total)` next to the driver's pair; a torch without that API is a blocking prerequisite with its error), records the installed version's sizing source lines, builds the final argv with the engine's vLLM adapter (`createVllmRunner` from the unstable internal subpath, DS001) and its `buildLaunch` over admission's utilization, which it takes from the agent-bound `admitVllm`, and writes one evidence document with its digest. The denominator is physical only when the total does not follow the pinned limit and matches the device's usable memory. Two declared checks cover the accelerator view under both limits: `torchViewsAgree` (the three torch totals are equal as integers) and `acceleratorAgreesWithDriver` (its total is within 1 MiB of the driver's and its free memory is not above its total); a disagreement fails the calibration. The calibration is qualifiable only when what is installed and queried is what the selected lock entry pins (the vLLM package version, the runnable copy's marker digest, torch's version under both limits, the CUDA runtime and every distribution the entry lists) and when the installed source holds the reviewed sizing statements of that exact vLLM version (`REVIEWED_SIZING`, each statement only in its own file, relative to the vllm package: the requested memory is the device total times `gpu_memory_utilization` in `v1/worker/utils.py`, with the total read in `utils/mem_utils.py`, class `MemorySnapshot`, method `measure`, as the second value of `torch.accelerator.get_memory_info` (vLLM 0.30.0: `free, total = ...`, bound by unpacking or by index `[1]`) or from `mem_get_info`; the same call in another file, such as the sleep-mode path of `v1/worker/gpu_worker.py`, a total taken from the first value, from another position or from another source is not that statement, and the evidence lines always list the hit of each rule); a keyword hit alone is never proof, the statements are matched against executable code only (Python comments and string literals, docstrings included, are removed first, line for line, so a commented-out or quoted expression proves nothing), and an incomplete scan, an unreviewed version or a source of another version blocks it. `render` turns qualifiable evidence into the reviewed entry, which production itself must accept for exactly that tuple; adding it to `REVIEWED_QUALIFICATIONS` is a separate, later step (`applyQualificationEntry`).

The overview adds `limits.budget.gpuShare={smPercent,vramBytes,assurance:"best-effort"}` only when configured. The dashboard adds a Limits from Ploinky line with the per-CUDA-process scope. Unlimited overview, runner argv and telemetry shapes retain their previous output.


### Rationale

#### Limits stay with the container

The engine is embedded by hosts that have no container (pworker is the second host in its DS000), so a cgroup budget or an MPS share has no meaning for it. Everything that depends on the container's budget stays where the container is configured, behind named hooks. A limit that cannot be read never turns into unlimited: admission refuses temporarily with `budget_unreadable` instead of running on unverified headroom.

#### GPU access through the grant

The agent reaches the GPU through its manifest declaration, which the operator can revoke, refuses before downloading when a model cannot fit, keeps runners on loopback behind a per-start key, and never exposes a runner port through the Router.


### Diagnostic provenance

Unavailable GPU observations may include a diagnostic with code, backend, driver, source and a bounded message. Sources are agent-query, image-contract or ploinky-environment. gpu_access_unavailable attributes sanitized text as "Ploinky reports"; gpu_unavailable leaves the host cause unknown and directs the operator to host ploinky gpu status. A render node with no valid declared runtime uses vulkan_runtime_unavailable, which says only that this image lacks a supported declared runtime. Query failure uses vulkan_query_unreadable. Parser, topology, device-match and memory checks emit vulkan_response_invalid, vulkan_device_unsupported, vulkan_device_mismatch and vulkan_memory_invalid respectively. Driver identity is included only after an actual device observation.

The agent cannot observe a wrong host provider or denied grant directly from the current free-text environment protocol. Those explanations remain attributed to Ploinky. CPU architecture, absent nvidia-smi and image Vulkan keys do not establish Apple hardware. No new host probe, grant operation or environment protocol is introduced.
