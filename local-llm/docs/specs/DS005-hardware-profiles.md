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

Everything this specification adds applies to the unified profile only. The dedicated profile keeps its snapshot values, parameters, defaults, launch arguments, environment and admission exactly as DS003 describes them.

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

Until a runner has a unified-memory policy, admission refuses it on a unified GPU as `incompatible`, naming the runner and the GPU.

### CPU threads

On a CPU whose cores differ in capacity (sysfs `cpu_capacity`), `performanceCoreCount()` counts the allowed CPUs within 10 % of the highest capacity: 10 on DGX Spark (Cortex-X925 at 997–1024, Cortex-A725 at 718–731). When capacities are equal or not reported it returns null. The dedicated default thread count (physical cores minus 2, DS003) is unchanged.

## Decisions & Questions

### Question #1: Why a known-unified device list beside the addressing mode?

Response: The addressing mode alone is not a unified-memory signal (GH200 reports ATS with dedicated HBM), and `[N/A]` memory is how this driver reports GB10 today. A future driver that reports numbers for GB10 would otherwise turn it into a dedicated GPU with a pool the size of system memory. The device id keeps GB10 unified whatever the driver reports; the addressing mode covers other GPUs that share memory, which the image then accepts or refuses by compute capability (DS000).

## Conclusion

The agent picks the dedicated or unified profile from what nvidia-smi reports, refuses a GPU whose memory model it cannot tell, and sizes unified runs from system memory, leaving the dedicated profile as it was.
