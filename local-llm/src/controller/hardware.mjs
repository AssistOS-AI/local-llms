// Point-in-time view of the resources a model competes for: GPU memory (and
// the other processes holding it), system RAM and free disk under /data.
// Re-read immediately before every launch, because the host Ollama service
// and other granted workspaces share the same GPU.
//
// The GPU's memory model decides the hardware profile (DS005): `dedicated`
// when nvidia-smi reports its memory in MiB, `unified` when the GPU shares
// system memory (it reports no memory figures and ATS or HMM addressing, or
// it is on the known-unified list), and anything else is unknown and refused.

import { execFile } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';

// The Box GPU wiring binds the host nvidia-smi here (the override is a test seam).
export const NVIDIA_SMI = process.env.LOCAL_LLM_NVIDIA_SMI || '/usr/local/nvidia/bin/nvidia-smi';
const MIB = 1024 * 1024;

function run(execFileImpl, command, args, timeoutMs = 10_000) {
    return new Promise((resolve) => {
        execFileImpl(command, args, { timeout: timeoutMs, encoding: 'utf8' }, (error, stdout, stderr) => {
            resolve({ ok: !error, stdout: String(stdout || ''), stderr: String(stderr || ''), error });
        });
    });
}

// A memory figure in MiB, or NaN when nvidia-smi gives none ("[N/A]", "Not Supported", empty).
function mib(text) {
    return /^\d+(?:\.\d+)?$/.test(text) ? Number(text) * MIB : Number.NaN;
}

// An optional telemetry figure: a number, or null when nvidia-smi gives none.
function optionalNumber(text) {
    return typeof text === 'string' && /^\d+(?:\.\d+)?$/.test(text) ? Number(text) : null;
}

// GPUs that share system memory whatever their driver reports (PCI device id
// as nvidia-smi prints it): NVIDIA GB10 (DGX Spark).
export const KNOWN_UNIFIED_DEVICE_IDS = Object.freeze(['0x2E1210DE']);
const UNIFIED_ADDRESSING = Object.freeze(['ATS', 'HMM']);
// The second, optional query (DS005). The first query stays exactly as it was,
// so an older driver that does not know a field here loses only these values.
export const DEVICE_QUERY_FIELDS = 'pci.device_id,compute_cap,addressing_mode,utilization.gpu,power.draw,temperature.gpu';

async function readDevice(execFileImpl, nvidiaSmi) {
    const query = await run(execFileImpl, nvidiaSmi, [`--query-gpu=${DEVICE_QUERY_FIELDS}`, '--format=csv,noheader,nounits']);
    if (!query.ok) return null;
    const [row] = csvRows(query.stdout);
    if (!row || row.length < 6) return null;
    const [pciDeviceId, computeCap, addressingMode, utilization, powerDraw, temperature] = row;
    return {
        device: {
            pciDeviceId: /^0x[0-9A-Fa-f]{8}$/.test(pciDeviceId) ? pciDeviceId.toUpperCase().replace(/^0X/, '0x') : null,
            computeCapability: /^\d+\.\d+$/.test(computeCap) ? computeCap : null,
            addressingMode: /^[A-Za-z]+$/.test(addressingMode) ? addressingMode : null,
        },
        telemetry: {
            utilizationPercent: optionalNumber(utilization),
            powerWatts: optionalNumber(powerDraw),
            temperatureC: optionalNumber(temperature),
        },
    };
}

/**
 * The memory model of a GPU from its memory figures and device facts:
 * `dedicated`, `unified` or `unknown`. Never inferred from the CPU
 * architecture: GH200 reports ATS with numeric HBM and is dedicated.
 */
export function memoryModelOf({ memoryNumeric, device }) {
    const knownUnified = KNOWN_UNIFIED_DEVICE_IDS.includes(device?.pciDeviceId);
    if (knownUnified) return 'unified';
    if (memoryNumeric) return 'dedicated';
    return UNIFIED_ADDRESSING.includes(device?.addressingMode) ? 'unified' : 'unknown';
}

function csvRows(text) {
    return text.split('\n').map((line) => line.trim()).filter(Boolean)
        .map((line) => line.split(',').map((cell) => cell.trim()));
}

export async function readGpu({ execFileImpl = execFile, nvidiaSmi = NVIDIA_SMI, env = process.env } = {}) {
    const query = await run(execFileImpl, nvidiaSmi, [
        '--query-gpu=name,memory.total,memory.used,memory.free,driver_version',
        '--format=csv,noheader,nounits',
    ]);
    if (!query.ok) {
        // Ploinky starts this agent without the GPU device when the Box has no
        // GPU for it, and says why (containerSecurity.gpu, Ploinky D14).
        const attachReason = env.PLOINKY_GPU_STATUS === 'unavailable' && env.PLOINKY_GPU_REASON
            ? String(env.PLOINKY_GPU_REASON).slice(0, 500)
            : null;
        const reason = query.error?.code === 'ENOENT'
            ? (attachReason
                ? `No GPU is attached to this agent: ${attachReason}`
                : 'No GPU is available to this agent. On the host, `ploinky gpu status` shows why: '
                    + 'revoked: run `ploinky gpu grant --agent local-llms/local-llm`; '
                    + 'not applied yet: run `ploinky start`; or the host has no usable GPU.')
            : `nvidia-smi failed: ${(query.stderr || query.error?.message || '').trim().slice(0, 300)}`;
        return { available: false, reason };
    }
    const [first] = csvRows(query.stdout);
    if (!first || first.length < 5) return { available: false, reason: 'nvidia-smi reported no GPU' };
    const [name, total, used, free, driver] = first;
    const memory = { totalBytes: mib(total), usedBytes: mib(used), freeBytes: mib(free) };
    const memoryNumeric = Object.values(memory).every(Number.isFinite);
    const facts = await readDevice(execFileImpl, nvidiaSmi);
    const memoryModel = memoryModelOf({ memoryNumeric, device: facts?.device });
    if (memoryModel === 'unknown') {
        // Admission sizes every runner from the memory figures, so a GPU that
        // gives none and is not known to share system memory is unknown, and
        // every Run is refused rather than admitted blind.
        return {
            available: false,
            name,
            driverVersion: driver,
            ...(facts || {}),
            reason: `nvidia-smi reports no memory figures for ${name} (total ${total}, used ${used}, free ${free}), `
                + 'so this agent cannot size models for it and refuses every Run. '
                + 'GPUs that share system memory are supported only when the driver reports ATS or HMM addressing.',
        };
    }
    const apps = await run(execFileImpl, nvidiaSmi, [
        '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits',
    ]);
    const processes = apps.ok
        ? csvRows(apps.stdout).map(([pid, processName, usedMemory]) => ({
            pid: Number(pid), name: processName, usedBytes: Number(usedMemory) * MIB,
        }))
        : [];
    if (memoryModel === 'unified') {
        // No GPU memory figures: the pool is system memory, read from
        // /proc/meminfo by the snapshot (never from nvidia-smi or cudaMemGetInfo).
        return {
            available: true,
            name,
            driverVersion: driver,
            memoryModel,
            totalBytes: null,
            usedBytes: null,
            freeBytes: null,
            processes,
            ...facts,
        };
    }
    return {
        available: true,
        name,
        driverVersion: driver,
        ...memory,
        processes,
        memoryModel,
        ...(facts || {}),
    };
}

export function readMemory({ fsApi = fs } = {}) {
    const text = fsApi.readFileSync('/proc/meminfo', 'utf8');
    const value = (key) => {
        const match = new RegExp(`^${key}:\\s+(\\d+) kB$`, 'm').exec(text);
        return match ? Number(match[1]) * 1024 : null;
    };
    return {
        totalBytes: value('MemTotal'),
        availableBytes: value('MemAvailable'),
        swapFreeBytes: value('SwapFree'),
        // For display on unified memory: the page cache the GPU can reclaim.
        freeBytes: value('MemFree'),
        cachedBytes: value('Cached'),
    };
}

/**
 * Memory pressure from /proc/pressure/memory: the `full` avg10 share, or null
 * when the kernel has no PSI. Recorded by the unified memory guard (DS005).
 */
export function readMemoryPressure({ fsApi = fs } = {}) {
    const text = readText(fsApi, '/proc/pressure/memory');
    const match = text && /^full avg10=(\d+(?:\.\d+)?)/m.exec(text);
    return match ? Number(match[1]) : null;
}

function readText(fsApi, file) {
    try {
        return fsApi.readFileSync(file, 'utf8');
    } catch {
        return null;
    }
}

// "0-3,8,10-11" -> [0, 1, 2, 3, 8, 10, 11]
function parseCpuList(text) {
    const cpus = [];
    for (const part of String(text).trim().split(',')) {
        const match = /^(\d+)(?:-(\d+))?$/.exec(part.trim());
        if (!match) return null;
        const first = Number(match[1]);
        const last = match[2] === undefined ? first : Number(match[2]);
        if (last < first || last - first > 4096) return null;
        for (let cpu = first; cpu <= last; cpu += 1) cpus.push(cpu);
    }
    return cpus.length ? cpus : null;
}

/**
 * Physical CPU cores this process may run on: the CPUs its affinity or
 * cpuset allows, counted once per (package, core) so SMT siblings share one
 * core, and capped by a cgroup v2 CPU quota. Falls back to /proc/cpuinfo when
 * sysfs has no topology, then to the logical CPU count.
 */
export function physicalCoreCount({ fsApi = fs, availableParallelism = () => os.availableParallelism?.() ?? os.cpus().length } = {}) {
    const logical = Math.max(1, availableParallelism());
    const status = readText(fsApi, '/proc/self/status');
    const allowedMatch = status && /^Cpus_allowed_list:\s*(\S+)\s*$/m.exec(status);
    const allowed = allowedMatch ? parseCpuList(allowedMatch[1]) : null;
    const allowedSet = allowed ? new Set(allowed) : null;

    let cores = null;
    if (allowed) {
        const seen = new Set();
        for (const cpu of allowed) {
            const base = `/sys/devices/system/cpu/cpu${cpu}/topology`;
            const pkg = readText(fsApi, `${base}/physical_package_id`);
            const core = readText(fsApi, `${base}/core_id`);
            if (pkg === null || core === null) {
                seen.clear();
                break;
            }
            seen.add(`${pkg.trim()}:${core.trim()}`);
        }
        if (seen.size) cores = seen.size;
    }
    if (cores === null) {
        const cpuinfo = readText(fsApi, '/proc/cpuinfo');
        const seen = new Set();
        for (const block of (cpuinfo || '').split(/\n\s*\n/)) {
            const field = (name) => new RegExp(`^${name}\\s*:\\s*(\\S+)`, 'm').exec(block)?.[1];
            const processor = field('processor');
            const core = field('core id');
            if (processor === undefined || core === undefined) continue;
            if (allowedSet && !allowedSet.has(Number(processor))) continue;
            seen.add(`${field('physical id') ?? 0}:${core}`);
        }
        cores = seen.size || logical;
    }
    const quota = readText(fsApi, '/sys/fs/cgroup/cpu.max');
    const quotaMatch = quota && /^(\d+)\s+(\d+)\s*$/.exec(quota.trim());
    if (quotaMatch && Number(quotaMatch[2]) > 0) {
        cores = Math.min(cores, Math.max(1, Math.ceil(Number(quotaMatch[1]) / Number(quotaMatch[2]))));
    }
    return cores;
}

/** The llama-server runners' default CPU threads: physical cores minus 2 (runners plan, I2). */
export function defaultThreads(cores) {
    return Math.max(1, cores - 2);
}

/**
 * The CPUs of the highest-capacity class this process may run on, when sysfs
 * `cpu_capacity` shows cores of different capacity (DGX Spark: 10 Cortex-X925
 * at 997-1024 and 10 Cortex-A725 at 718-731); null when every allowed CPU has
 * the same capacity or the kernel does not report it. A CPU within 10 % of the
 * highest capacity belongs to the class. Used by the unified profile only.
 */
export function performanceCoreCount({ fsApi = fs } = {}) {
    const status = readText(fsApi, '/proc/self/status');
    const allowedMatch = status && /^Cpus_allowed_list:\s*(\S+)\s*$/m.exec(status);
    const allowed = allowedMatch ? parseCpuList(allowedMatch[1]) : null;
    if (!allowed) return null;
    const capacities = [];
    for (const cpu of allowed) {
        const text = readText(fsApi, `/sys/devices/system/cpu/cpu${cpu}/cpu_capacity`);
        const value = text === null ? Number.NaN : Number(text.trim());
        if (!Number.isFinite(value) || value <= 0) return null;
        capacities.push(value);
    }
    const highest = Math.max(...capacities);
    const lowest = Math.min(...capacities);
    if (lowest >= highest * 0.9) return null;
    return capacities.filter((value) => value >= highest * 0.9).length;
}

export async function readDisk(dataDir, { statfs = (target) => fs.promises.statfs(target) } = {}) {
    const stats = await statfs(dataDir);
    return { freeBytes: Number(stats.bavail) * Number(stats.bsize), totalBytes: Number(stats.blocks) * Number(stats.bsize) };
}

export async function readSnapshot({ dataDir, execFileImpl, fsApi, statfs } = {}) {
    const [gpu, disk] = await Promise.all([
        readGpu({ execFileImpl }),
        readDisk(dataDir, { statfs }).catch((error) => ({ freeBytes: null, totalBytes: null, error: error.message })),
    ]);
    return {
        at: new Date().toISOString(),
        gpu,
        memory: readMemory({ fsApi }),
        disk,
        cpus: os.availableParallelism?.() ?? os.cpus().length,
    };
}
