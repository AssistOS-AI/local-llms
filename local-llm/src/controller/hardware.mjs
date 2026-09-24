// Point-in-time view of the resources a model competes for: GPU memory (and
// the other processes holding it), system RAM and free disk under /data.
// Re-read immediately before every launch, because the host Ollama service
// and other granted workspaces share the same GPU.

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
    const apps = await run(execFileImpl, nvidiaSmi, [
        '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits',
    ]);
    const processes = apps.ok
        ? csvRows(apps.stdout).map(([pid, processName, usedMemory]) => ({
            pid: Number(pid), name: processName, usedBytes: Number(usedMemory) * MIB,
        }))
        : [];
    return {
        available: true,
        name,
        driverVersion: driver,
        totalBytes: Number(total) * MIB,
        usedBytes: Number(used) * MIB,
        freeBytes: Number(free) * MIB,
        processes,
    };
}

export function readMemory({ fsApi = fs } = {}) {
    const text = fsApi.readFileSync('/proc/meminfo', 'utf8');
    const value = (key) => {
        const match = new RegExp(`^${key}:\\s+(\\d+) kB$`, 'm').exec(text);
        return match ? Number(match[1]) * 1024 : null;
    };
    return { totalBytes: value('MemTotal'), availableBytes: value('MemAvailable'), swapFreeBytes: value('SwapFree') };
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
