// Point-in-time view of the resources a model competes for: GPU memory (and
// the other processes holding it), system RAM and free disk under /data.
// Re-read immediately before every launch, because the host Ollama service
// and other granted workspaces share the same GPU.

import { execFile } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';

// The Box GPU grant binds the host nvidia-smi here (the override is a test seam).
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

export async function readGpu({ execFileImpl = execFile, nvidiaSmi = NVIDIA_SMI } = {}) {
    const query = await run(execFileImpl, nvidiaSmi, [
        '--query-gpu=name,memory.total,memory.used,memory.free,driver_version',
        '--format=csv,noheader,nounits',
    ]);
    if (!query.ok) {
        const reason = query.error?.code === 'ENOENT'
            ? 'No GPU is available to this agent: the workspace Box has no GPU grant for it '
                + '(on the host: ploinky gpu grant nvidia --agent local-llms/local-llm).'
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
