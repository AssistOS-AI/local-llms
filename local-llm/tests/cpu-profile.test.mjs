// The `cpu` hardware profile (DS005): chosen automatically when no NVIDIA GPU
// is usable, sized against the machine's RAM, and run by llama.cpp on the CPU.
// Hardware, time and processes are injected; nothing here needs a GPU, a
// container or the network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admissionResult, admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { chatLimits, cpuCompletionBudget, respond } from '../src/chatResponder.mjs';
import { createController } from '../src/controller/deployments.mjs';
import {
    DEVICE_QUERY_FIELDS,
    cpuDefaultThreads,
    readCgroupMemory,
    readGpu,
    readSnapshot,
} from '../src/controller/hardware.mjs';
import {
    UNIFIED,
    UNREADABLE_COMMIT_MS,
    admitCpuLlamaServer,
    cpuFloorBytes,
    cpuHostReserveBytes,
    cpuPool,
    decideProfile,
    estimateCpuLlamaServer,
} from '../src/controller/profiles.mjs';
import { ikLlamaCppRunner } from '../src/runners/ikLlamaCpp.mjs';
import { RUNNERS, getRunner, runnerSummary } from '../src/runners/index.mjs';
import { llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { createLlamaServerRunner } from '../src/runners/llamaServer.mjs';
import { ParamError } from '../src/runners/params.mjs';
import { parseRunnerReport } from '../src/controller/runnerProcess.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';

const MIB = 1024 * 1024;
const SMI = '/usr/local/nvidia/bin/nvidia-smi';
const GPU_QUERY = 'name,memory.total,memory.used,memory.free,driver_version';
const GB10_LINE = 'NVIDIA GB10, [N/A], [N/A], [N/A], 580.159.03\n';
const GB10_DEVICE = '0x2E1210DE, 12.1, ATS, 0, 4.52, 47\n';

// An nvidia-smi double: each answer is text, an Error to fail with, or absent (the query fails with exit 2).
function smi(answers) {
    const calls = [];
    const execFileImpl = (command, args, options, callback) => {
        calls.push(args[0]);
        const key = Object.keys(answers).find((name) => args[0].includes(name));
        const answer = key === undefined ? undefined : answers[key];
        if (answer instanceof Error) callback(answer, '', '');
        else if (answer === undefined) callback(Object.assign(new Error('unexpected query'), { code: 2 }), '', 'Field "x" is not a valid field to query.');
        else callback(null, answer, '');
    };
    return { calls, execFileImpl };
}

const read = (answers, env = {}) => {
    const host = smi(answers);
    return readGpu({ execFileImpl: host.execFileImpl, nvidiaSmi: SMI, env }).then((gpu) => ({ gpu, calls: host.calls }));
};

const enoent = () => Object.assign(new Error('spawn nvidia-smi ENOENT'), { code: 'ENOENT' });
const timedOut = () => Object.assign(new Error('Command failed'), { killed: true, signal: 'SIGKILL', code: null });

function fakeFs(files) {
    return { readFileSync(file) { if (Object.hasOwn(files, file)) return files[file]; throw Object.assign(new Error(`ENOENT: ${file}`), { code: 'ENOENT' }); } };
}

test('readGpu: a missing nvidia-smi is absent, a failed first query is unreadable, an answered unknown memory model is unusable', async () => {
    // Ploinky says why no GPU was attached; the reason is kept whatever the attach status.
    const attached = await read({ [GPU_QUERY]: enoent() }, { PLOINKY_GPU_STATUS: 'unavailable', PLOINKY_GPU_REASON: 'GPU not applied to this Box yet' });
    assert.equal(attached.gpu.available, false);
    assert.equal(attached.gpu.state, 'absent');
    assert.equal(attached.gpu.reason, 'No GPU is attached to this agent: GPU not applied to this Box yet');
    const status = await read({ [GPU_QUERY]: enoent() }, { PLOINKY_GPU_STATUS: 'attached' });
    assert.equal(status.gpu.state, 'absent', 'ENOENT is absent whatever Ploinky reports');
    assert.match(status.gpu.reason, /^No GPU is available to this agent\. On the host, `ploinky gpu status` shows why/);
    // A query that fails, times out or answers nothing may be transient: unreadable.
    const failed = await read({});
    assert.deepEqual([failed.gpu.available, failed.gpu.state], [false, 'unreadable']);
    assert.match(failed.gpu.reason, /^nvidia-smi failed: /);
    const timeout = await read({ [GPU_QUERY]: timedOut() });
    assert.deepEqual([timeout.gpu.available, timeout.gpu.state], [false, 'unreadable']);
    const empty = await read({ [GPU_QUERY]: '' });
    assert.deepEqual([empty.gpu.available, empty.gpu.state, empty.gpu.reason], [false, 'unreadable', 'nvidia-smi reported no GPU']);
    // The device answered, and it is neither known-unified nor ATS or HMM: unusable for good.
    const unusable = await read({ [GPU_QUERY]: 'Some GPU, [N/A], [N/A], [N/A], 580.159.03\n', [DEVICE_QUERY_FIELDS]: '0x11112222, 8.9, None, 0, 1, 30\n' });
    assert.equal(unusable.gpu.available, false);
    assert.equal(unusable.gpu.state, 'unusable');
    assert.equal(unusable.gpu.name, 'Some GPU');
    assert.deepEqual(unusable.gpu.device, { pciDeviceId: '0x11112222', computeCapability: '8.9', addressingMode: 'None' });
    assert.equal(unusable.gpu.reason, 'nvidia-smi reports no memory figures for Some GPU (total [N/A], used [N/A], free [N/A]), '
        + 'so this agent cannot use it and runs models on the CPU. '
        + 'GPUs that share system memory are supported only when the driver reports ATS or HMM addressing.');
});

test('readGpu: a GPU without memory figures whose device query fails is unreadable, not unusable', async () => {
    const failedQuery = await read({ [GPU_QUERY]: GB10_LINE });
    assert.deepEqual(failedQuery.calls, [`--query-gpu=${GPU_QUERY}`, `--query-gpu=${DEVICE_QUERY_FIELDS}`]);
    assert.equal(failedQuery.gpu.available, false);
    assert.equal(failedQuery.gpu.state, 'unreadable');
    assert.equal(failedQuery.gpu.name, 'NVIDIA GB10');
    assert.equal(failedQuery.gpu.driverVersion, '580.159.03');
    assert.equal(failedQuery.gpu.reason, 'nvidia-smi gave no memory figures for NVIDIA GB10 and its device query failed, '
        + 'so its memory model is not known yet');
    // The device query timing out, or answering too few fields, is the same: a verdict waits for a good read.
    for (const device of [timedOut(), '0x2E1210DE, 12.1\n', '']) {
        const { gpu } = await read({ [GPU_QUERY]: GB10_LINE, [DEVICE_QUERY_FIELDS]: device });
        assert.equal(gpu.state, 'unreadable', String(device));
    }
    // The next good read of the same GPU is usable and carries no state.
    const good = await read({ [GPU_QUERY]: GB10_LINE, [DEVICE_QUERY_FIELDS]: GB10_DEVICE, 'query-compute-apps': '' });
    assert.equal(good.gpu.available, true);
    assert.equal(good.gpu.memoryModel, 'unified');
    assert.equal('state' in good.gpu, false, 'a usable GPU keeps today\'s exact shape');
    const dedicated = await read({ [GPU_QUERY]: 'NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 4539, 1433, 595.91.07\n', 'query-compute-apps': '' });
    assert.equal(dedicated.gpu.available, true);
    assert.equal('state' in dedicated.gpu, false);
});

test('CPU default threads: 3 on four cores, cores minus 2 above four, at least 1', () => {
    const expected = new Map([[0, 1], [1, 1], [2, 1], [3, 2], [4, 3], [5, 3], [6, 4], [8, 6], [16, 14]]);
    for (const [cores, threads] of expected) assert.equal(cpuDefaultThreads(cores), threads, `${cores} cores`);
});

test('snapshot: physical cores and the container memory limit are read; max or unreadable means no limit', async () => {
    const files = {
        '/proc/meminfo': 'MemTotal:        6036128 kB\nMemAvailable:    3122576 kB\nSwapFree:              0 kB\nMemFree: 1 kB\nCached: 1 kB\n',
        '/proc/self/status': 'Cpus_allowed_list:\t0-3\n',
    };
    const statfs = async () => ({ bavail: 1000, bsize: 4096, blocks: 2000 });
    const execFileImpl = (command, args, options, callback) => callback(enoent(), '', '');
    const unlimited = await readSnapshot({ dataDir: '/data', execFileImpl, fsApi: fakeFs({ ...files, '/sys/fs/cgroup/memory.max': 'max\n', '/sys/fs/cgroup/memory.current': '123\n' }), statfs });
    assert.equal(unlimited.cgroupMemory, null);
    assert.equal(unlimited.gpu.state, 'absent');
    assert.equal(typeof unlimited.cores, 'number');
    assert.ok(unlimited.cores >= 1);
    const limited = await readSnapshot({ dataDir: '/data', execFileImpl, fsApi: fakeFs({ ...files, '/sys/fs/cgroup/memory.max': '2147483648\n', '/sys/fs/cgroup/memory.current': '536870912\n' }), statfs });
    assert.deepEqual(limited.cgroupMemory, { maxBytes: 2048 * MIB, currentBytes: 512 * MIB });
    assert.equal(readCgroupMemory({ fsApi: fakeFs({}) }), null, 'unreadable');
    assert.deepEqual(readCgroupMemory({ fsApi: fakeFs({ '/sys/fs/cgroup/memory.max': '1073741824\n' }) }), { maxBytes: 1024 * MIB, currentBytes: null });
    assert.equal(readCgroupMemory({ fsApi: fakeFs({ '/sys/fs/cgroup/memory.max': 'garbage\n' }) }), null);
    assert.equal(readCgroupMemory({ fsApi: fakeFs({ '/sys/fs/cgroup/memory.max': '9223372036854771712\n' }) }), null, 'a limit beyond any real memory is none');
});

// ---------------------------------------------------------------- decideProfile

const GIB = 1024 * MIB;
const unusableGpu = { available: false, state: 'unusable', name: 'Some GPU', reason: 'nvidia-smi reports no memory figures for Some GPU (…)' };
const absentGpu = { available: false, state: 'absent', reason: 'No GPU is attached to this agent: GPU not applied to this Box yet' };
const unreadableGpu = { available: false, state: 'unreadable', reason: 'nvidia-smi failed: timed out' };
const gb10 = { available: true, name: 'NVIDIA GB10', memoryModel: 'unified', device: { computeCapability: '12.1', addressingMode: 'ATS' } };
const x86 = { available: true, name: 'NVIDIA GeForce RTX 3060', memoryModel: 'dedicated', totalBytes: 6144 * MIB, device: { computeCapability: '8.6' } };
const thor = { ...gb10, name: 'NVIDIA Thor', device: { computeCapability: '11.0', addressingMode: 'HMM' } };
const decide = (gpu, extra = {}) => decideProfile({ snapshot: gpu === null ? null : { gpu }, nowMs: 1_000_000, ...extra });

test('decideProfile: absent and unusable decide cpu at once; a supported GPU decides dedicated or unified', () => {
    const absent = decide(absentGpu);
    assert.deepEqual([absent.profile, absent.cause, absent.reason, absent.unreadable], ['cpu', 'absent', absentGpu.reason, null]);
    const unusable = decide(unusableGpu);
    assert.deepEqual([unusable.profile, unusable.cause, unusable.reason, unusable.unreadable], ['cpu', 'unusable', unusableGpu.reason, null]);
    assert.deepEqual(decide(x86), { profile: 'dedicated', cause: 'gpu', reason: null, unreadable: null });
    assert.deepEqual(decide(gb10, { capabilities: ['12.1'] }), { profile: 'unified', cause: 'gpu', reason: null, unreadable: null });
    // An image that lists no capabilities (amd64) checks nothing.
    assert.equal(decide(thor).profile, 'unified');
    // A decision that is already positive ignores a stale unreadable window.
    assert.equal(decide(absentGpu, { unreadable: { firstAtMs: 0 } }).unreadable, null);
});

test('decideProfile: a reported compute capability the image was not built for decides cpu with the mismatch reason', () => {
    const decision = decide(thor, { capabilities: ['12.1'] });
    assert.deepEqual([decision.profile, decision.cause], ['cpu', 'mismatch']);
    assert.equal(decision.reason, 'This image\'s runners are built for GPUs of compute capability 12.1; NVIDIA Thor is 11.0.');
    assert.equal(decision.unreadable, null);
    // Any listed capability is enough, and a GPU on the list is not a mismatch.
    assert.equal(decide(thor, { capabilities: ['11.0', '12.1'] }).profile, 'unified');
    assert.equal(decide(x86, { capabilities: ['8.6'] }).profile, 'dedicated');
});

test('decideProfile: a GPU that does not report its compute capability on an image that lists capabilities is unreadable', () => {
    // Numeric memory, and the device query failed: readGpu gives no `device`. That is not a mismatch.
    const noDevice = { ...x86, device: undefined };
    delete noDevice.device;
    const waiting = decide(noDevice, { capabilities: ['12.1'] });
    assert.deepEqual([waiting.profile, waiting.cause], [null, 'unreadable']);
    assert.match(waiting.reason, /NVIDIA GeForce RTX 3060 did not report its compute capability\.$/);
    assert.deepEqual(waiting.unreadable, { firstAtMs: 1_000_000 });
    const nullCapability = decide({ ...x86, device: { computeCapability: null } }, { capabilities: ['12.1'] });
    assert.equal(nullCapability.profile, null);
    // Without a list nothing is checked, so the same GPU is dedicated at once.
    assert.equal(decide(noDevice, { capabilities: [] }).profile, 'dedicated');
    // Still missing after the window: cpu, with the same reason.
    const late = decide(noDevice, { capabilities: ['12.1'], unreadable: { firstAtMs: 1_000_000 - UNREADABLE_COMMIT_MS } });
    assert.deepEqual([late.profile, late.cause, late.reason], ['cpu', 'unreadable-timeout', waiting.reason]);
});

test('decideProfile: unreadable decides cpu only once 60 s have passed since the first unreadable snapshot', () => {
    assert.equal(UNREADABLE_COMMIT_MS, 60_000);
    const first = decide(unreadableGpu, { nowMs: 5_000 });
    assert.deepEqual([first.profile, first.cause, first.reason], [null, 'unreadable', unreadableGpu.reason]);
    assert.deepEqual(first.unreadable, { firstAtMs: 5_000 });
    // The window keeps its first timestamp while snapshots stay unreadable.
    const before = decide(unreadableGpu, { nowMs: 5_000 + 59_999, unreadable: first.unreadable });
    assert.equal(before.profile, null);
    assert.deepEqual(before.unreadable, { firstAtMs: 5_000 });
    const after = decide(unreadableGpu, { nowMs: 5_000 + 60_000, unreadable: first.unreadable });
    assert.deepEqual([after.profile, after.cause, after.reason], ['cpu', 'unreadable-timeout', unreadableGpu.reason]);
    // No state, no gpu and no snapshot are unreadable too, each with its own wording.
    for (const gpu of [{ available: false, reason: 'legacy' }, {}, null]) {
        const decision = decide(gpu, { nowMs: 0 });
        assert.deepEqual([decision.profile, decision.cause], [null, 'unreadable'], JSON.stringify(gpu));
        const timed = decide(gpu, { nowMs: 60_000, unreadable: decision.unreadable });
        assert.deepEqual([timed.profile, timed.cause], ['cpu', 'unreadable-timeout'], JSON.stringify(gpu));
    }
    assert.equal(decide(null, { nowMs: 60_000, unreadable: { firstAtMs: 0 } }).reason, 'the hardware snapshot could not be read');
    assert.equal(decideProfile({}).profile, null, 'no arguments at all');
    // The window is a parameter: a caller that wants a longer wait gets it.
    assert.equal(decide(unreadableGpu, { nowMs: 100_000, unreadable: { firstAtMs: 0 }, commitAfterMs: 120_000 }).profile, null);
});

test('decideProfile: a readable snapshot resets the unreadable window', () => {
    const t0 = decide(unreadableGpu, { nowMs: 0 });
    assert.deepEqual(t0.unreadable, { firstAtMs: 0 });
    // A good read half a minute later resets it ...
    const good = decide(gb10, { nowMs: 30_000, unreadable: t0.unreadable });
    assert.deepEqual([good.profile, good.unreadable], ['unified', null]);
    // ... so a failure at 70 s starts a new window and is not yet cpu, although 70 s passed since the first one.
    const again = decide(unreadableGpu, { nowMs: 70_000, unreadable: good.unreadable });
    assert.deepEqual([again.profile, again.unreadable], [null, { firstAtMs: 70_000 }]);
    assert.equal(decide(unreadableGpu, { nowMs: 70_000 + 59_999, unreadable: again.unreadable }).profile, null);
    assert.equal(decide(unreadableGpu, { nowMs: 70_000 + 60_000, unreadable: again.unreadable }).profile, 'cpu');
    // A mismatch decision and an absent one reset it as well.
    assert.equal(decide(thor, { capabilities: ['12.1'], unreadable: t0.unreadable }).unreadable, null);
    assert.equal(decide(absentGpu, { unreadable: t0.unreadable }).unreadable, null);
});

// ---------------------------------------------------------------- CPU admission

// The M1 Mac's Podman machine (Observed, SPEC Interfaces §2).
const M1_TOTAL = 6036128 * 1024;
const M1_AVAILABLE = 3122576 * 1024;
const M1_MEMORY = { totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE };
const DISK = { freeBytes: 100_000 * MIB, totalBytes: 400_000 * MIB };
const CPU_PARAMS = { ctxSize: 4096, parallel: 1 };
const absentDecision = { profile: 'cpu', cause: 'absent', reason: absentGpu.reason };
const cpuModel = (id, sizeMiB, { layers, kv, contextLength = 32768, seed = true } = {}) => ({
    id, displayName: id, seed, contextLength, architecture: 'dense', memory: { layers, kvBytesPerToken: kv },
});
const cpuSource = (sizeMiB) => ({ type: 'huggingface', size: Math.round(sizeMiB * MIB) });
const admitOn = ({ model, source, params = CPU_PARAMS, memory = M1_MEMORY, ...rest }) => admitCpuLlamaServer(
    { model, source, params, memory, disk: DISK, remainingDownloadBytes: 0, gpu: absentGpu, decision: absentDecision, ...rest }, admissionResult);
const mib = (bytes) => bytes / MIB;
const near = (actual, expected, message) => assert.ok(Math.abs(actual - expected) < 0.01, `${message}: ${actual} is not ${expected}`);

test('CPU admission on the M1 Podman machine figures: ok, insufficient-now and incompatible', () => {
    // The reserve and the floor of a 5.76 GiB machine.
    assert.equal(cpuHostReserveBytes(M1_TOTAL), 1536 * MIB);
    assert.equal(cpuFloorBytes(M1_TOTAL), Math.ceil(0.1 * M1_TOTAL));
    near(mib(cpuFloorBytes(M1_TOTAL)), 589.47, 'floor MiB');
    // Both bounds scale up to the unified values on a large host, and are whole bytes.
    assert.equal(cpuHostReserveBytes(125442396 * 1024), 16 * GIB);
    assert.equal(cpuFloorBytes(125442396 * 1024), 8 * GIB);
    assert.equal(cpuHostReserveBytes(1_000_000_001), Math.ceil(1.5 * GIB));
    assert.equal(cpuFloorBytes(1.01 * GIB) % 1, 0);

    const small = admitOn({ model: cpuModel('qwen2.5-0.5b', 380, { layers: 24, kv: 12288 }), source: cpuSource(380) });
    assert.equal(small.status, 'ok');
    near(mib(small.estimate.ramBytes), 1412.92, '0.5B need MiB');
    assert.equal(small.estimate.measured, false);
    assert.equal(small.estimate.isEstimate, true);
    const mid = admitOn({ model: cpuModel('qwen2.5-1.5b', 1068, { layers: 28, kv: 28672 }), source: cpuSource(1068) });
    assert.equal(mid.status, 'ok');
    near(mib(mid.estimate.ramBytes), 2164.92, '1.5B need MiB');
    near(mib(M1_AVAILABLE - cpuFloorBytes(M1_TOTAL) - mid.estimate.ramBytes), 295, '1.5B margin MiB');
    const four = admitOn({ model: cpuModel('qwen3-4b', 2384, { layers: 36, kv: 147456 }), source: cpuSource(2384) });
    assert.equal(four.status, 'insufficient-now');
    near(mib(four.estimate.ramBytes), 3944.92, '4B need MiB');
    assert.match(four.reason, /^Needs about 3\.85 GiB of memory and 589 MiB kept free; 2\.98 GiB is available now\./);
    // gpt-oss-20b as the catalog pins it, at its CPU context, cannot ever fit 5.76 GiB.
    const [gpt] = loadSeedCatalog();
    const big = admitOn({ model: gpt, source: gpt.sources.gguf, params: { ctxSize: 8192, parallel: 1 } });
    assert.equal(big.status, 'incompatible');
    assert.match(big.reason, /^Needs about 12\.\d\d GiB of the 5\.76 GiB of memory on this machine, which must keep 1\.50 GiB for the host\./);
    assert.equal(big.estimate.weightsBytes, gpt.sources.gguf.size);
    // The estimate carries what the guard and the cards need.
    assert.deepEqual([small.estimate.poolBytes, small.estimate.hostReserveBytes, small.estimate.floorBytes],
        [M1_TOTAL, 1536 * MIB, cpuFloorBytes(M1_TOTAL)]);
    for (const term of ['weightsBytes', 'kvBytes', 'computeBytes', 'runtimeBytes', 'cacheRamBytes']) {
        assert.ok(Number.isInteger(small.estimate[term]), term);
    }
    assert.equal(small.estimate.runtimeBytes, 512 * MIB);
    assert.equal(small.estimate.cacheRamBytes, 256 * MIB);
    // The disk decides last.
    const noDisk = admitOn({ model: cpuModel('m', 380, { layers: 24, kv: 12288 }), source: cpuSource(380), remainingDownloadBytes: 380 * MIB, disk: { freeBytes: 100 * MIB } });
    assert.equal(noDisk.status, 'insufficient-now');
    assert.match(noDisk.reason, /free disk/);
    // Boundary: exactly the need plus the floor is ok, one byte less is not.
    const model = cpuModel('edge', 380, { layers: 24, kv: 12288 });
    const need = estimateCpuLlamaServer({ model, source: cpuSource(380), params: CPU_PARAMS, pool: cpuPool(M1_MEMORY) }).ramBytes;
    const floor = cpuFloorBytes(M1_TOTAL);
    assert.equal(admitOn({ model, source: cpuSource(380), memory: { totalBytes: M1_TOTAL, availableBytes: need + floor } }).status, 'ok');
    assert.equal(admitOn({ model, source: cpuSource(380), memory: { totalBytes: M1_TOTAL, availableBytes: need + floor - 1 } }).status, 'insufficient-now');
    // And the same for the other bound: the reserve.
    const reserve = cpuHostReserveBytes(M1_TOTAL);
    const fit = { totalBytes: need + reserve, availableBytes: need + reserve };
    assert.equal(admitOn({ model, source: cpuSource(380), memory: fit }).status, 'ok');
    assert.equal(admitOn({ model, source: cpuSource(380), memory: { totalBytes: need + reserve - 1, availableBytes: need + reserve - 1 } }).status, 'incompatible');
});

test('CPU admission: unreadable meminfo and an unpinned size are incompatible', () => {
    const model = cpuModel('m', 380, { layers: 24, kv: 12288 });
    for (const memory of [{ totalBytes: null, availableBytes: M1_AVAILABLE }, { totalBytes: M1_TOTAL, availableBytes: null }, {}, { totalBytes: Number.NaN, availableBytes: 1 }]) {
        const result = admitOn({ model, source: cpuSource(380), memory });
        assert.equal(result.status, 'incompatible', JSON.stringify(memory));
        assert.match(result.reason, /cannot be read \(\/proc\/meminfo\), so nothing can be sized on the CPU/);
    }
    // A cgroup limit does not make unreadable memory readable.
    assert.equal(admitOn({ model, source: cpuSource(380), memory: {}, cgroupMemory: { maxBytes: 4 * GIB, currentBytes: 0 } }).status, 'incompatible');
    for (const source of [{ type: 'huggingface' }, { size: 0 }, { size: null }]) {
        const result = admitOn({ model, source });
        assert.equal(result.status, 'incompatible', JSON.stringify(source));
        assert.match(result.reason, /weight size is unknown, so it cannot be sized on the CPU; add the model again so its files are pinned/);
    }
});

test('CPU admission: a cgroup memory limit caps the pool and what is available', () => {
    // No limit ("max" reads as null): the host's figures.
    assert.deepEqual(cpuPool(M1_MEMORY, null), { totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE, capped: false });
    // A limit above the host's memory changes nothing while the container is small.
    assert.deepEqual(cpuPool(M1_MEMORY, { maxBytes: 64 * GIB, currentBytes: 100 * MIB }), { totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE, capped: false });
    // A limit below MemTotal caps the total; what is available is the smaller of MemAvailable and the limit less use.
    const limited = { maxBytes: 4 * GIB, currentBytes: 2 * GIB };
    assert.deepEqual(cpuPool(M1_MEMORY, limited), { totalBytes: 4 * GIB, availableBytes: 2 * GIB, capped: true });
    // Without a reading of the use, available is at most the limit.
    assert.deepEqual(cpuPool({ totalBytes: 8 * GIB, availableBytes: 7 * GIB }, { maxBytes: 4 * GIB, currentBytes: null }),
        { totalBytes: 4 * GIB, availableBytes: 4 * GIB, capped: true });
    const small = admitOn({ model: cpuModel('a', 380, { layers: 24, kv: 12288 }), source: cpuSource(380), cgroupMemory: limited });
    assert.equal(small.status, 'ok');
    assert.equal(small.estimate.poolBytes, 4 * GIB, 'the reserve and the floor come from the capped pool');
    assert.equal(small.estimate.hostReserveBytes, 1536 * MIB);
    assert.equal(small.estimate.floorBytes, cpuFloorBytes(4 * GIB));
    assert.ok(small.warnings.includes('A container memory limit of 4.00 GiB applies.'));
    // The 1.5B model fits the machine (see above) but not what is left of a 4 GiB limit with 2 GiB in use.
    const mid = admitOn({ model: cpuModel('b', 1068, { layers: 28, kv: 28672 }), source: cpuSource(1068), cgroupMemory: limited });
    assert.equal(mid.status, 'insufficient-now');
    assert.match(mid.reason, /2\.00 GiB is available now/);
    // A smaller limit makes the same model incompatible for good.
    const tight = admitOn({ model: cpuModel('b', 1068, { layers: 28, kv: 28672 }), source: cpuSource(1068), cgroupMemory: { maxBytes: 2 * GIB, currentBytes: 0 } });
    assert.equal(tight.status, 'incompatible');
    // No limit: no warning.
    assert.equal(admitOn({ model: cpuModel('a', 380, { layers: 24, kv: 12288 }), source: cpuSource(380), cgroupMemory: null })
        .warnings.some((warning) => /container memory limit/.test(warning)), false);
});

test('CPU admission warnings: the decision reason, and a restart hint only after an unreadable GPU recovers', () => {
    const model = cpuModel('m', 380, { layers: 24, kv: 12288 });
    const run = (decision, gpu = absentGpu) => admitOn({ model, source: cpuSource(380), decision, gpu }).warnings;
    const absent = run(absentDecision);
    assert.equal(absent[0], 'Runs on the CPU: no NVIDIA GPU is attached (No GPU is attached to this agent: GPU not applied to this Box yet). '
        + 'Generation is much slower than on a GPU.');
    assert.match(absent[1], /^Not measured on this machine: about 1\.38 GiB is an estimate\. The memory guard stops the runner at once if available memory falls below 589 MiB\.$/);
    // Every cause has its own lead, and a reason is cut at 300 characters.
    const leads = {
        unusable: /^Runs on the CPU: the NVIDIA GPU cannot be used \(why\)\./,
        mismatch: /^Runs on the CPU: this image's CUDA runners were not built for this GPU \(why\)\./,
        'unreadable-timeout': /^Runs on the CPU: the NVIDIA GPU could not be read for 60 s \(why\)\./,
    };
    for (const [cause, pattern] of Object.entries(leads)) assert.match(run({ cause, reason: 'why.' })[0], pattern, cause);
    const long = run({ cause: 'unusable', reason: 'x'.repeat(500) })[0];
    assert.ok(long.includes(`(${'x'.repeat(300)})`) && !long.includes('x'.repeat(301)));
    assert.equal(run(undefined)[0], 'Runs on the CPU: no NVIDIA GPU is used. Generation is much slower than on a GPU.');
    // The restart hint: only for unreadable-timeout, only once the GPU would now be chosen.
    const recovered = { cause: 'unreadable-timeout', reason: 'nvidia-smi failed', gpuRecovered: true };
    assert.equal(run(recovered, gb10).at(-1), 'The NVIDIA GPU is readable now (NVIDIA GB10); restart local-llm to use it.');
    assert.equal(run({ ...recovered, gpuRecovered: false }, gb10).some((warning) => /restart local-llm/.test(warning)), false, 'not chosen now (a mismatch, say)');
    assert.equal(run(recovered, unreadableGpu).some((warning) => /restart local-llm/.test(warning)), false, 'still unreadable');
    for (const cause of ['absent', 'unusable', 'mismatch']) {
        assert.equal(run({ cause, reason: 'r', gpuRecovered: true }, gb10).some((warning) => /restart local-llm/.test(warning)), false, cause);
    }
    // A model entry added at run time carries the defaulted-input and sizing warnings; a seed with full data has none.
    assert.equal(absent.some((warning) => /Estimated without/.test(warning)), false);
    const bare = admitOn({ model: { id: 'u', displayName: 'u', seed: false, memory: {}, contextLength: undefined }, source: cpuSource(380) });
    assert.ok(bare.warnings.some((warning) => /Estimated without memory\.kvBytesPerToken .*contextLength/.test(warning)));
    assert.deepEqual(bare.estimate.defaulted, ['memory.kvBytesPerToken', 'contextLength']);
    const sized = admitOn({ model: { ...cpuModel('u', 380, { layers: 24, kv: 12288 }), seed: false }, source: cpuSource(380) });
    assert.ok(sized.warnings.some((warning) => /^Sized with memory\.kvBytesPerToken from the model entry added at run time/.test(warning)));
});

test('admit(): the cpu profile calls the runner\'s CPU policy before any GPU check, and refuses a runner without one', () => {
    const model = cpuModel('m', 380, { layers: 24, kv: 12288 });
    const calls = [];
    const withCpu = {
        supported: true, displayName: 'Cpu Runner', admit() { throw new Error('the GPU policy must not run on cpu'); },
        admitCpu(input) { calls.push(input); return admissionResult('ok', null, {}); },
    };
    const snapshot = { gpu: absentGpu, memory: M1_MEMORY, cgroupMemory: { maxBytes: 4 * GIB, currentBytes: 0 }, disk: DISK };
    const decision = { cause: 'absent' };
    const ok = admit({ runner: withCpu, model, source: cpuSource(380), params: CPU_PARAMS, snapshot, profile: 'cpu', decision, remainingDownloadBytes: 7 });
    assert.equal(ok.status, 'ok');
    assert.deepEqual(Object.keys(calls[0]).sort(), ['cgroupMemory', 'decision', 'disk', 'gpu', 'memory', 'model', 'params', 'remainingDownloadBytes', 'source']);
    assert.deepEqual([calls[0].decision, calls[0].remainingDownloadBytes, calls[0].cgroupMemory.maxBytes, calls[0].gpu], [decision, 7, 4 * GIB, absentGpu]);
    // A snapshot without any GPU is not a reason to refuse on cpu.
    assert.equal(admit({ runner: withCpu, model, source: cpuSource(380), params: CPU_PARAMS, snapshot: { memory: M1_MEMORY }, profile: 'cpu' }).status, 'ok');
    const withoutCpu = { supported: true, displayName: 'Gpu Runner', admit() { throw new Error('the GPU policy must not run on cpu'); } };
    const refused = admit({ runner: withoutCpu, model, source: cpuSource(380), params: CPU_PARAMS, snapshot, profile: 'cpu', decision });
    assert.equal(refused.status, 'incompatible');
    assert.equal(refused.reason, 'Gpu Runner needs an NVIDIA GPU in this release; on this machine models run on the CPU with the runners listed in the Runners tab.');
    // Without a profile today's path runs: no GPU is refused with the GPU's own reason.
    assert.equal(admit({ runner: withCpu, model, source: cpuSource(380), params: CPU_PARAMS, snapshot }).reason, absentGpu.reason);
});

// ---------------------------------------------------------------- llama.cpp on the CPU

const LLAMA_MODEL = { id: 'm', contextLength: 32768 };
const isParamError = (field) => (error) => error instanceof ParamError && error.details.field === field;
const cpuParams = (params, model = LLAMA_MODEL) => llamaCppRunner.normalizeParams(params, { model, profile: 'cpu' });

test('llama.cpp CPU schema: defaults, bounds and the training-context cap', () => {
    const schema = llamaCppRunner.paramSchemaFor('cpu');
    assert.deepEqual(Object.keys(schema.properties), ['ctxSize', 'parallel', 'loadMode', 'threads', 'chatTemplateKwargs']);
    assert.equal(schema.additionalProperties, false);
    const defaults = cpuParams({});
    assert.deepEqual([defaults.ctxSize, defaults.parallel, defaults.loadMode, defaults.threads], [4096, 1, 'mmap', null]);
    // The other profiles are untouched.
    assert.equal(llamaCppRunner.paramSchemaFor('dedicated').properties.ctxSize.default, 16384);
    assert.equal(llamaCppRunner.paramSchemaFor('unified').properties.ctxSize.default, 32768);
    // Boundaries: ctxSize 511 / 512 / 131072 / 131073.
    assert.throws(() => cpuParams({ ctxSize: 511 }), isParamError('ctxSize'));
    assert.equal(cpuParams({ ctxSize: 512 }).ctxSize, 512);
    assert.equal(cpuParams({ ctxSize: 131072 }, { id: 'm' }).ctxSize, 131072);
    assert.throws(() => cpuParams({ ctxSize: 131073 }, { id: 'm' }), isParamError('ctxSize'));
    // The training context: an explicit value above it is refused, a default above it is lowered, per slot.
    const short = { id: 'short', contextLength: 2048 };
    assert.throws(() => cpuParams({ ctxSize: 4096 }, short), (error) => isParamError('ctxSize')(error) && /training context of 2048 tokens per slot/.test(error.message));
    assert.equal(cpuParams({}, short).ctxSize, 2048, 'the 4096 default is lowered');
    assert.equal(cpuParams({ parallel: 2 }, short).ctxSize, 4096, 'two slots may hold twice the training context');
    // A model's CPU recommendation is the default, and an explicit value still wins.
    const recommended = { id: 'rec', contextLength: 32768, recommended: { cpu: { 'llama.cpp': { ctxSize: 8192, loadMode: 'none' } } } };
    assert.deepEqual([cpuParams({}, recommended).ctxSize, cpuParams({}, recommended).loadMode], [8192, 'none']);
    assert.equal(cpuParams({ ctxSize: 1024 }, recommended).ctxSize, 1024);
    // parallel 0 / 1 / 4 / 5.
    assert.throws(() => cpuParams({ parallel: 0 }), isParamError('parallel'));
    assert.equal(cpuParams({ parallel: 1 }).parallel, 1);
    assert.equal(cpuParams({ parallel: 4 }).parallel, 4);
    assert.throws(() => cpuParams({ parallel: 5 }), isParamError('parallel'));
    // threads 0 / 1 / 256 / 257 / null.
    assert.throws(() => cpuParams({ threads: 0 }), isParamError('threads'));
    assert.equal(cpuParams({ threads: 1 }).threads, 1);
    assert.equal(cpuParams({ threads: 256 }).threads, 256);
    assert.throws(() => cpuParams({ threads: 257 }), isParamError('threads'));
    assert.equal(cpuParams({ threads: null }).threads, null);
    // The memory and device flags are fixed: no parameter offers them, and mlock and dio are not modes here.
    for (const params of [{ nGpuLayers: 10 }, { flashAttn: 'off' }, { cacheTypeK: 'q8_0' }, { batchSize: 64 }, { mtp: true }, { loadMode: 'mlock' }, { loadMode: 'dio' }, { loadMode: 'auto' }]) {
        assert.throws(() => cpuParams(params), (error) => error instanceof ParamError, JSON.stringify(params));
    }
    // ik_llama.cpp has no CPU policy, and the error says which profile.
    assert.equal(ikLlamaCppRunner.paramSchemaFor('cpu'), null);
    assert.throws(() => ikLlamaCppRunner.normalizeParams({}, { model: LLAMA_MODEL, profile: 'cpu' }),
        /ik_llama\.cpp has no parameters for the cpu profile/);
    assert.equal(typeof llamaCppRunner.admitCpu, 'function');
    assert.equal(ikLlamaCppRunner.admitCpu, undefined);
});

test('llama.cpp CPU launch: exact argv, no driver library path, GPUs hidden', () => {
    const key = 'k'.repeat(43);
    const runner = createLlamaServerRunner({
        id: 'llama.cpp', displayName: 'llama.cpp', executable: '/opt/llama.cpp/llama-server', pinnedVersion: 'b11159', port: 18080,
        // The real llama.cpp dialect, as the adapter builds it, with this test's CPU counts.
        dialect: { quietArgs: ['--no-webui', '-lv', '4'], unifiedKv: true, loadArgs: (mode) => (mode && mode !== 'auto' ? ['--load-mode', mode] : []),
            unified: true, cpu: true, jinja: (model) => Boolean(model?.requiresJinja), parseVersion: () => null },
        cpuCores: () => 4, perfCores: () => null,
    });
    const launch = (params, model = LLAMA_MODEL) => runner.buildLaunch({ artifactPath: '/data/models/gguf/m.gguf', params, port: 18080, apiKey: key, model, profile: 'cpu' });
    const base = launch({});
    assert.equal(base.command, '/opt/llama.cpp/llama-server');
    assert.deepEqual(base.args, ['-m', '/data/models/gguf/m.gguf', '--host', '127.0.0.1', '--port', '18080', '--api-key', key, '--no-webui', '-lv', '4', '--alias', 'm',
        '--ctx-size', '4096', '--device', 'none', '--n-gpu-layers', '0', '--fit', 'off', '--flash-attn', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
        '--threads', '3', '-np', '1', '--batch-size', '512', '--ubatch-size', '512', '--cache-ram', '256', '--load-mode', 'mmap']);
    // No library path (the CUDA driver is never loaded) and an empty device list.
    assert.deepEqual(base.env, { CUDA_VISIBLE_DEVICES: '' });
    assert.equal('LD_LIBRARY_PATH' in base.env, false);
    // Optional pieces: --kv-unified above one slot, the load mode, template arguments and --jinja.
    const shaped = launch({ parallel: 2, loadMode: 'none', threads: 2, chatTemplateKwargs: { reasoning_effort: 'low' } }, { ...LLAMA_MODEL, requiresJinja: true });
    assert.deepEqual(shaped.args.slice(shaped.args.indexOf('--threads')),
        ['--threads', '2', '-np', '2', '--kv-unified', '--batch-size', '512', '--ubatch-size', '512', '--cache-ram', '256',
            '--load-mode', 'none', '--chat-template-kwargs', '{"reasoning_effort":"low"}', '--jinja']);
    assert.equal(base.args.includes('--kv-unified'), false);
    // The other profiles keep today's launch: the driver library path, and no CPU flags.
    for (const profile of ['dedicated', 'unified']) {
        const other = runner.buildLaunch({ artifactPath: '/data/models/gguf/m.gguf', params: {}, port: 18080, apiKey: key, model: LLAMA_MODEL, profile });
        assert.deepEqual(other.env, { LD_LIBRARY_PATH: '/usr/local/nvidia/lib64' }, profile);
        assert.equal(other.args.includes('--device'), false, profile);
        assert.equal(other.args.includes('--fit'), false, profile);
    }
    // A runner with no CPU schema cannot be launched on cpu.
    assert.throws(() => ikLlamaCppRunner.buildLaunch({ artifactPath: '/data/models/gguf/m.gguf', params: {}, port: 18081, apiKey: key, model: LLAMA_MODEL, profile: 'cpu' }),
        /ik_llama\.cpp has no parameters for the cpu profile/);
    // start() hands the launch its runner directory (Phase 3 runs a runner from there).
    const seen = [];
    const probe = { ...runner, buildLaunch: (input) => { seen.push(input); return { command: 'x', args: [], env: {} }; } };
    return runner.start({
        runner: probe, weights: { path: '/data/models/gguf/m.gguf' }, params: {}, port: 18080, apiKey: key, model: LLAMA_MODEL, profile: 'cpu', runnerDir: '/opt/runners/x',
        launch: () => ({}), waitForHttp: async () => {},
    }).then(() => {
        assert.deepEqual([seen[0].profile, seen[0].runnerDir], ['cpu', '/opt/runners/x']);
    });
});

test('CPU threads through the runner: the default follows the machine, the admin\'s value wins', () => {
    const dialect = { quietArgs: [], unifiedKv: true, loadArgs: () => [], jinja: () => false, parseVersion: () => null, cpu: true };
    const threads = (cores, { perf = null, params = {} } = {}) => {
        const runner = createLlamaServerRunner({ id: 'llama.cpp', displayName: 'T', executable: '/opt/t/llama-server', pinnedVersion: 'b1', port: 18999, dialect,
            cpuCores: () => cores, perfCores: () => perf });
        const args = runner.buildLaunch({ artifactPath: '/data/models/m.gguf', params, port: 18999, apiKey: 'k'.repeat(43), model: { id: 'm' }, profile: 'cpu' }).args;
        return Number(args[args.indexOf('--threads') + 1]);
    };
    assert.equal(threads(4), 3);
    assert.equal(threads(1), 1, 'a 1-core host');
    assert.equal(threads(2), 1);
    assert.equal(threads(16), 14);
    assert.equal(threads(4, { perf: 10 }), 10, 'cores that differ in capacity: the high-performance ones');
    assert.equal(threads(4, { params: { threads: 2 } }), 2);
    assert.equal(threads(4, { params: { threads: null } }), 3);
    // Dedicated keeps cores minus 2 (defaultThreads) on the same machine.
    const dedicated = createLlamaServerRunner({ id: 'llama.cpp', displayName: 'T', executable: '/opt/t/llama-server', pinnedVersion: 'b1', port: 18999, dialect, cpuCores: () => 4 });
    const args = dedicated.buildLaunch({ artifactPath: '/data/models/m.gguf', params: {}, port: 18999, apiKey: 'k'.repeat(43), model: { id: 'm' } }).args;
    assert.equal(args[args.indexOf('--threads') + 1], '2');
});

test('runner summaries and the vLLM switch on the cpu profile', () => {
    // llama.cpp has a CPU schema; every other runner has none and says why. A decided profile only: undecided adds nothing.
    const summaries = Object.fromEntries(Object.values(RUNNERS).map((runner) => [runner.id, runnerSummary(runner, 'cpu')]));
    assert.ok(summaries['llama.cpp'].paramSchema);
    assert.equal('profileUnsupportedReason' in summaries['llama.cpp'], false);
    for (const id of ['ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio']) {
        assert.equal(summaries[id].paramSchema, null, id);
        assert.equal(summaries[id].profileUnsupportedReason,
            `${RUNNERS[id].displayName} needs an NVIDIA GPU in this release; on this machine models run on the CPU with the runners listed in the Runners tab.`, id);
    }
    for (const runner of Object.values(RUNNERS)) {
        assert.equal('profileUnsupportedReason' in runnerSummary(runner, null), false, `${runner.id} undecided`);
        assert.equal('profileUnsupportedReason' in runnerSummary(runner, 'dedicated'), false, `${runner.id} dedicated`);
    }
    // vLLM's switch is about unified memory: on the CPU it is not gated (admission refuses it), and the other profiles are as before.
    const vllm = getRunner('vllm');
    assert.deepEqual(vllm.enabled({}, 'cpu'), { enabled: true, reason: null });
    assert.equal(vllm.enabled({}, 'unified').enabled, false);
    assert.equal(vllm.enabled({}, null).enabled, false);
    assert.equal(vllm.enabled({}, 'dedicated').enabled, true);
});

// ---------------------------------------------------------------- the controller on cpu

const KEY = 'k'.repeat(43);
const HF = (file, size) => ({ type: 'huggingface', repo: 'acme/models', file, revision: 'main', commit: 'a'.repeat(40), size, sha256: 'b'.repeat(64) });
const SMALL = validateModel({ id: 'small-cpu', profiles: ['cpu', 'dedicated', 'unified'], contextLength: 32768,
    memory: { layers: 24, kvBytesPerToken: 12288 }, sources: { gguf: HF('small.gguf', 400 * MIB) } }, { seed: true });
const BIG = validateModel({ id: 'big-cpu', profiles: ['cpu'], sources: { gguf: HF('big.gguf', 12000 * MIB) } }, { seed: true });
const UNIFIED_ONLY = validateModel({ id: 'big-moe', profiles: ['unified'], contextLength: 131072, sources: { gguf: HF('moe.gguf', 60 * GIB) } }, { seed: true });
const ARM64_CONTRACT = Object.freeze({ architecture: 'arm64', llama_cpp: 'b11159', gpu_compute_capabilities: '12.1' });
const cpuSnapshot = ({ gpu = absentGpu, memory = M1_MEMORY, cgroupMemory = null } = {}) => ({
    gpu: structuredClone(gpu), memory: { ...memory }, cgroupMemory, disk: { ...DISK }, cpus: 4, cores: 4,
});

function deferred() {
    let resolve;
    const promise = new Promise((ok) => { resolve = ok; });
    return { promise, resolve };
}

async function until(predicate, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

// A controller on an injected snapshot, clock, memory and runner process. `dataDir` can be shared by a second controller (a restart).
function harness(t, { snap = () => cpuSnapshot(), seed = [SMALL, BIG, UNIFIED_ONLY], imageContract = ARM64_CONTRACT,
    fileExists = (file) => file === '/opt/llama.cpp/llama-server', registry = null, dataDir = null, clock = null, readMemory,
    readPressure = () => 0, installer } = {}) {
    let dir = dataDir;
    if (!dir) {
        dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-cpu-'));
        t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    }
    if (registry) {
        fs.mkdirSync(path.join(dir, 'state'), { recursive: true });
        fs.writeFileSync(path.join(dir, 'state', 'controller.json'), JSON.stringify({ version: 1, deployment: null, params: {}, requests: {}, registry }));
    }
    const started = [];
    const calls = { download: [], snapshots: 0 };
    const weights = path.join(dir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const controller = createController({
        dataDir: dir,
        env: { PATH: '/usr/bin' },
        seedCatalog: seed,
        stateStore: createStateStore({ dataDir: dir }),
        snapshot: async () => { calls.snapshots += 1; return typeof snap === 'function' ? snap(calls.snapshots) : structuredClone(snap); },
        download: async ({ artifact }) => { calls.download.push(artifact); return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        startRunner({ command, args, env, log }) {
            const exit = deferred();
            let running = true;
            const handle = {
                pid: 7000 + started.length, command, args, env, log, exited: exit.promise,
                get running() { return running; },
                async stop() { running = false; handle.stopped = 'SIGTERM'; exit.resolve({ code: 0, signal: 'SIGTERM', error: null }); return exit.promise; },
                async kill() { running = false; handle.stopped = 'SIGKILL'; exit.resolve({ code: null, signal: 'SIGKILL', error: null }); return exit.promise; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
        imageContract,
        fileExists,
        sharedModelsRoot: null,
        readMemory: readMemory || (() => ({ totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE })),
        readPressure,
        unifiedGuardMs: 5,
        dropCache: () => true,
        ...(clock ? { now: () => new Date(clock.ms) } : {}),
        ...(installer ? { installer } : {}),
    });
    const logText = () => {
        try { return fs.readFileSync(path.join(dir, 'logs', 'runner.log'), 'utf8'); } catch { return ''; }
    };
    const logLines = (text) => logText().split('\n').filter((line) => line.includes(text));
    const stored = () => JSON.parse(fs.readFileSync(path.join(dir, 'state', 'controller.json'), 'utf8'));
    return { controller, started, calls, dataDir: dir, weights, logLines, stored };
}

test('controller: an absent GPU commits cpu at the first overview and logs it once', async (t) => {
    const h = harness(t);
    const overview = await h.controller.overview();
    assert.equal(overview.profile, 'cpu');
    assert.deepEqual([overview.profileDecision.profile, overview.profileDecision.cause], ['cpu', 'absent']);
    assert.equal(overview.profileDecision.reason, absentGpu.reason);
    assert.equal(overview.profileDecision.gpuName, null);
    assert.match(overview.profileDecision.decidedAt, /^\d{4}-\d\d-\d\dT/);
    await h.controller.overview();
    await h.controller.status();
    const lines = h.logLines('hardware profile:');
    assert.equal(lines.length, 1);
    assert.match(lines[0], /hardware profile: cpu \(absent: No GPU is attached to this agent: GPU not applied to this Box yet\)$/);
    // One profile per container: a GPU that becomes usable later does not change it.
    const later = harness(t, { snap: (n) => (n === 1 ? cpuSnapshot() : { ...cpuSnapshot(), gpu: structuredClone(gb10) }) });
    assert.equal((await later.controller.overview()).profile, 'cpu');
    assert.equal((await later.controller.overview()).profile, 'cpu');
    assert.equal(later.logLines('hardware profile:').length, 1);
    // The decision is offered to the cpu profile's models only: a unified-only model is not listed.
    assert.deepEqual(overview.models.map((model) => model.id), ['small-cpu', 'big-cpu']);
    // A missing nvidia-smi answered through the real GPU reader, with Ploinky saying the GPU is attached: still absent, then cpu.
    const attached = harness(t, { snap: async () => ({ ...cpuSnapshot(), gpu: await readGpu({
        execFileImpl: (command, args, options, callback) => callback(enoent(), '', ''), nvidiaSmi: SMI, env: { PLOINKY_GPU_STATUS: 'attached' } }) }) });
    const decision = (await attached.controller.overview()).profileDecision;
    assert.deepEqual([decision.cause, decision.reason.startsWith('No GPU is available to this agent.')], ['absent', true]);
});

test('controller: GPU-only runners are refused on cpu before anything downloads and cannot be installed', async (t) => {
    // No image contract: every runner counts as present, so the refusal is the profile's, not the platform's.
    const installed = [];
    const installer = {
        installable: (id) => ['vllm', 'lmstudio', 'tabbyapi'].includes(id),
        describe: async () => ({ installed: false, runnable: false }),
        entryFor: () => { installed.push('entryFor'); throw new Error('an install must not get this far on cpu'); },
        pathsFor: () => ({}),
    };
    const h = harness(t, { imageContract: null, installer });
    await h.controller.overview();
    const refusal = (name) => `${name} needs an NVIDIA GPU in this release; on this machine models run on the CPU with the runners listed in the Runners tab.`;
    for (const runnerId of ['ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio']) {
        await assert.rejects(() => h.controller.run({ modelId: 'small-cpu', runnerId, requestId: `request-${runnerId.replace(/\W/g, '')}` }),
            (error) => error.code === 'runner_unavailable' && error.message === refusal(getRunner(runnerId).displayName), runnerId);
    }
    assert.equal(h.calls.download.length, 0, 'nothing downloaded');
    assert.equal(h.controller.state.deployment, null, 'nothing recorded');
    assert.deepEqual(h.controller.state.requests, {});
    // Install: a runner with no policy for the profile is refused, before its lock entry is read.
    for (const runnerId of ['vllm', 'lmstudio', 'tabbyapi']) {
        await assert.rejects(() => h.controller.installRunner({ runnerId, acceptLicence: true }),
            (error) => error.code === 'runner_unavailable' && error.message === refusal(getRunner(runnerId).displayName), runnerId);
    }
    assert.deepEqual(installed, []);
});

test('controller: a Run on cpu launches llama.cpp with the CPU flags and is guarded by the CPU floor', async (t) => {
    const h = harness(t);
    const first = await h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' });
    assert.equal(first.accepted, true);
    // Idempotency: the same request is answered as a duplicate and starts nothing more.
    assert.equal((await h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' })).duplicate, true);
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.equal(h.started.length, 1);
    const [process] = h.started;
    assert.equal(h.controller.state.deployment.profile, 'cpu');
    const args = process.args.join(' ');
    for (const flag of ['--device none', '--n-gpu-layers 0', '--fit off', '--flash-attn on', '--cache-type-k f16', '--cache-type-v f16',
        '-np 1', '--batch-size 512', '--ubatch-size 512', '--cache-ram 256', '--load-mode mmap', '--ctx-size 4096']) {
        assert.ok(args.includes(flag), flag);
    }
    assert.ok(process.args.includes('--threads'));
    assert.equal(process.env.CUDA_VISIBLE_DEVICES, '');
    assert.equal('LD_LIBRARY_PATH' in process.env, false);
    assert.equal(process.env.CUDA_CACHE_PATH, '/opt/runners/.cuda-cache');
    assert.equal(h.calls.download.length, 1);
    // The deployment's admission carries the floor the guard watches; the guard reports its samples.
    const floor = cpuFloorBytes(M1_TOTAL);
    assert.equal(h.controller.state.deployment.admission.estimate.floorBytes, floor);
    await until(async () => (await h.controller.status()).memoryGuard?.samples >= 3);
    const status = await h.controller.status();
    assert.deepEqual([status.profile, status.memoryGuard.floorBytes], ['cpu', floor]);
    assert.equal(h.controller.chatTarget().profile, 'cpu');
    await h.controller.stop();
    assert.equal(h.controller.state.deployment.phase, 'idle');
    // A Run that names no model, or a model not offered here, records nothing.
    const stored = Object.keys(h.controller.state.requests).length;
    await assert.rejects(() => h.controller.run({ modelId: 'nope', runnerId: 'llama.cpp', requestId: 'request-0002' }), (error) => error.code === 'unknown_model');
    await assert.rejects(() => h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0003' }), (error) => error.code === 'unknown_model');
    assert.equal(Object.keys(h.controller.state.requests).length, stored, 'nothing recorded');
    // A model that cannot fit is refused before a download.
    await assert.rejects(() => h.controller.run({ modelId: 'big-cpu', runnerId: 'llama.cpp', requestId: 'request-0004' }),
        (error) => error.code === 'admission_incompatible' && /must keep 1\.50 GiB for the host/.test(error.message));
    assert.equal(h.calls.download.length, 1);
});

test('controller: the CPU guard kills below the floor and on pressure with memory already low', async (t) => {
    const floor = cpuFloorBytes(M1_TOTAL);
    const gib1 = (bytes) => `${(bytes / GIB).toFixed(1)} GiB`;
    let available = M1_AVAILABLE;
    const floorRun = harness(t, { readMemory: () => ({ totalBytes: M1_TOTAL, availableBytes: available }) });
    await floorRun.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => floorRun.controller.state.deployment?.phase === 'ready');
    available = floor - 100 * MIB;
    await until(() => floorRun.controller.state.deployment?.phase === 'error');
    assert.equal(floorRun.started[0].stopped, 'SIGKILL');
    assert.equal(floorRun.controller.state.deployment.error, `stopped: host memory below the floor (${gib1(available)} available, ${gib1(floor)} required)`);
    // Boundary: exactly the floor is fine.
    let edge = floor;
    const atFloor = harness(t, { readMemory: () => ({ totalBytes: M1_TOTAL, availableBytes: edge }) });
    await atFloor.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => atFloor.controller.state.deployment?.phase === 'ready');
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.equal(atFloor.controller.state.deployment.phase, 'ready');
    edge = floor - 1;
    await until(() => atFloor.controller.state.deployment?.phase === 'error');
    // Unreadable memory stops the runner: it cannot be watched.
    const blind = harness(t, { readMemory: () => { throw new Error('EACCES'); } });
    await blind.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => blind.controller.state.deployment?.phase === 'error');
    assert.equal(blind.started[0].stopped, 'SIGKILL');
    assert.match(blind.controller.state.deployment.error, /host memory cannot be read/);
    // Pressure alone never stops a runner; with memory below twice the floor it does.
    let plenty = M1_AVAILABLE;
    const pressure = harness(t, { readPressure: () => 80, readMemory: () => ({ totalBytes: M1_TOTAL, availableBytes: plenty }) });
    await pressure.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => pressure.controller.state.deployment?.phase === 'ready');
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.equal(pressure.controller.state.deployment.phase, 'ready');
    plenty = 2 * floor - 100 * MIB;
    await until(() => pressure.controller.state.deployment?.phase === 'error');
    assert.equal(pressure.controller.state.deployment.error, `stopped: memory pressure 80.0 % with ${gib1(plenty)} available`);
    assert.equal(pressure.started[0].stopped, 'SIGKILL');
});

test('controller: the overview and status report the decision, the limits and per-profile refusals', async (t) => {
    const h = harness(t, { imageContract: null });
    const overview = await h.controller.overview();
    assert.equal(overview.profile, 'cpu');
    assert.deepEqual([overview.limits.floorBytes, overview.limits.hostReserveBytes], [cpuFloorBytes(M1_TOTAL), cpuHostReserveBytes(M1_TOTAL)]);
    assert.deepEqual([overview.limits.poolBytes, overview.limits.availableBytes], [M1_TOTAL, M1_AVAILABLE]);
    const byId = Object.fromEntries(overview.runners.map((runner) => [runner.id, runner]));
    assert.ok(byId['llama.cpp'].paramSchema);
    assert.equal('profileUnsupportedReason' in byId['llama.cpp'], false);
    for (const id of ['ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio']) {
        assert.equal(byId[id].paramSchema, null, id);
        assert.match(byId[id].profileUnsupportedReason, /needs an NVIDIA GPU in this release/, id);
    }
    // A model's row for such a runner says why, as an ordinary refusal and not as a parameter error.
    const rows = overview.models.find((model) => model.id === 'small-cpu').runners;
    assert.equal(rows['llama.cpp'].admission.status, 'ok');
    assert.equal(rows['llama.cpp'].params.ctxSize, 4096);
    assert.equal(rows['llama.cpp'].params.loadMode, 'mmap');
    assert.equal(rows['ik_llama.cpp'].params, null);
    assert.equal(rows['ik_llama.cpp'].admission.status, 'incompatible');
    assert.equal(rows['ik_llama.cpp'].admission.reason, byId['ik_llama.cpp'].profileUnsupportedReason);
    assert.doesNotMatch(rows['ik_llama.cpp'].admission.reason, /Invalid parameter|no parameters for/);
    assert.equal(overview.models.find((model) => model.id === 'big-cpu').runners['llama.cpp'].admission.status, 'incompatible');
    // The status carries the same decision, from one snapshot (it used to take two while undecided).
    const fresh = harness(t);
    const before = fresh.calls.snapshots;
    const status = await fresh.controller.status();
    assert.equal(fresh.calls.snapshots - before, 1);
    assert.deepEqual([status.profile, status.profileDecision.cause], ['cpu', 'absent']);
    assert.equal(status.gpu.state, 'absent');
    // Undecided: no decision, no limits, and every runner keeps its place.
    const undecided = harness(t, { imageContract: null, snap: () => cpuSnapshot({ gpu: unreadableGpu }) });
    const cold = await undecided.controller.overview();
    assert.deepEqual([cold.profile, cold.profileDecision, cold.limits], [null, null, null]);
    assert.equal(cold.runners.some((runner) => 'profileUnsupportedReason' in runner), false);
    // The unified profile reports its constants, and the other profiles none.
    const unified = harness(t, { imageContract: null, snap: () => ({ ...cpuSnapshot(), gpu: structuredClone(gb10) }) });
    const uni = await unified.controller.overview();
    assert.deepEqual([uni.profile, uni.profileDecision.cause, uni.limits], ['unified', 'gpu', { floorBytes: UNIFIED.floorBytes, hostReserveBytes: UNIFIED.hostReserveBytes }]);
    const dedicated = harness(t, { imageContract: null, snap: () => ({ ...cpuSnapshot(), gpu: structuredClone(x86) }) });
    const ded = await dedicated.controller.overview();
    assert.deepEqual([ded.profile, ded.limits], ['dedicated', null]);
});

test('controller: stored user entries offered in both GPU profiles are also offered on cpu after the upgrade, once', async (t) => {
    const entry = (id, profiles) => ({ id, ...(profiles ? { profiles } : {}), sources: { gguf: HF(`${id}.gguf`, 300 * MIB) } });
    const registry = [entry('user-both', ['dedicated', 'unified']), entry('user-swapped', ['unified', 'dedicated']), entry('user-one', ['dedicated']),
        entry('user-unified', ['unified']), entry('user-all', ['dedicated', 'unified', 'cpu']), entry('user-default', null)];
    const first = harness(t, { registry });
    const overview = await first.controller.overview();
    const stored = first.stored();
    const profilesOf = (id) => stored.registry.find((item) => item.id === id).profiles;
    assert.deepEqual(profilesOf('user-both'), ['dedicated', 'unified', 'cpu']);
    assert.deepEqual(profilesOf('user-swapped'), ['unified', 'dedicated', 'cpu']);
    assert.deepEqual(profilesOf('user-one'), ['dedicated']);
    assert.deepEqual(profilesOf('user-unified'), ['unified']);
    assert.deepEqual(profilesOf('user-all'), ['dedicated', 'unified', 'cpu']);
    assert.equal('profiles' in stored.registry.find((item) => item.id === 'user-default'), false);
    assert.equal(stored.cpuProfileMigration, '2026-09-30');
    assert.deepEqual(overview.models.map((model) => model.id).sort(), ['big-cpu', 'small-cpu', 'user-all', 'user-both', 'user-default', 'user-swapped']);
    const lines = first.logLines('is now also offered on the cpu profile');
    assert.deepEqual(lines.map((line) => /model entry (\S+) is now/.exec(line)[1]), ['user-both', 'user-swapped']);
    // A restart changes nothing and logs nothing more, even for an entry that lists both GPU profiles again.
    const again = fs.readFileSync(path.join(first.dataDir, 'state', 'controller.json'), 'utf8');
    const restarted = harness(t, { dataDir: first.dataDir });
    await restarted.controller.overview();
    assert.equal(fs.readFileSync(path.join(first.dataDir, 'state', 'controller.json'), 'utf8'), again);
    assert.equal(restarted.logLines('is now also offered on the cpu profile').length, 2, 'one line per entry in total');
    const later = { ...JSON.parse(again), registry: [...JSON.parse(again).registry, entry('added-after', ['dedicated', 'unified'])] };
    fs.writeFileSync(path.join(first.dataDir, 'state', 'controller.json'), JSON.stringify(later));
    const third = harness(t, { dataDir: first.dataDir });
    await third.controller.overview();
    assert.deepEqual(third.stored().registry.find((item) => item.id === 'added-after').profiles, ['dedicated', 'unified'], 'the migration ran once');
    // An empty registry still records the migration.
    const empty = harness(t);
    await empty.controller.overview();
    assert.equal(empty.stored().cpuProfileMigration, '2026-09-30');
});

test('controller: a user entry that shares a seed id is reported, not silently hidden', async (t) => {
    const shadow = { id: 'small-cpu', displayName: 'My own', sources: { gguf: HF('mine.gguf', 100 * MIB) } };
    const fine = { id: 'mine', sources: { gguf: HF('fine.gguf', 100 * MIB) } };
    const h = harness(t, { registry: [shadow, fine, { broken: true }] });
    const overview = await h.controller.overview();
    // The seed is what is offered.
    assert.equal(overview.models.find((model) => model.id === 'small-cpu').displayName, 'small-cpu');
    assert.ok(overview.models.some((model) => model.id === 'mine'));
    const reported = overview.unsupportedModels.find((entry) => entry.id === 'small-cpu');
    assert.match(reported.reason, /its id is also the id of a model that ships with this agent, which takes the id, so this entry is not offered/);
    // The text claims nothing about where the seed is offered: a seed is offered only in the profiles it lists.
    assert.doesNotMatch(reported.reason, /offered instead/);
    assert.equal(overview.unsupportedModels.some((entry) => entry.id === 'mine'), false);
    assert.equal(overview.unsupportedModels.length, 2, 'the shadowed entry and the invalid one');
    assert.equal(h.logLines('model entry small-cpu is not supported by this catalog and is not offered').length, 1);
});

test('controller: an API-only caller reaches the cpu decision 60 s after the first unreadable snapshot', async (t) => {
    const clock = { ms: Date.UTC(2026, 8, 30, 12, 0, 0) };
    const h = harness(t, { clock, snap: () => cpuSnapshot({ gpu: unreadableGpu }) });
    // Run before the window: refused with the wait and the promise, and nothing is recorded.
    await assert.rejects(() => h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' }),
        (error) => error.code === 'admission_incompatible' && error.message.startsWith('nvidia-smi failed: timed out. No usable GPU has been seen yet, so the hardware profile is not decided; run again once the GPU can be read.')
            && error.message.endsWith('If the GPU is still unreadable 60 s after the first failed read, the next Run uses the CPU.'));
    assert.deepEqual(h.controller.state.requests, {});
    clock.ms += 59_999;
    assert.equal((await h.controller.status()).profile, null);
    // No overview, no dashboard: just the next call after the window.
    clock.ms += 1;
    const status = await h.controller.status();
    assert.deepEqual([status.profile, status.profileDecision.cause], ['cpu', 'unreadable-timeout']);
    assert.equal(status.profileDecision.reason, unreadableGpu.reason);
    assert.equal((await h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' })).accepted, true);
    await until(() => h.controller.state.deployment?.phase === 'ready');
    // The warning of a Run decided this way names the wait and the reason.
    assert.match(h.controller.state.deployment.admission.warnings[0], /^Runs on the CPU: the NVIDIA GPU could not be read for 60 s \(nvidia-smi failed: timed out\)\./);
    await h.controller.stop();
    // Only a Run first: the first refused Run starts the window, and the next Run after 60 s is accepted.
    const runOnly = harness(t, { clock, snap: () => cpuSnapshot({ gpu: unreadableGpu }) });
    const attempt = () => runOnly.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0002' });
    await assert.rejects(attempt, (error) => error.code === 'admission_incompatible');
    clock.ms += 60_000;
    assert.equal((await attempt()).accepted, true);
    await runOnly.controller.stop();
});

test('controller: a transient unreadable GPU never locks a GPU host into cpu, and three parallel overviews commit once', async (t) => {
    // The first query answers and the device query fails (GB10 with [N/A] memory), then a good read: unified, never cpu.
    const clock = { ms: 1_000_000 };
    const gb10Unreadable = { available: false, state: 'unreadable', name: 'NVIDIA GB10', reason: 'nvidia-smi gave no memory figures for NVIDIA GB10 and its device query failed, so its memory model is not known yet' };
    const recovering = harness(t, { clock, snap: (n) => cpuSnapshot({ gpu: n === 1 ? gb10Unreadable : gb10 }) });
    assert.equal((await recovering.controller.overview()).profile, null);
    clock.ms += 59_000;
    const good = await recovering.controller.overview();
    assert.deepEqual([good.profile, good.profileDecision.cause], ['unified', 'gpu']);
    assert.equal(recovering.logLines('hardware profile: unified (NVIDIA GB10)').length, 1);
    assert.equal(recovering.logLines('hardware profile: cpu').length, 0);
    // A failure, then a good read, then another failure: each failure starts its own 60 s (a readable snapshot resets the wait).
    const flaky = harness(t, { clock, snap: (n) => cpuSnapshot({ gpu: n === 2 ? x86 : unreadableGpu }), imageContract: null });
    clock.ms = 2_000_000;
    await flaky.controller.overview();
    clock.ms += 30_000;
    assert.equal((await flaky.controller.overview()).profile, 'dedicated', 'a good read decides at once');
    // Three overviews in parallel while unreadable, spanning 60 s on the injected clock: one commit, one log line.
    // Each snapshot answers a little later than the one before and moves the clock to its own moment (0, 30 and 60 s).
    const spanning = harness(t, { clock, snap: async (n) => {
        await new Promise((resolve) => setTimeout(resolve, n * 10));
        clock.ms = 3_000_000 + (n - 1) * 30_000;
        return cpuSnapshot({ gpu: unreadableGpu });
    } });
    clock.ms = 3_000_000;
    const answers = await Promise.all([spanning.controller.overview(), spanning.controller.overview(), spanning.controller.overview()]);
    assert.equal(spanning.logLines('hardware profile:').length, 1);
    assert.equal(spanning.logLines('hardware profile: cpu (unreadable-timeout:').length, 1);
    assert.ok(answers.some((answer) => answer.profile === 'cpu'));
    assert.equal((await spanning.controller.overview()).profile, 'cpu');
    // A usable GPU that does not report its capability on an image that lists them waits too (a failed device query is not a mismatch).
    const noDevice = { ...x86 };
    delete noDevice.device;
    const capabilities = harness(t, { clock, snap: () => cpuSnapshot({ gpu: noDevice }), imageContract: { architecture: 'arm64', gpu_compute_capabilities: '12.1' } });
    clock.ms = 4_000_000;
    const waiting = await capabilities.controller.overview();
    assert.deepEqual([waiting.profile, waiting.profileDecision], [null, null]);
    assert.equal(capabilities.logLines('hardware profile:').length, 0);
    clock.ms += 60_000;
    const decided = await capabilities.controller.overview();
    assert.deepEqual([decided.profile, decided.profileDecision.cause], ['cpu', 'unreadable-timeout']);
    assert.match(decided.profileDecision.reason, /did not report its compute capability/);
});

test('controller: a GPU the image was not built for runs CPU-offered models on the CPU and offers no unified-only model', async (t) => {
    const thor = { ...gb10, name: 'NVIDIA Thor', device: { pciDeviceId: '0x2B0010DE', computeCapability: '11.0', addressingMode: 'HMM' } };
    const h = harness(t, { snap: () => cpuSnapshot({ gpu: thor }) });
    const overview = await h.controller.overview();
    assert.deepEqual([overview.profile, overview.profileDecision.cause, overview.profileDecision.gpuName], ['cpu', 'mismatch', 'NVIDIA Thor']);
    assert.equal(overview.profileDecision.reason, 'This image\'s runners are built for GPUs of compute capability 12.1; NVIDIA Thor is 11.0.');
    assert.ok(!overview.models.some((model) => model.id === 'big-moe'));
    const row = overview.models.find((model) => model.id === 'small-cpu').runners['llama.cpp'];
    assert.equal(row.admission.status, 'ok', 'the image\'s GPU list does not block a CPU run');
    assert.match(row.admission.warnings[0], /^Runs on the CPU: this image's CUDA runners were not built for this GPU \(This image's runners are built for GPUs of compute capability 12\.1; NVIDIA Thor is 11\.0\)\./);
    await assert.rejects(() => h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0001' }), (error) => error.code === 'unknown_model');
    assert.equal(h.calls.download.length, 0);
    // A CPU Run goes ahead although a GPU is attached: no driver library, no visible device.
    await h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0002' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.deepEqual([h.started[0].env.CUDA_VISIBLE_DEVICES, 'LD_LIBRARY_PATH' in h.started[0].env], ['', false]);
    await h.controller.stop();
});

test('controller: run and stop together on cpu fail the Run as cancelled, with no deployment and no download', async (t) => {
    const h = harness(t);
    await h.controller.overview();
    const run = h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' }).then((value) => ({ value }), (error) => ({ error }));
    const stop = h.controller.stop();
    const result = await run;
    await stop;
    assert.equal(result.error?.code, 'cancelled');
    assert.equal(h.controller.state.deployment, null);
    assert.equal(h.calls.download.length, 0);
    assert.deepEqual(h.controller.state.requests, {});
    assert.equal(h.started.length, 0);
});

// ---------------------------------------------------------------- chat budget and runner report

// Answers one chat request with a fake runner and returns the body the runner was sent.
async function sentBody({ profile, request = { messages: [{ role: 'user', content: 'hi' }] }, options = {} }) {
    let sent = null;
    const target = { runnerId: 'llama.cpp', modelId: 'm', baseUrl: 'http://127.0.0.1:18080', apiKey: null, model: 'm', requestOptions: null,
        ...(profile === undefined ? {} : { profile }) };
    const call = async (op) => (op === 'chatTarget' ? target : {});
    const fetchImpl = async (url, init) => {
        sent = JSON.parse(init.body);
        return { ok: true, status: 200, json: async () => ({ choices: [], usage: {} }) };
    };
    await respond({ request }, { call, fetchImpl, out: { write() {} }, limits: chatLimits({}), ...options });
    return sent;
}

test('chat responder: the cpu profile defaults to 2048 completion tokens unless the operator set a budget', async () => {
    assert.equal((await sentBody({ profile: 'cpu' })).max_tokens, 2048);
    // A caller's own limit is kept, up to the CPU budget.
    assert.equal((await sentBody({ profile: 'cpu', request: { messages: [], max_tokens: 100 } })).max_tokens, 100);
    assert.equal((await sentBody({ profile: 'cpu', request: { messages: [], max_tokens: 5000 } })).max_tokens, 2048);
    assert.equal((await sentBody({ profile: 'cpu', request: { messages: [], max_completion_tokens: 5000 } })).max_completion_tokens, 2048);
    // The operator's value wins over the CPU default, within its bounds; anything outside them is ignored.
    assert.equal(cpuCompletionBudget({}), 2048);
    assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '4096' }), 4096);
    assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '1' }), 1);
    assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '32768' }), 32768);
    for (const bad of ['0', '32769', 'abc', '', '-5', '1e3']) assert.equal(cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: bad }), 2048, bad);
    assert.equal((await sentBody({ profile: 'cpu', options: { cpuMaxCompletionTokens: cpuCompletionBudget({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '4096' }) } })).max_tokens, 4096);
    // Every other profile keeps the 8,192 budget, and so does a deployment recorded without a profile.
    for (const profile of ['dedicated', 'unified', null, undefined]) {
        assert.equal((await sentBody({ profile })).max_tokens, 8192, String(profile));
    }
    assert.deepEqual(chatLimits({}), { maxCompletionTokens: 8192, runnerTimeoutMs: 570_000 });
    assert.equal((await sentBody({ profile: 'unified', options: { limits: chatLimits({ LOCAL_LLM_MAX_COMPLETION_TOKENS: '3000' }) } })).max_tokens, 3000);
});

test('runner report: CPU buffer lines and the CPU backend variant are read, and no GPU layers are claimed', () => {
    const at = (line) => ({ line });
    const cpuRun = [
        'load_backend: loaded RPC backend from /opt/llama.cpp/libggml-rpc.so',
        'load_backend: loaded CPU backend from /opt/llama.cpp/libggml-cpu-armv8.2_2.so',
        'load_tensors: offloading 0 repeating layers to GPU',
        'load_tensors: offloaded 0/25 layers to GPU',
        'load_tensors:   CPU_Mapped model buffer size =   373.71 MiB',
        'load_tensors:   CPU_REPACK model buffer size =   100.00 MiB',
        'llama_kv_cache:        CPU KV buffer size =    48.00 MiB',
        'llama_context:        CPU compute buffer size =   112.30 MiB',
    ].map(at);
    const report = parseRunnerReport(cpuRun, { profile: 'cpu' });
    assert.deepEqual([report.modelMiB, report.kvMiB, report.computeMiB], [473.71, 48, 112.3]);
    assert.equal(report.totalMiB, 634);
    assert.equal(report.device, 'CPU (armv8.2_2)');
    assert.deepEqual(report.offloaded, { layers: 0, of: 25 }, 'zero layers on the GPU');
    // Without a profile the same log reads the same, since it shows no CUDA at all.
    assert.deepEqual(parseRunnerReport(cpuRun), report);
    // A build whose CPU backend has no variant suffix, and a model buffer named plainly.
    const plain = parseRunnerReport(['load_backend: loaded CPU backend from /opt/llama.cpp/libggml-cpu.so', 'load_tensors:          CPU model buffer size =   200.00 MiB'].map(at), { profile: 'cpu' });
    assert.deepEqual([plain.device, plain.modelMiB, plain.totalMiB], ['CPU', 200, 200]);
    // A GPU run: its CPU lines (the layers that stayed in RAM) and its CPU backend never replace the GPU's figures.
    const gpuRun = [
        'load_backend: loaded CUDA backend from /opt/llama.cpp/libggml-cuda.so',
        'load_backend: loaded CPU backend from /opt/llama.cpp/libggml-cpu-armv9.2_2.so',
        'llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GB10) - 0 MiB free',
        'load_tensors: offloaded 25/25 layers to GPU',
        'load_tensors:        CUDA0 model buffer size = 11548.00 MiB',
        'load_tensors:   CPU_Mapped model buffer size =  1104.00 MiB',
        'llama_kv_cache:      CUDA0 KV buffer size =   192.00 MiB',
        'llama_context:      CUDA0 compute buffer size =   216.92 MiB',
    ].map(at);
    for (const options of [{}, { profile: 'unified' }, { profile: 'dedicated' }]) {
        const gpu = parseRunnerReport(gpuRun, options);
        assert.deepEqual([gpu.device, gpu.modelMiB, gpu.kvMiB, gpu.computeMiB, gpu.totalMiB], ['CUDA0 (NVIDIA GB10)', 11548, 192, 216.92, 11957]);
        assert.deepEqual(gpu.offloaded, { layers: 25, of: 25 });
    }
    // A GPU run that has only loaded its backends so far shows no device rather than a CPU; on a cpu deployment (a GPU attached but not used) it is the CPU.
    const early = [
        'load_backend: loaded CUDA backend from /opt/llama.cpp/libggml-cuda.so',
        'load_backend: loaded CPU backend from /opt/llama.cpp/libggml-cpu-armv9.2_2.so',
    ].map(at);
    assert.equal(parseRunnerReport(early).device, null);
    assert.equal(parseRunnerReport(early, { profile: 'dedicated' }).device, null);
    assert.equal(parseRunnerReport(early, { profile: 'cpu' }).device, 'CPU (armv9.2_2)');
    // No lines, no claims.
    const empty = parseRunnerReport([], { profile: 'cpu' });
    assert.deepEqual([empty.device, empty.modelMiB, empty.totalMiB, empty.offloaded], [null, null, null, null]);
});

test('controller: the status report reads the CPU log lines of a cpu deployment', async (t) => {
    const h = harness(t);
    await h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.equal((await h.controller.status()).runnerReport.device, null, 'nothing logged yet');
    // The runner's output reaches the controller's log; a CUDA build on a host where the GPU is attached but unused also loads its CUDA backend.
    for (const line of [
        'load_backend: loaded CUDA backend from /opt/llama.cpp/libggml-cuda.so',
        'load_backend: loaded CPU backend from /opt/llama.cpp/libggml-cpu-armv8.2_2.so',
        'load_tensors: offloaded 0/25 layers to GPU',
        'load_tensors:   CPU_Mapped model buffer size =   373.71 MiB',
        'llama_kv_cache:        CPU KV buffer size =    48.00 MiB',
        'llama_context:        CPU compute buffer size =   112.30 MiB',
    ]) h.started[0].log.append('stdout', line);
    const { runnerReport } = await h.controller.status();
    assert.equal(runnerReport.device, 'CPU (armv8.2_2)');
    assert.deepEqual([runnerReport.modelMiB, runnerReport.kvMiB, runnerReport.computeMiB, runnerReport.totalMiB], [373.71, 48, 112.3, 534]);
    assert.deepEqual(runnerReport.offloaded, { layers: 0, of: 25 });
    await h.controller.stop();
});

test('a committed GPU profile whose GPU is gone refuses every Run until local-llm restarts, and the restart decides cpu', async (t) => {
    const model = cpuModel('m', 380, { layers: 24, kv: 12288 });
    const refusal = (gpu, profile) => admit({ runner: getRunner('llama.cpp'), model, source: cpuSource(380), params: CPU_PARAMS,
        snapshot: { memory: M1_MEMORY, disk: DISK, gpu }, profile });
    for (const profile of ['dedicated', 'unified']) {
        // Positive evidence (no nvidia-smi, or a GPU that cannot be used): the refusal says to restart.
        for (const gpu of [absentGpu, unusableGpu]) {
            const result = refusal(gpu, profile);
            assert.equal(result.status, 'incompatible', `${profile} ${gpu.state}`);
            assert.equal(result.reason, `${gpu.reason}. This agent started with the ${profile} profile; restart local-llm to run on the CPU.`, `${profile} ${gpu.state}`);
        }
        // A failed read may be transient, and a restart during it could lock a GPU host into cpu: the GPU's own reason, with no hint.
        for (const gpu of [unreadableGpu, { available: false, reason: 'a legacy snapshot without a state' }]) {
            const result = refusal(gpu, profile);
            assert.equal(result.status, 'incompatible', `${profile} ${gpu.state}`);
            assert.equal(result.reason, gpu.reason, `${profile} ${gpu.state}`);
        }
        assert.equal(refusal(null, profile).reason, 'No GPU is available to this agent.', 'no GPU reading at all');
    }
    // Without a committed profile the GPU's own reason stands, as before.
    assert.equal(refusal(absentGpu, undefined).reason, absentGpu.reason);
    assert.equal(refusal(unreadableGpu, undefined).reason, unreadableGpu.reason);
    // Through the controller: unified is committed, the GPU is then lost, a Run is refused and nothing downloads.
    let lost = null;
    const h = harness(t, { snap: () => (lost ? cpuSnapshot({ gpu: lost }) : { ...cpuSnapshot(), gpu: structuredClone(gb10) }), imageContract: null });
    assert.equal((await h.controller.overview()).profile, 'unified');
    lost = unreadableGpu;
    await assert.rejects(() => h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0002' }),
        (error) => error.code === 'admission_incompatible' && error.message === unreadableGpu.reason);
    lost = absentGpu;
    await assert.rejects(() => h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp', requestId: 'request-0001' }),
        (error) => error.code === 'admission_incompatible' && error.message.endsWith('This agent started with the unified profile; restart local-llm to run on the CPU.'));
    assert.equal(h.calls.download.length, 0);
    assert.equal((await h.controller.overview()).profile, 'unified', 'the committed profile is not changed by a later snapshot');
    // The restart: a new controller on the same state, with the GPU still gone, decides cpu.
    const restarted = harness(t, { dataDir: h.dataDir, snap: () => cpuSnapshot({ gpu: absentGpu }), imageContract: null });
    assert.equal((await restarted.controller.overview()).profile, 'cpu');
});

test('the shipped CPU seeds are pinned, offered on cpu with CPU defaults, and fit the M1 Podman machine as the SPEC worked out', () => {
    const seeds = loadSeedCatalog();
    const byId = Object.fromEntries(seeds.map((model) => [model.id, model]));
    const cpuIds = seeds.filter((model) => model.profiles.includes('cpu')).map((model) => model.id);
    assert.deepEqual(cpuIds, ['gpt-oss-20b', 'qwen2.5-0.5b-instruct-q4_k_m', 'qwen2.5-1.5b-instruct-q4_k_m', 'qwen3-4b-instruct-2507-q4_k_m']);
    // gpt-oss-20b keeps its GPU profiles and gains cpu; the small seeds are offered on every profile (SPEC D4).
    assert.deepEqual(byId['gpt-oss-20b'].profiles, ['dedicated', 'unified', 'cpu']);
    for (const id of cpuIds.slice(1)) assert.deepEqual(byId[id].profiles, ['cpu', 'dedicated', 'unified'], id);
    assert.deepEqual(byId['gpt-oss-20b'].recommended.cpu['llama.cpp'],
        { ctxSize: 8192, parallel: 1, loadMode: 'mmap', chatTemplateKwargs: { reasoning_effort: 'low' } });
    for (const id of cpuIds.slice(1)) {
        const model = byId[id];
        const source = model.sources.gguf;
        assert.deepEqual(model.recommended.cpu['llama.cpp'], { ctxSize: 4096 }, id);
        assert.match(source.commit, /^[0-9a-f]{40}$/, id);
        assert.match(source.sha256, /^[0-9a-f]{64}$/, id);
        assert.ok(source.size > 0, id);
        assert.equal(model.license, 'Apache-2.0', id);
        assert.equal(source.quantization, 'Q4_K_M', id);
        // A CPU Run with the recommended parameters validates against the CPU schema.
        assert.equal(llamaCppRunner.normalizeParams({}, { model, profile: 'cpu' }).ctxSize, 4096, id);
    }
    // The memory figures the estimate reads (f16 K and V: 2 x layers x KV heads x head size x 2 bytes).
    assert.deepEqual([byId['qwen2.5-0.5b-instruct-q4_k_m'].memory.layers, byId['qwen2.5-0.5b-instruct-q4_k_m'].memory.kvBytesPerToken], [24, 2 * 24 * 2 * 64 * 2]);
    assert.deepEqual([byId['qwen2.5-1.5b-instruct-q4_k_m'].memory.layers, byId['qwen2.5-1.5b-instruct-q4_k_m'].memory.kvBytesPerToken], [28, 2 * 28 * 2 * 128 * 2]);
    assert.deepEqual([byId['qwen3-4b-instruct-2507-q4_k_m'].memory.layers, byId['qwen3-4b-instruct-2507-q4_k_m'].memory.kvBytesPerToken], [36, 2 * 36 * 8 * 128 * 2]);
    // The SPEC's worked example on the M1 VM (MemTotal 5.76 GiB, 3.0 GiB available), now with the pinned sizes.
    const verdict = (id, ctxSize = 4096) => admitOn({ model: byId[id], source: byId[id].sources.gguf, params: { ctxSize, parallel: 1 } });
    assert.equal(verdict('qwen2.5-0.5b-instruct-q4_k_m').status, 'ok');
    assert.equal(verdict('qwen2.5-1.5b-instruct-q4_k_m').status, 'ok');
    assert.equal(verdict('qwen3-4b-instruct-2507-q4_k_m').status, 'insufficient-now');
    assert.equal(verdict('gpt-oss-20b', 8192).status, 'incompatible');
    // The need is the pinned size, the f16 KV cache for the context, the compute buffers at a 512 micro-batch (135 MiB + 0.16 MiB per
    // micro-batch token), 512 MiB for the runner and the 256 MiB prompt cache: worked out here without the estimate's code.
    const need = (id, ctxSize) => {
        const { sources, memory } = byId[id];
        return sources.gguf.size + memory.kvBytesPerToken * ctxSize + (memory.fixedKvBytes || 0) + Math.round((135 + 0.16 * 512) * MIB) + 512 * MIB + 256 * MIB;
    };
    for (const [id, ctxSize] of [['qwen2.5-0.5b-instruct-q4_k_m', 4096], ['qwen2.5-1.5b-instruct-q4_k_m', 4096], ['qwen3-4b-instruct-2507-q4_k_m', 4096], ['gpt-oss-20b', 8192]]) {
        assert.equal(verdict(id, ctxSize).estimate.ramBytes, need(id, ctxSize), id);
    }
    near(mib(need('qwen2.5-0.5b-instruct-q4_k_m', 4096)), 1501.56, '0.5B need MiB with its pinned size');
    near(mib(need('qwen2.5-1.5b-instruct-q4_k_m', 4096)), 2162.48, '1.5B need MiB with its pinned size');
    near(mib(need('gpt-oss-20b', 8192)), 12743.5, 'gpt-oss-20b need MiB');
    // Every seed's id is unique and a seed never shadows another (a catalog error otherwise).
    assert.equal(new Set(seeds.map((model) => model.id)).size, seeds.length);
});

test('the small seeds are offered on GPU hosts too, and run there with the profile\'s own defaults', async (t) => {
    const dedicatedGpu = { available: true, memoryModel: 'dedicated', name: 'RTX 3060', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB,
        processes: [], device: { computeCapability: '8.6' } };
    const dedicated = harness(t, { seed: loadSeedCatalog(), imageContract: null,
        snap: () => ({ ...cpuSnapshot({ memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB } }), gpu: structuredClone(dedicatedGpu) }) });
    const unified = harness(t, { seed: loadSeedCatalog(), imageContract: null,
        snap: () => ({ ...cpuSnapshot({ memory: { totalBytes: 125442396 * 1024, availableBytes: 108 * GIB } }), gpu: structuredClone(gb10) }) });
    const small = ['qwen2.5-0.5b-instruct-q4_k_m', 'qwen2.5-1.5b-instruct-q4_k_m', 'qwen3-4b-instruct-2507-q4_k_m'];
    for (const [profile, h] of [['dedicated', dedicated], ['unified', unified]]) {
        const overview = await h.controller.overview();
        assert.equal(overview.profile, profile);
        for (const id of small) assert.ok(overview.models.some((model) => model.id === id), `${id} is offered on ${profile}`);
        // The catalog asks for no per-profile defaults: the runner's own defaults apply.
        for (const id of small.slice(0, 2)) {
            const row = overview.models.find((model) => model.id === id).runners['llama.cpp'];
            assert.equal(row.admission.status, 'ok', `${id} on ${profile}: ${row.admission.reason}`);
        }
        const row = overview.models.find((model) => model.id === small[0]).runners['llama.cpp'];
        assert.equal(row.params.ctxSize, profile === 'dedicated' ? 16384 : 32768, profile);
    }
    // A Run of a small seed on a GPU host launches with the GPU flags, not the CPU ones.
    await unified.controller.run({ modelId: small[0], runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => unified.controller.state.deployment?.phase === 'ready');
    assert.ok(unified.started[0].args.includes('999') && !unified.started[0].args.includes('--device'));
    await unified.controller.stop();
});

test('controller: a shutdown is not a missing GPU, so a status or an install after the drain never decides the profile', async (t) => {
    const unreadableSnapshot = () => cpuSnapshot({ gpu: unreadableGpu });
    // Control: the same 61 s without a shutdown in between do decide cpu.
    const clock = { ms: 1_000_000 };
    const control = harness(t, { clock, snap: unreadableSnapshot });
    assert.equal((await control.controller.status()).profile, null);
    clock.ms += 61_000;
    assert.equal((await control.controller.status()).profile, 'cpu');
    // A status after the drain, 61 s after the first unreadable snapshot, leaves the profile undecided and logs nothing.
    const first = harness(t, { clock, snap: unreadableSnapshot });
    clock.ms = 2_000_000;
    assert.equal((await first.controller.status()).profile, null, 'the window starts');
    clock.ms += 30_000;
    await first.controller.drain();
    clock.ms += 31_000;
    const after = await first.controller.status();
    assert.deepEqual([after.profile, after.profileDecision, after.gpu], [null, null, null]);
    assert.equal(first.logLines('hardware profile:').length, 0);
    // A stop does not start the window either: two statuses after the drain, 61 s apart, decide nothing.
    const second = harness(t, { clock, snap: unreadableSnapshot });
    clock.ms = 3_000_000;
    await second.controller.drain();
    assert.equal((await second.controller.status()).profile, null);
    clock.ms += 61_000;
    assert.equal((await second.controller.status()).profile, null);
    assert.equal(second.logLines('hardware profile:').length, 0);
    // An install whose snapshot is cut short by the drain does not decide it either.
    const installer = {
        installable: (id) => id === 'vllm', describe: async () => ({ installed: false, runnable: false }),
        entryFor: () => { throw new Error('an install must not get this far'); }, pathsFor: () => ({}),
    };
    clock.ms = 4_000_000;
    // The second snapshot (the install's) never answers; the drain ends it.
    const race = harness(t, { clock, imageContract: null, installer, snap: (n) => (n === 2 ? new Promise(() => {}) : unreadableSnapshot()) });
    assert.equal((await race.controller.status()).profile, null, 'the window starts');
    clock.ms += 61_000;
    const install = race.controller.installRunner({ runnerId: 'vllm' }).then((value) => ({ value }), (error) => ({ error }));
    await until(() => race.calls.snapshots >= 2);
    await race.controller.drain();
    assert.ok((await install).error, 'the install ends with an error');
    assert.equal(race.logLines('hardware profile:').length, 0);
});

test('controller: the parameter preview of a runner with no cpu policy says why, not a parameter error', async (t) => {
    const h = harness(t, { imageContract: null });
    const refusal = (name) => `${name} needs an NVIDIA GPU in this release; on this machine models run on the CPU with the runners listed in the Runners tab.`;
    const previewOf = async (runnerId) => (await h.controller.overview({ preview: { modelId: 'small-cpu', runnerId } })).preview;
    const ik = await previewOf('ik_llama.cpp');
    assert.equal(ik.error, undefined);
    assert.deepEqual([ik.params, ik.context], [null, null]);
    assert.deepEqual([ik.admission.status, ik.admission.reason], ['incompatible', refusal('ik_llama.cpp')]);
    assert.doesNotMatch(JSON.stringify(ik), /Invalid parameter|no parameters for/);
    // The same text the overview gives the runner.
    const overview = await h.controller.overview();
    assert.equal(ik.admission.reason, overview.runners.find((runner) => runner.id === 'ik_llama.cpp').profileUnsupportedReason);
    // A runner that is switched off says so first, as in the overview's rows.
    assert.match((await previewOf('lmstudio')).admission.reason, /LM Studio is not enabled on this deployment/);
    // llama.cpp previews as usual.
    const llama = await previewOf('llama.cpp');
    assert.equal(llama.error, undefined);
    assert.equal(llama.params.ctxSize, 4096);
    assert.equal(llama.admission.status, 'ok');
    // A parameter error of a runner that has a policy is still a parameter error.
    const bad = (await h.controller.overview({ preview: { modelId: 'small-cpu', runnerId: 'llama.cpp', params: { ctxSize: 100 } } })).preview;
    assert.match(bad.error, /Invalid parameter ctxSize/);
});
