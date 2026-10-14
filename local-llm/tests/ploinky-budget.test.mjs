// Container budgets from Ploinky (ploinkyBudget.mjs, DS003, DS005): strict
// readings, the raw CPU quota, capping RAM once, the CPU profile's provenance,
// the guard's repeated-read rule, unchanged unlimited output and the unified
// vLLM denominator.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { GPU_TEST_LOCK, MPS_ENV, TEST_GPU, gpuHarness, testQualification, waitForLaunch } from './ploinkyGpuFixture.mjs';
import { createVllmMpsQualificationResolver, resolveVllmMpsQualification, vllmRunnerLockDigest } from '../src/controller/vllmMpsQualification.mjs';
import { attachGpuBudget, effectiveGpu, parseMpsBudget } from '../src/controller/ploinkyBudget.mjs';
import { validateModel } from '../src/controller/catalog.mjs';
import { createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { loadRunnerLocks } from '../src/controller/runnerLock.mjs';
import { collectOverviews, collectOllamaRuns } from './overview-scenarios.mjs';
import { QUALIFICATION_MISMATCHES, GPU_ADAPTERS, CPU_ADAPTERS, INVALID_MPS_VALUES, checkQualificationMismatch, checkGpuAdapter, checkCpuAdapter, checkInvalidMps } from './ploinkyGpuCases.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import {
    physicalCoreCount,
    readCgroupMemory,
    readCgroupMemoryObservation,
    readCpuQuota,
    readSnapshot,
} from '../src/controller/hardware.mjs';
import {
    attachCpuQuota,
    attachMemoryBudget,
    effectiveMemory,
    knownByte,
    memoryBudgetOf,
    observeMemoryBudget,
    parseCgroupBytes,
    parseCpuQuota,
} from '../src/controller/ploinkyBudget.mjs';
import { cpuPool } from '../src/controller/profiles.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { getRunner } from '../src/runners/index.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();
const SMALL = SEED.find((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const GPT = SEED.find((entry) => entry.id === 'gpt-oss-20b');
const KEY = 'k'.repeat(43);

function fakeFs(files) {
    return {
        readFileSync(file) {
            if (Object.hasOwn(files, file) && files[file] !== undefined) return files[file];
            throw Object.assign(new Error(`ENOENT: ${file}`), { code: 'ENOENT' });
        },
    };
}

const DEDICATED_GPU = Object.freeze({
    available: true, name: 'Test GPU', driverVersion: '595.91.07', memoryModel: 'dedicated',
    totalBytes: 24 * GIB, usedBytes: 256 * MIB, freeBytes: 24 * GIB - 256 * MIB, processes: [],
});
const UNIFIED_GPU = Object.freeze({
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
});

function snapshotWith({ gpu = DEDICATED_GPU, memory = { totalBytes: 128 * GIB, availableBytes: 120 * GIB }, max, current }) {
    const fsApi = fakeFs({ '/sys/fs/cgroup/memory.max': max, '/sys/fs/cgroup/memory.current': current });
    const snap = { gpu: structuredClone(gpu), memory: { ...memory }, disk: { freeBytes: 1024 * GIB }, cpus: 20, cgroupMemory: readCgroupMemory({ fsApi }) };
    return attachMemoryBudget(snap, readCgroupMemoryObservation({ fsApi, established: null }));
}

test('LL.raw-cpu-fractions', () => {
    assert.equal(parseCpuQuota('150000 100000'), 1.5);
    assert.equal(parseCpuQuota('50000 100000\n'), 0.5);
    assert.equal(parseCpuQuota('25000 100000'), 0.25);
    assert.equal(parseCpuQuota('max 100000'), null, 'max is no quota');
    for (const bad of ['', 'garbage', '0 100000', '100000 0', '-1 100000', null, undefined]) assert.equal(parseCpuQuota(bad), null, String(bad));
    const fsApi = fakeFs({ '/sys/fs/cgroup/cpu.max': '150000 100000\n' });
    assert.equal(readCpuQuota({ fsApi }), 1.5, 'no ceil, no minimum of one');
    assert.equal(readCpuQuota({ fsApi: fakeFs({}) }), null);
    // Integer thread selection keeps its own rounding: 1.5 CPUs is 2 cores there.
    const cores = physicalCoreCount({ fsApi: fakeFs({ '/sys/fs/cgroup/cpu.max': '150000 100000\n' }), availableParallelism: () => 8 });
    assert.equal(cores, 2);
    // The quota rides on the snapshot without being published with it.
    const snap = attachCpuQuota({ gpu: null }, 1.5);
    assert.equal(snap.cpuQuota, 1.5);
    assert.equal(JSON.stringify(snap).includes('cpuQuota'), false);
});

test('LL.cpu-warning-provenance', () => {
    // The CPU profile receives the original memory and cgroupMemory: its pool
    // caps them once and its warning still names the container limit.
    const runner = getRunner('llama.cpp');
    const params = runner.normalizeParams({}, { model: SMALL, profile: 'cpu' });
    const snap = snapshotWith({ gpu: { available: false, state: 'absent', reason: 'none' }, max: `${8 * GIB}\n`, current: `${2 * GIB}\n` });
    const result = admit({ runner, model: SMALL, source: SMALL.sources.gguf, params, snapshot: snap, profile: 'cpu', decision: { cause: 'absent', reason: 'none' } });
    assert.equal(result.status, 'ok', result.reason);
    assert.ok(result.warnings.includes(`A container memory limit of 8.00 GiB applies.`), result.warnings.join('\n'));
    assert.equal(result.estimate.poolBytes, 8 * GIB, 'pool capped by the limit');
    assert.equal(snap.memory.totalBytes, 128 * GIB, 'the raw object is not pre-capped');
});

test('LL.cap-once', () => {
    const observation = observeMemoryBudget({ maxText: `${8 * GIB}`, currentText: `${2 * GIB}` });
    const raw = { totalBytes: 128 * GIB, availableBytes: 120 * GIB, swapFreeBytes: 0 };
    const view = effectiveMemory(raw, observation);
    assert.equal(view.totalBytes, 8 * GIB);
    assert.equal(view.availableBytes, 6 * GIB);
    assert.equal(view.physicalTotalBytes, 128 * GIB);
    assert.equal(view.physicalAvailableBytes, 120 * GIB);
    assert.equal(view.swapFreeBytes, 0, 'other readings kept');
    assert.deepEqual(raw, { totalBytes: 128 * GIB, availableBytes: 120 * GIB, swapFreeBytes: 0 }, 'raw object unchanged');
    // Without a finite budget the very same object passes.
    assert.equal(effectiveMemory(raw, observeMemoryBudget({ maxText: 'max', currentText: '1' })), raw);
    // A dedicated policy sizes against the capped view (once): 8 GiB, not 128.
    const runner = getRunner('llama.cpp');
    const params = runner.normalizeParams({ nGpuLayers: 0 }, { model: GPT });
    const result = admit({ runner, model: GPT, source: GPT.sources.gguf, params, snapshot: snapshotWith({ max: `${8 * GIB}\n`, current: `${2 * GIB}\n` }) });
    assert.equal(result.status, 'incompatible');
    assert.match(result.reason, /this machine has 8\.0 GiB/);
    // The CPU pool caps from the raw inputs exactly once as well.
    assert.deepEqual(cpuPool(raw, { maxBytes: 8 * GIB, currentBytes: 2 * GIB }), { totalBytes: 8 * GIB, availableBytes: 6 * GIB, capped: true });
});

test('LL.known-zero', () => {
    assert.equal(knownByte(0), true);
    assert.deepEqual(parseCgroupBytes('0\n'), { state: 'known', bytes: 0 });
    const full = observeMemoryBudget({ maxText: `${8 * GIB}`, currentText: `${8 * GIB}` });
    assert.deepEqual({ ...full }, { memoryReadState: 'known', finiteMemoryBytes: 8 * GIB, headroomBytes: 0, reasonCode: null });
    const over = observeMemoryBudget({ maxText: `${8 * GIB}`, currentText: `${9 * GIB}` });
    assert.equal(over.headroomBytes, 0, 'current above the limit is known zero headroom');
    // Zero available cannot pass a falsy guard: the run is refused now, not waved through.
    const runner = getRunner('llama.cpp');
    const params = runner.normalizeParams({}, { model: SMALL });
    const zero = admit({ runner, model: SMALL, source: SMALL.sources.gguf, params, snapshot: snapshotWith({ max: `${8 * GIB}\n`, current: `${8 * GIB}\n` }) });
    assert.equal(zero.status, 'insufficient-now');
    assert.match(zero.reason, /0\.0 GiB is available now/);
    assert.equal(zero.reasonCode, undefined);
    assert.ok(zero.warnings.some((warning) => /0\.0 GiB is available now/.test(warning)), 'the RAM warning reports the shortage');
    // A known zero total is incompatible, not ignored.
    const none = admit({ runner, model: SMALL, source: SMALL.sources.gguf, params,
        snapshot: snapshotWith({ max: '0\n', current: '0\n' }) });
    assert.equal(none.status, 'incompatible');
    assert.deepEqual(cpuPool({ totalBytes: 8 * GIB, availableBytes: 4 * GIB }, { maxBytes: 0, currentBytes: 0 }), { totalBytes: 0, availableBytes: 0, capped: true });
});

test('LL.unknown-distinct', () => {
    const none = observeMemoryBudget({ maxText: 'max\n', currentText: '123\n' });
    assert.deepEqual({ ...none }, { memoryReadState: 'known', finiteMemoryBytes: null, headroomBytes: null, reasonCode: null });
    // An absent memory.max (undefined) is no limit, as on hosts without one.
    assert.deepEqual({ ...observeMemoryBudget({ maxText: undefined, currentText: undefined }) },
        { memoryReadState: 'known', finiteMemoryBytes: null, headroomBytes: null, reasonCode: null });
    // A memory.max that exists but cannot be read (null) or is malformed is
    // unknown, never unlimited, even with no limit established before.
    for (const maxText of [null, '', 'abc', '-1', '1.5', ' 12 34', 'MAX']) {
        assert.deepEqual({ ...observeMemoryBudget({ maxText, currentText: '1' }) },
            { memoryReadState: 'unknown', finiteMemoryBytes: null, headroomBytes: null, reasonCode: 'budget_unreadable' }, String(maxText));
    }
    // An established finite limit is kept whether the file vanished or broke.
    for (const maxText of [undefined, null, 'garbage']) {
        assert.equal(observeMemoryBudget({ maxText, currentText: '1', established: 4 * GIB }).finiteMemoryBytes, 4 * GIB, String(maxText));
    }
    const established = observeMemoryBudget({ maxText: null, currentText: '1', established: 4 * GIB });
    assert.deepEqual({ ...established }, { memoryReadState: 'unknown', finiteMemoryBytes: 4 * GIB, headroomBytes: null, reasonCode: 'budget_unreadable' });
    // Missing or malformed current use under a finite limit: unknown, null, never 0.
    for (const currentText of [null, undefined, '', 'abc', '-1', '1.5', ' 12 34', 'max']) {
        const observation = observeMemoryBudget({ maxText: `${4 * GIB}`, currentText });
        assert.equal(observation.memoryReadState, 'unknown', String(currentText));
        assert.equal(observation.headroomBytes, null);
        assert.equal(observation.reasonCode, 'budget_unreadable');
    }
    // Unsafe numbers are unknown internally, although the legacy raw projection remains null.
    for (const maxText of ['9223372036854771712', '9'.repeat(400)]) {
        for (const established of [null, 4 * GIB]) {
            assert.deepEqual(observeMemoryBudget({ maxText, currentText: '1', established }), {
                memoryReadState: 'unknown', finiteMemoryBytes: established, headroomBytes: null, reasonCode: 'budget_unreadable',
            });
        }
    }
    // The production reader keeps an established limit per filesystem reader.
    const limited = fakeFs({ '/sys/fs/cgroup/memory.max': `${4 * GIB}\n`, '/sys/fs/cgroup/memory.current': `${GIB}\n` });
    assert.equal(readCgroupMemoryObservation({ fsApi: limited }).headroomBytes, 3 * GIB);
    const gone = fakeFs({});
    assert.equal(readCgroupMemoryObservation({ fsApi: gone }).memoryReadState, 'known', 'a fresh reader without a limit has none');
    assert.equal(readCgroupMemoryObservation({ fsApi: gone, established: 4 * GIB }).memoryReadState, 'unknown');
    // A memory.max that exists but cannot be read (EACCES, EIO) is unknown.
    for (const code of ['EACCES', 'EIO']) {
        const denied = { readFileSync(file) { throw Object.assign(new Error(`${code}: ${file}`), { code }); } };
        const observation = readCgroupMemoryObservation({ fsApi: denied, established: null });
        assert.deepEqual([observation.memoryReadState, observation.reasonCode], ['unknown', 'budget_unreadable'], code);
    }
    // The legacy projection stays byte-identical: any read failure is null.
    assert.equal(readCgroupMemory({ fsApi: { readFileSync() { throw Object.assign(new Error('EACCES'), { code: 'EACCES' }); } } }), null);
    // Legacy-only snapshots: a finite limit with unknown current is unknown, too.
    assert.equal(memoryBudgetOf({ cgroupMemory: { maxBytes: 4 * GIB, currentBytes: null } }).memoryReadState, 'unknown');
    assert.equal(memoryBudgetOf({ cgroupMemory: null }).finiteMemoryBytes, null);
});

function deferred() {
    let resolve;
    const promise = new Promise((ok) => { resolve = ok; });
    return { promise, resolve };
}

function cpuHarness(t, { readBudget, max = `${16 * GIB}\n`, current = `${2 * GIB}\n` } = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-budget-guard-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const weights = path.join(dataDir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const started = [];
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: SEED,
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => snapshotWith({ gpu: { available: false, state: 'absent', reason: 'No NVIDIA GPU.' }, max, current }),
        download: async () => ({ status: 'complete', path: weights, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        startRunner({ command, args, env }) {
            const exit = deferred();
            let running = true;
            const handle = {
                pid: 8000 + started.length, command, args, env, exited: exit.promise,
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
        imageContract: null,
        fileExists: () => true,
        sharedModelsRoot: null,
        readMemory: () => ({ totalBytes: 128 * GIB, availableBytes: 120 * GIB }),
        readPressure: () => 0,
        readBudget,
        unifiedGuardMs: 5,
        pollMs: 2,
        stopGraceMs: 20,
    });
    return { controller, started };
}

async function until(predicate, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 2));
    }
    assert.fail('condition not reached');
}

test('LL.guard-read-failure', async (t) => {
    const known = observeMemoryBudget({ maxText: `${16 * GIB}`, currentText: `${2 * GIB}` });
    const unknown = observeMemoryBudget({ maxText: null, currentText: null, established: 16 * GIB });
    // Calls 1-5 known, one unreadable sample (6), then known again (7-9), then unreadable from 10 on.
    let calls = 0;
    const h = cpuHarness(t, {
        readBudget: () => {
            calls += 1;
            if (calls === 6) return unknown;
            return calls >= 10 ? unknown : known;
        },
    });
    await h.controller.run({ modelId: SMALL.id, runnerId: 'llama.cpp', requestId: 'request-guard' });
    await until(() => h.started[0]?.stopped);
    const [runner] = h.started;
    assert.equal(runner.stopped, 'SIGKILL', 'the owned process group is killed');
    assert.ok(calls >= 11, `a single unreadable sample did not stop it (stopped after ${calls} samples)`);
    await until(() => h.controller.state.deployment?.phase === 'error');
    assert.match(h.controller.state.deployment.error, /^stopped: budget_unreadable: /);
    await h.controller.stop();

    // Without a finite limit the guard never samples a budget rule.
    const unlimited = cpuHarness(t, { max: 'max\n', readBudget: () => observeMemoryBudget({ maxText: 'max', currentText: '1' }) });
    await unlimited.controller.run({ modelId: SMALL.id, runnerId: 'llama.cpp', requestId: 'request-unlimited' });
    await until(() => unlimited.started.length === 1);
    await new Promise((resolve) => setTimeout(resolve, 60));
    assert.equal(unlimited.started[0].stopped, undefined);
    await unlimited.controller.stop();
});

test('LL.unlimited-golden', async (t) => {
    // The production snapshot carries the budget facts without publishing them.
    const files = {
        '/proc/meminfo': 'MemTotal:        6036128 kB\nMemAvailable:    3122576 kB\nSwapFree:              0 kB\nMemFree: 1 kB\nCached: 1 kB\n',
        '/proc/self/status': 'Cpus_allowed_list:\t0-3\n',
    };
    const statfs = async () => ({ bavail: 1000, bsize: 4096, blocks: 2000 });
    const execFileImpl = (command, args, options, callback) => callback(Object.assign(new Error('ENOENT'), { code: 'ENOENT' }), '', '');
    const unlimited = await readSnapshot({ dataDir: '/data', execFileImpl, statfs,
        fsApi: fakeFs({ ...files, '/sys/fs/cgroup/memory.max': 'max\n', '/sys/fs/cgroup/memory.current': '123\n' }) });
    assert.equal(unlimited.cgroupMemory, null);
    assert.equal(unlimited.memoryBudget.finiteMemoryBytes, null);
    assert.deepEqual(Object.keys(unlimited).sort(), ['at', 'cgroupMemory', 'cores', 'cpus', 'disk', 'gpu', 'memory']);
    const limited = await readSnapshot({ dataDir: '/data', execFileImpl, statfs,
        fsApi: fakeFs({ ...files, '/sys/fs/cgroup/memory.max': `${2 * GIB}\n`, '/sys/fs/cgroup/memory.current': `${512 * MIB}\n`, '/sys/fs/cgroup/cpu.max': '150000 100000\n' }) });
    assert.deepEqual(Object.keys(limited).sort(), ['at', 'cgroupMemory', 'cores', 'cpus', 'disk', 'gpu', 'memory']);
    assert.equal(limited.memoryBudget.headroomBytes, 1536 * MIB);
    assert.equal(limited.cpuQuota, 1.5);

    // The overview: no budget field without a limit; exact values with one.
    const overviewFor = async (max, cpuQuota = null) => {
        const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-budget-overview-'));
        t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
        const controller = createController({
            dataDir, env: { PATH: '/usr/bin' }, seedCatalog: SEED, stateStore: createStateStore({ dataDir }),
            snapshot: async () => attachCpuQuota(snapshotWith({ max, current: `${GIB}\n` }), cpuQuota),
            inspect: async () => ({ state: 'absent', bytes: 0 }),
            detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
            imageContract: null, fileExists: () => true, sharedModelsRoot: null,
        });
        return controller.overview();
    };
    const plain = await overviewFor('max\n');
    assert.equal(plain.limits, null);
    assert.equal(JSON.stringify(plain.hardware).includes('memoryBudget'), false);
    assert.equal(JSON.stringify(plain).includes('budget'), false);
    const budgeted = await overviewFor(`${8 * GIB}\n`, 1.5);
    assert.deepEqual(budgeted.limits, { budget: { cpus: 1.5, memoryBytes: 8 * GIB, source: 'ploinky' } });
    assert.equal(JSON.stringify(budgeted.hardware).includes('memoryBudget'), false);
});

test('LL.unified-physical-denominator', () => {
    const runner = getRunner('vllm');
    const model = { ...SEED.find((entry) => entry.id === 'gpt-oss-20b') };
    const params = runner.normalizeParams({ gpuMemoryUtilization: 0.3 }, { model, profile: 'unified' });
    const physical = 128 * GIB;
    const snap = snapshotWith({ gpu: UNIFIED_GPU, memory: { totalBytes: physical, availableBytes: 120 * GIB }, max: `${64 * GIB}\n`, current: `${4 * GIB}\n` });
    const result = admit({ runner, model, source: model.sources.hf, params, snapshot: snap, profile: 'unified' });
    assert.equal(result.status, 'ok', result.reason);
    // The share is of the physical pool (vLLM's denominator), not of the 64 GiB budget...
    assert.equal(result.estimate.budgetBytes, Math.round(0.3 * physical));
    // ...while the absolute need is bounded by the budget itself, with no host reserve taken out of it (the
    // reserve belongs to the physical pool, DS005): 0.4 of the physical pool plus the runner's RAM fits a
    // 64 GiB budget, and 0.5 does not. The base refused 0.4 by taking the 16 GiB reserve out of the budget too.
    const fits = admit({ runner, model, source: model.sources.hf, params: runner.normalizeParams({ gpuMemoryUtilization: 0.4 }, { model, profile: 'unified' }),
        snapshot: snap, profile: 'unified' });
    assert.equal(fits.status, 'ok', fits.reason);
    const tooBig = admit({ runner, model, source: model.sources.hf, params: runner.normalizeParams({ gpuMemoryUtilization: 0.5 }, { model, profile: 'unified' }),
        snapshot: snap, profile: 'unified' });
    assert.equal(tooBig.status, 'incompatible');
    assert.equal(tooBig.reasonCode, 'budget_exceeded');
    assert.match(tooBig.reason, /this agent's Ploinky RAM budget is 64\.0 GiB/);
    // Without a limit, the denominator is today's MemTotal (unchanged).
    const plain = admit({ runner, model, source: model.sources.hf, params, profile: 'unified',
        snapshot: { gpu: structuredClone(UNIFIED_GPU), memory: { totalBytes: physical, availableBytes: 120 * GIB }, disk: { freeBytes: 1024 * GIB } } });
    assert.equal(plain.estimate.budgetBytes, Math.round(0.3 * physical));
    assert.equal(plain.estimate.poolBytes, physical);
});

test('LL.vllm-qualification-absent', async (t) => {
    const h = gpuHarness(t, { qualificationDataProvider: null, env: { ...MPS_ENV, LOCAL_LLM_VLLM_MPS_QUALIFIED: 'true' } });
    await assert.rejects(h.run({ maxModelLen: 512 }), (error) => error.code === 'admission_incompatible' && error.details?.admission?.reasonCode === 'vllm_mps_unqualified');
    assert.equal(h.started.length, 0);
    const { denominator, evidenceDigest, ...tuple } = testQualification();
    assert.equal(resolveVllmMpsQualification(tuple).qualified, false);
    assert.equal(resolveVllmMpsQualification({ ...tuple, qualified: true }).qualified, false);
});

test('LL.vllm-qualification-mismatch', async (t) => {
    for (const vector of QUALIFICATION_MISMATCHES) await checkQualificationMismatch(t, vector);
});
for (const vector of QUALIFICATION_MISMATCHES) test(`qualification mismatch vector: ${vector[0]}`, (t) => checkQualificationMismatch(t, vector));

test('LL.vllm-qualification-match', async (t) => {
    const entry = GPU_TEST_LOCK.runners.vllm;
    assert.equal(vllmRunnerLockDigest(entry), vllmRunnerLockDigest({ ...entry, check: { ...entry.check } }));
    assert.notEqual(vllmRunnerLockDigest(entry), vllmRunnerLockDigest({ ...entry, files: [{ ...entry.files[0], sha256: 'b'.repeat(64) }] }));
    assert.equal(vllmRunnerLockDigest({ id: 'vllm', digest: 'a'.repeat(64) }), null);
    const resolve = createVllmMpsQualificationResolver(() => [testQualification()]);
    assert.deepEqual(resolve(testQualification()), { qualified: true, denominator: 'physical-device', evidenceDigest: 'e'.repeat(64) });
    const h = gpuHarness(t);
    assert.equal((await h.run({ maxModelLen: 512 })).accepted, true);
    await waitForLaunch(h);
    assert.equal(h.controller.state.deployment.admission.status, 'ok');
});

test('LL.vllm-six-three-two', async (t) => {
    const h = gpuHarness(t);
    assert.equal((await h.run({ maxModelLen: 512 })).accepted, true);
    const launch = await waitForLaunch(h);
    const admission = h.controller.state.deployment.admission;
    assert.equal(admission.estimate.gpuBytes, 2 * GIB);
    const fraction = Number(launch.args[launch.args.indexOf('--gpu-memory-utilization') + 1]);
    assert.equal(fraction, admission.estimate.gpuMemoryUtilization);
    assert.ok(fraction * 6 * GIB <= 3 * GIB);
    assert.ok(fraction * 6 * GIB * 0.94 >= 2 * GIB, 'the physical denominator avoids multiplying the share twice');
    const overview = await h.controller.overview();
    assert.deepEqual(overview.limits.budget.gpuShare, { smPercent: 50, vramBytes: 3 * GIB, assurance: 'best-effort' });
    assert.equal(overview.hardware.gpu.totalBytes, 6 * GIB, 'raw telemetry is physical');
    assert.equal(JSON.stringify(overview.hardware).includes('gpuBudget'), false);
});

test('LL.vllm-over-cap-incompatible', async (t) => {
    const h = gpuHarness(t);
    await assert.rejects(h.run({ maxModelLen: 512, gpuMemoryUtilization: 0.6 }), (error) => error.code === 'admission_incompatible' && /Ploinky GPU share/.test(error.message));
    assert.equal(h.started.length, 0);
});

test('LL.vllm-free-shortage-temporary', async (t) => {
    const h = gpuHarness(t, { gpu: { ...TEST_GPU, freeBytes: 2 * GIB } });
    await assert.rejects(h.run({ maxModelLen: 512, gpuMemoryUtilization: 0.4 }), (error) => error.code === 'admission_insufficient_now');
    assert.equal(h.started.length, 0);
    h.setGpu({ ...TEST_GPU, freeBytes: 3 * GIB });
    assert.equal((await h.run({ maxModelLen: 512, gpuMemoryUtilization: 0.4 }, { requestId: 'gpu-budget-release-retry' })).accepted, true);
    await waitForLaunch(h);
});

test('LL.ollama-pinned-over-cap', async (t) => {
    const raw = structuredClone(Object.fromEntries(Object.entries(SEED.find((model) => model.id === 'gpt-oss-20b')).filter(([key]) => !['seed', 'recommended'].includes(key))));
    raw.id = 'budget-pinned-ollama';
    const expectedBytes = Math.round(4.445 * GIB);
    raw.sources = { ollama: { ...raw.sources.ollama, size: expectedBytes - 200 * MIB - raw.memory.kvBytesPerToken * 512 - raw.memory.fixedKvBytes } };
    const model = validateModel(raw, { seed: true });
    const h = gpuHarness(t, { seedCatalog: [model] });
    await assert.rejects(h.run({ numGpu: 24, numCtx: 512 }, { runnerId: 'ollama', modelId: model.id }), (error) => {
        assert.equal(error.code, 'admission_incompatible');
        assert.equal(error.details.admission.estimate.gpuBytes, expectedBytes);
        return true;
    });
    assert.equal(h.started.length, 0);
});

test('LL.ollama-auto-free', async (t) => {
    const h = gpuHarness(t);
    assert.equal((await h.run({ numGpu: null }, { runnerId: 'ollama', modelId: 'gpt-oss-20b' })).accepted, true);
    await waitForLaunch(h);
    const estimate = h.controller.state.deployment.admission.estimate;
    assert.equal(estimate.ramBytes, Math.max(0, estimate.totalBytes - (3 * GIB - 256 * MIB)) + 768 * MIB);
});

test('LL.gpu-minimal-env-all-adapters', async (t) => {
    for (const adapter of GPU_ADAPTERS) await checkGpuAdapter(t, adapter);
});
for (const adapter of GPU_ADAPTERS) test(`GPU environment adapter: ${adapter[0]}`, (t) => checkGpuAdapter(t, adapter));

test('LL.cpu-no-mps-env', async (t) => {
    for (const adapter of CPU_ADAPTERS) await checkCpuAdapter(t, adapter);
});
for (const adapter of CPU_ADAPTERS) test(`CPU environment adapter: ${adapter[0]}`, (t) => checkCpuAdapter(t, adapter));

test('LL.partial-mps-refused', async (t) => {
    for (const vector of INVALID_MPS_VALUES.entries()) await checkInvalidMps(t, vector);
});
for (const vector of INVALID_MPS_VALUES.entries()) test(`invalid MPS vector: ${vector[0]}`, (t) => checkInvalidMps(t, vector));

// A known Ploinky GPU share on a unified GPU (DS005): refused everywhere except the qualified GB10 tuple (PCI 0x2E1210DE with
// driver 580.178.04), where llama.cpp is admitted with the share bounding its GPU part and the RAM budget its host part.
const UNIFIED_GB10 = Object.freeze({ available: true, name: 'NVIDIA GB10', memoryModel: 'unified', driverVersion: '580.178.04',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' } });
const UNQUALIFIED_GB10 = Object.freeze({ ...UNIFIED_GB10, driverVersion: '580.159.03' });
const SHARE_REFUSAL = 'Ploinky GPU sharing requires a verified dedicated-memory GPU. Unified or unknown memory-model GPUs cannot use this MPS share; clear it through Ploinky.';
const SMALL_MODEL = 'qwen2.5-0.5b-instruct-q4_k_m';
const MPS_KEYS = ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT', 'CUDA_MPS_PIPE_DIRECTORY'];
const mpsEnvOf = (processEnv) => Object.fromEntries(Object.entries(processEnv).filter(([key]) => key.startsWith('CUDA_MPS_')));
const shareEnv = (vramMiB) => ({ ...MPS_ENV, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: `0=${vramMiB}M` });

// The admission verdict of the small llama.cpp model on a 128 GiB / 120 GiB unified host under a share and a RAM budget.
function shareVerdict({ gpu = UNIFIED_GB10, env = MPS_ENV, budget = null, memory = { totalBytes: 128 * GIB, availableBytes: 120 * GIB } } = {}) {
    const model = loadSeedCatalog().find((entry) => entry.id === SMALL_MODEL);
    const runner = getRunner('llama.cpp');
    const snapshot = { gpu: structuredClone(gpu), memory, disk: { freeBytes: 1024 * GIB }, cgroupMemory: budget };
    return admit({ runner, model, source: model.sources.gguf, params: runner.normalizeParams({}, { model, profile: 'unified' }), snapshot, profile: 'unified',
        gpuBudget: parseMpsBudget(env) });
}

test('LL.unified-share-refused: an unqualified unified GPU still refuses a share with the existing text; the qualified tuple accepts it', async (t) => {
    // The synthetic unified GPU of Phase 1 and the GB10 on driver 580.159.03 are not qualified.
    for (const gpu of [{ ...TEST_GPU, memoryModel: 'unified', totalBytes: null, freeBytes: null }, UNQUALIFIED_GB10,
        { ...UNIFIED_GB10, device: { ...UNIFIED_GB10.device, pciDeviceId: '0x2B0010DE' } }]) {
        const h = gpuHarness(t, { gpu });
        await assert.rejects(h.run({}, { runnerId: 'llama.cpp', modelId: SMALL_MODEL }), (error) => error.details?.admission?.reasonCode === 'gpu_share_unsupported'
            && error.details.admission.reason === SHARE_REFUSAL);
        assert.equal(h.started.length, 0);
    }
    // A-LL-24: the same share on the qualified tuple is accepted.
    const h = gpuHarness(t, { gpu: UNIFIED_GB10 });
    await h.run({}, { runnerId: 'llama.cpp', modelId: SMALL_MODEL });
    const runner = await waitForLaunch(h);
    assert.equal(h.controller.state.deployment.admission.status, 'ok');
    assert.equal(runner.env.PRIVATE_AGENT_SECRET, undefined);
});

test('A-LL-24: vLLM under a share on unified memory is vllm_mps_unqualified on the qualified tuple and gpu_share_unsupported elsewhere', () => {
    const model = validateModel({
        id: 'share-vllm-hf', displayName: 'Share vLLM fixture', architecture: 'dense', contextLength: 32768, profiles: ['unified'], memory: { layers: 24, kvBytesPerToken: 12288 },
        sources: { hf: { type: 'hf-snapshot', repo: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'main', commit: 'c'.repeat(40), files: [
            { path: 'config.json', size: 659, sha256: 'a'.repeat(64) }, { path: 'model.safetensors', size: 999585688, sha256: 'b'.repeat(64) },
        ] } },
    }, { seed: false });
    const vllm = getRunner('vllm');
    const verdict = (gpu) => admit({ runner: vllm, model, source: model.sources.hf, profile: 'unified', gpuBudget: parseMpsBudget(MPS_ENV),
        params: vllm.normalizeParams({}, { model, profile: 'unified' }),
        snapshot: { gpu: structuredClone(gpu), memory: { totalBytes: 128 * GIB, availableBytes: 120 * GIB }, disk: { freeBytes: 1024 * GIB } } });
    const qualified = verdict(UNIFIED_GB10);
    assert.deepEqual([qualified.status, qualified.reasonCode], ['incompatible', 'vllm_mps_unqualified']);
    const unqualified = verdict(UNQUALIFIED_GB10);
    assert.deepEqual([unqualified.status, unqualified.reasonCode, unqualified.reason], ['incompatible', 'gpu_share_unsupported', SHARE_REFUSAL]);
});

test('LL.unified-share-split (A-LL-9): the share bounds the GPU part, the budget the host part, and the runner gets exactly the three MPS variables', async (t) => {
    // G of the small model is 2.78 GiB: a 2 GiB share is too small, a 3 GiB share holds it.
    const small = shareVerdict({ env: shareEnv(2048) });
    assert.equal(small.status, 'incompatible');
    assert.equal(small.reasonCode, 'gpu_share_too_small');
    assert.match(small.reason, /^Needs about 2\.8 GiB of GPU memory; this agent's Ploinky GPU memory share is 2\.0 GiB\./);
    const { cacheRamBytes, gpuBytes, hostChargedBytes, unifiedBytes } = shareVerdict().estimate;
    assert.equal(gpuBytes, unifiedBytes - cacheRamBytes);
    assert.equal(hostChargedBytes, cacheRamBytes + 1.5 * GIB);
    // S >= G, C <= B and Hb >= C: ok, and the launched runner has exactly the three variables (and none of the agent's secrets).
    const h = gpuHarness(t, { gpu: UNIFIED_GB10, cgroupMemory: { maxBytes: 64 * GIB, currentBytes: GIB } });
    await h.run({}, { runnerId: 'llama.cpp', modelId: SMALL_MODEL });
    const runner = await waitForLaunch(h);
    assert.deepEqual(Object.keys(mpsEnvOf(runner.env)).sort(), MPS_KEYS);
    assert.deepEqual(mpsEnvOf(runner.env), MPS_ENV);
    assert.equal(runner.env.PRIVATE_AGENT_SECRET, undefined);
    // The order of the rules: U1 first, then the share (U2s), then the budget (U2c), then the floor (U3), then the headroom (U4c).
    assert.equal(shareVerdict({ env: shareEnv(2048), budget: { maxBytes: GIB, currentBytes: 0 } }).reasonCode, 'gpu_share_too_small');
    assert.equal(shareVerdict({ memory: { totalBytes: 12 * GIB, availableBytes: 11 * GIB } }).status, 'incompatible');
    assert.equal(shareVerdict({ memory: { totalBytes: 12 * GIB, availableBytes: 11 * GIB } }).reasonCode, undefined, 'U1 has no code');
    assert.match(shareVerdict({ memory: { totalBytes: 12 * GIB, availableBytes: 11 * GIB } }).reason, /must keep 16\.0 GiB for the host/);
    const floor = shareVerdict({ memory: { totalBytes: 128 * GIB, availableBytes: 18 * GIB }, budget: { maxBytes: 64 * GIB, currentBytes: 0 } });
    assert.equal(floor.status, 'insufficient-now');
    assert.equal(floor.reasonCode, undefined, 'U3 keeps its own text');
});

test('LL.unified-share-no-fabricated-capacity (A-LL-16): the unified policy sees the GPU with null memory figures, never 0, and the share', () => {
    let seen = null;
    const model = loadSeedCatalog().find((entry) => entry.id === SMALL_MODEL);
    const llama = getRunner('llama.cpp');
    const runner = { ...llama, admitUnified(input) { seen = input; return llama.admitUnified(input); } };
    const verdict = admit({ runner, model, source: model.sources.gguf, params: llama.normalizeParams({}, { model, profile: 'unified' }), profile: 'unified',
        snapshot: { gpu: structuredClone(UNIFIED_GB10), memory: { totalBytes: 128 * GIB, availableBytes: 120 * GIB }, disk: { freeBytes: 1024 * GIB } },
        gpuBudget: parseMpsBudget(MPS_ENV) });
    assert.equal(verdict.status, 'ok', verdict.reason);
    assert.equal(seen.gpu.totalBytes, null);
    assert.equal(seen.gpu.freeBytes, null);
    assert.equal(seen.gpu.usedBytes, null);
    assert.equal('budgetBytes' in seen.gpu || 'deviceTotalBytes' in seen.gpu, false, 'effectiveGpu is not applied on unified');
    assert.deepEqual(seen.gpuShare, { smPercent: 50, vramBytes: 3 * GIB });
    // No share: the policy gets none.
    admit({ runner, model, source: model.sources.gguf, params: llama.normalizeParams({}, { model, profile: 'unified' }), profile: 'unified',
        snapshot: { gpu: structuredClone(UNIFIED_GB10), memory: { totalBytes: 128 * GIB, availableBytes: 120 * GIB }, disk: { freeBytes: 1024 * GIB } } });
    assert.equal(seen.gpuShare, null);
});

test('LL.share-budget-boundary (A-LL-17..20): the RAM budget must hold C = prompt cache + runtime, now and ever, and nothing else', async (t) => {
    const { hostChargedBytes: C } = shareVerdict().estimate;
    // A-LL-17: B = C, nothing used: ok (the 10.8 GiB need alone would not fit a budget of C).
    assert.equal(shareVerdict({ budget: { maxBytes: C, currentBytes: 0 } }).status, 'ok');
    // A-LL-18: B = C - 1: refused for good, before any launch.
    const never = shareVerdict({ budget: { maxBytes: C - 1, currentBytes: 0 } });
    assert.deepEqual([never.status, never.reasonCode], ['incompatible', 'budget_exceeded']);
    assert.match(never.reason, /^Needs about 9\.5 GiB of host memory charged to this agent \(the prompt cache and runtime\); this agent's Ploinky RAM budget is 9\.5 GiB\./);
    // A-LL-19: B ample, Hb = C: ok. A-LL-20: Hb = C - 1: insufficient-now, budget_insufficient_now.
    assert.equal(shareVerdict({ budget: { maxBytes: 64 * GIB, currentBytes: 64 * GIB - C } }).status, 'ok');
    const busy = shareVerdict({ budget: { maxBytes: 64 * GIB, currentBytes: 64 * GIB - C + 1 } });
    assert.deepEqual([busy.status, busy.reasonCode], ['insufficient-now', 'budget_insufficient_now']);
    assert.match(busy.reason, /of this agent's 64\.0 GiB budget is free now\.$/);
    // The rules keep their order when a later one would also fire (18 GiB available: need 10.8 GiB plus the 8 GiB floor does not fit):
    // U2c comes before U3, and U3 (its own text, no code) before U4c.
    const crowded = { totalBytes: 128 * GIB, availableBytes: 18 * GIB };
    const budgetFirst = shareVerdict({ memory: crowded, budget: { maxBytes: C - 1, currentBytes: 0 } });
    assert.deepEqual([budgetFirst.status, budgetFirst.reasonCode], ['incompatible', 'budget_exceeded']);
    const floorFirst = shareVerdict({ memory: crowded, budget: { maxBytes: 64 * GIB, currentBytes: 64 * GIB - C + 1 } });
    assert.equal(floorFirst.status, 'insufficient-now');
    assert.equal(floorFirst.reasonCode, undefined);
    assert.match(floorFirst.reason, /kept free; 18\.0 GiB is available now\. Other processes on this machine hold the rest\.$/);
    // Without a share the same budgets are U2/U4 on the whole need, as before.
    const unshared = shareVerdict({ env: {}, budget: { maxBytes: C, currentBytes: 0 } });
    assert.deepEqual([unshared.status, unshared.reasonCode], ['incompatible', 'budget_exceeded']);
    // Through the controller: refused before launch, no runner started.
    for (const [budget, code] of [[{ maxBytes: C - 1, currentBytes: 0 }, 'budget_exceeded'], [{ maxBytes: 64 * GIB, currentBytes: 64 * GIB - C + 1 }, 'budget_insufficient_now']]) {
        const h = gpuHarness(t, { gpu: UNIFIED_GB10, cgroupMemory: budget });
        await assert.rejects(h.run({}, { runnerId: 'llama.cpp', modelId: SMALL_MODEL }), (error) => error.details?.admission?.reasonCode === code);
        assert.equal(h.started.length, 0);
    }
    const ok = gpuHarness(t, { gpu: UNIFIED_GB10, cgroupMemory: { maxBytes: C, currentBytes: 0 } });
    await ok.run({}, { runnerId: 'llama.cpp', modelId: SMALL_MODEL });
    await waitForLaunch(ok);
});

test('A-LL-16e: under a known share the Phase 1b guard reports not-applicable and never calls nvidia-smi', async (t) => {
    const calls = [];
    const h = gpuHarness(t, { gpu: UNIFIED_GB10, cgroupMemory: { maxBytes: 64 * GIB, currentBytes: GIB }, guardSampleMs: 5,
        controllerOptions: { gpuSampleMs: 5, readGpuProcesses: () => { calls.push(1); return Promise.resolve({ ok: false, reason: 'unexpected' }); },
            readProcGroup: () => null, readLeafMemoryStat: () => null } });
    await h.run({}, { runnerId: 'llama.cpp', modelId: SMALL_MODEL });
    await waitForLaunch(h);
    await new Promise((resolve) => setTimeout(resolve, 60));
    const guard = (await h.controller.status()).memoryGuard;
    assert.equal(guard.gpuBudgetSignal, 'not-applicable');
    assert.ok(guard.samples >= 3, 'the guard ticked');
    assert.equal(guard.gpuSamples, 0);
    assert.equal(calls.length, 0);
});

test('LL.gpu-unlimited-golden', async () => {
    const gpu = structuredClone(TEST_GPU);
    assert.equal(effectiveGpu(gpu, parseMpsBudget({})), gpu);
    const plain = { gpu, memory: { totalBytes: 128 * GIB, availableBytes: 120 * GIB }, cgroupMemory: null };
    const before = JSON.stringify(plain);
    attachGpuBudget(plain, parseMpsBudget({}));
    assert.equal(JSON.stringify(plain), before);
    const golden = JSON.parse(fs.readFileSync(new URL('./overview-golden-a84617d.json', import.meta.url), 'utf8'));
    const makeInstaller = ({ imageLockFile, dataDir }) => createRunnerInstaller({ lock: loadRunnerLocks({ image: imageLockFile, agent: null }), cacheRoot: path.join(dataDir, 'runners'), runRoot: path.join(dataDir, 'opt') });
    const args = { createController, createStateStore, validateModel, makeInstaller };
    assert.equal(JSON.stringify(await collectOverviews(args)), JSON.stringify(golden.scenarios));
    assert.equal(JSON.stringify(await collectOllamaRuns(args)), JSON.stringify(golden.ollamaRuns));
});
