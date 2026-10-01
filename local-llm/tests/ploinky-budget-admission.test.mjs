// Public admission under a Ploinky container memory budget (DS003, DS005):
// every runner path goes through createController().run(), the entry the
// application uses, with the snapshot's cgroup readings parsed by the
// production readers from file contents. For each path the required RAM R and
// its reserve F come from the path's own admission estimate; the budget's
// headroom is then set to 0, 1, R+F-1, R+F and R+F+1 bytes (and past the
// limit, and unreadable), holding the limit fixed. Nothing here needs a GPU,
// a container or the network.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { RAM_MARGIN_BYTES } from '../src/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { readCgroupMemory, readCgroupMemoryObservation } from '../src/controller/hardware.mjs';
import { attachMemoryBudget } from '../src/controller/ploinkyBudget.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { RUNNERS } from '../src/runners/index.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();
const KEY = 'k'.repeat(43);
// The container limit, held fixed in every vector, and the host's raw memory.
const CAP = 64 * GIB;
const RAW = Object.freeze({ totalBytes: 128 * GIB, availableBytes: 120 * GIB });
// Headroom for deriving R and F: everything fits.
const DERIVE_HEADROOM = 60 * GIB;

// A unified-memory copy of the AWQ snapshot, so the unified vLLM path launches
// a model without gpt-oss's tokenizer-vocabulary check.
const AWQ = SEED.find((entry) => entry.id === 'qwen3-4b-awq');
const AWQ_UNIFIED = validateModel({
    ...structuredClone(Object.fromEntries(Object.entries(AWQ).filter(([key]) => !['seed', 'recommended'].includes(key)))),
    id: 'qwen3-4b-awq-unified', profiles: ['unified'],
}, { seed: true });

const DEDICATED_GPU = Object.freeze({
    available: true, name: 'Test GPU', driverVersion: '595.91.07', memoryModel: 'dedicated',
    totalBytes: 24 * GIB, usedBytes: 256 * MIB, freeBytes: 24 * GIB - 256 * MIB, processes: [],
    device: { pciDeviceId: '0x252010DE', computeCapability: '8.6', addressingMode: 'None' },
});
const UNIFIED_GPU = Object.freeze({
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
});
const NO_GPU = Object.freeze({ available: false, state: 'absent', reason: 'No NVIDIA GPU is attached to this agent.' });

const reserveOf = {
    margin: () => RAM_MARGIN_BYTES,
    offloadFloor: (estimate) => estimate.ramFloorBytes,
    poolFloor: (estimate) => estimate.floorBytes,
};

// Each public path: the profile's GPU, the runner, the model and its params,
// the required RAM R (from the estimate) and the reserve F the policy keeps.
export const PATHS = Object.freeze({
    llama: { gpu: DEDICATED_GPU, runnerId: 'llama.cpp', modelId: 'qwen2.5-0.5b-instruct-q4_k_m', need: 'ramBytes', reserve: 'margin' },
    'ik-llama': { gpu: DEDICATED_GPU, runnerId: 'ik_llama.cpp', modelId: 'qwen2.5-0.5b-instruct-q4_k_m', need: 'ramBytes', reserve: 'margin' },
    'lm-studio': { gpu: DEDICATED_GPU, runnerId: 'lmstudio', modelId: 'qwen2.5-0.5b-instruct-q4_k_m', need: 'ramBytes', reserve: 'margin' },
    'ollama-auto': { gpu: DEDICATED_GPU, runnerId: 'ollama', modelId: 'gpt-oss-20b', need: 'ramBytes', reserve: 'margin' },
    'ollama-pinned': { gpu: DEDICATED_GPU, runnerId: 'ollama', modelId: 'gpt-oss-20b', params: { numGpu: 12 }, need: 'ramBytes', reserve: 'margin' },
    vllm: { gpu: DEDICATED_GPU, runnerId: 'vllm', modelId: 'qwen3-4b-awq', need: 'ramBytes', reserve: 'margin' },
    'vllm-offload': { gpu: DEDICATED_GPU, runnerId: 'vllm', modelId: 'qwen3-4b-awq', params: { cpuOffloadGb: 1 }, need: 'ramBytes', reserve: 'offloadFloor' },
    tabby: { gpu: DEDICATED_GPU, runnerId: 'tabbyapi', modelId: 'qwen3-8b-exl3', need: 'ramBytes', reserve: 'margin' },
    'cpu-llama': { gpu: NO_GPU, runnerId: 'llama.cpp', modelId: 'qwen2.5-0.5b-instruct-q4_k_m', need: 'ramBytes', reserve: 'poolFloor' },
    'cpu-ollama': { gpu: NO_GPU, runnerId: 'ollama', modelId: 'gpt-oss-20b', need: 'ramBytes', reserve: 'poolFloor' },
    'unified-llama': { gpu: UNIFIED_GPU, runnerId: 'llama.cpp', modelId: 'qwen2.5-0.5b-instruct-q4_k_m', need: 'unifiedBytes', reserve: 'poolFloor' },
    'unified-vllm': { gpu: UNIFIED_GPU, runnerId: 'vllm', modelId: 'qwen3-4b-awq-unified', params: { gpuMemoryUtilization: 0.3 }, need: 'unifiedBytes', reserve: 'poolFloor' },
});

// A file value of {error: CODE} exists but cannot be read; undefined is absent.
function cgroupFs({ max, current }) {
    const files = { '/sys/fs/cgroup/memory.max': max, '/sys/fs/cgroup/memory.current': current };
    return {
        readFileSync(file) {
            const value = files[file];
            if (value && typeof value === 'object') throw Object.assign(new Error(`${value.error}: ${file}`), { code: value.error });
            if (value !== undefined) return value;
            throw Object.assign(new Error(`ENOENT: ${file}`), { code: 'ENOENT' });
        },
    };
}

// The snapshot the controller sees: the production readers parse the cgroup
// files; the internal observation is attached as readSnapshot attaches it.
function budgetSnapshot(gpu, cgroup) {
    const fsApi = cgroupFs(cgroup);
    const snap = {
        gpu: structuredClone(gpu), memory: { ...RAW }, disk: { freeBytes: 1024 * GIB, totalBytes: 2048 * GIB }, cpus: 20, cores: 10,
        cgroupMemory: readCgroupMemory({ fsApi }),
    };
    return attachMemoryBudget(snap, readCgroupMemoryObservation({ fsApi, established: null }));
}

const withHeadroom = (headroom) => ({ max: `${CAP}\n`, current: `${CAP - headroom}\n` });

function harness(t, spec, initialCgroup) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-budget-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const dataDir = path.join(root, 'data');
    const weights = path.join(root, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const runDir = path.join(root, 'opt', 'runners', 'run');
    // TabbyAPI writes its token file into its runnable copy.
    fs.mkdirSync(path.join(runDir, 'tabbyAPI'), { recursive: true });
    // LM Studio's start first looks for stray processes in /proc; an empty
    // process table here, through the adapter's own procDir option.
    const procDir = path.join(root, 'proc');
    fs.mkdirSync(procDir);
    const runners = Object.freeze({
        ...RUNNERS,
        lmstudio: Object.freeze({ ...RUNNERS.lmstudio, start: (ctx) => RUNNERS.lmstudio.start(ctx, { procDir }) }),
    });
    let cgroup = initialCgroup;
    const started = [];
    const downloads = [];
    const installer = {
        installable: () => true,
        describe: async () => ({ installed: true, runnable: true, version: 'test', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
        ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }),
        entryFor: (id) => ({ id, version: 'test' }),
        pathsFor: () => ({ runDir }),
    };
    const controller = createController({
        dataDir,
        // The operator switches for the runners that need one (vLLM on unified memory, LM Studio).
        env: {
            PATH: '/usr/bin', LOCAL_LLM_VLLM_UNIFIED: 'experimental', LOCAL_LLM_LMSTUDIO: 'internal-use',
            LOCAL_LLM_RUN_ROOT: path.join(root, 'opt', 'runners'),
        },
        seedCatalog: [...SEED, AWQ_UNIFIED],
        runners,
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => budgetSnapshot(spec.gpu, cgroup),
        download: async ({ artifact }) => { downloads.push(artifact); return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }),
        downloadSnapshot: async () => { downloads.push('snapshot'); return { status: 'complete', bytesTransferred: 0 }; },
        installer,
        startRunner({ command, args, env }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 6000 + started.length, command, args, env,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                async kill() { running = false; resolveExit({ code: null, signal: 'SIGKILL', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion ?? 'test', reason: null }),
        imageContract: null,
        fileExists: () => true,
        sharedModelsRoot: null,
        shmDir: path.join(root, 'shm'),
        readMemory: () => ({ ...RAW }),
        readPressure: () => 0,
        readBudget: () => readCgroupMemoryObservation({ fsApi: cgroupFs(cgroup), established: null }),
        unifiedGuardMs: 60_000,
        memoryGuardLoadMs: 60_000,
        memoryGuardReadyMs: 60_000,
        dropCache: () => true,
        pollMs: 2,
        stopGraceMs: 20,
    });
    return {
        controller, started, downloads,
        setCgroup(next) { cgroup = next; },
        run: (requestId) => controller.run({ modelId: spec.modelId, runnerId: spec.runnerId, requestId, ...(spec.params ? { params: spec.params } : {}) }),
    };
}

async function until(predicate, timeoutMs = 4000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 2));
    }
    assert.fail('condition not reached');
}

// R and F for a path, from its own admission at ample headroom.
const derived = new Map();
async function requirement(t, name) {
    if (derived.has(name)) return derived.get(name);
    const spec = PATHS[name];
    const h = harness(t, spec, withHeadroom(DERIVE_HEADROOM));
    const accepted = await h.run(`request-derive-${name.replace(/\W/g, '')}`);
    assert.equal(accepted.accepted, true, `${name} fits permanently with ample headroom`);
    const estimate = h.controller.state.deployment.admission.estimate;
    await h.controller.stop();
    const need = estimate[spec.need];
    const reserve = reserveOf[spec.reserve](estimate);
    assert.ok(Number.isSafeInteger(need) && need > 0, `${name}: R is a positive byte count`);
    assert.ok(Number.isSafeInteger(reserve) && reserve > 0, `${name}: F is a positive byte count`);
    assert.ok(need + reserve + 1 < DERIVE_HEADROOM, `${name}: R+F fits the derivation headroom`);
    const value = { need, reserve };
    derived.set(name, value);
    return value;
}

async function assertRefusedNow(h, requestId, { reasonCode = undefined } = {}) {
    await assert.rejects(() => h.run(requestId), (error) => {
        assert.equal(error.code, 'admission_insufficient_now', error.message);
        assert.equal(error.details?.admission?.status, 'insufficient-now');
        if (reasonCode !== undefined) assert.equal(error.details.admission.reasonCode, reasonCode);
        else assert.equal(error.details.admission.reasonCode, undefined);
        return true;
    });
    assert.equal(h.started.length, 0, 'no runner launched');
    assert.equal(h.downloads.length, 0, 'nothing downloaded');
    assert.equal(h.controller.state.deployment ?? null, null, 'no deployment recorded');
}

async function assertAccepted(h, requestId) {
    const accepted = await h.run(requestId);
    assert.equal(accepted.accepted, true);
    assert.equal(h.controller.state.deployment.admission.status, 'ok', h.controller.state.deployment.admission.reason);
    await h.controller.stop();
}

async function releaseRetry(t, name) {
    const { need, reserve } = await requirement(t, name);
    const h = harness(t, PATHS[name], withHeadroom(0));
    await assertRefusedNow(h, 'request-held');
    // The limit is unchanged; only current use falls (memory released).
    h.setCgroup(withHeadroom(need + reserve));
    const accepted = await h.run('request-released');
    assert.equal(accepted.accepted, true);
    await until(() => h.started.length === 1).catch((error) => {
        const deployment = h.controller.state.deployment;
        throw new Error(`${error.message}: phase ${deployment?.phase}, error ${deployment?.error}`);
    });
    assert.equal(h.started.length, 1, 'launch spy 0 -> 1');
    await h.controller.stop();
}

const VECTORS = Object.freeze({
    zero: async (t, name) => assertRefusedNow(harness(t, PATHS[name], withHeadroom(0)), 'request-zero'),
    'over-current': async (t, name) => assertRefusedNow(
        harness(t, PATHS[name], { max: `${CAP}\n`, current: `${CAP + GIB}\n` }), 'request-over'),
    'one-byte': async (t, name) => assertRefusedNow(harness(t, PATHS[name], withHeadroom(1)), 'request-one'),
    'reserve-minus-one': async (t, name) => {
        const { need, reserve } = await requirement(t, name);
        await assertRefusedNow(harness(t, PATHS[name], withHeadroom(need + reserve - 1)), 'request-minus');
    },
    'exact-reserve': async (t, name) => {
        const { need, reserve } = await requirement(t, name);
        await assertAccepted(harness(t, PATHS[name], withHeadroom(need + reserve)), 'request-exact');
    },
    'reserve-plus-one': async (t, name) => {
        const { need, reserve } = await requirement(t, name);
        await assertAccepted(harness(t, PATHS[name], withHeadroom(need + reserve + 1)), 'request-plus');
    },
    'missing-current': async (t, name) => assertRefusedNow(
        harness(t, PATHS[name], { max: `${CAP}\n`, current: undefined }), 'request-missing', { reasonCode: 'budget_unreadable' }),
    'malformed-current': async (t, name) => assertRefusedNow(
        harness(t, PATHS[name], { max: `${CAP}\n`, current: 'not-a-number\n' }), 'request-malformed', { reasonCode: 'budget_unreadable' }),
    'release-retry': releaseRetry,
});

test('LL.controller-zero-no-launch', async (t) => {
    // Known zero headroom is exhausted memory, not an absent reading: refused now, nothing launched.
    for (const name of ['llama', 'cpu-llama', 'unified-llama']) {
        await assertRefusedNow(harness(t, PATHS[name], withHeadroom(0)), `request-zero-${name.replace(/\W/g, '')}`);
    }
});

test('LL.controller-unknown-no-launch', async (t) => {
    // A finite limit with unreadable current use refuses every profile, the CPU one included.
    for (const name of ['llama', 'cpu-llama', 'cpu-ollama', 'unified-llama']) {
        const h = harness(t, PATHS[name], { max: `${CAP}\n`, current: undefined });
        await assertRefusedNow(h, `request-unknown-${name.replace(/\W/g, '')}`, { reasonCode: 'budget_unreadable' });
        await assert.rejects(() => h.run('request-unknown-again'), (error) => {
            assert.match(error.message, /current memory use cannot be read/);
            assert.doesNotMatch(error.message, /never|cannot run/i, 'not a permanent incompatibility');
            return true;
        });
    }
});

test('LL.controller-limit-absent-admitted', async (t) => {
    // No memory.max at all (ENOENT), as on hosts without a container limit:
    // admitted exactly as before, no budget applied.
    for (const name of ['llama', 'cpu-llama']) {
        const h = harness(t, PATHS[name], { max: undefined, current: undefined });
        const accepted = await h.run(`request-absent-${name.replace(/\W/g, '')}`);
        assert.equal(accepted.accepted, true);
        assert.equal(h.controller.state.deployment.admission.status, 'ok', h.controller.state.deployment.admission.reason);
        assert.equal(h.controller.state.deployment.admission.reasonCode, undefined);
        await until(() => h.started.length === 1);
        await h.controller.stop();
    }
});

test('LL.controller-limit-unreadable-no-launch', async (t) => {
    // memory.max exists but cannot be read (EACCES, EIO) or holds garbage:
    // the limit is unknown, never unlimited, so a new start is refused before
    // any runner dispatch, as a temporary condition.
    for (const max of [{ error: 'EACCES' }, { error: 'EIO' }, 'garbage\n', '\n']) {
        const h = harness(t, PATHS.llama, { max, current: `${GIB}\n` });
        await assertRefusedNow(h, 'request-limit-unreadable', { reasonCode: 'budget_unreadable' });
        await assert.rejects(() => h.run('request-limit-unreadable-again'), (error) => {
            assert.match(error.message, /container memory limit cannot be read/);
            assert.doesNotMatch(error.message, /never|cannot run/i, 'not a permanent incompatibility');
            return true;
        });
    }
});

test('LL.controller-release-retry', async (t) => {
    await releaseRetry(t, 'llama');
    await releaseRetry(t, 'cpu-llama');
});

for (const name of Object.keys(PATHS)) {
    for (const [vector, run] of Object.entries(VECTORS)) {
        test(`LL-RAM.${name}.${vector}`, async (t) => run(t, name));
    }
}
