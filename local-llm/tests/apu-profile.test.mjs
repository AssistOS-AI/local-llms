import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createStateStore } from 'pworker/local-llm';
import { APU, decideProfile } from 'pworker/local-llm/internal/controller/profiles.mjs';
import { parseRunnerReport } from 'pworker/local-llm/internal/controller/runnerProcess.mjs';
import { catalogProfileOf } from 'pworker/local-llm/internal/controller/vulkan.mjs';

import { createAgentController as createController, getRunner, loadSeedCatalog, vulkanContractOf } from '../src/engineOptions.mjs';
import { admit, readSnapshot } from '../src/hardwareLimits.mjs';
import { attachMemoryBudget, mpsRunnerEnvironment } from '../src/controller/ploinkyBudget.mjs';
import { hardwareCardsHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';
import { cpuFs, VULKAN_KEYS } from './overview-cpu-scenarios.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const seed = loadSeedCatalog();
const model = seed.find((entry) => entry.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const runner = getRunner('llama.cpp-vulkan');
const contract = vulkanContractOf(VULKAN_KEYS);
const fixture = (name) => fs.readFileSync(new URL(`./fixtures/vulkan/${name}`, import.meta.url), 'utf8');
const gpu = () => ({ available: true, backend: 'vulkan', driver: 'radv', memoryModel: 'shared', name: 'AMD Radeon Graphics',
    totalBytes: 16 * GIB, freeBytes: 16 * GIB, icdFile: contract.vulkan_icd_radv, processes: [] });
const snapshot = () => ({ gpu: gpu(), memory: { totalBytes: 32 * GIB, availableBytes: 24 * GIB }, disk: { freeBytes: 100 * GIB } });
const reportLines = [
    'using device Vulkan0 (AMD Radeon Graphics) (RADV) - 16384 MiB free',
    'offloaded 25/25 layers to GPU',
    'Vulkan0 model buffer size = 325.02 MiB',
    'Vulkan0 KV buffer size = 48.00 MiB',
];

function probe({ summary = fixture('radv-integrated-summary.txt'), devices = fixture('radv-integrated-devices.txt') } = {}) {
    const calls = [];
    return { calls, execFileImpl(command, args, options, callback) {
        calls.push({ command, args, options });
        if (command.endsWith('nvidia-smi')) callback(Object.assign(new Error('absent'), { code: 'ENOENT' }), '', '');
        else callback(null, command === contract.vulkaninfo ? summary : devices, '');
        return null;
    } };
}
async function readApu({ enabled = true, memoryMax = undefined, ...options } = {}) {
    const host = probe(options);
    const fake = cpuFs();
    const snap = await readSnapshot({ dataDir: '/tmp', apu: enabled, vulkan: contract, execFileImpl: host.execFileImpl,
        env: {}, statfs: async () => ({ bsize: 4096, blocks: 1000000, bavail: 999000 }),
        fsApi: { ...fake, readdirSync: () => ['renderD128'], lstatSync: () => ({ isCharacterDevice: () => true }),
            readFileSync(file) {
                if (String(file).endsWith('/memory.max') && memoryMax !== undefined) {
                    if (memoryMax === null) throw Object.assign(new Error('unreadable'), { code: 'EACCES' });
                    return memoryMax;
                }
                if (String(file).endsWith('/memory.current')) return '0';
                return fake.readFileSync(file);
            } } });
    return { snap, calls: host.calls };
}
function verdict({ snap = snapshot(), definition = runner, chosenModel = model, params = {}, profile = 'apu' } = {}) {
    return admit({ runner: definition, model: chosenModel, source: chosenModel.sources.gguf,
        params: definition.normalizeParams ? definition.normalizeParams(params, { model: chosenModel, profile }) : params,
        snapshot: snap, profile, gpuBudget: { state: 'absent' } });
}

test('T5: finite and unreadable cgroup limits refuse APU discovery; unlimited remains available', async () => {
    for (const memoryMax of [String(8 * GIB), null]) {
        const { snap } = await readApu({ memoryMax });
        assert.equal(snap.gpu.state, 'unusable');
        assert.match(snap.gpu.reason, /container memory limit/);
        assert.equal(decideProfile({ snapshot: snap }).profile, 'cpu');
    }
    assert.equal((await readApu({ memoryMax: 'max' })).snap.gpu.available, true);
});

test('T8-T10: APU uses Vulkan report patterns, never MPS, CPU catalog eligibility', () => {
    const report = parseRunnerReport(reportLines.map((line) => ({ line })), { profile: 'apu' });
    assert.equal(report.modelMiB, 325.02); assert.equal(report.kvMiB, 48);
    assert.match(report.device, /^Vulkan0/);
    assert.deepEqual(mpsRunnerEnvironment({ CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: 'broken' }, 'apu'), {});
    assert.equal(catalogProfileOf('apu'), 'cpu');
});

test('T12: committed APU refuses finite and unknown budgets without a policy call', () => {
    const finite = snapshot(); finite.cgroupMemory = { maxBytes: GIB, currentBytes: 0 };
    assert.equal(verdict({ snap: finite }).estimate.constraint, 'C3');
    const unknown = attachMemoryBudget(snapshot(), { memoryReadState: 'unknown', finiteMemoryBytes: null, headroomBytes: null });
    assert.equal(verdict({ snap: unknown }).reasonCode, 'budget_unreadable');
});

function harness(t, { lines = reportLines, ready = true, env = {}, initialState = null, lookup = undefined,
    installer = undefined, downloadImpl = null, killImpl = null, monotonicNow = null, activityCounterLimit = undefined,
    onSnapshot = null } = {}) {
    const root = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-apu-'));
    const memory = { totalBytes: 32 * GIB, availableBytes: 24 * GIB };
    const time = { monotonic: 0, wall: Date.now() }; const handles = []; const calls = [];
    let snap = snapshot(); let downloads = 0; let responseReady = ready;
    let budget = { memoryReadState: 'known', finiteMemoryBytes: null, headroomBytes: null };
    const store = createStateStore({ dataDir: root });
    if (initialState) store.save(initialState);
    const controller = createController({ dataDir: root, env: { PATH: '/usr/bin', LOCAL_LLM_AMD_APU: 'experimental', SECRET_SENTINEL: 'private-value', ...env },
        stateStore: store, seedCatalog: seed, imageContract: VULKAN_KEYS, hostArch: 'x64', fileExists: () => true,
        sharedModelsRoot: null, detectRunner: () => ({ installed: true, version: 'b11159', reason: null }),
        ...(lookup ? { lookup } : {}), ...(installer ? { installer } : {}),
        ...(activityCounterLimit === undefined ? {} : { apuActivityCounterLimit: activityCounterLimit }),
        readHardwareSnapshot: async (options) => { calls.push(options); await onSnapshot?.(); return snap; },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async (...args) => { downloads += 1; return downloadImpl ? downloadImpl(...args)
            : { status: 'complete', path: path.join(root, 'test.gguf'), bytesTransferred: 0 }; },
        startRunner({ command, args, env: runnerEnv, log }) {
            let running = true; let settle; let killed = false;
            const handle = { pid: 4242, command, args, env: runnerEnv, log, exited: new Promise((resolve) => { settle = resolve; }),
                get running() { return running; }, get killed() { return killed; },
                async stop() { running = false; settle({ code: 0, signal: 'SIGTERM' }); return handle.exited; },
                async kill() { killed = true; return killImpl ? killImpl(handle) : handle.stop(); } };
            handles.push(handle);
            for (const line of lines) log.append('stderr', line);
            return handle;
        },
        fetchImpl: async () => ({ ok: responseReady, status: responseReady ? 200 : 503, json: async () => ({}) }),
        pollMs: 2, unifiedGuardMs: 5, readMemory: () => memory, readPressure: () => 0,
        readBudget: () => { if (budget instanceof Error) throw budget; return budget; },
        monotonicNow: monotonicNow || (() => time.monotonic), now: () => new Date(time.wall),
    });
    t.after(async () => { await controller.drain(); fs.rmSync(root, { recursive: true, force: true }); });
    return { controller, handles, memory, time, calls, get downloads() { return downloads; },
        setSnapshot(value) { snap = value; }, setBudget(value) { budget = value; }, ready() { responseReady = true; } };
}
async function waitFor(check) {
    for (let i = 0; i < 500; i += 1) {
        if (await check()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('Timed out waiting for APU controller state');
}
const run = (h, params = {}, requestId = 'apu-run-000001') => h.controller.run({ modelId: model.id, runnerId: runner.id, params, requestId });

for (const [name, budget] of [['finite', { memoryReadState: 'known', finiteMemoryBytes: GIB }],
    ['unknown', { memoryReadState: 'unknown', finiteMemoryBytes: null }], ['missing', null], ['thrown', new Error('unreadable budget')]]) {
    test(`T13: ${name} budget observation kills an APU runner without CPU fallback`, async (t) => {
        const h = harness(t); await run(h); await waitFor(() => h.controller.state.deployment.phase === 'ready');
        h.setBudget(budget); await waitFor(() => h.controller.state.deployment.phase === 'error');
        assert.match(h.controller.state.deployment.error, /finite or unknown container memory limit/);
        assert.equal(h.handles[0].killed, true); assert.equal(h.handles.length, 1);
    });
}

test('T16: APU dashboard is experimental, shows shared pool, admission floor and observed ratio', () => {
    const limits = { floorBytes: GIB, poolBytes: 32 * GIB, availableBytes: 24 * GIB,
        offloadReport: { modelBufferRatio: 0.75, modelBufferDenominatorBasis: 'full pinned weights' } };
    const html = hardwareCardsHtml(snapshot(), null, { profile: 'apu', limits });
    assert.match(html, /AMD integrated GPU/); assert.match(html, /Experimental, unqualified/);
    assert.match(html, /Shared memory/); assert.match(html, /0.750/); assert.match(html, /full pinned weights/);
});
test('T17: optional APU manifest switches have no defaults and declare no memory limit', () => {
    const manifest = JSON.parse(fs.readFileSync(new URL('../manifest.json', import.meta.url), 'utf8'));
    for (const key of ['LOCAL_LLM_AMD_APU', 'LOCAL_LLM_APU_GUARD_FLOOR_BYTES']) {
        assert.deepEqual(manifest.profiles.default.env[key], { required: false });
    }
    assert.equal(manifest.resources?.memory, undefined);
});

async function refuse(h, requestId = 'apu-refusal-0001') {
    const snap = snapshot(); snap.memory.availableBytes = 512 * MIB; h.setSnapshot(snap);
    let caught;
    try { await run(h, {}, requestId); } catch (error) { caught = error; }
    assert.equal(caught.code, 'admission_insufficient_now');
    return caught.details.apuAttempt;
}
