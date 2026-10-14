// The unified hardware profile (DS005): a GPU that shares system memory
// (NVIDIA GB10 in DGX Spark). Launch, envelope admission, profile selection,
// runner availability from the image, and the memory guard.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController, dropPageCache, readImageContract } from '../src/controller/deployments.mjs';
import { attachMemoryBudget, observeMemoryBudget } from '../src/controller/ploinkyBudget.mjs';
import { UNIFIED, envelopeFor, profileOf } from '../src/controller/profiles.mjs';
import { createLogBuffer, startRunnerProcess } from '../src/controller/runnerProcess.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { getRunner } from '../src/runners/index.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const SEED = loadSeedCatalog();
const GGUF = Object.freeze({
    type: 'huggingface', repo: 'ggml-org/gpt-oss-120b-GGUF', file: 'gpt-oss-120b-MXFP4.gguf', revision: 'main',
    commit: 'a'.repeat(40), size: 63387346208, sha256: 'b'.repeat(64),
});
// A model measured on unified memory with dio: 128k x 4 slots, and 256k x 1
// with MTP, which its weights' declared prediction head (mtp) allows.
const MEASURED = validateModel({
    id: 'big-moe', displayName: 'Big MoE', architecture: 'moe', contextLength: 131072, mtp: true, sources: { gguf: GGUF },
    profiles: ['unified'],
    recommended: { unified: { 'llama.cpp': { ctxSize: 32768 } } },
    unified: { envelope: [
        { runner: 'llama.cpp', loadMode: 'dio', maxCtx: 131072, maxParallel: 4, mtp: false, bufferBytes: 66 * GIB, transientBytes: 13 * GIB },
        { runner: 'llama.cpp', loadMode: 'dio', maxCtx: 262144, maxParallel: 1, mtp: true, bufferBytes: 70 * GIB, transientBytes: 2 * GIB },
    ] },
}, { seed: true });
const GB10 = Object.freeze({
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
    telemetry: { utilizationPercent: 0, powerWatts: 4.5, temperatureC: 47 },
});
const TOTAL = 125442396 * 1024; // 119.63 GiB

function unifiedSnapshot({ available = 108 * GIB, gpu = GB10 } = {}) {
    return { gpu: structuredClone(gpu), memory: { totalBytes: TOTAL, availableBytes: available }, disk: { freeBytes: 400 * GIB }, cpus: 20 };
}

test('the snapshot decides the profile; no usable GPU keeps the dedicated rules', () => {
    assert.equal(profileOf(unifiedSnapshot()), 'unified');
    assert.equal(profileOf({ gpu: { available: true, memoryModel: 'dedicated', totalBytes: 6144 * MIB } }), 'dedicated');
    assert.equal(profileOf({ gpu: { available: false, reason: 'no GPU' } }), 'dedicated');
    assert.equal(profileOf(null), 'dedicated');
});

test('llama.cpp on unified memory: the envelope flags are fixed, the admin chooses inside them', () => {
    const PATH = '/data/models/gguf/big.gguf';
    const launch = (params, extra = {}) => llamaCppRunner.buildLaunch({
        artifactPath: PATH, params, port: 18080, apiKey: KEY, model: MEASURED, profile: 'unified', ...extra,
    });
    const values = llamaCppRunner.normalizeParams({}, { model: MEASURED, profile: 'unified' });
    assert.deepEqual(values, { ctxSize: 32768, parallel: 1, loadMode: 'dio', mtp: false, threads: null });
    const { args, env } = launch({ threads: 10 });
    assert.deepEqual(args, [
        '-m', PATH, '--host', '127.0.0.1', '--port', '18080', '--api-key', KEY, '--no-webui', '-lv', '4', '--alias', 'big-moe',
        '--ctx-size', '32768', '--n-gpu-layers', '999', '--flash-attn', 'on', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
        '--threads', '10', '-np', '1', '--batch-size', '2048', '--ubatch-size', '2048', '--cache-ram', String(UNIFIED.cacheRamMiB),
        '--load-mode', 'dio',
    ]);
    assert.deepEqual(env, { LD_LIBRARY_PATH: '/usr/local/nvidia/lib64' });
    const parallel = launch({ ctxSize: 131072, parallel: 4, loadMode: 'none', mtp: true, threads: 10,
        chatTemplateKwargs: { preserve_thinking: true } }).args;
    assert.ok(parallel.includes('--kv-unified'));
    assert.deepEqual(parallel.slice(parallel.indexOf('--spec-type'), parallel.indexOf('--spec-type') + 4),
        ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3']);
    assert.ok(parallel.includes('--chat-template-kwargs') && parallel.includes('{"preserve_thinking":true}'));
    // The dedicated-only parameters do not exist here, and dio is not a dedicated default.
    for (const param of ['nCpuMoe', 'nGpuLayers', 'batchSize', 'cacheTypeK', 'flashAttn']) {
        assert.throws(() => llamaCppRunner.normalizeParams({ [param]: 1 }, { model: MEASURED, profile: 'unified' }), /unknown parameter/);
    }
    assert.throws(() => llamaCppRunner.normalizeParams({ loadMode: 'mmap' }, { model: MEASURED, profile: 'unified' }), /loadMode/);
    // Context is capped at the model's training context per slot.
    assert.throws(() => llamaCppRunner.normalizeParams({ ctxSize: 262144, parallel: 1 }, { model: MEASURED, profile: 'unified' }),
        /ctxSize.*131072/);
    assert.equal(llamaCppRunner.normalizeParams({ ctxSize: 262144, parallel: 2 }, { model: MEASURED, profile: 'unified' }).ctxSize, 262144);
    // The unified schema is what the overview shows for this profile.
    assert.equal(llamaCppRunner.paramSchemaFor('unified').properties.ctxSize.maximum, 262144);
    assert.equal(llamaCppRunner.paramSchemaFor('dedicated').properties.ctxSize.maximum, 131072);
});

test('unified threads default to the high-performance cores, else physical cores minus 2', async () => {
    const { createLlamaServerRunner, LOAD_MODES } = await import('../src/runners/llamaServer.mjs');
    const make = (perf) => createLlamaServerRunner({
        id: 'llama.cpp', displayName: 'llama.cpp', executable: '/x', pinnedVersion: 'b1', port: 18080,
        dialect: { quietArgs: [], unifiedKv: true, loadModes: LOAD_MODES, loadArgs: () => [], jinja: () => false, parseVersion: () => null, unified: true },
        cpuCores: () => 20, perfCores: () => perf,
    });
    const threadsOf = (runner, profile) => {
        const { args } = runner.buildLaunch({ artifactPath: '/m.gguf', params: {}, port: 18080, apiKey: KEY, model: { id: 'm' }, profile });
        return args[args.indexOf('--threads') + 1];
    };
    assert.equal(threadsOf(make(10), 'unified'), '10');
    assert.equal(threadsOf(make(null), 'unified'), '18');
    assert.equal(threadsOf(make(10), 'dedicated'), '18');
});

test('unified admission: the measured envelope, the whole known allocation, the host reserve and the floor', () => {
    const runner = getRunner('llama.cpp');
    const decide = (params, snap = unifiedSnapshot()) => admit({
        runner, model: MEASURED, source: GGUF, params: runner.normalizeParams(params, { model: MEASURED, profile: 'unified' }),
        snapshot: snap, profile: 'unified',
    });
    const inside = decide({ ctxSize: 131072, parallel: 4 });
    assert.equal(inside.status, 'ok', inside.reason);
    const need = 66 * GIB + 13 * GIB + UNIFIED.runtimeBytes + UNIFIED.cacheRamMiB * MIB;
    assert.equal(inside.estimate.unifiedBytes, need);
    assert.deepEqual(inside.estimate.envelope, { loadMode: 'dio', maxCtx: 131072, maxParallel: 4, mtp: false });
    assert.equal(inside.estimate.cacheRamBytes, 8192 * MIB);
    // The corner's figures cover every configuration inside the rectangle.
    assert.equal(decide({ ctxSize: 8192, parallel: 1 }).estimate.unifiedBytes, need);
    // MTP only where it was measured.
    assert.equal(decide({ ctxSize: 65536, parallel: 1, mtp: true }).estimate.envelope.maxCtx, 262144);
    // Outside the rectangles the run is sized by estimate, not refused (owner, 2026-09-29: no benchmark gate).
    const outside = decide({ ctxSize: 131072, parallel: 8 });
    assert.equal(outside.status, 'ok', outside.reason);
    assert.equal(outside.estimate.envelope, null);
    assert.equal(outside.estimate.measured, false);
    assert.match(outside.warnings[0], /^Outside the catalog's envelope \(context 131072 x 4 slots with load mode dio; context 262144 x 1 slot \(MTP allowed\) with load mode dio\): context 131072 x 8 slots with load mode dio is sized by estimate\./);
    // Less available than need plus the floor: busy now, naming the numbers.
    const busy = decide({ ctxSize: 131072, parallel: 4 }, unifiedSnapshot({ available: need + UNIFIED.floorBytes - GIB }));
    assert.equal(busy.status, 'insufficient-now');
    assert.match(busy.reason, /Other processes on this machine hold the rest/);
    // More than the pool less the host reserve: never.
    const huge = validateModel({ ...structuredClone(MEASURED), unified: { envelope: [
        { runner: 'llama.cpp', loadMode: 'dio', maxCtx: 131072, maxParallel: 1, mtp: false, bufferBytes: 100 * GIB, transientBytes: 0 },
    ] } }, { seed: true });
    const never = admit({ runner, model: huge, source: GGUF, params: runner.normalizeParams({}, { model: huge, profile: 'unified' }),
        snapshot: unifiedSnapshot(), profile: 'unified' });
    assert.equal(never.status, 'incompatible');
    assert.match(never.reason, /must keep 16\.0 GiB for the host/);
    // A model never measured here runs by a labelled estimate, still inside the reserve and the floor.
    const [gpt] = SEED;
    const unmeasured = admit({ runner, model: gpt, source: gpt.sources.gguf, params: runner.normalizeParams({}, { model: gpt, profile: 'unified' }),
        snapshot: unifiedSnapshot(), profile: 'unified' });
    assert.equal(unmeasured.status, 'ok', unmeasured.reason);
    assert.equal(unmeasured.estimate.measured, false);
    assert.match(unmeasured.warnings[0], /^Not measured on unified memory/);
    // No readable pool: refused.
    const blind = decide({}, { ...unifiedSnapshot(), memory: { totalBytes: null, availableBytes: null } });
    assert.equal(blind.status, 'incompatible');
    assert.match(blind.reason, /cannot be read/);
    assert.equal(envelopeFor(MEASURED, 'ollama', { ctxSize: 1, parallel: 1, loadMode: 'dio' }), null);
});

test('a profile that no longer matches the GPU refuses every Run until the agent restarts', () => {
    const runner = getRunner('llama.cpp');
    const params = runner.normalizeParams({}, { model: MEASURED, profile: 'unified' });
    const result = admit({ runner, model: MEASURED, source: GGUF, params, profile: 'dedicated', snapshot: unifiedSnapshot() });
    assert.equal(result.status, 'incompatible');
    assert.match(result.reason, /now reports unified memory, but this agent started with the dedicated profile; restart local-llm/);
});

function deferred() {
    let resolve;
    const promise = new Promise((ok) => { resolve = ok; });
    return { promise, resolve };
}

function harness(t, { snap = unifiedSnapshot(), seed = [...SEED, MEASURED], imageContract = null, fileExists = () => true,
    readMemory, readPressure = () => 0, readBudget, unifiedGuardMs = 5, downloads = true, dropCache, controllerOptions = {}, exitDelayMs = 0 } = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-unified-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const started = [];
    const calls = { download: [], dropped: [] };
    const weights = path.join(dataDir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: seed,
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => (typeof snap === 'function' ? snap() : structuredClone(snap)),
        download: async ({ artifact }) => { calls.download.push(artifact); return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        inspect: async () => ({ state: downloads ? 'absent' : 'complete', bytes: 0 }),
        remove: async () => 0,
        startRunner({ command, args, env }) {
            const exit = deferred();
            let running = true;
            const handle = {
                pid: 7000 + started.length, command, args, env, exited: exit.promise,
                get running() { return running; },
                async stop() { running = false; handle.stopped = 'SIGTERM'; exit.resolve({ code: 0, signal: 'SIGTERM', error: null }); return exit.promise; },
                async kill() {
                    running = false;
                    handle.stopped = 'SIGKILL';
                    // A real process takes a moment to exit after SIGKILL; a test may model that.
                    if (exitDelayMs) setTimeout(() => exit.resolve({ code: null, signal: 'SIGKILL', error: null }), exitDelayMs);
                    else exit.resolve({ code: null, signal: 'SIGKILL', error: null });
                    return exit.promise;
                },
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
        readMemory: readMemory || (() => ({ totalBytes: TOTAL, availableBytes: 100 * GIB })),
        readPressure,
        readBudget,
        unifiedGuardMs,
        dropCache: dropCache || ((file) => { calls.dropped.push(file); return true; }),
        ...controllerOptions,
    });
    return { controller, started, calls, dataDir, weights };
}

async function until(predicate, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

test('the controller takes the unified profile from its first snapshot and offers only the models for it', async (t) => {
    const h = harness(t);
    const overview = await h.controller.overview();
    assert.equal(overview.profile, 'unified');
    const ids = overview.models.map((model) => model.id);
    assert.ok(ids.includes('big-moe') && ids.includes('gpt-oss-20b'));
    assert.ok(!ids.includes('qwen3-4b-awq') && !ids.includes('qwen3-8b-exl3'), 'dedicated-only seeds are not offered');
    const llama = overview.runners.find((runner) => runner.id === 'llama.cpp');
    assert.equal(llama.paramSchema.properties.ctxSize.maximum, 262144);
    assert.equal(overview.runners.find((runner) => runner.id === 'ollama').paramSchema, null);
    const big = overview.models.find((model) => model.id === 'big-moe');
    assert.equal(big.runners['llama.cpp'].admission.status, 'ok');
    assert.equal(big.runners['llama.cpp'].params.loadMode, 'dio');
    // A dedicated host does not see the unified-only model.
    const dedicated = harness(t, { snap: { gpu: { available: true, memoryModel: 'dedicated', name: 'RTX', totalBytes: 6144 * MIB, usedBytes: 0, freeBytes: 6144 * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB }, disk: { freeBytes: 300 * GIB } } });
    const dedicatedIds = (await dedicated.controller.overview()).models.map((model) => model.id);
    assert.ok(!dedicatedIds.includes('big-moe'));
});

test('a runner the image lacks is not available on this platform, and Run refuses it before any download', async (t) => {
    const present = new Set(['/opt/llama.cpp/llama-server']);
    const h = harness(t, { imageContract: { architecture: 'arm64', gpu_compute_capabilities: '12.1' }, fileExists: (file) => present.has(file) });
    const overview = await h.controller.overview();
    const byId = Object.fromEntries(overview.runners.map((runner) => [runner.id, runner]));
    assert.equal(byId['llama.cpp'].supported, true);
    for (const id of ['ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio']) {
        assert.equal(byId[id].supported, false, id);
        assert.match(byId[id].unsupportedReason, /not available on this platform: this image does not include it/, id);
        // The Runners tab reads `reason`: it names the platform, never the version probe's ENOENT text.
        assert.match(byId[id].reason, /not available on this platform: this image does not include it/, id);
        assert.doesNotMatch(byId[id].reason, /Executable not found/, id);
        assert.equal(byId[id].installed, false, id);
    }
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'ik_llama.cpp', requestId: 'request-0001' }),
        (error) => error.code === 'runner_unavailable');
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'vllm', requestId: 'request-0002' }),
        (error) => error.code === 'runner_unavailable');
    assert.equal(h.calls.download.length, 0);
    const gpt = overview.models.find((model) => model.id === 'gpt-oss-20b');
    assert.deepEqual(gpt.weights.gguf.runners, ['llama.cpp']);
    assert.match(gpt.runners.ollama.admission.reason, /not available on this platform/);
    // Availability comes before the operator switch: LM Studio is off here and also absent from the image.
    assert.match(gpt.runners.lmstudio.admission.reason, /not available on this platform/);
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'lmstudio', requestId: 'request-0003' }),
        (error) => error.code === 'runner_unavailable');
});

// Deliberate change (CPU profile, DS005): a GPU the image's CUDA runners were not
// built for used to be refused; it now decides the cpu profile, with the mismatch as its reason.
test('a GPU the image\'s runners were not built for decides the cpu profile with the mismatch reason; only CPU-offered models run, and nothing downloads', async (t) => {
    const thor = { ...GB10, name: 'NVIDIA Thor', device: { ...GB10.device, pciDeviceId: '0x2B0010DE', computeCapability: '11.0', addressingMode: 'HMM' } };
    const tiny = validateModel({
        id: 'tiny-cpu', displayName: 'Tiny CPU', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 }, profiles: ['cpu'],
        sources: { gguf: { ...GGUF, file: 'tiny.gguf', size: 400 * MIB } },
    }, { seed: true });
    const h = harness(t, { snap: unifiedSnapshot({ gpu: thor }), imageContract: { gpu_compute_capabilities: '12.1' }, seed: [...SEED, MEASURED, tiny] });
    const overview = await h.controller.overview();
    assert.equal(overview.profile, 'cpu');
    assert.equal(overview.profileDecision.cause, 'mismatch');
    assert.match(overview.profileDecision.reason, /built for GPUs of compute capability 12\.1; NVIDIA Thor is 11\.0/);
    // A CPU-offered model is admitted on llama.cpp; the unified-only model is not offered at all.
    assert.equal(overview.models.find((model) => model.id === 'tiny-cpu').runners['llama.cpp'].admission.status, 'ok');
    assert.ok(!overview.models.some((model) => model.id === 'big-moe'));
    await assert.rejects(() => h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0001' }),
        (error) => error.code === 'unknown_model');
    assert.equal(h.calls.download.length, 0);
});

// GPU hosts stay as they were (the cpu profile's rule 3): on unified memory a runner with no unified policy is
// refused exactly as before the cpu profile existed, never with the cpu profile's "needs an NVIDIA GPU" refusal.
// The outcomes below were observed at the commit before the cpu profile (349f8b0).
test('on unified memory a runner with no unified policy is refused exactly as before the cpu profile', async (t) => {
    const h = harness(t);
    const overview = await h.controller.overview();
    assert.equal(overview.profile, 'unified');
    for (const runner of overview.runners) assert.equal('profileUnsupportedReason' in runner, false, runner.id);
    const run = (runnerId) => h.controller.run({ modelId: 'gpt-oss-20b', runnerId, requestId: `request-${runnerId.replace(/\W/g, '')}` });
    // Ollama has no unified policy: admission refuses it, naming the GPU.
    await assert.rejects(() => run('ollama'), (error) => error.code === 'admission_incompatible'
        && error.message === 'Ollama is not available on a GPU that shares system memory (NVIDIA GB10) in this release.');
    // LM Studio is off: the operator's switch text, not a profile refusal.
    await assert.rejects(() => run('lmstudio'), (error) => error.code === 'runner_disabled'
        && /LM Studio is not enabled on this deployment \(internal use only\)/.test(error.message)
        && /ploinky var LOCAL_LLM_LMSTUDIO internal-use/.test(error.message));
    // ik_llama.cpp has no unified parameters: a parameter error, as before.
    await assert.rejects(() => run('ik_llama.cpp'), (error) => error.code === 'invalid_params'
        && error.message === 'Invalid parameter (runner): ik_llama.cpp has no parameters for the unified profile');
    assert.equal(h.calls.download.length, 0);
    assert.deepEqual(h.controller.state.requests, {}, 'nothing recorded');
    // The overview's rows and the parameter preview say the same things.
    const rows = overview.models.find((model) => model.id === 'gpt-oss-20b').runners;
    assert.equal(rows['ik_llama.cpp'].params, null);
    assert.equal(rows['ik_llama.cpp'].admission.reason, 'Invalid parameter (runner): ik_llama.cpp has no parameters for the unified profile');
    assert.notEqual(rows.ollama.params, null);
    assert.equal(rows.ollama.admission.reason, 'Ollama is not available on a GPU that shares system memory (NVIDIA GB10) in this release.');
    assert.match(rows.lmstudio.admission.reason, /LM Studio is not enabled on this deployment/);
    const { preview } = await h.controller.overview({ preview: { modelId: 'gpt-oss-20b', runnerId: 'ik_llama.cpp' } });
    assert.deepEqual(preview, { modelId: 'gpt-oss-20b', runnerId: 'ik_llama.cpp',
        error: 'Invalid parameter (runner): ik_llama.cpp has no parameters for the unified profile', field: '(runner)' });
    // A dedicated GPU has none either.
    const dedicated = harness(t, { snap: { gpu: { available: true, memoryModel: 'dedicated', name: 'RTX', totalBytes: 6144 * MIB, usedBytes: 0, freeBytes: 6144 * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB }, disk: { freeBytes: 300 * GIB } } });
    const ded = await dedicated.controller.overview();
    assert.equal(ded.profile, 'dedicated');
    assert.equal(ded.runners.some((runner) => 'profileUnsupportedReason' in runner), false);
});

test('a unified run downloads, drops the file from the page cache, launches with the unified flags and the container-local JIT cache', async (t) => {
    const h = harness(t);
    await h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0001', params: { ctxSize: 65536, parallel: 2 } });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.deepEqual(h.calls.dropped, [h.weights]);
    const [process] = h.started;
    assert.ok(process.args.includes('--kv-unified') && process.args.includes('--cache-ram'));
    assert.equal(process.env.CUDA_CACHE_PATH, '/opt/runners/.cuda-cache');
    await h.controller.stop();
});

test('the unified guard samples in every phase and kills the process group below the floor', async (t) => {
    let available = 100 * GIB;
    const reads = [];
    const h = harness(t, { readMemory: () => { reads.push(Date.now()); return { totalBytes: TOTAL, availableBytes: available }; } });
    await h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const before = reads.length;
    await until(() => reads.length >= before + 3);
    const status = await h.controller.status();
    assert.equal(status.profile, 'unified');
    assert.equal(status.memoryGuard.floorBytes, UNIFIED.floorBytes);
    assert.ok(status.memoryGuard.samples >= 3);
    available = UNIFIED.floorBytes - GIB;
    await until(() => h.controller.state.deployment?.phase === 'error');
    assert.equal(h.started[0].stopped, 'SIGKILL');
    assert.match(h.controller.state.deployment.error, /stopped: host memory below the floor \(7\.0 GiB available, 8\.0 GiB required\)/);
});

// The Ploinky RAM budget of the guard tests (DS005): admission sees an ample budget so the run launches, then the
// guard's samples are constrained. The budget is latched finite when the guard starts, so it is finite from the first sample.
const SMALL_ID = 'qwen2.5-0.5b-instruct-q4_k_m';
const observed = (finiteMemoryBytes, headroomBytes) => ({ memoryReadState: 'known', finiteMemoryBytes, headroomBytes, reasonCode: null });
function budgetedHarness(t, { available = () => 100 * GIB, headroom, readPressure } = {}) {
    const admitted = observed(64 * GIB, 60 * GIB);
    return harness(t, {
        snap: () => attachMemoryBudget(unifiedSnapshot(), admitted),
        readMemory: () => ({ totalBytes: TOTAL, availableBytes: available() }),
        readBudget: () => headroom(),
        readPressure,
    });
}
const startSmall = async (h, requestId) => {
    await h.controller.run({ modelId: SMALL_ID, runnerId: 'llama.cpp', requestId });
    await until(() => h.controller.state.deployment?.phase === 'ready');
};
const sampled = async (h, count) => {
    const before = (await h.controller.status()).memoryGuard.samples;
    // A stop ends the wait too, so a runner the guard killed fails the caller's phase check with the guard's own reason.
    await until(async () => h.controller.state.deployment?.phase !== 'ready' || (await h.controller.status()).memoryGuard.samples >= before + count);
};

test('LL.guard-no-budget-floor: a budget\'s headroom below the floor does not stop a unified runner while the host has memory', async (t) => {
    let headroom = observed(12 * GIB, 12 * GIB);
    const h = budgetedHarness(t, { headroom: () => headroom });
    await startSmall(h, 'request-guard-budget');
    // The admissible run is up; now the container has 6 GiB of its 12 GiB budget left and the host 100 GiB free.
    headroom = observed(12 * GIB, 6 * GIB);
    await sampled(h, 4);
    assert.equal(h.controller.state.deployment.phase, 'ready', h.controller.state.deployment.error);
    assert.equal(h.started[0].stopped, undefined, 'not stopped');
    const { memoryGuard } = await h.controller.status();
    assert.equal(memoryGuard.minAvailableBytes, 100 * GIB, 'the floor is compared with the physical MemAvailable');
    await h.controller.stop();
});

test('LL.guard-physical-floor: below the floor on the physical pool stops a budgeted unified runner', async (t) => {
    let available = 100 * GIB;
    const h = budgetedHarness(t, { available: () => available, headroom: () => observed(64 * GIB, 60 * GIB) });
    await startSmall(h, 'request-guard-physical');
    available = 7 * GIB;
    await until(() => h.controller.state.deployment?.phase === 'error');
    assert.equal(h.started[0].stopped, 'SIGKILL');
    assert.match(h.controller.state.deployment.error, /stopped: host memory below the floor \(7\.0 GiB available, 8\.0 GiB required\)/);
});

test('the unified guard keeps its pressure rule on the physical pool and its unreadable-budget rule under a budget', async (t) => {
    // Heavy pressure with a small budget headroom (below twice the floor) but a free host: no stop.
    let headroom = observed(40 * GIB, 40 * GIB);
    const calm = budgetedHarness(t, { headroom: () => headroom, readPressure: () => 80 });
    await startSmall(calm, 'request-guard-psi');
    headroom = observed(40 * GIB, 6 * GIB);
    await sampled(calm, 4);
    assert.equal(calm.controller.state.deployment.phase, 'ready', calm.controller.state.deployment.error);
    await calm.controller.stop();
    // Two unreadable budget samples in a row still stop the runner (budget_unreadable).
    let reading = observed(40 * GIB, 40 * GIB);
    const blind = budgetedHarness(t, { headroom: () => reading });
    await startSmall(blind, 'request-guard-unreadable');
    reading = { memoryReadState: 'unknown', finiteMemoryBytes: 40 * GIB, headroomBytes: null, reasonCode: 'budget_unreadable' };
    await until(() => blind.controller.state.deployment?.phase === 'error');
    assert.equal(blind.started[0].stopped, 'SIGKILL');
    assert.match(blind.controller.state.deployment.error, /^stopped: budget_unreadable/);
});

test('the unified guard kills when /proc/meminfo cannot be read, and on pressure only with memory already low', async (t) => {
    const blind = harness(t, { readMemory: () => { throw new Error('EACCES'); } });
    await blind.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => blind.controller.state.deployment?.phase === 'error');
    assert.equal(blind.started[0].stopped, 'SIGKILL');
    assert.match(blind.controller.state.deployment.error, /host memory cannot be read/);

    // Pressure alone (page-cache reclaim with plenty available) never stops a runner.
    let available = 100 * GIB;
    const h = harness(t, { readPressure: () => 80, readMemory: () => ({ totalBytes: TOTAL, availableBytes: available }) });
    await h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0002' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.equal(h.controller.state.deployment.phase, 'ready');
    available = UNIFIED.floorBytes * 2 - GIB;
    await until(() => h.controller.state.deployment?.phase === 'error');
    assert.match(h.controller.state.deployment.error, /stopped: memory pressure 80\.0 % with 15\.0 GiB available/);
    assert.equal(h.started[0].stopped, 'SIGKILL');
});

test('a runner is marked as the OOM killer\'s first choice right after spawn; one that cannot be marked is killed', async (t) => {
    const log = createLogBuffer({});
    const runner = startRunnerProcess({ command: '/bin/sleep', args: ['5'], env: { PATH: '/usr/bin:/bin' }, log });
    t.after(() => runner.kill());
    assert.equal(fs.readFileSync(`/proc/${runner.pid}/oom_score_adj`, 'utf8').trim(), '1000');
    const killed = await runner.kill();
    assert.equal(killed.signal, 'SIGKILL');

    const refused = startRunnerProcess({
        command: '/bin/sleep', args: ['5'], env: { PATH: '/usr/bin:/bin' }, log,
        setOomScore: () => { throw Object.assign(new Error('denied'), { code: 'EACCES' }); },
    });
    const result = await refused.exited;
    assert.equal(result.signal, 'SIGKILL');
    assert.ok(log.all().some((entry) => /runner killed: its OOM score could not be set \(EACCES\)/.test(entry.line)));
});

test('the image contract is read as key=value lines; none means no platform limits', (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-contract-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, 'source.contract');
    fs.writeFileSync(file, 'architecture=arm64\nllama_cpp=b11159\ngpu_compute_capabilities=12.1\nuv=0.12.18\n');
    assert.deepEqual(readImageContract(file), { architecture: 'arm64', llama_cpp: 'b11159', gpu_compute_capabilities: '12.1', uv: '0.12.18' });
    assert.equal(readImageContract(path.join(dir, 'missing')), null);
});

test('the page cache is dropped with dd iflag=nocache count=0, in a minimal environment', () => {
    const calls = [];
    const ok = dropPageCache('/data/models/x.gguf', { spawnSyncImpl: (command, args, options) => { calls.push({ command, args, env: options.env }); return { status: 0 }; } });
    assert.equal(ok, true);
    assert.deepEqual(calls, [{ command: 'dd', args: ['if=/data/models/x.gguf', 'iflag=nocache', 'count=0', 'status=none'], env: { PATH: '/usr/bin:/bin', LANG: 'C.UTF-8' } }]);
    assert.equal(dropPageCache('/x', { spawnSyncImpl: () => ({ status: 1 }) }), false);
});

test('state this controller cannot read is reported, never misread: old registry entries, saved parameters, a recorded deployment', async (t) => {
    const v2 = { id: 'user-v2', sources: { gguf: { ...GGUF, file: 'v2.gguf' } }, recommended: { 'llama.cpp': { ctxSize: 8192 } } };
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-old-state-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const stateStore = createStateStore({ dataDir });
    stateStore.save({
        version: 1, registry: [v2], requests: {},
        params: { 'gpt-oss-20b|llama.cpp': { ctxSize: 8192, noMmap: true } },
        deployment: { id: 'd-1', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', phase: 'idle', params: { ctxSize: 8192, mlock: true } },
    });
    const dedicated = { gpu: { available: true, memoryModel: 'dedicated', name: 'RTX', totalBytes: 6144 * MIB, usedBytes: 0, freeBytes: 6000 * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB }, disk: { freeBytes: 300 * GIB } };
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin' }, seedCatalog: SEED, stateStore, snapshot: async () => structuredClone(dedicated),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
    });
    const overview = await controller.overview();
    assert.deepEqual(overview.unsupportedModels, [{ id: 'user-v2', reason: "recommended has an unknown profile 'llama.cpp'" }]);
    assert.ok(!overview.models.some((model) => model.id === 'user-v2'));
    // Saved parameters that no longer validate give way to the defaults, once, with a log line.
    const llama = overview.models.find((model) => model.id === 'gpt-oss-20b').runners['llama.cpp'];
    assert.equal(llama.params.ctxSize, 16384);
    const status = await controller.status();
    assert.equal(status.context, null, 'a recorded deployment with old parameters has no context, and no error');
    const lines = status.logs.map((entry) => entry.line);
    assert.ok(lines.some((line) => /model entry user-v2 is not supported by this catalog and is not offered/.test(line)));
    assert.equal(lines.filter((line) => /saved parameters for gpt-oss-20b on llama\.cpp no longer apply \(Invalid parameter noMmap: unknown parameter\)/.test(line)).length, 1);
    // The unsupported entry can be removed.
    assert.deepEqual(await controller.removeModel({ modelId: 'user-v2' }), {
        removed: 'user-v2', note: 'The entry was not supported by this catalog; downloaded weights, if any, stay on disk.',
    });
    assert.deepEqual((await controller.overview()).unsupportedModels, []);
});

test('a split GGUF downloads every shard, drops each from the page cache, and launches the first', async (t) => {
    const first = 'Q-00001-of-00003.gguf';
    const shards = [1, 2, 3].map((k) => ({ file: `Q-0000${k}-of-00003.gguf`, size: 10 * GIB, sha256: String(k).repeat(64) }));
    const split = validateModel({ ...structuredClone(MEASURED), id: 'split-moe',
        sources: { gguf: { type: 'huggingface', repo: 'org/split', file: first, revision: 'main', commit: 'd'.repeat(40), shards } } }, { seed: true });
    const h = harness(t, { seed: [split] });
    await h.controller.run({ modelId: 'split-moe', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.deepEqual(h.calls.download.map((artifact) => artifact.file), shards.map((shard) => shard.file));
    assert.equal(h.calls.dropped.length, 3);
    assert.equal(h.controller.state.deployment.artifact.size, 30 * GIB);
    await h.controller.stop();
});

test('the profile waits for a snapshot with a usable GPU; a transient failure does not lock it to dedicated', async (t) => {
    let first = true;
    const h = harness(t, { snap: () => {
        if (first) { first = false; return { gpu: { available: false, reason: 'nvidia-smi failed: timeout' }, memory: { totalBytes: TOTAL, availableBytes: 100 * GIB }, disk: { freeBytes: 400 * GIB } }; }
        return unifiedSnapshot();
    } });
    const early = await h.controller.status();
    assert.equal(early.profile, null);
    const overview = await h.controller.overview();
    assert.equal(overview.profile, 'unified');
    assert.ok(overview.models.some((model) => model.id === 'big-moe'));
});

test('the unified guard keeps watching a runner through a graceful Stop until it exits', async (t) => {
    let available = 100 * GIB;
    const h = harness(t, { readMemory: () => ({ totalBytes: TOTAL, availableBytes: available }) });
    await h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const [process] = h.started;
    // A runner that ignores SIGTERM: like the real one, its stop() settles only when the process exits.
    process.stop = () => process.exited;
    const stopping = h.controller.stop();
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(process.stopped, undefined, 'still running during the graceful stop');
    available = UNIFIED.floorBytes - GIB;
    await until(() => process.stopped === 'SIGKILL');
    await stopping;
});

test('a user model never reuses a seed id or a stored entry\'s id, whatever the profile; Delete weights finds any model', async (t) => {
    const dedicated = { gpu: { available: true, memoryModel: 'dedicated', name: 'RTX', totalBytes: 6144 * MIB, usedBytes: 0, freeBytes: 6000 * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB }, disk: { freeBytes: 300 * GIB } };
    const h = harness(t, { snap: dedicated });
    await h.controller.overview();
    // big-moe is a unified-only seed: not offered here, still reserved.
    await assert.rejects(() => h.controller.addModel({ id: 'big-moe', sources: { gguf: { type: 'huggingface', repo: 'x/y', file: 'z.gguf' } } }),
        { code: 'duplicate_model' });
    // Its weights can still be deleted on this host.
    const deleted = await h.controller.deleteWeights({ modelId: 'big-moe', runnerId: 'llama.cpp' });
    assert.equal(typeof deleted.freedBytes, 'number');
});

// ---------------------------------------------------------------------------
// The GPU-aware budget stop (DS005, B1): on the qualified GB10 (PCI 0x2E1210DE, driver 580.178.04) under a finite
// Ploinky RAM budget and no GPU share, the guard adds the runner's nvidia-smi memory to the memory the cgroup charges.
// Every double below replaces nvidia-smi, /proc, the cgroup and the clock sources: no hardware is touched. The
// existing GB10 stub (driver 580.159.03) stays the negative control.
// ---------------------------------------------------------------------------
const QUALIFIED = Object.freeze({ ...GB10, driverVersion: '580.178.04' });
const RUNNER_PID = 7000; // the first runner the harness starts
const budgetOf = (limit, current) => observeMemoryBudget({ maxText: String(limit), currentText: String(current) });
const statText = ({ file = 0, shmem = 0, active = 0, inactive = 0 } = {}) =>
    `anon 4096\nfile ${file}\nshmem ${shmem}\nactive_file ${active}\ninactive_file ${inactive}\n`;
const heldRows = (bytes, extra = []) => ({
    ok: true, rows: [{ pid: RUNNER_PID, name: '/opt/llama.cpp/llama-server', usedBytes: bytes }, ...extra], execMs: 3,
});

// An nvidia-smi double. `script(index)` returns the call's result, or undefined to leave the call pending until
// `answer(index, result)` or the sampler's stop (when it honours it). Every call is recorded with its signal.
function gpuStub(script, { honorsAbort = true } = {}) {
    const stub = { calls: [], inFlight: 0, settle: [] };
    stub.read = ({ signal }) => {
        const index = stub.calls.length;
        stub.calls.push({ signal });
        stub.inFlight += 1;
        return new Promise((resolve, reject) => {
            let done = false;
            const finish = (result, failure) => {
                if (done) return;
                done = true;
                stub.inFlight -= 1;
                signal.removeEventListener('abort', onAbort);
                if (failure) reject(failure); else resolve(result);
            };
            const onAbort = () => { if (honorsAbort) finish({ ok: false, reason: 'aborted', execMs: 1 }); };
            signal.addEventListener('abort', onAbort, { once: true });
            stub.settle[index] = finish;
            try {
                const result = script(index);
                if (result !== undefined) finish(result);
            } catch (error) {
                finish(null, error);
            }
        });
    };
    stub.answer = (index, result) => stub.settle[index](result);
    return stub;
}

// A budgeted unified controller whose hardware reads are all doubles. `state` is what the doubles answer now.
function gpuHarness(t, { limit = 12 * GIB, current = 2 * GIB, held = 0, stat = statText(), groups, script = (_, st) => heldRows(st.held),
    honorsAbort = true, gpuSampleMs = 10, gpu = QUALIFIED, budgeted = true, readMemory, readPressure, controllerOptions = {}, exitDelayMs = 0 } = {}) {
    const state = { limit, current, held, stat, groups: groups ?? new Map([[RUNNER_PID, RUNNER_PID]]) };
    const stub = gpuStub((index) => script(index, state), { honorsAbort });
    const h = harness(t, {
        // Admission sees an ample budget so the run launches; the guard's own reads are constrained.
        snap: () => {
            const snap = unifiedSnapshot({ gpu });
            return budgeted ? attachMemoryBudget(snap, observed(64 * GIB, 60 * GIB)) : snap;
        },
        readBudget: () => (budgeted ? budgetOf(state.limit, state.current) : observeMemoryBudget({ maxText: 'max', currentText: '0' })),
        readMemory,
        readPressure,
        exitDelayMs,
        controllerOptions: {
            readGpuProcesses: stub.read,
            readProcGroup: (pid) => state.groups.get(pid) ?? null,
            readLeafMemoryStat: () => {
                if (state.stat === null) throw Object.assign(new Error('EACCES'), { code: 'EACCES' });
                return state.stat;
            },
            gpuSampleMs,
            ...controllerOptions,
        },
    });
    return { ...h, state, stub };
}

const guardOf = async (h) => (await h.controller.status()).memoryGuard;
const OVER_MESSAGE = /^stopped: over the Ploinky RAM budget \(/;
const stoppedWith = async (h) => {
    await until(() => h.controller.state.deployment?.phase === 'error');
    assert.equal(h.started[0].stopped, 'SIGKILL', 'the process group is killed');
    return h.controller.state.deployment.error;
};
// The run stays up through `ticks` guard ticks; with `samples`, also after that many completed GPU samples.
async function staysUp(h, { ticks = 8, samples = 0 } = {}) {
    await sampled(h, ticks);
    if (samples) await until(async () => (await guardOf(h)).gpuSamples >= samples);
    await sampled(h, 3);
    assert.equal(h.controller.state.deployment.phase, 'ready', h.controller.state.deployment.error);
    assert.equal(h.started[0].stopped, undefined, 'not stopped');
}

test('A-LL-13: a qualified GB10 stops when the charged memory plus the runner\'s GPU memory exceeds the budget', async (t) => {
    const h = gpuHarness(t, { limit: 12 * GIB, current: 2 * GIB, held: 11 * GIB, readMemory: () => ({ totalBytes: TOTAL, availableBytes: 100 * GIB }) });
    await startSmall(h, 'request-gpu-13');
    const message = await stoppedWith(h);
    assert.match(message, OVER_MESSAGE);
    assert.equal(message, 'stopped: over the Ploinky RAM budget (2.0 GiB charged + 11.0 GiB GPU, budget 12.0 GiB)');
    // One GiB over is not past the fast path: a second, newer sample confirmed it.
    assert.ok(h.stub.calls.length >= 2, 'a newer sample confirms an overrun of 1 GiB or less');
    const guard = await guardOf(h);
    assert.equal(guard.gpuBudgetSignal, 'observed');
    assert.equal(guard.gpuHeldBytes, 11 * GIB);
    assert.equal(guard.gpuChargedBytes, 2 * GIB);
    assert.ok(guard.gpuSamples >= 2);
    assert.equal(typeof guard.gpuSampleAt, 'number');
    assert.ok(Number.isFinite(guard.gpuSampleExecMsMedian));
});

test('A-LL-13b: the budget is a strict limit; a newer sample confirms a small overrun and a reading under budget resets it', async (t) => {
    // Exactly at the budget (2 + 10 = 12 GiB): no stop. One MiB more: a stop.
    const h = gpuHarness(t, { held: 10 * GIB });
    await startSmall(h, 'request-gpu-13b');
    await staysUp(h, { ticks: 8, samples: 3 });
    assert.equal((await guardOf(h)).gpuBudgetSignal, 'observed');
    h.state.held = 10 * GIB + MIB;
    assert.match(await stoppedWith(h), OVER_MESSAGE);

    // Over by 1 MiB on S1, under on S2: no stop. Over on S3, under on S4: still none (the streak was reset). Then
    // over on S5 and still over on the newer S6: a stop. Each sample is answered after the guard has ticked on the last.
    const calls = [10 * GIB + MIB, 10 * GIB, 10 * GIB + MIB, 10 * GIB, 10 * GIB + MIB, 10 * GIB + MIB];
    const m = gpuHarness(t, { gpuSampleMs: 100, script: () => undefined });
    await startSmall(m, 'request-gpu-13b-reset');
    for (let index = 0; index < 4; index += 1) {
        await until(() => m.stub.calls.length > index);
        m.stub.answer(index, heldRows(calls[index]));
        await sampled(m, 3);
        assert.equal(m.controller.state.deployment.phase, 'ready', `sample ${index + 1}: ${m.controller.state.deployment.error}`);
    }
    await until(() => m.stub.calls.length > 4);
    m.stub.answer(4, heldRows(calls[4]));
    await sampled(m, 3);
    assert.equal(m.controller.state.deployment.phase, 'ready', 'one sample over after a reset is not confirmed');
    await until(() => m.stub.calls.length > 5);
    m.stub.answer(5, heldRows(calls[5]));
    assert.match(await stoppedWith(m), OVER_MESSAGE);
    assert.equal(m.stub.calls.length, 6);

    // More than 1 GiB over on a fresh sample stops at once, before any second sample.
    const fast = gpuHarness(t, { gpuSampleMs: 100, script: (index) => (index === 0 ? heldRows(12 * GIB) : undefined) });
    await startSmall(fast, 'request-gpu-13b-fast');
    assert.match(await stoppedWith(fast), /^stopped: over the Ploinky RAM budget \(2\.0 GiB charged \+ 12\.0 GiB GPU, budget 12\.0 GiB\)$/);
    assert.equal(fast.stub.calls.length, 1);
});

test('c1: a pending confirmation survives ticks without a valid reading', async (t) => {
    // Over on S1; S2 fails (unavailable); still over on S3: a stop, although no tick saw S2.
    const h = gpuHarness(t, { gpuSampleMs: 100, script: () => undefined });
    await startSmall(h, 'request-gpu-c1');
    await until(() => h.stub.calls.length > 0);
    h.stub.answer(0, heldRows(10 * GIB + MIB));
    await sampled(h, 3);
    await until(() => h.stub.calls.length > 1);
    h.stub.answer(1, { ok: false, reason: 'nvidia-smi failed: exit 1', execMs: 2 });
    await sampled(h, 3);
    assert.equal((await guardOf(h)).gpuBudgetSignal, 'unavailable');
    assert.equal(h.controller.state.deployment.phase, 'ready');
    await until(() => h.stub.calls.length > 2);
    h.stub.answer(2, heldRows(10 * GIB + MIB));
    assert.match(await stoppedWith(h), OVER_MESSAGE);
    assert.equal(h.stub.calls.length, 3);
});

test('A-LL-13c: the page-cache term is active_file + inactive_file of memory.stat; file and shmem are ignored', async (t) => {
    // memory.stat lines that would mislead a file - shmem rule: file 5 GiB, shmem 1 GiB.
    const h = gpuHarness(t, { current: 6 * GIB, held: 9 * GIB,
        stat: statText({ file: 5 * GIB, shmem: GIB, active: 3 * GIB, inactive: GIB }) });
    await startSmall(h, 'request-gpu-13c');
    await staysUp(h, { ticks: 8, samples: 3 });
    assert.equal((await guardOf(h)).gpuChargedBytes, 2 * GIB, 'charged = 6 - (3 + 1) GiB; 2 + 9 is within 12');
    // Page cache of 2 GiB (1.5 active + 0.5 inactive) against file 5 GiB and shmem 1 GiB: charged 4 GiB.
    h.state.held = 7 * GIB;
    h.state.stat = statText({ file: 5 * GIB, shmem: GIB, active: 1.5 * GIB, inactive: 0.5 * GIB });
    await until(async () => (await guardOf(h)).gpuChargedBytes === 4 * GIB);
    await staysUp(h, { ticks: 4 });
    // No reclaimable cache left: 6 GiB charged + 9 GiB of GPU memory is past the budget (pinned host memory stays counted).
    h.state.held = 9 * GIB;
    h.state.stat = statText({ file: GIB, shmem: GIB, active: 0, inactive: 0 });
    const message = await stoppedWith(h);
    assert.equal(message, 'stopped: over the Ploinky RAM budget (6.0 GiB charged + 9.0 GiB GPU, budget 12.0 GiB)');
});

test('A-LL-14: a qualified run within the budget is not stopped (regression guard)', async (t) => {
    const h = gpuHarness(t, { held: 9 * GIB });
    await startSmall(h, 'request-gpu-14');
    await staysUp(h, { ticks: 8 });
});

test('A-LL-14b: within the budget the sampler really ran, and the guard reports an observed signal', async (t) => {
    const h = gpuHarness(t, { held: 9 * GIB });
    await startSmall(h, 'request-gpu-14b');
    await staysUp(h, { ticks: 8, samples: 3 });
    await until(async () => (await guardOf(h)).gpuBudgetSignal === 'observed');
    const guard = await guardOf(h);
    assert.equal(guard.gpuHeldBytes, 9 * GIB);
    assert.equal(guard.gpuChargedBytes, 2 * GIB);
    assert.ok(guard.gpuSampleAgeMs >= 0);
});

test('A-LL-15: rows whose process group is not the runner\'s are unmapped: unavailable, no stop', async (t) => {
    // The only row is in a foreign group (and a second PID no longer exists): 50 GiB of it must not count.
    const h = gpuHarness(t, { groups: new Map([[RUNNER_PID, 1234]]), script: () => heldRows(50 * GIB, [{ pid: 9001, name: 'gone', usedBytes: GIB }]) });
    await startSmall(h, 'request-gpu-15');
    await staysUp(h, { ticks: 8, samples: 3 });
    const guard = await guardOf(h);
    assert.equal(guard.gpuBudgetSignal, 'unavailable');
    assert.equal(guard.gpuHeldBytes, null);
});

test('A-LL-15b: a hung nvidia-smi is bounded, single flight, and ended with the guard', async (t) => {
    // Never settles, even when stopped: after two intervals there is no usable reading, and still one call.
    const hung = gpuHarness(t, { script: () => undefined, honorsAbort: false });
    await startSmall(hung, 'request-gpu-15b');
    await sampled(hung, 8);
    await new Promise((resolve) => setTimeout(resolve, 60));
    assert.equal((await guardOf(hung)).gpuBudgetSignal, 'unavailable');
    assert.equal(hung.stub.calls.length, 1, 'exactly one call in flight over five intervals');
    assert.equal(hung.stub.calls[0].signal.aborted, true, 'the per-exec bound ended it');
    assert.equal(hung.controller.state.deployment.phase, 'ready');
    await hung.controller.stop();
    assert.equal(hung.stub.calls[0].signal.aborted, true);
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.equal(hung.stub.calls.length, 1, 'no further call after cancel');

    // Settles only when aborted: the sampler's per-exec timer aborts it, the next sample starts, and a settling one restores 'observed'.
    const abortOnly = gpuHarness(t, { script: (index, st) => (index === 0 ? undefined : heldRows(st.held)), held: GIB });
    await startSmall(abortOnly, 'request-gpu-15b-abort');
    await until(() => abortOnly.stub.calls.length >= 2);
    assert.equal(abortOnly.stub.calls[0].signal.aborted, true);
    await until(async () => (await guardOf(abortOnly)).gpuBudgetSignal === 'observed');
    assert.equal((await guardOf(abortOnly)).gpuHeldBytes, GIB);
    assert.equal(abortOnly.stub.inFlight <= 1, true);
    await abortOnly.controller.stop();
});

test('A-LL-15 stale: a sample older than two intervals is not used, even when a later overrun would have stopped the run', async (t) => {
    // S1 is under budget; every later call hangs (and ignores the stop), so S1 only ages.
    const h = gpuHarness(t, { gpuSampleMs: 30, held: 5 * GIB, honorsAbort: false, script: (index, st) => (index === 0 ? heldRows(st.held) : undefined) });
    await startSmall(h, 'request-gpu-stale');
    await until(async () => (await guardOf(h)).gpuBudgetSignal === 'observed');
    await until(async () => (await guardOf(h)).gpuBudgetSignal === 'unavailable');
    assert.ok((await guardOf(h)).gpuSampleAgeMs > 60, 'unavailable because the sample is older than 2 x 30 ms');
    // 11 + 5 GiB would be 4 GiB past the budget of 12 if the old sample were still used.
    h.state.current = 11 * GIB;
    await staysUp(h, { ticks: 8 });
    assert.equal((await guardOf(h)).gpuBudgetSignal, 'unavailable');
    assert.equal(h.stub.calls.length, 2, 'one hung call after the first');
});

test('A-LL-15c: a failing, throwing or garbage nvidia-smi is unavailable, never a stop and never a throw', async (t) => {
    const answers = [
        { ok: false, reason: 'nvidia-smi failed: exit 1', execMs: 2 },
        { ok: true, rows: [], execMs: 2 }, // what readGpuProcesses returns for garbage output
        { ok: false, reason: 'aborted', execMs: 1 },
    ];
    const h = gpuHarness(t, { script: (index) => {
        if (index % 4 === 3) throw new Error('spawn failed');
        return answers[index % 4] ?? answers[0];
    } });
    await startSmall(h, 'request-gpu-15c');
    await staysUp(h, { ticks: 8, samples: 5 });
    assert.equal((await guardOf(h)).gpuBudgetSignal, 'unavailable');
    assert.equal((await guardOf(h)).gpuHeldBytes, null);
});

test('A-LL-15d: the runner\'s GPU memory is the sum of the rows in its process group', async (t) => {
    const groups = new Map([[RUNNER_PID, RUNNER_PID], [7001, RUNNER_PID], [9000, 9000]]);
    const rows = [{ pid: 7001, name: 'child', usedBytes: 3 * GIB }, { pid: 9000, name: 'foreign', usedBytes: 50 * GIB },
        { pid: 9001, name: 'gone', usedBytes: 40 * GIB }];
    const h = gpuHarness(t, { groups, script: () => heldRows(4 * GIB, rows) });
    await startSmall(h, 'request-gpu-15d');
    await staysUp(h, { ticks: 8, samples: 3 });
    await until(async () => (await guardOf(h)).gpuHeldBytes === 7 * GIB);
});

test('A-LL-15e: an unreadable memory.stat at the decision is unavailable that tick; reading it again decides', async (t) => {
    const h = gpuHarness(t, { stat: null, held: 30 * GIB });
    await startSmall(h, 'request-gpu-15e');
    await staysUp(h, { ticks: 8, samples: 3 });
    assert.equal((await guardOf(h)).gpuBudgetSignal, 'unavailable');
    assert.equal((await guardOf(h)).gpuChargedBytes, null);
    h.state.stat = statText();
    assert.match(await stoppedWith(h), OVER_MESSAGE);
    // memory.stat without the two file terms is unreadable too.
    const partial = gpuHarness(t, { stat: 'anon 1\nfile 2\n', held: 30 * GIB });
    await startSmall(partial, 'request-gpu-15e-partial');
    await staysUp(partial, { ticks: 8, samples: 3 });
    assert.equal((await guardOf(partial)).gpuBudgetSignal, 'unavailable');
});

test('A-LL-15f: a budget breach cancels the sampler at once, not when the runner has exited', async (t) => {
    // The killed runner takes 800 ms to exit, so only the breach itself can have cancelled the sampler. Sample 0 is
    // under budget; sample 1 is in flight (a 200 ms bound) when an overrun appears.
    const h = gpuHarness(t, { gpuSampleMs: 200, held: 5 * GIB, exitDelayMs: 800, script: (index, st) => (index === 0 ? heldRows(st.held) : undefined) });
    await startSmall(h, 'request-gpu-15f');
    await until(() => h.stub.calls.length === 2);
    const inFlight = h.stub.calls[1].signal;
    assert.equal(inFlight.aborted, false);
    h.state.current = 11 * GIB; // 11 + 5 GiB against 12: past the fast path
    await until(() => h.started[0].stopped === 'SIGKILL');
    assert.equal(h.controller.state.deployment.phase, 'ready', 'the runner has not exited yet');
    assert.equal(inFlight.aborted, true, 'the breach aborts the in-flight call before the runner exits');
    await new Promise((resolve) => setTimeout(resolve, 600));
    assert.equal(h.stub.calls.length, 2, 'no readGpuProcesses call between the breach and the exit');
    assert.equal(h.stub.inFlight, 0);
    assert.match(await stoppedWith(h), OVER_MESSAGE);
    assert.equal(h.stub.calls.length, 2);
});

test('a late result after cancel is dropped: the guard\'s GPU fields stay as they were', async (t) => {
    // The stub ignores the stop, so call 1 answers after the Stop.
    const h = gpuHarness(t, { held: GIB, honorsAbort: false, script: (index, st) => (index === 0 ? heldRows(st.held) : undefined) });
    await startSmall(h, 'request-gpu-late');
    await until(() => h.stub.calls.length === 2);
    await until(async () => (await guardOf(h)).gpuSamples === 1);
    await h.controller.stop();
    const before = await guardOf(h);
    assert.equal(before.gpuSamples, 1);
    assert.equal(before.gpuHeldBytes, GIB);
    assert.equal(typeof before.gpuSampleAt, 'number');
    h.stub.answer(1, heldRows(5 * GIB));
    await new Promise((resolve) => setTimeout(resolve, 60));
    const after = await guardOf(h);
    assert.equal(after.gpuSamples, before.gpuSamples);
    assert.equal(after.gpuHeldBytes, before.gpuHeldBytes);
    assert.equal(after.gpuSampleAt, before.gpuSampleAt);
    assert.equal(h.stub.calls.length, 2, 'and no further sample is scheduled');
});

test('A-LL-15g and 15h: the floor and the pressure checks keep priority over the budget stop in the same tick', async (t) => {
    let available = 100 * GIB;
    let pressure = 0;
    const set = { readMemory: () => ({ totalBytes: TOTAL, availableBytes: available }), readPressure: () => pressure };
    for (const [name, flip, message] of [
        ['floor', () => { available = 7 * GIB; }, /^stopped: host memory below the floor \(7\.0 GiB available, 8\.0 GiB required\)$/],
        ['pressure', () => { available = 15 * GIB; pressure = 80; }, /^stopped: memory pressure 80\.0 % with 15\.0 GiB available$/],
    ]) {
        available = 100 * GIB;
        pressure = 0;
        const h = gpuHarness(t, { held: 5 * GIB, ...set });
        await startSmall(h, `request-gpu-15-${name}`);
        await staysUp(h, { ticks: 4, samples: 3 });
        await until(async () => (await guardOf(h)).gpuBudgetSignal === 'observed');
        // One synchronous step: the next tick sees both an overrun (16 GiB against 12) and the lower host reading.
        h.state.current = 11 * GIB;
        flip();
        const error = await stoppedWith(h);
        assert.match(error, message, name);
        assert.doesNotMatch(error, OVER_MESSAGE, name);
    }
});

test('A-LL-16b: the existing GB10 stub (driver 580.159.03) is unqualified: no sampler, no stop', async (t) => {
    const h = gpuHarness(t, { gpu: GB10, current: 10 * GIB, held: 8 * GIB });
    await startSmall(h, 'request-gpu-16b');
    await staysUp(h, { ticks: 8 });
    const guard = await guardOf(h);
    assert.equal(guard.gpuBudgetSignal, 'unqualified');
    assert.equal(guard.gpuSamples, 0);
    assert.equal(h.stub.calls.length, 0);
});

test('A-LL-16c: a unified GPU with another PCI id is unqualified, whatever its driver', async (t) => {
    const other = { ...QUALIFIED, device: { ...QUALIFIED.device, pciDeviceId: '0x2B0010DE' } };
    const h = gpuHarness(t, { gpu: other, current: 10 * GIB, held: 8 * GIB });
    await startSmall(h, 'request-gpu-16c');
    await staysUp(h, { ticks: 8 });
    assert.equal((await guardOf(h)).gpuBudgetSignal, 'unqualified');
    assert.equal(h.stub.calls.length, 0);
});

test('A-LL-16d: a qualified GB10 without a finite budget has no GPU-aware stop: not-applicable, no call', async (t) => {
    const h = gpuHarness(t, { budgeted: false, current: 10 * GIB, held: 80 * GIB });
    await startSmall(h, 'request-gpu-16d');
    await staysUp(h, { ticks: 8 });
    assert.equal((await guardOf(h)).gpuBudgetSignal, 'not-applicable');
    assert.equal(h.stub.calls.length, 0);
});

// The qualifying decision itself, with the facts spelled out; the controller vector for a known share (admitted on the
// qualified tuple) is A-LL-16e in ploinky-budget.test.mjs.
test('A-LL-16e and A-LL-16i: a known share, or no admitted GPU facts at all, never samples', async () => {
    const { gpuGuardSignal } = await import('../src/controller/gpuBudgetGuard.mjs');
    const qualified = { gpu: { driverVersion: '580.178.04', device: { pciDeviceId: '0x2E1210DE' } }, shareKnown: false };
    assert.equal(gpuGuardSignal(qualified, true), 'unavailable', 'the sampler runs, with no valid reading yet');
    assert.equal(gpuGuardSignal({ ...qualified, shareKnown: true }, true), 'not-applicable');
    assert.equal(gpuGuardSignal(qualified, false), 'not-applicable');
    assert.equal(gpuGuardSignal(undefined, true), 'unqualified', 'a missing entry is unqualified');
    assert.equal(gpuGuardSignal(null, false), 'unqualified');
    assert.equal(gpuGuardSignal({ ...qualified, gpu: { ...qualified.gpu, driverVersion: '580.159.03' } }, true), 'unqualified');
});

test('A-LL-16g: nothing about the sampler is persisted, and the published shapes differ only by the guardStats fields', async (t) => {
    const shapes = [];
    for (const gpu of [GB10, QUALIFIED]) {
        const h = gpuHarness(t, { gpu, held: 5 * GIB });
        await startSmall(h, 'request-gpu-16g');
        await staysUp(h, { ticks: 4, samples: gpu === QUALIFIED ? 3 : 0 });
        const status = await h.controller.status();
        const overview = await h.controller.overview();
        const stateText = fs.readFileSync(path.join(h.dataDir, 'state', 'controller.json'), 'utf8');
        assert.match(stateText, /request-gpu-16g/, 'the scan reads the real state file');
        assert.doesNotMatch(stateText, /580\.178\.04|shareKnown|gpuBudgetSignal|gpuHeldBytes|gpuSampleAt|pciDeviceId/);
        assert.doesNotMatch(JSON.stringify(overview), /shareKnown|gpuBudgetSignal|gpuHeldBytes/);
        // The persisted deployment, byte for byte, apart from its random id and its clock readings.
        const persisted = JSON.parse(stateText.replace(/"\d{4}-\d\d-\d\dT[\d:.]+Z"/g, '"<time>"')).deployment;
        persisted.id = '<id>';
        shapes.push({
            persisted,
            status: Object.keys(status).sort(),
            guard: Object.keys(status.memoryGuard),
            overview: Object.keys(overview).sort(),
        });
        await h.controller.stop();
    }
    const [base, sampled_] = shapes;
    assert.ok(Object.keys(base.persisted).length > 5, 'a persisted deployment was compared');
    assert.deepEqual(sampled_.persisted, base.persisted, 'the persisted deployment is the same with and without the sampler');
    assert.deepEqual(sampled_.status, base.status);
    assert.deepEqual(sampled_.overview, base.overview);
    assert.deepEqual(sampled_.guard, base.guard, 'the unified guardStats fields are the same on every unified host');
    assert.deepEqual(base.guard.slice(0, 4), ['floorBytes', 'minAvailableBytes', 'maxPressureAvg10', 'samples']);
    assert.deepEqual(base.guard.slice(4), ['gpuBudgetSignal', 'gpuHeldBytes', 'gpuChargedBytes', 'gpuSampleAt', 'gpuSampleAgeMs',
        'gpuSampleExecMsMedian', 'gpuSamples']);
});

// The other pool profiles keep their guardStats: the GPU fields exist only on the unified profile.
test('the cpu profile\'s guardStats are unchanged: no GPU fields, no sampler', async (t) => {
    const h = gpuHarness(t, { gpu: GB10, controllerOptions: { profile: 'cpu' } });
    await h.controller.run({ modelId: SMALL_ID, runnerId: 'llama.cpp', requestId: 'request-gpu-cpu' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    await until(async () => (await guardOf(h))?.samples >= 3);
    assert.equal((await h.controller.status()).profile, 'cpu');
    assert.deepEqual(Object.keys(await guardOf(h)), ['floorBytes', 'minAvailableBytes', 'maxPressureAvg10', 'samples']);
    assert.equal(h.stub.calls.length, 0);
    await h.controller.stop();
});

// Counts the timers set with one delay, so a test can say how many sampler timers are alive: the sampler's interval and its
// per-exec bound both use `delay`, and no other timer of these tests does.
function trackTimers(t, delay) {
    const live = new Set();
    const created = [];
    const realSet = globalThis.setTimeout;
    const realClear = globalThis.clearTimeout;
    globalThis.setTimeout = (fn, ms, ...args) => {
        if (ms !== delay) return realSet(fn, ms, ...args);
        const timer = realSet(() => { live.delete(timer); fn(...args); }, ms);
        live.add(timer);
        created.push(timer);
        return timer;
    };
    globalThis.clearTimeout = (timer) => { live.delete(timer); return realClear(timer); };
    t.after(() => { globalThis.setTimeout = realSet; globalThis.clearTimeout = realClear; });
    return { live, created, wait: (ms) => new Promise((resolve) => realSet(resolve, ms)) };
}

// Cancel leaves nothing behind: no sampler timer, no exec in flight, and a finished sample leaves no timer either.
test('A-LL-15 timers: after a sample settles and after each cancel there are 0 sampler timers and 0 execs in flight', async (t) => {
    const DELAY = 13;
    const { live, created, wait } = trackTimers(t, DELAY);
    const h = gpuHarness(t, { gpuSampleMs: DELAY, held: GIB });
    for (const round of [1, 2]) {
        await startSmall(h, `request-gpu-timers-${round}`);
        const before = h.stub.calls.length;
        await until(() => h.stub.calls.length >= before + 3);
        // Between samples only the next-sample timer is alive; the settled exec's timer is gone.
        assert.ok(live.size <= 1, `${live.size} sampler timers alive between samples`);
        assert.equal(h.stub.calls.at(-2).signal.aborted, false, 'a sample that settled is not aborted by its own bound');
        await h.controller.stop();
        assert.equal(live.size, 0, `round ${round}: sampler timers after cancel`);
        assert.equal(h.stub.inFlight, 0, `round ${round}: execs in flight after cancel`);
        const calls = h.stub.calls.length;
        await wait(4 * DELAY);
        assert.equal(h.stub.calls.length, calls, `round ${round}: no call after cancel`);
    }
    assert.ok(created.length >= 6);
    assert.ok(created.every((timer) => timer.hasRef() === false), 'every sampler timer is unref\'d');

    // Cancel while an exec is in flight: its bound timer goes with it, and a late answer breaches nothing.
    const pending = gpuHarness(t, { gpuSampleMs: DELAY, script: () => undefined });
    await startSmall(pending, 'request-gpu-timers-pending');
    await until(() => pending.stub.calls.length >= 1);
    assert.equal(pending.stub.inFlight, 1);
    await pending.controller.stop();
    assert.equal(live.size, 0);
    assert.equal(pending.stub.inFlight, 0);
    assert.equal(pending.stub.calls.length, 1);
    assert.notEqual(pending.controller.state.deployment?.phase, 'error');
});

test('A-LL-15 replace: the old runner stays sampled through its graceful stop; the new guard gets its own stats, reads its own group, and leaves nothing behind', async (t) => {
    const DELAY = 13;
    const { live, wait } = trackTimers(t, DELAY);
    // Each runner has its own rows: 1 GiB in the old group (7000), 2 GiB in the new one (7001).
    const groups = new Map([[7000, 7000], [7001, 7001]]);
    const h = gpuHarness(t, { gpuSampleMs: DELAY, groups, held: GIB,
        script: (_, st) => heldRows(st.held, [{ pid: 7001, name: 'new', usedBytes: 2 * GIB }]) });
    await startSmall(h, 'request-gpu-replace-1');
    await until(async () => (await guardOf(h)).gpuHeldBytes === GIB);
    const [old] = h.started;
    old.stop = () => old.exited; // like the real one: a graceful stop settles when the process exits
    const replacing = h.controller.run({ modelId: SMALL_ID, runnerId: 'llama.cpp', requestId: 'request-gpu-replace-2', replace: true });
    const calls = h.stub.calls.length;
    await until(() => h.stub.calls.length >= calls + 3);
    assert.equal(h.started.length, 1, 'the new runner starts only after the old one has exited');
    assert.equal(old.stopped, undefined);
    await old.kill();
    await replacing;
    await until(() => h.controller.state.deployment?.requestId === 'request-gpu-replace-2' && h.controller.state.deployment.phase === 'ready');
    assert.equal(h.started.length, 2);
    // The new guard has its own stats (counted from zero) and matches only its own process group.
    await until(async () => (await guardOf(h)).gpuHeldBytes === 2 * GIB);
    assert.ok(live.size <= 1, `${live.size} sampler timers alive with one guard`);
    await h.controller.stop();
    assert.equal(live.size, 0);
    assert.equal(h.stub.inFlight, 0);
    const after = h.stub.calls.length;
    await wait(4 * DELAY);
    assert.equal(h.stub.calls.length, after, 'no call after both guards are cancelled');
});
