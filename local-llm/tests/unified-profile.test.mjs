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
// A model measured on unified memory: 128k x 4 slots, and 256k x 1 with MTP.
const MEASURED = validateModel({
    id: 'big-moe', displayName: 'Big MoE', architecture: 'moe', contextLength: 131072, sources: { gguf: GGUF },
    profiles: ['unified'],
    recommended: { unified: { 'llama.cpp': { ctxSize: 32768 } } },
    unified: { envelope: [
        { runner: 'llama.cpp', maxCtx: 131072, maxParallel: 4, mtp: false, bufferBytes: 66 * GIB, transientBytes: 13 * GIB },
        { runner: 'llama.cpp', maxCtx: 262144, maxParallel: 1, mtp: true, bufferBytes: 70 * GIB, transientBytes: 2 * GIB },
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
    assert.deepEqual(inside.estimate.envelope, { maxCtx: 131072, maxParallel: 4, mtp: false });
    assert.equal(inside.estimate.cacheRamBytes, 8192 * MIB);
    // The corner's figures cover every configuration inside the rectangle.
    assert.equal(decide({ ctxSize: 8192, parallel: 1 }).estimate.unifiedBytes, need);
    // MTP only where it was measured.
    assert.equal(decide({ ctxSize: 65536, parallel: 1, mtp: true }).estimate.envelope.maxCtx, 262144);
    const outside = decide({ ctxSize: 131072, parallel: 8 });
    assert.equal(outside.status, 'incompatible');
    assert.match(outside.reason, /measured on unified memory with llama\.cpp only up to context 131072 x 4 slots; context 262144 x 1 slot \(MTP allowed\); context 131072 x 8 slots is outside it/);
    // Less available than need plus the floor: busy now, naming the numbers.
    const busy = decide({ ctxSize: 131072, parallel: 4 }, unifiedSnapshot({ available: need + UNIFIED.floorBytes - GIB }));
    assert.equal(busy.status, 'insufficient-now');
    assert.match(busy.reason, /Other processes on this machine hold the rest/);
    // More than the pool less the host reserve: never.
    const huge = validateModel({ ...structuredClone(MEASURED), unified: { envelope: [
        { runner: 'llama.cpp', maxCtx: 131072, maxParallel: 1, mtp: false, bufferBytes: 100 * GIB, transientBytes: 0 },
    ] } }, { seed: true });
    const never = admit({ runner, model: huge, source: GGUF, params: runner.normalizeParams({}, { model: huge, profile: 'unified' }),
        snapshot: unifiedSnapshot(), profile: 'unified' });
    assert.equal(never.status, 'incompatible');
    assert.match(never.reason, /must keep 16\.0 GiB for the host/);
    // A model never measured here is refused, whatever its size.
    const [gpt] = SEED;
    const unmeasured = admit({ runner, model: gpt, source: gpt.sources.gguf, params: runner.normalizeParams({}, { model: gpt, profile: 'unified' }),
        snapshot: unifiedSnapshot(), profile: 'unified' });
    assert.equal(unmeasured.status, 'incompatible');
    assert.match(unmeasured.reason, /has not been measured on unified memory/);
    // No readable pool: refused.
    const blind = decide({}, { ...unifiedSnapshot(), memory: { totalBytes: null, availableBytes: null } });
    assert.equal(blind.status, 'incompatible');
    assert.match(blind.reason, /cannot be read/);
    assert.equal(envelopeFor(MEASURED, 'ollama', { ctxSize: 1, parallel: 1 }), null);
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
    readMemory, readPressure = () => 0, unifiedGuardMs = 5, downloads = true, dropCache } = {}) {
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
        readMemory: readMemory || (() => ({ totalBytes: TOTAL, availableBytes: 100 * GIB })),
        readPressure,
        unifiedGuardMs,
        dropCache: dropCache || ((file) => { calls.dropped.push(file); return true; }),
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
    }
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'ik_llama.cpp', requestId: 'request-0001' }),
        (error) => error.code === 'runner_unavailable');
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'vllm', requestId: 'request-0002' }),
        (error) => error.code === 'runner_unavailable');
    assert.equal(h.calls.download.length, 0);
    const gpt = overview.models.find((model) => model.id === 'gpt-oss-20b');
    assert.deepEqual(gpt.weights.gguf.runners, ['llama.cpp']);
    assert.match(gpt.runners.ollama.admission.reason, /not available on this platform/);
});

test('a GPU the image\'s runners were not built for is refused before any download', async (t) => {
    const thor = { ...GB10, name: 'NVIDIA Thor', device: { ...GB10.device, pciDeviceId: '0x2B0010DE', computeCapability: '11.0', addressingMode: 'HMM' } };
    const h = harness(t, { snap: unifiedSnapshot({ gpu: thor }), imageContract: { gpu_compute_capabilities: '12.1' } });
    await assert.rejects(() => h.controller.run({ modelId: 'big-moe', runnerId: 'llama.cpp', requestId: 'request-0001' }),
        (error) => error.code === 'admission_incompatible' && /built for GPUs of compute capability 12\.1; NVIDIA Thor is 11\.0/.test(error.message));
    assert.equal(h.calls.download.length, 0);
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
