// Runtime models on unified memory without benchmark gates (owner, 2026-09-29),
// the Qwen MTP default with MTP as a declared model capability, experimental
// vLLM behind the operator's switch, and complete installed-distribution
// validation (DS002, DS004, DS005).
//
// Hardware, downloads, verification, runners and the runner installer are
// injected: no /proc, GPU, network or real model is used. Every number here is
// synthetic; no envelope, margin or constant is approved by these tests.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { controllerHandlers } from '../src/controlHandlers.mjs';
import { admit, computeBufferBytes, DEFAULT_KV_BYTES_PER_TOKEN, VLLM_OVERHEAD_BYTES, VLLM_RUNNER_RAM_BYTES } from '../src/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { UNIFIED, estimateUnifiedLlamaServer } from '../src/controller/profiles.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { VLLM_UNIFIED_SWITCH, vllmRunner } from '../src/runners/vllm.mjs';
import { TOOL_OPERATIONS, handleTool } from '../tools/local_llm_tool.mjs';
import { compareInstalledDistributions, lockedDistributions } from '../tools/runner_install_check.mjs';
import { modelEntryFromForm } from '../IDE-plugins/local-llm-settings/local-llm-settings-model.js';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const TOTAL = 120 * GIB;
const SEED = loadSeedCatalog();
const seedModel = (id) => SEED.find((model) => model.id === id);
const GB10 = Object.freeze({
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
});

function unifiedSnap(available = 100 * GIB, total = TOTAL) {
    return { gpu: structuredClone(GB10), memory: { totalBytes: total, availableBytes: available }, disk: { freeBytes: 400 * GIB }, cpus: 20 };
}

function unifiedParams(model, params = {}) {
    return llamaCppRunner.normalizeParams(params, { model, profile: 'unified' });
}

function admitUnified(model, params, snapshot = unifiedSnap(), runner = llamaCppRunner, source = model.sources.gguf) {
    return admit({ runner, model, source, params, snapshot, profile: 'unified' });
}

async function until(predicate, label, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 3));
    }
    assert.fail(`timed out waiting for ${label}`);
}

/** The Run form's preview, through the overview as the dashboard asks for it. */
async function previewOf(h, request) {
    return (await h.controller.overview({ preview: request })).preview;
}

/** A unified-memory controller; `dataDir` lets a second controller reload the first one's state. */
function harness(t, { seed = SEED, available = () => 100 * GIB, env = {}, dataDir, installer, resolveHf, snapshot = null, imageContract = null,
    hostArch = 'arm64' } = {}) {
    const dir = dataDir || fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-runtime-models-'));
    const weights = path.join(dir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const started = [];
    const calls = { download: 0, resolveHf: 0 };
    const controller = createController({
        dataDir: dir,
        env: { PATH: '/usr/bin', ...env },
        seedCatalog: seed,
        stateStore: createStateStore({ dataDir: dir }),
        snapshot: async () => (snapshot ? snapshot() : unifiedSnap(available())),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async () => { calls.download += 1; return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        verify: async () => ({ notes: [] }),
        remove: async () => 0,
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }),
        downloadSnapshot: async () => ({ status: 'complete', bytesTransferred: 0 }),
        resolveHf: resolveHf || (async () => { calls.resolveHf += 1; return { commit: 'c'.repeat(40), size: 20 * GIB, sha256: 'd'.repeat(64) }; }),
        sharedModelsRoot: null,
        imageContract,
        hostArch,
        installer: installer || { installable: () => false },
        shmDir: path.join(dir, 'shm'),
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner({ command, args, env: runnerEnv }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 7100 + started.length, command, args, env: runnerEnv,
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
        pollMs: 1,
        stopGraceMs: 20,
        readMemory: () => ({ totalBytes: TOTAL, availableBytes: available() }),
        readPressure: () => 0,
        unifiedGuardMs: 5,
        dropCache: () => true,
    });
    cleanupAfter(t, controller, dataDir ? null : dir);
    return { controller, started, calls, dataDir: dir };
}

// One cleanup per test: every controller of the test drains (and saves its
// state) before any of the test's directories is removed, so no drain can
// recreate a directory that was already removed.
const cleanups = new WeakMap();
function cleanupAfter(t, controller, dir) {
    let entry = cleanups.get(t);
    if (!entry) {
        entry = { controllers: [], dirs: [] };
        cleanups.set(t, entry);
        t.after(async () => {
            for (const each of entry.controllers) await each.drain().catch(() => {});
            for (const each of entry.dirs) fs.rmSync(each, { recursive: true, force: true });
        });
    }
    entry.controllers.push(controller);
    if (dir) entry.dirs.push(dir);
}

const valueOf = (args, flag) => args[args.indexOf(flag) + 1];

// ------------------------------------------------ bundled models by estimate

test('a bundled unified model without an envelope is admitted by a labelled estimate, not refused for lack of a benchmark', () => {
    const model = seedModel('gpt-oss-120b');
    assert.equal(model.unified, null, 'the bundled 120b entry carries no envelope');
    const params = unifiedParams(model);
    const result = admitUnified(model, params);
    assert.equal(result.status, 'ok', result.reason);
    const { estimate } = result;
    const kv = DEFAULT_KV_BYTES_PER_TOKEN * params.ctxSize;
    const compute = Math.round(computeBufferBytes({ ubatchSize: UNIFIED.batchSize, flashAttn: 'on' }));
    assert.equal(estimate.unifiedBytes, model.sources.gguf.size + kv + compute + UNIFIED.runtimeBytes + UNIFIED.cacheRamMiB * MIB);
    assert.equal(estimate.measured, false);
    assert.equal(estimate.envelope, null);
    assert.equal(estimate.isEstimate, true);
    assert.match(estimate.basis, /^estimate, not measured on this machine/);
    assert.deepEqual(estimate.defaulted, ['memory.kvBytesPerToken']);
    assert.deepEqual(estimate.userSizing, [], 'a seed entry is not user sizing');
    assert.equal(estimate.poolBytes, TOTAL, 'the pool is the host snapshot, not the model entry');
    assert.ok(result.warnings.some((warning) => /Not measured on unified memory: about .* is an estimate\. The memory guard stops/.test(warning)));
    assert.ok(result.warnings.some((warning) => /^Estimated without memory\.kvBytesPerToken \(the 64 KiB per token default can understate the KV cache several times for large dense models\); add them to the model entry/.test(warning)));
    assert.ok(!result.warnings.some((warning) => /Sized with .* from the model entry added at run time/.test(warning)));
    // The entry's own sizing data is used when it has it (gpt-oss-20b does).
    const small = seedModel('gpt-oss-20b');
    const smallEstimate = admitUnified(small, unifiedParams(small)).estimate;
    assert.equal(smallEstimate.kvBytes, small.memory.kvBytesPerToken * 32768 + small.memory.fixedKvBytes);
    assert.deepEqual(smallEstimate.defaulted, []);
});

test('the estimate still refuses what the pool cannot hold, and fails closed on unknown memory or size', () => {
    const model = seedModel('gpt-oss-120b');
    const params = unifiedParams(model);
    const need = admitUnified(model, params).estimate.unifiedBytes;
    // Beyond MemTotal less the host reserve: never.
    const small = admitUnified(model, params, unifiedSnap(60 * GIB, need + UNIFIED.hostReserveBytes - 1));
    assert.equal(small.status, 'incompatible');
    assert.match(small.reason, /must keep 16\.0 GiB for the host/);
    // Beyond MemAvailable less the floor: not now.
    const busy = admitUnified(model, params, unifiedSnap(need + UNIFIED.floorBytes - 1));
    assert.equal(busy.status, 'insufficient-now');
    assert.match(busy.reason, /8\.0 GiB kept free/);
    assert.equal(admitUnified(model, params, unifiedSnap(need + UNIFIED.floorBytes)).status, 'ok');
    // /proc/meminfo unreadable.
    const unreadable = admitUnified(model, params, { ...unifiedSnap(), memory: { totalBytes: TOTAL, availableBytes: null } });
    assert.equal(unreadable.status, 'incompatible');
    assert.match(unreadable.reason, /cannot be read/);
    // No pinned size.
    const unsized = admitUnified(model, params, unifiedSnap(), llamaCppRunner, { ...model.sources.gguf, size: 0 });
    assert.equal(unsized.status, 'incompatible');
    assert.match(unsized.reason, /weight size is unknown/);
});

test('a trusted envelope that covers the parameters still sizes the run; outside it the estimate applies and says so', () => {
    const rectangle = { runner: 'llama.cpp', loadMode: 'dio', maxCtx: 65536, maxParallel: 2, mtp: false, bufferBytes: 30 * GIB, transientBytes: 2 * GIB };
    const model = validateModel({
        id: 'trusted', displayName: 'Trusted', architecture: 'moe', contextLength: 262144, profiles: ['unified'],
        sources: { gguf: { type: 'huggingface', repo: 'a/b', file: 'm.gguf', revision: 'main', commit: 'a'.repeat(40), size: 20 * GIB, sha256: 'b'.repeat(64) } },
        unified: { envelope: [rectangle] },
    }, { seed: true });
    const inside = admitUnified(model, unifiedParams(model, { ctxSize: 32768 }));
    assert.equal(inside.status, 'ok');
    assert.equal(inside.estimate.unifiedBytes, 32 * GIB + UNIFIED.runtimeBytes + UNIFIED.cacheRamMiB * MIB);
    assert.equal(inside.estimate.envelope.maxCtx, 65536);
    // Catalog envelope data, never presented as calibration for this host.
    assert.equal(inside.estimate.measured, false);
    assert.equal(Object.hasOwn(inside.estimate, 'validated'), false);
    assert.match(inside.estimate.basis, /^catalog envelope data .*; not validated calibration for this host$/);
    assert.doesNotMatch(inside.estimate.basis, /measured envelope/);
    assert.match(inside.warnings[0], /^Sized from the catalog's envelope figures, which are not validated calibration for this host; the memory guard stops/);
    assert.equal(inside.estimate.poolBytes, TOTAL);
    for (const params of [{ ctxSize: 131072 }, { ctxSize: 32768, loadMode: 'none' }]) {
        const outside = admitUnified(model, unifiedParams(model, params));
        assert.equal(outside.status, 'ok', 'outside the rectangle is not a refusal any more');
        assert.equal(outside.estimate.envelope, null);
        assert.equal(outside.estimate.measured, false);
        assert.equal(outside.estimate.poolBytes, TOTAL);
        assert.match(outside.warnings[0], /^Outside the catalog's envelope \(context 65536 x 2 slots with load mode dio\)/);
    }
});

// ------------------------------------------------------------ MTP defaults

test('Qwen3.6 MTP is on by default and an explicit false wins; the default, the flags and the estimate agree', () => {
    const qwen = seedModel('qwen3.6-35b-a3b-mtp');
    assert.equal(qwen.mtp, true);
    const on = unifiedParams(qwen);
    const off = unifiedParams(qwen, { mtp: false });
    assert.equal(on.mtp, true);
    assert.equal(off.mtp, false);
    const launch = (params) => llamaCppRunner.buildLaunch({ artifactPath: '/m/q.gguf', params, port: 18080, apiKey: KEY, model: qwen, profile: 'unified' }).args;
    assert.equal(valueOf(launch({}), '--spec-type'), 'draft-mtp');
    assert.equal(valueOf(launch({}), '--spec-draft-n-max'), '3');
    assert.ok(!launch({ mtp: false }).includes('--spec-type'));
    const withMtp = estimateUnifiedLlamaServer({ model: qwen, source: qwen.sources.gguf, params: on });
    const without = estimateUnifiedLlamaServer({ model: qwen, source: qwen.sources.gguf, params: off });
    assert.ok(withMtp.mtpBytes > 0);
    assert.equal(without.mtpBytes, 0);
    assert.equal(withMtp.unifiedBytes - without.unifiedBytes, withMtp.mtpBytes);
    assert.ok(withMtp.defaulted.includes('memory.layers'));
    assert.match(withMtp.basis, /an MTP draft context/);
});

test('MTP is a declared capability: gpt-oss never gets it by default, and an explicit true is refused, not estimated', () => {
    for (const id of ['gpt-oss-20b', 'gpt-oss-120b', 'glm-4.7-flash']) {
        const model = seedModel(id);
        assert.equal(model.mtp, false);
        assert.equal(unifiedParams(model).mtp, false);
        assert.throws(() => unifiedParams(model, { mtp: true }), (error) => error.details?.field === 'mtp'
            && /has no multi-token-prediction head/.test(error.message));
    }
    // A recommended MTP default needs the declared head.
    assert.throws(() => validateModel({
        id: 'no-head', sources: { gguf: { type: 'huggingface', repo: 'a/b', file: 'm.gguf', revision: 'main' } },
        recommended: { unified: { 'llama.cpp': { mtp: true } } },
    }), (error) => error.details.field === 'recommended.unified.llama.cpp.mtp');
    assert.throws(() => validateModel({ id: 'bad-flag', mtp: 'yes', sources: { gguf: { type: 'huggingface', repo: 'a/b', file: 'm.gguf', revision: 'main' } } }),
        (error) => error.details.field === 'mtp');
});

test('the overview and the preview show the MTP default, and a Run launches with it or with an explicit false', async (t) => {
    const h = harness(t);
    const overview = await h.controller.overview();
    const row = overview.models.find((model) => model.id === 'qwen3.6-35b-a3b-mtp').runners['llama.cpp'];
    assert.equal(row.params.mtp, true);
    assert.equal(row.admission.status, 'ok', row.admission.reason);
    const gpt = overview.models.find((model) => model.id === 'gpt-oss-120b').runners['llama.cpp'];
    assert.equal(gpt.params.mtp, false);
    const refused = await previewOf(h, { modelId: 'gpt-oss-120b', runnerId: 'llama.cpp', params: { mtp: true } });
    assert.equal(refused.field, 'mtp');
    await h.controller.run({ requestId: 'request-mtp-off', modelId: 'qwen3.6-35b-a3b-mtp', runnerId: 'llama.cpp', params: { mtp: false } });
    await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
    assert.ok(!h.started[0].args.includes('--spec-type'));
    assert.equal(h.controller.state.deployment.params.mtp, false);
    // The explicit false is kept for the next Run of this model.
    assert.equal((await h.controller.overview()).models.find((model) => model.id === 'qwen3.6-35b-a3b-mtp').runners['llama.cpp'].params.mtp, false);
});

// ------------------------------------------------------ models added at run time

const ADDED = Object.freeze({
    id: 'user-moe', displayName: 'User MoE', architecture: 'moe', contextLength: 65536, profiles: ['unified'],
    memory: { layers: 40, kvBytesPerToken: 40960 },
    sources: { gguf: { type: 'huggingface', repo: 'user/Model-GGUF', file: 'model-Q4_K_M.gguf', revision: 'main' } },
});

test('a model added at run time is listed, sized by estimate, run, persisted and still there after a reload', async (t) => {
    const first = harness(t);
    const { model } = await first.controller.addModel(structuredClone(ADDED));
    assert.equal(model.sources.gguf.commit, 'c'.repeat(40), 'pinned when added');
    assert.equal(model.sources.gguf.size, 20 * GIB);
    assert.equal(first.calls.resolveHf, 1);
    const listed = (await first.controller.overview()).models.find((entry) => entry.id === 'user-moe');
    assert.ok(listed, 'listed in the unified profile');
    const row = listed.runners['llama.cpp'];
    assert.equal(row.admission.status, 'ok', row.admission.reason);
    assert.equal(row.admission.estimate.kvBytes, 40960 * row.params.ctxSize);
    assert.match(row.admission.estimate.basis, /from the model entry added at run time/);
    assert.equal(row.admission.estimate.measured, false);
    // Select it with explicit parameters, which win over the defaults.
    const preview = await previewOf(first, { modelId: 'user-moe', runnerId: 'llama.cpp', params: { ctxSize: 16384 } });
    assert.equal(preview.params.ctxSize, 16384);
    assert.equal(preview.admission.status, 'ok');
    await first.controller.run({ requestId: 'request-user-01', modelId: 'user-moe', runnerId: 'llama.cpp', params: { ctxSize: 16384 } });
    await until(() => first.controller.state.deployment?.phase === 'ready', 'ready');
    assert.equal(valueOf(first.started[0].args, '--ctx-size'), '16384');
    assert.equal(first.controller.state.deployment.admission.estimate.measured, false);
    await first.controller.stop();
    await first.controller.drain();

    // A new controller on the same state: the entry and its chosen parameters are back.
    const second = harness(t, { dataDir: first.dataDir });
    const reloaded = (await second.controller.overview()).models.find((entry) => entry.id === 'user-moe');
    assert.ok(reloaded);
    assert.equal(reloaded.runners['llama.cpp'].params.ctxSize, 16384);
    assert.equal(second.calls.resolveHf, 0, 'nothing is resolved again');
    // A model id cannot be added twice, and a seed id cannot be reused.
    await assert.rejects(second.controller.addModel(structuredClone(ADDED)), (error) => /already/.test(error.message));
});

test('an added model that does not fit now is refused before anything downloads; one too large for the pool, always', async (t) => {
    let available = 100 * GIB;
    const h = harness(t, { available: () => available });
    await h.controller.addModel(structuredClone(ADDED));
    available = 20 * GIB;
    const busy = await previewOf(h, { modelId: 'user-moe', runnerId: 'llama.cpp' });
    assert.equal(busy.admission.status, 'insufficient-now');
    await assert.rejects(h.controller.run({ requestId: 'request-user-busy', modelId: 'user-moe', runnerId: 'llama.cpp' }),
        (error) => error.code === 'admission_insufficient_now');
    assert.equal(h.calls.download, 0);
    assert.equal(h.started.length, 0);
    available = 100 * GIB;
    const huge = await previewOf(h, { modelId: 'user-moe', runnerId: 'llama.cpp', params: { ctxSize: 131072, parallel: 2 } });
    assert.equal(huge.admission.status, 'ok', 'a larger context is an estimate, not a refusal');
    const enormous = harness(t, { resolveHf: async () => ({ commit: 'c'.repeat(40), size: 110 * GIB, sha256: 'd'.repeat(64) }) });
    await enormous.controller.addModel({ ...structuredClone(ADDED), id: 'user-enormous' });
    const never = await previewOf(enormous, { modelId: 'user-enormous', runnerId: 'llama.cpp' });
    assert.equal(never.admission.status, 'incompatible');
    assert.match(never.admission.reason, /must keep 16\.0 GiB for the host/);
});

test('the admin tools add a model at run time and plan and start its Run through the operations the agent serves', async (t) => {
    const h = harness(t);
    // main.mjs serves controllerHandlers(controller) on the control socket; each tool maps its input onto one operation.
    const handlers = controllerHandlers(h.controller);
    for (const { op } of Object.values(TOOL_OPERATIONS)) assert.equal(typeof handlers[op], 'function', op);
    const ops = [];
    // The socket carries JSON both ways; so does this stand-in.
    const call = async (op, args) => {
        ops.push(op);
        return JSON.parse(JSON.stringify(await handlers[op](JSON.parse(JSON.stringify(args)))));
    };
    const admin = { user: { roles: ['admin'], email: 'admin@example.com' } };
    const tool = (name, input, authInfo = admin) => handleTool(name, input, { authInfo, call });
    await assert.rejects(tool('local_llm_model_add', { model: structuredClone(ADDED) }, { user: { roles: ['user'] } }),
        (error) => error.code === 'admin_required');
    assert.deepEqual(ops, [], 'refused before any operation');
    const added = await tool('local_llm_model_add', { model: structuredClone(ADDED) });
    assert.equal(added.model.id, 'user-moe');
    assert.equal(added.model.sources.gguf.commit, 'c'.repeat(40));
    const listed = (await tool('local_llm_overview', {})).models.find((entry) => entry.id === 'user-moe');
    assert.equal(listed.runners['llama.cpp'].admission.status, 'ok');
    assert.equal(listed.runners['llama.cpp'].admission.estimate.measured, false);
    const { preview } = await tool('local_llm_overview', { preview: { modelId: 'user-moe', runnerId: 'llama.cpp', params: { ctxSize: 16384 } } });
    assert.equal(preview.params.ctxSize, 16384, 'explicit parameters win');
    assert.equal(preview.admission.status, 'ok');
    const run = await tool('local_llm_run', { requestId: 'request-tool-01', modelId: 'user-moe', runnerId: 'llama.cpp', params: { ctxSize: 16384 } });
    assert.equal(run.accepted, true);
    await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
    assert.equal(valueOf(h.started[0].args, '--ctx-size'), '16384');
    const status = await tool('local_llm_status', {});
    assert.equal(status.deployment.modelId, 'user-moe');
    await tool('local_llm_stop', {});
    assert.deepEqual(ops, ['addModel', 'overview', 'overview', 'run', 'status', 'stop']);
});

test('user sizing can lower only the KV and MTP terms, and every run it lowers says the memory guard is then the only backstop', async (t) => {
    const h = harness(t);
    // A 40 GiB dense entry that claims one byte per token of KV cache.
    const huge = { commit: 'c'.repeat(40), size: 40 * GIB, sha256: 'd'.repeat(64) };
    const understated = harness(t, { resolveHf: async () => huge });
    await understated.controller.addModel({ ...structuredClone(ADDED), id: 'user-tiny-kv', architecture: 'dense', contextLength: 262144,
        memory: { kvBytesPerToken: 1 } });
    await understated.controller.addModel({ ...structuredClone(ADDED), id: 'user-no-sizing', architecture: 'dense', contextLength: 262144,
        memory: undefined });
    const params = { ctxSize: 262144 };
    const tiny = (await previewOf(understated, { modelId: 'user-tiny-kv', runnerId: 'llama.cpp', params })).admission;
    const defaulted = (await previewOf(understated, { modelId: 'user-no-sizing', runnerId: 'llama.cpp', params })).admission;
    assert.equal(tiny.status, 'ok', 'user sizing is trusted as the admin\'s data, not refused');
    assert.deepEqual(tiny.estimate.userSizing, ['memory.kvBytesPerToken']);
    assert.ok(tiny.warnings.some((warning) => /^Sized with memory\.kvBytesPerToken from the model entry added at run time\. Nothing checks these against the weights: an understated value makes the estimate too small, and then only the memory guard stands behind the run\.$/.test(warning)));
    assert.ok(!defaulted.warnings.some((warning) => /^Sized with/.test(warning)), 'a defaulted estimate has no user-sizing warning');
    // Only the KV (and MTP) terms moved; weights, compute, runtime and the prompt cache are the same.
    assert.ok(tiny.estimate.unifiedBytes < defaulted.estimate.unifiedBytes);
    for (const term of ['weightsBytes', 'computeBytes', 'runtimeBytes', 'cacheRamBytes']) {
        assert.equal(tiny.estimate[term], defaulted.estimate[term], term);
    }
    assert.equal(defaulted.estimate.unifiedBytes - tiny.estimate.unifiedBytes, defaulted.estimate.kvBytes - tiny.estimate.kvBytes);
    // An added entry with a declared head and its own layers: layers feed the MTP draft, and are named.
    await h.controller.addModel({ ...structuredClone(ADDED), id: 'user-mtp-sized', mtp: true });
    const mtp = (await previewOf(h, { modelId: 'user-mtp-sized', runnerId: 'llama.cpp', params: { mtp: true } })).admission;
    assert.deepEqual(mtp.estimate.userSizing, ['memory.kvBytesPerToken', 'memory.layers']);
    // vLLM reads the same field, and says so too.
    const vllmModel = validateModel({ ...structuredClone(ADDED), id: 'user-hf', profiles: ['unified'],
        sources: { hf: { ...HF_MODEL.sources.hf } } }, { seed: false });
    const vllm = admitUnified(vllmModel, vllmRunner.normalizeParams({}, { model: vllmModel, profile: 'unified' }), unifiedSnap(), vllmRunner,
        vllmModel.sources.hf);
    assert.deepEqual(vllm.estimate.userSizing, ['memory.kvBytesPerToken']);
    assert.ok(vllm.warnings.some((warning) => /^Sized with memory\.kvBytesPerToken from the model entry added at run time/.test(warning)));
    assert.equal(vllm.estimate.poolBytes, TOTAL);
});

test('malformed metadata is refused with the field to fix, and nothing is stored or resolved', async (t) => {
    const h = harness(t);
    for (const [change, field] of [
        [{ contextLength: 'abc' }, 'contextLength'],
        [{ contextLength: 100 }, 'contextLength'],
        [{ memory: { kvBytesPerToken: 0 } }, 'memory.kvBytesPerToken'],
        [{ memory: { layers: 2000 } }, 'memory.layers'],
        [{ mtp: 'yes' }, 'mtp'],
        [{ profiles: ['mainframe'] }, 'profiles'],
        [{ sources: { gguf: { type: 'huggingface', repo: 'user/Model-GGUF', file: '', revision: 'main' } } }, 'sources.gguf.file'],
        [{ unified: { envelope: [{ runner: 'llama.cpp', loadMode: 'dio', maxCtx: 1024, maxParallel: 1, mtp: false, bufferBytes: 1, transientBytes: 0 }] } }, 'unified'],
    ]) {
        await assert.rejects(h.controller.addModel({ ...structuredClone(ADDED), ...change }),
            (error) => error.code === 'invalid_model' && (error.details?.field ?? '').startsWith(field), JSON.stringify(change));
    }
    assert.equal(h.calls.resolveHf, 0);
    assert.equal((await h.controller.overview()).models.some((entry) => entry.id === 'user-moe'), false);
});

test('an added model without sizing data runs by estimate with named defaults; one that declares an MTP head can use it', async (t) => {
    const h = harness(t);
    const { memory: _memory, contextLength: _context, ...bare } = structuredClone(ADDED);
    await h.controller.addModel({ ...bare, id: 'user-bare' });
    await h.controller.addModel({ ...structuredClone(ADDED), id: 'user-mtp', mtp: true });
    const rows = Object.fromEntries((await h.controller.overview()).models.map((entry) => [entry.id, entry.runners['llama.cpp']]));
    assert.equal(rows['user-bare'].admission.status, 'ok');
    assert.deepEqual(rows['user-bare'].admission.estimate.defaulted, ['memory.kvBytesPerToken', 'contextLength']);
    assert.ok(rows['user-bare'].admission.warnings.some((warning) => /^Estimated without memory\.kvBytesPerToken \(the 64 KiB per token default can understate .*\), contextLength \(the context is not capped at the model's training context\)/.test(warning)));
    assert.deepEqual(rows['user-bare'].admission.estimate.userSizing, [], 'no sizing supplied, so none to warn about');
    assert.equal(rows['user-mtp'].params.mtp, false, 'a declared head is not a default');
    const withMtp = await previewOf(h, { modelId: 'user-mtp', runnerId: 'llama.cpp', params: { mtp: true } });
    assert.equal(withMtp.params.mtp, true);
    assert.ok(withMtp.admission.estimate.mtpBytes > 0);
    const refused = await previewOf(h, { modelId: 'user-bare', runnerId: 'llama.cpp', params: { mtp: true } });
    assert.equal(refused.field, 'mtp');
});

test('the Add model form carries the optional sizing data and the MTP head; malformed numbers reach validation', () => {
    // What the dashboard's Add model form sends: its inputs' names, and the MTP checkbox's value when it is checked.
    const html = fs.readFileSync(new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.html',
        import.meta.url), 'utf8');
    const inputs = [...html.matchAll(/<input\b[^>]*>/g)].map(([tag]) => ({
        name: /\bname="([^"]+)"/.exec(tag)?.[1], type: /\btype="([^"]+)"/.exec(tag)?.[1], value: /\bvalue="([^"]*)"/.exec(tag)?.[1] }));
    for (const name of ['contextLength', 'layers', 'kvBytesPerToken']) {
        assert.equal(inputs.filter((input) => input.name === name && input.type === 'number').length, 1, name);
    }
    const mtpInputs = inputs.filter((input) => input.name === 'mtp');
    assert.deepEqual(mtpInputs, [{ name: 'mtp', type: 'checkbox', value: 'true' }]);
    const base = { id: 'Form-Model', sourceKind: 'huggingface', repo: 'u/m', file: 'm.gguf' };
    const full = modelEntryFromForm({ ...base, contextLength: '65536', layers: '40', kvBytesPerToken: '40960', mtp: mtpInputs[0].value });
    assert.equal(full.contextLength, 65536);
    assert.deepEqual(full.memory, { layers: 40, kvBytesPerToken: 40960 });
    assert.equal(full.mtp, true);
    const empty = modelEntryFromForm({ ...base, contextLength: '', layers: ' ', kvBytesPerToken: '' });
    assert.equal(Object.hasOwn(empty, 'contextLength'), false);
    assert.equal(Object.hasOwn(empty, 'memory'), false);
    assert.equal(Object.hasOwn(empty, 'mtp'), false);
    const typo = modelEntryFromForm({ ...base, contextLength: '64k' });
    assert.equal(typo.contextLength, '64k');
    assert.throws(() => validateModel(typo), (error) => error.details.field === 'contextLength');
});

// ------------------------------------------------- experimental vLLM, unified

const HF_MODEL = seedModel('gpt-oss-20b');

function vllmInstaller(runDir) {
    return {
        installable: (id) => id === 'vllm',
        describe: async () => ({ installed: true, runnable: true, version: '0.30.0', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
        ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }),
        entryFor: () => ({ id: 'vllm', version: '0.30.0', totalBytes: 1, licence: { name: 'Apache-2.0', requiresAcceptance: false } }),
        pathsFor: () => ({ runDir }),
    };
}

// The source.contract each published image writes (container-image-builds Dockerfile and Dockerfile.arm64): the
// amd64 one has no architecture line and names ik_llama.cpp, which only it builds.
const AMD64_CONTRACT = Object.freeze({ llama_cpp: 'b11159', llama_cpp_cuda: '12.8', ollama: '0.34.4', ik_llama_cpp: '20f7a72',
    ik_llama_cpp_cuda: '12.8', uv: '0.12.18', lmstudio_sdk: '1.5.0' });
const ARM64_CONTRACT = Object.freeze({ architecture: 'arm64', llama_cpp: 'b11159', llama_cpp_cuda: '13.4', llama_cpp_build: 'release',
    gpu_compute_capabilities: '12.1', uv: '0.12.18' });
const GPU_UNREADABLE = () => ({ gpu: { available: false, reason: 'nvidia-smi failed: the GPU grant was revoked' },
    memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB }, disk: { freeBytes: 400 * GIB }, cpus: 16 });
// A readable arm64 GPU with memory of its own (GH200-like) and a readable GB10.
const GH200_LIKE = () => ({ gpu: { available: true, name: 'NVIDIA GH200', memoryModel: 'dedicated', totalBytes: 96 * GIB, freeBytes: 95 * GIB,
    usedBytes: GIB, processes: [], device: { computeCapability: '12.1' } }, memory: { totalBytes: 480 * GIB, availableBytes: 470 * GIB },
disk: { freeBytes: 400 * GIB }, cpus: 72 });

test('an image that can never run on unified memory keeps vLLM as before while the GPU cannot be read', async (t) => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-amd64-'));
    t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
    // The real amd64 contract (positive evidence: ik_llama.cpp) on any CPU, or an architecture-less contract on an x64 CPU.
    for (const [imageContract, hostArch] of [[AMD64_CONTRACT, 'arm64'], [AMD64_CONTRACT, 'x64'], [{ llama_cpp: 'b11159' }, 'x64']]) {
        const h = harness(t, { installer: vllmInstaller(runDir), imageContract, hostArch, snapshot: GPU_UNREADABLE });
        const overview = await h.controller.overview();
        const row = overview.runners.find((runner) => runner.id === 'vllm');
        assert.equal(row.enabled, true, 'as at 50f39385: no switch on a dedicated-only image');
        assert.equal(Object.hasOwn(row, 'disabledReason'), false);
        // The install is not refused by the switch (here the stub reports it already installed and runnable).
        const install = await h.controller.installRunner({ runnerId: 'vllm' });
        assert.equal(install.installed, true);
    }
});

test('an install before any overview decides the profile first: a readable dedicated arm64 GPU installs, a readable GB10 needs the switch', async (t) => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-first-'));
    t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
    const dedicated = harness(t, { installer: vllmInstaller(runDir), imageContract: ARM64_CONTRACT, hostArch: 'arm64', snapshot: GH200_LIKE });
    assert.equal((await dedicated.controller.installRunner({ runnerId: 'vllm' })).installed, true);
    assert.equal((await dedicated.controller.overview()).profile, 'dedicated');
    const gb10 = harness(t, { installer: vllmInstaller(runDir), imageContract: ARM64_CONTRACT, hostArch: 'arm64' });
    await assert.rejects(gb10.controller.installRunner({ runnerId: 'vllm' }), (error) => error.code === 'runner_disabled'
        && /^vLLM is experimental on unified memory and is not enabled on this deployment/.test(error.message)
        && !/could not be read/.test(error.message));
});

test('an image that can run on unified memory keeps vLLM off while the GPU cannot be read, and says what turns it on', async (t) => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-arm64-'));
    t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
    // arm64's contract, none (development), and on an arm64 CPU an empty, malformed or architecture-less one:
    // only positive evidence of the amd64 image opens the gate.
    for (const [imageContract, hostArch] of [[ARM64_CONTRACT, 'arm64'], [null, 'arm64'], [null, 'x64'], [{}, 'arm64'],
        [{ llama_cpp: 'b11159', uv: '0.12.18' }, 'arm64']]) {
        const h = harness(t, { installer: vllmInstaller(runDir), imageContract, hostArch, snapshot: GPU_UNREADABLE });
        const row = (await h.controller.overview()).runners.find((runner) => runner.id === 'vllm');
        assert.equal(row.enabled, false, JSON.stringify({ imageContract, hostArch }));
        assert.match(row.disabledReason, new RegExp(`the GPU could not be read, so it is not known whether it shares system memory, where vLLM is experimental and needs the operator's switch\\. The operator can turn it on with ploinky var ${VLLM_UNIFIED_SWITCH} experimental`));
        assert.doesNotMatch(row.disabledReason, /this image runs on GPUs that share/, 'no claim about the hardware it could not read');
        assert.doesNotMatch(row.disabledReason, /try again|open the overview/i, 'no retry that cannot succeed');
        await assert.rejects(h.controller.installRunner({ runnerId: 'vllm' }), (error) => error.code === 'runner_disabled');
        // With the operator's switch it is not refused by the gate.
        const on = harness(t, { installer: vllmInstaller(runDir), imageContract, hostArch, snapshot: GPU_UNREADABLE, env: { [VLLM_UNIFIED_SWITCH]: 'experimental' } });
        assert.equal((await on.controller.installRunner({ runnerId: 'vllm' })).installed, true);
    }
});

test('vLLM on unified memory stays off without the operator switch; no parameter, entry or runner choice turns it on', async (t) => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-off-'));
    t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
    const h = harness(t, { installer: vllmInstaller(runDir) });
    const overview = await h.controller.overview();
    const row = overview.models.find((model) => model.id === HF_MODEL.id).runners.vllm;
    assert.equal(row.admission.status, 'incompatible');
    assert.match(row.admission.reason, new RegExp(`experimental on unified memory.*ploinky var ${VLLM_UNIFIED_SWITCH} experimental`));
    assert.equal(overview.runners.find((runner) => runner.id === 'vllm').enabled, false);
    // The Run form's preview is refused the same way, before anything is sized.
    const preview = await previewOf(h, { modelId: HF_MODEL.id, runnerId: 'vllm', params: { gpuMemoryUtilization: 0.3 } });
    assert.equal(preview.admission.status, 'incompatible');
    assert.match(preview.admission.reason, new RegExp(`ploinky var ${VLLM_UNIFIED_SWITCH} experimental`));
    assert.equal(preview.params, null);
    for (const env of [{}, { [VLLM_UNIFIED_SWITCH]: 'yes' }, { [VLLM_UNIFIED_SWITCH]: 'EXPERIMENTAL' }]) {
        const gated = harness(t, { installer: vllmInstaller(runDir), env });
        await gated.controller.overview();
        await assert.rejects(gated.controller.run({ requestId: 'request-vllm-off', modelId: HF_MODEL.id, runnerId: 'vllm', params: { gpuMemoryUtilization: 0.3 } }),
            (error) => error.code === 'runner_disabled');
        await assert.rejects(gated.controller.installRunner({ runnerId: 'vllm' }), (error) => error.code === 'runner_disabled');
        assert.equal(gated.started.length, 0);
    }
    // A dedicated GPU is not gated (the switch is for unified memory only), and an undecided profile fails closed.
    assert.equal(vllmRunner.enabled({}, 'dedicated').enabled, true);
    assert.equal(vllmRunner.enabled({}, null).enabled, false);
});

test('with the switch, experimental vLLM is sized from a fresh snapshot and launched with that share, never as measured', async (t) => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-on-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    let available = 100 * GIB;
    const h = harness(t, { installer: vllmInstaller(path.join(root, 'opt', 'runners', 'vllm', '0.30.0')), env: { [VLLM_UNIFIED_SWITCH]: 'experimental' },
        available: () => available });
    const preview = await previewOf(h, { modelId: HF_MODEL.id, runnerId: 'vllm' });
    assert.equal(preview.admission.status, 'ok', preview.admission.reason);
    assert.equal(Object.hasOwn(preview.params, 'cpuOffloadGb'), false, 'no CPU offload in one pool');
    const { estimate } = preview.admission;
    assert.equal(estimate.experimental, true);
    assert.equal(estimate.measured, false);
    assert.equal(estimate.envelope, null);
    assert.match(preview.admission.warnings[0], /^Experimental: vLLM has not been measured on unified memory/);
    // The share is computed from this snapshot: the floor and the runner's RAM stay free, at most 0.9.
    const expected = Math.min(0.9, Math.floor(((TOTAL - UNIFIED.hostReserveBytes - VLLM_RUNNER_RAM_BYTES) / TOTAL) * 100) / 100,
        Math.floor(((available - UNIFIED.floorBytes - VLLM_RUNNER_RAM_BYTES) / TOTAL) * 100) / 100);
    assert.equal(estimate.gpuMemoryUtilization, expected);
    assert.equal(estimate.unifiedBytes, Math.round(expected * TOTAL) + VLLM_RUNNER_RAM_BYTES);
    // The Run is re-admitted from a fresh snapshot before launch: less memory by then gives a smaller share.
    await h.controller.run({ requestId: 'request-vllm-on', modelId: HF_MODEL.id, runnerId: 'vllm' });
    available = 60 * GIB;
    await until(() => h.controller.state.deployment?.phase === 'ready' || h.controller.state.deployment?.phase === 'error', 'settled');
    assert.equal(h.controller.state.deployment.phase, 'ready', h.controller.state.deployment.error);
    const share = h.controller.state.deployment.admission.estimate.gpuMemoryUtilization;
    assert.equal(share, Math.floor(((60 * GIB - UNIFIED.floorBytes - VLLM_RUNNER_RAM_BYTES) / TOTAL) * 100) / 100);
    const [launched] = h.started;
    assert.equal(valueOf(launched.args, '--gpu-memory-utilization'), String(share));
    assert.ok(!launched.args.includes('--cpu-offload-gb'));
    assert.equal(launched.env.VLLM_USE_FLASHINFER_SAMPLER, '0');
    assert.equal(launched.env.HF_HUB_OFFLINE, '1');
    await h.controller.stop();
});

test('experimental vLLM admission: an admin share that cannot hold the model is refused, and a busy pool is not now', () => {
    const at = (params, snap = unifiedSnap()) => admitUnified(HF_MODEL, vllmRunner.normalizeParams(params, { model: HF_MODEL, profile: 'unified' }),
        snap, vllmRunner, HF_MODEL.sources.hf);
    const tiny = at({ gpuMemoryUtilization: 0.1 });
    assert.equal(tiny.status, 'incompatible');
    assert.match(tiny.reason, /gpuMemoryUtilization 0\.1 gives vLLM about/);
    const greedy = at({ gpuMemoryUtilization: 0.95 });
    assert.equal(greedy.status, 'incompatible');
    assert.match(greedy.reason, /must keep 16\.0 GiB for the host/);
    const busy = at({}, unifiedSnap(20 * GIB));
    assert.equal(busy.status, 'insufficient-now');
    assert.match(busy.reason, /at least a 0\.1 share/);
    const adminBusy = at({ gpuMemoryUtilization: 0.3 }, unifiedSnap(30 * GIB));
    assert.equal(adminBusy.status, 'insufficient-now');
    const unreadable = at({}, { ...unifiedSnap(), memory: { totalBytes: null, availableBytes: 100 * GIB } });
    assert.equal(unreadable.status, 'incompatible');
    assert.equal(at({ gpuMemoryUtilization: 0.3 }).status, 'ok');
    assert.throws(() => vllmRunner.normalizeParams({ cpuOffloadGb: 4 }, { model: HF_MODEL, profile: 'unified' }), /cpuOffloadGb/);
    // The dedicated profile keeps its own schema and policy.
    assert.equal(vllmRunner.paramSchemaFor('dedicated').properties.cpuOffloadGb.default, 0);
});

const vllmAt = (params, snap = unifiedSnap(), model = HF_MODEL) => admitUnified(model,
    vllmRunner.normalizeParams(params, { model, profile: 'unified' }), snap, vllmRunner, model.sources.hf);

test('a bundled entry\'s own sizing is not user sizing, for llama.cpp and vLLM alike', () => {
    const bundled = seedModel('gpt-oss-20b');
    assert.ok(Number.isInteger(bundled.memory.kvBytesPerToken) && Number.isInteger(bundled.memory.fixedKvBytes));
    const llama = admitUnified(bundled, unifiedParams(bundled));
    const vllm = vllmAt({}, unifiedSnap(), bundled);
    for (const result of [llama, vllm]) {
        assert.deepEqual(result.estimate.userSizing, []);
        assert.ok(!result.warnings.some((warning) => /^Sized with/.test(warning)), result.warnings.join(' | '));
    }
    assert.equal(llama.estimate.kvBytes, bundled.memory.kvBytesPerToken * 32768 + bundled.memory.fixedKvBytes, 'its sizing is used');
});

test('vLLM\'s share must hold the weights, the KV cache and its 768 MiB overhead: just below refuses, just above admits', () => {
    const base = vllmAt({ gpuMemoryUtilization: 0.5 }).estimate;
    assert.equal(base.overheadBytes, VLLM_OVERHEAD_BYTES);
    const needed = base.weightsBytes + base.kvBytes + VLLM_OVERHEAD_BYTES;
    const exact = needed / TOTAL;
    const below = vllmAt({ gpuMemoryUtilization: exact - 0.0005 });
    const above = vllmAt({ gpuMemoryUtilization: exact + 0.0005 });
    assert.equal(below.status, 'incompatible', below.reason);
    assert.match(below.reason, /and vLLM's overhead need about/);
    assert.equal(above.status, 'ok', above.reason);
    // Just above the weights and KV cache alone, but short of the overhead: still refused.
    const withoutOverhead = (base.weightsBytes + base.kvBytes) / TOTAL;
    assert.equal(vllmAt({ gpuMemoryUtilization: withoutOverhead + 0.0005 }).status, 'incompatible');
});

test('vLLM\'s automatic share keeps the host reserve and never exceeds 0.9', () => {
    // A small pool with ample free memory: the reserve (with the runner's RAM) binds.
    const small = vllmAt({}, unifiedSnap(39 * GIB, 40 * GIB));
    const reserveShare = Math.floor(((40 * GIB - UNIFIED.hostReserveBytes - VLLM_RUNNER_RAM_BYTES) / (40 * GIB)) * 100) / 100;
    const freeShare = Math.floor(((39 * GIB - UNIFIED.floorBytes - VLLM_RUNNER_RAM_BYTES) / (40 * GIB)) * 100) / 100;
    assert.ok(reserveShare < freeShare && reserveShare < 0.9);
    assert.equal(small.status, 'ok', small.reason);
    assert.equal(small.estimate.gpuMemoryUtilization, reserveShare);
    assert.ok(small.estimate.unifiedBytes <= 40 * GIB - UNIFIED.hostReserveBytes);
    // A large pool, nearly all free: the 0.9 cap binds.
    const large = vllmAt({}, unifiedSnap(990 * GIB, 1000 * GIB));
    assert.equal(large.status, 'ok', large.reason);
    assert.equal(large.estimate.gpuMemoryUtilization, 0.9);
});

test('vLLM\'s fp8 KV cache is half the size of the default one', () => {
    const auto = vllmAt({ maxModelLen: 8192 }).estimate;
    const fp8 = vllmAt({ maxModelLen: 8192, kvCacheDtype: 'fp8' }).estimate;
    assert.equal(auto.kvBytes, HF_MODEL.memory.kvBytesPerToken * 8192);
    assert.equal(fp8.kvBytes, Math.round(auto.kvBytes / 2));
});

test('the MTP draft context is exactly one more layer of KV cache plus one more set of compute buffers', () => {
    const model = validateModel({ ...structuredClone(ADDED), id: 'user-mtp-exact', mtp: true,
        sources: { gguf: { ...ADDED.sources.gguf, commit: 'c'.repeat(40), size: 20 * GIB, sha256: 'd'.repeat(64) } } }, { seed: false });
    const params = unifiedParams(model, { mtp: true, ctxSize: 32768 });
    const estimate = estimateUnifiedLlamaServer({ model, source: model.sources.gguf, params, memory: { totalBytes: TOTAL } });
    const compute = Math.round(computeBufferBytes({ ubatchSize: UNIFIED.batchSize, flashAttn: 'on' }));
    assert.equal(estimate.kvBytes, 40960 * 32768);
    assert.equal(estimate.mtpBytes, Math.round(estimate.kvBytes / 40 + compute));
});

test('vLLM refuses to size anything when the pool cannot be read, whatever the share', () => {
    for (const memory of [
        { totalBytes: Number.NaN, availableBytes: 100 * GIB },
        { totalBytes: TOTAL, availableBytes: Number.NaN },
        { totalBytes: Number.POSITIVE_INFINITY, availableBytes: 100 * GIB },
        { totalBytes: TOTAL, availableBytes: undefined },
    ]) {
        for (const params of [{}, { gpuMemoryUtilization: 0.3 }]) {
            const result = vllmAt(params, { ...unifiedSnap(), memory });
            assert.equal(result.status, 'incompatible', JSON.stringify({ memory, params }));
            assert.match(result.reason, /cannot be read/);
        }
    }
});

test('the vLLM switch fails closed for a caller that names no profile', () => {
    assert.equal(vllmRunner.enabled({}).enabled, false);
    assert.equal(vllmRunner.enabled({}, undefined).enabled, false);
    assert.equal(vllmRunner.enabled({}, 'dedicated').enabled, true);
    assert.equal(vllmRunner.enabled({ [VLLM_UNIFIED_SWITCH]: 'experimental' }).enabled, true);
});

test('the vLLM user-sizing warning says what an understated value does there: a looser fit check, not a smaller need', () => {
    const model = validateModel({ ...structuredClone(ADDED), id: 'user-hf-sized', sources: { hf: { ...HF_MODEL.sources.hf } } }, { seed: false });
    const result = vllmAt({}, unifiedSnap(), model);
    assert.ok(result.warnings.some((warning) => warning === 'Sized with memory.kvBytesPerToken from the model entry added at run time. '
        + 'Nothing checks these against the weights: an understated value only loosens the check that the weights and KV cache fit '
        + 'vLLM\'s share, so vLLM may fail to start; its memory need (the share plus its runner RAM) does not depend on it.'),
    result.warnings.join(' | '));
});

test('the llama-server estimate\'s basis names every defaulted input', () => {
    const { memory: _memory, contextLength: _context, ...bare } = structuredClone(ADDED);
    const model = validateModel({ ...bare, id: 'user-bare-basis', mtp: true,
        sources: { gguf: { ...ADDED.sources.gguf, commit: 'c'.repeat(40), size: 20 * GIB, sha256: 'd'.repeat(64) } } }, { seed: false });
    const estimate = estimateUnifiedLlamaServer({ model, source: model.sources.gguf, params: unifiedParams(model, { mtp: true }), memory: { totalBytes: TOTAL } });
    assert.deepEqual(estimate.defaulted, ['memory.kvBytesPerToken', 'memory.layers', 'contextLength']);
    assert.match(estimate.basis, /; defaults used for memory\.kvBytesPerToken, memory\.layers, contextLength$/);
    const sized = estimateUnifiedLlamaServer({ model: seedModel('gpt-oss-20b'), source: seedModel('gpt-oss-20b').sources.gguf,
        params: unifiedParams(seedModel('gpt-oss-20b')), memory: { totalBytes: TOTAL } });
    assert.doesNotMatch(sized.basis, /defaults used/);
});

// ------------------------------------------------ complete install validation

test('the install check compares every installed distribution with the lock: none extra, none missing, exact versions', () => {
    const entry = { files: [
        { name: 'vllm-0.30.0-cp38-abi3-manylinux_2_28_aarch64.whl' },
        { name: 'typing_extensions-4.15.0-py3-none-any.whl' },
        { name: 'Jinja2-3.1.6-py3-none-any.whl' },
        { name: 'o200k_base.tiktoken', into: 'tiktoken' },
        { name: 'source.tar.gz', extract: 'src' },
    ] };
    assert.deepEqual(lockedDistributions(entry), { vllm: '0.30.0', 'typing-extensions': '4.15.0', jinja2: '3.1.6' });
    const exact = compareInstalledDistributions(entry, { vllm: '0.30.0', typing_extensions: '4.15.0', jinja2: '3.1.6' });
    assert.deepEqual(exact.problems, []);
    assert.equal(exact.installed, 3);
    const drift = compareInstalledDistributions(entry, { vllm: '0.30.0', 'Typing.Extensions': '4.14.0', pip: '25.0' });
    assert.deepEqual(drift.extra, ['pip']);
    assert.deepEqual(drift.missing, ['jinja2']);
    assert.deepEqual(drift.mismatched, ['typing-extensions 4.14.0 (lock 4.15.0)']);
    assert.equal(drift.problems.length, 3);
});

test('a name installed twice (a stale copy beside the locked one) fails the install check, whatever the order', () => {
    const entry = { files: [{ name: 'foo.bar-2.0-py3-none-any.whl' }, { name: 'vllm-0.30.0-cp38-abi3-manylinux_2_28_aarch64.whl' }] };
    for (const installed of [
        [['foo.bar', '2.0'], ['Foo_Bar', '1.0'], ['vllm', '0.30.0']],
        [['Foo_Bar', '1.0'], ['foo.bar', '2.0'], ['vllm', '0.30.0']],
        [['foo-bar', '2.0'], ['foo-bar', '2.0'], ['vllm', '0.30.0']],
    ]) {
        const result = compareInstalledDistributions(entry, installed);
        assert.equal(result.duplicated.length, 1, JSON.stringify(installed));
        assert.match(result.duplicated[0], /^foo-bar \(/);
        assert.ok(result.problems.some((problem) => /^installed more than once under one name: foo-bar/.test(problem)));
        assert.equal(result.installed, 3);
    }
    // Pairs and the object form agree when nothing is duplicated.
    assert.deepEqual(compareInstalledDistributions(entry, [['foo.bar', '2.0'], ['vllm', '0.30.0']]).problems, []);
    assert.deepEqual(compareInstalledDistributions(entry, { 'foo.bar': '2.0', vllm: '0.30.0' }).problems, []);
});

// ------------------------------------------------ the tool entry's operations

test('the operations main.mjs serves are exactly those the tools and the chat responder call', () => {
    const called = new Set(Object.values(TOOL_OPERATIONS).map(({ op }) => op));
    for (const file of ['../src/chatResponder.mjs', '../src/testPrompt.mjs']) {
        const source = fs.readFileSync(new URL(file, import.meta.url), 'utf8');
        for (const [, op] of source.matchAll(/\bcall\('([A-Za-z]+)'/g)) called.add(op);
    }
    assert.ok(called.has('chatTarget') && called.has('recordCompletion'));
    const served = Object.keys(controllerHandlers({}));
    assert.deepEqual(served.sort(), [...called].sort());
    const main = fs.readFileSync(new URL('../src/main.mjs', import.meta.url), 'utf8');
    assert.match(main, /handlers: controllerHandlers\(controller\)/);
});
