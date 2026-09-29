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

import { admit, computeBufferBytes, DEFAULT_KV_BYTES_PER_TOKEN, VLLM_RUNNER_RAM_BYTES } from '../src/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { UNIFIED, estimateUnifiedLlamaServer } from '../src/controller/profiles.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { VLLM_UNIFIED_SWITCH, vllmRunner } from '../src/runners/vllm.mjs';
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
function harness(t, { seed = SEED, available = () => 100 * GIB, env = {}, dataDir, installer, resolveHf, snapshot = null, imageContract = null } = {}) {
    const dir = dataDir || fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-runtime-models-'));
    if (!dataDir) t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
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
    t.after(() => controller.drain());
    return { controller, started, calls, dataDir: dir };
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
    const base = { id: 'Form-Model', sourceKind: 'huggingface', repo: 'u/m', file: 'm.gguf' };
    const full = modelEntryFromForm({ ...base, contextLength: '65536', layers: '40', kvBytesPerToken: '40960', mtp: 'on' });
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

// The source.contract of each published image (container-image-builds): amd64 has no architecture line.
const AMD64_CONTRACT = Object.freeze({ llama_cpp: 'b11159', llama_cpp_cuda: '12.8', ollama: '0.34.4', uv: '0.12.18' });
const ARM64_CONTRACT = Object.freeze({ architecture: 'arm64', llama_cpp: 'b11159', llama_cpp_cuda: '13.4', gpu_compute_capabilities: '12.1', uv: '0.12.18' });
const GPU_UNREADABLE = () => ({ gpu: { available: false, reason: 'nvidia-smi failed: the GPU grant was revoked' },
    memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB }, disk: { freeBytes: 400 * GIB }, cpus: 16 });

test('an image that can never run on unified memory keeps vLLM as before while the GPU cannot be read', async (t) => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-amd64-'));
    t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
    const h = harness(t, { installer: vllmInstaller(runDir), imageContract: AMD64_CONTRACT, snapshot: GPU_UNREADABLE });
    const overview = await h.controller.overview();
    const row = overview.runners.find((runner) => runner.id === 'vllm');
    assert.equal(row.enabled, true, 'as at 50f39385: no switch on a dedicated-only image');
    assert.equal(Object.hasOwn(row, 'disabledReason'), false);
    // The install is not refused by the switch (here the stub reports it already installed and runnable).
    const install = await h.controller.installRunner({ runnerId: 'vllm' });
    assert.equal(install.installed, true);
});

test('an image that can run on unified memory keeps vLLM off while the GPU cannot be read, and says what turns it on', async (t) => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-arm64-'));
    t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
    for (const imageContract of [ARM64_CONTRACT, null]) {
        const h = harness(t, { installer: vllmInstaller(runDir), imageContract, snapshot: GPU_UNREADABLE });
        const row = (await h.controller.overview()).runners.find((runner) => runner.id === 'vllm');
        assert.equal(row.enabled, false);
        assert.match(row.disabledReason, new RegExp(`the GPU could not be read to tell\\. The operator can turn it on with ploinky var ${VLLM_UNIFIED_SWITCH} experimental`));
        assert.doesNotMatch(row.disabledReason, /try again|open the overview/i, 'no retry that cannot succeed');
        await assert.rejects(h.controller.installRunner({ runnerId: 'vllm' }), (error) => error.code === 'runner_disabled');
        // With the operator's switch it is not refused by the gate.
        const on = harness(t, { installer: vllmInstaller(runDir), imageContract, snapshot: GPU_UNREADABLE, env: { [VLLM_UNIFIED_SWITCH]: 'experimental' } });
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
