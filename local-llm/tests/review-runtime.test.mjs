// Runtime and envelope-authority regressions: one committed hardware profile
// for normalization, admission, launch and the guard; a last admission from
// fresh resources immediately before the runner starts; unified envelopes and
// `validated` labels only from the trusted seed; and envelopes keyed on the
// load mode they were measured with (DS001, DS003, DS005).
//
// Hardware, downloads, verification and runners are injected: no /proc, GPU,
// network or real model is used, so these run on macOS and in Linux alike.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { UNIFIED, UNIFIED_LOAD_MODES, envelopeFor } from '../src/controller/profiles.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { getRunner } from '../src/runners/index.mjs';
import { LLAMA_SERVER_UNIFIED_PARAM_SCHEMA } from '../src/runners/llamaServer.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const TOTAL = 120 * GIB;
const GB10 = Object.freeze({
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
});
const COLD = Object.freeze({ available: false, name: 'NVIDIA GB10', reason: 'nvidia-smi failed: timeout' });

function gguf(file = 'fixture.gguf') {
    return { type: 'huggingface', repo: 'review/fixture', file, revision: 'main', commit: 'a'.repeat(40), size: 1, sha256: 'b'.repeat(64) };
}

function rectangle(fields = {}) {
    return { runner: 'llama.cpp', loadMode: 'dio', maxCtx: 131072, maxParallel: 4, mtp: false, bufferBytes: 60 * GIB, transientBytes: 4 * GIB, ...fields };
}

// A trusted seed entry (shipped or operator catalog) with a measured envelope.
// Synthetic numbers: no envelope, margin or constant is approved by these tests.
function trustedModel({ id = 'trusted-moe', profiles = ['unified'], envelope = [rectangle()] } = {}) {
    return validateModel({
        id, displayName: id, architecture: 'moe', contextLength: 131072, profiles, sources: { gguf: gguf() },
        unified: { envelope },
    }, { seed: true });
}

function unifiedNeed(entry) {
    return entry.bufferBytes + entry.transientBytes + UNIFIED.runtimeBytes + UNIFIED.cacheRamMiB * MIB;
}

function unifiedSnap(available = 108 * GIB) {
    return { gpu: structuredClone(GB10), memory: { totalBytes: TOTAL, availableBytes: available }, disk: { freeBytes: 400 * GIB }, cpus: 20 };
}

function dedicatedSnap(freeBytes = 6 * GIB) {
    return {
        gpu: { available: true, name: 'RTX', memoryModel: 'dedicated', totalBytes: 6 * GIB, freeBytes, usedBytes: 6 * GIB - freeBytes, processes: [] },
        memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB },
        disk: { freeBytes: 400 * GIB },
        cpus: 16,
    };
}

function deferred() {
    let resolve;
    const promise = new Promise((ok) => { resolve = ok; });
    return { promise, resolve };
}

async function until(predicate, label, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 3));
    }
    assert.fail(`timed out waiting for ${label}`);
}

/**
 * A controller with injected hardware. `snap(read)` gives the n-th snapshot
 * (it may throw); `onVerify` runs inside the final verification of the weights.
 */
function harness(t, { seed, snap, onVerify = async () => {}, readMemory, dataDir, env = {}, resolveHf } = {}) {
    const dir = dataDir || fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-review-runtime-'));
    if (!dataDir) t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const weights = path.join(dir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const events = [];
    const started = [];
    const calls = { download: 0, resolveHf: 0, snapshots: 0 };
    const controller = createController({
        dataDir: dir,
        env: { PATH: '/usr/bin', ...env },
        ...(seed ? { seedCatalog: seed } : {}),
        stateStore: createStateStore({ dataDir: dir }),
        snapshot: async () => {
            calls.snapshots += 1;
            events.push('snapshot');
            return snap(calls.snapshots);
        },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async () => { calls.download += 1; return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        verify: async () => { events.push('verify'); await onVerify(); return { notes: [] }; },
        remove: async () => 0,
        resolveHf: resolveHf || (async () => { calls.resolveHf += 1; return { commit: 'c'.repeat(40), size: 1, sha256: 'd'.repeat(64) }; }),
        sharedModelsRoot: null,
        imageContract: null,
        installer: { installable: () => false },
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner({ command, args, env: runnerEnv }) {
            events.push('launch');
            const exit = deferred();
            let running = true;
            const handle = {
                pid: 7000 + started.length, command, args, env: runnerEnv, exited: exit.promise,
                get running() { return running; },
                async stop() { running = false; handle.stopped = 'SIGTERM'; exit.resolve({ code: 0, signal: 'SIGTERM', error: null }); return exit.promise; },
                async kill() { running = false; handle.stopped = 'SIGKILL'; exit.resolve({ code: null, signal: 'SIGKILL', error: null }); return exit.promise; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        pollMs: 1,
        stopGraceMs: 20,
        readMemory: readMemory || (() => ({ totalBytes: TOTAL, availableBytes: 100 * GIB })),
        readPressure: () => 0,
        unifiedGuardMs: 5,
        dropCache: () => true,
    });
    t.after(() => controller.drain());
    return { controller, events, started, calls, weights, dataDir: dir };
}

function settled(h) {
    return until(() => ['ready', 'error', 'idle'].includes(h.controller.state.deployment?.phase) && h.controller.state.deployment.phase !== 'downloading',
        'the deployment to settle');
}

// ------------------------------------------------------------ one profile

for (const [label, firstSnapshot] of [
    ['reports the GPU unavailable', () => ({ gpu: structuredClone(COLD), memory: { totalBytes: TOTAL, availableBytes: 108 * GIB }, disk: { freeBytes: 400 * GIB } })],
    ['cannot be read at all', () => { throw new Error('nvidia-smi: cold start'); }],
]) {
    test(`a Run is refused while the first snapshot ${label}, and its retry launches, admits and guards under the unified profile`, async (t) => {
        // Offered in both profiles: an undecided profile must not fall back to the dedicated rules for it.
        const model = trustedModel({ profiles: ['dedicated', 'unified'] });
        let available = 108 * GIB;
        const h = harness(t, {
            seed: [model],
            snap: (read) => (read === 1 ? firstSnapshot() : unifiedSnap(available)),
            readMemory: () => ({ totalBytes: TOTAL, availableBytes: available }),
        });
        await assert.rejects(
            () => h.controller.run({ requestId: 'request-cold-01', modelId: model.id, runnerId: 'llama.cpp' }),
            (error) => error.code === 'admission_incompatible' && /hardware profile is not decided; run again once the GPU can be read/.test(error.message)
                && (label !== 'reports the GPU unavailable' || error.message.startsWith('nvidia-smi failed: timeout. ')),
        );
        // Nothing was recorded or started, so the same request may be sent again.
        assert.equal(h.controller.state.deployment, null);
        assert.deepEqual(Object.keys(h.controller.state.requests), []);
        assert.deepEqual(h.controller.state.params, {});
        assert.equal(h.calls.download, 0);
        assert.equal(h.started.length, 0);

        const accepted = await h.controller.run({ requestId: 'request-cold-01', modelId: model.id, runnerId: 'llama.cpp' });
        assert.equal(accepted.accepted, true);
        await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
        const deployment = h.controller.state.deployment;
        assert.equal(deployment.profile, 'unified');
        // Normalized with the unified schema, admitted by the measured envelope, launched with its fixed flags.
        assert.deepEqual(deployment.params, { ctxSize: 32768, parallel: 1, loadMode: 'dio', mtp: false, threads: null });
        assert.equal(deployment.admission.status, 'ok');
        assert.equal(deployment.admission.estimate.unifiedBytes, unifiedNeed(model.unified.envelope[0]));
        const { args } = h.started[0];
        for (const [flag, value] of [['--cache-ram', String(UNIFIED.cacheRamMiB)], ['--load-mode', 'dio'], ['--n-gpu-layers', '999'], ['--flash-attn', 'on']]) {
            assert.equal(args[args.indexOf(flag) + 1], value, flag);
        }
        // The unified guard is installed and stops the runner below the floor.
        await until(async () => (await h.controller.status()).memoryGuard?.samples >= 2, 'guard samples');
        const status = await h.controller.status();
        assert.equal(status.profile, 'unified');
        assert.equal(status.memoryGuard.floorBytes, UNIFIED.floorBytes);
        available = GIB;
        await until(() => h.controller.state.deployment?.phase === 'error', 'guard stop');
        assert.equal(h.started[0].stopped, 'SIGKILL');
        assert.match(h.controller.state.deployment.error, /stopped: host memory below the floor/);
    });
}

test('the overview normalizes and admits under the profile of the one snapshot it reads, and answers while it is undecided', async (t) => {
    // Offered in both profiles, so an undecided (dedicated-default) catalog lists it too.
    const model = trustedModel({ id: 'both-moe', profiles: ['dedicated', 'unified'] });
    const h = harness(t, { seed: [model], snap: (read) => (read === 1 ? { gpu: structuredClone(COLD), memory: { totalBytes: TOTAL, availableBytes: 108 * GIB }, disk: { freeBytes: 400 * GIB } } : unifiedSnap()) });
    const cold = await h.controller.overview();
    assert.equal(cold.profile, null);
    const coldRow = cold.models.find((entry) => entry.id === model.id).runners['llama.cpp'];
    // Dedicated-default parameters are never admitted by the unified rules of another snapshot.
    assert.equal(coldRow.params.loadMode, 'auto');
    assert.equal(coldRow.admission.status, 'incompatible');
    assert.equal(coldRow.admission.reason, COLD.reason);
    const warm = await h.controller.overview();
    assert.equal(warm.profile, 'unified');
    const warmRow = warm.models.find((entry) => entry.id === model.id).runners['llama.cpp'];
    assert.equal(warmRow.params.loadMode, 'dio');
    assert.equal(warmRow.admission.status, 'ok', warmRow.admission.reason);
    assert.equal(h.calls.snapshots, 2, 'each overview reads exactly one snapshot');
});

// ----------------------------------------- the last admission before launch

function assertAdmittedAfterVerify(events) {
    const verify = events.lastIndexOf('verify');
    assert.ok(verify >= 0, 'the weights were verified');
    const launch = events.indexOf('launch');
    const after = events.slice(verify + 1, launch < 0 ? undefined : launch);
    assert.ok(after.includes('snapshot'), `a fresh snapshot is read after the final verification (events: ${events.join(', ')})`);
}

test('dedicated: free GPU memory taken during the final verification refuses the start, and nothing is launched', async (t) => {
    const [gpt] = loadSeedCatalog();
    assert.equal(gpt.id, 'gpt-oss-20b');
    let free = 6 * GIB;
    const h = harness(t, { snap: () => dedicatedSnap(free), onVerify: async () => { free = 128 * MIB; } });
    await h.controller.run({ requestId: 'request-dedicated-drop', modelId: gpt.id, runnerId: 'llama.cpp' });
    await settled(h);
    const deployment = h.controller.state.deployment;
    assert.equal(deployment.phase, 'error');
    assert.equal(deployment.profile, 'dedicated');
    assert.equal(deployment.admission.status, 'insufficient-now');
    assert.ok(deployment.admission.estimate.gpuBytes > free);
    assert.match(deployment.error, /GPU memory; 0\.1 GiB is free now/);
    assert.equal(h.started.length, 0, 'no runner is started after the refusal');
    assertAdmittedAfterVerify(h.events);
});

test('dedicated: with nothing taken meanwhile the start is unchanged', async (t) => {
    const [gpt] = loadSeedCatalog();
    const h = harness(t, { snap: () => dedicatedSnap(6 * GIB) });
    await h.controller.run({ requestId: 'request-dedicated-keep', modelId: gpt.id, runnerId: 'llama.cpp' });
    await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
    const deployment = h.controller.state.deployment;
    assert.equal(h.started.length, 1);
    const expected = getRunner('llama.cpp').buildLaunch({
        artifactPath: h.weights, params: deployment.params, port: 18080, apiKey: KEY, model: gpt, profile: 'dedicated',
    });
    assert.deepEqual(h.started[0].args, expected.args);
    assert.ok(!h.started[0].args.includes('--cache-ram'));
    assertAdmittedAfterVerify(h.events);
});

for (const [label, drop, expected] of [
    ['to 16 GiB (above the guard floor) refuses the start', () => 16 * GIB, 'error'],
    ['to exactly the need plus the floor still starts', (need) => need + UNIFIED.floorBytes, 'ready'],
    ['to one byte less than the need plus the floor refuses the start', (need) => need + UNIFIED.floorBytes - 1, 'error'],
]) {
    test(`unified: available memory falling during the final verification ${label}`, async (t) => {
        const model = trustedModel();
        const need = unifiedNeed(model.unified.envelope[0]);
        let available = 108 * GIB;
        const h = harness(t, {
            seed: [model],
            snap: () => unifiedSnap(available),
            onVerify: async () => { available = drop(need); },
            readMemory: () => ({ totalBytes: TOTAL, availableBytes: available }),
        });
        await h.controller.run({ requestId: 'request-unified-drop', modelId: model.id, runnerId: 'llama.cpp' });
        await settled(h);
        const deployment = h.controller.state.deployment;
        assert.equal(deployment.phase, expected, deployment.error ?? '');
        if (expected === 'error') {
            assert.equal(deployment.admission.status, 'insufficient-now');
            assert.match(deployment.error, /Other processes on this machine hold the rest/);
            assert.equal(h.started.length, 0, 'no runner is started after the refusal');
        } else {
            assert.equal(h.started.length, 1);
            assert.ok(h.started[0].args.includes('--cache-ram'));
        }
        assertAdmittedAfterVerify(h.events);
    });
}

test('a Stop during the final verification leaves nothing launched', async (t) => {
    const model = trustedModel();
    const inVerify = deferred();
    const release = deferred();
    const h = harness(t, {
        seed: [model],
        snap: () => unifiedSnap(),
        onVerify: async () => { inVerify.resolve(); await release.promise; },
    });
    await h.controller.run({ requestId: 'request-stop-verify', modelId: model.id, runnerId: 'llama.cpp' });
    await inVerify.promise;
    const stopping = h.controller.stop();
    release.resolve();
    await stopping;
    assert.equal(h.controller.state.deployment.phase, 'idle');
    assert.equal(h.started.length, 0);
});

// ------------------------------------------------------ envelope authority

test('model add refuses a unified envelope or validated labels before anything is resolved', async (t) => {
    const h = harness(t, { snap: () => unifiedSnap() });
    const unpinned = { type: 'huggingface', repo: 'someone/model', file: 'model.gguf', revision: 'main' };
    for (const [extra, field] of [
        [{ unified: { envelope: [rectangle({ bufferBytes: 1, transientBytes: 0 })] } }, 'unified'],
        [{ validated: { unified: { 'llama.cpp': 'measured by me' } } }, 'validated.unified'],
        [{ validated: { dedicated: { 'llama.cpp': 'measured by me' } } }, 'validated.dedicated'],
    ]) {
        await assert.rejects(() => h.controller.addModel({ id: 'mine', profiles: ['unified'], sources: { gguf: unpinned }, ...extra }),
            (error) => error.code === 'invalid_model' && error.details.field === field
                && /only the trusted seed catalog \(the shipped catalog or the operator's LOCAL_LLM_CATALOG_FILE\) may provide it/.test(error.message));
    }
    assert.equal(h.calls.resolveHf, 0, 'refused before any Hugging Face call');
    assert.deepEqual(h.controller.state.registry, []);
    // The empty forms a stored entry carries are not certification.
    const { model } = await h.controller.addModel({ id: 'mine', sources: { gguf: unpinned }, unified: null, validated: {} });
    assert.equal(model.unified, null);
    assert.deepEqual(model.validated, {});
    assert.equal(h.calls.resolveHf, 1);
});

test('a stored user model survives a restart with its round-tripped empty fields', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-review-runtime-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const first = harness(t, { dataDir, snap: () => unifiedSnap() });
    await first.controller.addModel({ id: 'mine', profiles: ['unified'], sources: { gguf: gguf('mine.gguf') } });
    await first.controller.drain();
    const stored = JSON.parse(fs.readFileSync(path.join(dataDir, 'state', 'controller.json'), 'utf8')).registry;
    assert.equal(stored[0].unified, null);
    assert.deepEqual(stored[0].validated, {});
    const second = harness(t, { dataDir, snap: () => unifiedSnap() });
    const overview = await second.controller.overview();
    assert.ok(overview.models.some((entry) => entry.id === 'mine'));
    assert.deepEqual(overview.unsupportedModels, []);
});

test('model update refuses to add an envelope or labels and leaves the stored entry unchanged', async (t) => {
    const h = harness(t, { snap: () => unifiedSnap() });
    await h.controller.addModel({ id: 'mine', profiles: ['unified'], sources: { gguf: gguf('mine.gguf') } });
    const before = structuredClone(h.controller.state.registry);
    for (const extra of [
        { unified: { envelope: [rectangle()] } },
        { validated: { unified: { 'llama.cpp': 'measured by me' } } },
    ]) {
        await assert.rejects(() => h.controller.updateModel({ id: 'mine', profiles: ['unified'], sources: { gguf: gguf('mine.gguf') }, ...extra }),
            { code: 'invalid_model' });
    }
    assert.deepEqual(h.controller.state.registry, before);
    // An ordinary update still works.
    const { model } = await h.controller.updateModel({ id: 'mine', displayName: 'Mine', profiles: ['unified'], sources: { gguf: gguf('mine.gguf') } });
    assert.equal(model.displayName, 'Mine');
});

test('a user alias of a trusted seed\'s file does not inherit its envelope; the seed keeps it', async (t) => {
    const seed = trustedModel({ id: 'seed-moe' });
    const h = harness(t, { seed: [seed], snap: () => unifiedSnap() });
    await h.controller.addModel({ id: 'alias-moe', profiles: ['unified'], sources: { gguf: { ...seed.sources.gguf } } });
    const overview = await h.controller.overview();
    const rows = Object.fromEntries(overview.models.map((entry) => [entry.id, entry.runners['llama.cpp'].admission]));
    assert.equal(rows['seed-moe'].status, 'ok', rows['seed-moe'].reason);
    assert.equal(rows['alias-moe'].status, 'incompatible');
    assert.match(rows['alias-moe'].reason, /alias-moe has not been measured on unified memory with llama\.cpp/);
    await assert.rejects(() => h.controller.run({ requestId: 'request-alias-01', modelId: 'alias-moe', runnerId: 'llama.cpp' }),
        { code: 'admission_incompatible' });
    assert.equal(h.started.length, 0);
    assert.equal(h.calls.download, 0);
});

test('the operator catalog (LOCAL_LLM_CATALOG_FILE) is trusted: its envelope and labels are kept and admit', async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-review-catalog-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, 'models.json');
    fs.writeFileSync(file, JSON.stringify({
        schema: 'local-llm.catalog/v3',
        models: [{
            id: 'operator-moe', architecture: 'moe', contextLength: 131072, profiles: ['unified'], sources: { gguf: gguf() },
            validated: { unified: { 'llama.cpp': 'operator measurement record' } },
            unified: { envelope: [rectangle({ measured: 'operator fixture' })] },
        }],
    }));
    const [entry] = loadSeedCatalog(file);
    assert.equal(entry.seed, true);
    assert.equal(entry.unified.envelope[0].loadMode, 'dio');
    const h = harness(t, { env: { LOCAL_LLM_CATALOG_FILE: file }, snap: () => unifiedSnap() });
    const overview = await h.controller.overview();
    const row = overview.models.find((model) => model.id === 'operator-moe');
    assert.deepEqual(row.validated, { 'llama.cpp': 'operator measurement record' });
    assert.equal(row.runners['llama.cpp'].admission.status, 'ok', row.runners['llama.cpp'].admission.reason);
});

test('a stored user entry that carries an envelope is kept on disk, reported, never offered, and removable', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-review-runtime-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const certified = {
        id: 'self-certified', profiles: ['unified'], sources: { gguf: gguf('self.gguf') }, validated: {},
        unified: { envelope: [rectangle({ bufferBytes: 1, transientBytes: 0 })] },
    };
    createStateStore({ dataDir }).save({ version: 1, registry: [certified], requests: {}, params: {}, deployment: null });
    const h = harness(t, { dataDir, snap: () => unifiedSnap() });
    const overview = await h.controller.overview();
    assert.ok(!overview.models.some((entry) => entry.id === 'self-certified'));
    assert.equal(overview.unsupportedModels.length, 1);
    assert.equal(overview.unsupportedModels[0].id, 'self-certified');
    assert.match(overview.unsupportedModels[0].reason, /unified is a measured envelope; only the trusted seed catalog/);
    const logs = (await h.controller.status()).logs.map((entry) => entry.line);
    assert.ok(logs.some((line) => /model entry self-certified is not supported by this catalog and is not offered/.test(line)));
    // Not migrated or deleted: the state file still holds it as it was.
    const stored = JSON.parse(fs.readFileSync(path.join(dataDir, 'state', 'controller.json'), 'utf8')).registry;
    assert.deepEqual(stored, [certified]);
    await assert.rejects(() => h.controller.run({ requestId: 'request-self-01', modelId: 'self-certified', runnerId: 'llama.cpp' }),
        { code: 'unknown_model' });
    assert.equal(h.started.length, 0);
    assert.equal((await h.controller.removeModel({ modelId: 'self-certified' })).removed, 'self-certified');
});

test('only the seed loader validates with seed trust', () => {
    const root = path.join(import.meta.dirname, '..', 'src');
    const hits = [];
    const walk = (dir) => {
        for (const name of fs.readdirSync(dir)) {
            const file = path.join(dir, name);
            if (fs.statSync(file).isDirectory()) walk(file);
            else if (name.endsWith('.mjs')) {
                for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
                    if (/seed:\s*true/.test(line)) hits.push(`${path.relative(root, file)}: ${line.trim()}`);
                }
            }
        }
    };
    walk(root);
    assert.deepEqual(hits, ['controller/catalog.mjs: const models = document.models.map((entry) => validateModel(entry, { seed: true }));']);
});

// --------------------------------------------------------------- load mode

function decide(model, params) {
    const runner = getRunner('llama.cpp');
    return admit({
        runner, model, source: model.sources.gguf, snapshot: unifiedSnap(), profile: 'unified',
        params: runner.normalizeParams(params, { model, profile: 'unified' }),
    });
}

test('an envelope measured with dio only refuses none, naming the load mode', () => {
    const model = trustedModel();
    const dio = decide(model, { loadMode: 'dio' });
    assert.equal(dio.status, 'ok', dio.reason);
    assert.equal(dio.estimate.envelope.loadMode, 'dio');
    const none = decide(model, { loadMode: 'none' });
    assert.equal(none.status, 'incompatible');
    assert.match(none.reason, /only up to context 131072 x 4 slots with load mode dio; context 32768 x 1 slot with load mode none is outside it/);
    assert.equal(envelopeFor(model, 'llama.cpp', { ctxSize: 512, parallel: 1, mtp: false, loadMode: 'none' }), null);
    assert.equal(envelopeFor(model, 'llama.cpp', { ctxSize: 512, parallel: 1, mtp: false }), null, 'no mode is no match');
});

test('modes measured separately are rectangles of their own, each admitting only its mode with its own figures', () => {
    // `none` is listed first and is larger: a dio request must still take the dio rectangle.
    const none = rectangle({ loadMode: 'none', bufferBytes: 60 * GIB, transientBytes: 20 * GIB });
    const dio = rectangle({ loadMode: 'dio', maxCtx: 65536, bufferBytes: 58 * GIB, transientBytes: 4 * GIB });
    const model = trustedModel({ envelope: [none, dio] });
    const byMode = Object.fromEntries(['dio', 'none'].map((loadMode) => [loadMode, decide(model, { loadMode, ctxSize: 65536 })]));
    assert.equal(byMode.dio.status, 'ok', byMode.dio.reason);
    assert.equal(byMode.none.status, 'ok', byMode.none.reason);
    assert.equal(byMode.dio.estimate.unifiedBytes, unifiedNeed(dio));
    assert.equal(byMode.none.estimate.unifiedBytes, unifiedNeed(none));
    assert.notEqual(byMode.dio.estimate.unifiedBytes, byMode.none.estimate.unifiedBytes);
    // Each rectangle bounds only its own mode: dio was measured only up to 64k.
    assert.equal(decide(model, { loadMode: 'dio', ctxSize: 131072 }).status, 'incompatible');
    assert.equal(decide(model, { loadMode: 'none', ctxSize: 131072 }).status, 'ok');
});

test('a Run asking for an unmeasured load mode is refused before anything is downloaded', async (t) => {
    const model = trustedModel();
    const h = harness(t, { seed: [model], snap: () => unifiedSnap() });
    await assert.rejects(() => h.controller.run({ requestId: 'request-none-01', modelId: model.id, runnerId: 'llama.cpp', params: { loadMode: 'none' } }),
        (error) => error.code === 'admission_incompatible' && /with load mode none is outside it/.test(error.message));
    assert.equal(h.calls.download, 0);
    assert.equal(h.started.length, 0);
});

test('the envelope\'s load modes are exactly the unified parameter\'s, in the validator and the published schema', () => {
    assert.deepEqual([...UNIFIED_LOAD_MODES], LLAMA_SERVER_UNIFIED_PARAM_SCHEMA.properties.loadMode.enum);
    const schema = JSON.parse(fs.readFileSync(new URL('../catalog/schema.json', import.meta.url), 'utf8'));
    const item = schema.$defs.model.properties.unified.properties.envelope.items;
    assert.ok(item.required.includes('loadMode'));
    assert.deepEqual(item.properties.loadMode.enum, [...UNIFIED_LOAD_MODES]);
});
