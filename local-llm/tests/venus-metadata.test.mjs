import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { loadSeedCatalog, validateModel, mergeCatalog, unsupportedRegistryEntries } from '../src/controller/catalog.mjs';
import { exportModelMetadata, measurementStatus } from '../src/controller/venusMetadata.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { getRunner } from '../src/runners/index.mjs';
import { admit } from '../src/controller/admission.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { loadSeedCatalog as loadV3SeedCatalog, mergeCatalog as mergeV3Catalog, validateModel as validateV3Model } from './fixtures/catalog-v3/src/controller/catalog.mjs';
import { createStateStore as createV3StateStore } from './fixtures/catalog-v3/src/controller/stateStore.mjs';

const source = { type: 'huggingface', repo: 'test/model', file: 'model.gguf', commit: 'a'.repeat(40), revision: 'main', size: 400000000, sha256: 'b'.repeat(64) };
const params = { ctxSize: 2048, parallel: 1, loadMode: 'none' };
const record = { schema: 'local-llm.venus-measurement/v1', id: 'trial-01', measuredAt: '2026-10-13T10:00:00.000Z',
    evidence: { runId: 'run.1', manifestSha256: 'c'.repeat(64), reference: 'offline fixture' }, modelSource: source,
    configuration: { params, resolvedThreads: 4, hostChip: 'M1', hostRamBytes: 16 * 2 ** 30, vmCpus: 4, vmRamBytes: 6 * 2 ** 30 } };
const model = () => ({ id: 'test-model', sources: { gguf: source }, recommended: { venus: { 'llama.cpp-vulkan': params } }, validated: { venus: { 'llama.cpp-vulkan': [structuredClone(record)] } } });

test('v4 Venus history is seed-only and never grants eligibility or alias certification', () => {
    const entry = validateModel(model(), { seed: true });
    assert.deepEqual(entry.profiles, ['dedicated', 'unified', 'cpu']);
    assert.throws(() => validateModel({ ...model(), seed: true }), /trusted seed/);
    assert.equal(mergeCatalog([], [{ ...model(), seed: true }]).length, 0);
    assert.equal(unsupportedRegistryEntries([model()]).length, 1);
    assert.throws(() => validateModel({ ...model(), profiles: ['venus'] }, { seed: true }), /profiles/);
    assert.equal(measurementStatus({ ...entry, seed: false }, 'llama.cpp-vulkan', params).state, 'unverified');
    assert.equal(measurementStatus(entry, 'llama.cpp-vulkan', params).applicability, 'unknown');
    assert.equal(measurementStatus(entry, 'llama.cpp-vulkan', { ...params, ctxSize: 4096 }).applicability, 'mismatch');
});

test('strict bounded records reject malformed timestamps, fields, identity, and duplicates', () => {
    for (const mutate of [
        (v) => { v.extra = true; }, (v) => { v.measuredAt = '2026-02-30T10:00:00.000Z'; },
        (v) => { v.configuration.vmCpus = 1.5; }, (v) => { v.configuration.hostRamBytes = Number.MAX_SAFE_INTEGER + 1; },
        (v) => { v.evidence.reference = '\n'; }, (v) => { v.modelSource.sha256 = 'd'.repeat(64); },
        (v) => { v.configuration.params.threads = 4; }, (v) => { v.evidence.trusted = true; },
    ]) {
        const value = model(); mutate(value.validated.venus['llama.cpp-vulkan'][0]);
        assert.throws(() => validateModel(value, { seed: true }), { code: 'invalid_model' });
    }
    for (const count of [0, 2, 17]) {
        const value = model(); value.validated.venus['llama.cpp-vulkan'] = Array(count).fill(record);
        assert.throws(() => validateModel(value, { seed: true }), { code: 'invalid_model' });
    }
    const empty = model(); empty.validated.venus = {}; delete empty.recommended;
    assert.doesNotThrow(() => validateModel(empty));
});

test('v3 remains readable, cannot smuggle Venus, and lossy exports are explicit', (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'venus-metadata-')); t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, 'catalog.json');
    fs.writeFileSync(file, JSON.stringify({ schema: 'local-llm.catalog/v3', models: [{ id: 'legacy', sources: { gguf: source } }] }));
    assert.equal(loadSeedCatalog(file).length, 1);
    fs.writeFileSync(file, JSON.stringify({ schema: 'local-llm.catalog/v3', models: [model()] }));
    assert.throws(() => loadSeedCatalog(file), /unknown profile/);
    assert.throws(() => exportModelMetadata([model()], { seed: true, targetSchema: 'local-llm.catalog/v3' }), { code: 'catalog_downgrade_lossy' });
    const out = exportModelMetadata([model()], { seed: true, targetSchema: 'local-llm.catalog/v3', dropVenusMetadata: true });
    assert.deepEqual(out.dropped, [{ modelId: 'test-model', field: 'recommended.venus' }, { modelId: 'test-model', field: 'validated.venus' }]);
    assert.equal(out.models[0].seed, undefined);
    assert.deepEqual(out.models[0].sources.gguf, source);
    assert.ok(model().validated.venus);
});

test('captured v3 reader hides valid Venus recommendations, preserves raw state, and restores visibility after upgrade', (t) => {
    const root = new URL('./fixtures/catalog-v3/', import.meta.url);
    const provenance = JSON.parse(fs.readFileSync(new URL('provenance.json', root), 'utf8'));
    assert.equal(provenance.revision, 'c70ecd013793b919fd9a51ce8ae5001a2fd1dae9');
    for (const entry of provenance.files) {
        const bytes = fs.readFileSync(new URL(entry.path, root));
        assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'), entry.sha256, entry.path);
        if (entry.capture === 'whole-file-unmodified') assert.equal(entry.sha256, entry.sourceSha256);
    }
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'venus-v3-roundtrip-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const user = { id: 'user-venus', sources: { gguf: source }, profiles: ['cpu'],
        recommended: { venus: { 'llama.cpp-vulkan': params }, cpu: { 'llama.cpp': { ctxSize: 1024 } } } };
    const store = createStateStore({ dataDir: dir }); const state = store.load(); state.registry = [user];
    state.cpuProfileMigration = '2026-10-01'; store.save(state);
    assert.deepEqual(mergeCatalog([], state.registry).map((entry) => entry.id), ['user-venus']);
    assert.equal(mergeCatalog([], state.registry)[0].seed, false);
    const rawBefore = JSON.stringify(state.registry);
    const oldStore = createV3StateStore({ dataDir: dir }); const oldState = oldStore.load();
    assert.throws(() => validateV3Model(oldState.registry[0]), /unknown profile 'venus'/);
    assert.deepEqual(mergeV3Catalog([], oldState.registry), []);
    oldState.params.unrelated = { ctxSize: 4096 }; oldStore.save(oldState);
    assert.equal(JSON.stringify(JSON.parse(fs.readFileSync(oldStore.file, 'utf8')).registry), rawBefore);
    assert.equal(JSON.stringify(oldStore.load().registry), rawBefore);
    const upgraded = store.load();
    assert.equal(JSON.stringify(upgraded.registry), rawBefore);
    assert.deepEqual(mergeCatalog([], upgraded.registry).map((entry) => entry.id), ['user-venus']);
    assert.deepEqual(mergeCatalog([], upgraded.registry)[0].recommended.venus['llama.cpp-vulkan'], params);
    assert.deepEqual(upgraded.params.unrelated, { ctxSize: 4096 });

    const file = path.join(dir, 'catalog.json');
    const writeCatalog = ({ schema, models }) => fs.writeFileSync(file, JSON.stringify({ schema, models }));
    writeCatalog(exportModelMetadata([user]));
    assert.throws(() => loadV3SeedCatalog(file), /Unsupported catalog/);
    const exported = exportModelMetadata([user], { targetSchema: 'local-llm.catalog/v3', dropVenusMetadata: true });
    assert.deepEqual(exported.dropped, [{ modelId: 'user-venus', field: 'recommended.venus' }]);
    writeCatalog(exported);
    const consumed = loadV3SeedCatalog(file);
    assert.deepEqual(consumed[0].sources.gguf, source);
    assert.deepEqual(consumed[0].profiles, ['cpu']);
    assert.deepEqual(consumed[0].recommended, { cpu: { 'llama.cpp': { ctxSize: 1024 } } });
    assert.deepEqual(mergeV3Catalog([], exported.models).map((entry) => entry.id), ['user-venus']);
});

test('Venus defaults reach normalization and history never widens admission', () => {
    const entry = validateModel(model(), { seed: true }); const runner = getRunner('llama.cpp-vulkan');
    assert.deepEqual(runner.normalizeParams({}, { model: entry, profile: 'venus' }), params);
    assert.equal(runner.normalizeParams({ ctxSize: 4096 }, { model: entry, profile: 'venus' }).ctxSize, 4096);
    const normalized = runner.normalizeParams({ ctxSize: 131072, parallel: 4 }, { model: entry, profile: 'venus' });
    const result = admit({ runner, model: entry, source, params: normalized, profile: 'venus',
        snapshot: { gpu: { available: true, backend: 'vulkan', driver: 'venus', memoryModel: 'venus', totalBytes: 8 * 2 ** 30, freeBytes: 8 * 2 ** 30 }, memory: { totalBytes: 16 * 2 ** 30, availableBytes: 15 * 2 ** 30 }, disk: { freeBytes: 1e12 } } });
    assert.equal(result.status, 'incompatible'); assert.equal(result.estimate.measured, false);
});

test('measurement source equality includes ordered split shards, and applicability compares records independently', () => {
    const shards = [1, 2].map((index) => ({ file: `model-0000${index}-of-00002.gguf`, size: 200000000, sha256: String(index).repeat(64) }));
    const split = { ...source, file: shards[0].file, sha256: undefined, shards };
    const value = model(); value.sources.gguf = split; value.validated.venus['llama.cpp-vulkan'][0].modelSource = structuredClone(split);
    assert.doesNotThrow(() => validateModel(value, { seed: true }));
    value.validated.venus['llama.cpp-vulkan'][0].modelSource.shards[1].sha256 = 'd'.repeat(64);
    assert.throws(() => validateModel(value, { seed: true }), /match the enclosing/);
    const oversized = model(); const huge = { ...split, size: 10000000000000000, shards: shards.map((shard) => ({ ...shard, size: 5000000000000000 })) };
    oversized.sources.gguf = huge; oversized.validated.venus['llama.cpp-vulkan'][0].modelSource = structuredClone(huge);
    assert.throws(() => validateModel(oversized, { seed: true }), /modelSource.size/);
    const multiple = model(); const next = structuredClone(record); next.id = 'trial-02'; next.configuration.params.ctxSize = 4096;
    multiple.validated.venus['llama.cpp-vulkan'].push(next);
    const result = measurementStatus(validateModel(multiple, { seed: true }), 'llama.cpp-vulkan', params);
    assert.equal(result.applicability, 'unknown'); assert.deepEqual(result.records.map((record) => record.applicability), ['unknown', 'mismatch']);
});

test('Add and Update reject forged history before any pinning request', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'venus-trust-')); let pins = 0;
    const controller = createController({ dataDir, seedCatalog: [], imageContract: null, installer: { installable: () => false },
        sharedModelsRoot: null, resolveHf: async () => { pins++; throw new Error('must not pin'); } });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
    const forged = { ...model(), seed: true };
    await assert.rejects(() => controller.addModel(forged), { code: 'invalid_model' });
    controller.state.registry.push({ id: forged.id, sources: { gguf: source } });
    await assert.rejects(() => controller.updateModel(forged), { code: 'invalid_model' });
    assert.equal(pins, 0);
});
