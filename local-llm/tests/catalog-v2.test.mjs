// Catalog schema v2: model sources are keyed by weight format (gguf, ollama),
// not by runner, so runners that read the same format share one download.
// Registries persisted by the v1 controller are migrated when state loads.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { WEIGHT_FORMATS, loadSeedCatalog, mergeCatalog, migrateModelEntry, validateModel } from '../src/controller/catalog.mjs';
import { STATE_VERSION, createStateStore } from '../src/controller/stateStore.mjs';

const GGUF = Object.freeze({
    type: 'huggingface', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', revision: 'main',
    commit: 'a'.repeat(40), size: 639446688, sha256: 'b'.repeat(64),
});
const OLLAMA = Object.freeze({ type: 'ollama', tag: 'qwen3:0.6b' });

function tempStore(t, content) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-catalog-v2-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const store = createStateStore({ dataDir });
    if (content !== undefined) {
        fs.mkdirSync(path.dirname(store.file), { recursive: true });
        fs.writeFileSync(store.file, JSON.stringify(content));
    }
    return store;
}

test('the seed catalog is schema v2 and keys gpt-oss-20b sources by weight format', () => {
    const document = JSON.parse(fs.readFileSync(new URL('../catalog/models.json', import.meta.url), 'utf8'));
    assert.equal(document.schema, 'local-llm.catalog/v2');
    const [gpt] = loadSeedCatalog();
    assert.deepEqual(Object.keys(gpt.sources), ['gguf', 'ollama', 'hf']);
    assert.equal(gpt.sources.gguf.sha256, '27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901');
    assert.equal(gpt.sources.ollama.tag, 'gpt-oss:20b');
    // Parameters and measurements stay per runner.
    assert.deepEqual(Object.keys(gpt.recommended), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'lmstudio']);
    // LM Studio gets the llama.cpp runner's gpt-oss placement, with flash attention on (it has no auto).
    assert.deepEqual(gpt.recommended.lmstudio, { ctxSize: 16384, nGpuLayers: 99, nCpuMoe: 17, parallel: 1, batchSize: 256, ubatchSize: 256, flashAttn: 'on' });
    // Measured in the R1 benchmark at the default threads (physical cores minus 2).
    assert.deepEqual(Object.keys(gpt.validated), ['llama.cpp', 'ik_llama.cpp', 'ollama']);
    assert.match(gpt.validated['ik_llama.cpp'], /12 threads: 5,340 MiB VRAM, 40\.5 tok\/s generation, 386\.6 tok\/s prompt/);
    assert.match(gpt.validated['llama.cpp'], /b11159, 16k context, 12 threads: 4,798 MiB VRAM, 38\.4 tok\/s generation/);
    const schema = JSON.parse(fs.readFileSync(new URL('../catalog/schema.json', import.meta.url), 'utf8'));
    assert.equal(schema.$id, 'local-llm.catalog/v2');
    assert.deepEqual(Object.keys(schema.$defs.model.properties.sources.properties), ['gguf', 'ollama', 'hf', 'exl3']);
    assert.deepEqual(Object.keys(schema.$defs.model.properties.sources.properties), Object.keys(WEIGHT_FORMATS));
});

test('a v2 entry is validated per format; runner-keyed sources are refused', () => {
    const model = validateModel({ id: 'qwen-small', sources: { gguf: GGUF, ollama: OLLAMA } });
    assert.deepEqual(Object.keys(model.sources), ['gguf', 'ollama']);
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { 'llama.cpp': GGUF } }),
        { code: 'invalid_model', message: /unknown weight format 'llama\.cpp'/ });
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { gguf: OLLAMA } }), /sources\.gguf\.type must be huggingface/);
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { ollama: GGUF } }), /sources\.ollama\.type must be ollama/);
});

test('migrating a v1 entry maps llama.cpp to gguf, drops LM Studio and the vLLM GGUF fallback, and is idempotent', () => {
    const v1 = {
        id: 'user-qwen', displayName: 'Qwen', sources: { 'llama.cpp': GGUF, lmstudio: { ...GGUF, file: 'other.gguf' }, ollama: OLLAMA },
        recommended: { 'llama.cpp': { ctxSize: 8192 }, lmstudio: { contextLength: 4096 } },
    };
    const migrated = migrateModelEntry(v1);
    assert.deepEqual(migrated.sources, { gguf: GGUF, ollama: OLLAMA });
    // Only the v1 LM Studio stub's parameters are dropped (that stub never ran).
    assert.deepEqual(migrated.recommended, { 'llama.cpp': { ctxSize: 8192 } });
    assert.deepEqual(migrateModelEntry(migrated), migrated);
    // An entry whose only GGUF came from LM Studio or vLLM keeps it as the gguf source.
    assert.deepEqual(migrateModelEntry({ id: 'only-lms', sources: { lmstudio: GGUF } }).sources, { gguf: GGUF });
    assert.deepEqual(migrateModelEntry({ id: 'only-vllm', sources: { vllm: GGUF } }).sources, { gguf: GGUF });
    // Entries that are not objects are returned as they are, for validation to skip.
    assert.equal(migrateModelEntry(null), null);
    assert.equal(validateModel(migrateModelEntry(v1)).sources.gguf.commit, GGUF.commit);
});

// LM Studio is a runner again (runners plan I9, Phase R7), with parameters of
// its own. Only a v1 entry, which still keys sources by runner, carries the old
// stub's parameters, which never ran; a v2 entry's LM Studio parameters stay.
test('migration keeps the LM Studio parameters and measurements of a v2 entry', () => {
    const v2 = {
        id: 'user-qwen', sources: { gguf: GGUF },
        recommended: { 'llama.cpp': { ctxSize: 8192 }, lmstudio: { contextLength: 8192 } },
        validated: { lmstudio: 'PONG on CUDA0' },
    };
    assert.deepEqual(migrateModelEntry(v2), v2);
    // A v1 entry still loses the stub's entries, whichever v1 source it had.
    const v1 = { id: 'user-qwen', sources: { ollama: OLLAMA, 'llama.cpp': GGUF }, recommended: { lmstudio: { contextLength: 4096 } }, validated: { lmstudio: 'stub' } };
    assert.deepEqual(migrateModelEntry(v1).recommended, {});
    assert.deepEqual(migrateModelEntry(v1).validated, {});
    assert.deepEqual(migrateModelEntry(migrateModelEntry(v1)), migrateModelEntry(v1));
});

// The state file keeps version 1: registry entries are migrated one by one,
// idempotently. A controller from before this change reads the migrated
// entries as invalid and hides them, but keeps and saves them unchanged, so a
// downgrade and a later upgrade lose no user model.
test('a state file written before catalog v2 loads with its registry migrated and everything else unchanged', (t) => {
    const deployment = {
        id: 'd-1', requestId: 'request-0001', modelId: 'user-qwen', runnerId: 'llama.cpp', params: { ctxSize: 8192 },
        artifact: GGUF, phase: 'paused', pausedReason: 'The agent restarted during the download; press Run to resume.',
    };
    const v1 = {
        version: 1,
        deployment,
        params: { 'user-qwen|llama.cpp': { ctxSize: 8192 }, 'user-qwen|ollama': { numCtx: 4096 } },
        requests: { 'request-0001': { deploymentId: 'd-1', at: '2026-09-24T10:00:00.000Z' } },
        registry: [{ id: 'user-qwen', displayName: 'Qwen', sources: { 'llama.cpp': GGUF, ollama: OLLAMA } }],
        ollamaPulls: { 'qwen3:0.6b': [`sha256:${'c'.repeat(64)}`] },
    };
    const store = tempStore(t, v1);
    const state = store.load();
    assert.equal(STATE_VERSION, 1);
    assert.equal(state.version, 1);
    assert.deepEqual(state.registry, [{ id: 'user-qwen', displayName: 'Qwen', sources: { gguf: GGUF, ollama: OLLAMA } }]);
    assert.deepEqual(state.deployment, deployment);
    assert.deepEqual(state.params, v1.params);
    assert.deepEqual(state.requests, v1.requests);
    assert.deepEqual(state.ollamaPulls, v1.ollamaPulls);
    // The migrated entry is a valid catalog entry, so it is not silently skipped.
    assert.deepEqual(mergeCatalog([], state.registry).map((model) => model.id), ['user-qwen']);
    // Saving and loading again changes nothing.
    store.save(state);
    assert.deepEqual(store.load(), state);
});

test('two user entries with one id keep only the first, so lookups and removal agree', () => {
    // Possible after a downgrade: the old controller hides a migrated entry and
    // lets the same id be added again.
    const first = { id: 'user-qwen', displayName: 'First', sources: { gguf: GGUF } };
    const second = { id: 'user-qwen', displayName: 'Second', sources: { ollama: OLLAMA } };
    assert.deepEqual(mergeCatalog([], [first, second]).map((model) => model.displayName), ['First']);
});

test('a state file from a newer controller is kept aside, not overwritten silently', (t) => {
    const store = tempStore(t, { version: 99, registry: [{ id: 'x' }] });
    const state = store.load();
    assert.equal(state.version, STATE_VERSION);
    assert.deepEqual(state.registry, []);
    const kept = fs.readdirSync(path.dirname(store.file)).filter((name) => name.startsWith('controller.json.unsupported-'));
    assert.equal(kept.length, 1);
});
