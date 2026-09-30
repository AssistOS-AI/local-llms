// Catalog schema v3: model sources are keyed by weight format (gguf, ollama,
// hf, exl3), so runners that read the same format share one download;
// `recommended` and `validated` are keyed by hardware profile, then by runner;
// `profiles` says where a model is offered; `unified.envelope` is the unified
// profile's measured envelope (DS005). Earlier schemas are not migrated.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { WEIGHT_FORMATS, loadSeedCatalog, mergeCatalog, validateModel } from '../src/controller/catalog.mjs';
import { STATE_VERSION, createStateStore } from '../src/controller/stateStore.mjs';

const GGUF = Object.freeze({
    type: 'huggingface', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', revision: 'main',
    commit: 'a'.repeat(40), size: 639446688, sha256: 'b'.repeat(64),
});
const OLLAMA = Object.freeze({ type: 'ollama', tag: 'qwen3:0.6b' });
const RECTANGLE = Object.freeze({ runner: 'llama.cpp', loadMode: 'dio', maxCtx: 131072, maxParallel: 4, mtp: false, bufferBytes: 17_000_000_000, transientBytes: 2_000_000_000 });

function tempStore(t, content) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-catalog-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const store = createStateStore({ dataDir });
    if (content !== undefined) {
        fs.mkdirSync(path.dirname(store.file), { recursive: true });
        fs.writeFileSync(store.file, JSON.stringify(content));
    }
    return store;
}

test('the seed catalog is schema v3: sources by weight format, parameters and measurements by profile, then runner', () => {
    const document = JSON.parse(fs.readFileSync(new URL('../catalog/models.json', import.meta.url), 'utf8'));
    assert.equal(document.schema, 'local-llm.catalog/v3');
    const [gpt, awq, exl3] = loadSeedCatalog();
    assert.deepEqual(Object.keys(gpt.sources), ['gguf', 'ollama', 'hf']);
    assert.equal(gpt.sources.gguf.sha256, '27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901');
    assert.equal(gpt.sources.ollama.tag, 'gpt-oss:20b');
    assert.deepEqual(gpt.profiles, ['dedicated', 'unified', 'cpu']);
    assert.deepEqual([awq.profiles, exl3.profiles], [['dedicated'], ['dedicated']]);
    assert.deepEqual(Object.keys(gpt.recommended.dedicated), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'lmstudio']);
    // LM Studio gets the llama.cpp runner's gpt-oss placement, with flash attention on (it has no auto).
    assert.deepEqual(gpt.recommended.dedicated.lmstudio, { ctxSize: 16384, nGpuLayers: 99, nCpuMoe: 17, parallel: 1, batchSize: 256, ubatchSize: 256, flashAttn: 'on' });
    assert.deepEqual(Object.keys(gpt.recommended.unified), ['llama.cpp']);
    // Measured in the R1 benchmark at the default threads (physical cores minus 2).
    assert.deepEqual(Object.keys(gpt.validated.dedicated), ['llama.cpp', 'ik_llama.cpp', 'ollama']);
    assert.match(gpt.validated.dedicated['ik_llama.cpp'], /12 threads: 5,340 MiB VRAM, 40\.5 tok\/s generation, 386\.6 tok\/s prompt/);
    assert.match(gpt.validated.dedicated['llama.cpp'], /b11159, 16k context, 12 threads: 4,798 MiB VRAM, 38\.4 tok\/s generation/);
    const schema = JSON.parse(fs.readFileSync(new URL('../catalog/schema.json', import.meta.url), 'utf8'));
    assert.equal(schema.$id, 'local-llm.catalog/v3');
    assert.deepEqual(Object.keys(schema.$defs.model.properties.sources.properties), Object.keys(WEIGHT_FORMATS));
    assert.deepEqual(Object.keys(schema.$defs.model.properties.recommended.properties), ['dedicated', 'unified', 'cpu']);
});

test('an entry is validated per format; runner-keyed sources are refused', () => {
    const model = validateModel({ id: 'qwen-small', sources: { gguf: GGUF, ollama: OLLAMA } });
    assert.deepEqual(Object.keys(model.sources), ['gguf', 'ollama']);
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { 'llama.cpp': GGUF } }),
        { code: 'invalid_model', message: /unknown weight format 'llama\.cpp'/ });
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { gguf: OLLAMA } }), /sources\.gguf\.type must be huggingface/);
    assert.throws(() => validateModel({ id: 'qwen-small', sources: { ollama: GGUF } }), /sources\.ollama\.type must be ollama/);
});

test('profiles default to every profile; recommended and validated are keyed by profile, then runner', () => {
    // `validated` is a measurement record: only a trusted seed entry carries one.
    const model = validateModel({
        id: 'qwen-small', sources: { gguf: GGUF },
        recommended: { dedicated: { 'llama.cpp': { ctxSize: 8192 } }, unified: { 'llama.cpp': { ctxSize: 32768 } } },
        validated: { unified: { 'llama.cpp': 'DGX Spark, b11159' } },
    }, { seed: true });
    assert.deepEqual(model.profiles, ['dedicated', 'unified', 'cpu']);
    assert.deepEqual(model.recommended.unified['llama.cpp'], { ctxSize: 32768 });
    assert.equal(model.validated.unified['llama.cpp'], 'DGX Spark, b11159');
    assert.equal(model.unified, null);
    assert.deepEqual(validateModel({ id: 'x-only', sources: { gguf: GGUF }, profiles: ['unified'] }).profiles, ['unified']);
    assert.deepEqual(validateModel({ id: 'cpu-only', sources: { gguf: GGUF }, profiles: ['cpu'] }).profiles, ['cpu']);
    for (const profiles of [[], ['gpu'], ['unified', 'unified'], ['cpu', 'cpu'], 'unified']) {
        assert.throws(() => validateModel({ id: 'bad-profiles', sources: { gguf: GGUF }, profiles }),
            { message: /profiles must list one or more of dedicated, unified, cpu/ });
    }
    // `recommended` and `validated` are keyed by cpu too, and an unknown profile is still refused.
    const cpuKeyed = validateModel({ id: 'cpu-keyed', sources: { gguf: GGUF }, recommended: { cpu: { 'llama.cpp': { ctxSize: 4096 } } },
        validated: { cpu: { 'llama.cpp': 'M1 Podman machine' } } }, { seed: true });
    assert.deepEqual(cpuKeyed.recommended.cpu['llama.cpp'], { ctxSize: 4096 });
    assert.throws(() => validateModel({ id: 'gpu-keyed', sources: { gguf: GGUF }, recommended: { gpu: { 'llama.cpp': {} } } }),
        /recommended has an unknown profile 'gpu'/);
    // A v2 entry (recommended keyed by runner) is not valid v3.
    assert.throws(() => validateModel({ id: 'v2-entry', sources: { gguf: GGUF }, recommended: { 'llama.cpp': { ctxSize: 8192 } } }),
        /recommended has an unknown profile 'llama\.cpp'/);
    assert.throws(() => validateModel({ id: 'v2-entry', sources: { gguf: GGUF }, validated: { unified: { 'llama.cpp': 42 } } }, { seed: true }),
        /validated\.unified\.llama\.cpp must be plain text/);
});

test('the unified envelope is measured rectangles, each with its load mode, buffers and transient', () => {
    const model = validateModel({ id: 'enveloped', sources: { gguf: GGUF }, unified: { envelope: [RECTANGLE, { ...RECTANGLE, maxCtx: 262144, maxParallel: 1, mtp: true, measured: 'L1, 2026-09-28' }] } }, { seed: true });
    assert.equal(model.unified.envelope.length, 2);
    assert.deepEqual(model.unified.envelope[0], { ...RECTANGLE, measured: undefined });
    assert.equal(model.unified.envelope[1].measured, 'L1, 2026-09-28');
    for (const [change, pattern] of [
        [{ bufferBytes: undefined }, /bufferBytes is required/],
        [{ transientBytes: -1 }, /transientBytes must be an integer/],
        [{ maxParallel: 17 }, /maxParallel must be an integer from 1 to 16/],
        [{ mtp: 'yes' }, /mtp must be true or false/],
        [{ runner: '' }, /runner must name a runner/],
        [{ extra: 1 }, /unsupported field 'extra'/],
        // The load mode is part of what was measured: it is required, one mode per rectangle.
        [{ loadMode: undefined }, /loadMode must be the load mode it was measured with \(dio or none\)/],
        [{ loadMode: 'mmap' }, /loadMode must be the load mode it was measured with/],
        [{ loadMode: ['dio', 'none'] }, /loadMode must be the load mode it was measured with/],
    ]) {
        const entry = { ...RECTANGLE, ...change };
        for (const [key, value] of Object.entries(change)) if (value === undefined) delete entry[key];
        assert.throws(() => validateModel({ id: 'enveloped', sources: { gguf: GGUF }, unified: { envelope: [entry] } }, { seed: true }), pattern);
    }
    assert.throws(() => validateModel({ id: 'enveloped', sources: { gguf: GGUF }, unified: { envelope: [] } }, { seed: true }), /1-16 measured rectangles/);
});

test('a registry entry of an earlier schema is kept in the state file but hidden from the catalog', (t) => {
    const v1 = { id: 'user-qwen', displayName: 'Qwen', sources: { 'llama.cpp': GGUF, ollama: OLLAMA } };
    const v2 = { id: 'user-v2', sources: { gguf: GGUF }, recommended: { 'llama.cpp': { ctxSize: 8192 } } };
    const v3 = { id: 'user-v3', sources: { gguf: GGUF }, recommended: { dedicated: { 'llama.cpp': { ctxSize: 8192 } } } };
    const store = tempStore(t, { version: 1, registry: [v1, v2, v3] });
    const state = store.load();
    assert.equal(state.version, STATE_VERSION);
    assert.deepEqual(state.registry, [v1, v2, v3]);
    assert.deepEqual(mergeCatalog([], state.registry).map((model) => model.id), ['user-v3']);
});

test('two user entries with one id keep only the first, so lookups and removal agree', () => {
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

test('the unified seed lists the models measured or to be measured on DGX Spark, every source pinned', () => {
    const byId = Object.fromEntries(loadSeedCatalog().map((model) => [model.id, model]));
    for (const id of ['gpt-oss-120b', 'qwen3.6-35b-a3b-mtp', 'qwen3-coder-30b-a3b', 'glm-4.7-flash', 'qwen3.5-122b-a10b']) {
        const model = byId[id];
        assert.deepEqual(model.profiles, ['unified'], id);
        assert.match(model.sources.gguf.commit, /^[0-9a-f]{40}$/, id);
        assert.equal(model.sources.gguf.revision, model.sources.gguf.commit, id);
    }
    assert.equal(byId['gpt-oss-120b'].sources.gguf.sha256, '582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d');
    assert.equal(byId['qwen3.6-35b-a3b-mtp'].sources.gguf.sha256, '55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1');
    // Qwen3.5-122B-A10B is a split GGUF: three shards, 74.7 GB together.
    assert.equal(byId['qwen3.5-122b-a10b'].sources.gguf.shards.length, 3);
    assert.equal(byId['qwen3.5-122b-a10b'].sources.gguf.size, 74664408608);
    // Listed models carry no measured envelope; on unified memory they are sized by estimate (DS005).
    for (const id of ['qwen3-coder-30b-a3b', 'glm-4.7-flash', 'qwen3.5-122b-a10b']) assert.equal(byId[id].unified, null, id);
});
