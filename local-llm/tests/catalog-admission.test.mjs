import assert from 'node:assert/strict';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog, mergeCatalog, validateModel } from '../src/controller/catalog.mjs';
import { getRunner } from '../src/runners/index.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const [GPT] = loadSeedCatalog();

function snapshot({ gpuFree = 6000 * MIB, processes = [], available = 24 * GIB } = {}) {
    return {
        gpu: { available: true, totalBytes: 6144 * MIB, freeBytes: gpuFree, usedBytes: 6144 * MIB - gpuFree, processes },
        memory: { totalBytes: 31 * GIB, availableBytes: available },
        disk: { freeBytes: 300 * GIB },
    };
}

function admitGpt(runnerId, params = {}, snap = snapshot()) {
    const runner = getRunner(runnerId);
    const source = GPT.sources[runner.weightFormat];
    const normalized = runner.supported ? runner.normalizeParams(params, { model: GPT }) : {};
    return admit({ runner, model: GPT, source, params: normalized, snapshot: snap });
}

test('the seed catalog validates and pins every Hugging Face source to a commit', () => {
    assert.equal(GPT.id, 'gpt-oss-20b');
    assert.equal(GPT.seed, true);
    const source = GPT.sources.gguf;
    assert.match(source.commit, /^[0-9a-f]{40}$/);
    assert.equal(source.size, 12109566624);
    assert.equal(source.sha256, '27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901');
    assert.match(GPT.sources.ollama.manifestDigest, /^sha256:17052f91a42e/);
    const { commit: _commit, ...unpinned } = source;
    assert.throws(() => validateModel({ ...GPT, sources: { gguf: unpinned } }, { seed: true }), /commit/);
});

test('user model entries are validated before they are accepted', () => {
    assert.throws(() => validateModel({ id: 'no-sources', displayName: 'x' }), { code: 'invalid_model' });
    assert.throws(() => validateModel({ id: 'xx', sources: {} }), /sources/);
    assert.throws(() => validateModel({ id: 'Bad Id', sources: { ollama: { type: 'ollama', tag: 'x:1' } } }), /id must be/);
    assert.throws(() => validateModel({
        id: 'multi', sources: { gguf: { type: 'huggingface', repo: 'a/b', file: '../x.gguf' } },
    }), /single \.gguf/);
    assert.throws(() => validateModel({
        id: 'shell', sources: { ollama: { type: 'ollama', tag: 'x; rm -rf /' } },
    }), /Ollama library tag/);
    assert.throws(() => validateModel({ id: 'extra', sources: { ollama: { type: 'ollama', tag: 'x:1' } }, script: 'x' }), /unsupported field/);
    const qwen = validateModel({
        id: 'qwen3.6-35b-a3b',
        architecture: 'moe',
        sources: { gguf: { type: 'huggingface', repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', file: 'Qwen3.6-35B-A3B-UD-Q3_K_M.gguf' } },
    });
    assert.equal(qwen.sources.gguf.revision, 'main');
    assert.equal(qwen.seed, false);
    // A user entry cannot shadow a seed id; invalid persisted entries are skipped.
    const merged = mergeCatalog([GPT], [{ ...qwen, seed: undefined }, { id: 'gpt-oss-20b', sources: GPT.sources }, { broken: true }]);
    assert.deepEqual(merged.map((model) => model.id), ['gpt-oss-20b', 'qwen3.6-35b-a3b']);
});

test('admission: the Phase 0 defaults fit a 6 GB GPU and the estimate matches the measurement', () => {
    const result = admitGpt('llama.cpp');
    assert.equal(result.status, 'ok');
    assert.equal(result.estimate.isEstimate, true);
    const gpuMiB = result.estimate.gpuBytes / MIB;
    // Phase 0 measured 4,817 MiB in the container at these parameters.
    assert.ok(gpuMiB > 4600 && gpuMiB < 5100, `estimate ${gpuMiB} MiB`);
    assert.equal(result.estimate.basis, 'measured memory profile');
});

test('admission: 128k context with every expert on the GPU is incompatible, for a VRAM reason', () => {
    const result = admitGpt('llama.cpp', { ctxSize: 131072, nCpuMoe: 0 });
    assert.equal(result.status, 'incompatible');
    assert.match(result.reason, /GPU memory; the GPU has 6\.0 GiB/);
});

test('admission: enough hardware but VRAM held by another process is insufficient now', () => {
    const result = admitGpt('llama.cpp', {}, snapshot({
        gpuFree: 1500 * MIB,
        processes: [{ pid: 7, name: 'ollama', usedBytes: 4500 * MIB }],
    }));
    assert.equal(result.status, 'insufficient-now');
    assert.match(result.reason, /1\.5 GiB is free now; other GPU users: ollama/);
});

test('admission: an unsupported runner is incompatible with the reason its adapter gives', () => {
    const vllm = getRunner('vllm');
    const result = admit({ runner: vllm, model: GPT, source: undefined, params: {}, snapshot: snapshot() });
    assert.equal(result.status, 'incompatible');
    assert.equal(result.reason, 'Not supported or tested in this release; installable in a later release.');
});

test('admission: no GPU grant is an actionable incompatibility; big RAM users get a warning', () => {
    const noGpu = admitGpt('llama.cpp', {}, { gpu: { available: false, reason: 'No GPU is available to this agent: grant it' } });
    assert.equal(noGpu.status, 'incompatible');
    assert.match(noGpu.reason, /grant/);
    const qwen = validateModel({
        id: 'qwen3.6-35b-a3b',
        architecture: 'moe',
        sources: { gguf: { type: 'huggingface', repo: 'unsloth/Qwen3.6-35B-A3B-GGUF', file: 'q.gguf',
            commit: 'c'.repeat(40), size: 16_600_000_000, sha256: 'd'.repeat(64) } },
    });
    const runner = getRunner('llama.cpp');
    const params = runner.normalizeParams({ ctxSize: 8192, nCpuMoe: 48 }, { model: qwen });
    const result = admit({ runner, model: qwen, source: qwen.sources.gguf, params, snapshot: snapshot({ available: 20 * GIB }) });
    assert.equal(result.status, 'ok');
    assert.equal(result.estimate.basis, 'file size heuristic (no memory profile for this model)');
    assert.match(result.warnings[0], /Uses about 1[45]\.\d GiB of system RAM; 20\.0 GiB is available now/);
    const tight = admit({ runner, model: qwen, source: qwen.sources.gguf, params, snapshot: snapshot({ available: 12 * GIB }) });
    assert.equal(tight.status, 'insufficient-now');
});

test('admission: Ollama x gpt-oss:20b fits with its automatic layer split', () => {
    const result = admitGpt('ollama');
    assert.equal(result.status, 'ok');
    assert.ok(result.estimate.ramBytes > 8 * GIB);
    const pinned = admitGpt('ollama', { numGpu: 999 });
    assert.equal(pinned.status, 'incompatible');
});
