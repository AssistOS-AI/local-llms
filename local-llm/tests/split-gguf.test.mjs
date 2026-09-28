// Split GGUF (DS002): one gguf source whose `file` is the first shard and whose
// `shards` pin every shard by size and sha256, in canonical order, in one
// directory; one artifact for in-use checks, disk, progress and deletion.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { splitGgufFiles, splitGgufName, validateModel } from '../src/controller/catalog.mjs';
import { artifactPaths } from '../src/controller/downloader.mjs';
import { createWeightStores } from '../src/controller/weightStores.mjs';

const COMMIT = 'c'.repeat(40);
const REPO = 'unsloth/Qwen3.5-122B-A10B-GGUF';
const FIRST = 'MXFP4_MOE/Qwen3.5-122B-A10B-MXFP4_MOE-00001-of-00003.gguf';
const SHARDS = Object.freeze([
    { file: FIRST, size: 30, sha256: '1'.repeat(64) },
    { file: 'MXFP4_MOE/Qwen3.5-122B-A10B-MXFP4_MOE-00002-of-00003.gguf', size: 20, sha256: '2'.repeat(64) },
    { file: 'MXFP4_MOE/Qwen3.5-122B-A10B-MXFP4_MOE-00003-of-00003.gguf', size: 10, sha256: '3'.repeat(64) },
]);
const SOURCE = Object.freeze({ type: 'huggingface', repo: REPO, file: FIRST, revision: 'main', commit: COMMIT, shards: SHARDS });

const entry = (gguf) => ({ id: 'qwen-split', architecture: 'moe', sources: { gguf } });
// The controller only ever sees validated sources (size is the sum of the shards).
const VALIDATED = validateModel(entry(SOURCE), { seed: true }).sources.gguf;

test('split GGUF names: the shard number and count, and every shard from the first', () => {
    assert.deepEqual(splitGgufName('a/b-00002-of-00003.gguf'), { prefix: 'a/b', index: 2, count: 3 });
    assert.equal(splitGgufName('gpt-oss-120b-MXFP4.gguf'), null);
    assert.equal(splitGgufName('x-00001-of-00001.gguf'), null, 'one shard is a single file');
    assert.equal(splitGgufName('x-00004-of-00003.gguf'), null);
    assert.deepEqual(splitGgufFiles(FIRST), SHARDS.map((shard) => shard.file));
    assert.equal(splitGgufFiles(SHARDS[1].file), null);
});

test('a split source pins every shard in canonical order; its size is their sum', () => {
    const model = validateModel(entry(SOURCE), { seed: true });
    assert.equal(model.sources.gguf.size, 60);
    assert.deepEqual(model.sources.gguf.shards, SHARDS);
    assert.equal(validateModel(entry({ ...SOURCE, size: 60 })).sources.gguf.size, 60);
    const refuse = (gguf, pattern, seed = false) => assert.throws(() => validateModel(entry(gguf), { seed }), pattern);
    refuse({ ...SOURCE, size: 61 }, /size must equal the sum of the shard sizes \(60\)/);
    refuse({ ...SOURCE, shards: SHARDS.slice(0, 2) }, /must list all 3 shards/);
    refuse({ ...SOURCE, shards: [SHARDS[1], SHARDS[0], SHARDS[2]] }, /shards\[0\]\.file must be .*00001-of-00003/);
    refuse({ ...SOURCE, shards: [SHARDS[0], { ...SHARDS[1], sha256: 'x' }, SHARDS[2]] }, /shards\[1\]\.sha256/);
    refuse({ ...SOURCE, sha256: '4'.repeat(64) }, /sha256 is per shard/);
    refuse({ ...SOURCE, file: SHARDS[1].file }, /names shard 2 of 3; name the first shard/);
    refuse({ ...SOURCE, file: 'single.gguf' }, /shards is only for a split GGUF/);
    refuse({ ...SOURCE, commit: undefined }, /commit must be a 40-hex commit/);
    // A seed must pin its shards; a user entry may name the first shard only, for Add model to resolve.
    refuse({ type: 'huggingface', repo: REPO, file: FIRST }, /commit must be a 40-hex commit/, true);
    assert.deepEqual(validateModel(entry({ type: 'huggingface', repo: REPO, file: FIRST })).sources.gguf,
        { type: 'huggingface', repo: REPO, file: FIRST, revision: 'main' });
});

function stores(t, { inspected = {}, resolved = {} } = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-split-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const calls = { download: [], remove: [], resolve: [] };
    const all = createWeightStores({
        dataDir,
        env: {},
        hfBaseUrl: 'https://huggingface.co',
        inspect: async ({ artifact }) => inspected[artifact.file] || { state: 'absent', bytes: 0 },
        download: async ({ artifact, onProgress }) => {
            calls.download.push(artifact);
            onProgress({ bytes: artifact.size / 2, total: artifact.size, transferred: artifact.size / 2 });
            onProgress({ bytes: artifact.size, total: artifact.size, transferred: artifact.size });
            return { path: `/data/models/gguf/${artifact.file}`, bytesTransferred: artifact.size };
        },
        remove: async ({ artifact }) => { calls.remove.push(artifact.file); return artifact.size; },
        resolveHf: async ({ file, revision }) => {
            calls.resolve.push([file, revision]);
            const shard = SHARDS.find((entry) => entry.file === file);
            return { commit: COMMIT, size: shard.size, sha256: shard.sha256, ...(resolved[file] || {}) };
        },
        state: () => ({}),
        save: () => {},
        activeArtifact: () => null,
    });
    return { store: all.huggingface, calls };
}

test('the store handles the set as one artifact: key, state, download progress, deletion', async (t) => {
    const partial = stores(t, { inspected: { [FIRST]: { state: 'complete', bytes: 30 }, [SHARDS[1].file]: { state: 'partial', bytes: 5 } } });
    const { store } = partial;
    assert.equal(store.key(VALIDATED), `gguf:${REPO}@${COMMIT}/${FIRST}#3`);
    assert.equal(store.isPinned(VALIDATED), true);
    assert.equal(store.isPinned({ ...VALIDATED, shards: undefined }), false, 'a split file without its shards is not pinned');
    assert.deepEqual(await store.state(VALIDATED), { state: 'partial', bytes: 35, total: 60, shards: 3 });
    const complete = stores(t, { inspected: Object.fromEntries(SHARDS.map((shard) => [shard.file, { state: 'complete', bytes: shard.size }])) });
    assert.equal((await complete.store.state(VALIDATED)).state, 'complete');
    assert.equal((await stores(t).store.state(VALIDATED)).state, 'absent');

    const progress = [];
    const fetched = await store.fetch({ artifact: VALIDATED, onProgress: (entry) => progress.push([entry.bytes, entry.total]) });
    assert.deepEqual(partial.calls.download.map((artifact) => [artifact.file, artifact.size, artifact.sha256, artifact.commit]),
        SHARDS.map((shard) => [shard.file, shard.size, shard.sha256, COMMIT]));
    assert.deepEqual(progress, [[15, 60], [30, 60], [40, 60], [50, 60], [55, 60], [60, 60]]);
    assert.equal(fetched.path, `/data/models/gguf/${FIRST}`, 'llama.cpp opens the first shard');
    assert.equal(fetched.files.length, 3);
    assert.equal(fetched.bytes, 60);
    assert.equal(await store.remove(VALIDATED), 60);
    assert.deepEqual(partial.calls.remove, SHARDS.map((shard) => shard.file));
    // Every shard lands in the same directory as the first.
    const first = store.paths(VALIDATED).file;
    const root = first.slice(0, first.indexOf('/models/gguf/') + '/models/gguf'.length);
    const last = artifactPaths({ root, artifact: { ...VALIDATED, shards: undefined, ...SHARDS[2] } });
    assert.equal(path.dirname(last.file), path.dirname(store.paths(VALIDATED).file));
});

test('Add model pins a split GGUF from its first shard: every shard at the first shard\'s commit', async (t) => {
    const { store, calls } = stores(t);
    const pinned = await store.pin({ type: 'huggingface', repo: REPO, file: FIRST, revision: 'main' });
    assert.deepEqual(calls.resolve, [[FIRST, 'main'], [SHARDS[1].file, COMMIT], [SHARDS[2].file, COMMIT]]);
    assert.deepEqual(pinned, { type: 'huggingface', repo: REPO, file: FIRST, revision: 'main', commit: COMMIT, size: 60, shards: SHARDS });
    assert.equal(validateModel(entry(pinned)).sources.gguf.size, 60);
    // An update of the same entry keeps the pin.
    assert.deepEqual(store.carryPin({ type: 'huggingface', repo: REPO, file: FIRST, revision: 'main' }, pinned).shards, SHARDS);
});
