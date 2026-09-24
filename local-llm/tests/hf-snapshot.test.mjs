// Hugging Face snapshot store (runners plan §5.3, Phase R3): a model that is
// a directory of files (safetensors for vLLM, EXL3 for TabbyAPI), pinned to
// a commit with every file's size and digest, downloaded file by file with
// resume and verification, ready only when every file verified, and deleted
// as one unit.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { WEIGHT_FORMATS, isSnapshotFile, validateModel } from '../src/controller/catalog.mjs';
import {
    DownloadError,
    downloadSnapshotFile,
    inspectSnapshotFile,
    resolveHuggingFaceSnapshot,
    snapshotPaths,
} from '../src/controller/downloader.mjs';
import { createWeightStores } from '../src/controller/weightStores.mjs';

const REPO = 'org/model-AWQ';
const COMMIT = 'b'.repeat(40);
const noSleep = async () => {};
const sha256 = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');
// What the Hugging Face tree API reports as `oid` for a file stored in git (not LFS).
const gitOid = (buffer) => crypto.createHash('sha1').update(`blob ${buffer.length}\0`).update(buffer).digest('hex');

const FILES = Object.freeze({
    'config.json': Buffer.from('{"model_type":"qwen3","num_hidden_layers":36}\n'),
    'model.safetensors': crypto.randomBytes(2 * 1024 * 1024 + 7),
    'tokenizer.json': Buffer.from('{"version":"1.0","model":{}}\n'),
});
const LFS = new Set(['model.safetensors', 'pytorch_model.bin']);
// In the repository, never part of the snapshot.
const EXTRA = Object.freeze({
    'README.md': Buffer.from('# model\n'),
    '.gitattributes': Buffer.from('*.safetensors filter=lfs\n'),
    'modeling_custom.py': Buffer.from('import os\n'),
    'pytorch_model.bin': crypto.randomBytes(4096),
});

function pinnedFiles() {
    return Object.entries(FILES).map(([name, bytes]) => (LFS.has(name)
        ? { path: name, size: bytes.length, sha256: sha256(bytes) }
        : { path: name, size: bytes.length, gitOid: gitOid(bytes) }));
}

function tempRoot(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-snapshot-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    return root;
}

// A tiny Hugging Face: revision and tree APIs and the resolve URL with Range.
async function hub(t, { cutFirst = null, wrong = null, ignoreRange = false, badLfs = false } = {}) {
    const requests = [];
    const all = { ...FILES, ...EXTRA };
    const server = http.createServer((req, res) => {
        const url = new URL(req.url, 'http://hub');
        requests.push({ path: url.pathname, range: req.headers.range ?? null });
        if (url.pathname === `/api/models/${REPO}/revision/main`) {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ sha: COMMIT }));
            return;
        }
        if (url.pathname === `/api/models/${REPO}/tree/${COMMIT}`) {
            const entries = Object.entries(all).map(([name, bytes]) => ({
                type: 'file', path: name, size: bytes.length,
                oid: gitOid(LFS.has(name) ? Buffer.from(`lfs pointer ${name}`) : bytes),
                ...(LFS.has(name) ? { lfs: { oid: sha256(bytes), size: badLfs ? bytes.length + 1 : bytes.length, pointerSize: 134 } } : {}),
            }));
            entries.push({ type: 'directory', path: 'original', size: 0, oid: 'c'.repeat(40) });
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify(entries));
            return;
        }
        const prefix = `/${REPO}/resolve/${COMMIT}/`;
        const name = url.pathname.startsWith(prefix) ? decodeURIComponent(url.pathname.slice(prefix.length)) : null;
        let body = name ? all[name] : null;
        if (!body) { res.writeHead(404); res.end(); return; }
        if (wrong === name) body = Buffer.from(body.map((byte) => byte ^ 0xff));
        const range = ignoreRange ? null : /^bytes=(\d+)-$/.exec(req.headers.range ?? '');
        const start = range ? Number(range[1]) : 0;
        const headers = { 'Content-Length': body.length - start };
        if (range) headers['Content-Range'] = `bytes ${start}-${body.length - 1}/${body.length}`;
        res.writeHead(range ? 206 : 200, headers);
        const hits = requests.filter((request) => request.path === url.pathname).length;
        if (cutFirst === name && hits === 1) {
            res.write(body.subarray(start, start + 512 * 1024), () => res.destroy());
            return;
        }
        res.end(body.subarray(start));
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => server.close());
    return { base: `http://127.0.0.1:${server.address().port}`, requests };
}

const snapshotSource = () => ({ type: 'hf-snapshot', repo: REPO, revision: 'main', commit: COMMIT, files: pinnedFiles() });

test('a snapshot source pins every file with its digest, sums the size, and takes only top-level model files', () => {
    assert.equal(WEIGHT_FORMATS.hf.sourceType, 'hf-snapshot');
    assert.equal(WEIGHT_FORMATS.exl3.sourceType, 'hf-snapshot');
    const model = validateModel({ id: 'snap-model', sources: { hf: snapshotSource(), exl3: snapshotSource() } }, { seed: true });
    const total = Object.values(FILES).reduce((sum, bytes) => sum + bytes.length, 0);
    assert.equal(model.sources.hf.size, total);
    assert.deepEqual(model.sources.hf.files.map((file) => file.path), ['config.json', 'model.safetensors', 'tokenizer.json']);
    // A user entry may name only the revision; Add pins it.
    const user = validateModel({ id: 'user-snap', sources: { hf: { type: 'hf-snapshot', repo: REPO, revision: 'main' } } });
    assert.equal(user.sources.hf.commit, undefined);
    assert.throws(() => validateModel({ id: 'seed-snap', sources: { hf: { type: 'hf-snapshot', repo: REPO } } }, { seed: true }), /commit/);
    for (const [files, message] of [
        [[{ path: 'original/model.safetensors', size: 1, sha256: 'a'.repeat(64) }], /files\[0\]\.path/],
        [[{ path: '../x.json', size: 1, sha256: 'a'.repeat(64) }], /files\[0\]\.path/],
        [[{ path: 'modeling.py', size: 1, sha256: 'a'.repeat(64) }], /files\[0\]\.path/],
        [[{ path: 'config.json', size: 1 }], /sha256 or gitOid/],
        [[{ path: 'config.json', size: 1, gitOid: 'z'.repeat(40) }], /gitOid/],
        [[{ path: 'config.json', size: 0, gitOid: 'a'.repeat(40) }], /size/],
        [[...pinnedFiles(), pinnedFiles()[0]], /twice/],
        [[], /files/],
    ]) {
        assert.throws(() => validateModel({ id: 'bad-snap', sources: { hf: { ...snapshotSource(), files } } }, { seed: true }), message);
    }
    assert.throws(() => validateModel({ id: 'bad-snap', sources: { hf: { ...snapshotSource(), size: 5 } } }, { seed: true }), /size/);
    for (const name of ['config.json', 'model-00001-of-00002.safetensors', 'merges.txt', 'tokenizer.model', 'chat_template.jinja', 'o200k_base.tiktoken']) {
        assert.equal(isSnapshotFile(name), true, name);
    }
    for (const name of ['README.md', '.gitattributes', 'pytorch_model.bin', 'modeling_x.py', 'model.gguf', 'original/config.json', 'LICENSE']) {
        assert.equal(isSnapshotFile(name), false, name);
    }
});

test('pinning lists the top-level model files at the commit: sha256 for LFS files, the git oid for the rest', async (t) => {
    const { base } = await hub(t);
    const pinned = await resolveHuggingFaceSnapshot({ repo: REPO, revision: 'main', baseUrl: base });
    assert.equal(pinned.commit, COMMIT);
    assert.deepEqual(pinned.files, pinnedFiles());
});

test('each file is verified before it lands in the snapshot directory, which never holds partial or bookkeeping files', async (t) => {
    const { base } = await hub(t);
    const root = tempRoot(t);
    const { dir, stateDir } = snapshotPaths({ root, repo: REPO, commit: COMMIT });
    for (const file of pinnedFiles()) {
        const artifact = { repo: REPO, commit: COMMIT, file: file.path, size: file.size, sha256: file.sha256, gitOid: file.gitOid };
        const result = await downloadSnapshotFile({ root, artifact, baseUrl: base, sleep: noSleep });
        assert.equal(result.path, path.join(dir, file.path));
        assert.equal((await inspectSnapshotFile({ root, artifact })).state, 'complete');
    }
    assert.deepEqual(fs.readdirSync(dir).sort(), Object.keys(FILES).sort());
    for (const [name, bytes] of Object.entries(FILES)) assert.ok(fs.readFileSync(path.join(dir, name)).equals(bytes), name);
    assert.ok(path.relative(dir, stateDir).startsWith('..'), 'bookkeeping lives outside the snapshot directory');
    assert.ok(fs.readdirSync(stateDir).every((name) => name.endsWith('.json')));
});

test('bytes that match neither the sha256 nor the git oid are refused and nothing is kept', async (t) => {
    const root = tempRoot(t);
    for (const name of ['config.json', 'model.safetensors']) {
        const { base } = await hub(t, { wrong: name });
        const file = pinnedFiles().find((entry) => entry.path === name);
        const artifact = { repo: REPO, commit: COMMIT, file: name, size: file.size, sha256: file.sha256, gitOid: file.gitOid };
        await assert.rejects(() => downloadSnapshotFile({ root, artifact, baseUrl: base, sleep: noSleep }),
            (error) => error instanceof DownloadError && error.code === 'SHA256_MISMATCH');
        assert.equal((await inspectSnapshotFile({ root, artifact })).state, 'absent');
    }
    assert.equal(fs.existsSync(snapshotPaths({ root, repo: REPO, commit: COMMIT }).dir)
        && fs.readdirSync(snapshotPaths({ root, repo: REPO, commit: COMMIT }).dir).length, 0);
});

test('a cut transfer resumes from the partial with Range', async (t) => {
    const { base, requests } = await hub(t, { cutFirst: 'model.safetensors' });
    const root = tempRoot(t);
    const file = pinnedFiles().find((entry) => entry.path === 'model.safetensors');
    const artifact = { repo: REPO, commit: COMMIT, file: file.path, size: file.size, sha256: file.sha256 };
    const result = await downloadSnapshotFile({ root, artifact, baseUrl: base, sleep: noSleep });
    assert.equal(result.status, 'complete');
    const ranges = requests.filter((request) => request.path.endsWith('/model.safetensors')).map((request) => request.range);
    assert.deepEqual(ranges, [null, `bytes=${512 * 1024}-`]);
});

function stores(t, root, base) {
    let state = {};
    return createWeightStores({
        dataDir: root,
        env: {},
        hfBaseUrl: base,
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async () => assert.fail('the GGUF store is not used here'),
        remove: async () => 0,
        resolveHf: async () => assert.fail('the GGUF store is not used here'),
        state: () => state,
        save: () => {},
        activeArtifact: () => null,
        downloadSnapshot: (options) => downloadSnapshotFile({ ...options, sleep: noSleep }),
        inspectSnapshot: inspectSnapshotFile,
        resolveSnapshot: resolveHuggingFaceSnapshot,
    })['hf-snapshot'];
}

test('a partial snapshot is never ready; fetch completes it; Delete removes the snapshot and its bookkeeping', async (t) => {
    const { base } = await hub(t);
    const root = tempRoot(t);
    const store = stores(t, root, base);
    const source = validateModel({ id: 'snap-model', sources: { hf: snapshotSource() } }, { seed: true }).sources.hf;
    assert.equal(store.type, 'hf-snapshot');
    assert.equal(store.fetchedBy, 'controller');
    assert.equal(store.key(source), `hf:${REPO}@${COMMIT}`);
    assert.equal(store.isPinned(source), true);
    assert.deepEqual(await store.state(source), { state: 'absent', bytes: 0, total: source.size });
    // One file of three: partial, never complete.
    const [first] = source.files;
    await downloadSnapshotFile({ root: path.join(root, 'models', 'hf'), artifact: { repo: REPO, commit: COMMIT, file: first.path, size: first.size, gitOid: first.gitOid }, baseUrl: base });
    const partial = await store.state(source);
    assert.equal(partial.state, 'partial');
    assert.equal(partial.bytes, first.size);
    const progress = [];
    const fetched = await store.fetch({ artifact: source, onProgress: (update) => progress.push(update) });
    assert.equal(fetched.path, snapshotPaths({ root: path.join(root, 'models', 'hf'), repo: REPO, commit: COMMIT }).dir);
    assert.equal(fetched.bytes, source.size);
    assert.equal(fetched.bytesTransferred, source.size - first.size);
    assert.ok(progress.length > 0 && progress.at(-1).bytes === source.size);
    assert.deepEqual(await store.state(source), { state: 'complete', bytes: source.size, total: source.size });
    const freed = await store.remove(source);
    assert.ok(freed >= source.size);
    assert.deepEqual(await store.state(source), { state: 'absent', bytes: 0, total: source.size });
    assert.equal(fs.existsSync(path.join(root, 'models', 'hf', ...REPO.split('/'), COMMIT)), false);
    assert.equal(fs.existsSync(path.join(root, 'models', 'hf', '.state', ...REPO.split('/'), COMMIT)), false);
});

test('Add pins a revision to a commit and its files; an update keeps the pin while repository and revision stay', async (t) => {
    const { base } = await hub(t);
    const store = stores(t, tempRoot(t), base);
    const pinned = await store.pin({ type: 'hf-snapshot', repo: REPO, revision: 'main' });
    assert.equal(pinned.commit, COMMIT);
    assert.deepEqual(pinned.files, pinnedFiles());
    assert.equal(store.isPinned({ type: 'hf-snapshot', repo: REPO, revision: 'main' }), false);
    assert.deepEqual(store.carryPin({ type: 'hf-snapshot', repo: REPO, revision: 'main' }, pinned), pinned);
    assert.equal(store.carryPin({ type: 'hf-snapshot', repo: REPO, revision: 'v2' }, pinned).commit, undefined);
});

test('a git-oid file whose partial must restart from zero still verifies by its git oid', async (t) => {
    // A server that ignores Range answers 200 with the whole file: the partial restarts.
    const { base, requests } = await hub(t, { ignoreRange: true });
    const root = tempRoot(t);
    const file = pinnedFiles().find((entry) => entry.path === 'config.json');
    const artifact = { repo: REPO, commit: COMMIT, file: file.path, size: file.size, gitOid: file.gitOid };
    const { stateDir } = snapshotPaths({ root, repo: REPO, commit: COMMIT });
    fs.mkdirSync(stateDir, { recursive: true });
    fs.writeFileSync(path.join(stateDir, 'config.json.partial'), FILES['config.json'].subarray(0, 10));
    fs.writeFileSync(path.join(stateDir, 'config.json.partial.json'), JSON.stringify({ repo: REPO, commit: COMMIT, file: file.path, size: file.size, gitOid: file.gitOid }));
    const result = await downloadSnapshotFile({ root, artifact, baseUrl: base, sleep: noSleep });
    assert.equal(result.status, 'complete');
    assert.equal(requests.at(-1).range, 'bytes=10-');
    assert.equal((await inspectSnapshotFile({ root, artifact })).state, 'complete');
});

test('pinning refuses an LFS entry whose sizes disagree instead of trusting its pointer', async (t) => {
    const { base } = await hub(t, { badLfs: true });
    await assert.rejects(() => resolveHuggingFaceSnapshot({ repo: REPO, revision: 'main', baseUrl: base }),
        (error) => error instanceof DownloadError && error.code === 'RESOLVE_FAILED');
});

test('a snapshot without a safetensors file is refused', () => {
    const files = pinnedFiles().filter((file) => !file.path.endsWith('.safetensors'));
    assert.throws(() => validateModel({ id: 'no-weights', sources: { hf: { ...snapshotSource(), files } } }, { seed: true }), /safetensors/);
});
