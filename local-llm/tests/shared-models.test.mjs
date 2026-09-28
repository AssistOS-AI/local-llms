// C12 (DS002, DS003): a pinned model file already in the workspace is used
// instead of downloaded. In order: the agent's own verified file; a file at
// the store path, adopted in place; a private, verified copy of a candidate
// in /shared/models (never a link); otherwise a download or its resume.
// Only the pinned identity decides a match. These are the plan's regression
// tests 1-14, on small disposable fixtures, with no GPU and no network.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import {
    DownloadError,
    artifactAcquisition,
    artifactPaths,
    downloadArtifact,
    downloadSnapshotFile,
    inspectArtifact,
    removeArtifact,
    snapshotPaths,
    verifyArtifact,
} from '../src/controller/downloader.mjs';
import { createWeightStores } from '../src/controller/weightStores.mjs';

const SIZE = 256 * 1024 + 7;
const CHUNK = 64 * 1024;
const PAYLOAD = crypto.randomBytes(SIZE);
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const gitOid = (bytes) => crypto.createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const COMMIT = 'c'.repeat(40);
const ARTIFACT = Object.freeze({ repo: 'org/model-GGUF', file: 'model-Q8_0.gguf', revision: 'main', commit: COMMIT, size: SIZE, sha256: sha256(PAYLOAD) });
const GIB = 1024 ** 3;
const MIB = 1024 ** 2;

function corrupt(bytes, at = 1000) {
    const copy = Buffer.from(bytes);
    copy[at] ^= 0xff;
    return copy;
}

function setup(t) {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-c12-'));
    t.after(() => fs.rmSync(base, { recursive: true, force: true }));
    const dataDir = path.join(base, 'data');
    const root = path.join(dataDir, 'models', 'gguf');
    const shared = path.join(base, 'shared', 'models');
    fs.mkdirSync(shared, { recursive: true });
    const requests = [];
    // Serves any pinned fixture by its file name.
    const bodies = new Map([[ARTIFACT.file, PAYLOAD]]);
    const fetchImpl = async (url, options = {}) => {
        const range = new Headers(options.headers).get('range');
        requests.push({ url: String(url), range });
        const body = bodies.get(decodeURIComponent(String(url).split('/').pop()));
        if (!body) return new Response('missing', { status: 404 });
        const start = range ? Number(/bytes=(\d+)-/.exec(range)[1]) : 0;
        const slice = body.subarray(start);
        return new Response(slice, range
            ? { status: 206, headers: { 'content-length': String(slice.length), 'content-range': `bytes ${start}-${body.length - 1}/${body.length}` } }
            : { status: 200, headers: { 'content-length': String(body.length) } });
    };
    const common = { fetchImpl, baseUrl: 'http://hf.invalid', sleep: async () => {}, chunkBytes: CHUNK };
    const get = (options = {}) => downloadArtifact({ artifact: ARTIFACT, root, adoptFrom: [shared], ...common, ...options });
    const place = (relative, bytes = PAYLOAD) => {
        const file = path.join(shared, relative);
        fs.mkdirSync(path.dirname(file), { recursive: true });
        fs.writeFileSync(file, bytes);
        return file;
    };
    return { base, dataDir, root, shared, requests, bodies, fetchImpl, common, get, place, paths: artifactPaths({ root, artifact: ARTIFACT }) };
}

const meta = (paths) => JSON.parse(fs.readFileSync(paths.meta, 'utf8'));
const fileSha = (file) => sha256(fs.readFileSync(file));
const stagingLeft = (paths) => fs.existsSync(path.dirname(paths.partial))
    ? fs.readdirSync(path.dirname(paths.partial)).filter((name) => name.includes('.copy-'))
    : [];

function seedPartial(s, bytes) {
    fs.mkdirSync(path.dirname(s.paths.partial), { recursive: true });
    fs.writeFileSync(s.paths.partial, PAYLOAD.subarray(0, bytes));
    fs.writeFileSync(s.paths.identity, JSON.stringify({ repo: ARTIFACT.repo, file: ARTIFACT.file, commit: COMMIT, size: SIZE, sha256: ARTIFACT.sha256 }));
}

// ------------------------------------------------------------------ copies

test('a verified candidate in /shared/models is copied, hashed as it is written, and never linked', async (t) => {
    const s = setup(t);
    // Any name, any depth: only the size and digest decide.
    const source = s.place('someone/else/renamed.bin');
    const progress = [];
    const result = await s.get({ onProgress: (p) => progress.push(p.phase), progressIntervalMs: 0 });
    assert.deepEqual(result.provenance, { file: ARTIFACT.file, source, method: 'copy', bytes: SIZE });
    assert.equal(s.requests.length, 0, 'nothing downloaded');
    assert.notEqual(fs.statSync(s.paths.file).ino, fs.statSync(source).ino, 'a private copy, not a link');
    assert.equal(fs.statSync(s.paths.file).nlink, 1);
    assert.ok(progress.includes('copying'));
    assert.deepEqual(meta(s.paths).provenance, result.provenance);
    const stat = fs.lstatSync(s.paths.file);
    assert.deepEqual(meta(s.paths).stat, { dev: stat.dev, ino: stat.ino, size: stat.size, mtimeMs: stat.mtimeMs, ctimeMs: stat.ctimeMs });
    // A second Run finds it verified and unchanged: no hashing, no download, no copy.
    const again = await s.get();
    assert.equal(again.current, true);
    assert.equal(again.reverified, undefined);
    assert.deepEqual(stagingLeft(s.paths), []);
    // The agent's own copy frees its size on Delete; the shared file stays.
    const freed = await removeArtifact({ root: s.root, artifact: ARTIFACT });
    assert.ok(freed >= SIZE && freed < SIZE + 4096, String(freed));
    assert.ok(fs.existsSync(source));
});

test('1. a fresh HF snapshot is adopted from /shared/models with git-oid and sha256 files, then a partial and a mixed one', async (t) => {
    const s = setup(t);
    const config = Buffer.from('{"architectures":["Test"],"hidden_size":8}\n');
    const weights = PAYLOAD;
    const extra = corrupt(PAYLOAD, 5);
    const files = [
        { path: 'config.json', size: config.length, gitOid: gitOid(config) },
        { path: 'model.safetensors', size: weights.length, sha256: sha256(weights) },
    ];
    const source = { type: 'hf-snapshot', repo: 'org/model', commit: COMMIT, files, size: config.length + weights.length };
    s.bodies.set('extra.safetensors', extra);
    const make = () => createWeightStores({
        dataDir: s.dataDir,
        env: {},
        hfBaseUrl: 'http://hf.invalid',
        sharedModelsRoot: s.shared,
        downloadSnapshot: (options) => downloadSnapshotFile({ ...options, fetchImpl: s.fetchImpl, sleep: async () => {} }),
        state: () => ({}),
        save: () => {},
        activeArtifact: () => null,
    })['hf-snapshot'];
    const store = make();
    const hfRoot = path.join(s.dataDir, 'models', 'hf');
    const { dir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
    s.place('a/config.json', config);
    s.place('b/weights.bin', weights);
    assert.equal(fs.existsSync(dir), false, 'no final directory beforehand');
    const plan = await store.plan(source);
    assert.deepEqual(plan.files.map((entry) => entry.method), ['copy', 'copy']);
    const fetched = await store.fetch({ artifact: source });
    assert.equal(fetched.bytesTransferred, 0);
    assert.equal(s.requests.length, 0);
    assert.deepEqual(fetched.provenance.map((entry) => [entry.file, entry.method]), [['config.json', 'copy'], ['model.safetensors', 'copy']]);
    assert.equal(fileSha(path.join(dir, 'model.safetensors')), sha256(weights));
    assert.equal(gitOid(fs.readFileSync(path.join(dir, 'config.json'))), files[0].gitOid);
    assert.equal((await store.state(source)).state, 'complete');

    // A partly present snapshot plus a file only the network has: mixed sources, complete only when all verify.
    const bigger = { ...source, files: [...files, { path: 'extra.safetensors', size: extra.length, sha256: sha256(extra) }] };
    bigger.size += extra.length;
    const second = await make().fetch({ artifact: bigger });
    assert.deepEqual(second.provenance.map((entry) => [entry.file, entry.method, Boolean(entry.current)]),
        [['config.json', 'copy', true], ['model.safetensors', 'copy', true], ['extra.safetensors', 'download', false]]);
    assert.equal(second.bytesTransferred, extra.length);
    assert.equal((await make().state(bigger)).state, 'complete');
});

// -------------------------------------------------------- bound verification

test('2. a write during hashing, even after the last chunk, is never recorded as verified, and the next Run does not trust it', async (t) => {
    // In place: the file at the store path is rewritten while it is hashed.
    const s = setup(t);
    fs.mkdirSync(path.dirname(s.paths.file), { recursive: true });
    fs.writeFileSync(s.paths.file, PAYLOAD);
    let written = false;
    const rewriteAtEnd = (file) => (p) => {
        if (p.phase === 'verifying' && p.bytes === SIZE && !written) {
            written = true;
            const fd = fs.openSync(file, 'r+');
            fs.writeSync(fd, Buffer.from([PAYLOAD[0] ^ 0xff]), 0, 1, 0);
            fs.closeSync(fd);
        }
    };
    await assert.rejects(s.get({ adoptFrom: [], onProgress: rewriteAtEnd(s.paths.file), progressIntervalMs: 0 }),
        (err) => err instanceof DownloadError && err.code === 'CHANGED_WHILE_VERIFYING');
    assert.equal(fs.existsSync(s.paths.meta), false, 'no record pairs the digest with bytes that were not hashed');
    const next = await s.get({ adoptFrom: [] });
    assert.equal(next.provenance.method, 'download', 'the changed bytes were hashed again, refused and replaced');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);

    // Copy: the shared source is rewritten after the last chunk was read; that candidate is not used.
    const c = setup(t);
    const source = c.place('model-Q8_0.gguf');
    written = false;
    const copied = await c.get({ onProgress: (p) => {
        if (p.phase === 'copying' && p.bytes === SIZE && !written) {
            written = true;
            const fd = fs.openSync(source, 'r+');
            fs.writeSync(fd, Buffer.from([PAYLOAD[0] ^ 0xff]), 0, 1, 0);
            fs.closeSync(fd);
        }
    }, progressIntervalMs: 0 });
    assert.equal(copied.provenance.method, 'download');
    assert.ok(copied.notes.some((note) => note.includes('changed while it was copied')), copied.notes.join('; '));
    assert.equal(fileSha(c.paths.file), ARTIFACT.sha256);
    assert.equal(meta(c.paths).stat.ino, fs.statSync(c.paths.file).ino);
});

test('2b. a recorded file whose bytes change later (same size) is hashed again at the next Run and replaced', async (t) => {
    const s = setup(t);
    await s.get({ adoptFrom: [] });
    const fd = fs.openSync(s.paths.file, 'r+');
    fs.writeSync(fd, Buffer.from([PAYLOAD[10] ^ 0xff]), 0, 1, 10);
    fs.closeSync(fd);
    const again = await s.get({ adoptFrom: [] });
    assert.equal(again.provenance.method, 'download');
    assert.equal(s.requests.length, 2);
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
});

test('3. a symbolic link at the store path, or in an ancestor, is never adopted or trusted', async (t) => {
    const s = setup(t);
    const outside = path.join(s.base, 'outside.gguf');
    fs.writeFileSync(outside, PAYLOAD);
    fs.mkdirSync(path.dirname(s.paths.file), { recursive: true });
    fs.symlinkSync(outside, s.paths.file);
    const result = await s.get({ adoptFrom: [] });
    assert.equal(result.provenance.method, 'download');
    assert.ok(result.notes.some((note) => note.includes('symbolic link')));
    assert.equal(fs.lstatSync(s.paths.file).isFile(), true, 'the link was replaced, not followed');
    assert.deepEqual(fs.readFileSync(outside), PAYLOAD, 'its target is untouched');
    // A verified file later swapped for a link to a correct copy: refused, then its target's edits never count.
    fs.rmSync(s.paths.file);
    fs.symlinkSync(outside, s.paths.file);
    assert.equal((await inspectArtifact({ root: s.root, artifact: ARTIFACT })).state, 'absent');
    await assert.rejects(verifyArtifact({ root: s.root, artifact: ARTIFACT }), (err) => err.code === 'CHANGED_AFTER_VERIFY');

    // A link in an ancestor directory: refused before anything is read, created or downloaded.
    const a = setup(t);
    const elsewhere = path.join(a.base, 'elsewhere');
    fs.mkdirSync(path.join(elsewhere, COMMIT), { recursive: true });
    fs.writeFileSync(path.join(elsewhere, COMMIT, ARTIFACT.file), PAYLOAD);
    fs.mkdirSync(path.join(a.root, 'org'), { recursive: true });
    fs.symlinkSync(elsewhere, path.join(a.root, 'org', 'model-GGUF'));
    await assert.rejects(a.get({ adoptFrom: [] }), (err) => err.code === 'UNSAFE_PATH');
    assert.equal(a.requests.length, 0);
    assert.equal(fs.existsSync(path.join(elsewhere, COMMIT, `${ARTIFACT.file}.json`)), false, 'nothing recorded through the link');
});

// ------------------------------------------------------- staging and resume

test('4. a wrong same-size candidate beside a valid partial: the download resumes from the partial (also after a copy error or a Stop)', async (t) => {
    const s = setup(t);
    seedPartial(s, 131_072);
    s.place('model-Q8_0.gguf', corrupt(PAYLOAD));
    const result = await s.get();
    assert.deepEqual(s.requests.map((request) => request.range), ['bytes=131072-']);
    assert.equal(result.provenance.method, 'download');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
    assert.deepEqual(stagingLeft(s.paths), []);

    // A read error while copying is an error, not "no candidate"; the partial stays.
    const e = setup(t);
    seedPartial(e, 131_072);
    e.place('model-Q8_0.gguf');
    const failingRead = {
        ...fs,
        promises: {
            ...fs.promises,
            open: async (file, flags, mode) => {
                const handle = await fs.promises.open(file, flags, mode);
                if (!String(file).startsWith(e.shared)) return handle;
                return { fd: handle.fd, stat: () => handle.stat(), close: () => handle.close(),
                    read: async () => { throw Object.assign(new Error('EIO'), { code: 'EIO' }); } };
            },
        },
    };
    await assert.rejects(e.get({ fsApi: failingRead }), (err) => err.code === 'COPY_FAILED');
    assert.equal(fs.statSync(e.paths.partial).size, 131_072);
    assert.ok(fs.existsSync(e.paths.identity));
    assert.deepEqual(stagingLeft(e.paths), []);
    fs.rmSync(path.join(e.shared, 'model-Q8_0.gguf'));
    await e.get();
    assert.deepEqual(e.requests.map((request) => request.range), ['bytes=131072-']);

    // A Stop while copying.
    const c = setup(t);
    seedPartial(c, 131_072);
    const source = c.place('model-Q8_0.gguf');
    const stop = new AbortController();
    await assert.rejects(c.get({ signal: stop.signal, progressIntervalMs: 0, onProgress: (p) => p.phase === 'copying' && stop.abort() }),
        (err) => err.code === 'ABORTED');
    assert.equal(fs.statSync(c.paths.partial).size, 131_072);
    assert.deepEqual(stagingLeft(c.paths), []);
    fs.rmSync(source);
    await c.get();
    assert.deepEqual(c.requests.map((request) => request.range), ['bytes=131072-']);
});

test('5. a copy with too little free space is refused before copying, and admission and its preview say so', async (t) => {
    const s = setup(t);
    s.place('model-Q8_0.gguf');
    let reads = 0;
    const counting = { ...fs, promises: { ...fs.promises, open: async (file, flags, mode) => {
        const handle = await fs.promises.open(file, flags, mode);
        if (!String(file).startsWith(s.shared)) return handle;
        return { fd: handle.fd, stat: () => handle.stat(), close: () => handle.close(),
            read: (...args) => { reads += 1; return handle.read(...args); } };
    } } };
    await assert.rejects(s.get({ fsApi: counting, statfs: async () => ({ bavail: 1, bsize: 4096 }) }),
        (err) => err instanceof DownloadError && err.code === 'INSUFFICIENT_SPACE' && err.message.includes('private copy'));
    assert.equal(reads, 0);
    assert.deepEqual(stagingLeft(s.paths), []);
    assert.deepEqual(await artifactAcquisition({ root: s.root, artifact: ARTIFACT, candidates: [{ path: path.join(s.shared, 'model-Q8_0.gguf'), root: s.shared }] }),
        { file: ARTIFACT.file, method: 'copy', bytesNeeded: SIZE, source: path.join(s.shared, 'model-Q8_0.gguf') });

    const h = controllerHarness(t, s, { freeBytes: SIZE / 2 });
    const overview = await h.controller.overview({ preview: { modelId: 'tiny', runnerId: 'llama.cpp' } });
    assert.equal(overview.preview.acquisition.files[0].method, 'copy');
    assert.equal(overview.preview.admission.status, 'insufficient-now');
    await assert.rejects(h.controller.run({ requestId: 'request-copy-0001', ...h.run }), (err) => err.code === 'admission_insufficient_now');
    assert.equal(h.downloads.length, 0);
});

// ------------------------------------------------------ in place, controller

function controllerHarness(t, s, { freeBytes = 300 * GIB, download } = {}) {
    const GPT = loadSeedCatalog().find((model) => model.id === 'gpt-oss-20b');
    const tiny = validateModel({ ...GPT, id: 'tiny', displayName: 'Tiny', sources: { gguf: { type: 'huggingface', ...ARTIFACT } } }, { seed: true });
    const downloads = [];
    const started = [];
    const controller = createController({
        dataDir: s.dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: [tiny],
        sharedModelsRoot: s.shared,
        snapshot: async () => ({
            gpu: { available: true, name: 'Test GPU', totalBytes: 24 * GIB, usedBytes: 0, freeBytes: 24 * GIB, processes: [] },
            memory: { totalBytes: 64 * GIB, availableBytes: 60 * GIB },
            disk: { freeBytes, totalBytes: 500 * GIB },
            cpus: 8,
        }),
        download: (options) => {
            downloads.push(options.artifact.file);
            return (download || downloadArtifact)({ ...options, ...s.common });
        },
        verify: verifyArtifact,
        fileExists: () => true,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner: ({ log }) => {
            let exit;
            const exited = new Promise((resolve) => { exit = resolve; });
            const handle = { pid: 4242, exited, running: true, async stop() { handle.running = false; exit({ code: 0, signal: 'SIGTERM', error: null }); return exited; } };
            log.append('stdout', 'load_tensors: offloaded 25/25 layers to GPU');
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        pollMs: 2,
        stopGraceMs: 50,
    });
    t.after(async () => { await controller.stop().catch(() => {}); });
    return { controller, downloads, started, run: { modelId: 'tiny', runnerId: 'llama.cpp' } };
}

async function until(predicate, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

test('6 and 7. a correct file at the store path is adopted in place with no download, even with less free disk than its size', async (t) => {
    const s = setup(t);
    // The operator hard-links a retained file into the canonical store path.
    const retained = path.join(s.base, 'retained.gguf');
    fs.writeFileSync(retained, PAYLOAD);
    fs.mkdirSync(path.dirname(s.paths.file), { recursive: true });
    fs.linkSync(retained, s.paths.file);
    assert.deepEqual(await artifactAcquisition({ root: s.root, artifact: ARTIFACT }), { file: ARTIFACT.file, method: 'in-place', bytesNeeded: 0 });
    const h = controllerHarness(t, s, { freeBytes: SIZE / 2 });
    const accepted = await h.controller.run({ requestId: 'request-inplace-01', ...h.run });
    assert.equal(accepted.accepted, true);
    assert.deepEqual(accepted.deployment.acquisition, [{ file: ARTIFACT.file, method: 'in-place', bytesNeeded: 0 }]);
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.equal(s.requests.length, 0, 'no network bytes');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
    assert.equal(fs.statSync(s.paths.file).ino, fs.statSync(retained).ino, 'adopted where it is');
    const status = await h.controller.status();
    const lines = status.logs.map((line) => line.line);
    assert.ok(lines.includes(`${ARTIFACT.file}: adopted in place at ${s.paths.file} (0 bytes), verified`), lines.join('\n'));
    assert.ok(lines.some((line) => line.includes('has 1 other hard link')));
    assert.deepEqual(status.deployment.provenance, [{ file: ARTIFACT.file, source: s.paths.file, method: 'in-place', bytes: 0 }]);
    assert.ok(!lines.some((line) => line.includes('undefined')));
});

test('8. provenance names every file\'s source and method, for a single GGUF, each shard of a split GGUF and each snapshot file', async (t) => {
    const s = setup(t);
    // Distinct sizes: the plan matches candidates by size alone.
    const shards = [0, 1, 2].map((index) => corrupt(PAYLOAD.subarray(0, SIZE - index), 100 + index));
    const names = shards.map((_, index) => `split/m-0000${index + 1}-of-00003.gguf`);
    names.forEach((name, index) => s.bodies.set(path.basename(name), shards[index]));
    const source = validateModel({ id: 'split', architecture: 'moe', sources: { gguf: {
        type: 'huggingface', repo: 'org/split-GGUF', file: names[0], revision: 'main', commit: COMMIT,
        shards: names.map((file, index) => ({ file, size: shards[index].length, sha256: sha256(shards[index]) })),
    } } }, { seed: true }).sources.gguf;
    const store = createWeightStores({
        dataDir: s.dataDir, env: {}, hfBaseUrl: 'http://hf.invalid', sharedModelsRoot: s.shared,
        download: (options) => downloadArtifact({ ...options, ...s.common }),
        inspect: inspectArtifact,
        remove: removeArtifact,
        state: () => ({}), save: () => {}, activeArtifact: () => null,
    }).huggingface;
    // 12: shard 1 in place, shard 2 copied (after a corrupt decoy with the same name), shard 3 downloaded.
    const first = artifactPaths({ root: s.root, artifact: { ...source, shards: undefined, file: names[0], size: shards[0].length, sha256: sha256(shards[0]) } });
    fs.mkdirSync(path.dirname(first.file), { recursive: true });
    fs.writeFileSync(first.file, shards[0]);
    s.place(path.basename(names[1]), corrupt(shards[1], 7));
    const good = s.place('elsewhere/second.gguf', shards[1]);
    const plan = await store.plan(source);
    assert.deepEqual(plan.files.map((entry) => entry.method), ['in-place', 'copy', 'download']);
    assert.equal(plan.bytesNeeded, shards[1].length + shards[2].length);
    const fetched = await store.fetch({ artifact: source });
    assert.deepEqual(fetched.provenance, [
        { file: names[0], source: first.file, method: 'in-place', bytes: 0 },
        { file: names[1], source: good, method: 'copy', bytes: shards[1].length },
        { file: names[2], source: `http://hf.invalid/org/split-GGUF/resolve/${COMMIT}/split/m-00003-of-00003.gguf`, method: 'download', bytes: shards[2].length },
    ]);
    assert.ok(fetched.notes.some((note) => note.includes('its bytes do not match the pinned digest')));
    assert.equal((await store.state(source)).state, 'complete');
    assert.ok(!JSON.stringify(fetched).includes('undefined'));
});

test('9. a /shared directory swapped for a link during the walk leads to no adoption from outside', async (t) => {
    const s = setup(t);
    const sub = path.join(s.shared, 'sub');
    fs.mkdirSync(sub);
    const outside = path.join(s.base, 'external');
    fs.mkdirSync(outside);
    fs.writeFileSync(path.join(outside, ARTIFACT.file), PAYLOAD);
    // The walk sees `sub` as a directory; it is swapped for a link before the walk opens it.
    const fsApi = { ...fs, promises: { ...fs.promises, opendir: async (dir, options) => {
        if (dir === sub && !fs.lstatSync(sub).isSymbolicLink()) {
            fs.renameSync(sub, `${sub}-old`);
            fs.symlinkSync(outside, sub);
        }
        return fs.promises.opendir(dir, options);
    } } };
    const result = await s.get({ fsApi });
    assert.equal(result.provenance.method, 'download');
    assert.ok(result.notes.some((note) => note.includes('reached through a symbolic link')), result.notes.join('; '));
});

test('10. Stop, Cancel and drain during a copy or a verification settle within a chunk; only the staging file goes', async (t) => {
    // Directly, during in-place verification.
    const v = setup(t);
    fs.mkdirSync(path.dirname(v.paths.file), { recursive: true });
    fs.writeFileSync(v.paths.file, PAYLOAD);
    const stop = new AbortController();
    let ticks = 0;
    await assert.rejects(v.get({ adoptFrom: [], signal: stop.signal, progressIntervalMs: 0, onProgress: (p) => {
        if (p.phase === 'verifying') { ticks += 1; stop.abort(); }
    } }), (err) => err.code === 'ABORTED');
    assert.equal(ticks, 1, 'stopped after one chunk');
    assert.equal(fs.existsSync(v.paths.meta), false);

    // Through the controller: a slow copy (each chunk waits) with a partial beside it.
    for (const how of ['stop', 'cancel', 'drain']) {
        const s = setup(t);
        seedPartial(s, 131_072);
        s.place('model-Q8_0.gguf');
        let chunks = 0;
        const slow = { ...fs, promises: { ...fs.promises, open: async (file, flags, mode) => {
            const handle = await fs.promises.open(file, flags, mode);
            if (!String(file).startsWith(s.shared)) return handle;
            return { fd: handle.fd, stat: () => handle.stat(), close: () => handle.close(),
                read: async (...args) => { chunks += 1; await new Promise((resolve) => setTimeout(resolve, 20)); return handle.read(...args); } };
        } } };
        const h = controllerHarness(t, s, { download: (options) => downloadArtifact({ ...options, fsApi: slow, chunkBytes: 4096 }) });
        await h.controller.run({ requestId: `request-${how}-000001`, ...h.run });
        await until(() => chunks >= 2);
        const began = Date.now();
        if (how === 'stop') await h.controller.stop();
        else if (how === 'cancel') await h.controller.cancelDownload();
        else await h.controller.drain();
        const settled = Date.now() - began;
        assert.ok(settled < 1000, `${how} took ${settled} ms`);
        const seen = chunks;
        await new Promise((resolve) => setTimeout(resolve, 60));
        assert.equal(chunks, seen, `${how}: no chunk read after it settled`);
        assert.deepEqual(stagingLeft(s.paths), [], how);
        assert.equal(fs.statSync(s.paths.partial).size, 131_072, how);
        assert.ok(fs.existsSync(s.paths.identity), how);
        assert.equal(fs.existsSync(s.paths.file), false, how);
    }
});

test('11. Delete counts an inode once when all its links go, and not at all while a link remains elsewhere', async (t) => {
    const s = setup(t);
    const files = [
        { path: 'model-a.safetensors', size: SIZE, sha256: ARTIFACT.sha256 },
        { path: 'model-b.safetensors', size: SIZE, sha256: ARTIFACT.sha256 },
    ];
    const source = { type: 'hf-snapshot', repo: 'org/model', commit: COMMIT, files, size: 2 * SIZE };
    const store = () => createWeightStores({ dataDir: s.dataDir, env: {}, sharedModelsRoot: s.shared,
        state: () => ({}), save: () => {}, activeArtifact: () => null })['hf-snapshot'];
    const hfRoot = path.join(s.dataDir, 'models', 'hf');
    const { dir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
    const make = () => {
        fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(path.join(dir, files[0].path), PAYLOAD);
        fs.linkSync(path.join(dir, files[0].path), path.join(dir, files[1].path));
    };
    make();
    assert.equal(await store().remove(source), SIZE);
    assert.equal(fs.existsSync(dir), false);
    make();
    fs.linkSync(path.join(dir, files[0].path), path.join(s.base, 'kept-elsewhere'));
    assert.equal(await store().remove(source), 0);
});

test('13. many same-size decoys: at most 3 are hashed per file, and the walk stops promptly on a Stop', async (t) => {
    const s = setup(t);
    for (let index = 0; index < 10; index += 1) s.place(`decoy-${index}.bin`, corrupt(PAYLOAD, index));
    const result = await s.get();
    assert.equal(result.notes.filter((note) => note.includes('do not match')).length, 3);
    assert.equal(result.provenance.method, 'download');

    const w = setup(t);
    for (let index = 0; index < 50; index += 1) w.place(`d${index}/x.bin`, Buffer.alloc(1));
    const stop = new AbortController();
    let opened = 0;
    const fsApi = { ...fs, promises: { ...fs.promises, opendir: async (dir, options) => {
        opened += 1;
        if (opened === 3) stop.abort();
        return fs.promises.opendir(dir, options);
    } } };
    await assert.rejects(w.get({ fsApi, signal: stop.signal }), (err) => err.code === 'ABORTED');
    assert.ok(opened <= 4, `the walk went on: ${opened} directories`);
});

test('14. a file whose stat changed at the last check before loading is hashed again; changed bytes are refused', async (t) => {
    const s = setup(t);
    await s.get({ adoptFrom: [] });
    // Same bytes, new timestamps: hashed again and kept.
    const later = new Date(Date.now() + 5000);
    fs.utimesSync(s.paths.file, later, later);
    assert.equal((await verifyArtifact({ root: s.root, artifact: ARTIFACT })).reverified, true);
    assert.equal((await verifyArtifact({ root: s.root, artifact: ARTIFACT })).reverified, false);
    // Other bytes: refused before any load.
    const fd = fs.openSync(s.paths.file, 'r+');
    fs.writeSync(fd, Buffer.from([PAYLOAD[3] ^ 0xff]), 0, 1, 3);
    fs.closeSync(fd);
    await assert.rejects(verifyArtifact({ root: s.root, artifact: ARTIFACT }), (err) => err.code === 'CHANGED_AFTER_VERIFY');
});

test('a file verified before stats were recorded is checked once, then trusted by its stat', async (t) => {
    const s = setup(t);
    await s.get({ adoptFrom: [] });
    const record = meta(s.paths);
    delete record.stat;
    fs.writeFileSync(s.paths.meta, JSON.stringify(record));
    assert.equal((await s.get()).reverified, true);
    assert.equal((await s.get()).reverified, undefined);
    assert.equal(s.requests.length, 1);
});
