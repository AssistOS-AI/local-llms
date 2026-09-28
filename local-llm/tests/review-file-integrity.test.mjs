// Model-file integrity and resume (C12, DS002): a private copy is published
// only as the bytes on disk that were read back through its own descriptor;
// a /shared candidate is opened from the checked descriptor of its anchored
// root, so a root swapped at the open cannot supply it; and an unchecked
// same-size candidate cannot turn a feasible resume into a disk refusal.
// Small disposable fixtures only; no GPU, no network. Linux: needs a real
// /proc/self/fd.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import {
    artifactAcquisition,
    artifactPaths,
    downloadArtifact,
    inspectArtifact,
    removeArtifact,
    verifyArtifact,
} from '../src/controller/downloader.mjs';
import { createWeightStores } from '../src/controller/weightStores.mjs';
import { walkShared } from '../src/controller/workspaceReuse.mjs';

const SIZE = 256 * 1024 + 7;
const CHUNK = 64 * 1024;
const PAYLOAD = crypto.randomBytes(SIZE);
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const COMMIT = 'c'.repeat(40);
const ARTIFACT = Object.freeze({ repo: 'org/model-GGUF', file: 'model-Q8_0.gguf', revision: 'main', commit: COMMIT, size: SIZE, sha256: sha256(PAYLOAD) });
const GIB = 1024 ** 3;
const PARTIAL = SIZE - 50_000;
const { O_DIRECTORY } = fs.constants;

function setup(t) {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-integrity-'));
    t.after(() => fs.rmSync(base, { recursive: true, force: true }));
    const dataDir = path.join(base, 'data');
    const root = path.join(dataDir, 'models', 'gguf');
    const shared = path.join(base, 'shared', 'models');
    fs.mkdirSync(shared, { recursive: true });
    const requests = [];
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
    return { base, dataDir, root, shared, requests, bodies, common, get, place, paths: artifactPaths({ root, artifact: ARTIFACT }) };
}

const meta = (paths) => JSON.parse(fs.readFileSync(paths.meta, 'utf8'));
const fileSha = (file) => sha256(fs.readFileSync(file));
const stagingLeft = (paths) => (fs.existsSync(path.dirname(paths.partial))
    ? fs.readdirSync(path.dirname(paths.partial)).filter((name) => name.includes('.copy-'))
    : []);
const stagingPath = (paths) => {
    const names = stagingLeft(paths);
    assert.equal(names.length, 1, 'exactly one staging file');
    return path.join(path.dirname(paths.partial), names[0]);
};

function flipByte(file, at) {
    const fd = fs.openSync(file, 'r+');
    try {
        const one = Buffer.alloc(1);
        fs.readSync(fd, one, 0, 1, at);
        one[0] ^= 0xff;
        fs.writeSync(fd, one, 0, 1, at);
    } finally {
        fs.closeSync(fd);
    }
}

function seedPartial(paths, artifact, bytes, payload = PAYLOAD) {
    fs.mkdirSync(path.dirname(paths.partial), { recursive: true });
    fs.writeFileSync(paths.partial, payload.subarray(0, bytes));
    fs.writeFileSync(paths.identity, JSON.stringify({ repo: artifact.repo, file: artifact.file, commit: artifact.commit, size: artifact.size, sha256: artifact.sha256 }));
    return fs.statSync(paths.partial).ino;
}

// A same-size file that takes no disk: any agent can plant one under /shared.
function sparseDecoy(dir, name, size) {
    const file = path.join(dir, name);
    const fd = fs.openSync(file, 'w');
    fs.ftruncateSync(fd, size);
    fs.closeSync(fd);
    return file;
}

// A statfs whose free space is `free` less what the store's model files grew
// by since it was made, so a copy or a resume uses up what it writes.
function diskModel(dir, free) {
    const usage = () => {
        let total = 0;
        const walk = (current) => {
            for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
                const full = path.join(current, entry.name);
                if (entry.isDirectory()) walk(full);
                else if (entry.isFile() && /\.(gguf|partial)$|\.partial\.copy-/.test(entry.name)) total += fs.statSync(full).size;
            }
        };
        if (fs.existsSync(dir)) walk(dir);
        return total;
    };
    const start = usage();
    return async () => ({ bavail: Math.max(0, free - (usage() - start)), bsize: 1 });
}

// Counts the reads of every /shared file the downloader opens.
function countingShared(s) {
    const counts = { opens: 0, reads: 0 };
    const fsApi = { ...fs, promises: { ...fs.promises, open: async (file, flags, mode) => {
        const handle = await fs.promises.open(file, flags, mode);
        if (flags & O_DIRECTORY) return handle;
        if (!(await fs.promises.readlink(`/proc/self/fd/${handle.fd}`)).startsWith(`${fs.realpathSync(s.shared)}${path.sep}`)) return handle;
        counts.opens += 1;
        return { fd: handle.fd, stat: () => handle.stat(), close: () => handle.close(),
            read: (...args) => { counts.reads += 1; return handle.read(...args); } };
    } } };
    return { counts, fsApi };
}

// ------------------------------------------------------------ C3: the copy

test('C3 positive control: an intact copy is read back as the verifying phase and published with the stat of what was hashed', async (t) => {
    const s = setup(t);
    seedPartial(s.paths, ARTIFACT, 131_072);
    const source = s.place('elsewhere/any.bin');
    const progress = [];
    const result = await s.get({ progressIntervalMs: 0, onProgress: (p) => progress.push([p.phase, p.bytes]) });
    assert.deepEqual(result.provenance, { file: ARTIFACT.file, source, method: 'copy', bytes: SIZE });
    assert.equal(s.requests.length, 0);
    const firstVerify = progress.findIndex(([phase]) => phase === 'verifying');
    const lastCopy = progress.findLastIndex(([phase]) => phase === 'copying');
    assert.ok(lastCopy >= 0 && firstVerify > lastCopy, 'the copy is read back after it was written');
    assert.equal(Math.max(...progress.filter(([phase]) => phase === 'verifying').map(([, bytes]) => bytes)), SIZE, 'every byte was read back');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
    const stat = fs.lstatSync(s.paths.file);
    assert.deepEqual(meta(s.paths).stat, { dev: stat.dev, ino: stat.ino, size: stat.size, mtimeMs: stat.mtimeMs, ctimeMs: stat.ctimeMs });
    assert.equal(fs.existsSync(s.paths.partial), false, 'the partial goes once a verified copy is published');
    assert.deepEqual(stagingLeft(s.paths), []);
    assert.equal((await verifyArtifact({ root: s.root, artifact: ARTIFACT })).reverified, false);
});

test('C3: a byte of the staging file changed while the copy continues is never published, recorded or trusted', async (t) => {
    const s = setup(t);
    seedPartial(s.paths, ARTIFACT, 131_072);
    s.place(ARTIFACT.file);
    let mutated = false;
    const mutateStaging = (p) => {
        if (mutated || p.phase !== 'copying' || p.bytes >= SIZE) return;
        flipByte(stagingPath(s.paths), 0);
        mutated = true;
    };
    await assert.rejects(s.get({ progressIntervalMs: 0, onProgress: mutateStaging }),
        (err) => err.code === 'SHA256_MISMATCH' && err.message.includes('partial download is kept'));
    assert.equal(mutated, true);
    assert.equal(fs.existsSync(s.paths.file), false, 'nothing published');
    assert.equal(fs.existsSync(s.paths.meta), false, 'nothing recorded');
    assert.deepEqual(stagingLeft(s.paths), [], 'the staging file is removed');
    assert.equal(fs.statSync(s.paths.partial).size, 131_072, 'the partial is kept');
    assert.deepEqual(await inspectArtifact({ root: s.root, artifact: ARTIFACT }), { state: 'partial', bytes: 131_072 });
    await assert.rejects(verifyArtifact({ root: s.root, artifact: ARTIFACT }), (err) => err.code === 'CHANGED_AFTER_VERIFY');
    assert.equal(s.requests.length, 0);
    // The next Run, undisturbed, copies again and publishes the pinned bytes.
    const again = await s.get();
    assert.equal(again.provenance.method, 'copy');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
});

test('C3: a write to the staging file while it is read back is CHANGED_WHILE_VERIFYING, with nothing recorded', async (t) => {
    const s = setup(t);
    s.place(ARTIFACT.file);
    let mutated = false;
    await assert.rejects(s.get({ progressIntervalMs: 0, onProgress: (p) => {
        if (mutated || p.phase !== 'verifying' || p.bytes >= SIZE) return;
        flipByte(stagingPath(s.paths), SIZE - 1);
        mutated = true;
    } }), (err) => err.code === 'CHANGED_WHILE_VERIFYING' && err.retryable === true);
    assert.equal(mutated, true);
    assert.equal(fs.existsSync(s.paths.file), false);
    assert.equal(fs.existsSync(s.paths.meta), false);
    assert.deepEqual(stagingLeft(s.paths), []);
});

test('C3: a change after the read-back that only the descriptor\'s ctime shows stops publication before anything is recorded', async (t) => {
    // The mode change keeps the bytes, size and mtime: only the fstat of the
    // descriptor the bytes were hashed through can show it before the rename.
    const s = setup(t);
    s.place(ARTIFACT.file);
    let readBack = false;
    let changed = false;
    const fsApi = { ...fs, promises: { ...fs.promises, mkdir: async (...args) => {
        if (readBack && !changed) {
            fs.chmodSync(stagingPath(s.paths), 0o600);
            changed = true;
        }
        return fs.promises.mkdir(...args);
    } } };
    await assert.rejects(s.get({ fsApi, progressIntervalMs: 0, onProgress: (p) => {
        if (p.phase === 'verifying' && p.bytes === SIZE) readBack = true;
    } }), (err) => err.code === 'CHANGED_WHILE_VERIFYING');
    assert.equal(changed, true);
    assert.equal(fs.existsSync(s.paths.file), false);
    assert.equal(fs.existsSync(s.paths.meta), false);
    assert.deepEqual(stagingLeft(s.paths), []);
});

test('C3: a write after the read-back but before the rename is caught at publication; no stat is recorded for it', async (t) => {
    const s = setup(t);
    s.place(ARTIFACT.file);
    let mutated = false;
    const fsApi = { ...fs, promises: { ...fs.promises, rename: async (from, to) => {
        if (!mutated && to === s.paths.file) {
            flipByte(from, 1);
            mutated = true;
        }
        return fs.promises.rename(from, to);
    } } };
    await assert.rejects(s.get({ fsApi }), (err) => err.code === 'CHANGED_WHILE_VERIFYING');
    assert.equal(mutated, true);
    assert.equal(fs.existsSync(s.paths.meta), false, 'no record pairs the digest with the changed bytes');
    // The unrecorded file at the store path is hashed at the next Run, refused and replaced.
    const next = await s.get({ adoptFrom: [] });
    assert.equal(next.provenance.method, 'download');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
});

test('C3: a Stop while the copy is read back settles within a chunk; only the staging file goes', async (t) => {
    const s = setup(t);
    seedPartial(s.paths, ARTIFACT, 131_072);
    s.place(ARTIFACT.file);
    const stop = new AbortController();
    let verifyTicks = 0;
    await assert.rejects(s.get({ signal: stop.signal, progressIntervalMs: 0, onProgress: (p) => {
        if (p.phase !== 'verifying') return;
        verifyTicks += 1;
        stop.abort();
    } }), (err) => err.code === 'ABORTED');
    assert.equal(verifyTicks, 1, 'stopped after one chunk of the read-back');
    assert.deepEqual(stagingLeft(s.paths), []);
    assert.equal(fs.existsSync(s.paths.file), false);
    assert.equal(fs.existsSync(s.paths.meta), false);
    assert.equal(fs.statSync(s.paths.partial).size, 131_072);
    assert.ok(fs.existsSync(s.paths.identity));
});

// ------------------------------------------------------- C4: the anchored root

// The candidate found by the walk, an outside directory holding another copy
// of the pinned bytes, and an fs whose opens record each file inode opened.
async function rootCase(t, { at, swapBack = false } = {}) {
    const s = setup(t);
    const candidatePath = s.place(ARTIFACT.file);
    const outside = path.join(s.base, 'outside-root');
    fs.mkdirSync(outside);
    fs.writeFileSync(path.join(outside, ARTIFACT.file), PAYLOAD);
    const originalInode = fs.statSync(candidatePath).ino;
    const replacementInode = fs.statSync(path.join(outside, ARTIFACT.file)).ino;
    const candidates = (await walkShared(fs, [s.shared], [SIZE])).get(SIZE);
    assert.equal(candidates.length, 1);
    const opened = [];
    let swapped = false;
    const swap = () => {
        fs.renameSync(s.shared, `${s.shared}-original`);
        fs.renameSync(outside, s.shared);
        swapped = true;
    };
    const restore = () => {
        fs.renameSync(s.shared, outside);
        fs.renameSync(`${s.shared}-original`, s.shared);
    };
    const fsApi = { ...fs, promises: { ...fs.promises, open: async (file, flags, mode) => {
        const directory = Boolean(flags & O_DIRECTORY);
        // `root`: the first open of the root directory itself; `file`: the first open of the candidate's name.
        const hit = !swapped && ((at === 'root' && directory && file === s.shared)
            || (at === 'file' && !directory && path.basename(String(file)) === ARTIFACT.file
                && (file === candidatePath || String(file).startsWith('/proc/self/fd/'))));
        if (hit) swap();
        let handle;
        try {
            handle = await fs.promises.open(file, flags, mode);
        } finally {
            if (hit && swapBack) restore();
        }
        if (!directory) opened.push((await handle.stat()).ino);
        return handle;
    } } };
    const result = await downloadArtifact({ root: s.root, artifact: ARTIFACT, candidates, fsApi, ...s.common });
    return { s, result, opened, swapped: () => swapped, originalInode, replacementInode };
}

test('C4 positive control: an unmoved root supplies its own candidate, as a private copy', async (t) => {
    const c = await rootCase(t, { at: 'none' });
    assert.equal(c.result.provenance.method, 'copy');
    assert.ok(c.opened.includes(c.originalInode));
    assert.ok(!c.opened.includes(c.replacementInode));
    assert.notEqual(fs.statSync(c.result.path).ino, c.originalInode, 'a private copy, not the shared inode');
    assert.equal(c.s.requests.length, 0);
});

test('C4: a real directory substituted for the root at the candidate open never supplies the file', async (t) => {
    const c = await rootCase(t, { at: 'file' });
    assert.equal(c.swapped(), true);
    assert.ok(!c.opened.includes(c.replacementInode), 'the replacement root\'s file was opened');
    assert.equal(c.result.provenance.method, 'download');
    assert.ok(c.result.notes.some((note) => note.includes('moved while it was opened') || note.includes('root directory changed')), c.result.notes.join('; '));
    assert.equal(fileSha(c.result.path), ARTIFACT.sha256);
});

test('C4: a root substituted just before its descriptor is opened is refused as a changed root', async (t) => {
    const c = await rootCase(t, { at: 'root' });
    assert.equal(c.swapped(), true);
    assert.ok(!c.opened.includes(c.replacementInode));
    assert.equal(c.result.provenance.method, 'download');
    assert.ok(c.result.notes.some((note) => note.includes('root directory changed')), c.result.notes.join('; '));
});

test('C4: a root swapped in for the open and back right after cannot supply the file either', async (t) => {
    const c = await rootCase(t, { at: 'file', swapBack: true });
    assert.equal(c.swapped(), true);
    assert.ok(!c.opened.includes(c.replacementInode), 'the replacement root\'s file was opened');
    // The file came from the anchored root itself, which is back at its path.
    assert.equal(c.result.provenance.method, 'copy');
    assert.ok(c.opened.includes(c.originalInode));
    assert.equal(fileSha(c.result.path), ARTIFACT.sha256);
});

test('C4: without a readable /proc/self/fd a candidate is refused closed, never taken as missing', async (t) => {
    const s = setup(t);
    s.place(ARTIFACT.file);
    const fsApi = { ...fs, promises: { ...fs.promises, readlink: async () => { throw Object.assign(new Error('no proc'), { code: 'ENOENT' }); } } };
    await assert.rejects(s.get({ fsApi }), (err) => err.code === 'UNSAFE_PATH');
    assert.equal(s.requests.length, 0);
    assert.equal(fs.existsSync(s.paths.file), false);
});

// ------------------------------------------- F2: an unchecked candidate and a resume

test('F2: beside a resumable partial, a sparse same-size decoy that does not fit as a copy leaves the partial to resume in place', async (t) => {
    const s = setup(t);
    const inode = seedPartial(s.paths, ARTIFACT, PARTIAL);
    sparseDecoy(s.shared, 'anything.gguf', SIZE);
    const remaining = SIZE - PARTIAL;
    // Room for the resume and its reserve, not for a whole copy.
    const statfs = diskModel(s.root, Math.ceil(remaining * 1.05) + 16);
    const { counts, fsApi } = countingShared(s);
    const result = await s.get({ statfs, fsApi });
    assert.equal(result.provenance.method, 'download');
    assert.deepEqual(s.requests.map((request) => request.range), [`bytes=${PARTIAL}-`], 'one request, from the partial\'s end');
    assert.equal(result.bytesTransferred, remaining);
    assert.equal(fs.statSync(s.paths.file).ino, inode, 'the partial itself was completed, not rewritten');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
    assert.equal(counts.reads, 0, 'the unchecked decoy was not read');
    assert.ok(result.notes.some((note) => note.includes('does not fit') && note.includes(`${PARTIAL} of ${SIZE}`)), result.notes.join('; '));
    assert.deepEqual(stagingLeft(s.paths), []);
});

test('F2: the plan of a candidate beside a partial needs only the resume; without a partial a copy still needs its size', async (t) => {
    const s = setup(t);
    const candidates = [{ path: path.join(s.shared, 'anything.gguf') }];
    const inspected = { state: 'partial', bytes: PARTIAL };
    assert.deepEqual(await artifactAcquisition({ root: s.root, artifact: ARTIFACT, candidates, inspected }), {
        file: ARTIFACT.file, method: 'copy-or-resume', bytesNeeded: SIZE - PARTIAL, copyBytesNeeded: SIZE, source: candidates[0].path,
    });
    // The same rule from the files on disk.
    seedPartial(s.paths, ARTIFACT, PARTIAL);
    assert.equal((await artifactAcquisition({ root: s.root, artifact: ARTIFACT, candidates })).bytesNeeded, SIZE - PARTIAL);
    // An empty partial saves nothing: that is a copy of the whole size.
    assert.deepEqual(await artifactAcquisition({ root: s.root, artifact: ARTIFACT, candidates, inspected: { state: 'partial', bytes: 0 } }),
        { file: ARTIFACT.file, method: 'copy', bytesNeeded: SIZE, source: candidates[0].path });
    assert.deepEqual(await artifactAcquisition({ root: s.root, artifact: ARTIFACT, candidates, inspected: { state: 'absent', bytes: 0 } }),
        { file: ARTIFACT.file, method: 'copy', bytesNeeded: SIZE, source: candidates[0].path });
    assert.deepEqual(await artifactAcquisition({ root: s.root, artifact: ARTIFACT, inspected }),
        { file: ARTIFACT.file, method: 'download', bytesNeeded: SIZE - PARTIAL });
});

test('F2 positive control: with room for the whole copy a valid candidate beside a partial is still copied', async (t) => {
    const s = setup(t);
    seedPartial(s.paths, ARTIFACT, PARTIAL);
    const source = s.place('model-Q8_0.gguf');
    const result = await s.get({ statfs: diskModel(s.root, Math.ceil(SIZE * 1.05)) });
    assert.deepEqual(result.provenance, { file: ARTIFACT.file, source, method: 'copy', bytes: SIZE });
    assert.equal(s.requests.length, 0);
    assert.equal(fs.existsSync(s.paths.partial), false);
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
});

test('F2: with room for neither, the refusal is the resume\'s and the partial stays as it was', async (t) => {
    const s = setup(t);
    seedPartial(s.paths, ARTIFACT, PARTIAL);
    sparseDecoy(s.shared, 'anything.gguf', SIZE);
    const before = fs.statSync(s.paths.partial);
    await assert.rejects(s.get({ statfs: async () => ({ bavail: 1000, bsize: 1 }) }),
        (err) => err.code === 'INSUFFICIENT_SPACE' && err.details.required === Math.ceil((SIZE - PARTIAL) * 1.05));
    const after = fs.statSync(s.paths.partial);
    assert.deepEqual([after.ino, after.size, after.mtimeMs], [before.ino, before.size, before.mtimeMs]);
    assert.equal(s.requests.length, 0);
});

test('F2: both shards of a split GGUF resume beside their decoys when the free disk only covers the resumes', async (t) => {
    const s = setup(t);
    // Distinct sizes: candidates are matched by size alone.
    const shards = [PAYLOAD, PAYLOAD.subarray(0, SIZE - 1)];
    const names = ['split/m-00001-of-00002.gguf', 'split/m-00002-of-00002.gguf'];
    names.forEach((name, index) => s.bodies.set(path.basename(name), shards[index]));
    const source = validateModel({ id: 'split', architecture: 'moe', sources: { gguf: {
        type: 'huggingface', repo: 'org/split-GGUF', file: names[0], revision: 'main', commit: COMMIT,
        shards: names.map((file, index) => ({ file, size: shards[index].length, sha256: sha256(shards[index]) })),
    } } }, { seed: true }).sources.gguf;
    const inodes = names.map((file, index) => {
        const artifact = { ...source, shards: undefined, file, size: shards[index].length, sha256: sha256(shards[index]) };
        sparseDecoy(s.shared, `decoy-${index}.gguf`, artifact.size);
        return seedPartial(artifactPaths({ root: s.root, artifact }), artifact, PARTIAL, shards[index]);
    });
    const remaining = shards.map((bytes) => bytes.length - PARTIAL);
    const statfs = diskModel(s.root, Math.ceil((remaining[0] + remaining[1]) * 1.05));
    const store = createWeightStores({
        dataDir: s.dataDir, env: {}, hfBaseUrl: 'http://hf.invalid', sharedModelsRoot: s.shared,
        download: (options) => downloadArtifact({ ...options, ...s.common, statfs }),
        inspect: inspectArtifact, remove: removeArtifact,
        state: () => ({}), save: () => {}, activeArtifact: () => null,
    }).huggingface;
    const plan = await store.plan(source);
    assert.deepEqual(plan.files.map((entry) => [entry.method, entry.bytesNeeded]), remaining.map((bytes) => ['copy-or-resume', bytes]));
    assert.equal(plan.bytesNeeded, remaining[0] + remaining[1]);
    const fetched = await store.fetch({ artifact: source });
    assert.deepEqual(fetched.provenance.map((entry) => entry.method), ['download', 'download']);
    assert.deepEqual(s.requests.map((request) => request.range), [`bytes=${PARTIAL}-`, `bytes=${PARTIAL}-`]);
    assert.deepEqual(fetched.files.map((file) => fs.statSync(file).ino), inodes, 'each partial completed in place');
    assert.equal((await store.state(source)).state, 'complete');
});

// The controller: admission, its preview and the Run agree with the transfer.
function controllerHarness(t, s, { freeBytes, statfs }) {
    const GPT = loadSeedCatalog().find((model) => model.id === 'gpt-oss-20b');
    const tiny = validateModel({ ...GPT, id: 'tiny', displayName: 'Tiny', sources: { gguf: { type: 'huggingface', ...ARTIFACT } } }, { seed: true });
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
        // The transfer sees the same free disk as the snapshot.
        download: (options) => downloadArtifact({ ...options, ...s.common, statfs }),
        verify: verifyArtifact,
        fileExists: () => true,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner: ({ log }) => {
            let exit;
            const exited = new Promise((resolve) => { exit = resolve; });
            const handle = { pid: 4242, exited, running: true, async stop() { handle.running = false; exit({ code: 0, signal: 'SIGTERM', error: null }); return exited; } };
            log.append('stdout', 'load_tensors: offloaded 25/25 layers to GPU');
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        pollMs: 2,
        stopGraceMs: 50,
    });
    t.after(async () => { await controller.stop().catch(() => {}); });
    return controller;
}

async function until(predicate, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

test('F2: through the controller, a sparse decoy on a tight disk neither blocks admission nor stops the resume', async (t) => {
    const s = setup(t);
    const inode = seedPartial(s.paths, ARTIFACT, PARTIAL);
    sparseDecoy(s.shared, 'anything.gguf', SIZE);
    const remaining = SIZE - PARTIAL;
    const freeBytes = Math.ceil(remaining * 1.05) + 16;
    const controller = controllerHarness(t, s, { freeBytes, statfs: diskModel(s.root, freeBytes) });
    const preview = (await controller.overview({ preview: { modelId: 'tiny', runnerId: 'llama.cpp' } })).preview;
    assert.deepEqual(preview.acquisition.files.map((entry) => [entry.method, entry.bytesNeeded]), [['copy-or-resume', remaining]]);
    assert.notEqual(preview.admission.status, 'insufficient-now', preview.admission.reason);
    const accepted = await controller.run({ requestId: 'request-f2-resume-01', modelId: 'tiny', runnerId: 'llama.cpp' });
    assert.equal(accepted.accepted, true);
    assert.deepEqual(accepted.deployment.acquisition, [{ file: ARTIFACT.file, method: 'copy-or-resume', bytesNeeded: remaining }]);
    await until(() => controller.state.deployment?.phase === 'ready');
    assert.deepEqual(s.requests.map((request) => request.range), [`bytes=${PARTIAL}-`]);
    assert.equal(fs.statSync(s.paths.file).ino, inode);
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
    const status = await controller.status();
    assert.deepEqual(status.deployment.provenance.map((entry) => entry.method), ['download']);
});

test('F2 control: the same tight disk with no partial is still refused for the whole copy', async (t) => {
    const s = setup(t);
    sparseDecoy(s.shared, 'anything.gguf', SIZE);
    const freeBytes = Math.ceil((SIZE - PARTIAL) * 1.05) + 16;
    const controller = controllerHarness(t, s, { freeBytes, statfs: diskModel(s.root, freeBytes) });
    const preview = (await controller.overview({ preview: { modelId: 'tiny', runnerId: 'llama.cpp' } })).preview;
    assert.deepEqual(preview.acquisition.files.map((entry) => [entry.method, entry.bytesNeeded]), [['copy', SIZE]]);
    assert.equal(preview.admission.status, 'insufficient-now');
    await assert.rejects(controller.run({ requestId: 'request-f2-nopart-1', modelId: 'tiny', runnerId: 'llama.cpp' }),
        (err) => err.code === 'admission_insufficient_now');
    assert.equal(s.requests.length, 0);
});

// ------------------------------------ F2: free space that changes between readings

const COPY_NEED = Math.ceil(SIZE * 1.05);
const RESUME_NEED = Math.ceil((SIZE - PARTIAL) * 1.05);

// A statfs that returns `values` in turn, then the last one; `reads` records each reading.
function readings(values) {
    const reads = [];
    const statfs = async () => {
        const free = values[Math.min(reads.length, values.length - 1)];
        reads.push(free);
        return { bavail: free, bsize: 1 };
    };
    return { reads, statfs };
}

// Counts reads per /shared file (by the kernel's path of each opened candidate).
function readsPerSharedFile(s) {
    const reads = new Map();
    const fsApi = { ...fs, promises: { ...fs.promises, open: async (file, flags, mode) => {
        const handle = await fs.promises.open(file, flags, mode);
        if (flags & O_DIRECTORY) return handle;
        const real = await fs.promises.readlink(`/proc/self/fd/${handle.fd}`);
        if (!real.startsWith(`${fs.realpathSync(s.shared)}${path.sep}`)) return handle;
        reads.set(real, reads.get(real) ?? 0);
        return { fd: handle.fd, stat: () => handle.stat(), close: () => handle.close(),
            read: (...args) => { reads.set(real, reads.get(real) + 1); return handle.read(...args); } };
    } } };
    return { reads, fsApi };
}

test('F2 changing space: a copy that fits by its one reading is made, however later readings change', async (t) => {
    // The first reading fits the copy; every later one fits only the resume.
    const s = setup(t);
    seedPartial(s.paths, ARTIFACT, PARTIAL);
    const source = s.place(ARTIFACT.file);
    const { reads, statfs } = readings([COPY_NEED, RESUME_NEED]);
    let readsAtCopy = null;
    const result = await s.get({ statfs, progressIntervalMs: 0, onProgress: (p) => {
        if (p.phase === 'copying' && readsAtCopy === null) readsAtCopy = reads.length;
    } });
    assert.equal(readsAtCopy, 1, 'one reading decided the copy');
    assert.deepEqual(result.provenance, { file: ARTIFACT.file, source, method: 'copy', bytes: SIZE });
    assert.equal(s.requests.length, 0);
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
    assert.equal(fs.existsSync(s.paths.partial), false, 'the partial went only once the copy was published');
});

test('F2 changing space: space that shrinks between candidates resumes at the next one, without reading it', async (t) => {
    const s = setup(t);
    const inode = seedPartial(s.paths, ARTIFACT, PARTIAL);
    // The pinned name goes first: a same-size decoy with other bytes; then the valid file under another name.
    const wrong = Buffer.from(PAYLOAD);
    wrong[5] ^= 0xff;
    const decoy = s.place(ARTIFACT.file, wrong);
    const valid = s.place('other/valid.bin');
    const { reads: statReads, statfs } = readings([COPY_NEED, RESUME_NEED]);
    const { reads, fsApi } = readsPerSharedFile(s);
    const result = await s.get({ statfs, fsApi });
    assert.ok(reads.get(fs.realpathSync(decoy)) > 0, 'the decoy was copied and hashed');
    assert.equal(reads.get(fs.realpathSync(valid)) ?? 0, 0, 'the second candidate was not read');
    assert.ok(result.notes.some((note) => note.includes('do not match')), result.notes.join('; '));
    assert.ok(result.notes.some((note) => note.includes(`${valid} does not fit`)), result.notes.join('; '));
    assert.equal(result.provenance.method, 'download');
    assert.deepEqual(s.requests.map((request) => request.range), [`bytes=${PARTIAL}-`]);
    assert.equal(fs.statSync(s.paths.file).ino, inode, 'the partial was completed in place');
    assert.equal(fileSha(s.paths.file), ARTIFACT.sha256);
    assert.deepEqual(statReads.slice(0, 2), [COPY_NEED, RESUME_NEED], 'one reading per candidate');
    assert.deepEqual(stagingLeft(s.paths), []);
});

test('F2 changing space: with no partial, a copy that fits by its reading is made; one that does not is refused with that reading', async (t) => {
    const fits = setup(t);
    fits.place(ARTIFACT.file);
    const first = readings([COPY_NEED, 10]);
    const copied = await fits.get({ statfs: first.statfs });
    assert.equal(copied.provenance.method, 'copy');
    assert.equal(first.reads.length, 1);

    const low = setup(t);
    low.place(ARTIFACT.file);
    const second = readings([COPY_NEED - 1, COPY_NEED]);
    const { reads, fsApi } = readsPerSharedFile(low);
    await assert.rejects(low.get({ statfs: second.statfs, fsApi }), (err) => err.code === 'INSUFFICIENT_SPACE'
        && err.message.includes('private copy') && err.details.required === COPY_NEED && err.details.available === COPY_NEED - 1);
    assert.equal(second.reads.length, 1, 'the refusal gives the figures of the one reading that decided');
    assert.equal([...reads.values()].reduce((sum, count) => sum + count, 0), 0, 'no candidate byte was read');
    assert.deepEqual(stagingLeft(low.paths), []);
    assert.equal(low.requests.length, 0);
});

test('F2 changing space: a full disk while copying beside a partial still pauses, keeping the partial', async (t) => {
    const s = setup(t);
    seedPartial(s.paths, ARTIFACT, PARTIAL);
    s.place(ARTIFACT.file);
    let hit = 0;
    const fsApi = { ...fs, promises: { ...fs.promises, open: async (file, flags, mode) => {
        const handle = await fs.promises.open(file, flags, mode);
        if (!String(file).includes('.copy-')) return handle;
        return { fd: handle.fd, stat: () => handle.stat(), close: () => handle.close(), sync: () => handle.sync(),
            read: (...args) => handle.read(...args),
            write: async () => { hit += 1; throw Object.assign(new Error('no space'), { code: 'ENOSPC' }); } };
    } } };
    await assert.rejects(s.get({ fsApi, statfs: async () => ({ bavail: COPY_NEED, bsize: 1 }) }),
        (err) => err.code === 'PAUSED_ENOSPC' && err.retryable === true);
    assert.equal(hit, 1, 'the staging write hook fired');
    assert.deepEqual(stagingLeft(s.paths), []);
    assert.equal(fs.statSync(s.paths.partial).size, PARTIAL);
    assert.ok(fs.existsSync(s.paths.identity));
    assert.equal(s.requests.length, 0);
});
