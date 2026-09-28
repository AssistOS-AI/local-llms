// C12 regression tests 15-21, from the recheck of the workspace model reuse
// (findings F1-F5): anchored roots, non-blocking opens, in-place snapshot
// adoption, downloads bound to the bytes on disk, equal-size mixed shards,
// the controller's last check and cancellable planning. Small disposable
// fixtures only; no GPU, no network. Run under the exclusive measurement
// lock on the Spark (flock -x /home/research/phase0/.memlock).
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import {
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
const PAYLOAD = crypto.randomBytes(SIZE);
const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const COMMIT = 'c'.repeat(40);
const ARTIFACT = Object.freeze({ repo: 'org/model-GGUF', file: 'model-Q8_0.gguf', revision: 'main', commit: COMMIT, size: SIZE, sha256: sha256(PAYLOAD) });
const GIB = 1024 ** 3;

function setup(t) {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-c12r-'));
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
    const common = { fetchImpl, baseUrl: 'http://hf.invalid', sleep: async () => {}, chunkBytes: 64 * 1024 };
    const get = (options = {}) => downloadArtifact({ artifact: ARTIFACT, root, adoptFrom: [shared], ...common, ...options });
    return { base, dataDir, root, shared, requests, bodies, fetchImpl, common, get, paths: artifactPaths({ root, artifact: ARTIFACT }) };
}

function flipByte(file, at) {
    const fd = fs.openSync(file, 'r+');
    const one = Buffer.alloc(1);
    fs.readSync(fd, one, 0, 1, at);
    one[0] ^= 0xff;
    fs.writeSync(fd, one, 0, 1, at);
    fs.closeSync(fd);
}

test('15. the /shared/models root swapped for a link during lookup: nothing is adopted from outside the anchored root', async (t) => {
    const s = setup(t);
    const outside = path.join(s.base, 'outside');
    fs.mkdirSync(outside);
    fs.writeFileSync(path.join(outside, ARTIFACT.file), PAYLOAD);
    let swapped = false;
    const fsApi = { ...fs, promises: { ...fs.promises, opendir: async (dir, options) => {
        if (dir === s.shared && !swapped) {
            swapped = true;
            fs.renameSync(s.shared, `${s.shared}-old`);
            fs.symlinkSync(outside, s.shared);
        }
        return fs.promises.opendir(dir, options);
    } } };
    const result = await s.get({ fsApi });
    assert.equal(swapped, true);
    assert.equal(result.provenance.method, 'download');
    assert.equal(s.requests.length, 1);
    assert.ok(result.notes.some((note) => note.includes('root directory changed')), result.notes.join('; '));

    // A root that is already a link when the lookup starts is not a root at all.
    const l = setup(t);
    fs.rmSync(l.shared, { recursive: true });
    fs.symlinkSync(outside, l.shared);
    assert.equal((await l.get()).provenance.method, 'download');
});

test('16. a candidate replaced by a FIFO just before it is opened: no block, rejected, and an abort settles promptly', async (t) => {
    const withFifo = (s, onSwap = () => {}) => {
        const source = path.join(s.shared, ARTIFACT.file);
        fs.writeFileSync(source, PAYLOAD);
        // Release any reader a regression would leave blocked, so the test process can exit.
        t.after(() => {
            try {
                fs.closeSync(fs.openSync(source, fs.constants.O_WRONLY | fs.constants.O_NONBLOCK));
            } catch {}
        });
        let swapped = false;
        return { ...fs, promises: { ...fs.promises, open: async (file, flags, mode) => {
            if (file === source && !swapped) {
                swapped = true;
                fs.unlinkSync(source);
                execFileSync('mkfifo', [source]);
                onSwap();
            }
            return fs.promises.open(file, flags, mode);
        } } };
    };
    const s = setup(t);
    const result = await s.get({ fsApi: withFifo(s) });
    assert.equal(result.provenance.method, 'download');
    assert.ok(result.notes.some((note) => note.includes('not a regular file')), result.notes.join('; '));

    const a = setup(t);
    const stop = new AbortController();
    const began = Date.now();
    await assert.rejects(a.get({ fsApi: withFifo(a, () => stop.abort()), signal: stop.signal }), (err) => err.code === 'ABORTED');
    assert.ok(Date.now() - began < 1000);
    assert.equal(a.requests.length, 0);
});

test('17. a canonical, unrecorded HF snapshot with no .state is adopted in place; a mixed snapshot completes too', async (t) => {
    const s = setup(t);
    const config = Buffer.from('{"hidden_size": 8}\n');
    const second = Buffer.from(PAYLOAD.subarray(0, SIZE - 3));
    const third = Buffer.from(PAYLOAD.subarray(0, SIZE - 5));
    const files = [
        { path: 'config.json', size: config.length, sha256: sha256(config) },
        { path: 'model.safetensors', size: PAYLOAD.length, sha256: sha256(PAYLOAD) },
    ];
    const source = { type: 'hf-snapshot', repo: 'org/model', commit: COMMIT, files, size: config.length + PAYLOAD.length };
    const store = () => createWeightStores({
        dataDir: s.dataDir, env: {}, hfBaseUrl: 'http://hf.invalid', sharedModelsRoot: s.shared,
        downloadSnapshot: (options) => downloadSnapshotFile({ ...options, fetchImpl: s.fetchImpl, sleep: async () => {} }),
        state: () => ({}), save: () => {}, activeArtifact: () => null,
    })['hf-snapshot'];
    const hfRoot = path.join(s.dataDir, 'models', 'hf');
    const { dir, stateDir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'config.json'), config);
    fs.writeFileSync(path.join(dir, 'model.safetensors'), PAYLOAD);
    assert.equal(fs.existsSync(stateDir), false);
    const fetched = await store().fetch({ artifact: source });
    assert.deepEqual(fetched.provenance.map((entry) => entry.method), ['in-place', 'in-place']);
    assert.equal(s.requests.length, 0);
    assert.equal((await store().state(source)).state, 'complete');

    // One in place, one copied from /shared, one downloaded.
    const mixed = { ...source, commit: 'd'.repeat(40), files: [
        { path: 'config.json', size: config.length, sha256: sha256(config) },
        { path: 'model-a.safetensors', size: second.length, sha256: sha256(second) },
        { path: 'model-b.safetensors', size: third.length, sha256: sha256(third) },
    ] };
    mixed.size = config.length + second.length + third.length;
    const other = snapshotPaths({ root: hfRoot, repo: mixed.repo, commit: mixed.commit });
    fs.mkdirSync(other.dir, { recursive: true });
    fs.writeFileSync(path.join(other.dir, 'config.json'), config);
    fs.writeFileSync(path.join(s.shared, 'weights-a.bin'), second);
    s.bodies.set('model-b.safetensors', third);
    const result = await store().fetch({ artifact: mixed });
    assert.deepEqual(result.provenance.map((entry) => entry.method), ['in-place', 'copy', 'download']);
    assert.equal((await store().state(mixed)).state, 'complete');
});

test('18. a byte changed in the partial before publication is never recorded as verified, also in a resumed prefix', async (t) => {
    // After the last received chunk, before publication.
    const s = setup(t);
    let changed = false;
    await assert.rejects(s.get({ adoptFrom: [], progressIntervalMs: 0, onProgress: (p) => {
        if (p.phase === 'downloading' && p.bytes === SIZE && !changed) {
            changed = true;
            flipByte(s.paths.partial, 0);
        }
    } }), (err) => err.code === 'SHA256_MISMATCH');
    assert.equal(fs.existsSync(s.paths.meta), false);
    assert.equal(fs.existsSync(s.paths.file), false);
    assert.equal((await s.get({ adoptFrom: [] })).provenance.method, 'download');
    assert.equal(sha256(fs.readFileSync(s.paths.file)), ARTIFACT.sha256);

    // During the final verification itself.
    const v = setup(t);
    changed = false;
    await assert.rejects(v.get({ adoptFrom: [], progressIntervalMs: 0, onProgress: (p) => {
        if (p.phase === 'verifying' && !changed) {
            changed = true;
            flipByte(v.paths.partial, SIZE - 1);
        }
    } }), (err) => ['CHANGED_WHILE_VERIFYING', 'SHA256_MISMATCH'].includes(err.code));
    assert.equal(fs.existsSync(v.paths.meta), false);

    // A resumed prefix changed between attempts, and one changed during the transfer after it was read.
    for (const when of ['between', 'during']) {
        const r = setup(t);
        fs.mkdirSync(path.dirname(r.paths.partial), { recursive: true });
        fs.writeFileSync(r.paths.partial, PAYLOAD.subarray(0, 131_072));
        fs.writeFileSync(r.paths.identity, JSON.stringify({ repo: ARTIFACT.repo, file: ARTIFACT.file, commit: COMMIT, size: SIZE, sha256: ARTIFACT.sha256 }));
        if (when === 'between') flipByte(r.paths.partial, 100);
        let flipped = false;
        await assert.rejects(r.get({ adoptFrom: [], progressIntervalMs: 0, onProgress: (p) => {
            if (when === 'during' && p.phase === 'downloading' && !flipped) {
                flipped = true;
                flipByte(r.paths.partial, 100);
            }
        } }), (err) => err.code === 'SHA256_MISMATCH', when);
        assert.equal(fs.existsSync(r.paths.meta), false, when);
        assert.deepEqual(r.requests.map((request) => request.range), ['bytes=131072-'], when);
    }
});

test('19. equal-size split shards from mixed sources, with a same-name wrong decoy and a bad download, complete only when every digest matches', async (t) => {
    const s = setup(t);
    const blobs = [Buffer.alloc(65536, 65), Buffer.alloc(65536, 66), Buffer.alloc(65536, 67)];
    const names = ['m-00001-of-00003.gguf', 'm-00002-of-00003.gguf', 'm-00003-of-00003.gguf'];
    const source = validateModel({ id: 'split', architecture: 'moe', sources: { gguf: {
        type: 'huggingface', repo: 'org/split-GGUF', file: names[0], revision: 'main', commit: COMMIT,
        shards: names.map((file, index) => ({ file, size: 65536, sha256: sha256(blobs[index]) })),
    } } }, { seed: true }).sources.gguf;
    const first = artifactPaths({ root: s.root, artifact: { ...source, shards: undefined, file: names[0], size: 65536, sha256: sha256(blobs[0]) } });
    fs.mkdirSync(path.dirname(first.file), { recursive: true });
    fs.writeFileSync(first.file, blobs[0]);
    fs.writeFileSync(path.join(s.shared, names[1]), blobs[0]);
    fs.writeFileSync(path.join(s.shared, 'valid-second.bin'), blobs[1]);
    let badThird = true;
    const store = createWeightStores({
        dataDir: s.dataDir, env: {}, hfBaseUrl: 'http://hf.invalid', sharedModelsRoot: s.shared, inspect: inspectArtifact, remove: removeArtifact,
        download: (options) => downloadArtifact({ ...options, chunkBytes: 4096, sleep: async () => {}, fetchImpl: async () => {
            const bytes = badThird ? blobs[0] : blobs[2];
            return new Response(bytes, { headers: { 'content-length': String(bytes.length) } });
        } }),
        state: () => ({}), save: () => {}, activeArtifact: () => null,
    }).huggingface;
    await assert.rejects(store.fetch({ artifact: source }), (err) => err.code === 'SHA256_MISMATCH');
    assert.equal((await store.state(source)).state, 'partial');
    badThird = false;
    const fetched = await store.fetch({ artifact: source });
    assert.equal((await store.state(source)).state, 'complete');
    assert.deepEqual(fetched.provenance.map((entry) => entry.method), ['in-place', 'copy', 'download']);
    for (let index = 0; index < 3; index += 1) {
        assert.equal(sha256(fs.readFileSync(path.join(path.dirname(first.file), names[index]))), sha256(blobs[index]));
    }
});

function controllerFor(t, s, extra = {}) {
    const GPT = loadSeedCatalog().find((model) => model.id === 'gpt-oss-20b');
    const tiny = validateModel({ ...GPT, id: 'tiny', displayName: 'Tiny', sources: { gguf: { type: 'huggingface', ...ARTIFACT } } }, { seed: true });
    const launches = [];
    const controller = createController({
        dataDir: s.dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: [tiny],
        sharedModelsRoot: s.shared,
        snapshot: async () => ({
            gpu: { available: true, name: 'Test GPU', totalBytes: 24 * GIB, usedBytes: 0, freeBytes: 24 * GIB, processes: [] },
            memory: { totalBytes: 64 * GIB, availableBytes: 60 * GIB },
            disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
            cpus: 8,
        }),
        download: (options) => downloadArtifact({ ...options, ...s.common }),
        verify: verifyArtifact,
        fileExists: () => true,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner: ({ log }) => {
            let exit;
            const exited = new Promise((resolve) => { exit = resolve; });
            const handle = { pid: 4242, exited, running: true, async stop() { handle.running = false; exit({ code: 0, signal: 'SIGTERM', error: null }); return exited; } };
            log.append('stdout', 'load_tensors: offloaded 25/25 layers to GPU');
            launches.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        pollMs: 2,
        stopGraceMs: 50,
        ...extra,
    });
    t.after(async () => { await controller.stop().catch(() => {}); });
    return { controller, launches, run: { modelId: 'tiny', runnerId: 'llama.cpp' } };
}

async function until(predicate, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

test('20. in the real controller, an artifact changed after the fetch and before launch fails the start with no launch', async (t) => {
    const s = setup(t);
    const h = controllerFor(t, s, { dropCache: (file) => { flipByte(file, 7); return true; } });
    await h.controller.run({ requestId: 'request-lastcheck-01', ...h.run });
    await until(() => h.controller.state.deployment?.phase === 'error');
    assert.match(h.controller.state.deployment.error, /no longer the verified file/);
    assert.equal(h.launches.length, 0);
    const phases = (await h.controller.status()).logs.map((line) => line.line).filter((line) => line.startsWith('phase '));
    assert.ok(phases.includes('phase verifying'), 'the last check shows as verifying: ' + phases.join(', '));
});

test('21. a Stop during acquisition planning stops the lookup after the pending read, and settles promptly', async (t) => {
    const s = setup(t);
    for (let index = 0; index < 20; index += 1) fs.mkdirSync(path.join(s.shared, `d${index}`));
    const original = fs.promises.opendir;
    let opened = 0;
    let release;
    let entered;
    const gate = new Promise((resolve) => { release = resolve; });
    const started = new Promise((resolve) => { entered = resolve; });
    let held = false;
    fs.promises.opendir = async (dir, options) => {
        opened += 1;
        if (dir === s.shared && !held) {
            held = true;
            entered();
            await gate;
        }
        return original(dir, options);
    };
    t.after(() => { fs.promises.opendir = original; });
    const h = controllerFor(t, s);
    const running = h.controller.run({ requestId: 'request-planstop-01', ...h.run }).catch((error) => error);
    await started;
    const began = Date.now();
    const stopping = h.controller.stop();
    release();
    const outcome = await running;
    await stopping;
    assert.ok(Date.now() - began < 1000);
    assert.equal(outcome.code, 'cancelled');
    assert.equal(opened, 1, 'no directory opened after the Stop');
    assert.equal(h.launches.length, 0);
});

// Holds the first open of /shared/models until released, so a Stop, Cancel or drain lands during planning.
function holdSharedOpen(t, s) {
    const original = fs.promises.opendir;
    const hold = { opened: 0 };
    const gate = new Promise((resolve) => { hold.release = resolve; });
    hold.started = new Promise((resolve) => { hold.entered = resolve; });
    let held = false;
    fs.promises.opendir = async (dir, options) => {
        hold.opened += 1;
        if (dir === s.shared && !held) {
            held = true;
            hold.entered();
            await gate;
        }
        return original(dir, options);
    };
    t.after(() => { fs.promises.opendir = original; });
    return hold;
}

const lowDisk = {
    snapshot: async () => ({
        gpu: { available: true, name: 'Test GPU', totalBytes: 24 * GIB, usedBytes: 0, freeBytes: 24 * GIB, processes: [] },
        memory: { totalBytes: 64 * GIB, availableBytes: 60 * GIB },
        disk: { freeBytes: 1024, totalBytes: 500 * GIB },
        cpus: 8,
    }),
};

test('21b. a Stop or drain during an EMPTY lookup still fails the Run with cancelled: no job, no admission verdict', async (t) => {
    for (const [how, extra] of [['stop', {}], ['stop', lowDisk], ['drain', {}]]) {
        await t.test(`${how}${extra === lowDisk ? ', disk below the model size' : ''}`, async (tt) => {
            const s = setup(tt);
            const hold = holdSharedOpen(tt, s);
            const h = controllerFor(tt, s, extra);
            const running = h.controller.run({ requestId: `request-emptystop-${how}`, ...h.run }).catch((error) => error);
            await hold.started;
            const settling = how === 'drain' ? h.controller.drain() : h.controller.stop();
            hold.release();
            const outcome = await running;
            await settling;
            assert.equal(outcome.code, 'cancelled', `the Run reports cancelled, not ${outcome.code ?? 'accepted'}`);
            assert.equal(h.controller.state.deployment ?? null, null, 'no deployment was created');
            assert.equal(s.requests.length, 0);
            assert.equal(h.launches.length, 0);
            assert.equal(hold.opened, 1);
        });
    }
});

test('21c. a Cancel during acquisition planning succeeds, empty or not; an idle Cancel is still not_downloading', async (t) => {
    for (const [kind, subdirs, extra] of [['empty', 0, lowDisk], ['empty', 0, {}], ['nonempty', 20, {}]]) {
        await t.test(`${kind}${extra === lowDisk ? ', disk below the model size' : ''}`, async (tt) => {
            const s = setup(tt);
            for (let index = 0; index < subdirs; index += 1) fs.mkdirSync(path.join(s.shared, `d${index}`));
            const hold = holdSharedOpen(tt, s);
            const h = controllerFor(tt, s, extra);
            const running = h.controller.run({ requestId: `request-plancancel-${kind}-${subdirs}`, ...h.run }).catch((error) => error);
            await hold.started;
            const cancelling = h.controller.cancelDownload();
            hold.release();
            const outcome = await running;
            const cancelled = await cancelling;
            assert.equal(outcome.code, 'cancelled');
            assert.equal(cancelled.deployment, null, 'the Cancel succeeded and no deployment exists');
            assert.equal(hold.opened, 1, 'no directory opened after the Cancel');
            assert.equal(s.requests.length, 0);
            assert.equal(h.launches.length, 0);
            await assert.rejects(h.controller.cancelDownload(), (error) => error.code === 'not_downloading');
        });
    }
});
