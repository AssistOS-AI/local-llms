import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import util from 'node:util';

import {
    DownloadError,
    artifactPaths,
    downloadArtifact,
    inspectArtifact,
    removeArtifact,
    resolveHuggingFaceArtifact,
} from '../src/controller/downloader.mjs';

const REPO = 'org/model-GGUF';
const FILE = 'model-Q4_K_M.gguf';
const COMMIT = 'a'.repeat(8) + '0123456789abcdef0123456789abcdef';
const SIZE = 3 * 1024 * 1024 + 12_345;
const MIB = 1024 * 1024;
const TIMEOUT = 20_000;
const noSleep = async () => {};

// Deterministic payload from a seeded mulberry32 PRNG.
function makePayload(size, seed) {
    const buffer = Buffer.alloc(size);
    let state = seed >>> 0;
    for (let i = 0; i < size; i += 1) {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        buffer[i] = ((t ^ (t >>> 14)) >>> 0) & 0xff;
    }
    return buffer;
}

const PAYLOAD = makePayload(SIZE, 1234);
const SHA256 = crypto.createHash('sha256').update(PAYLOAD).digest('hex');
const ARTIFACT = Object.freeze({
    source: 'huggingface',
    repo: REPO,
    file: FILE,
    revision: 'main',
    commit: COMMIT,
    size: SIZE,
    sha256: SHA256,
});

function sha256File(filePath) {
    return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function tempRoot(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-dl-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    return root;
}

function exists(filePath) {
    return fs.existsSync(filePath);
}

function sendBody(res, body) {
    res.end(body);
}

function serveFile(state, req, res) {
    const body = state.options.wrongBytes ? corrupt(PAYLOAD) : PAYLOAD;
    const range = /^bytes=(\d+)-$/.exec(req.headers.range ?? '');
    const fileHits = state.hits('file');
    if (state.options.truncateFirst && fileHits === 1) {
        res.writeHead(200, { 'Content-Length': SIZE });
        res.write(body.subarray(0, MIB), () => res.destroy());
        return;
    }
    if (range && state.options.badContentRange && !state.badRangeSent) {
        state.badRangeSent = true;
        const start = Number(range[1]);
        res.writeHead(206, {
            'Content-Range': `bytes 0-${SIZE - 1}/${SIZE}`,
            'Content-Length': SIZE - start,
        });
        sendBody(res, body.subarray(start));
        return;
    }
    if (range && !state.options.ignoreRange) {
        const start = Number(range[1]);
        if (start >= SIZE) {
            res.writeHead(416, { 'Content-Range': `bytes */${SIZE}` });
            res.end();
            return;
        }
        res.writeHead(206, {
            'Content-Range': `bytes ${start}-${SIZE - 1}/${SIZE}`,
            'Content-Length': SIZE - start,
        });
        sendBody(res, body.subarray(start));
        return;
    }
    res.writeHead(200, { 'Content-Length': SIZE });
    sendBody(res, body);
}

function corrupt(buffer) {
    const copy = Buffer.from(buffer);
    copy[copy.length >> 1] ^= 0xff;
    return copy;
}

function sendJson(res, status, value) {
    const body = JSON.stringify(value);
    res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) });
    res.end(body);
}

function treeEntries() {
    return [
        { type: 'file', path: 'README.md', size: 120, oid: 'b'.repeat(40) },
        {
            type: 'file',
            path: FILE,
            size: SIZE,
            oid: 'c'.repeat(40),
            lfs: { oid: SHA256, size: SIZE, pointerSize: 135 },
        },
    ];
}

function route(state, req, res) {
    const url = new URL(req.url, 'http://127.0.0.1');
    const resolvePath = `/${REPO}/resolve/${COMMIT}/${FILE}`;
    if (url.pathname === `/api/models/${REPO}/revision/main`) {
        if (state.options.failWhenAuthorized && req.headers.authorization) {
            sendJson(res, 500, { error: 'boom' });
            return;
        }
        sendJson(res, 200, { id: REPO, sha: COMMIT });
        return;
    }
    if (url.pathname === `/api/models/${REPO}/tree/${COMMIT}`) {
        sendJson(res, 200, treeEntries());
        return;
    }
    if (url.pathname === `/api/models/${REPO}/tree/${COMMIT}/sub`) {
        sendJson(res, 200, [{
            type: 'file',
            path: `sub/${FILE}`,
            size: SIZE,
            oid: 'd'.repeat(40),
            lfs: { oid: SHA256, size: SIZE, pointerSize: 135 },
        }]);
        return;
    }
    if (url.pathname === resolvePath || url.pathname === `/${REPO}/resolve/${COMMIT}/sub/${FILE}`) {
        if (state.options.redirect) {
            res.writeHead(302, { Location: `/cdn/${FILE}?sig=${state.hits('resolve')}` });
            res.end();
            return;
        }
        serveFile(state, req, res);
        return;
    }
    if (url.pathname === `/cdn/${FILE}`) {
        if (state.hits('cdn') === 1) {
            res.writeHead(403, { 'Content-Length': 0 });
            res.end();
            return;
        }
        serveFile(state, req, res);
        return;
    }
    res.writeHead(404, { 'Content-Length': 0 });
    res.end();
}

function classify(pathname) {
    if (pathname.includes('/resolve/')) {
        return 'resolve';
    }
    if (pathname.startsWith('/cdn/')) {
        return 'cdn';
    }
    return 'api';
}

async function startServer(t, options = {}) {
    const state = {
        options,
        requests: [],
        badRangeSent: false,
        hits(kind) {
            if (kind === 'file') {
                return this.requests.filter((r) => r.kind === (options.redirect ? 'cdn' : 'resolve')).length;
            }
            return this.requests.filter((r) => r.kind === kind).length;
        },
    };
    const server = http.createServer((req, res) => {
        const pathname = new URL(req.url, 'http://127.0.0.1').pathname;
        state.requests.push({ kind: classify(pathname), url: req.url, headers: { ...req.headers } });
        res.on('error', () => {});
        route(state, req, res);
    });
    server.on('clientError', (err, socket) => socket.destroy());
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => {
        server.closeAllConnections();
        return new Promise((resolve) => server.close(resolve));
    });
    state.baseUrl = `http://127.0.0.1:${server.address().port}`;
    return state;
}

function seedPartial(root, bytes, identity = ARTIFACT) {
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    fs.mkdirSync(paths.dir, { recursive: true });
    fs.writeFileSync(paths.partial, PAYLOAD.subarray(0, bytes));
    fs.writeFileSync(paths.identity, JSON.stringify({
        repo: identity.repo,
        file: identity.file,
        commit: identity.commit,
        size: identity.size,
        sha256: identity.sha256,
    }));
    return paths;
}

function download(server, root, extra = {}) {
    return downloadArtifact({
        artifact: ARTIFACT,
        root,
        baseUrl: server.baseUrl,
        progressIntervalMs: 0,
        sleep: noSleep,
        ...extra,
    });
}

function assertComplete(root) {
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    assert.equal(sha256File(paths.file), SHA256);
    assert.equal(exists(paths.partial), false);
    assert.equal(exists(paths.identity), false);
    const meta = JSON.parse(fs.readFileSync(paths.meta, 'utf8'));
    assert.equal(meta.sha256, SHA256);
    assert.equal(meta.commit, COMMIT);
    assert.equal(meta.size, SIZE);
    return paths;
}

async function abortAt(server, root, fraction) {
    const controller = new AbortController();
    const onProgress = ({ bytes }) => {
        if (bytes >= SIZE * fraction) {
            controller.abort();
        }
    };
    await assert.rejects(
        download(server, root, { signal: controller.signal, onProgress }),
        (err) => err instanceof DownloadError && err.code === 'ABORTED' && err.retryable === true,
    );
}

function rangeStarts(server) {
    return server.requests
        .filter((r) => r.kind !== 'api' && r.headers.range)
        .map((r) => Number(/^bytes=(\d+)-$/.exec(r.headers.range)[1]));
}

test('resolve pins main to the commit and reads size and sha256 from the tree', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const artifact = await resolveHuggingFaceArtifact({ repo: REPO, file: FILE, baseUrl: server.baseUrl });
    assert.deepEqual(artifact, { ...ARTIFACT });
    assert.ok(Object.isFrozen(artifact));
    assert.deepEqual(server.requests.map((r) => r.url), [
        `/api/models/${REPO}/revision/main`,
        `/api/models/${REPO}/tree/${COMMIT}`,
    ]);
    assert.equal(server.requests[0].headers.authorization, undefined);
});

test('resolve skips the revision lookup for a commit and lists subdirectories', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const artifact = await resolveHuggingFaceArtifact({
        repo: REPO,
        file: `sub/${FILE}`,
        revision: COMMIT,
        baseUrl: server.baseUrl,
    });
    assert.equal(artifact.commit, COMMIT);
    assert.equal(artifact.size, SIZE);
    assert.deepEqual(server.requests.map((r) => r.url), [`/api/models/${REPO}/tree/${COMMIT}/sub`]);
});

test('resolve rejects an unknown file with NOT_FOUND', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    await assert.rejects(
        resolveHuggingFaceArtifact({ repo: REPO, file: 'missing.gguf', baseUrl: server.baseUrl }),
        (err) => err instanceof DownloadError && err.code === 'NOT_FOUND',
    );
});

test('resolve rejects invalid sources before any request', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const cases = [
        { repo: 'no-owner', file: FILE },
        { repo: 'a/b/c', file: FILE },
        { repo: '../etc', file: FILE },
        { repo: 'org/..', file: FILE },
        { repo: REPO, file: '../x.gguf' },
        { repo: REPO, file: 'a/../b.gguf' },
        { repo: REPO, file: 'model.bin' },
        { repo: REPO, file: '/abs.gguf' },
        { repo: REPO, file: 'a/b/c/d/e.gguf' },
        { repo: REPO, file: FILE, revision: '../main' },
        { repo: REPO, file: FILE, revision: 'a b' },
    ];
    for (const input of cases) {
        await assert.rejects(
            resolveHuggingFaceArtifact({ ...input, baseUrl: server.baseUrl }),
            (err) => err instanceof DownloadError && err.code === 'INVALID_SOURCE',
            JSON.stringify(input),
        );
    }
    assert.equal(server.requests.length, 0);
});

test('resolve sends the token as Bearer and never exposes it', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t, { failWhenAuthorized: true });
    const token = 'hf_SuperSecretToken123';
    const err = await resolveHuggingFaceArtifact({ repo: REPO, file: FILE, token, baseUrl: server.baseUrl })
        .then(() => null, (error) => error);
    assert.ok(err instanceof DownloadError);
    assert.equal(err.code, 'RESOLVE_FAILED');
    assert.equal(err.retryable, true);
    assert.equal(server.requests[0].headers.authorization, `Bearer ${token}`);
    for (const text of [err.message, err.stack, util.inspect(err, { depth: null }), JSON.stringify(err.details ?? {})]) {
        assert.equal(text.includes(token), false);
    }
});

test('an aborted download keeps its partial and resumes with Range', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    await abortAt(server, root, 0.4);
    assert.ok(exists(paths.partial));
    assert.ok(exists(paths.identity));
    const kept = fs.statSync(paths.partial).size;
    assert.ok(kept >= SIZE * 0.4 && kept < SIZE);
    assert.deepEqual(await inspectArtifact({ root, artifact: ARTIFACT }), { state: 'partial', bytes: kept });

    const result = await download(server, root);
    assert.equal(result.status, 'complete');
    assert.equal(result.path, paths.file);
    assert.ok(result.bytesTransferred < SIZE);
    assert.equal(result.bytesTransferred, SIZE - kept);
    assert.deepEqual(rangeStarts(server), [kept]);
    assertComplete(root);
});

test('a server that ignores Range restarts from zero', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t, { ignoreRange: true });
    const root = tempRoot(t);
    seedPartial(root, 1_500_000);
    const result = await download(server, root);
    assert.equal(result.bytesTransferred, SIZE);
    assert.deepEqual(rangeStarts(server), [1_500_000]);
    assertComplete(root);
});

test('an invalid Content-Range discards the partial and restarts', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t, { badContentRange: true });
    const root = tempRoot(t);
    seedPartial(root, 1_500_000);
    const result = await download(server, root);
    assert.equal(result.bytesTransferred, SIZE);
    const fileRequests = server.requests.filter((r) => r.kind === 'resolve');
    assert.equal(fileRequests.length, 2);
    assert.equal(fileRequests[0].headers.range, 'bytes=1500000-');
    assert.equal(fileRequests[1].headers.range, undefined);
    assertComplete(root);
});

test('a partial with a different upstream identity is discarded before any Range request', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    seedPartial(root, 1_500_000, { ...ARTIFACT, sha256: 'f'.repeat(64) });
    assert.deepEqual(await inspectArtifact({ root, artifact: ARTIFACT }), { state: 'absent', bytes: 0 });
    const result = await download(server, root);
    assert.equal(result.bytesTransferred, SIZE);
    assert.deepEqual(rangeStarts(server), []);
    assertComplete(root);
});

test('an expired signed redirect is re-resolved from the resolve URL', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t, { redirect: true });
    const root = tempRoot(t);
    const result = await download(server, root);
    assert.equal(result.bytesTransferred, SIZE);
    assert.equal(server.hits('resolve'), 2);
    assert.equal(server.hits('cdn'), 2);
    assert.deepEqual(
        server.requests.filter((r) => r.kind === 'cdn').map((r) => r.url),
        [`/cdn/${FILE}?sig=1`, `/cdn/${FILE}?sig=2`],
    );
    assertComplete(root);
});

test('wrong bytes of the right length fail with SHA256_MISMATCH and leave nothing', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t, { wrongBytes: true });
    const root = tempRoot(t);
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    await assert.rejects(
        download(server, root),
        (err) => err instanceof DownloadError && err.code === 'SHA256_MISMATCH' && err.retryable === false,
    );
    assert.equal(exists(paths.partial), false);
    assert.equal(exists(paths.identity), false);
    assert.equal(exists(paths.file), false);
    assert.equal(exists(paths.meta), false);
});

test('insufficient free space fails before any request or byte', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    const probed = [];
    const available = Math.floor(SIZE * 1.04);
    const statfs = async (p) => {
        probed.push(p);
        return { bavail: available, bsize: 1 };
    };
    const err = await download(server, root, { statfs }).then(() => null, (error) => error);
    assert.ok(err instanceof DownloadError);
    assert.equal(err.code, 'INSUFFICIENT_SPACE');
    assert.equal(err.retryable, false);
    assert.deepEqual(err.details, { required: Math.ceil(SIZE * 1.05), available });
    assert.equal(server.requests.length, 0);
    assert.equal(exists(paths.partial), false);
    assert.equal(exists(paths.identity), false);
    assert.deepEqual(probed, [path.resolve(root)]);
});

test('a resume needs space only for the remaining bytes', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    const have = 2 * MIB;
    seedPartial(root, have);
    const available = Math.ceil((SIZE - have) * 1.05) + 16;
    assert.ok(available < SIZE);
    const statfs = async () => ({ bavail: available, bsize: 1 });
    const result = await download(server, root, { statfs });
    assert.equal(result.bytesTransferred, SIZE - have);
    assert.deepEqual(rangeStarts(server), [have]);
    assertComplete(root);
});

test('ENOSPC mid-transfer pauses and keeps the partial', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    let written = 0;
    const open = async (...args) => {
        const handle = await fs.promises.open(...args);
        return {
            write: async (buffer, offset, length) => {
                if (written >= MIB) {
                    throw Object.assign(new Error('no space left on device'), { code: 'ENOSPC' });
                }
                const result = await handle.write(buffer, offset, length);
                written += result.bytesWritten;
                return result;
            },
            sync: () => handle.sync(),
            truncate: (len) => handle.truncate(len),
            close: () => handle.close(),
        };
    };
    const fsApi = { ...fs, promises: { ...fs.promises, open } };
    await assert.rejects(
        download(server, root, { fsApi }),
        (err) => err instanceof DownloadError && err.code === 'PAUSED_ENOSPC' && err.retryable === true,
    );
    assert.ok(exists(paths.identity));
    const kept = fs.statSync(paths.partial).size;
    assert.equal(kept, written);
    assert.ok(kept >= MIB);

    const result = await download(server, root);
    assert.equal(result.bytesTransferred, SIZE - kept);
    assertComplete(root);
});

test('a truncated connection resumes with Range in the same call', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t, { truncateFirst: true });
    const root = tempRoot(t);
    const result = await download(server, root);
    assert.equal(result.bytesTransferred, SIZE);
    const starts = rangeStarts(server);
    assert.equal(starts.length, 1);
    assert.ok(starts[0] > 0);
    assertComplete(root);
});

test('repeated server errors give up with NETWORK and keep the partial', { timeout: TIMEOUT }, async (t) => {
    const root = tempRoot(t);
    let calls = 0;
    const fetchImpl = async () => {
        calls += 1;
        return new Response('busy', { status: 503 });
    };
    const err = await downloadArtifact({
        artifact: ARTIFACT,
        root,
        fetchImpl,
        sleep: noSleep,
        maxAttempts: 3,
    }).then(() => null, (error) => error);
    assert.equal(err.code, 'NETWORK');
    assert.equal(err.retryable, true);
    assert.equal(calls, 3);
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    assert.ok(exists(paths.partial));
    assert.ok(exists(paths.identity));
});

test('inspect reports states and a complete download is a no-op', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    assert.deepEqual(await inspectArtifact({ root, artifact: ARTIFACT }), { state: 'absent', bytes: 0 });
    seedPartial(root, 1000);
    assert.deepEqual(await inspectArtifact({ root, artifact: ARTIFACT }), { state: 'partial', bytes: 1000 });

    const progress = [];
    await download(server, root, { onProgress: (p) => progress.push(p), progressIntervalMs: 60_000 });
    assert.deepEqual(await inspectArtifact({ root, artifact: ARTIFACT }), { state: 'complete', bytes: SIZE });
    assert.equal(progress.length, 2);
    const last = progress.at(-1);
    assert.equal(last.bytes, SIZE);
    assert.equal(last.total, SIZE);
    assert.equal(last.transferred, SIZE - 1000);

    const before = server.requests.length;
    const again = await download(server, root);
    assert.deepEqual(again, { status: 'complete', path: artifactPaths({ root, artifact: ARTIFACT }).file, bytesTransferred: 0 });
    assert.equal(server.requests.length, before);
});

test('artifactPaths keeps every path under the root', () => {
    const paths = artifactPaths({ root: '/data/weights', artifact: { ...ARTIFACT, file: `sub/${FILE}` } });
    assert.deepEqual(paths, {
        dir: `/data/weights/org/model-GGUF/${COMMIT}`,
        file: `/data/weights/org/model-GGUF/${COMMIT}/sub/${FILE}`,
        partial: `/data/weights/org/model-GGUF/${COMMIT}/sub/${FILE}.partial`,
        identity: `/data/weights/org/model-GGUF/${COMMIT}/sub/${FILE}.partial.json`,
        meta: `/data/weights/org/model-GGUF/${COMMIT}/sub/${FILE}.json`,
    });
    for (const bad of [{ repo: '../x' }, { repo: 'a/..' }, { file: '../../x.gguf' }, { commit: '../..' }]) {
        assert.throws(
            () => artifactPaths({ root: '/data/weights', artifact: { ...ARTIFACT, ...bad } }),
            (err) => err instanceof DownloadError && err.code === 'INVALID_SOURCE',
        );
    }
});

test('removeArtifact deletes every file and the empty directories', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    const paths = artifactPaths({ root, artifact: ARTIFACT });
    await download(server, root);
    fs.writeFileSync(paths.identity, '{}');
    const metaSize = fs.statSync(paths.meta).size;
    const freed = await removeArtifact({ root, artifact: ARTIFACT });
    assert.equal(freed, SIZE + metaSize + 2);
    for (const target of [paths.file, paths.partial, paths.identity, paths.meta, paths.dir]) {
        assert.equal(exists(target), false);
    }
    assert.ok(exists(root));
    assert.deepEqual(fs.readdirSync(root), []);
    assert.equal(await removeArtifact({ root, artifact: ARTIFACT }), 0);
});

test('a stop during the resume re-hash aborts before any request', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    const paths = seedPartial(root, 2 * MIB);
    const controller = new AbortController();
    let chunks = 0;
    const createReadStream = (...args) => (async function* () {
        for await (const chunk of fs.createReadStream(...args)) {
            chunks += 1;
            yield chunk;
            controller.abort();
        }
    })();
    const fsApi = { ...fs, createReadStream };
    await assert.rejects(
        download(server, root, { fsApi, signal: controller.signal }),
        (err) => err instanceof DownloadError && err.code === 'ABORTED',
    );
    assert.ok(chunks <= 2, `re-hash kept reading: ${chunks} chunks`);
    assert.equal(server.requests.length, 0);
    assert.equal(fs.statSync(paths.partial).size, 2 * MIB);
    assert.ok(exists(paths.identity));
});

test('files with the same name in different folders of one commit get different paths', () => {
    const base = { type: 'huggingface', repo: 'owner/repo', revision: 'main', commit: 'c'.repeat(40), size: 1, sha256: 'd'.repeat(64) };
    const root = '/data/models/gguf';
    const q4 = artifactPaths({ root, artifact: { ...base, file: 'Q4/m.gguf' } });
    const q8 = artifactPaths({ root, artifact: { ...base, file: 'Q8/m.gguf' } });
    assert.notEqual(q4.file, q8.file);
    assert.notEqual(q4.identity, q8.identity);
    assert.equal(q4.file, `/data/models/gguf/owner/repo/${'c'.repeat(40)}/Q4/m.gguf`);
    // A single-segment file keeps the layout used so far.
    assert.equal(artifactPaths({ root, artifact: { ...base, file: 'm.gguf' } }).file, `/data/models/gguf/owner/repo/${'c'.repeat(40)}/m.gguf`);
});

test('a file in a subfolder downloads into that folder and is removed with it', { timeout: TIMEOUT }, async (t) => {
    const server = await startServer(t);
    const root = tempRoot(t);
    const artifact = { ...ARTIFACT, file: `sub/${FILE}` };
    const result = await download(server, root, { artifact });
    assert.equal(result.status, 'complete');
    assert.equal(result.path, path.join(root, 'org', 'model-GGUF', COMMIT, 'sub', FILE));
    await removeArtifact({ root, artifact });
    assert.equal(fs.existsSync(path.join(root, 'org')), false);
});
