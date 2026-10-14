// ModelScope registries (DS002): a registry of kind `modelscope` is looked up, pinned and downloaded through its own
// API (commits, a recursive listing, /repo downloads) behind the controller's guarded transport. A pin records each
// file's size and sha256 at a 40-hex commit; the credential is a Bearer token for the registry's origin only, never
// the CDN a download is redirected to. The fake registry, the fake CDN, a fake huggingface.co and a bystander are
// loopback servers behind a transport that maps their https names onto them (net-guard.test.mjs holds the A5 rows).
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import zlib from 'node:zlib';

import { createCommandQueue } from '../src/controller/commandQueue.mjs';
import {
    DownloadError, artifactPaths, downloadArtifact, downloadSnapshotFile, inspectArtifact, removeArtifact, resolveHuggingFaceArtifact, snapshotPaths,
} from '../src/controller/downloader.mjs';
import {
    LISTING_LIMIT, listModelScopeFiles, lookupModelScope, modelScopeFileUrl, resolveModelScopeCommit, resolveModelScopeGguf, resolveModelScopeSnapshot,
} from '../src/controller/modelScope.mjs';
import { followRedirects } from '../src/controller/netGuard.mjs';
import { createRegistryService } from '../src/controller/registries.mjs';
import { createWeightStores } from '../src/controller/weightStores.mjs';
import { ggufBytes, modelPairs } from './gguf-fixture.mjs';
import { createTestTransport, fakeServer } from './net-fixture.mjs';
import { KEY, codeOf, deferred, registryHarness, until } from './registry-fixture.mjs';

const REPO = 'acme/models';
const COMMIT = 'c'.repeat(40);
const OTHER_COMMIT = 'd'.repeat(40);
const HF_TOKEN = 'hf_home_token_value';
const MS_TOKEN = 'modelscope-credential-value';
const MS_URL = 'https://ms.test';
const SHARD_1 = 'm-Q4_K_M-00001-of-00002.gguf';
const SHARD_2 = 'm-Q4_K_M-00002-of-00002.gguf';
const MIB = 1024 * 1024;

const sha256Of = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const pad = (header, size) => Buffer.concat([header, Buffer.alloc(Math.max(0, size - header.length), 0xee)]);
// A real GGUF header (24 layers, 14 heads, 2 KV heads) padded to a small file, so it can be looked up and downloaded.
const HEADER = pad(ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896 }) }).bytes, 65536);
const FILES = Object.freeze({
    '.gitattributes': Buffer.from('*.gguf filter=lfs\n'),
    'README.md': Buffer.from('# models\n'),
    'm-Q4_K_M.gguf': HEADER,
    'quant/m-Q8_0.gguf': pad(HEADER, 20_000 + HEADER.length),
    [SHARD_1]: Buffer.alloc(40_000, 0x61),
    [SHARD_2]: Buffer.alloc(30_000, 0x62),
    'config.json': Buffer.from('{"num_hidden_layers":2,"hidden_size":64,"num_attention_heads":4,"max_position_embeddings":2048}\n'),
    'model.safetensors': Buffer.alloc(50_000, 0x63),
    'original/model.safetensors': Buffer.alloc(10_000, 0x64),
});
// What the registry serves through a redirect to the CDN; the rest it serves itself.
const LFS = new Set(['m-Q4_K_M.gguf', 'quant/m-Q8_0.gguf', SHARD_1, SHARD_2, 'model.safetensors', 'original/model.safetensors']);
const blob = (name, bytes, extra = {}) => ({
    Type: 'blob', Name: name.split('/').pop(), Path: name, Size: bytes.length, Sha256: sha256Of(bytes), IsLFS: LFS.has(name), Mode: '33188', ...extra,
});
const tree = (name) => ({ Type: 'tree', Name: name, Path: name, Size: 0, Sha256: '', Mode: '16384' });
const LISTING = Object.freeze([tree('quant'), tree('original'), ...Object.entries(FILES).map(([name, bytes]) => blob(name, bytes))]);
const envelope = (Data) => ({ Code: 200, Data, Message: 'success', RequestId: 'r', Success: true });
const commitsOf = (id = COMMIT) => envelope({ Commit: [{ Id: id, Title: 't' }], TotalCount: 1 });

// ------------------------------------------------------------------ the fakes

const bare = (res, status, headers = {}) => {
    res.writeHead(status, { 'Content-Length': 0, ...headers });
    res.end();
};
// JSON as the registry sends it; a gzip mode answers compressed (always, or only when the request allows it).
function reply(world, req, res, status, body, headers = {}) {
    let bytes = Buffer.from(typeof body === 'string' ? body : JSON.stringify(body));
    const extra = { 'Content-Type': 'application/json', ...headers };
    if (world.gzip === 'always' || (world.gzip === 'asked' && /gzip/.test(req.headers['accept-encoding'] ?? ''))) {
        bytes = zlib.gzipSync(bytes);
        extra['Content-Encoding'] = 'gzip';
    }
    res.writeHead(status, { 'Content-Length': bytes.length, ...extra });
    res.end(bytes);
}
// Bytes with the range semantics of a file host: a ranged request gets 206 and Content-Range.
function serve(req, res, bytes) {
    const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
    if (!range) {
        res.writeHead(200, { 'Content-Length': bytes.length });
        res.end(bytes);
        return;
    }
    const from = Number(range[1]);
    const to = Math.min(range[2] === '' ? bytes.length - 1 : Number(range[2]), bytes.length - 1);
    if (from >= bytes.length) return bare(res, 416);
    res.writeHead(206, { 'Content-Length': to - from + 1, 'Content-Range': `bytes ${from}-${to}/${bytes.length}` });
    res.end(bytes.subarray(from, to + 1));
}

const registryHandler = (world) => (req, res, url) => {
    const base = `/api/v1/models/${REPO}`;
    if (url.pathname === `${base}/commits`) {
        const send = () => (world.commitsReply
            ? world.commitsReply(req, res, url)
            : reply(world, req, res, 200, world.commits(url.searchParams.get('Ref'))));
        if (world.gate) world.gate.promise.then(send); else send();
    } else if (url.pathname === `${base}/repo/files`) {
        if (url.searchParams.get('Revision') !== COMMIT || url.searchParams.get('Recursive') !== 'true') return bare(res, 404);
        if (world.filesReply) return world.filesReply(req, res);
        return reply(world, req, res, 200, envelope({ Files: world.listing }));
    } else if (url.pathname === `/openapi/v1/models/${REPO}`) {
        if (world.openapiReply) return world.openapiReply(req, res);
        return reply(world, req, res, 200, { success: true, data: { gated: false, license: 'apache-2.0', private: false } });
    } else if (url.pathname === `${base}/repo`) {
        const file = url.searchParams.get('FilePath');
        if (url.searchParams.get('Revision') !== COMMIT || !world.files[file]) return bare(res, 404);
        if (!LFS.has(file)) return serve(req, res, world.files[file]);
        return bare(res, 302, { Location: `https://cdn.test/blob/${encodeURIComponent(file)}?auth_key=signed-url-secret` });
    } else {
        bare(res, 404);
    }
};

/** The registry `ms.test`, a fake huggingface.co (`hf.test`), a CDN (`cdn.test`) and a bystander (`other.test`). */
async function createWorld(t, options = {}) {
    const world = {
        files: { ...FILES },
        listing: LISTING,
        commits: (ref) => (ref === 'master' || ref === 'v1.0' || ref === COMMIT ? commitsOf() : envelope({ Commit: null, TotalCount: 0 })),
        gzip: null,
        gate: null,
        cdnHook: null,
        ...options,
    };
    const start = async (handler) => {
        const server = await fakeServer(handler);
        t.after(() => server.close());
        return server;
    };
    world.ms = await start(registryHandler(world));
    world.hf = await start((req, res) => bare(res, 404));
    world.other = await start((req, res) => bare(res, 404));
    world.cdn = await start((req, res, url) => {
        const bytes = world.files[decodeURIComponent(url.pathname.replace('/blob/', ''))];
        if (!bytes) bare(res, 404);
        else if (!world.cdnHook?.(req, res, bytes, url)) serve(req, res, bytes);
    });
    const routes = { 'ms.test': world.ms.origin, 'hf.test': world.hf.origin, 'cdn.test': world.cdn.origin, 'other.test': world.other.origin };
    const inner = createTestTransport();
    t.after(() => inner.close());
    world.transport = Object.assign(async (url, init) => {
        const target = new URL(url);
        const real = routes[target.host];
        if (!real) throw new Error(`unrouted host ${target.host}`);
        const to = new URL(real);
        target.protocol = to.protocol;
        target.host = to.host;
        return inner(target.href, init);
    }, { allowHttp: false });
    // The fetch the adapter functions take: redirects followed by hand, credentials home only, metadata at home.
    world.fetchImpl = followRedirects(world.transport, { homeOrigin: MS_URL, sameOriginOnly: true });
    return world;
}

const authOf = (server) => server.requests.map((request) => request.headers.authorization);
const pathsOf = (server) => server.requests.map((request) => request.url.replace(/\?.*$/, ''));
const downloadsOf = (server) => server.requests.filter((request) => request.url.startsWith(`/api/v1/models/${REPO}/repo?`));
const fileOf = (request) => new URL(request.url, 'http://x').searchParams.get('FilePath');

function scratch(t) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-ms-'));
    const cleanups = [];
    t.after(async () => {
        for (const cleanup of cleanups.reverse()) await cleanup();
        fs.rmSync(dir, { recursive: true, force: true });
    });
    return { dir, later: (cleanup) => cleanups.push(cleanup) };
}

// A real controller (real downloader and stores) whose model paths go through the world's transport.
function controllerOn(t, world, { disk = scratch(t), env = {} } = {}) {
    const h = registryHarness(t, {
        dataDir: disk.dir,
        env: { PATH: '/usr/bin', HF_TOKEN, ...env },
        extra: { registryTransport: world.transport, hfBaseUrl: 'https://hf.test', download: undefined, inspect: undefined, remove: undefined },
    });
    disk.later(() => h.controller.drain());
    return h;
}

const MS = Object.freeze({ id: 'ms', kind: 'modelscope', url: MS_URL });
async function addMs(h, { token = MS_TOKEN } = {}) {
    await h.controller.saveRegistry({ registry: MS });
    if (token) await h.controller.setRegistryCredential({ id: 'ms', token });
}
const lookupOf = (h, args) => h.controller.lookupModel({ registry: 'ms', ...args });
const gguf = (extra = {}) => ({ type: 'huggingface', registry: 'ms', repo: REPO, file: 'm-Q4_K_M.gguf', revision: 'master', ...extra });
const entryOf = (id, sources) => ({ id, displayName: id, contextLength: 4096, architecture: 'dense', memory: { layers: 4, kvBytesPerToken: 1000 }, sources });
const pinnedGguf = (file = 'm-Q4_K_M.gguf') => gguf({ file, commit: COMMIT, size: FILES[file].length, sha256: sha256Of(FILES[file]) });
const shardsOf = (...names) => names.map((file) => ({ file, size: FILES[file].length, sha256: sha256Of(FILES[file]) }));
const pinnedSplit = () => gguf({ file: SHARD_1, commit: COMMIT, size: FILES[SHARD_1].length + FILES[SHARD_2].length, shards: shardsOf(SHARD_1, SHARD_2) });
const pinnedSnapshot = (extra = {}) => ({
    type: 'hf-snapshot', registry: 'ms', repo: REPO, revision: 'master', commit: COMMIT,
    files: ['config.json', 'model.safetensors'].map((name) => ({ path: name, size: FILES[name].length, sha256: sha256Of(FILES[name]) })),
    size: FILES['config.json'].length + FILES['model.safetensors'].length,
    ...extra,
});

// A store over the real downloader and a real registry service, with one attempt's retry not waiting.
function storeOn(t, world, disk = scratch(t)) {
    const registries = createRegistryService({ dataDir: disk.dir, credentialsKey: KEY, hfBaseUrl: 'https://hf.test' });
    const ops = registries.operations({ queue: createCommandQueue(), references: () => ({ entries: [], active: null }) });
    const stores = createWeightStores({
        dataDir: disk.dir,
        env: { HF_TOKEN },
        hfBaseUrl: 'https://hf.test',
        inspect: inspectArtifact,
        download: (options) => downloadArtifact({ ...options, sleep: async () => {} }),
        downloadSnapshot: (options) => downloadSnapshotFile({ ...options, sleep: async () => {} }),
        remove: removeArtifact,
        resolveHf: resolveHuggingFaceArtifact,
        state: () => ({}),
        save: () => {},
        activeArtifact: () => null,
        registryTransport: world?.transport ?? null,
        registries,
    });
    return { registries, ops, stores, dir: disk.dir };
}
async function storeWithRegistry(t, world) {
    const made = storeOn(t, world);
    await made.ops.saveRegistry({ registry: MS });
    await made.ops.setRegistryCredential({ id: 'ms', token: MS_TOKEN });
    return made;
}

// ------------------------------------------------------------------ lookup

test('lookup: commit, then the listing, then the advisory OpenAPI, then the header; every request carries the bearer to ms only', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    const found = await lookupOf(h, { repo: REPO, file: 'm-Q4_K_M.gguf' });
    assert.deepEqual(found.registry, { id: 'ms', kind: 'modelscope', label: 'ms' });
    assert.deepEqual([found.provider, found.revision, found.commit, found.format, found.gated, found.license, found.truncated], ['modelscope', 'master', COMMIT, 'gguf', false, 'apache-2.0', false]);
    assert.deepEqual(found.files.map((row) => row.file), [SHARD_1, 'm-Q4_K_M.gguf', 'quant/m-Q8_0.gguf']);
    const [split, single] = found.files;
    assert.deepEqual([single.size, single.sha256, single.gitOid, single.quantization, single.shards], [HEADER.length, sha256Of(HEADER), null, 'Q4_K_M', null]);
    assert.deepEqual([split.size, split.sha256, split.shards], [FILES[SHARD_1].length + FILES[SHARD_2].length, null, shardsOf(SHARD_1, SHARD_2)]);
    assert.deepEqual([found.sizing.source, found.sizing.memory.layers], ['gguf-header', 24]);
    // The order and the shape of the requests.
    assert.deepEqual(world.ms.requests.map((request) => request.url), [
        `/api/v1/models/${REPO}/commits?Ref=master&PageNumber=1&PageSize=1`,
        `/api/v1/models/${REPO}/repo/files?Revision=${COMMIT}&Recursive=true`,
        `/openapi/v1/models/${REPO}`,
        `/api/v1/models/${REPO}/repo?Revision=${COMMIT}&FilePath=m-Q4_K_M.gguf`,
    ]);
    assert.ok(authOf(world.ms).every((value) => value === `Bearer ${MS_TOKEN}`), 'the exact bearer, on every request at home');
    assert.ok(world.ms.requests.every((request) => request.headers.cookie === undefined), 'a Bearer header, not a session cookie');
    assert.ok(world.ms.requests.every((request) => request.headers['accept-encoding'] === 'identity'));
    assert.ok(world.cdn.requests.length >= 1 && authOf(world.cdn).every((value) => value === undefined), 'the CDN gets no authorization');
    assert.equal(world.cdn.requests.every((request) => request.headers.range?.startsWith('bytes=0-')), true, 'a ranged header read');
    assert.equal(world.hf.requests.length, 0);
    assert.equal(JSON.stringify(found).includes(MS_TOKEN), false);
});

test('lookup: the HF_TOKEN stays at huggingface.co and the ModelScope token never leaves ms', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    await assert.rejects(h.controller.lookupModel({ repo: REPO }), { code: 'not_found' });
    assert.ok(world.hf.requests.length >= 1 && authOf(world.hf).every((value) => value === `Bearer ${HF_TOKEN}`));
    assert.equal(world.ms.requests.length, 0);
    await lookupOf(h, { repo: REPO });
    assert.ok(authOf(world.ms).length >= 3 && authOf(world.ms).every((value) => value === `Bearer ${MS_TOKEN}`));
    assert.ok(authOf(world.hf).every((value) => value === `Bearer ${HF_TOKEN}`), 'nothing new reached huggingface.co');
    // Without a stored credential nothing is sent, and HF_TOKEN does not stand in for it.
    await h.controller.clearRegistryCredential({ id: 'ms' });
    world.ms.requests.length = 0;
    await lookupOf(h, { repo: REPO });
    assert.ok(world.ms.requests.length >= 3 && authOf(world.ms).every((value) => value === undefined));
});

test('lookup: a snapshot lists the top-level files only and sizes from config.json checked against its digest', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    const found = await lookupOf(h, { repo: REPO, format: 'hf', revision: 'v1.0' });
    assert.equal(found.revision, 'v1.0');
    assert.equal(found.commit, COMMIT);
    assert.deepEqual(found.files.map((row) => row.file), ['config.json', 'model.safetensors']);
    assert.ok(found.files.every((row) => /^[a-f0-9]{64}$/.test(row.sha256) && row.gitOid === null));
    assert.deepEqual([found.sizing.source, found.sizing.memory.layers, found.sizing.contextLength], ['config.json', 2, 2048]);
    assert.ok(world.ms.requests.some((request) => request.url.includes('Ref=v1.0')));
    // The same size with other bytes does not match the listing's digest.
    world.files['config.json'] = Buffer.from(FILES['config.json'].toString('utf8').replace('"num_hidden_layers":2', '"num_hidden_layers":9'));
    assert.equal(world.files['config.json'].length, FILES['config.json'].length);
    await assert.rejects(lookupOf(h, { repo: REPO, format: 'hf' }), (error) => error.code === 'lookup_failed' && /digest/.test(error.message));
});

test('lookup: the commit rules; /commits is asked even for a 40-hex revision', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    const answer = (body, status = 200) => { world.commitsReply = (req, res) => reply(world, req, res, status, body); };
    const outcome = async (args = {}) => {
        try {
            const found = await lookupOf(h, { repo: REPO, ...args });
            return { commit: found.commit };
        } catch (error) {
            return { code: error.code, reason: error.details?.reason };
        }
    };
    answer(envelope({ Commit: null, TotalCount: 0 }));
    assert.deepEqual(await outcome(), { code: 'not_found', reason: undefined });
    answer(envelope({ Commit: [], TotalCount: 0 }));
    assert.equal((await outcome()).code, 'not_found');
    answer({ Code: 10010205001, Message: 'record not found', Success: false });
    assert.deepEqual(await outcome(), { code: 'lookup_failed', reason: undefined });
    answer({ Code: 200, Data: { Commit: [{ Id: COMMIT }] }, Success: false });
    assert.equal((await outcome()).code, 'lookup_failed', 'Success must be true');
    answer({ Code: 200, Data: { Commit: [{ Id: COMMIT }] } });
    assert.deepEqual(await outcome(), { code: 'lookup_failed', reason: 'nonconforming-registry' });
    answer(envelope(null));
    assert.deepEqual(await outcome(), { code: 'lookup_failed', reason: 'nonconforming-registry' });
    for (const id of ['abc123', 'z'.repeat(40), COMMIT.slice(1), `${COMMIT}0`, 7, null]) {
        answer(commitsOf(id));
        assert.deepEqual(await outcome(), { code: 'lookup_failed', reason: 'nonconforming-registry' }, String(id));
    }
    answer(commitsOf(COMMIT.toUpperCase()));
    assert.deepEqual(await outcome(), { commit: COMMIT }, 'an upper-case id is the same commit');
    for (const [status, code] of [[404, 'not_found'], [401, 'access_denied'], [403, 'access_denied'], [429, 'lookup_failed'], [500, 'lookup_failed']]) {
        answer(envelope(null), status);
        assert.equal((await outcome()).code, code, String(status));
    }
    // A 40-hex revision is asked, and must come back as itself.
    world.commitsReply = null;
    world.commits = () => commitsOf(COMMIT);
    world.ms.requests.length = 0;
    assert.deepEqual(await outcome({ revision: COMMIT }), { commit: COMMIT });
    assert.ok(world.ms.requests[0].url.includes(`Ref=${COMMIT}`), 'the commit endpoint was asked');
    assert.deepEqual(await outcome({ revision: OTHER_COMMIT }), { code: 'lookup_failed', reason: 'nonconforming-registry' });
    // A redirect of a metadata request to another origin is refused before it is made.
    world.commitsReply = (req, res) => bare(res, 302, { Location: 'https://other.test/commits' });
    assert.deepEqual(await outcome(), { code: 'lookup_failed', reason: 'redirect' });
    assert.equal(world.other.requests.length, 0);
});

test('the listing rules: unsafe paths, duplicates, trees, bad sizes and non-64-hex digests are skipped; null or empty is not found', async (t) => {
    const world = await createWorld(t);
    const good = blob('g-Q4_K_M.gguf', HEADER);
    const upper = blob('upper.json', FILES['config.json'], { Sha256: sha256Of(FILES['config.json']).toUpperCase() });
    const bad = (path_, extra = {}) => blob(path_, HEADER, extra);
    world.listing = [
        good, upper,
        bad('../x.gguf'), bad('/abs.gguf'), bad('a//b.gguf'), bad('a/./b.gguf'), bad('a\\b.gguf'), bad('a/../b.gguf'), bad('nul\u0000.gguf'), bad(''),
        bad('dup.gguf'), bad('dup.gguf', { Sha256: 'e'.repeat(64) }),
        bad('short.gguf', { Sha256: 'a'.repeat(63) }), bad('long.gguf', { Sha256: 'a'.repeat(65) }), bad('nothex.gguf', { Sha256: 'z'.repeat(64) }),
        bad('empty.gguf', { Sha256: '' }), bad('none.gguf', { Sha256: undefined }),
        bad('zero.gguf', { Size: 0 }), bad('text.gguf', { Size: '5' }), bad('float.gguf', { Size: 1.5 }),
        tree('dir.gguf'), { ...bad('kind.gguf'), Type: 'commit' }, null, 'junk', { Type: 'blob' },
    ];
    const options = { repo: REPO, commit: COMMIT, fetchImpl: world.fetchImpl, baseUrl: MS_URL };
    const listing = await listModelScopeFiles(options);
    assert.deepEqual(listing.files.map((file) => file.path), ['g-Q4_K_M.gguf', 'upper.json']);
    assert.equal(listing.files[1].sha256, sha256Of(FILES['config.json']), 'a digest is kept in lower case');
    assert.equal(listing.incomplete, false);
    for (const Files of [null, []]) {
        world.listing = Files;
        await assert.rejects(listModelScopeFiles(options), { code: 'NOT_FOUND' });
        const h = controllerOn(t, world);
        await addMs(h);
        await assert.rejects(lookupOf(h, { repo: REPO }), { code: 'not_found' });
    }
    world.listing = 'text';
    await assert.rejects(listModelScopeFiles(options), { code: 'RESOLVE_FAILED', details: { reason: 'nonconforming-registry' } });
    // HTTP 404 for the listing (an unknown commit) is not found too.
    world.listing = LISTING;
    await assert.rejects(listModelScopeFiles({ ...options, commit: OTHER_COMMIT }), { code: 'RESOLVE_FAILED', details: { status: 404 } });
});

test('A14: 2999 entries work; 3000 are truncated for a lookup and listing-incomplete for a pin', async (t) => {
    const world = await createWorld(t);
    const filler = (count) => Array.from({ length: count }, (_, index) => blob(`f${index}.bin`, Buffer.from([index % 251, 1])));
    const h = controllerOn(t, world);
    await addMs(h);
    assert.equal(LISTING_LIMIT, 3000);
    world.listing = [blob('m-Q4_K_M.gguf', HEADER), ...filler(LISTING_LIMIT - 2)];
    assert.equal(world.listing.length, 2999);
    const found = await lookupOf(h, { repo: REPO });
    assert.equal(found.truncated, false);
    assert.deepEqual(found.files.map((row) => row.file), ['m-Q4_K_M.gguf']);
    assert.equal((await h.controller.addModel(entryOf('boundary', { gguf: gguf() }))).model.sources.gguf.commit, COMMIT);
    world.listing = [blob('m-Q4_K_M.gguf', HEADER), ...filler(LISTING_LIMIT - 1)];
    assert.equal(world.listing.length, 3000);
    const cut = await lookupOf(h, { repo: REPO });
    assert.equal(cut.truncated, true);
    assert.deepEqual(cut.files.map((row) => row.file), ['m-Q4_K_M.gguf'], 'what the listing has is still offered');
    for (const source of [gguf({ file: SHARD_1 }), { type: 'hf-snapshot', registry: 'ms', repo: REPO, revision: 'master' }, gguf()]) {
        await assert.rejects(h.controller.addModel(entryOf('too-long', { [source.type === 'huggingface' ? 'gguf' : 'hf']: source })),
            (error) => error instanceof DownloadError && error.code === 'RESOLVE_FAILED' && error.details.reason === 'listing-incomplete', JSON.stringify(source));
    }
    assert.equal(h.controller.state.registry.length, 1, 'nothing was saved by the refused Adds');
});

test('the caps: a commits or openapi answer of 1 MiB is read and 1 MiB + 1 is not; a listing of 4 MiB is read and 4 MiB + 1 is not', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    // A body of exactly `size` bytes: the envelope with its Message padded.
    const sized = (make, size) => {
        const empty = JSON.stringify(make(''));
        return JSON.stringify(make('x'.repeat(size - empty.length)));
    };
    const send = (size, make) => (req, res) => reply(world, req, res, 200, sized(make, size));
    const outcome = async () => lookupOf(h, { repo: REPO }).then((found) => ({ gated: found.gated, files: found.files.length }), (error) => ({ code: error.code, reason: error.details?.reason }));
    const commits = (message) => ({ ...commitsOf(), Message: message });
    world.commitsReply = send(MIB, commits);
    assert.deepEqual(await outcome(), { gated: false, files: 3 });
    world.commitsReply = send(MIB + 1, commits);
    assert.deepEqual(await outcome(), { code: 'lookup_failed', reason: 'too-large' });
    world.commitsReply = null;
    const listing = (message) => ({ ...envelope({ Files: LISTING }), Message: message });
    world.filesReply = send(4 * MIB, listing);
    assert.deepEqual(await outcome(), { gated: false, files: 3 });
    world.filesReply = send(4 * MIB + 1, listing);
    assert.deepEqual(await outcome(), { code: 'lookup_failed', reason: 'too-large' });
    world.filesReply = null;
    // The advisory answer is optional: too large, a failure or nonsense leave gating and licence unknown, not failed.
    const advisory = (message) => ({ success: true, data: { gated: true, license: 'mit', readme: message } });
    world.openapiReply = send(MIB + 1, advisory);
    assert.deepEqual(await outcome(), { gated: null, files: 3 });
    world.openapiReply = send(MIB, advisory);
    assert.deepEqual(await outcome(), { gated: 'manual', files: 3 });
    for (const answer of [(req, res) => bare(res, 404), (req, res) => bare(res, 500), (req, res) => reply(world, req, res, 200, 'not json'),
        (req, res) => reply(world, req, res, 200, { success: false }), (req, res) => reply(world, req, res, 200, [])]) {
        world.openapiReply = answer;
        const found = await lookupOf(h, { repo: REPO });
        assert.deepEqual([found.gated, found.license], [null, null]);
    }
    world.openapiReply = (req, res) => reply(world, req, res, 200, { success: true, data: { gated: 'weird', license: 'a‮b' } });
    assert.deepEqual((({ gated, license }) => ({ gated, license }))(await lookupOf(h, { repo: REPO })), { gated: null, license: 'ab' });
});

test('gzip: the requests ask for identity, so a registry that compresses only when allowed answers plain JSON; one that always compresses fails cleanly', async (t) => {
    const world = await createWorld(t, { gzip: 'asked' });
    const h = controllerOn(t, world);
    await addMs(h);
    assert.equal((await lookupOf(h, { repo: REPO })).commit, COMMIT);
    assert.ok(world.ms.requests.every((request) => request.headers['accept-encoding'] === 'identity'));
    world.gzip = 'always';
    await assert.rejects(lookupOf(h, { repo: REPO }), (error) => error.code === 'lookup_failed' && !/token|Bearer/.test(error.message));
    // The guarded transport refuses an encoded body outright.
    await assert.rejects(resolveModelScopeCommit({ repo: REPO, revision: 'master', fetchImpl: world.fetchImpl, baseUrl: MS_URL }), { code: 'HTTP_ERROR' });
});

test('a lookup of a file the listing does not offer is not_found; the revision default is master, not main; a/b/c is refused before any request (A19)', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    await assert.rejects(lookupOf(h, { repo: REPO, file: 'nope.gguf' }), { code: 'not_found' });
    await assert.rejects(lookupOf(h, { repo: REPO, file: SHARD_2 }), { code: 'invalid_request' });
    world.ms.requests.length = 0;
    await assert.rejects(lookupOf(h, { repo: 'a/b/c' }), { code: 'invalid_request' });
    await assert.rejects(lookupOf(h, { repo: REPO, provider: 'huggingface' }), { code: 'invalid_request' });
    assert.equal(world.ms.requests.length, 0);
    await lookupOf(h, { repo: REPO, revision: 'v1.0' });
    assert.ok(world.ms.requests[0].url.includes('Ref=v1.0'), 'a revision that was typed is used');
});

test('every adapter function throws a TypeError without a fetchImpl (no default)', async () => {
    const base = { repo: REPO, baseUrl: MS_URL };
    await assert.rejects(resolveModelScopeCommit({ ...base, revision: 'master' }), { name: 'TypeError', message: 'fetchImpl is required' });
    await assert.rejects(listModelScopeFiles({ ...base, commit: COMMIT }), { name: 'TypeError', message: 'fetchImpl is required' });
    await assert.rejects(resolveModelScopeGguf({ ...base, file: 'm.gguf' }), { name: 'TypeError', message: 'fetchImpl is required' });
    await assert.rejects(resolveModelScopeSnapshot(base), { name: 'TypeError', message: 'fetchImpl is required' });
    await assert.rejects(lookupModelScope({ repo: REPO }, { baseUrl: MS_URL }), { name: 'TypeError', message: 'fetchImpl is required' });
});

// ------------------------------------------------------------------ pins

test('Add pins through ModelScope: GGUF, split GGUF and snapshot record size and sha256 at the commit, with the bearer and no download', async (t) => {
    const world = await createWorld(t);
    const disk = scratch(t);
    const h = controllerOn(t, world, { disk });
    await addMs(h);
    assert.deepEqual({ ...(await h.controller.addModel(entryOf('ms-gguf', { gguf: gguf() }))).model.sources.gguf }, pinnedGguf());
    assert.deepEqual({ ...(await h.controller.addModel(entryOf('ms-nested', { gguf: gguf({ file: 'quant/m-Q8_0.gguf' }) }))).model.sources.gguf }, pinnedGguf('quant/m-Q8_0.gguf'));
    assert.deepEqual({ ...(await h.controller.addModel(entryOf('ms-split', { gguf: gguf({ file: SHARD_1 }) }))).model.sources.gguf }, pinnedSplit());
    const snapshot = (await h.controller.addModel(entryOf('ms-snapshot', { hf: { type: 'hf-snapshot', registry: 'ms', repo: REPO, revision: 'master' } }))).model.sources.hf;
    assert.deepEqual(JSON.parse(JSON.stringify(snapshot)), JSON.parse(JSON.stringify(pinnedSnapshot())));
    assert.ok(snapshot.files.every((file) => file.gitOid === undefined && /^[a-f0-9]{64}$/.test(file.sha256)), 'sha256 only');
    assert.ok(authOf(world.ms).length >= 8 && authOf(world.ms).every((value) => value === `Bearer ${MS_TOKEN}`));
    assert.equal(downloadsOf(world.ms).length, 0, 'a pin reads metadata only');
    assert.equal(world.cdn.requests.length, 0);
    assert.equal(world.hf.requests.length, 0);
    const stored = fs.readFileSync(path.join(disk.dir, 'state', 'controller.json'), 'utf8');
    assert.equal(stored.includes(MS_TOKEN), false);
    // Twin of the 40-hex rule: a typed commit is asked for and must come back.
    assert.equal((await h.controller.addModel(entryOf('ms-commit', { gguf: gguf({ revision: COMMIT }) }))).model.sources.gguf.commit, COMMIT);
    world.commits = () => commitsOf(COMMIT);
    await assert.rejects(h.controller.addModel(entryOf('ms-other', { gguf: gguf({ revision: OTHER_COMMIT }) })), { code: 'RESOLVE_FAILED' });
    await assert.rejects(h.controller.addModel(entryOf('no-rev', { gguf: { ...gguf(), revision: undefined } })), { code: 'invalid_model' });
});

test('a pin refuses what the listing cannot support: a missing file, a split set missing a shard, a snapshot without safetensors, an unknown revision', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    const add = (sources) => h.controller.addModel(entryOf('refused', sources));
    await assert.rejects(add({ gguf: gguf({ file: 'missing.gguf' }) }), { code: 'NOT_FOUND' });
    world.listing = LISTING.filter((entry) => entry.Path !== SHARD_2);
    await assert.rejects(add({ gguf: gguf({ file: SHARD_1 }) }), { code: 'NOT_FOUND' });
    world.listing = LISTING.filter((entry) => !entry.Path.endsWith('.safetensors'));
    await assert.rejects(add({ hf: { type: 'hf-snapshot', registry: 'ms', repo: REPO, revision: 'master' } }), { code: 'NOT_FOUND' });
    world.listing = LISTING;
    await assert.rejects(add({ gguf: gguf({ revision: 'nope' }) }), { code: 'NOT_FOUND' });
    // A file with a digest that is not 64 hex is not a file the pin can use, even though the listing names it.
    world.listing = LISTING.map((entry) => (entry.Path === 'm-Q4_K_M.gguf' ? { ...entry, Sha256: 'a'.repeat(63) } : entry));
    await assert.rejects(add({ gguf: gguf() }), { code: 'NOT_FOUND' });
    assert.equal(h.controller.state.registry.length, 0);
});

// ------------------------------------------------------------------ keys, paths, downloads

test('ModelScope files and keys live under _r/<id> like a mirror\'s, derived from the source alone', async (t) => {
    const { stores, dir } = storeOn(t, null);
    const gguf_ = path.join(dir, 'models', 'gguf');
    assert.equal(stores.huggingface.paths(pinnedGguf()).file, path.join(gguf_, '_r', 'ms', 'acme', 'models', COMMIT, 'm-Q4_K_M.gguf'));
    assert.equal(stores.huggingface.key(pinnedGguf()), `gguf:_r/ms/${REPO}@${COMMIT}/m-Q4_K_M.gguf`);
    assert.equal(stores.huggingface.key(pinnedSplit()), `gguf:_r/ms/${REPO}@${COMMIT}/${SHARD_1}#2`);
    assert.equal(stores['hf-snapshot'].key(pinnedSnapshot()), `hf:_r/ms/${REPO}@${COMMIT}`);
    assert.equal((await stores['hf-snapshot'].state(pinnedSnapshot())).state, 'absent');
    assert.equal(modelScopeFileUrl({ baseUrl: MS_URL, repo: REPO, commit: COMMIT, file: 'quant/m-Q8_0.gguf' }),
        `${MS_URL}/api/v1/models/${REPO}/repo?Revision=${COMMIT}&FilePath=quant%2Fm-Q8_0.gguf`);
});

test('Run downloads a ModelScope GGUF into _r/<id>: ms sees the bearer, the CDN it redirects to none; provenance is the ms URL only (A7)', async (t) => {
    const world = await createWorld(t);
    const disk = scratch(t);
    const h = controllerOn(t, world, { disk });
    await addMs(h);
    await h.controller.addModel(entryOf('ms-run', { gguf: gguf() }));
    await h.controller.run({ modelId: 'ms-run', runnerId: 'llama.cpp', requestId: 'request-ms-run' });
    await until(() => ['ready', 'error'].includes(h.controller.state.deployment?.phase), 5000);
    assert.equal(h.controller.state.deployment.phase, 'ready', h.controller.state.deployment.error);
    const file = path.join(disk.dir, 'models', 'gguf', '_r', 'ms', 'acme', 'models', COMMIT, 'm-Q4_K_M.gguf');
    assert.ok(fs.readFileSync(file).equals(HEADER));
    const downloads = downloadsOf(world.ms);
    assert.deepEqual(downloads.map(fileOf), ['m-Q4_K_M.gguf']);
    assert.equal(downloads[0].headers.authorization, `Bearer ${MS_TOKEN}`);
    assert.ok(downloads[0].headers['accept-encoding'] === 'identity');
    assert.ok(world.cdn.requests.length >= 1 && authOf(world.cdn).every((value) => value === undefined));
    const [entry] = h.controller.state.deployment.provenance;
    assert.equal(entry.source, `${MS_URL}/api/v1/models/${REPO}/repo?Revision=${COMMIT}&FilePath=m-Q4_K_M.gguf`);
    const text = JSON.stringify(h.controller.state.deployment.provenance);
    assert.ok(!text.includes(MS_TOKEN) && !text.includes('cdn.test') && !text.includes('auth_key'));
    assert.equal(world.hf.requests.length, 0);
    await h.controller.stop();
});

test('a split GGUF and a snapshot download from ModelScope: every file from its /repo URL, inline files without a redirect, the CDN without the bearer', async (t) => {
    const world = await createWorld(t);
    const { stores, dir } = await storeWithRegistry(t, world);
    const split = await stores.huggingface.fetch({ artifact: pinnedSplit(), signal: undefined });
    assert.equal(split.files.length, 2);
    for (const name of [SHARD_1, SHARD_2]) {
        assert.ok(fs.readFileSync(path.join(dir, 'models', 'gguf', '_r', 'ms', 'acme', 'models', COMMIT, name)).equals(FILES[name]), name);
    }
    assert.deepEqual(split.provenance.map((entry) => fileOf({ url: entry.source.slice(MS_URL.length) })), [SHARD_1, SHARD_2]);
    const nested = await stores.huggingface.fetch({ artifact: pinnedGguf('quant/m-Q8_0.gguf'), signal: undefined });
    assert.ok(fs.readFileSync(nested.path).equals(FILES['quant/m-Q8_0.gguf']));
    assert.equal(downloadsOf(world.ms).at(-1).url.includes('FilePath=quant%2Fm-Q8_0.gguf'), true, 'the path is one encoded query value');
    const cdnBefore = world.cdn.requests.length;
    const snapshot = await stores['hf-snapshot'].fetch({ artifact: pinnedSnapshot(), signal: undefined });
    const expected = snapshotPaths({ root: path.join(dir, 'models', 'hf'), repo: REPO, commit: COMMIT, registry: 'ms' });
    assert.equal(snapshot.path, expected.dir);
    for (const name of ['config.json', 'model.safetensors']) assert.ok(fs.readFileSync(path.join(expected.dir, name)).equals(FILES[name]), name);
    assert.equal(world.cdn.requests.length - cdnBefore, 1, 'config.json was served by ms itself, model.safetensors by the CDN');
    assert.ok(downloadsOf(world.ms).every((request) => request.headers.authorization === `Bearer ${MS_TOKEN}`));
    assert.ok(authOf(world.cdn).every((value) => value === undefined));
    assert.equal((await stores['hf-snapshot'].state(pinnedSnapshot())).state, 'complete');
    assert.equal((await stores['hf-snapshot'].state({ ...pinnedSnapshot(), registry: undefined })).state, 'absent');
});

test('A13: the right size with the wrong bytes is SHA256_MISMATCH, the partial is removed and no final file exists', async (t) => {
    const world = await createWorld(t);
    const { stores, dir } = await storeWithRegistry(t, world);
    for (const name of ['m-Q4_K_M.gguf', 'model.safetensors']) world.files[name] = Buffer.alloc(FILES[name].length, 0x5a);
    const isMismatch = (error) => error instanceof DownloadError && error.code === 'SHA256_MISMATCH';
    await assert.rejects(stores.huggingface.fetch({ artifact: pinnedGguf(), signal: undefined }), isMismatch);
    const paths = artifactPaths({ root: path.join(dir, 'models', 'gguf'), artifact: pinnedGguf() });
    assert.deepEqual([fs.existsSync(paths.file), fs.existsSync(paths.partial), fs.existsSync(paths.meta)], [false, false, false]);
    await assert.rejects(stores['hf-snapshot'].fetch({ artifact: pinnedSnapshot(), signal: undefined }), isMismatch);
    const { dir: snapshotDir, stateDir } = snapshotPaths({ root: path.join(dir, 'models', 'hf'), repo: REPO, commit: COMMIT, registry: 'ms' });
    assert.equal(fs.existsSync(path.join(snapshotDir, 'model.safetensors')), false);
    assert.equal(fs.existsSync(path.join(stateDir, 'model.safetensors.partial')), false);
    // Twin: the right bytes verify.
    world.files['m-Q4_K_M.gguf'] = FILES['m-Q4_K_M.gguf'];
    assert.ok(fs.existsSync((await stores.huggingface.fetch({ artifact: pinnedGguf(), signal: undefined })).path));
});

// ------------------------------------------------------------------ A26

test('A26: a credential cleared and a registry disabled during a running download do not change that job\'s requests, even on a retry; the next lookup sees both', async (t) => {
    const hold = deferred();
    const arrived = deferred();
    let seen = 0;
    const world = await createWorld(t, {
        cdnHook: (req, res, bytes, url) => {
            if (!url.pathname.endsWith(encodeURIComponent(SHARD_1))) return false;
            seen += 1;
            if (seen > 1) return false;
            // The first answer is cut short once the test has changed the registry, so the job asks ms again.
            arrived.resolve();
            hold.promise.then(() => {
                res.writeHead(200, { 'Content-Length': bytes.length });
                res.write(bytes.subarray(0, 1000));
                setTimeout(() => res.destroy(), 20);
            });
            return true;
        },
    });
    const { registries, ops, stores } = await storeWithRegistry(t, world);
    const job = stores.huggingface.fetch({ artifact: pinnedSplit(), signal: undefined });
    await arrived.promise;
    await ops.clearRegistryCredential({ id: 'ms' });
    await ops.saveRegistry({ registry: { id: 'ms', enabled: false } });
    hold.resolve();
    assert.equal((await job).files.length, 2);
    const downloads = downloadsOf(world.ms);
    assert.deepEqual(downloads.map(fileOf), [SHARD_1, SHARD_1, SHARD_2], 'the cut-off shard was asked for again');
    assert.ok(downloads.every((request) => request.headers.authorization === `Bearer ${MS_TOKEN}`), 'the snapshot\'s credential, on the retry and the next shard');
    assert.ok(authOf(world.cdn).every((value) => value === undefined));
    const options = { fetchImpl: world.transport, registries };
    const { lookupInRegistry } = await import('../src/controller/registryLookup.mjs');
    await assert.rejects(lookupInRegistry({ registry: 'ms', repo: REPO }, options), { code: 'registry_disabled' });
    await assert.rejects(stores.huggingface.pin(gguf()), { code: 'registry_disabled' });
    await ops.saveRegistry({ registry: { id: 'ms', enabled: true } });
    world.ms.requests.length = 0;
    await lookupInRegistry({ registry: 'ms', repo: REPO, format: 'hf' }, options);
    assert.ok(world.ms.requests.length >= 3 && authOf(world.ms).every((value) => value === undefined), 'the cleared credential is not sent');
});

test('A26 (N5): a registry removed, re-created elsewhere, re-kinded, removed or disabled during an Add\'s pin fails the Add and saves nothing', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await h.controller.saveRegistry({ registry: MS });
    const during = async (change, id = 'during') => {
        world.gate = deferred();
        const arrivals = world.ms.requests.length;
        const running = codeOf(h.controller.addModel(entryOf(id, { gguf: gguf() })));
        await until(() => world.ms.requests.length > arrivals);
        await change();
        world.gate.resolve();
        return running;
    };
    const recreate = (changed) => async () => {
        await h.controller.removeRegistry({ id: 'ms' });
        await h.controller.saveRegistry({ registry: { ...MS, ...changed } });
    };
    assert.equal(await during(recreate({ url: 'https://elsewhere.test' })), 'registry_changed');
    await recreate({})();
    assert.equal(await during(recreate({ kind: 'huggingface' })), 'registry_changed');
    await recreate({})();
    assert.equal(await during(recreate({}), 'same-again'), 'resolved', 'twin: created again as it was');
    assert.equal(h.controller.state.registry.length, 1);
    const before = JSON.stringify(h.controller.state.registry);
    // The stored entry now uses ms, so ms cannot be removed; an unused second ModelScope registry can.
    await h.controller.saveRegistry({ registry: { id: 'ms-two', kind: 'modelscope', url: MS_URL } });
    const second = async (change) => {
        world.gate = deferred();
        const arrivals = world.ms.requests.length;
        const running = codeOf(h.controller.addModel(entryOf('second', { gguf: gguf({ registry: 'ms-two' }) })));
        await until(() => world.ms.requests.length > arrivals);
        await change();
        world.gate.resolve();
        return running;
    };
    assert.equal(await second(() => h.controller.removeRegistry({ id: 'ms-two' })), 'registry_unknown');
    await h.controller.saveRegistry({ registry: { id: 'ms-two', kind: 'modelscope', url: MS_URL } });
    assert.equal(await second(() => h.controller.saveRegistry({ registry: { id: 'ms-two', enabled: false } })), 'registry_disabled');
    assert.equal(JSON.stringify(h.controller.state.registry), before, 'state.registry unchanged');
    // An oci source cannot name a ModelScope registry.
    const oci = { type: 'oci', registry: 'ms', repository: 'library/x', tag: 'latest' };
    assert.equal(await codeOf(h.controller.addModel(entryOf('kind-mismatch', { gguf: oci }))), 'invalid_model');
});

test('A26: a drain ends a ModelScope pin and lookups with shutting_down; the third concurrent lookup is busy', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMs(h);
    world.gate = deferred();
    t.after(() => world.gate.resolve());
    const outcome = (promise) => promise.then(() => 'resolved', (error) => error.code);
    const first = outcome(lookupOf(h, { repo: REPO }));
    const second = outcome(lookupOf(h, { repo: REPO }));
    await until(() => world.ms.requests.length === 2);
    await assert.rejects(lookupOf(h, { repo: REPO }), { code: 'busy' });
    assert.equal(world.ms.requests.length, 2, 'the refused lookup sent nothing');
    const adding = outcome(h.controller.addModel(entryOf('draining', { gguf: gguf() })));
    await until(() => world.ms.requests.length === 3);
    const drain = h.controller.drain();
    assert.deepEqual(await Promise.all([first, second, adding]), ['shutting_down', 'shutting_down', 'shutting_down']);
    await drain;
    await assert.rejects(lookupOf(h, { repo: REPO }), { code: 'shutting_down' });
});
