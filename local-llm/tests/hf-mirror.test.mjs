// Hugging Face-compatible mirrors (DS002): a registry of kind `huggingface` at an https URL with an optional path
// prefix. Lookups, pins and downloads read it through the controller's guarded transport; its credential reaches its
// own origin only (never a CDN, never huggingface.co), `HF_TOKEN` never reaches it, its files and keys live under
// `_r/<id>`, and a registry that is gone or disabled never hides a model or its weights. The fake mirror, the fake
// huggingface.co and the fake CDN are loopback servers behind a transport that maps their https names onto them.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import zlib from 'node:zlib';

import { validateModel } from '../src/controller/catalog.mjs';
import { createCommandQueue } from '../src/controller/commandQueue.mjs';
import {
    DownloadError, artifactPaths, downloadArtifact, downloadSnapshotFile, inspectArtifact, removeArtifact, resolveHuggingFaceArtifact,
    snapshotPaths,
} from '../src/controller/downloader.mjs';
import { createRegistryService, REGISTRY_ID_RE } from '../src/controller/registries.mjs';
import { lookupInRegistry } from '../src/controller/registryLookup.mjs';
import { statIdentity } from '../src/controller/workspaceReuse.mjs';
import { createWeightStores } from '../src/controller/weightStores.mjs';
import { ggufBytes, modelPairs } from './gguf-fixture.mjs';
import { createTestTransport, fakeServer } from './net-fixture.mjs';
import { KEY, OTHER_KEY, codeOf, deferred, registryHarness, until } from './registry-fixture.mjs';

const REPO = 'acme/models';
const COMMIT = 'c'.repeat(40);
const HF_TOKEN = 'hf_home_token_value';
const MIRROR_TOKEN = 'mirror-credential-value';
const MIRROR_URL = 'https://mirror.test/p';
const SHARD_1 = 'm-Q4_K_M-00001-of-00002.gguf';
const SHARD_2 = 'm-Q4_K_M-00002-of-00002.gguf';

const sha256Of = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const gitOidOf = (bytes) => crypto.createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const pad = (header, size) => Buffer.concat([header, Buffer.alloc(Math.max(0, size - header.length), 0xee)]);
// A real GGUF header (24 layers, 14 heads, 2 KV heads) padded to a small file, so it can be looked up and downloaded.
const HEADER = pad(ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896 }) }).bytes, 65536);
const FILES = Object.freeze({
    'm-Q4_K_M.gguf': HEADER,
    [SHARD_1]: Buffer.alloc(40_000, 0x61),
    [SHARD_2]: Buffer.alloc(30_000, 0x62),
    'config.json': Buffer.from('{"num_hidden_layers":2,"hidden_size":64,"num_attention_heads":4,"max_position_embeddings":2048}\n'),
    'model.safetensors': Buffer.alloc(50_000, 0x63),
});
const lfsEntry = (name) => ({ type: 'file', path: name, size: FILES[name].length, oid: 'f'.repeat(40), lfs: { oid: sha256Of(FILES[name]), size: FILES[name].length, pointerSize: 134 } });
const gitEntry = (name) => ({ type: 'file', path: name, size: FILES[name].length, oid: gitOidOf(FILES[name]) });
const TREE = [lfsEntry('m-Q4_K_M.gguf'), lfsEntry(SHARD_1), lfsEntry(SHARD_2), gitEntry('config.json'), lfsEntry('model.safetensors')];

// ------------------------------------------------------------------ the fakes

const json = (res, status, body, headers = {}) => {
    const bytes = Buffer.from(JSON.stringify(body));
    res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': bytes.length, ...headers });
    res.end(bytes);
};
const bare = (res, status, headers = {}) => {
    res.writeHead(status, { 'Content-Length': 0, ...headers });
    res.end();
};

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
    if (from >= bytes.length) {
        bare(res, 416);
        return;
    }
    res.writeHead(206, { 'Content-Length': to - from + 1, 'Content-Range': `bytes ${from}-${to}/${bytes.length}` });
    res.end(bytes.subarray(from, to + 1));
}

// The repository's API and resolve URLs under `prefix`, answered from `world`.
const repoHandler = (world, prefix) => (req, res, url) => {
    const pathname = decodeURIComponent(url.pathname);
    if (!pathname.startsWith(`${prefix}/`)) {
        bare(res, 404);
        return;
    }
    const rest = pathname.slice(prefix.length);
    const api = `/api/models/${REPO}`;
    if (rest === api) {
        json(res, 200, world.info);
    } else if (rest.startsWith(`${api}/revision/`)) {
        const revision = rest.slice(`${api}/revision/`.length);
        if (revision === 'main' || revision === COMMIT) {
            if (world.revisionReply) world.revisionReply(res);
            else json(res, 200, { sha: COMMIT });
        } else if (world.canary === 'catchall') {
            json(res, 200, { sha: COMMIT });
        } else {
            bare(res, world.canary);
        }
    } else if (rest === `${api}/tree/${COMMIT}`) {
        const page = Number(url.searchParams.get('page') ?? 0);
        const link = world.link?.(page);
        json(res, 200, world.tree[page] ?? [], link ? { Link: link } : {});
    } else if (rest.startsWith(`/${REPO}/resolve/${COMMIT}/`)) {
        const file = rest.slice(`/${REPO}/resolve/${COMMIT}/`.length);
        if (!world.files[file]) bare(res, 404);
        else bare(res, 302, { Location: `https://cdn.test/blob/${encodeURIComponent(file)}` });
    } else {
        bare(res, 404);
    }
};

/**
 * One mirror (`mirror.test`, prefix /p), a fake huggingface.co (`hf.test`, no prefix) over the same repository, a CDN
 * (`cdn.test`) and a bystander (`other.test`), behind a transport that maps those https names onto loopback servers.
 * Its inner transport is the production guard with loopback allowed; the mapped transport does not follow http.
 */
async function createWorld(t, options = {}) {
    const world = {
        info: { id: REPO, gated: false, cardData: { license: 'apache-2.0' } },
        canary: 404,
        revisionReply: null,
        files: { ...FILES },
        tree: [TREE],
        link: null,
        cdnHook: null,
        ...options,
    };
    const start = async (handler) => {
        const server = await fakeServer(handler);
        t.after(() => server.close());
        return server;
    };
    world.mirror = await start(repoHandler(world, '/p'));
    world.hf = await start(repoHandler(world, ''));
    world.other = await start((req, res) => bare(res, 404));
    world.cdn = await start((req, res, url) => {
        const bytes = world.files[decodeURIComponent(url.pathname.replace('/blob/', ''))];
        if (!bytes) bare(res, 404);
        else if (!world.cdnHook?.(req, res, bytes, url)) serve(req, res, bytes);
    });
    const routes = { 'mirror.test': world.mirror.origin, 'hf.test': world.hf.origin, 'cdn.test': world.cdn.origin, 'other.test': world.other.origin };
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
    return world;
}

const authOf = (server) => server.requests.map((request) => request.headers.authorization);
const pathsOf = (server) => server.requests.map((request) => request.url.replace(/\?.*$/, ''));
const resolvesOf = (server) => server.requests.filter((request) => request.url.includes('/resolve/'));

// A directory removed after every deferred cleanup (a controller's drain writes into it first).
function scratch(t) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-mirror-'));
    const cleanups = [];
    t.after(async () => {
        for (const cleanup of cleanups.reverse()) await cleanup();
        fs.rmSync(dir, { recursive: true, force: true });
    });
    return { dir, later: (cleanup) => cleanups.push(cleanup) };
}

// A real controller (real downloader and store) whose model paths go through the world's transport.
function controllerOn(t, world, { disk = scratch(t), credentialsKey = KEY, env = {}, extra = {}, initialState = null, resolveHf = null } = {}) {
    const h = registryHarness(t, {
        dataDir: disk.dir,
        credentialsKey,
        env: { PATH: '/usr/bin', HF_TOKEN, ...env },
        initialState,
        resolveHf,
        extra: { registryTransport: world.transport, hfBaseUrl: 'https://hf.test', download: undefined, inspect: undefined, remove: undefined, ...extra },
    });
    disk.later(() => h.controller.drain());
    return h;
}

const MIRROR = Object.freeze({ id: 'mirror', kind: 'huggingface', url: MIRROR_URL });
async function addMirror(h, { token = MIRROR_TOKEN } = {}) {
    await h.controller.saveRegistry({ registry: MIRROR });
    if (token) await h.controller.setRegistryCredential({ id: 'mirror', token });
}

const ggufSource = (extra = {}) => ({ type: 'huggingface', registry: 'mirror', repo: REPO, file: 'm-Q4_K_M.gguf', revision: 'main', ...extra });
const pinnedGguf = (extra = {}) => ggufSource({ commit: COMMIT, size: HEADER.length, sha256: sha256Of(HEADER), ...extra });
const splitSource = (extra = {}) => ggufSource({
    file: SHARD_1,
    commit: COMMIT,
    size: FILES[SHARD_1].length + FILES[SHARD_2].length,
    shards: [SHARD_1, SHARD_2].map((file) => ({ file, size: FILES[file].length, sha256: sha256Of(FILES[file]) })),
    ...extra,
});
const snapshotSource = (extra = {}) => ({
    type: 'hf-snapshot',
    registry: 'mirror',
    repo: REPO,
    revision: 'main',
    commit: COMMIT,
    files: [
        { path: 'config.json', size: FILES['config.json'].length, gitOid: gitOidOf(FILES['config.json']) },
        { path: 'model.safetensors', size: FILES['model.safetensors'].length, sha256: sha256Of(FILES['model.safetensors']) },
    ],
    size: FILES['config.json'].length + FILES['model.safetensors'].length,
    ...extra,
});
const entryOf = (id, sources) => ({ id, displayName: id, contextLength: 4096, architecture: 'dense', memory: { layers: 4, kvBytesPerToken: 1000 }, sources });
const lookupOf = (h, args) => h.controller.lookupModel(args);

// A store over the real downloader and a real registry service, for the rules that need no controller.
function storeOn(t, world, disk = scratch(t), { credentialsKey = KEY, download = downloadArtifact } = {}) {
    const registries = createRegistryService({ dataDir: disk.dir, credentialsKey, hfBaseUrl: 'https://hf.test' });
    const ops = registries.operations({ queue: createCommandQueue(), references: () => ({ entries: [], active: null }) });
    const stores = createWeightStores({
        dataDir: disk.dir,
        env: { HF_TOKEN },
        hfBaseUrl: 'https://hf.test',
        inspect: inspectArtifact,
        download,
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

// ------------------------------------------------------------------ source shape

test('a source names a registry: huggingface is normalized away, any other id needs an explicit revision', () => {
    const base = { type: 'huggingface', repo: REPO, file: 'm-Q4_K_M.gguf', revision: 'main' };
    const plain = validateModel({ id: 'plain', sources: { gguf: base } });
    const named = validateModel({ id: 'plain', sources: { gguf: { ...base, registry: 'huggingface' } } });
    assert.deepEqual(named.sources.gguf, plain.sources.gguf);
    assert.equal(Object.hasOwn(named.sources.gguf, 'registry'), false);
    assert.equal(validateModel({ id: 'mm', sources: { gguf: { ...base, registry: 'mirror' } } }).sources.gguf.registry, 'mirror');
    assert.equal(validateModel({ id: 'mm', sources: { hf: { type: 'hf-snapshot', registry: 'mirror', repo: REPO, revision: 'main' } } }).sources.hf.registry, 'mirror');
    assert.equal(Object.hasOwn(validateModel({ id: 'mm', sources: { hf: { type: 'hf-snapshot', registry: 'huggingface', repo: REPO } } }).sources.hf, 'registry'), false);
    // Without a revision the default branch would be assumed; a mirror's is not known.
    const { revision, ...noRevision } = base;
    for (const [format, source] of [['gguf', { ...noRevision, registry: 'mirror' }], ['hf', { type: 'hf-snapshot', registry: 'mirror', repo: REPO }]]) {
        assert.throws(() => validateModel({ id: 'mm', sources: { [format]: source } }), { code: 'invalid_model', details: { field: `sources.${format}.revision` } });
    }
    assert.equal(revision, 'main');
    for (const registry of ['Hub', 'x', 'a'.repeat(33), 'a/b', '../x', 7, null, '']) {
        assert.throws(() => validateModel({ id: 'mm', sources: { gguf: { ...base, registry } } }), { code: 'invalid_model' }, String(registry));
    }
    // The published schema says the same.
    const schema = JSON.parse(fs.readFileSync(new URL('../catalog/schema.json', import.meta.url), 'utf8'));
    for (const name of ['huggingface', 'hfSnapshot']) {
        assert.equal(schema.$defs[name].properties.registry.pattern, REGISTRY_ID_RE.source);
        assert.deepEqual(schema.$defs[name].dependentRequired, { registry: ['revision'] });
    }
    assert.equal(schema.$id, 'local-llm.catalog/v4');
});

// ------------------------------------------------------------------ A17 paths and keys, A16 records

test('A17: a mirror\'s files and keys live under _r/<id>; huggingface.co keeps its own; delete weights touches only one', async (t) => {
    const { stores, dir } = storeOn(t, null);
    const gguf = path.join(dir, 'models', 'gguf');
    const hf = path.join(dir, 'models', 'hf');
    const plainGguf = pinnedGguf({ registry: undefined });
    const plainSplit = splitSource({ registry: undefined });
    // The store table.
    assert.equal(stores.huggingface.paths(plainGguf).file, path.join(gguf, 'acme', 'models', COMMIT, 'm-Q4_K_M.gguf'));
    assert.equal(stores.huggingface.paths(pinnedGguf()).file, path.join(gguf, '_r', 'mirror', 'acme', 'models', COMMIT, 'm-Q4_K_M.gguf'));
    assert.equal(stores.huggingface.paths(splitSource()).file, path.join(gguf, '_r', 'mirror', 'acme', 'models', COMMIT, SHARD_1));
    assert.equal(stores.huggingface.key(plainGguf), `gguf:${REPO}@${COMMIT}/m-Q4_K_M.gguf`);
    assert.equal(stores.huggingface.key(pinnedGguf()), `gguf:_r/mirror/${REPO}@${COMMIT}/m-Q4_K_M.gguf`);
    assert.equal(stores.huggingface.key(plainSplit), `gguf:${REPO}@${COMMIT}/${SHARD_1}#2`);
    assert.equal(stores.huggingface.key(splitSource()), `gguf:_r/mirror/${REPO}@${COMMIT}/${SHARD_1}#2`);
    assert.deepEqual(snapshotPaths({ root: hf, repo: REPO, commit: COMMIT }), {
        dir: path.join(hf, 'acme', 'models', COMMIT),
        stateDir: path.join(hf, '.state', 'acme', 'models', COMMIT),
    });
    assert.deepEqual(snapshotPaths({ root: hf, repo: REPO, commit: COMMIT, registry: 'mirror' }), {
        dir: path.join(hf, '_r', 'mirror', 'acme', 'models', COMMIT),
        stateDir: path.join(hf, '.state', '_r', 'mirror', 'acme', 'models', COMMIT),
    });
    assert.equal(stores['hf-snapshot'].key({ ...snapshotSource(), registry: undefined }), `hf:${REPO}@${COMMIT}`);
    assert.equal(stores['hf-snapshot'].key(snapshotSource()), `hf:_r/mirror/${REPO}@${COMMIT}`);
    // An id is joined into a path only in its validated form.
    for (const registry of ['..', 'a/b', 'Hub', '', 7]) {
        assert.throws(() => artifactPaths({ root: gguf, artifact: pinnedGguf({ registry }) }), { code: 'INVALID_SOURCE' }, String(registry));
        assert.throws(() => snapshotPaths({ root: hf, repo: REPO, commit: COMMIT, registry }), { code: 'INVALID_SOURCE' }, String(registry));
    }

    // Delete weights: the same repository and commit on both registries, byte-checked after one is deleted.
    const place = (target, bytes) => {
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, bytes);
    };
    const filesOf = (root) => {
        const found = new Map();
        const walk = (current) => {
            for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
                const full = path.join(current, entry.name);
                if (entry.isDirectory()) walk(full);
                else found.set(path.relative(dir, full), fs.readFileSync(full));
            }
        };
        if (fs.existsSync(root)) walk(root);
        return found;
    };
    place(stores.huggingface.paths(plainGguf).file, HEADER);
    place(stores.huggingface.paths(plainGguf).meta, '{"kept":true}');
    place(stores.huggingface.paths(pinnedGguf()).file, HEADER);
    place(stores.huggingface.paths(pinnedGguf()).meta, '{"mirror":true}');
    for (const registry of [undefined, 'mirror']) {
        const { dir: snapshotDir, stateDir } = snapshotPaths({ root: hf, repo: REPO, commit: COMMIT, registry });
        place(path.join(snapshotDir, 'config.json'), registry ? 'mirror config' : 'plain config');
        place(path.join(stateDir, 'config.json.json'), registry ? '{"mirror":true}' : '{"kept":true}');
    }
    const plainBefore = new Map([...filesOf(path.join(gguf, 'acme')), ...filesOf(path.join(hf, 'acme')), ...filesOf(path.join(hf, '.state', 'acme'))]);
    assert.equal(plainBefore.size, 4);
    assert.ok(await stores.huggingface.remove(pinnedGguf()) > 0, 'the mirror GGUF freed bytes');
    assert.ok(await stores['hf-snapshot'].remove(snapshotSource()) > 0, 'the mirror snapshot freed bytes');
    assert.equal(filesOf(path.join(gguf, '_r')).size, 0);
    assert.equal(filesOf(path.join(hf, '_r')).size, 0);
    assert.equal(filesOf(path.join(hf, '.state', '_r')).size, 0);
    const plainAfter = new Map([...filesOf(path.join(gguf, 'acme')), ...filesOf(path.join(hf, 'acme')), ...filesOf(path.join(hf, '.state', 'acme'))]);
    assert.deepEqual([...plainAfter.keys()].sort(), [...plainBefore.keys()].sort(), 'huggingface.co files remain');
    for (const [name, bytes] of plainBefore) assert.ok(plainAfter.get(name).equals(bytes), `${name} is byte-identical`);
    // Twin: deleting the huggingface.co GGUF takes only its own file.
    assert.ok(await stores.huggingface.remove(plainGguf) > 0);
    assert.equal(fs.existsSync(stores.huggingface.paths(plainGguf).file), false);
    assert.equal(fs.existsSync(path.join(hf, 'acme', 'models', COMMIT, 'config.json')), true);
});

test('A16: records written before registries existed are complete, with no request and no re-hash', async (t) => {
    const { stores, dir } = storeOn(t, null);
    const gguf = path.join(dir, 'models', 'gguf');
    const hf = path.join(dir, 'models', 'hf');
    const source = pinnedGguf({ registry: undefined });
    const paths = artifactPaths({ root: gguf, artifact: source });
    fs.mkdirSync(path.dirname(paths.file), { recursive: true });
    fs.writeFileSync(paths.file, HEADER);
    // The record as the base version wrote it: no `registry` key at all.
    const record = {
        repo: REPO, file: source.file, commit: COMMIT, size: HEADER.length, sha256: sha256Of(HEADER),
        verifiedAt: '2026-01-01T00:00:00.000Z',
        provenance: { file: source.file, source: `https://huggingface.co/${REPO}/resolve/${COMMIT}/${source.file}`, method: 'download', bytes: HEADER.length },
        stat: statIdentity(fs.statSync(paths.file)),
    };
    fs.writeFileSync(paths.meta, `${JSON.stringify(record, null, 2)}\n`);
    assert.deepEqual(await inspectArtifact({ root: gguf, artifact: source }), { state: 'complete', bytes: HEADER.length });
    assert.equal((await stores.huggingface.state(source)).state, 'complete');
    const refuse = async () => assert.fail('no request may be made for a current record');
    const kept = await downloadArtifact({ root: gguf, artifact: source, fetchImpl: refuse });
    assert.equal(kept.current, true, 'verified by its recorded stat, not hashed again');
    assert.equal(kept.bytesTransferred, 0);
    assert.deepEqual(JSON.parse(fs.readFileSync(paths.meta, 'utf8')).provenance, record.provenance);

    // A snapshot file.
    const file = snapshotSource().files[1];
    const artifact = { repo: REPO, commit: COMMIT, file: file.path, size: file.size, sha256: file.sha256 };
    const { dir: snapshotDir, stateDir } = snapshotPaths({ root: hf, repo: REPO, commit: COMMIT });
    fs.mkdirSync(snapshotDir, { recursive: true });
    fs.mkdirSync(stateDir, { recursive: true });
    fs.writeFileSync(path.join(snapshotDir, file.path), FILES[file.path]);
    fs.writeFileSync(path.join(stateDir, `${file.path}.json`), `${JSON.stringify({
        ...artifact,
        verifiedAt: '2026-01-01T00:00:00.000Z',
        provenance: { file: file.path, source: `https://huggingface.co/${REPO}`, method: 'download', bytes: file.size },
        stat: statIdentity(fs.statSync(path.join(snapshotDir, file.path))),
    }, null, 2)}\n`);
    const snapshot = { type: 'hf-snapshot', repo: REPO, revision: 'main', commit: COMMIT, files: [{ path: file.path, size: file.size, sha256: file.sha256 }], size: file.size };
    assert.equal((await stores['hf-snapshot'].state(snapshot)).state, 'complete');
    const again = await downloadSnapshotFile({ root: hf, artifact, fetchImpl: refuse });
    assert.equal(again.current, true);
    assert.equal(again.bytesTransferred, 0);
    // A mirror's identity is another one: the same bytes recorded without a registry are not its file.
    assert.equal((await stores.huggingface.state({ ...source, registry: 'mirror' })).state, 'absent');
});

// ------------------------------------------------------------------ lookup through a mirror

test('lookup through a mirror: a canary first, every request under the prefix, the credential to the mirror only (A7, A8)', async (t) => {
    const world = await createWorld(t);
    const h = controllerOn(t, world);
    await addMirror(h);
    const found = await lookupOf(h, { registry: 'mirror', repo: REPO, file: 'm-Q4_K_M.gguf' });
    assert.deepEqual(found.registry, { id: 'mirror', kind: 'huggingface', label: 'mirror' });
    assert.equal(found.commit, COMMIT);
    assert.equal(found.sizing.source, 'gguf-header');
    assert.deepEqual(found.files.map((row) => row.file), [SHARD_1, 'm-Q4_K_M.gguf']);
    const paths = pathsOf(world.mirror);
    assert.ok(paths.every((entry) => entry.startsWith('/p/')), paths.join(' '));
    const canary = paths.findIndex((entry) => /\/revision\/local-llm-canary-[0-9a-f]{16}$/.test(entry));
    assert.ok(canary >= 0 && paths.indexOf(`/p/api/models/${REPO}/revision/main`) > canary, 'the canary comes before the revision');
    // A7: the mirror gets the exact bearer on every request; the CDN it redirected the ranged read to gets none.
    assert.ok(authOf(world.mirror).every((value) => value === `Bearer ${MIRROR_TOKEN}`), authOf(world.mirror).join(' '));
    assert.ok(world.cdn.requests.length >= 1 && authOf(world.cdn).every((value) => value === undefined));
    // A8: huggingface.co's own lookup carries HF_TOKEN, and the mirror was never sent it.
    const before = world.mirror.requests.length;
    const home = await lookupOf(h, { repo: REPO, format: 'hf' });
    assert.deepEqual(home.registry, { id: 'huggingface', kind: 'huggingface', label: 'Hugging Face' });
    assert.ok(world.hf.requests.length >= 3);
    assert.ok(authOf(world.hf).every((value) => value === `Bearer ${HF_TOKEN}`), authOf(world.hf).join(' '));
    assert.equal(world.mirror.requests.length, before, 'huggingface.co lookups never touch the mirror');
    assert.ok(world.mirror.requests.every((request) => !JSON.stringify(request.headers).includes(HF_TOKEN)));
    assert.ok(world.cdn.requests.every((request) => !JSON.stringify(request.headers).includes(HF_TOKEN)));
    // The snapshot lookup reads config.json from the mirror.
    const snapshot = await lookupOf(h, { registry: 'mirror', repo: REPO, format: 'hf' });
    assert.equal(snapshot.sizing.source, 'config.json');
    // A provider names huggingface.co only.
    assert.equal(await codeOf(lookupOf(h, { registry: 'mirror', repo: REPO, provider: 'huggingface' })), 'invalid_request');
});

test('A10: without a readable key the mirror\'s home server sees no Authorization header', async (t) => {
    const world = await createWorld(t);
    const disk = scratch(t);
    const keyed = controllerOn(t, world, { disk });
    await addMirror(keyed);
    await lookupOf(keyed, { registry: 'mirror', repo: REPO, format: 'hf' });
    assert.ok(authOf(world.mirror).every((value) => value === `Bearer ${MIRROR_TOKEN}`), 'positive control: the stored credential is sent');
    await keyed.controller.drain();
    // The same data directory under another key: the credential cannot be read, and nothing is sent.
    world.mirror.requests.length = 0;
    const rotated = controllerOn(t, world, { disk, credentialsKey: OTHER_KEY });
    assert.equal((await rotated.controller.listRegistries()).registries.find((entry) => entry.id === 'mirror').credential, 'unreadable');
    await lookupOf(rotated, { registry: 'mirror', repo: REPO, format: 'hf' });
    assert.ok(world.mirror.requests.length >= 3);
    assert.ok(authOf(world.mirror).every((value) => value === undefined), authOf(world.mirror).join(' '));
    // No key at all: the same.
    world.mirror.requests.length = 0;
    const keyless = controllerOn(t, world, { disk, credentialsKey: null });
    await lookupOf(keyless, { registry: 'mirror', repo: REPO, format: 'hf' });
    assert.ok(world.mirror.requests.length >= 3);
    assert.ok(authOf(world.mirror).every((value) => value === undefined));
    // A pin goes the same way, with HF_TOKEN set in this controller's environment all along (A8).
    world.mirror.requests.length = 0;
    await keyless.controller.addModel(entryOf('keyless-pin', { gguf: ggufSource() }));
    assert.ok(world.mirror.requests.length >= 3);
    assert.ok(authOf(world.mirror).every((value) => value === undefined), authOf(world.mirror).join(' '));
});

test('a mirror that does not answer a revision that cannot exist with 404 is nonconforming, for the lookup and the pin', async (t) => {
    const catchAll = await createWorld(t, { canary: 'catchall' });
    const h = controllerOn(t, catchAll);
    await addMirror(h);
    await assert.rejects(lookupOf(h, { registry: 'mirror', repo: REPO }), (error) => error.code === 'lookup_failed' && error.details?.reason === 'nonconforming-registry');
    await assert.rejects(h.controller.addModel(entryOf('catch-all', { gguf: ggufSource() })),
        (error) => error instanceof DownloadError && error.details?.reason === 'nonconforming-registry');
    assert.equal(catchAll.mirror.requests.some((request) => request.url.endsWith('/revision/main')), false, 'the real revision was never asked');
    assert.equal(h.controller.state.registry.length, 0);
    // A commit needs no canary: it names itself.
    assert.equal((await lookupOf(h, { registry: 'mirror', repo: REPO, revision: COMMIT })).commit, COMMIT);
    // A gated repository answers the canary like any other: 404 passes, and the gating is reported.
    const gated = await createWorld(t, { info: { id: REPO, gated: 'manual' } });
    const g = controllerOn(t, gated);
    await addMirror(g);
    assert.equal((await lookupOf(g, { registry: 'mirror', repo: REPO })).gated, 'manual');
    // A refusal of the canary is the refusal the real request would get, said with the registry's own name and hint.
    for (const status of [401, 403]) {
        const refusing = await createWorld(t, { canary: status });
        const r = controllerOn(t, refusing);
        await addMirror(r, { token: null });
        await assert.rejects(lookupOf(r, { registry: 'mirror', repo: REPO }),
            (error) => error.code === 'access_denied' && /^mirror did not find/.test(error.message) && /local_llm_registry_credential_set/.test(error.message), String(status));
    }
    // Server errors are not "nonconforming": they are failures of the request.
    const broken = await createWorld(t, { canary: 503 });
    const b = controllerOn(t, broken);
    await addMirror(b);
    await assert.rejects(lookupOf(b, { registry: 'mirror', repo: REPO }), (error) => error.code === 'lookup_failed' && error.details?.status === 503);
});

test('A14: a root-relative next link is fetched under the prefix; a cross-origin one is listing-incomplete and is never requested', async (t) => {
    const second = [lfsEntry(SHARD_2), lfsEntry('model.safetensors')];
    const pages = (link) => ({ tree: [[lfsEntry('m-Q4_K_M.gguf'), lfsEntry(SHARD_1), gitEntry('config.json')], second], link: (page) => (page === 0 ? link : null) });
    const relative = await createWorld(t, pages(`</api/models/${REPO}/tree/${COMMIT}?page=1>; rel="next"`));
    const h = controllerOn(t, relative);
    await addMirror(h);
    const found = await lookupOf(h, { registry: 'mirror', repo: REPO });
    assert.deepEqual(found.files.map((row) => row.file), [SHARD_1, 'm-Q4_K_M.gguf'], 'the split set is complete only with its second page');
    assert.equal(pathsOf(relative.mirror).filter((entry) => entry === `/p/api/models/${REPO}/tree/${COMMIT}`).length, 2, 'page 2 under the prefix');
    const pinned = await h.controller.addModel(entryOf('paged-gguf', { gguf: ggufSource({ file: SHARD_1 }) }));
    assert.equal(pinned.model.sources.gguf.shards.length, 2);
    const snapshotPin = await h.controller.addModel(entryOf('paged-snapshot', { hf: { type: 'hf-snapshot', registry: 'mirror', repo: REPO, revision: 'main' } }));
    assert.deepEqual(snapshotPin.model.sources.hf.files.map((file) => file.path), ['config.json', 'model.safetensors']);
    assert.equal(relative.other.requests.length, 0);

    const crossOrigin = await createWorld(t, pages(`<https://other.test/api/models/${REPO}/tree/${COMMIT}?page=1>; rel="next"`));
    const c = controllerOn(t, crossOrigin);
    await addMirror(c);
    await assert.rejects(lookupOf(c, { registry: 'mirror', repo: REPO }), (error) => error.code === 'lookup_failed' && error.details?.reason === 'listing-incomplete');
    for (const entry of [entryOf('x-gguf', { gguf: ggufSource({ file: SHARD_1 }) }), entryOf('x-snapshot', { hf: { type: 'hf-snapshot', registry: 'mirror', repo: REPO, revision: 'main' } })]) {
        await assert.rejects(c.controller.addModel(entry), (error) => error.code === 'RESOLVE_FAILED' && error.details?.reason === 'listing-incomplete', entry.id);
    }
    assert.equal(crossOrigin.other.requests.length, 0, 'the other origin got no request');
    assert.equal(c.controller.state.registry.length, 0);
});

test('metadata errors from a mirror: a 307 without Location, a gzip answer and a redirect to another origin end the request', async (t) => {
    const noLocation = await createWorld(t, { revisionReply: (res) => bare(res, 307) });
    const h = controllerOn(t, noLocation);
    await addMirror(h);
    await assert.rejects(lookupOf(h, { registry: 'mirror', repo: REPO }), (error) => error.code === 'lookup_failed' && error.details?.reason === 'redirect');
    const gzip = await createWorld(t, {
        revisionReply: (res) => {
            const bytes = zlib.gzipSync(JSON.stringify({ sha: COMMIT }));
            res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Encoding': 'gzip', 'Content-Length': bytes.length });
            res.end(bytes);
        },
    });
    const g = controllerOn(t, gzip);
    await addMirror(g);
    await assert.rejects(lookupOf(g, { registry: 'mirror', repo: REPO }), (error) => error.code === 'lookup_failed' && error.details?.reason === 'encoding');
    await assert.rejects(g.controller.addModel(entryOf('gzipped', { gguf: ggufSource() })),
        (error) => error instanceof DownloadError && error.details?.reason === 'encoding');
    // A redirect from the mirror to another origin is not followed for metadata, and nothing is sent there.
    const away = await createWorld(t, { revisionReply: (res) => bare(res, 302, { Location: 'https://other.test/elsewhere' }) });
    const a = controllerOn(t, away);
    await addMirror(a);
    await assert.rejects(lookupOf(a, { registry: 'mirror', repo: REPO }), (error) => error.code === 'lookup_failed' && error.details?.reason === 'redirect');
    assert.equal(away.other.requests.length, 0);
});

// ------------------------------------------------------------------ pins

test('Add pins through a mirror: GGUF, split GGUF and snapshot carry the registry; the credential is only ever on the wire', async (t) => {
    const world = await createWorld(t);
    const disk = scratch(t);
    const h = controllerOn(t, world, { disk });
    await addMirror(h);
    const gguf = (await h.controller.addModel(entryOf('mirror-gguf', { gguf: ggufSource() }))).model.sources.gguf;
    assert.deepEqual({ ...gguf }, pinnedGguf());
    const split = (await h.controller.addModel(entryOf('mirror-split', { gguf: ggufSource({ file: SHARD_1 }) }))).model.sources.gguf;
    assert.deepEqual({ ...split }, splitSource());
    const snapshot = (await h.controller.addModel(entryOf('mirror-snapshot', { hf: { type: 'hf-snapshot', registry: 'mirror', repo: REPO, revision: 'main' } }))).model.sources.hf;
    assert.deepEqual(JSON.parse(JSON.stringify(snapshot)), JSON.parse(JSON.stringify(snapshotSource())));
    assert.ok(world.mirror.requests.length >= 6);
    assert.ok(authOf(world.mirror).every((value) => value === `Bearer ${MIRROR_TOKEN}`));
    assert.equal(world.cdn.requests.length, 0, 'a pin downloads nothing');
    assert.equal(world.hf.requests.length, 0);
    const stored = fs.readFileSync(path.join(disk.dir, 'state', 'controller.json'), 'utf8');
    assert.equal(stored.includes(MIRROR_TOKEN), false, 'no credential in the model entries');
    assert.ok(JSON.parse(stored).registry.some((entry) => entry.sources.gguf?.registry === 'mirror'), 'the entries name the registry');
    // A8: the same Add against huggingface.co sends HF_TOKEN there and nothing to the mirror.
    const before = world.mirror.requests.length;
    const home = (await h.controller.addModel(entryOf('home-gguf', { gguf: { type: 'huggingface', repo: REPO, file: 'm-Q4_K_M.gguf', revision: 'main' } }))).model.sources.gguf;
    assert.equal(Object.hasOwn(home, 'registry'), false);
    assert.ok(world.hf.requests.length >= 2 && authOf(world.hf).every((value) => value === `Bearer ${HF_TOKEN}`));
    assert.equal(world.mirror.requests.length, before);
    // A revision is required for a named registry.
    await assert.rejects(h.controller.addModel(entryOf('no-revision', { gguf: { type: 'huggingface', registry: 'mirror', repo: REPO, file: 'm-Q4_K_M.gguf' } })), { code: 'invalid_model' });
});

// ------------------------------------------------------------------ downloads

test('Run downloads a mirror GGUF into _r/<id>: the mirror sees the bearer, the CDN it redirects to sees none (A7)', async (t) => {
    const world = await createWorld(t);
    const disk = scratch(t);
    const h = controllerOn(t, world, { disk });
    await addMirror(h);
    await h.controller.addModel(entryOf('mirror-run', { gguf: ggufSource() }));
    await h.controller.run({ modelId: 'mirror-run', runnerId: 'llama.cpp', requestId: 'request-mirror-run' });
    await until(() => ['ready', 'error'].includes(h.controller.state.deployment?.phase), 5000);
    assert.equal(h.controller.state.deployment.phase, 'ready', h.controller.state.deployment.error);
    const file = path.join(disk.dir, 'models', 'gguf', '_r', 'mirror', 'acme', 'models', COMMIT, 'm-Q4_K_M.gguf');
    assert.ok(fs.readFileSync(file).equals(HEADER));
    assert.equal(fs.existsSync(path.join(disk.dir, 'models', 'gguf', 'acme')), false, 'huggingface.co\'s tree is untouched');
    const resolves = resolvesOf(world.mirror);
    assert.equal(resolves.length, 1);
    assert.equal(resolves[0].headers.authorization, `Bearer ${MIRROR_TOKEN}`);
    assert.ok(world.cdn.requests.length >= 1 && authOf(world.cdn).every((value) => value === undefined));
    // Provenance names the registry URL, never the CDN or a token.
    const [entry] = h.controller.state.deployment.provenance;
    assert.equal(entry.source, `${MIRROR_URL}/${REPO}/resolve/${COMMIT}/m-Q4_K_M.gguf`);
    assert.ok(!JSON.stringify(h.controller.state.deployment.provenance).includes(MIRROR_TOKEN));
    assert.equal(world.hf.requests.length, 0);
    await h.controller.stop();
});

test('a snapshot downloads from a mirror: every file from its registry URL under _r/<id>, the CDN without the bearer (A7)', async (t) => {
    const world = await createWorld(t);
    const { ops, stores, dir } = storeOn(t, world);
    await ops.saveRegistry({ registry: MIRROR });
    await ops.setRegistryCredential({ id: 'mirror', token: MIRROR_TOKEN });
    const result = await stores['hf-snapshot'].fetch({ artifact: snapshotSource(), signal: undefined });
    const expected = snapshotPaths({ root: path.join(dir, 'models', 'hf'), repo: REPO, commit: COMMIT, registry: 'mirror' });
    assert.equal(result.path, expected.dir);
    for (const name of ['config.json', 'model.safetensors']) assert.ok(fs.readFileSync(path.join(expected.dir, name)).equals(FILES[name]), name);
    assert.ok(fs.existsSync(path.join(expected.stateDir, 'config.json.json')));
    assert.deepEqual(resolvesOf(world.mirror).map((request) => request.url), [`/p/${REPO}/resolve/${COMMIT}/config.json`, `/p/${REPO}/resolve/${COMMIT}/model.safetensors`]);
    assert.ok(resolvesOf(world.mirror).every((request) => request.headers.authorization === `Bearer ${MIRROR_TOKEN}`));
    assert.ok(world.cdn.requests.length === 2 && authOf(world.cdn).every((value) => value === undefined));
    assert.deepEqual(result.provenance.map((entry) => entry.source), [
        `${MIRROR_URL}/${REPO}/resolve/${COMMIT}/config.json`, `${MIRROR_URL}/${REPO}/resolve/${COMMIT}/model.safetensors`,
    ]);
    assert.equal((await stores['hf-snapshot'].state(snapshotSource())).state, 'complete');
    assert.equal((await stores['hf-snapshot'].state({ ...snapshotSource(), registry: undefined })).state, 'absent');
});

test('a CDN that answers 403 after a dropped connection ends the download; the retry goes back through the mirror, and the CDN never sees the bearer', async (t) => {
    let cdnRequests = 0;
    const world = await createWorld(t, {
        cdnHook: (req, res, bytes) => {
            cdnRequests += 1;
            if (cdnRequests === 1) {
                res.writeHead(200, { 'Content-Length': bytes.length });
                res.write(bytes.subarray(0, 10_000));
                setTimeout(() => res.destroy(), 20);
                return true;
            }
            bare(res, 403);
            return true;
        },
    });
    const { ops, stores, dir } = storeOn(t, world, scratch(t), { download: (options) => downloadArtifact({ ...options, maxAttempts: 2, sleep: async () => {} }) });
    await ops.saveRegistry({ registry: MIRROR });
    await ops.setRegistryCredential({ id: 'mirror', token: MIRROR_TOKEN });
    await assert.rejects(stores.huggingface.fetch({ artifact: pinnedGguf(), signal: undefined }),
        (error) => error instanceof DownloadError && error.code === 'NETWORK' && error.details.lastReason === 'expired-redirect' && error.details.lastStatus === 403);
    assert.equal(cdnRequests, 2);
    assert.equal(resolvesOf(world.mirror).length, 2, 'each attempt started again at the mirror');
    assert.ok(resolvesOf(world.mirror).every((request) => request.headers.authorization === `Bearer ${MIRROR_TOKEN}`));
    assert.ok(authOf(world.cdn).every((value) => value === undefined));
    const paths = artifactPaths({ root: path.join(dir, 'models', 'gguf'), artifact: pinnedGguf() });
    assert.equal(fs.existsSync(paths.file), false, 'no complete file');
    assert.ok(fs.statSync(paths.partial).size <= 10_000, 'the partial is kept');
});

// ------------------------------------------------------------------ A26, A28

test('A26: a credential cleared and a registry disabled during a running download do not change that job\'s requests; the next lookup sees both', async (t) => {
    const hold = deferred();
    const arrived = deferred();
    const world = await createWorld(t, {
        cdnHook: (req, res, bytes, url) => {
            if (!url.pathname.endsWith(encodeURIComponent(SHARD_1))) return false;
            arrived.resolve();
            hold.promise.then(() => serve(req, res, bytes));
            return true;
        },
    });
    const { registries, ops, stores } = storeOn(t, world);
    await ops.saveRegistry({ registry: MIRROR });
    await ops.setRegistryCredential({ id: 'mirror', token: MIRROR_TOKEN });
    const job = stores.huggingface.fetch({ artifact: splitSource(), signal: undefined });
    await arrived.promise;
    // The job is inside its first shard. The credential goes, and so does the registry.
    await ops.clearRegistryCredential({ id: 'mirror' });
    await ops.saveRegistry({ registry: { id: 'mirror', enabled: false } });
    hold.resolve();
    const result = await job;
    assert.equal(result.files.length, 2);
    assert.deepEqual(resolvesOf(world.mirror).map((request) => request.url), [`/p/${REPO}/resolve/${COMMIT}/${SHARD_1}`, `/p/${REPO}/resolve/${COMMIT}/${SHARD_2}`]);
    assert.ok(resolvesOf(world.mirror).every((request) => request.headers.authorization === `Bearer ${MIRROR_TOKEN}`), 'the second shard still carries the snapshot\'s credential');
    // The next operation sees the change.
    const options = { fetchImpl: world.transport, registries };
    await assert.rejects(lookupInRegistry({ registry: 'mirror', repo: REPO }, options), { code: 'registry_disabled' });
    await assert.rejects(stores.huggingface.pin({ ...ggufSource() }), { code: 'registry_disabled' });
    await ops.saveRegistry({ registry: { id: 'mirror', enabled: true } });
    world.mirror.requests.length = 0;
    await lookupInRegistry({ registry: 'mirror', repo: REPO, format: 'hf' }, options);
    assert.ok(world.mirror.requests.length >= 3 && authOf(world.mirror).every((value) => value === undefined), 'the cleared credential is not sent');
});

test('A26: a registry removed and re-created with another url or kind during an Add or Update pin gives registry_changed (the real re-check)', async (t) => {
    const world = await createWorld(t);
    let gate = deferred();
    let started = deferred();
    const resolveHf = async () => {
        started.resolve();
        await gate.promise;
        return { commit: COMMIT, size: HEADER.length, sha256: sha256Of(HEADER) };
    };
    const h = controllerOn(t, world, { resolveHf });
    const redefine = async (id, changed) => {
        await h.controller.removeRegistry({ id });
        await h.controller.saveRegistry({ registry: { ...MIRROR, id, ...changed } });
    };
    const during = async (operation, id, changed) => {
        gate = deferred();
        started = deferred();
        const running = operation();
        await started.promise;
        await redefine(id, changed);
        gate.resolve();
        return codeOf(running);
    };
    await h.controller.saveRegistry({ registry: MIRROR });
    const adding = (id) => () => h.controller.addModel(entryOf(id, { gguf: ggufSource() }));
    assert.equal(await during(adding('url-moved'), 'mirror', { url: 'https://elsewhere.test/p' }), 'registry_changed');
    assert.equal(await during(adding('kind-changed'), 'mirror', { url: 'https://mirror.test', kind: 'modelscope' }), 'registry_changed');
    assert.equal(h.controller.state.registry.length, 0, 'state.registry unchanged');
    // Twin: re-created identically, the Add is saved.
    await h.controller.removeRegistry({ id: 'mirror' });
    await h.controller.saveRegistry({ registry: MIRROR });
    assert.equal(await during(adding('same-again'), 'mirror', {}), 'resolved');
    assert.equal(h.controller.state.registry.length, 1);
    // Update: the stored entry uses `mirror`; the update moves it to a second registry that changes during the pin.
    await h.controller.saveRegistry({ registry: { ...MIRROR, id: 'mirror-two' } });
    const before = JSON.stringify(h.controller.state.registry);
    const update = () => h.controller.updateModel(entryOf('same-again', { gguf: ggufSource({ registry: 'mirror-two', revision: 'other' }) }));
    assert.equal(await during(update, 'mirror-two', { url: 'https://elsewhere.test/p' }), 'registry_changed');
    assert.equal(JSON.stringify(h.controller.state.registry), before, 'state.registry unchanged');
    assert.equal(await codeOf((async () => {
        gate = deferred();
        started = deferred();
        const running = update();
        await started.promise;
        await h.controller.saveRegistry({ registry: { id: 'mirror-two', enabled: false } });
        gate.resolve();
        return running;
    })()), 'registry_disabled');
});

test('A28: a stored mirror entry whose registry is missing or disabled stays in the overview; Delete weights and Remove work; lookup, Add, Update and Run are refused', async (t) => {
    const world = await createWorld(t);
    const disk = scratch(t);
    const stored = (id, registry) => entryOf(id, { gguf: pinnedGguf({ registry }) });
    const h = controllerOn(t, world, {
        disk,
        initialState: { version: 1, deployment: null, params: {}, requests: {}, registry: [stored('orphan-gone', 'gone'), stored('orphan-off', 'mirror')] },
    });
    await h.controller.saveRegistry({ registry: MIRROR });
    await h.controller.saveRegistry({ registry: { id: 'mirror', enabled: false } });
    // Each entry's weights are on disk, recorded under its own registry.
    const root = path.join(disk.dir, 'models', 'gguf');
    for (const registry of ['gone', 'mirror']) {
        const paths = artifactPaths({ root, artifact: pinnedGguf({ registry }) });
        fs.mkdirSync(path.dirname(paths.file), { recursive: true });
        fs.writeFileSync(paths.file, HEADER);
        fs.writeFileSync(paths.meta, JSON.stringify({
            registry, repo: REPO, file: 'm-Q4_K_M.gguf', commit: COMMIT, size: HEADER.length, sha256: sha256Of(HEADER),
            verifiedAt: '2026-01-01T00:00:00.000Z', provenance: { file: 'm-Q4_K_M.gguf', source: MIRROR_URL, method: 'download', bytes: 1 },
            stat: statIdentity(fs.statSync(paths.file)),
        }));
    }
    const overview = await h.controller.overview();
    for (const id of ['orphan-gone', 'orphan-off']) {
        const model = overview.models.find((entry) => entry.id === id);
        assert.ok(model, `${id} is listed`);
        assert.equal(model.weights.gguf.download.state, 'complete', `${id}: its weights are seen`);
    }
    // The orphan probe, and the same four operations for each case.
    assert.equal(await codeOf(lookupOf(h, { registry: 'nope', repo: REPO })), 'registry_unknown');
    for (const [id, registry, code] of [['orphan-gone', 'gone', 'registry_unknown'], ['orphan-off', 'mirror', 'registry_disabled']]) {
        assert.equal(await codeOf(lookupOf(h, { registry, repo: REPO })), code, `${id}: lookup`);
        assert.equal(await codeOf(h.controller.addModel(entryOf(`new-${id}`, { gguf: ggufSource({ registry }) }))), code, `${id}: Add`);
        assert.equal(await codeOf(h.controller.updateModel(stored(id, registry))), code, `${id}: Update`);
        assert.equal(await codeOf(h.controller.run({ modelId: id, runnerId: 'llama.cpp', requestId: `request-${id}` })), code, `${id}: Run`);
    }
    assert.equal(world.mirror.requests.length, 0, 'nothing reached the mirror');
    assert.deepEqual((await h.controller.listRegistries()).registries.find((entry) => entry.id === 'mirror').usedBy, ['orphan-off']);
    // Weights and entries can still be removed.
    for (const id of ['orphan-gone', 'orphan-off']) {
        const freed = (await h.controller.deleteWeights({ modelId: id, runnerId: 'llama.cpp' })).freedBytes;
        assert.ok(freed >= HEADER.length, `${id}: freed ${freed}`);
        assert.deepEqual(await h.controller.removeModel({ modelId: id }), { removed: id });
    }
    assert.equal(fs.existsSync(path.join(root, '_r', 'mirror', 'acme')), false, 'the mirror\'s files are gone');
    assert.equal(h.controller.state.registry.length, 0);
});
