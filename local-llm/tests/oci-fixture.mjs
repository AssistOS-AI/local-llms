// Test seams for OCI registries (DS002): a fake registry (`reg.test`), its token realm (`auth.test`), the CDN its blobs
// redirect to (`cdn.test`), a bystander (`other.test`) and a fake huggingface.co (`hf.test`), all loopback servers
// behind a transport that maps their https names onto them; `localhost` goes to the production guard. Manifests and
// configs are built from bytes, so every digest is real. Nothing here is used by the agent itself.
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { createGuardedTransport } from '../src/controller/netGuard.mjs';
import { ggufBytes, GGUF_TYPE, modelPairs } from './gguf-fixture.mjs';
import { createTestTransport, fakeServer } from './net-fixture.mjs';
import { registryHarness } from './registry-fixture.mjs';

export const REPO = 'ai/smollm2';
export const TAG = '135m';
export const REG_URL = 'https://reg.test';
export const AUTH_URL = 'https://auth.test';
export const USERNAME = 'robot-user';
export const SECRET = 'oci-credential-secret-value';
export const BASIC = `Basic ${Buffer.from(`${USERNAME}:${SECRET}`).toString('base64')}`;
export const CDN_SIGNATURE = 'cdn-signature-secret';
export const MEDIA = Object.freeze({
    manifest: 'application/vnd.oci.image.manifest.v1+json',
    index: 'application/vnd.oci.image.index.v1+json',
    cncf: 'application/vnd.cncf.model.manifest.v1+json',
    cncfConfig: 'application/vnd.cncf.model.config.v1+json',
    legacyConfig: 'application/vnd.docker.ai.model.config.v0.1+json',
    legacyGguf: 'application/vnd.docker.ai.gguf.v3',
    weight: 'application/vnd.cncf.model.weight.v1.raw',
    weightConfig: 'application/vnd.cncf.model.weight.config.v1.raw',
    doc: 'application/vnd.cncf.model.doc.v1.raw',
});

export const sha256Of = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
export const digestOf = (bytes) => `sha256:${sha256Of(bytes)}`;
export const json = (value) => Buffer.from(JSON.stringify(value));
const pad = (header, size) => Buffer.concat([header, Buffer.alloc(Math.max(0, size - header.length), 0xee)]);
// A real GGUF header (24 layers) padded to a small file; a split shard's header also says which shard it is.
export const GGUF = pad(ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896 }) }).bytes, 40_000);
export const shardBytes = (no, count, size = 20_000) => pad(ggufBytes({ kv: modelPairs({ extra: [
    ['split.no', GGUF_TYPE.u16, no], ['split.count', GGUF_TYPE.u16, count], ['split.tensors.count', GGUF_TYPE.i32, 10],
] }) }).bytes, size + no);
export const CONFIG_JSON = Buffer.from('{"num_hidden_layers":2,"hidden_size":64,"num_attention_heads":4,"max_position_embeddings":2048}\n');
export const SAFETENSORS = Buffer.alloc(30_000, 0x63);

/** A layer descriptor for `bytes`, with an `org.cncf.model.filepath` when `filepath` is given. */
export const layer = (mediaType, bytes, filepath = undefined) => ({
    mediaType, digest: digestOf(bytes), size: bytes.length, ...(filepath === undefined ? {} : { annotations: { 'org.cncf.model.filepath': filepath } }),
});

/** A manifest and the blobs it names: `files` are `[mediaType, bytes, filepath?]`; `config` is the config's bytes. */
export function image({ files, config = json({ config: { format: 'gguf' }, descriptor: { licenses: ['Apache-2.0'] } }), layout = 'cncf', extra = {} }) {
    const blobs = new Map([[digestOf(config), config]]);
    const layers = files.map(([mediaType, bytes, filepath]) => {
        blobs.set(digestOf(bytes), bytes);
        return layer(mediaType, bytes, filepath);
    });
    const manifest = {
        schemaVersion: 2,
        mediaType: MEDIA.manifest,
        ...(layout === 'cncf' ? { artifactType: MEDIA.cncf } : {}),
        config: layer(layout === 'cncf' ? MEDIA.cncfConfig : MEDIA.legacyConfig, config),
        layers,
        ...extra,
    };
    return { manifest, bytes: json(manifest), blobs };
}

export const cncfGguf = () => image({ files: [[MEDIA.weight, GGUF, 'SmolLM2-135M-Instruct-Q4_K_M.gguf']] });
export const legacyGguf = () => image({ layout: 'docker', config: json({ config: { format: 'gguf' } }), files: [[MEDIA.legacyGguf, GGUF], ['application/vnd.docker.ai.license', Buffer.from('license')]] });
export const legacySplit = (shards = [shardBytes(0, 2), shardBytes(1, 2)]) => image({ layout: 'docker', config: json({ config: { format: 'gguf' } }), files: shards.map((bytes) => [MEDIA.legacyGguf, bytes]) });
export const cncfSnapshot = (names = ['config.json', 'model.safetensors', 'README.md']) => image({
    config: json({ config: { format: 'safetensors' } }),
    files: names.map((name) => [name.endsWith('.md') ? MEDIA.doc : name.endsWith('.safetensors') || name.endsWith('.bin') ? MEDIA.weight : MEDIA.weightConfig,
        name === 'config.json' ? CONFIG_JSON : name === 'model.safetensors' ? SAFETENSORS : Buffer.from(`file ${name}\n`), name]),
});

const bare = (res, status, headers = {}) => {
    res.writeHead(status, { 'Content-Length': 0, ...headers });
    res.end();
};

// Bytes with a file host's range semantics.
export function serve(req, res, bytes) {
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

/**
 * The fake registry and its neighbours. `world.publish(tag, image)` serves a manifest by tag and by digest;
 * `world.auth` is `'bearer'` (a token from auth.test is required), `'basic'` or `null`; `world.challenge` is the
 * `WWW-Authenticate` value; hooks may answer a request first (return true).
 */
export async function createOciWorld(t, options = {}) {
    const world = {
        repo: REPO,
        manifests: new Map(),
        blobs: new Map(),
        auth: 'bearer',
        challenge: `Bearer realm="${AUTH_URL}/token",service="reg.test"`,
        tokens: new Set(),
        issued: 0,
        expiresIn: 300,
        digestHeader: true,
        gate: null,
        tokenGate: null,
        registryHook: null,
        tokenHook: null,
        cdnHook: null,
        ...options,
    };
    world.publish = (tag, made) => {
        world.manifests.set(tag, made.bytes);
        world.manifests.set(digestOf(made.bytes), made.bytes);
        for (const [digest, bytes] of made.blobs) world.blobs.set(digest, bytes);
        return digestOf(made.bytes);
    };
    const start = async (handler) => {
        const server = await fakeServer(handler);
        t.after(() => server.close());
        return server;
    };
    world.reg = await start(async (req, res, url) => {
        if (world.gate) await world.gate.promise;
        if (world.registryHook?.(req, res, url)) return;
        const match = /^\/v2\/(.+)\/(manifests|blobs)\/([^/]+)$/.exec(url.pathname);
        if (!match) return bare(res, 404);
        const authorization = req.headers.authorization ?? '';
        const allowed = world.auth === null || (world.auth === 'bearer' && world.tokens.has(authorization.replace(/^Bearer /, '')))
            || (world.auth === 'basic' && authorization === BASIC);
        if (!allowed) return bare(res, 401, { 'WWW-Authenticate': world.challenge });
        const [, repo, kind, ref] = match;
        if (world.repo !== null && repo !== world.repo) return bare(res, 404);
        if (kind === 'manifests') {
            const bytes = world.manifests.get(ref);
            if (!bytes) return bare(res, 404);
            const headers = { 'Content-Type': MEDIA.manifest, 'Content-Length': bytes.length,
                ...(world.digestHeader ? { 'Docker-Content-Digest': typeof world.digestHeader === 'string' ? world.digestHeader : digestOf(bytes) } : {}) };
            res.writeHead(200, headers);
            res.end(req.method === 'HEAD' ? undefined : bytes);
            return;
        }
        if (!world.blobs.has(ref)) return bare(res, 404);
        bare(res, 307, { Location: `https://cdn.test/blobs/${ref}?sig=${CDN_SIGNATURE}` });
    });
    world.auth_ = await start(async (req, res, url) => {
        if (world.tokenGate) await world.tokenGate.promise;
        if (world.tokenHook?.(req, res, url)) return;
        if (url.pathname !== '/token') return bare(res, 404);
        world.issued += 1;
        const token = `issued-token-${world.issued}`;
        world.tokens.add(token);
        const body = json({ token, access_token: token, expires_in: world.expiresIn, issued_at: new Date().toISOString() });
        res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': body.length });
        res.end(body);
    });
    world.cdn = await start((req, res, url) => {
        const bytes = (world.cdnBytes ?? world.blobs).get(decodeURIComponent(url.pathname.replace('/blobs/', '')));
        if (!bytes) return bare(res, 404);
        if (!world.cdnHook?.(req, res, bytes)) serve(req, res, bytes);
    });
    world.other = await start((req, res) => bare(res, 404));
    world.hf = await start((req, res) => bare(res, 404));
    const routes = { 'reg.test': world.reg.origin, 'auth.test': world.auth_.origin, 'cdn.test': world.cdn.origin, 'other.test': world.other.origin, 'hf.test': world.hf.origin };
    const inner = createTestTransport();
    const production = createGuardedTransport();
    t.after(() => { inner.close(); production.close(); });
    world.calls = [];
    world.transport = Object.assign(async (url, init) => {
        const target = new URL(url);
        world.calls.push({ url: target.href, authorization: Object.entries(init?.headers ?? {}).find(([name]) => name.toLowerCase() === 'authorization')?.[1] });
        if (target.hostname === 'localhost') return production(url, init);
        const real = routes[target.host];
        if (!real) throw new Error(`unrouted host ${target.host}`);
        const to = new URL(real);
        target.protocol = to.protocol;
        target.host = to.host;
        return inner(target.href, init);
    }, { allowHttp: false });
    return world;
}

export const authOf = (server) => server.requests.map((request) => request.headers.authorization);
export const blobRequests = (world) => [...world.reg.requests, ...world.cdn.requests].filter((request) => request.url.includes('/blobs/'));

/** A registry snapshot as registries.mjs takes it: the credential is a non-enumerable property. */
export function snapshotOf({ url = REG_URL, authOrigins = [AUTH_URL], credential = { token: SECRET, username: USERNAME }, id = 'reg' } = {}) {
    const taken = { id, kind: 'oci', url, label: id, authOrigins };
    Object.defineProperty(taken, 'credential', { value: credential, enumerable: false });
    return Object.freeze(taken);
}

export function scratch(t, prefix = 'local-llm-oci-') {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
    const cleanups = [];
    t.after(async () => {
        for (const cleanup of cleanups.reverse()) await cleanup();
        fs.rmSync(dir, { recursive: true, force: true });
    });
    return { dir, later: (cleanup) => cleanups.push(cleanup) };
}

/** A real controller (real downloader and stores) whose model paths go through the world's transport. */
export function ociController(t, world, { disk = scratch(t), extra = {} } = {}) {
    const h = registryHarness(t, {
        dataDir: disk.dir,
        env: { PATH: '/usr/bin', HF_TOKEN: 'hf_home_token_value' },
        extra: { registryTransport: world.transport, hfBaseUrl: 'https://hf.test', download: undefined, inspect: undefined, remove: undefined, ...extra },
    });
    disk.later(() => h.controller.drain());
    return { ...h, disk };
}

export const REG = Object.freeze({ id: 'reg', kind: 'oci', url: REG_URL, authOrigins: [AUTH_URL] });
export async function addReg(h, { registry = REG, credential = true } = {}) {
    await h.controller.saveRegistry({ registry });
    if (credential) await h.controller.setRegistryCredential({ id: registry.id, token: SECRET, tokenUsername: USERNAME });
}
export const entryOf = (id, sources, extra = {}) => ({ id, displayName: id, contextLength: 4096, architecture: 'dense', memory: { layers: 4, kvBytesPerToken: 1000 }, sources, ...extra });
export const ociSource = (extra = {}) => ({ type: 'oci', registry: 'reg', repository: REPO, tag: TAG, ...extra });
export const ociSnapshotSource = (extra = {}) => ({ type: 'oci-snapshot', registry: 'reg', repository: REPO, tag: TAG, ...extra });
