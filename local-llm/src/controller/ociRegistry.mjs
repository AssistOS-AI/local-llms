// OCI distribution registries (DS002): Docker Hub, GHCR or any registry serving GGUF (the legacy Docker dialect or
// CNCF ModelPack) or CNCF per-file safetensors. A session is one registry snapshot and one repository. Each request
// goes to the registry origin, anonymously or with a cached bearer; a 401 is answered once: a Bearer challenge's
// realm must be https and at the registry origin or one of the registry's `authOrigins`, and the stored credential is
// sent there as Basic only when that origin is in `authOrigins`; the bearer it returns goes to the registry origin
// only. A Basic challenge gets the credential only when the registry origin itself is in `authOrigins`. Redirects are
// followed by hand and every hop off the registry origin (a blob's CDN) drops Authorization.
//
// A manifest is read by digest and must hash to it; it is classified before any blob is read, so a refusal reads
// none; a CNCF config is read by digest and checked too. Nothing here defaults `fetchImpl`.

import crypto from 'node:crypto';

import { LocalLlmError } from '../errors.mjs';
import { isSnapshotFile, splitGgufFiles, splitGgufName } from './catalog.mjs';
import { DownloadError, METADATA_TIMEOUT_MS } from './downloader.mjs';
import { createGgufHeaderReader, ggufSizing } from './ggufHeader.mjs';
import { LOOKUP_TIMEOUT_MS, REGISTRY_TOKEN_HINT, licenseOf, lookupError, quantizationOf, sizingFromConfig } from './modelLookup.mjs';
import { followRedirects } from './netGuard.mjs';
import { LEGACY_SINGLE_GGUF, OCI_DIGEST_RE, isOciGgufFile, sanitizeOciFilepath } from './ociSource.mjs';
import { normalizeLookupInput } from './registryLookup.mjs';

const MIB = 1024 * 1024;
export const MANIFEST_MAX_BYTES = MIB;
export const CONFIG_MAX_BYTES = MIB;
export const TOKEN_RESPONSE_MAX_BYTES = 64 * 1024;
export const TOKEN_MAX = 16 * 1024;
export const MAX_GGUF_LAYERS = 64;
export const MAX_LAYERS = 1024;
const HEADER_MAX_BYTES = 32 * MIB;
const MAX_SNAPSHOT_FILES = 200;
const MEDIA = Object.freeze({
    ociManifest: 'application/vnd.oci.image.manifest.v1+json',
    ociIndex: 'application/vnd.oci.image.index.v1+json',
    dockerManifest: 'application/vnd.docker.distribution.manifest.v2+json',
    dockerList: 'application/vnd.docker.distribution.manifest.list.v2+json',
    cncfArtifact: 'application/vnd.cncf.model.manifest.v1+json',
    legacyConfig: 'application/vnd.docker.ai.model.config.v0.1+json',
    cncfConfig: 'application/vnd.cncf.model.config.v1+json',
    legacyGguf: 'application/vnd.docker.ai.gguf.v3',
    weight: 'application/vnd.cncf.model.weight.v1.raw',
    weightConfig: 'application/vnd.cncf.model.weight.config.v1.raw',
});
// Every type is asked for, so a registry answers an index as an index (refused here) instead of 404 or 406.
const MANIFEST_ACCEPT = [MEDIA.ociManifest, MEDIA.ociIndex, MEDIA.dockerManifest, MEDIA.dockerList].join(', ');
// Layers whose digest is not a per-file digest (tars, Ollama), and the legacy Docker safetensors and per-file layouts.
const REFUSED_LAYER = /ollama|\.tar(?:$|\+)|^application\/vnd\.docker\.ai\.(?:safetensors|model\.file)$/;
const FILEPATH = 'org.cncf.model.filepath';

const unsupported = (message) => new LocalLlmError('unsupported_artifact', message);
const sha256Hex = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const plainObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const withTimeout = (signal, ms) => AbortSignal.any([AbortSignal.timeout(ms), ...(signal ? [signal] : [])]);

function needFetch(fetchImpl) {
    if (typeof fetchImpl !== 'function') throw new TypeError('fetchImpl is required');
}

async function discard(response) {
    try {
        await response.body?.cancel();
    } catch {
        // The connection is being dropped anyway.
    }
}

// A body as raw bytes (a digest is over bytes, not text), never more than `maxBytes`.
async function readBytes(response, maxBytes, what) {
    const tooLarge = () => new DownloadError('RESOLVE_FAILED', `${what} is larger than ${maxBytes} bytes`, { details: { reason: 'too-large' } });
    if (Number(response.headers.get('content-length')) > maxBytes) {
        await discard(response);
        throw tooLarge();
    }
    const parts = [];
    let total = 0;
    for await (const chunk of response.body ?? []) {
        total += chunk.length;
        if (total > maxBytes) {
            await discard(response);
            throw tooLarge();
        }
        parts.push(chunk);
    }
    return Buffer.concat(parts, total);
}

async function statusFailure(response, what) {
    await discard(response);
    if (response.status === 404) return new DownloadError('NOT_FOUND', `The registry has no ${what}`, { details: { status: 404 } });
    return new DownloadError('RESOLVE_FAILED', `The registry answered HTTP ${response.status} for ${what}`, { details: { status: response.status } });
}

/**
 * The scheme and parameters of a `WWW-Authenticate` challenge, or null when it is not one well-formed Bearer (with a
 * realm) or Basic challenge. A malformed challenge is never answered.
 */
export function parseChallenge(header) {
    if (typeof header !== 'string' || header.length > 4096) return null;
    const match = /^\s*([A-Za-z]+)(?:\s+(.*?))?\s*$/.exec(header);
    if (!match || !['bearer', 'basic'].includes(match[1].toLowerCase())) return null;
    const scheme = match[1].toLowerCase();
    const rest = match[2] ?? '';
    const params = Object.create(null);
    const param = /\s*([A-Za-z][A-Za-z0-9_-]*)=(?:"((?:[^"\\\u0000-\u001f]|\\[^\u0000-\u001f])*)"|([^\s",]+))\s*(,|$)/y;
    for (let at = 0; at < rest.length;) {
        param.lastIndex = at;
        const found = param.exec(rest);
        if (!found) return null;
        const name = found[1].toLowerCase();
        if (name in params) return null;
        params[name] = found[2] !== undefined ? found[2].replace(/\\(.)/g, '$1') : found[3];
        if (param.lastIndex === at) return null;
        at = param.lastIndex;
    }
    if (scheme === 'bearer' && !params.realm) return null;
    return Object.freeze({ scheme, realm: params.realm ?? null, service: params.service ?? null });
}

const realmRefused = (subject, why) => new DownloadError('ACCESS_DENIED', `${subject}: token realm not allowed (${why})`, { details: { reason: 'realm' } });

/**
 * The session of one lookup, pin or download job: `registry` is the snapshot (`url`, `authOrigins`, `label` and the
 * non-enumerable `credential` `{ token, username? }`), `fetchImpl` the guarded transport. `request(url, init,
 * { sameOriginOnly })` sends a request to the registry origin; `fetch(url, init)` is the same for a blob, whose
 * redirect to a CDN is followed. `now` is the clock of the token cache.
 */
export function createOciSession({ registry, repository, fetchImpl, now = Date.now, timeoutMs = METADATA_TIMEOUT_MS }) {
    needFetch(fetchImpl);
    const origin = new URL(registry.url).origin;
    const authOrigins = new Set((registry.authOrigins ?? []).map((value) => new URL(value).origin));
    const credential = registry.credential?.token ? registry.credential : null;
    const subject = registry.label || registry.id;
    let bearer = null;
    let basicAtHome = false;
    const basic = () => `Basic ${Buffer.from(`${credential.username ?? ''}:${credential.token}`).toString('base64')}`;

    function send(url, init, sameOriginOnly, authorization) {
        const headers = {};
        for (const [name, value] of Object.entries(init.headers ?? {})) {
            if (!['authorization', 'cookie', 'proxy-authorization'].includes(name.toLowerCase())) headers[name] = value;
        }
        if (authorization) headers.Authorization = authorization;
        return followRedirects(fetchImpl, { homeOrigin: origin, credentialOrigins: [origin], sameOriginOnly, subject })(url, { ...init, headers });
    }

    async function fetchToken(challenge, signal) {
        let realm;
        try {
            realm = new URL(challenge.realm);
        } catch {
            throw realmRefused(subject, 'not a URL');
        }
        if (realm.protocol !== 'https:' || realm.username || realm.password) throw realmRefused(subject, 'not https');
        if (realm.origin !== origin && !authOrigins.has(realm.origin)) throw realmRefused(subject, `${realm.origin} is not the registry or one of its auth origins`);
        realm.hash = '';
        if (challenge.service !== null) realm.searchParams.set('service', challenge.service);
        // The scope is this session's own, whatever the challenge named.
        realm.searchParams.set('scope', `repository:${repository}:pull`);
        const headers = { Accept: 'application/json' };
        if (credential && authOrigins.has(realm.origin)) headers.Authorization = basic();
        const follow = followRedirects(fetchImpl, { homeOrigin: realm.origin, credentialOrigins: [realm.origin], sameOriginOnly: true, subject });
        const response = await follow(realm.href, { headers, signal: withTimeout(signal, timeoutMs) });
        if (response.status !== 200) throw await statusFailure(response, 'token');
        let answer;
        try {
            answer = JSON.parse((await readBytes(response, TOKEN_RESPONSE_MAX_BYTES, 'The token response')).toString('utf8'));
        } catch (error) {
            if (error instanceof DownloadError) throw error;
            answer = null;
        }
        const value = plainObject(answer) ? (answer.token || answer.access_token) : undefined;
        if (typeof value !== 'string' || value.length > TOKEN_MAX || !/^[\x21-\x7e]+$/.test(value)) {
            throw new DownloadError('RESOLVE_FAILED', `${subject}'s token service gave no usable token`, { details: { reason: 'token' } });
        }
        const lifetime = Number.isFinite(answer.expires_in) && answer.expires_in > 0 ? answer.expires_in : 60;
        bearer = { value, until: now() + (Math.min(lifetime, 300) - 10) * 1000 };
        return `Bearer ${value}`;
    }

    async function request(url, init = {}, { sameOriginOnly = false } = {}) {
        if (new URL(url).origin !== origin) throw new DownloadError('RESOLVE_FAILED', `${subject}: only the registry's own URLs are requested`, { details: { reason: 'origin' } });
        const cached = bearer && now() < bearer.until ? `Bearer ${bearer.value}` : (basicAtHome ? basic() : null);
        const first = await send(url, init, sameOriginOnly, cached);
        if (first.status !== 401) return first;
        // One answer to the challenge per request: a 401 after it stands.
        const challenge = parseChallenge(first.headers.get('www-authenticate'));
        if (!challenge || (challenge.scheme === 'basic' && (!credential || !authOrigins.has(origin) || basicAtHome))) return first;
        await discard(first);
        let authorization;
        if (challenge.scheme === 'basic') {
            basicAtHome = true;
            authorization = basic();
        } else {
            bearer = null;
            authorization = await fetchToken(challenge, init.signal);
        }
        return send(url, init, sameOriginOnly, authorization);
    }

    return Object.freeze({
        origin,
        repository,
        subject,
        request,
        fetch: (url, init = {}) => request(url, init),
        blobUrl: (digest) => `${origin}/v2/${repository}/blobs/${digest}`,
        manifestUrl: (reference) => `${origin}/v2/${repository}/manifests/${reference}`,
    });
}

/**
 * The manifest `tag` (or `manifestDigest`) names, with its digest: a HEAD gives the digest, the GET is by digest and
 * the body must hash to it. Without a digest header, the GET is by tag and the digest is the body's.
 */
export async function readOciManifest(session, { tag, manifestDigest = undefined, signal = undefined, timeoutMs = METADATA_TIMEOUT_MS }) {
    const accept = { Accept: MANIFEST_ACCEPT };
    let digest = manifestDigest;
    if (!digest) {
        const head = await session.request(session.manifestUrl(tag), { method: 'HEAD', headers: accept, signal: withTimeout(signal, timeoutMs) }, { sameOriginOnly: true });
        if (head.status === 200) {
            const named = head.headers.get('docker-content-digest');
            if (OCI_DIGEST_RE.test(named ?? '')) digest = named;
        } else if (head.status === 404 || head.status === 401 || head.status === 403) {
            throw await statusFailure(head, `manifest ${tag}`);
        }
    }
    const response = await session.request(session.manifestUrl(digest ?? tag), { headers: accept, signal: withTimeout(signal, timeoutMs) }, { sameOriginOnly: true });
    if (response.status !== 200) throw await statusFailure(response, `manifest ${tag}`);
    const bytes = await readBytes(response, MANIFEST_MAX_BYTES, 'The manifest');
    const actual = `sha256:${sha256Hex(bytes)}`;
    if (digest && actual !== digest) {
        throw new DownloadError('SHA256_MISMATCH', `The manifest does not hash to its digest ${digest}`, { details: { reason: 'digest-mismatch', expected: digest, actual } });
    }
    let manifest = null;
    try {
        manifest = JSON.parse(bytes.toString('utf8'));
    } catch {
        // Classified below as not a manifest.
    }
    return { digest: actual, manifest, contentType: response.headers.get('content-type') };
}

function descriptor(value, what) {
    if (!plainObject(value) || typeof value.mediaType !== 'string' || value.mediaType.length > 255 || !OCI_DIGEST_RE.test(String(value.digest))
        || !Number.isSafeInteger(value.size) || value.size < 0) {
        throw unsupported(`The manifest's ${what} is not a valid descriptor.`);
    }
    return value;
}

// CNCF split shards must be named in order and stand in layer order; anything else is refused, never reordered.
function cncfGguf(found) {
    const byName = new Map(found.map((entry, index) => [entry.file, { ...entry, index }]));
    const choices = [];
    for (const entry of found) {
        const split = splitGgufName(entry.file);
        if (!split) {
            choices.push({ file: entry.file, size: entry.size, sha256: entry.sha256 });
            continue;
        }
        if (split.index !== 1) continue;
        const shards = (splitGgufFiles(entry.file) ?? []).map((name) => byName.get(name) ?? null);
        if (shards.length === 0 || shards.includes(null)) continue;
        if (shards.some((shard, k) => k > 0 && shard.index < shards[k - 1].index)) throw unsupported(`The shards of ${entry.file} are not in layer order.`);
        choices.push({ file: entry.file, size: shards.reduce((sum, shard) => sum + shard.size, 0), shards: shards.map(({ file, size, sha256 }) => ({ file, size, sha256 })) });
    }
    return choices;
}

/**
 * What an image manifest offers, decided from the manifest alone: `layout` (`docker` or `cncf`), the config
 * descriptor, `gguf` (single files and complete split sets, each `{ file, size, sha256 }` or `{ file, size, shards }`)
 * and `snapshot` (top-level safetensors model files). Indexes, other artifact and config types, refused layer types,
 * unsafe or duplicated filepaths and too many layers are `unsupported_artifact`.
 */
export function classifyOciManifest(manifest, contentType = null) {
    if (!plainObject(manifest)) throw unsupported('The registry answered with something that is not a manifest.');
    const media = manifest.mediaType ?? String(contentType ?? '').split(';')[0].trim();
    if (Array.isArray(manifest.manifests) || media === MEDIA.ociIndex || media === MEDIA.dockerList) {
        throw unsupported('The tag names an index or manifest list, not one model manifest.');
    }
    if (manifest.schemaVersion !== 2 || ![MEDIA.ociManifest, MEDIA.dockerManifest].includes(media)) throw unsupported('The registry answered with an unknown manifest type.');
    if (manifest.artifactType !== undefined && manifest.artifactType !== MEDIA.cncfArtifact) {
        throw unsupported(`The artifact type ${String(manifest.artifactType).slice(0, 100)} is not a model this agent reads.`);
    }
    const config = descriptor(manifest.config, 'config');
    const layout = config.mediaType === MEDIA.legacyConfig ? 'docker' : config.mediaType === MEDIA.cncfConfig ? 'cncf' : null;
    if (!layout || (layout === 'docker' && manifest.artifactType !== undefined)) {
        throw unsupported(`The config type ${config.mediaType.slice(0, 100)} is not a model config this agent reads.`);
    }
    if (!Array.isArray(manifest.layers) || manifest.layers.length === 0 || manifest.layers.length > MAX_LAYERS) {
        throw unsupported(`The manifest must list 1-${MAX_LAYERS} layers.`);
    }
    const layers = manifest.layers.map((layer, index) => descriptor(layer, `layer ${index + 1}`));
    const refused = layers.find((layer) => REFUSED_LAYER.test(layer.mediaType));
    if (refused) throw unsupported(`A ${refused.mediaType.slice(0, 100)} layer has no per-file digest, or is a layout this agent does not read.`);
    const fileOf = (layer) => ({ size: layer.size, sha256: layer.digest.slice('sha256:'.length) });
    if (layout === 'docker') {
        // Legacy layers are named by their place, so an empty one is refused rather than skipped (it would rename the rest).
        const gguf = layers.filter((layer) => layer.mediaType === MEDIA.legacyGguf);
        if (gguf.length > MAX_GGUF_LAYERS) throw unsupported(`The manifest has more than ${MAX_GGUF_LAYERS} GGUF layers.`);
        if (gguf.some((layer) => layer.size === 0)) throw unsupported('A GGUF layer is empty.');
        const names = gguf.length === 1 ? [LEGACY_SINGLE_GGUF] : splitGgufFiles(`model-00001-of-${String(gguf.length).padStart(5, '0')}.gguf`) ?? [];
        const shards = gguf.map((layer, index) => ({ file: names[index], ...fileOf(layer) }));
        const choices = shards.length === 1 ? shards : shards.length > 1
            ? [{ file: names[0], size: shards.reduce((sum, shard) => sum + shard.size, 0), shards }] : [];
        return { layout, config, gguf: choices, snapshot: [] };
    }
    const named = [];
    const seen = new Set();
    for (const layer of layers) {
        const raw = layer.annotations?.[FILEPATH];
        if (raw === undefined) continue;
        const file = sanitizeOciFilepath(raw);
        if (file === null) throw unsupported('A layer names an unsafe file path.');
        if (seen.has(file)) throw unsupported(`Two layers name the file ${file}.`);
        seen.add(file);
        if (layer.size > 0) named.push({ layer, file });
    }
    const ggufFound = named.filter(({ layer, file }) => layer.mediaType === MEDIA.weight && file.endsWith('.gguf')
        && !file.split('/').pop().toLowerCase().includes('mmproj'));
    if (ggufFound.length > MAX_GGUF_LAYERS) throw unsupported(`The manifest has more than ${MAX_GGUF_LAYERS} GGUF layers.`);
    const gguf = cncfGguf(ggufFound.filter(({ file }) => isOciGgufFile(file)).map(({ layer, file }) => ({ file, ...fileOf(layer) })));
    const snapshot = named.filter(({ layer, file }) => [MEDIA.weight, MEDIA.weightConfig].includes(layer.mediaType) && isSnapshotFile(file))
        .map(({ layer, file }) => ({ path: file, ...fileOf(layer) }))
        .sort((left, right) => (left.path < right.path ? -1 : left.path > right.path ? 1 : 0));
    return { layout, config, gguf, snapshot };
}

// A small blob read whole through the session (a CDN redirect is followed), checked against its size and digest.
async function readBlob(session, { digest, size }, { maxBytes, signal, timeoutMs, what }) {
    if (size > maxBytes) throw new DownloadError('RESOLVE_FAILED', `${what} is larger than ${maxBytes} bytes`, { details: { reason: 'too-large' } });
    const response = await session.request(session.blobUrl(digest), { headers: {}, signal: withTimeout(signal, timeoutMs) });
    if (response.status !== 200) throw await statusFailure(response, what);
    const bytes = await readBytes(response, maxBytes, what);
    const actual = `sha256:${sha256Hex(bytes)}`;
    if (bytes.length !== size || actual !== digest) {
        throw new DownloadError('SHA256_MISMATCH', `${what} does not match its digest`, { details: { reason: 'digest-mismatch', expected: digest, actual } });
    }
    return bytes;
}

/** The CNCF config: `format` and the first licence, read by digest and checked. */
export async function readOciConfig(session, config, { signal = undefined, timeoutMs = METADATA_TIMEOUT_MS } = {}) {
    const bytes = await readBlob(session, config, { maxBytes: CONFIG_MAX_BYTES, signal, timeoutMs, what: 'The model config' });
    let parsed = null;
    try {
        parsed = JSON.parse(bytes.toString('utf8'));
    } catch {
        // Refused below.
    }
    if (!plainObject(parsed) || !plainObject(parsed.config)) throw unsupported('The model config is not a CNCF model config.');
    const licenses = Array.isArray(parsed.descriptor?.licenses) ? parsed.descriptor.licenses : [];
    return { format: parsed.config.format, license: licenseOf({ cardData: { license: licenses[0] } }) };
}

/** The GGUF key-value section of a blob, from a ranged read of at most 32 MiB that stops once the header is parsed. */
export async function readOciGgufHeader(session, sha256, { signal = undefined } = {}) {
    const stop = new AbortController();
    try {
        const response = await session.request(session.blobUrl(`sha256:${sha256}`),
            { headers: { Range: `bytes=0-${HEADER_MAX_BYTES - 1}` }, signal: AbortSignal.any([stop.signal, withTimeout(signal, LOOKUP_TIMEOUT_MS)]) });
        if (response.status !== 200 && response.status !== 206) throw await statusFailure(response, 'GGUF file');
        const reader = createGgufHeaderReader();
        let readBytes = 0;
        for await (const chunk of response.body ?? []) {
            readBytes += chunk.length;
            if (reader.push(chunk) === 'done' || readBytes > HEADER_MAX_BYTES) break;
        }
        return { kv: reader.end().kv, readBytes };
    } finally {
        stop.abort();
    }
}

// A legacy split names no shard, so each shard's own header must agree with its place: split.no k-1 of split.count N.
async function checkLegacySplit(session, shards, signal) {
    for (const [index, shard] of shards.entries()) {
        const { kv } = await readOciGgufHeader(session, shard.sha256, { signal });
        if (Number(kv['split.no']) !== index || Number(kv['split.count']) !== shards.length) {
            throw unsupported(`The header of layer ${index + 1} does not say it is shard ${index + 1} of ${shards.length}.`);
        }
    }
}

/** Pin an `oci` source: the manifest digest, the layout and the GGUF file (derived when it is the only one), or every shard. */
export async function pinOciGguf(session, source, { signal = undefined } = {}) {
    const { digest, manifest, contentType } = await readOciManifest(session, { tag: source.tag, manifestDigest: source.manifestDigest, signal });
    const found = classifyOciManifest(manifest, contentType);
    if (!found.gguf.length) throw unsupported('The manifest has no GGUF file this agent can add.');
    if (found.layout === 'cncf' && (await readOciConfig(session, found.config, { signal })).format !== 'gguf') {
        throw unsupported('The model config does not say format gguf.');
    }
    let choice;
    if (source.file !== undefined) {
        choice = found.gguf.find((entry) => entry.file === source.file);
        if (!choice) throw new DownloadError('NOT_FOUND', `${source.file} is not a GGUF file of the manifest`);
    } else if (found.gguf.length === 1) {
        [choice] = found.gguf;
    } else {
        throw new LocalLlmError('invalid_model', `The manifest offers ${found.gguf.length} GGUF files; name one in file.`, { field: 'file' });
    }
    if (found.layout === 'docker' && choice.shards) await checkLegacySplit(session, choice.shards, signal);
    const { file, size, sha256, shards } = choice;
    return { ...source, manifestDigest: digest, layout: found.layout, file, size, ...(shards ? { shards } : { sha256 }) };
}

/** Pin an `oci-snapshot` source: the manifest digest and its top-level safetensors model files. */
export async function pinOciSnapshot(session, source, { signal = undefined } = {}) {
    const { digest, manifest, contentType } = await readOciManifest(session, { tag: source.tag, manifestDigest: source.manifestDigest, signal });
    const found = classifyOciManifest(manifest, contentType);
    const files = snapshotFilesOf(found);
    if ((await readOciConfig(session, found.config, { signal })).format !== 'safetensors') throw unsupported('The model config does not say format safetensors.');
    return { ...source, manifestDigest: digest, files, size: files.reduce((sum, file) => sum + file.size, 0) };
}

function snapshotFilesOf(found) {
    if (found.layout !== 'cncf' || !found.snapshot.some((file) => file.path.endsWith('.safetensors'))) {
        throw unsupported('The manifest has no CNCF safetensors files.');
    }
    if (found.snapshot.length > MAX_SNAPSHOT_FILES) throw unsupported(`The manifest has more than ${MAX_SNAPSHOT_FILES} model files.`);
    return found.snapshot;
}

const rowOf = ({ file, path, size, sha256, shards }, gguf) => ({
    file: file ?? path, size, sha256: shards ? null : sha256, gitOid: null, quantization: gguf ? quantizationOf(file) : null, shards: shards ?? null,
});

/**
 * Look a model up in an OCI registry (`registry` is its snapshot): the manifest digest, layout and licence, the files
 * of the format, and the sizing from one GGUF header or the snapshot's `config.json`. `revision` is the tag
 * (`latest` when absent). The answer has the Hugging Face lookup's shape with `commit: null`; the caller adds `registry`.
 */
export async function lookupOci(args, { registry, fetchImpl, signal = undefined, timeoutMs = METADATA_TIMEOUT_MS, totalMs = LOOKUP_TIMEOUT_MS, now = Date.now } = {}) {
    needFetch(fetchImpl);
    const input = normalizeLookupInput(args, { kind: 'oci' });
    const deadline = withTimeout(signal, totalMs);
    const token = registry.credential?.token ?? '';
    try {
        const session = createOciSession({ registry, repository: input.repo, fetchImpl, now, timeoutMs });
        const { digest, manifest, contentType } = await readOciManifest(session, { tag: input.revision, signal: deadline, timeoutMs });
        const found = classifyOciManifest(manifest, contentType);
        const gguf = input.format === 'gguf';
        const files = gguf ? found.gguf : snapshotFilesOf(found);
        if (!files.length) throw unsupported('The manifest has no GGUF file this agent can add.');
        let license = null;
        if (found.layout === 'cncf') {
            const config = await readOciConfig(session, found.config, { signal: deadline, timeoutMs });
            if (config.format !== (gguf ? 'gguf' : 'safetensors')) throw unsupported(`The model config says format ${String(config.format).slice(0, 40)}.`);
            license = config.license;
        }
        let sizing = null;
        if (gguf && input.file !== undefined) {
            const choice = files.find((entry) => entry.file === input.file);
            if (!choice) throw new LocalLlmError('not_found', `${input.file} is not a GGUF file of ${input.repo}:${input.revision} that can be added.`);
            const { kv, readBytes } = await readOciGgufHeader(session, (choice.shards?.[0] ?? choice).sha256, { signal: deadline });
            const read = ggufSizing(kv);
            sizing = { contextLength: read.contextLength, architecture: read.architecture, memory: { layers: read.layers, kvBytesPerToken: read.kvBytesPerToken },
                source: 'gguf-header', readBytes, notes: read.notes };
        } else if (!gguf) {
            const config = files.find((entry) => entry.path === 'config.json');
            if (config && config.size <= CONFIG_MAX_BYTES) {
                const bytes = await readBlob(session, { digest: `sha256:${config.sha256}`, size: config.size }, { maxBytes: CONFIG_MAX_BYTES, signal: deadline, timeoutMs, what: 'config.json' });
                let parsed = null;
                try {
                    parsed = JSON.parse(bytes.toString('utf8'));
                } catch {
                    // Refused below.
                }
                if (!plainObject(parsed)) throw new LocalLlmError('lookup_failed', 'config.json is not a JSON object.');
                sizing = { ...sizingFromConfig(parsed), readBytes: bytes.length };
            }
        }
        return {
            provider: 'oci', repo: input.repo, revision: input.revision, commit: null, manifestDigest: digest, layout: found.layout, format: input.format,
            gated: null, license, files: files.map((entry) => rowOf(entry, gguf)), truncated: false, sizing,
        };
    } catch (error) {
        if (error instanceof DownloadError && error.code === 'ACCESS_DENIED') throw new LocalLlmError('access_denied', error.message);
        throw lookupError(error, signal, token, { subject: registry.label, hint: REGISTRY_TOKEN_HINT });
    }
}
