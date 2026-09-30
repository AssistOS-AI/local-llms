// Ollama's on-disk store under /data/models/ollama: which tags are present,
// their exact manifest identity, and deletion without a running server.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';

const REGISTRY = 'registry.ollama.ai';
const REGISTRY_URL = `https://${REGISTRY}`;
// What Add sends to read a tag's manifest (Docker distribution manifest v2; DS002).
const MANIFEST_ACCEPT = 'application/vnd.docker.distribution.manifest.v2+json';
const MAX_MANIFEST_BYTES = 1024 * 1024;
const MAX_LAYERS = 512;
const MAX_REDIRECTS = 3;
const REGISTRY_TIMEOUT_MS = 30_000;

export function parseTag(tag) {
    const [nameWithNamespace, version = 'latest'] = String(tag).split(':');
    const [namespace, name] = nameWithNamespace.includes('/')
        ? nameWithNamespace.split('/')
        : ['library', nameWithNamespace];
    return { namespace, name, version };
}

export function manifestPath(modelsDir, tag) {
    const { namespace, name, version } = parseTag(tag);
    return path.join(modelsDir, 'manifests', REGISTRY, namespace, name, version);
}

// Layer digests come from a manifest on disk; only this exact form may be
// turned into a file name, so a crafted digest cannot reach outside blobs/.
const DIGEST_RE = /^sha256:[0-9a-f]{64}$/;

export class InvalidOllamaManifestError extends Error {
    constructor(message) {
        super(message);
        this.name = 'InvalidOllamaManifestError';
        this.code = 'invalid_manifest';
    }
}

function blobPath(modelsDir, digest) {
    if (typeof digest !== 'string' || !DIGEST_RE.test(digest)) {
        throw new InvalidOllamaManifestError(`Ollama manifest has an invalid layer digest: ${String(digest).slice(0, 80)}`);
    }
    return path.join(modelsDir, 'blobs', digest.replace(':', '-'));
}

/**
 * A manifest's identity: the sha256 of its exact bytes, the blobs it names
 * (config first, then the layers) and their sizes added up. The store and the
 * pin at Add use this one formula, so a pinned size is the size the store reports.
 */
export function ollamaManifestIdentity(bytes) {
    const manifest = JSON.parse(bytes.toString('utf8'));
    const blobs = [manifest.config, ...(manifest.layers || [])].filter(Boolean)
        .map((entry) => ({ digest: entry.digest, size: entry.size, mediaType: entry.mediaType }));
    return {
        manifestDigest: `sha256:${crypto.createHash('sha256').update(bytes).digest('hex')}`,
        size: blobs.reduce((total, blob) => total + (blob.size || 0), 0),
        blobs,
    };
}

/** @returns {null | { manifestDigest, size, blobs: [{ digest, size, mediaType }], complete }} */
export function readOllamaManifest(modelsDir, tag, { fsApi = fs } = {}) {
    let bytes;
    try {
        bytes = fsApi.readFileSync(manifestPath(modelsDir, tag));
    } catch (error) {
        if (error?.code === 'ENOENT') return null;
        throw error;
    }
    const { manifestDigest, size, blobs } = ollamaManifestIdentity(bytes);
    for (const blob of blobs) blobPath(modelsDir, blob.digest);
    const complete = blobs.every((blob) => {
        try {
            return fsApi.statSync(blobPath(modelsDir, blob.digest)).size === blob.size;
        } catch {
            return false;
        }
    });
    return { manifestDigest, size, blobs, complete };
}

// The body of a response, at most `limit` bytes: a larger one is refused, not read on.
async function readBounded(response, limit) {
    if (!response.body || typeof response.body[Symbol.asyncIterator] !== 'function') {
        const buffer = Buffer.from(await response.arrayBuffer());
        if (buffer.length > limit) throw new LocalLlmError('pin_failed', `The registry's manifest is larger than ${limit} bytes.`);
        return buffer;
    }
    const chunks = [];
    let total = 0;
    for await (const chunk of response.body) {
        total += chunk.length;
        if (total > limit) throw new LocalLlmError('pin_failed', `The registry's manifest is larger than ${limit} bytes.`);
        chunks.push(chunk);
    }
    return Buffer.concat(chunks);
}

/**
 * The identity an Ollama tag has in the registry right now, to pin it at Add
 * (DS002): `manifestDigest` (the sha256 of the exact manifest bytes the
 * registry sends) and `size` (config plus layers, the store's own formula).
 * Nothing but the manifest is fetched: no blob, no credential. A redirect is
 * followed only within the registry's own origin.
 *
 * ASSUMPTIONS, not verified against the live registry: the manifest endpoint
 * is GET <registry>/v2/<namespace>/<name>/manifests/<tag> with the Docker v2
 * Accept header; it answers directly; and Ollama stores the manifest bytes it
 * was sent, so the digest of the stored file equals the one pinned here. If a
 * pull then fails with `identity_changed`, the assumption is wrong and the
 * pin must be reworked, never the check weakened.
 */
export async function fetchOllamaRegistryManifest(tag, { fetchImpl = globalThis.fetch, baseUrl = REGISTRY_URL, timeoutMs = REGISTRY_TIMEOUT_MS, signal } = {}) {
    const { namespace, name, version } = parseTag(tag);
    const origin = new URL(baseUrl).origin;
    const stop = signal ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs);
    let url = `${baseUrl}/v2/${encodeURIComponent(namespace)}/${encodeURIComponent(name)}/manifests/${encodeURIComponent(version)}`;
    const fail = (message, code = 'pin_failed') => new LocalLlmError(code, message);
    let bytes;
    try {
        let response;
        for (let hop = 0; ; hop += 1) {
            response = await fetchImpl(url, { method: 'GET', headers: { Accept: MANIFEST_ACCEPT }, redirect: 'manual', signal: stop });
            if (![301, 302, 303, 307, 308].includes(response.status)) break;
            const next = response.headers?.get?.('location') ? new URL(response.headers.get('location'), url) : null;
            if (next && next.origin !== origin) {
                throw fail(`The registry redirected the manifest request of ${tag} away from ${new URL(baseUrl).host}, which is not followed.`);
            }
            if (hop >= MAX_REDIRECTS || !next) {
                throw fail(`The registry redirected the manifest request of ${tag} ${next ? 'more than ' + MAX_REDIRECTS + ' times' : 'without saying where'}, which is not followed.`);
            }
            url = next.href;
        }
        if (response.status === 404) throw fail(`${new URL(baseUrl).host} has no tag ${tag}.`, 'not_found');
        if (!response.ok) throw fail(`${new URL(baseUrl).host} answered HTTP ${response.status} for the manifest of ${tag}.`);
        bytes = await readBounded(response, MAX_MANIFEST_BYTES);
    } catch (error) {
        if (error instanceof LocalLlmError) throw error;
        throw fail(`Could not read the manifest of ${tag} from ${new URL(baseUrl).host}: ${error?.name === 'TimeoutError' ? 'timed out' : (error?.message || error)}`);
    }
    let identity;
    try {
        const manifest = JSON.parse(bytes.toString('utf8'));
        if (manifest?.schemaVersion !== 2 || !Array.isArray(manifest.layers) || manifest.layers.length === 0 || manifest.layers.length > MAX_LAYERS) {
            throw new Error('not a version 2 manifest with layers');
        }
        identity = ollamaManifestIdentity(bytes);
        if (identity.blobs.length !== manifest.layers.length + 1) throw new Error('the manifest has no config');
        for (const blob of identity.blobs) {
            if (typeof blob.digest !== 'string' || !DIGEST_RE.test(blob.digest)) throw new Error('a layer digest is not sha256:<64 hex>');
            if (!Number.isSafeInteger(blob.size) || blob.size <= 0) throw new Error('a layer size is not a positive byte count');
        }
        if (!Number.isSafeInteger(identity.size)) throw new Error('the sizes add up to more than a safe integer');
    } catch (error) {
        throw fail(`The manifest of ${tag} from ${new URL(baseUrl).host} is not usable: ${error.message}.`);
    }
    return { manifestDigest: identity.manifestDigest, size: identity.size };
}

function referencedDigests(modelsDir, fsApi, skip) {
    const referenced = new Set();
    const root = path.join(modelsDir, 'manifests');
    const walk = (directory) => {
        let entries = [];
        try { entries = fsApi.readdirSync(directory, { withFileTypes: true }); } catch { return; }
        for (const entry of entries) {
            const target = path.join(directory, entry.name);
            if (entry.isDirectory()) walk(target);
            else if (entry.isFile() && target !== skip) {
                try {
                    const manifest = JSON.parse(fsApi.readFileSync(target, 'utf8'));
                    for (const item of [manifest.config, ...(manifest.layers || [])]) {
                        if (typeof item?.digest === 'string' && DIGEST_RE.test(item.digest)) referenced.add(item.digest);
                    }
                } catch {}
            }
        }
    };
    walk(root);
    return referenced;
}

/** Delete a tag's manifest and every blob no other manifest references. */
export function deleteOllamaModel(modelsDir, tag, { fsApi = fs } = {}) {
    const target = manifestPath(modelsDir, tag);
    let manifest;
    try {
        manifest = readOllamaManifest(modelsDir, tag, { fsApi });
    } catch (error) {
        // A manifest with an invalid digest is left alone rather than trusted.
        if (error instanceof InvalidOllamaManifestError) return 0;
        throw error;
    }
    if (!manifest) return 0;
    const keep = referencedDigests(modelsDir, fsApi, target);
    let freed = 0;
    for (const blob of manifest.blobs) {
        if (keep.has(blob.digest)) continue;
        const file = blobPath(modelsDir, blob.digest);
        try {
            freed += fsApi.statSync(file).size;
            fsApi.rmSync(file, { force: true });
        } catch {}
    }
    fsApi.rmSync(target, { force: true });
    return freed;
}

// Ollama resumes a blob download from `sha256-<hex>-partial` and keeps its
// part records in `sha256-<hex>-partial-<n>` files next to it.
const PARTIAL_RE = /^sha256-([0-9a-f]{64})-partial(?:-\d+)?$/;

/** Partial pull files, grouped by the blob digest they belong to. */
export function partialPullFiles(modelsDir, { fsApi = fs } = {}) {
    const byDigest = new Map();
    let entries = [];
    try {
        entries = fsApi.readdirSync(path.join(modelsDir, 'blobs'));
    } catch {
        return byDigest;
    }
    for (const entry of entries) {
        const match = PARTIAL_RE.exec(entry);
        if (!match) continue;
        const digest = `sha256:${match[1]}`;
        if (!byDigest.has(digest)) byDigest.set(digest, []);
        byDigest.get(digest).push(path.join(modelsDir, 'blobs', entry));
    }
    return byDigest;
}

/** Bytes of partial blob data for the given digests (one tag's pull). */
export function partialPullBytes(modelsDir, digests = [], { fsApi = fs } = {}) {
    const wanted = new Set(digests);
    let total = 0;
    for (const [digest, files] of partialPullFiles(modelsDir, { fsApi })) {
        if (!wanted.has(digest)) continue;
        for (const file of files) {
            if (!file.endsWith('-partial')) continue;
            try { total += fsApi.statSync(file).size; } catch {}
        }
    }
    return total;
}

/**
 * Delete one tag's partial pull files, and orphans: partials that no manifest
 * references and no other tag's recorded pull claims. A partial that another
 * tag also claims is kept, since that tag's pull resumes from it. Orphans are
 * kept too when `keepOrphans` is set (another pull is running and may not
 * have reported its digests yet).
 */
export function deleteOllamaPartials(modelsDir, {
    digests = [],
    claimedByOthers = [],
    keepOrphans = false,
    fsApi = fs,
} = {}) {
    const own = new Set(digests);
    const others = new Set(claimedByOthers);
    const referenced = referencedDigests(modelsDir, fsApi, null);
    let freed = 0;
    for (const [digest, files] of partialPullFiles(modelsDir, { fsApi })) {
        if (others.has(digest)) continue;
        const orphan = !keepOrphans && !referenced.has(digest);
        if (!own.has(digest) && !orphan) continue;
        for (const file of files) {
            try {
                freed += fsApi.statSync(file).size;
                fsApi.rmSync(file, { force: true });
            } catch {}
        }
    }
    return freed;
}
