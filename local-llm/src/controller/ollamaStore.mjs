// Ollama's on-disk store under /data/models/ollama: which tags are present,
// their exact manifest identity, and deletion without a running server.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const REGISTRY = 'registry.ollama.ai';

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

/** @returns {null | { manifestDigest, size, blobs: [{ digest, size, mediaType }], complete }} */
export function readOllamaManifest(modelsDir, tag, { fsApi = fs } = {}) {
    let bytes;
    try {
        bytes = fsApi.readFileSync(manifestPath(modelsDir, tag));
    } catch (error) {
        if (error?.code === 'ENOENT') return null;
        throw error;
    }
    const manifest = JSON.parse(bytes.toString('utf8'));
    const blobs = [manifest.config, ...(manifest.layers || [])].filter(Boolean)
        .map((entry) => ({ digest: entry.digest, size: entry.size, mediaType: entry.mediaType }));
    for (const blob of blobs) blobPath(modelsDir, blob.digest);
    const complete = blobs.every((blob) => {
        try {
            return fsApi.statSync(blobPath(modelsDir, blob.digest)).size === blob.size;
        } catch {
            return false;
        }
    });
    return {
        manifestDigest: `sha256:${crypto.createHash('sha256').update(bytes).digest('hex')}`,
        size: blobs.reduce((total, blob) => total + (blob.size || 0), 0),
        blobs,
        complete,
    };
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
 * references and no other tag's recorded pull claims.
 */
export function deleteOllamaPartials(modelsDir, { digests = [], claimedByOthers = [], fsApi = fs } = {}) {
    const own = new Set(digests);
    const others = new Set(claimedByOthers);
    const referenced = referencedDigests(modelsDir, fsApi, null);
    let freed = 0;
    for (const [digest, files] of partialPullFiles(modelsDir, { fsApi })) {
        const orphan = !others.has(digest) && !referenced.has(digest);
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
