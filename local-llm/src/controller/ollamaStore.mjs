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

function blobPath(modelsDir, digest) {
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
                        if (item?.digest) referenced.add(item.digest);
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
    const manifest = readOllamaManifest(modelsDir, tag, { fsApi });
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

/** Bytes of partial pull data Ollama keeps for resuming (blob-*-partial files). */
export function partialPullBytes(modelsDir, { fsApi = fs } = {}) {
    let total = 0;
    try {
        for (const entry of fsApi.readdirSync(path.join(modelsDir, 'blobs'))) {
            if (entry.endsWith('-partial')) total += fsApi.statSync(path.join(modelsDir, 'blobs', entry)).size;
        }
    } catch {}
    return total;
}
