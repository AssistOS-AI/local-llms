// OCI model sources (DS002): the shapes of an `oci` source (one GGUF file, or every shard of a split GGUF, from an
// OCI image manifest) and an `oci-snapshot` source (CNCF per-file safetensors), the repository, tag and digest
// patterns, and the filepath annotations a manifest gives, which are untrusted. A pinned source names its manifest by
// digest and every file by size and sha256 (a layer's digest is the sha256 of the raw file).
//
// The catalog imports this module and this module imports the catalog, so nothing here uses a catalog binding at
// module load: only inside functions.

import { LocalLlmError } from '../errors.mjs';
import { HF_FILE_SEGMENT_RE, isSnapshotFile, splitGgufName, validateShards } from './catalog.mjs';
import { REGISTRY_ID_RE } from './registries.mjs';

// The OCI distribution name grammar, 2-4 path components, each lowercase alphanumerics joined by `.`, `_`, `__` or dashes.
export const OCI_REPOSITORY_RE = /^[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*(?:\/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*){1,3}$/;
export const OCI_REPOSITORY_MAX = 255;
export const OCI_TAG_RE = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;
export const OCI_DIGEST_RE = /^sha256:[0-9a-f]{64}$/;
export const OCI_LAYOUTS = Object.freeze(['docker', 'cncf']);
// The names a legacy (docker) GGUF layer gets: it has none of its own, so its place in the manifest names it.
export const LEGACY_SINGLE_GGUF = 'model.gguf';
const SHA256_RE = /^[0-9a-f]{64}$/;
const MAX_FILEPATH = 512;
const MAX_SNAPSHOT_FILES = 200;
const MAX_SHARDS = 64;

function invalid(message, field) {
    return new LocalLlmError('invalid_model', message, { field });
}

function plainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
        && [Object.prototype, null].includes(Object.getPrototypeOf(value));
}

function onlyKeys(value, allowed, field) {
    for (const key of Object.keys(value)) {
        if (!allowed.includes(key)) throw invalid(`${field} has an unsupported field '${key}'`, `${field}.${key}`);
    }
}

export function isOciRepository(value) {
    return typeof value === 'string' && value.length <= OCI_REPOSITORY_MAX && OCI_REPOSITORY_RE.test(value);
}

/**
 * An `org.cncf.model.filepath` annotation in its safe form, or null when it is unsafe: not a string, empty, longer than
 * 512 characters, absolute, with a backslash, a control character or a `..` segment. `.` and empty segments are
 * dropped, so `./a//b` and `a/b` are the same path (and a manifest naming both is refused as a duplicate).
 */
export function sanitizeOciFilepath(value) {
    if (typeof value !== 'string' || !value || value.length > MAX_FILEPATH) return null;
    if (/[\u0000-\u001f\u007f\\]/.test(value) || value.startsWith('/')) return null;
    const segments = value.split('/').filter((segment) => segment !== '' && segment !== '.');
    if (segments.length === 0 || segments.includes('..')) return null;
    return segments.join('/');
}

/** A GGUF file name a store can hold: 1-4 plain segments ending `.gguf`, as for Hugging Face. */
export function isOciGgufFile(file) {
    if (typeof file !== 'string' || !file.endsWith('.gguf')) return false;
    const segments = file.split('/');
    return segments.length <= 4 && segments.every((segment) => HF_FILE_SEGMENT_RE.test(segment));
}

function common(value, field, type, keys) {
    if (!plainObject(value)) throw invalid(`${field} must be an object`, field);
    if (value.type !== type) throw invalid(`${field}.type must be ${type}`, `${field}.type`);
    onlyKeys(value, keys, field);
    if (typeof value.registry !== 'string' || !REGISTRY_ID_RE.test(value.registry) || value.registry === 'huggingface') {
        throw invalid(`${field}.registry must name an oci registry (2-32 lowercase letters, digits or dashes)`, `${field}.registry`);
    }
    if (!isOciRepository(value.repository)) {
        throw invalid(`${field}.repository must be an OCI repository of 2-4 lowercase segments, such as ai/smollm2`, `${field}.repository`);
    }
    if (typeof value.tag !== 'string' || !OCI_TAG_RE.test(value.tag)) throw invalid(`${field}.tag must be an OCI tag`, `${field}.tag`);
    if (value.manifestDigest !== undefined && !OCI_DIGEST_RE.test(String(value.manifestDigest))) {
        throw invalid(`${field}.manifestDigest must be sha256:<64 hex>`, `${field}.manifestDigest`);
    }
    return {
        type,
        registry: value.registry,
        repository: value.repository,
        tag: value.tag,
        ...(value.manifestDigest !== undefined ? { manifestDigest: value.manifestDigest } : {}),
    };
}

/**
 * An `oci` source (weight format gguf). Unpinned (a user entry before Add pins it): registry, repository, tag, and
 * optionally the manifest digest and the file. Pinned: the manifest digest, the layout (`docker` names its layers
 * model.gguf or model-0000k-of-0000N.gguf in layer order; `cncf` takes the sanitized filepath), the file, its size and
 * sha256, or every shard of a split GGUF.
 */
export function validateOciSource(value, field, { requirePin = false } = {}) {
    const base = common(value, field, 'oci', [
        'type', 'registry', 'repository', 'tag', 'manifestDigest', 'layout', 'file', 'size', 'sha256', 'shards', 'quantization',
    ]);
    if (value.quantization !== undefined && (typeof value.quantization !== 'string' || value.quantization.length > 64
        || /[\u0000-\u001f]/.test(value.quantization))) {
        throw invalid(`${field}.quantization must be plain text of at most 64 characters`, `${field}.quantization`);
    }
    const quantization = value.quantization ? { quantization: value.quantization.trim() } : {};
    if (value.file !== undefined) {
        if (!isOciGgufFile(value.file)) throw invalid(`${field}.file must be a .gguf file of the manifest (the first shard of a split GGUF)`, `${field}.file`);
        const split = splitGgufName(value.file);
        if (split && split.count > MAX_SHARDS) throw invalid(`${field}.file names ${split.count} shards; at most ${MAX_SHARDS} are supported`, `${field}.file`);
        if (split && split.index !== 1) throw invalid(`${field}.file names shard ${split.index} of ${split.count}; name the first shard`, `${field}.file`);
    }
    const pinned = ['layout', 'size', 'sha256', 'shards'].some((key) => value[key] !== undefined);
    if (!requirePin && !pinned) return Object.freeze({ ...base, ...(value.file !== undefined ? { file: value.file } : {}), ...quantization });
    if (!base.manifestDigest) throw invalid(`${field}.manifestDigest is required for a pinned source`, `${field}.manifestDigest`);
    if (!OCI_LAYOUTS.includes(value.layout)) throw invalid(`${field}.layout must be docker or cncf`, `${field}.layout`);
    if (value.file === undefined) throw invalid(`${field}.file is required for a pinned source`, `${field}.file`);
    const split = splitGgufName(value.file);
    if (value.layout === 'docker' && (split ? split.prefix !== 'model' : value.file !== LEGACY_SINGLE_GGUF)) {
        throw invalid(`${field}.file must be ${LEGACY_SINGLE_GGUF} or model-0000k-of-0000N.gguf for the docker layout`, `${field}.file`);
    }
    if (split) {
        if (value.sha256 !== undefined) throw invalid(`${field}.sha256 is per shard for a split GGUF`, `${field}.sha256`);
        const { shards, size } = validateShards(value, field);
        return Object.freeze({ ...base, layout: value.layout, file: value.file, size, shards, ...quantization });
    }
    if (value.shards !== undefined) throw invalid(`${field}.shards is only for a split GGUF (…-00001-of-0000N.gguf)`, `${field}.shards`);
    if (!Number.isSafeInteger(value.size) || value.size <= 0) throw invalid(`${field}.size must be the file size in bytes`, `${field}.size`);
    if (!SHA256_RE.test(String(value.sha256 || ''))) throw invalid(`${field}.sha256 must be the layer's 64 hex digest`, `${field}.sha256`);
    return Object.freeze({ ...base, layout: value.layout, file: value.file, size: value.size, sha256: value.sha256, ...quantization });
}

/**
 * An `oci-snapshot` source (weight format hf): CNCF per-file safetensors. Pinned: the manifest digest and 1-200 files,
 * each a top-level model file with its size and sha256, at least one `.safetensors`; `size` is their sum.
 */
export function validateOciSnapshotSource(value, field, { requirePin = false } = {}) {
    const base = common(value, field, 'oci-snapshot', ['type', 'registry', 'repository', 'tag', 'manifestDigest', 'files', 'size']);
    if (!requirePin && value.files === undefined) {
        if (value.size !== undefined) throw invalid(`${field}.size is computed from the pinned files`, `${field}.size`);
        return Object.freeze(base);
    }
    if (!base.manifestDigest) throw invalid(`${field}.manifestDigest is required for a pinned source`, `${field}.manifestDigest`);
    if (!Array.isArray(value.files) || value.files.length === 0 || value.files.length > MAX_SNAPSHOT_FILES) {
        throw invalid(`${field}.files must list 1-${MAX_SNAPSHOT_FILES} files`, `${field}.files`);
    }
    const seen = new Set();
    const files = value.files.map((file, index) => {
        const at = `${field}.files[${index}]`;
        if (!plainObject(file)) throw invalid(`${at} must be an object`, at);
        onlyKeys(file, ['path', 'size', 'sha256'], at);
        if (!isSnapshotFile(file.path)) throw invalid(`${at}.path must be a top-level model file`, `${at}.path`);
        if (seen.has(file.path)) throw invalid(`${field}.files lists ${file.path} twice`, at);
        seen.add(file.path);
        if (!Number.isSafeInteger(file.size) || file.size <= 0) throw invalid(`${at}.size must be the file size in bytes`, `${at}.size`);
        if (!SHA256_RE.test(String(file.sha256 || ''))) throw invalid(`${at}.sha256 must be 64 hex`, `${at}.sha256`);
        return Object.freeze({ path: file.path, size: file.size, sha256: file.sha256 });
    });
    if (!files.some((file) => file.path.endsWith('.safetensors'))) throw invalid(`${field}.files has no .safetensors file`, `${field}.files`);
    const size = files.reduce((sum, file) => sum + file.size, 0);
    if (value.size !== undefined && value.size !== size) throw invalid(`${field}.size must equal the sum of the file sizes (${size})`, `${field}.size`);
    return Object.freeze({ ...base, files: Object.freeze(files), size });
}
