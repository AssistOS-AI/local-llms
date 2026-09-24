// Model catalog: the read-only seed shipped with the agent plus the user
// registry persisted under /data. Every entry is validated here; the same
// rules apply to seed entries (at load) and to user entries (at add/update).
//
// Schema v2 keys a model's sources by weight format, not by runner: every
// runner declares the format it reads, so runners that read the same GGUF
// file share one download. `recommended` and `validated` stay per runner.

import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';

export const CATALOG_SCHEMA = 'local-llm.catalog/v2';

const ID_RE = /^[a-z0-9][a-z0-9._-]{1,63}$/;
const HF_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const HF_FILE_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
const HF_REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;
const COMMIT_RE = /^[0-9a-f]{40}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const GIT_OID_RE = /^[0-9a-f]{40}$/;
const OLLAMA_TAG_RE = /^[a-z0-9][a-z0-9._-]{0,63}(?:\/[a-z0-9][a-z0-9._-]{0,63})?(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$/;
const TEXT_MAX = 400;

// v1 registries keyed sources by runner id. Where several v1 keys held a GGUF
// file, the first in this order becomes the gguf source.
const V1_GGUF_KEYS = Object.freeze(['llama.cpp', 'lmstudio', 'vllm']);
// Runners that no longer exist; their per-runner entries are dropped.
const REMOVED_RUNNERS = Object.freeze(['lmstudio']);

function invalid(message, field) {
    return new LocalLlmError('invalid_model', message, { field });
}

function plainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
        && [Object.prototype, null].includes(Object.getPrototypeOf(value));
}

function optionalText(value, field, max = TEXT_MAX) {
    if (value === undefined || value === null || value === '') return undefined;
    if (typeof value !== 'string' || value.length > max || /[\u0000-\u001f]/.test(value)) {
        throw invalid(`${field} must be plain text of at most ${max} characters`, field);
    }
    return value.trim();
}

function optionalInteger(value, field, min, max) {
    if (value === undefined || value === null) return undefined;
    if (!Number.isInteger(value) || value < min || value > max) {
        throw invalid(`${field} must be an integer from ${min} to ${max}`, field);
    }
    return value;
}

function onlyKeys(value, allowed, field) {
    for (const key of Object.keys(value)) {
        if (!allowed.includes(key)) throw invalid(`${field} has an unsupported field '${key}'`, `${field}.${key}`);
    }
}

export function validateHuggingFaceSource(value, field, { requirePin = false } = {}) {
    if (!plainObject(value)) throw invalid(`${field} must be an object`, field);
    if (value.type !== 'huggingface') throw invalid(`${field}.type must be huggingface`, `${field}.type`);
    onlyKeys(value, ['type', 'repo', 'file', 'revision', 'commit', 'size', 'sha256', 'quantization'], field);
    if (typeof value.repo !== 'string' || !HF_REPO_RE.test(value.repo)) {
        throw invalid(`${field}.repo must be a Hugging Face repository id (owner/name)`, `${field}.repo`);
    }
    const segments = typeof value.file === 'string' ? value.file.split('/') : [];
    if (!segments.length || segments.length > 4 || !segments.every((segment) => HF_FILE_SEGMENT_RE.test(segment))
        || !value.file.endsWith('.gguf')) {
        throw invalid(`${field}.file must be a single .gguf file in the repository`, `${field}.file`);
    }
    const revision = value.revision ?? 'main';
    if (typeof revision !== 'string' || !HF_REVISION_RE.test(revision) || revision.includes('..')) {
        throw invalid(`${field}.revision is invalid`, `${field}.revision`);
    }
    const pinned = value.commit !== undefined || value.size !== undefined || value.sha256 !== undefined;
    if (requirePin || pinned) {
        if (!COMMIT_RE.test(String(value.commit || ''))) throw invalid(`${field}.commit must be a 40-hex commit`, `${field}.commit`);
        if (!Number.isSafeInteger(value.size) || value.size <= 0) throw invalid(`${field}.size must be the file size in bytes`, `${field}.size`);
        if (!SHA256_RE.test(String(value.sha256 || ''))) throw invalid(`${field}.sha256 must be the LFS sha256`, `${field}.sha256`);
    }
    return Object.freeze({
        type: 'huggingface',
        repo: value.repo,
        file: value.file,
        revision,
        ...(pinned || requirePin ? { commit: value.commit, size: value.size, sha256: value.sha256 } : {}),
        ...(value.quantization ? { quantization: optionalText(value.quantization, `${field}.quantization`, 64) } : {}),
    });
}

// The files a Hugging Face snapshot takes: top-level model files only. No
// subdirectories (gpt-oss-20b keeps a second copy of its weights in
// original/), no Python (remote code is never trusted), and no other weight
// formats (.bin, .pt, .gguf, .onnx).
const SNAPSHOT_FILE_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\.(json|safetensors|txt|model|tiktoken|jinja)$/;
const MAX_SNAPSHOT_FILES = 200;

export function isSnapshotFile(name) {
    return typeof name === 'string' && SNAPSHOT_FILE_RE.test(name) && !name.includes('..');
}

/**
 * A Hugging Face snapshot: every top-level model file of a repository at one
 * commit, each pinned by size and by its LFS sha256 or, for a small file kept
 * in git, its git blob oid. `size` is the sum of the files.
 */
export function validateHfSnapshotSource(value, field, { requirePin = false } = {}) {
    if (!plainObject(value)) throw invalid(`${field} must be an object`, field);
    if (value.type !== 'hf-snapshot') throw invalid(`${field}.type must be hf-snapshot`, `${field}.type`);
    onlyKeys(value, ['type', 'repo', 'revision', 'commit', 'files', 'size', 'quantization'], field);
    if (typeof value.repo !== 'string' || !HF_REPO_RE.test(value.repo)) {
        throw invalid(`${field}.repo must be a Hugging Face repository id (owner/name)`, `${field}.repo`);
    }
    const revision = value.revision ?? 'main';
    if (typeof revision !== 'string' || !HF_REVISION_RE.test(revision) || revision.includes('..')) {
        throw invalid(`${field}.revision is invalid`, `${field}.revision`);
    }
    const pinned = value.commit !== undefined || value.files !== undefined;
    const base = {
        type: 'hf-snapshot',
        repo: value.repo,
        revision,
        ...(value.quantization ? { quantization: optionalText(value.quantization, `${field}.quantization`, 64) } : {}),
    };
    if (!requirePin && !pinned) {
        if (value.size !== undefined) throw invalid(`${field}.size is computed from the pinned files`, `${field}.size`);
        return Object.freeze(base);
    }
    if (!COMMIT_RE.test(String(value.commit || ''))) throw invalid(`${field}.commit must be a 40-hex commit`, `${field}.commit`);
    if (!Array.isArray(value.files) || value.files.length === 0 || value.files.length > MAX_SNAPSHOT_FILES) {
        throw invalid(`${field}.files must list 1-${MAX_SNAPSHOT_FILES} files`, `${field}.files`);
    }
    const seen = new Set();
    const files = value.files.map((file, index) => {
        const at = `${field}.files[${index}]`;
        if (!plainObject(file)) throw invalid(`${at} must be an object`, at);
        onlyKeys(file, ['path', 'size', 'sha256', 'gitOid'], at);
        if (!isSnapshotFile(file.path)) {
            throw invalid(`${at}.path must be a top-level model file (.json, .safetensors, .txt, .model, .tiktoken or .jinja)`, `${at}.path`);
        }
        if (seen.has(file.path)) throw invalid(`${field}.files lists ${file.path} twice`, at);
        seen.add(file.path);
        if (!Number.isSafeInteger(file.size) || file.size <= 0) throw invalid(`${at}.size must be the file size in bytes`, `${at}.size`);
        if (file.sha256 === undefined && file.gitOid === undefined) {
            throw invalid(`${at} needs its sha256 or gitOid`, at);
        }
        if (file.sha256 !== undefined && !SHA256_RE.test(String(file.sha256))) throw invalid(`${at}.sha256 must be 64 hex`, `${at}.sha256`);
        if (file.gitOid !== undefined && !GIT_OID_RE.test(String(file.gitOid))) throw invalid(`${at}.gitOid must be 40 hex`, `${at}.gitOid`);
        return Object.freeze({
            path: file.path,
            size: file.size,
            ...(file.sha256 !== undefined ? { sha256: file.sha256 } : { gitOid: file.gitOid }),
        });
    });
    if (!files.some((file) => file.path.endsWith('.safetensors'))) {
        throw invalid(`${field}.files has no .safetensors file; vLLM and TabbyAPI read their weights from safetensors files`, `${field}.files`);
    }
    const size = files.reduce((sum, file) => sum + file.size, 0);
    if (value.size !== undefined && value.size !== size) {
        throw invalid(`${field}.size must equal the sum of the file sizes (${size})`, `${field}.size`);
    }
    return Object.freeze({ ...base, commit: value.commit, files: Object.freeze(files), size });
}

export function validateOllamaSource(value, field) {
    if (!plainObject(value)) throw invalid(`${field} must be an object`, field);
    if (value.type !== 'ollama') throw invalid(`${field}.type must be ollama`, `${field}.type`);
    onlyKeys(value, ['type', 'tag', 'manifestDigest', 'size'], field);
    if (typeof value.tag !== 'string' || !OLLAMA_TAG_RE.test(value.tag)) {
        throw invalid(`${field}.tag must be an Ollama library tag such as gpt-oss:20b`, `${field}.tag`);
    }
    if (value.manifestDigest !== undefined && !/^sha256:[0-9a-f]{64}$/.test(String(value.manifestDigest))) {
        throw invalid(`${field}.manifestDigest must be sha256:<64 hex>`, `${field}.manifestDigest`);
    }
    if (value.size !== undefined && (!Number.isSafeInteger(value.size) || value.size <= 0)) {
        throw invalid(`${field}.size must be a positive byte count`, `${field}.size`);
    }
    return Object.freeze({
        type: 'ollama',
        tag: value.tag,
        ...(value.manifestDigest ? { manifestDigest: value.manifestDigest } : {}),
        ...(value.size ? { size: value.size } : {}),
    });
}

/**
 * The weight formats a source may be given for, and the source type each
 * takes. A runner reads exactly one format (its `weightFormat`).
 */
export const WEIGHT_FORMATS = Object.freeze({
    gguf: Object.freeze({ sourceType: 'huggingface', label: 'GGUF file', validate: validateHuggingFaceSource }),
    ollama: Object.freeze({ sourceType: 'ollama', label: 'Ollama tag', validate: (value, field) => validateOllamaSource(value, field) }),
    // Hugging Face snapshots: safetensors for vLLM, EXL3 for TabbyAPI.
    hf: Object.freeze({ sourceType: 'hf-snapshot', label: 'Hugging Face snapshot', validate: validateHfSnapshotSource }),
    exl3: Object.freeze({ sourceType: 'hf-snapshot', label: 'EXL3 snapshot', validate: validateHfSnapshotSource }),
});

function validateMemory(value) {
    if (value === undefined || value === null) return undefined;
    if (!plainObject(value)) throw invalid('memory must be an object', 'memory');
    onlyKeys(value, ['layers', 'nonExpertBytes', 'expertBytesPerLayer', 'kvBytesPerToken', 'fixedKvBytes', 'embeddingBytes'], 'memory');
    return Object.freeze({
        layers: optionalInteger(value.layers, 'memory.layers', 1, 1024),
        nonExpertBytes: optionalInteger(value.nonExpertBytes, 'memory.nonExpertBytes', 0, 2 ** 50),
        expertBytesPerLayer: optionalInteger(value.expertBytesPerLayer, 'memory.expertBytesPerLayer', 0, 2 ** 45),
        kvBytesPerToken: optionalInteger(value.kvBytesPerToken, 'memory.kvBytesPerToken', 1, 2 ** 30),
        fixedKvBytes: optionalInteger(value.fixedKvBytes, 'memory.fixedKvBytes', 0, 2 ** 40),
        // The input embedding a runner keeps in system RAM (ExLlamaV3 does).
        embeddingBytes: optionalInteger(value.embeddingBytes, 'memory.embeddingBytes', 0, 2 ** 40),
    });
}

/**
 * Validate one catalog or registry entry. Seed entries must pin every
 * Hugging Face source (commit, size, sha256); user entries are pinned when
 * they are added, by resolving the revision through the Hugging Face API.
 */
export function validateModel(value, { seed = false } = {}) {
    if (!plainObject(value)) throw invalid('A model must be an object', 'model');
    // `seed` may come back from an earlier validation; it is never taken from input.
    onlyKeys(value, [
        'id', 'displayName', 'description', 'license', 'architecture', 'totalParams', 'activeParams',
        'contextLength', 'requiresJinja', 'sources', 'memory', 'recommended', 'validated', 'seed',
    ], 'model');
    if (typeof value.id !== 'string' || !ID_RE.test(value.id)) {
        throw invalid('id must be 2-64 lowercase letters, digits, dot, dash or underscore', 'id');
    }
    if (!plainObject(value.sources) || Object.keys(value.sources).length === 0) {
        throw invalid('sources must name at least one weight format', 'sources');
    }
    const sources = {};
    for (const [format, source] of Object.entries(value.sources)) {
        const field = `sources.${format}`;
        if (!Object.hasOwn(WEIGHT_FORMATS, format)) {
            throw invalid(`sources has an unknown weight format '${format}'; use one of ${Object.keys(WEIGHT_FORMATS).join(', ')}`, field);
        }
        sources[format] = WEIGHT_FORMATS[format].validate(source, field, { requirePin: seed });
    }
    const architecture = value.architecture ?? 'dense';
    if (!['moe', 'dense'].includes(architecture)) throw invalid('architecture must be moe or dense', 'architecture');
    if (value.recommended !== undefined && !plainObject(value.recommended)) {
        throw invalid('recommended must be an object keyed by runner', 'recommended');
    }
    if (value.validated !== undefined && !plainObject(value.validated)) {
        throw invalid('validated must be an object keyed by runner', 'validated');
    }
    return Object.freeze({
        id: value.id,
        displayName: optionalText(value.displayName, 'displayName', 120) || value.id,
        description: optionalText(value.description, 'description'),
        license: optionalText(value.license, 'license', 80),
        architecture,
        totalParams: optionalText(value.totalParams, 'totalParams', 16),
        activeParams: optionalText(value.activeParams, 'activeParams', 16),
        contextLength: optionalInteger(value.contextLength, 'contextLength', 512, 2 ** 22),
        requiresJinja: value.requiresJinja === true,
        sources: Object.freeze(sources),
        memory: validateMemory(value.memory),
        recommended: value.recommended ? structuredClone(value.recommended) : {},
        validated: value.validated ? structuredClone(value.validated) : {},
        seed,
    });
}

/**
 * Bring a registry entry written by the v1 controller (sources keyed by
 * runner id) to schema v2 (keyed by weight format). Idempotent, and it never
 * throws: anything it cannot map is left for validation to refuse, so the
 * entry is hidden but kept rather than lost.
 */
export function migrateModelEntry(entry) {
    if (!plainObject(entry) || !plainObject(entry.sources)) return entry;
    const sources = {};
    for (const [key, source] of Object.entries(entry.sources)) {
        if (!V1_GGUF_KEYS.includes(key)) sources[key] = source;
    }
    if (!sources.gguf) {
        const legacy = V1_GGUF_KEYS.map((key) => entry.sources[key]).find((source) => source?.type === 'huggingface');
        if (legacy) sources.gguf = legacy;
    }
    const migrated = { ...entry, sources };
    for (const field of ['recommended', 'validated']) {
        if (!plainObject(entry[field])) continue;
        migrated[field] = Object.fromEntries(Object.entries(entry[field]).filter(([runnerId]) => !REMOVED_RUNNERS.includes(runnerId)));
    }
    return migrated;
}

export function loadSeedCatalog(file = path.join(import.meta.dirname, '..', '..', 'catalog', 'models.json')) {
    const document = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (document.schema !== CATALOG_SCHEMA || !Array.isArray(document.models)) {
        throw new Error(`Unsupported catalog ${file}`);
    }
    const models = document.models.map((entry) => validateModel(entry, { seed: true }));
    const ids = new Set();
    for (const model of models) {
        if (ids.has(model.id)) throw new Error(`Duplicate catalog id ${model.id}`);
        ids.add(model.id);
    }
    return Object.freeze(models);
}

/**
 * Seed entries first, then valid user entries whose ids do not shadow a seed
 * or an earlier user entry (two can exist after a downgrade hid one).
 */
export function mergeCatalog(seed, registry = []) {
    const seenIds = new Set(seed.map((model) => model.id));
    const user = [];
    for (const entry of registry) {
        try {
            const model = validateModel(entry, { seed: false });
            if (seenIds.has(model.id)) continue;
            seenIds.add(model.id);
            user.push(model);
        } catch {
            // An invalid persisted entry is skipped, never allowed to break the catalog.
        }
    }
    return Object.freeze([...seed, ...user]);
}
