// Model catalog: the read-only seed shipped with the agent plus the user
// registry persisted under /data. Every entry is validated here; the same
// rules apply to seed entries (at load) and to user entries (at add/update).

import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';

const ID_RE = /^[a-z0-9][a-z0-9._-]{1,63}$/;
const HF_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const HF_FILE_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
const HF_REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;
const COMMIT_RE = /^[0-9a-f]{40}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const OLLAMA_TAG_RE = /^[a-z0-9][a-z0-9._-]{0,63}(?:\/[a-z0-9][a-z0-9._-]{0,63})?(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$/;
const TEXT_MAX = 400;

export const SOURCE_RUNNERS = Object.freeze({
    huggingface: ['llama.cpp', 'lmstudio'],
    ollama: ['ollama'],
});

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
    onlyKeys(value, ['type', 'repo', 'file', 'revision', 'commit', 'size', 'sha256', 'quantization'], field);
    if (value.type !== 'huggingface') throw invalid(`${field}.type must be huggingface`, `${field}.type`);
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

export function validateOllamaSource(value, field) {
    if (!plainObject(value)) throw invalid(`${field} must be an object`, field);
    onlyKeys(value, ['type', 'tag', 'manifestDigest', 'size'], field);
    if (value.type !== 'ollama') throw invalid(`${field}.type must be ollama`, `${field}.type`);
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

function validateMemory(value) {
    if (value === undefined || value === null) return undefined;
    if (!plainObject(value)) throw invalid('memory must be an object', 'memory');
    onlyKeys(value, ['layers', 'nonExpertBytes', 'expertBytesPerLayer', 'kvBytesPerToken', 'fixedKvBytes'], 'memory');
    return Object.freeze({
        layers: optionalInteger(value.layers, 'memory.layers', 1, 1024),
        nonExpertBytes: optionalInteger(value.nonExpertBytes, 'memory.nonExpertBytes', 0, 2 ** 50),
        expertBytesPerLayer: optionalInteger(value.expertBytesPerLayer, 'memory.expertBytesPerLayer', 0, 2 ** 45),
        kvBytesPerToken: optionalInteger(value.kvBytesPerToken, 'memory.kvBytesPerToken', 1, 2 ** 30),
        fixedKvBytes: optionalInteger(value.fixedKvBytes, 'memory.fixedKvBytes', 0, 2 ** 40),
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
        throw invalid('sources must name at least one runner source', 'sources');
    }
    const sources = {};
    for (const [runnerId, source] of Object.entries(value.sources)) {
        const field = `sources.${runnerId}`;
        if (runnerId === 'llama.cpp' || runnerId === 'lmstudio') {
            sources[runnerId] = validateHuggingFaceSource(source, field, { requirePin: seed });
        } else if (runnerId === 'ollama') {
            sources[runnerId] = validateOllamaSource(source, field);
        } else if (runnerId === 'vllm') {
            sources[runnerId] = validateHuggingFaceSource(source, field, { requirePin: seed });
        } else {
            throw invalid(`sources has an unknown runner '${runnerId}'`, field);
        }
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

export function loadSeedCatalog(file = path.join(import.meta.dirname, '..', '..', 'catalog', 'models.json')) {
    const document = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (document.schema !== 'local-llm.catalog/v1' || !Array.isArray(document.models)) {
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

/** Seed entries first, then valid user entries whose ids do not shadow a seed. */
export function mergeCatalog(seed, registry = []) {
    const seedIds = new Set(seed.map((model) => model.id));
    const user = [];
    for (const entry of registry) {
        try {
            const model = validateModel(entry, { seed: false });
            if (!seedIds.has(model.id)) user.push(model);
        } catch {
            // An invalid persisted entry is skipped, never allowed to break the catalog.
        }
    }
    return Object.freeze([...seed, ...user]);
}
