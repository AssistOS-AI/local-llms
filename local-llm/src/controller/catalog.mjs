// Model catalog: the read-only seed shipped with the agent plus the user
// registry persisted under /data. Every entry is validated here; seed entries
// (at load) and user entries (at add/update) share the shape, but only the
// seed may certify measurements (below).
//
// Schema v3 keys a model's sources by weight format (every runner declares
// the format it reads, so runners that read the same GGUF file share one
// download), keys `recommended` and `validated` by hardware profile and then
// by runner (DS005), says in which profiles the model is offered, and carries
// the unified profile's measured envelope. Entries of earlier schemas are not
// migrated: a registry entry that is not valid v3 is hidden.
//
// Trust boundary (DS005): `unified.envelope` admits runs on unified memory and
// `validated` says what was measured, so both come only from the seed: the
// shipped catalog or the operator's LOCAL_LLM_CATALOG_FILE, whose provenance
// is controlled outside the model-management API. A user entry (model add or
// update) that carries either is refused, and a stored one is not offered.

import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import { PROFILES, UNIFIED_LOAD_MODES } from './profiles.mjs';

export const CATALOG_SCHEMA = 'local-llm.catalog/v3';


const ID_RE = /^[a-z0-9][a-z0-9._-]{1,63}$/;
export const HF_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
export const HF_FILE_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
export const HF_REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;
const COMMIT_RE = /^[0-9a-f]{40}$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
const GIT_OID_RE = /^[0-9a-f]{40}$/;
const OLLAMA_TAG_RE = /^[a-z0-9][a-z0-9._-]{0,63}(?:\/[a-z0-9][a-z0-9._-]{0,63})?(?::[A-Za-z0-9][A-Za-z0-9._-]{0,63})?$/;
const TEXT_MAX = 400;
// Where an entry's sizing fields (contextLength, memory.layers, memory.kvBytesPerToken) came from (DS002): the GGUF header
// read at lookup (read again from the verified file after the download), the pinned config.json of a snapshot, or typed by hand.
export const SIZING_SOURCES = Object.freeze(['gguf-header', 'config.json', 'manual']);

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

function requiredInteger(value, field, min, max) {
    if (value === undefined || value === null) throw invalid(`${field} is required`, field);
    return optionalInteger(value, field, min, max);
}

function onlyKeys(value, allowed, field) {
    for (const key of Object.keys(value)) {
        if (!allowed.includes(key)) throw invalid(`${field} has an unsupported field '${key}'`, `${field}.${key}`);
    }
}

// llama.cpp's split GGUF names: <prefix>-<k>-of-<n>.gguf, k and n five digits.
const SPLIT_GGUF_RE = /^(.+)-(\d{5})-of-(\d{5})\.gguf$/;
const MAX_SHARDS = 64;

/** The shard number and count of a split GGUF file name, or null for a single file. */
export function splitGgufName(file) {
    const match = SPLIT_GGUF_RE.exec(String(file || ''));
    if (!match) return null;
    const index = Number(match[2]);
    const count = Number(match[3]);
    return count > 1 && index >= 1 && index <= count ? { prefix: match[1], index, count } : null;
}

/** Every shard's file name of a split GGUF, in canonical order, from its first shard's name. */
export function splitGgufFiles(firstFile) {
    const split = splitGgufName(firstFile);
    if (!split || split.index !== 1) return null;
    const pad = (value) => String(value).padStart(5, '0');
    return Array.from({ length: split.count }, (_, i) => `${split.prefix}-${pad(i + 1)}-of-${pad(split.count)}.gguf`);
}

// A split GGUF (DS002): still the gguf weight format; `file` is the first
// shard, which llama.cpp opens, and `shards` pins every shard by size and
// sha256, in canonical order, in one directory. `size` is their sum.
function validateShards(value, field) {
    const expected = splitGgufFiles(value.file);
    if (!expected) throw invalid(`${field}.file must be the first shard (…-00001-of-0000N.gguf) when shards are given`, `${field}.file`);
    if (expected.length > MAX_SHARDS) throw invalid(`${field} has more than ${MAX_SHARDS} shards`, `${field}.shards`);
    if (!Array.isArray(value.shards) || value.shards.length !== expected.length) {
        throw invalid(`${field}.shards must list all ${expected.length} shards`, `${field}.shards`);
    }
    const shards = value.shards.map((shard, index) => {
        const at = `${field}.shards[${index}]`;
        if (!plainObject(shard)) throw invalid(`${at} must be an object`, at);
        onlyKeys(shard, ['file', 'size', 'sha256'], at);
        if (shard.file !== expected[index]) throw invalid(`${at}.file must be ${expected[index]}`, `${at}.file`);
        if (!Number.isSafeInteger(shard.size) || shard.size <= 0) throw invalid(`${at}.size must be the file size in bytes`, `${at}.size`);
        if (!SHA256_RE.test(String(shard.sha256 || ''))) throw invalid(`${at}.sha256 must be the LFS sha256`, `${at}.sha256`);
        return Object.freeze({ file: shard.file, size: shard.size, sha256: shard.sha256 });
    });
    const size = shards.reduce((sum, shard) => sum + shard.size, 0);
    if (value.size !== undefined && value.size !== size) throw invalid(`${field}.size must equal the sum of the shard sizes (${size})`, `${field}.size`);
    return { shards: Object.freeze(shards), size };
}

export function validateHuggingFaceSource(value, field, { requirePin = false } = {}) {
    if (!plainObject(value)) throw invalid(`${field} must be an object`, field);
    if (value.type !== 'huggingface') throw invalid(`${field}.type must be huggingface`, `${field}.type`);
    onlyKeys(value, ['type', 'repo', 'file', 'revision', 'commit', 'size', 'sha256', 'shards', 'quantization'], field);
    if (typeof value.repo !== 'string' || !HF_REPO_RE.test(value.repo)) {
        throw invalid(`${field}.repo must be a Hugging Face repository id (owner/name)`, `${field}.repo`);
    }
    const segments = typeof value.file === 'string' ? value.file.split('/') : [];
    if (!segments.length || segments.length > 4 || !segments.every((segment) => HF_FILE_SEGMENT_RE.test(segment))
        || !value.file.endsWith('.gguf')) {
        throw invalid(`${field}.file must be a .gguf file in the repository (the first shard of a split GGUF)`, `${field}.file`);
    }
    const revision = value.revision ?? 'main';
    if (typeof revision !== 'string' || !HF_REVISION_RE.test(revision) || revision.includes('..')) {
        throw invalid(`${field}.revision is invalid`, `${field}.revision`);
    }
    const split = splitGgufName(value.file);
    // Bounded before anything is resolved: pinning asks Hugging Face once per shard.
    if (split && split.count > MAX_SHARDS) throw invalid(`${field}.file names ${split.count} shards; at most ${MAX_SHARDS} are supported`, `${field}.file`);
    if (split && split.index !== 1) {
        throw invalid(`${field}.file names shard ${split.index} of ${split.count}; name the first shard of a split GGUF`, `${field}.file`);
    }
    const quantization = value.quantization ? { quantization: optionalText(value.quantization, `${field}.quantization`, 64) } : {};
    if (split) {
        // Pinned: the commit and every shard. Unpinned (a user entry before Add
        // model resolves it): the first shard's name only.
        const pinnedSplit = value.commit !== undefined || value.shards !== undefined || value.size !== undefined;
        if (value.sha256 !== undefined) throw invalid(`${field}.sha256 is per shard for a split GGUF`, `${field}.sha256`);
        if (!requirePin && !pinnedSplit) return Object.freeze({ type: 'huggingface', repo: value.repo, file: value.file, revision, ...quantization });
        if (!COMMIT_RE.test(String(value.commit || ''))) throw invalid(`${field}.commit must be a 40-hex commit`, `${field}.commit`);
        const { shards, size } = validateShards(value, field);
        return Object.freeze({ type: 'huggingface', repo: value.repo, file: value.file, revision, commit: value.commit, size, shards, ...quantization });
    }
    if (value.shards !== undefined) throw invalid(`${field}.shards is only for a split GGUF (…-00001-of-0000N.gguf)`, `${field}.shards`);
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
        ...quantization,
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
 * Only a seed entry may carry a unified envelope or `validated` labels.
 */
export function validateModel(value, { seed = false } = {}) {
    if (!plainObject(value)) throw invalid('A model must be an object', 'model');
    // `seed` may come back from an earlier validation; it is never taken from input.
    onlyKeys(value, [
        'id', 'displayName', 'description', 'license', 'architecture', 'totalParams', 'activeParams',
        'contextLength', 'requiresJinja', 'mtp', 'profiles', 'sources', 'memory', 'recommended', 'validated', 'unified', 'seed',
        'sizingSource',
    ], 'model');
    // A capability of the weights (a multi-token-prediction head), not a
    // measurement, so a user entry may declare it too. Without it, MTP is
    // refused for the model (DS005).
    if (value.mtp !== undefined && typeof value.mtp !== 'boolean') throw invalid('mtp must be true or false', 'mtp');
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
    const profiles = value.profiles === undefined ? [...PROFILES] : value.profiles;
    if (!Array.isArray(profiles) || profiles.length === 0 || profiles.some((profile) => !PROFILES.includes(profile))
        || new Set(profiles).size !== profiles.length) {
        throw invalid(`profiles must list one or more of ${PROFILES.join(', ')}`, 'profiles');
    }
    const recommended = validatePerProfile(value.recommended, 'recommended', (entry, field) => {
        if (!plainObject(entry)) throw invalid(`${field} must be an object of parameters`, field);
        if (entry.mtp === true && value.mtp !== true) {
            throw invalid(`${field}.mtp recommends MTP for a model that does not declare a multi-token-prediction head (mtp)`, `${field}.mtp`);
        }
        return entry;
    });
    const validated = validatePerProfile(value.validated, 'validated', (entry, field) => optionalText(entry, field, 600));
    if (!seed) assertNoCertification(value, validated);
    const sizingSource = validateSizingSource(value.sizingSource, sources);
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
        mtp: value.mtp === true,
        sources: Object.freeze(sources),
        memory: validateMemory(value.memory),
        profiles: Object.freeze([...profiles]),
        recommended,
        validated,
        unified: validateUnified(value.unified),
        ...(sizingSource ? { sizingSource } : {}),
        seed,
    });
}

// The sizing source must match a source the entry has, so its warning and the re-read after a download never describe
// a file the entry does not use.
function validateSizingSource(value, sources) {
    if (value === undefined || value === null) return undefined;
    if (!SIZING_SOURCES.includes(value)) throw invalid(`sizingSource must be one of ${SIZING_SOURCES.join(', ')}`, 'sizingSource');
    if (value === 'gguf-header' && !sources.gguf) throw invalid('sizingSource gguf-header needs a gguf source', 'sizingSource');
    if (value === 'config.json' && !sources.hf && !sources.exl3) throw invalid('sizingSource config.json needs an hf or exl3 source', 'sizingSource');
    return value;
}

// A user entry cannot certify itself. Stored user entries carry the empty
// forms (`unified: null`, `validated: {}`) that validation writes back, so
// only an actual envelope or label is refused.
function assertNoCertification(value, validated) {
    const trusted = 'only the trusted seed catalog (the shipped catalog or the operator\'s LOCAL_LLM_CATALOG_FILE) may provide it';
    if (value.unified !== undefined && value.unified !== null) {
        throw invalid(`unified is a measured envelope; ${trusted}`, 'unified');
    }
    const labelled = Object.entries(validated).find(([, byRunner]) => Object.keys(byRunner).length > 0);
    if (labelled) throw invalid(`validated records measurements; ${trusted}`, `validated.${labelled[0]}`);
}

// `recommended` and `validated`: { <profile>: { <runner id>: value } }.
function validatePerProfile(value, field, validateEntry) {
    if (value === undefined || value === null) return Object.freeze({});
    if (!plainObject(value)) throw invalid(`${field} must be an object keyed by profile`, field);
    const out = {};
    for (const [profile, byRunner] of Object.entries(value)) {
        if (!PROFILES.includes(profile)) throw invalid(`${field} has an unknown profile '${profile}'`, `${field}.${profile}`);
        if (!plainObject(byRunner)) throw invalid(`${field}.${profile} must be an object keyed by runner`, `${field}.${profile}`);
        out[profile] = Object.freeze(Object.fromEntries(Object.entries(byRunner)
            .map(([runnerId, entry]) => [runnerId, validateEntry(entry, `${field}.${profile}.${runnerId}`)])));
    }
    return Object.freeze(structuredClone(out));
}

const ENVELOPE_KEYS = Object.freeze(['runner', 'loadMode', 'maxCtx', 'maxParallel', 'mtp', 'bufferBytes', 'transientBytes', 'measured']);

/**
 * The unified profile's measured envelope (DS005): rectangles of context and
 * parallel slots, each measured at its corner in one load mode, with the
 * deterministic buffer bytes llama.cpp logged there and the measured transient
 * margin beyond them. A mode measured separately is a rectangle of its own.
 * A model without an envelope, or parameters outside it, are sized by estimate.
 */
function validateUnified(value) {
    if (value === undefined || value === null) return null;
    if (!plainObject(value)) throw invalid('unified must be an object', 'unified');
    onlyKeys(value, ['envelope'], 'unified');
    if (!Array.isArray(value.envelope) || value.envelope.length === 0 || value.envelope.length > 16) {
        throw invalid('unified.envelope must list 1-16 measured rectangles', 'unified.envelope');
    }
    const envelope = value.envelope.map((entry, index) => {
        const at = `unified.envelope[${index}]`;
        if (!plainObject(entry)) throw invalid(`${at} must be an object`, at);
        onlyKeys(entry, ENVELOPE_KEYS, at);
        if (typeof entry.runner !== 'string' || !entry.runner) throw invalid(`${at}.runner must name a runner`, `${at}.runner`);
        if (typeof entry.mtp !== 'boolean') throw invalid(`${at}.mtp must be true or false`, `${at}.mtp`);
        if (!UNIFIED_LOAD_MODES.includes(entry.loadMode)) {
            throw invalid(`${at}.loadMode must be the load mode it was measured with (${UNIFIED_LOAD_MODES.join(' or ')})`, `${at}.loadMode`);
        }
        return Object.freeze({
            runner: entry.runner,
            loadMode: entry.loadMode,
            maxCtx: requiredInteger(entry.maxCtx, `${at}.maxCtx`, 512, 2 ** 22),
            maxParallel: requiredInteger(entry.maxParallel, `${at}.maxParallel`, 1, 16),
            mtp: entry.mtp,
            bufferBytes: requiredInteger(entry.bufferBytes, `${at}.bufferBytes`, 1, 2 ** 45),
            transientBytes: requiredInteger(entry.transientBytes, `${at}.transientBytes`, 0, 2 ** 40),
            measured: optionalText(entry.measured, `${at}.measured`, 600),
        });
    });
    return Object.freeze({ envelope: Object.freeze(envelope) });
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
/**
 * Registry entries the catalog cannot read (an earlier schema, or invalid) and
 * valid ones a seed's id hides (a seed added after the entry was), with the
 * reason, so the overview can say so instead of hiding them silently.
 */
export function unsupportedRegistryEntries(registry = [], seed = []) {
    const seedIds = new Set(seed.map((model) => model.id));
    const unsupported = [];
    for (const entry of registry) {
        try {
            const model = validateModel(entry, { seed: false });
            if (seedIds.has(model.id)) {
                unsupported.push({
                    id: model.id,
                    // A seed is offered only in the profiles it lists, so this says nothing about where it is offered.
                    reason: 'its id is also the id of a model that ships with this agent, which takes the id, so this entry is not offered; remove it, or add the model again under another id',
                });
            }
        } catch (error) {
            unsupported.push({ id: typeof entry?.id === 'string' ? entry.id.slice(0, 64) : null, reason: error.message });
        }
    }
    return unsupported;
}

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
