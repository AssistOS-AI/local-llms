// The Hugging Face lookup behind the Add model form (DS002): what a
// repository offers at a revision, so an admin picks a file instead of typing
// its name, and the sizing the memory estimate needs, read from the file's own
// GGUF header or from a snapshot's config.json. It reads metadata and, for one
// GGUF file, at most 32 MiB from the start of the file; it writes nothing and
// downloads no weights.
//
// The token (`HF_TOKEN`) goes to the Hugging Face origin only. Every request,
// the metadata calls and the ranged read alike, follows redirects by hand and
// drops `Authorization` from any hop whose origin is not the Hugging Face one
// (the file hosts' CDN is another origin), and a tree page named by a `Link`
// header on another origin is never followed. Nothing here relies on the
// fetch implementation's own redirect handling.

import crypto from 'node:crypto';

import { LocalLlmError } from '../errors.mjs';
import { HF_FILE_SEGMENT_RE, isSnapshotFile, splitGgufFiles, splitGgufName, validateHuggingFaceSource } from './catalog.mjs';
import {
    DownloadError,
    MAX_TREE_PAGES,
    METADATA_TIMEOUT_MS,
    assertFile,
    assertRepo,
    assertRevision,
    authHeaders,
    fetchJson,
    lfsIdentity,
    nextTreePage,
    resolveCommit,
} from './downloader.mjs';
import { GGUF_LIMITS, createGgufHeaderReader, ggufSizing } from './ggufHeader.mjs';

const FORMATS = Object.freeze(['gguf', 'hf', 'exl3']);
const INPUT_KEYS = Object.freeze(['provider', 'repo', 'revision', 'format', 'file']);
const MAX_DIRECTORIES = 32;
const MAX_FILES = 500;
// Tree entries held while listing, whatever their kind, so a huge directory cannot grow memory.
const MAX_ENTRIES = 5000;
const MAX_REDIRECTS = 5;
// The tool's own timeout is 60 s; the lookup ends before it.
export const LOOKUP_TIMEOUT_MS = 55_000;
// A config.json is a few kilobytes; one beyond this is not read.
const CONFIG_MAX_BYTES = 1024 * 1024;
const REDIRECTS = new Set([301, 302, 303, 307, 308]);
const SHA1_RE = /^[a-f0-9]{40}$/;
const TOKEN_HINT = 'set a token with `ploinky var HF_TOKEN <token>` and restart local-llm';

function failure(code, message, details = undefined) {
    return new LocalLlmError(code, message, details);
}

function statusError(status, what) {
    if (status === 401 || status === 403) {
        return failure('access_denied', `Hugging Face refused access to ${what} (HTTP ${status}). For a gated or private repository, ${TOKEN_HINT}.`);
    }
    if (status === 404) return failure('not_found', `Hugging Face has no ${what}.`);
    return failure('lookup_failed', `Hugging Face answered HTTP ${status} for ${what}.`, { status });
}

// Every failure of a lookup is one of the documented codes; a stop of the agent is `shutting_down`.
function lookupError(error, signal) {
    if (error instanceof LocalLlmError) return error;
    if (signal?.aborted) return failure('shutting_down', 'The agent is restarting; look the model up again once it is back.');
    if (error instanceof DownloadError) {
        if (error.code === 'INVALID_SOURCE') return failure('invalid_request', error.message);
        if (error.code === 'NOT_FOUND') return failure('not_found', error.message);
        const status = error.details?.status;
        if (Number.isInteger(status)) return statusError(status, 'this repository, revision or file');
        return failure('lookup_failed', error.message, error.details?.reason ? { reason: error.details.reason } : undefined);
    }
    if (error?.name === 'TimeoutError' || error?.name === 'AbortError') {
        return failure('lookup_failed', 'The Hugging Face lookup took too long.', { reason: 'timeout' });
    }
    return failure('lookup_failed', 'The Hugging Face lookup failed.');
}

/** The lookup's input, checked before any request; a bad value is `invalid_request`. */
export function normalizeLookupInput(args) {
    if (args === null || typeof args !== 'object' || Array.isArray(args)) throw failure('invalid_request', 'The lookup needs a repo.');
    for (const key of Object.keys(args)) {
        if (!INPUT_KEYS.includes(key)) throw failure('invalid_request', `The lookup has no '${key}' field.`);
    }
    const format = args.format ?? 'gguf';
    if (!FORMATS.includes(format)) throw failure('invalid_request', `format must be one of ${FORMATS.join(', ')}.`);
    if (args.provider !== undefined && args.provider !== 'huggingface') throw failure('invalid_request', 'provider must be huggingface.');
    const revision = args.revision ?? 'main';
    const file = args.file;
    try {
        assertRepo(args.repo);
        assertRevision(revision);
        if (file !== undefined) {
            if (format !== 'gguf') throw failure('invalid_request', 'file applies to a GGUF lookup only.');
            if (typeof file !== 'string' || file.length > 512) throw failure('invalid_request', 'file must be a path of at most 512 characters.');
            assertFile(file);
            const split = splitGgufName(file);
            if (split && split.index !== 1) {
                throw failure('invalid_request', `file names shard ${split.index} of ${split.count}; name the first shard of a split GGUF.`);
            }
        }
    } catch (error) {
        throw error instanceof DownloadError ? failure('invalid_request', error.message) : error;
    }
    return { repo: args.repo, revision, format, ...(file === undefined ? {} : { file }) };
}

const encodePath = (value) => value.split('/').map(encodeURIComponent).join('/');

async function discard(response) {
    try {
        await response.body?.cancel();
    } catch {
        // The connection is being dropped anyway.
    }
}

/**
 * A fetch that follows redirects itself, up to five, and drops `Authorization`
 * from every request whose origin is not the Hugging Face one. Each request
 * also ends at `deadline`.
 */
export function originGuardedFetch({ fetchImpl, baseUrl, deadline }) {
    const home = new URL(baseUrl);
    return async function guarded(url, init = {}) {
        let current = new URL(url);
        for (let hops = 0; ; hops += 1) {
            const headers = { ...init.headers };
            if (current.origin !== home.origin) {
                for (const name of Object.keys(headers)) {
                    if (name.toLowerCase() === 'authorization') delete headers[name];
                }
            }
            const signal = init.signal ? AbortSignal.any([init.signal, deadline]) : deadline;
            const response = await fetchImpl(current.href, { ...init, headers, signal, redirect: 'manual' });
            if (!REDIRECTS.has(response.status)) return response;
            const location = response.headers?.get?.('location');
            await discard(response);
            if (!location || hops >= MAX_REDIRECTS) throw failure('lookup_failed', 'Hugging Face redirected the request too often, or without saying where.');
            const next = new URL(location, current);
            if (!['http:', 'https:'].includes(next.protocol) || (current.protocol === 'https:' && next.protocol !== 'https:')) {
                throw failure('lookup_failed', 'Hugging Face redirected the request to an address that is not https.');
            }
            current = next;
        }
    };
}

function gatedOf(info) {
    return info.gated === false || info.gated === 'auto' || info.gated === 'manual' ? info.gated : null;
}

// The licence id from the model card, else from a `license:` tag; plain text only.
function licenseOf(info) {
    const card = info.cardData?.license;
    const fromTags = Array.isArray(info.tags)
        ? info.tags.filter((tag) => typeof tag === 'string' && tag.startsWith('license:')).map((tag) => tag.slice('license:'.length))
        : [];
    for (const candidate of [...(Array.isArray(card) ? card : [card]), ...fromTags]) {
        if (typeof candidate !== 'string') continue;
        const text = candidate.trim();
        if (text && text.length <= 80 && !/[\u0000-\u001f\u007f]/.test(text)) return text;
    }
    return null;
}

// One directory's entries, page by page: at most 50 pages for the whole lookup, same-origin next links only.
async function listTree(context, dir = '') {
    const { get, baseUrl, repo, commit, token, timeoutMs, budget } = context;
    const entries = [];
    let url = `${baseUrl}/api/models/${encodePath(repo)}/tree/${commit}${dir ? `/${encodePath(dir)}` : ''}`;
    while (url) {
        if (budget.pages >= MAX_TREE_PAGES) {
            budget.truncated = true;
            break;
        }
        budget.pages += 1;
        const { json, link } = await fetchJson(url, { token, fetchImpl: get, timeoutMs });
        if (!Array.isArray(json)) throw failure('lookup_failed', 'The Hugging Face tree response is not a list.');
        for (const entry of json) {
            if (entries.length >= MAX_ENTRIES) {
                budget.truncated = true;
                break;
            }
            entries.push(entry);
        }
        const next = nextTreePage(link, baseUrl);
        // A next page on another origin is not followed, so what was listed is not the whole directory.
        if (next === null && /rel="?next"?/.test(link ?? '')) budget.truncated = true;
        url = next;
    }
    return entries;
}

const QUANTIZATION_RE = /(?:^|[-_.])((?:UD-)?(?:IQ\d[A-Z0-9_]{0,30}|Q\d[A-Z0-9_]{0,30}|BF16|F16|F32|MXFP4[A-Z0-9_]{0,30}))(?=[-_.]|$)/i;

/** The quantization named in a GGUF file name (Q4_K_M, IQ3_XS, BF16 …), or null. */
export function quantizationOf(file) {
    const base = String(file).split('/').pop().replace(/-\d{5}-of-\d{5}\.gguf$/i, '').replace(/\.gguf$/i, '');
    const match = QUANTIZATION_RE.exec(base);
    return match ? match[1].toUpperCase() : null;
}

const byFile = (left, right) => (left.file < right.file ? -1 : left.file > right.file ? 1 : 0);

// The GGUF files the Add model form could add: single files, and split sets whose every shard is present, each
// listed once under its first shard. Every row is checked by the validator Add uses, so a listed file can be added.
function ggufRows(found, repo, commit) {
    const rows = [];
    for (const [file, identity] of found) {
        const split = splitGgufName(file);
        let source;
        let row;
        if (!split) {
            source = { type: 'huggingface', repo, file, revision: 'main', commit, size: identity.size, sha256: identity.sha256 };
            row = { file, size: identity.size, sha256: identity.sha256, gitOid: null, quantization: quantizationOf(file), shards: null };
        } else if (split.index === 1) {
            const shards = (splitGgufFiles(file) ?? []).map((name) => (found.has(name) ? { file: name, ...found.get(name) } : null));
            if (shards.length === 0 || shards.includes(null)) continue;
            const size = shards.reduce((sum, shard) => sum + shard.size, 0);
            source = { type: 'huggingface', repo, file, revision: 'main', commit, size, shards };
            row = { file, size, sha256: null, gitOid: null, quantization: quantizationOf(file), shards };
        } else {
            continue;
        }
        try {
            validateHuggingFaceSource(source, 'source');
        } catch {
            continue;
        }
        rows.push(row);
    }
    return rows.sort(byFile);
}

async function listGguf(context) {
    const { budget } = context;
    const found = new Map();
    const directories = [];
    const take = (entries, prefix) => {
        for (const entry of entries) {
            if (entry === null || typeof entry !== 'object' || typeof entry.path !== 'string') continue;
            if (entry.type === 'directory') {
                if (!prefix && HF_FILE_SEGMENT_RE.test(entry.path)) directories.push(entry.path);
                continue;
            }
            if (entry.type !== 'file' || !entry.path.endsWith('.gguf')) continue;
            // A directory's listing names only its own files.
            if (prefix && !entry.path.startsWith(`${prefix}/`)) continue;
            try {
                assertFile(entry.path);
            } catch {
                continue;
            }
            const identity = lfsIdentity(entry);
            if (!identity) continue;
            if (found.size >= MAX_FILES) budget.truncated = true;
            else found.set(entry.path, identity);
        }
    };
    take(await listTree(context), '');
    directories.sort();
    if (directories.length > MAX_DIRECTORIES) budget.truncated = true;
    // One level of subdirectories: larger quantizations are often kept in one.
    for (const dir of directories.slice(0, MAX_DIRECTORIES)) take(await listTree(context, dir), dir);
    return ggufRows(new Map([...found].sort(([left], [right]) => (left < right ? -1 : 1))), context.repo, context.commit);
}

// The top-level files a snapshot pins (the same filter `resolveHuggingFaceSnapshot` applies), each with its digest.
async function listSnapshot(context) {
    const rows = [];
    for (const entry of await listTree(context)) {
        if (entry === null || typeof entry !== 'object' || entry.type !== 'file' || !isSnapshotFile(entry.path)) continue;
        if (entry.lfs !== undefined && entry.lfs !== null) {
            const identity = lfsIdentity(entry);
            if (identity) rows.push({ file: entry.path, size: identity.size, sha256: identity.sha256, gitOid: null, quantization: null, shards: null });
            continue;
        }
        const oid = typeof entry.oid === 'string' ? entry.oid.toLowerCase() : '';
        if (SHA1_RE.test(oid) && Number.isSafeInteger(entry.size) && entry.size > 0) {
            rows.push({ file: entry.path, size: entry.size, sha256: null, gitOid: oid, quantization: null, shards: null });
        }
    }
    if (rows.length > MAX_FILES) context.budget.truncated = true;
    return rows.sort(byFile).slice(0, MAX_FILES);
}

function resolveUrl({ baseUrl, repo, commit }, file) {
    return `${baseUrl}/${encodePath(repo)}/resolve/${commit}/${encodePath(file)}`;
}

/**
 * The key-value section of the GGUF file, from a ranged read of its first 32 MiB that ends once the header is
 * parsed (the request is aborted, so the rest of the file is never sent).
 */
async function readGgufKeyValues(context, file) {
    const { get, token } = context;
    const stop = new AbortController();
    try {
        const response = await get(resolveUrl(context, file), {
            headers: { 'Accept-Encoding': 'identity', Range: `bytes=0-${GGUF_LIMITS.maxBytes - 1}`, ...authHeaders(token) },
            signal: stop.signal,
        });
        if (response.status === 416) throw failure('invalid_gguf', 'The file is empty.');
        if (response.status !== 200 && response.status !== 206) throw statusError(response.status, `the file ${file}`);
        if (!response.body) throw failure('lookup_failed', 'Hugging Face sent no content for the file.');
        const reader = createGgufHeaderReader();
        let readBytes = 0;
        for await (const chunk of response.body) {
            readBytes += chunk.length;
            // A server that ignores the range is cut off at the same bound.
            if (reader.push(chunk) === 'done' || readBytes > GGUF_LIMITS.maxBytes) break;
        }
        return { kv: reader.end().kv, readBytes };
    } finally {
        stop.abort();
    }
}

async function ggufSizingOf(context, file) {
    const { kv, readBytes } = await readGgufKeyValues(context, file);
    const sizing = ggufSizing(kv);
    return {
        contextLength: sizing.contextLength,
        architecture: sizing.architecture,
        memory: { layers: sizing.layers, kvBytesPerToken: sizing.kvBytesPerToken },
        source: 'gguf-header',
        readBytes,
        notes: sizing.notes,
    };
}

// config.json: read whole (it is small), checked against the digest the tree gave, then parsed.
async function readConfig(context, row) {
    const { get, token } = context;
    const response = await get(resolveUrl(context, row.file), { headers: { 'Accept-Encoding': 'identity', ...authHeaders(token) } });
    if (response.status !== 200) throw statusError(response.status, `the file ${row.file}`);
    if (!response.body) throw failure('lookup_failed', 'Hugging Face sent no content for config.json.');
    const parts = [];
    let total = 0;
    for await (const chunk of response.body) {
        total += chunk.length;
        if (total > row.size) throw failure('lookup_failed', 'config.json is longer than its pinned size.');
        parts.push(chunk);
    }
    const bytes = Buffer.concat(parts, total);
    // An LFS file is pinned by its sha256, a file kept in git by its blob oid (sha1 of "blob <size>\0" and the content).
    const digest = row.sha256
        ? crypto.createHash('sha256').update(bytes).digest('hex')
        : crypto.createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
    if (bytes.length !== row.size || digest !== (row.sha256 ?? row.gitOid)) {
        throw failure('lookup_failed', 'config.json does not match the digest pinned in the repository listing.');
    }
    try {
        const config = JSON.parse(bytes.toString('utf8'));
        if (config === null || typeof config !== 'object' || Array.isArray(config)) throw new Error('not an object');
        return { config, readBytes: total };
    } catch {
        throw failure('lookup_failed', 'config.json is not valid JSON.');
    }
}

function integerField(object, names, min, max) {
    for (const name of names) {
        const value = object[name];
        if (Number.isSafeInteger(value) && value >= min && value <= max) return value;
    }
    return null;
}

/** Layers, context, f16 KV bytes per token and dense or mixture-of-experts from a Transformers-style config.json. */
export function sizingFromConfig(config) {
    // Multimodal configs keep the language model's settings under text_config.
    const root = config.text_config && typeof config.text_config === 'object' && !Array.isArray(config.text_config)
        && Number.isSafeInteger(config.text_config.num_hidden_layers) ? config.text_config : config;
    const notes = [];
    const layers = integerField(root, ['num_hidden_layers', 'n_layer', 'num_layers'], 1, 1024);
    if (layers === null) notes.push('config.json has no layer count');
    const contextLength = integerField(root, ['max_position_embeddings', 'n_positions', 'max_sequence_length'], 512, 2 ** 22);
    if (contextLength === null) notes.push('config.json has no usable maximum context');
    const experts = integerField(root, ['num_local_experts', 'num_experts', 'n_routed_experts', 'moe_num_experts'], 0, 2 ** 20);
    let kvBytesPerToken = null;
    const heads = integerField(root, ['num_attention_heads', 'n_head'], 1, 2 ** 20);
    const latent = Object.keys(root).find((name) => /^(kv_lora_rank|q_lora_rank|mamba|ssm|linear_)/.test(name))
        ?? (Array.isArray(root.layer_types) && root.layer_types.some((type) => !['full_attention', 'sliding_attention'].includes(type)) ? 'layer_types' : null);
    if (layers !== null && latent) {
        notes.push(`${latent} marks a hybrid, recurrent or latent-attention model, so its KV cache is not sized from the head counts`);
    } else if (layers !== null && heads !== null) {
        const kvHeads = integerField(root, ['num_key_value_heads', 'num_kv_heads'], 1, 2 ** 20) ?? heads;
        const hidden = integerField(root, ['hidden_size', 'n_embd'], 1, 2 ** 24);
        const headSize = integerField(root, ['head_dim'], 1, 2 ** 20) ?? (hidden === null ? null : Math.floor(hidden / heads));
        if (headSize > 0) {
            // f16: two bytes per element, for the keys and the values.
            const bytes = layers * kvHeads * headSize * 2 * 2;
            if (bytes >= 1 && bytes <= 2 ** 30) kvBytesPerToken = bytes;
            else notes.push(`the KV cache size computed from config.json (${bytes}) is outside 1 to ${2 ** 30}`);
        } else {
            notes.push('config.json gives no head size');
        }
    } else if (layers !== null) {
        notes.push('config.json has no attention head count');
    }
    return { contextLength, architecture: experts > 0 ? 'moe' : 'dense', memory: { layers, kvBytesPerToken }, source: 'config.json', notes };
}

/**
 * Look a model up on Hugging Face. `args`: `{ repo, revision?, format?, file?, provider? }`. Returns the commit, the
 * gated status and licence, the files of the format, and the sizing (for one GGUF `file`, or a snapshot's
 * config.json). Options are the seams the controller and the tests use.
 */
export async function lookupHuggingFaceModel(args, {
    token = '',
    fetchImpl = globalThis.fetch,
    baseUrl = 'https://huggingface.co',
    timeoutMs = METADATA_TIMEOUT_MS,
    totalMs = LOOKUP_TIMEOUT_MS,
    signal = undefined,
} = {}) {
    const input = normalizeLookupInput(args);
    const deadline = AbortSignal.any([AbortSignal.timeout(totalMs), ...(signal ? [signal] : [])]);
    try {
        const get = originGuardedFetch({ fetchImpl, baseUrl, deadline });
        const { repo, revision, format } = input;
        const { json: info } = await fetchJson(`${baseUrl}/api/models/${encodePath(repo)}`, { token, fetchImpl: get, timeoutMs });
        if (info === null || typeof info !== 'object' || Array.isArray(info)) throw failure('lookup_failed', 'The Hugging Face model response is not an object.');
        const commit = await resolveCommit({ repo, revision, token, fetchImpl: get, baseUrl, timeoutMs });
        const context = { get, baseUrl, repo, commit, token, timeoutMs, budget: { pages: 0, truncated: false } };
        const files = format === 'gguf' ? await listGguf(context) : await listSnapshot(context);
        let sizing = null;
        if (format === 'gguf' && input.file !== undefined) {
            if (!files.some((row) => row.file === input.file)) {
                throw failure('not_found', `${input.file} is not a GGUF file of ${repo} at ${commit.slice(0, 12)} that can be added (an LFS file, or the first shard of a complete split set).`);
            }
            sizing = await ggufSizingOf(context, input.file);
        } else if (format !== 'gguf') {
            const row = files.find((entry) => entry.file === 'config.json');
            if (row && row.size <= CONFIG_MAX_BYTES) {
                const { config, readBytes } = await readConfig(context, row);
                sizing = { ...sizingFromConfig(config), readBytes };
            }
        }
        return {
            provider: 'huggingface',
            repo,
            revision,
            commit,
            format,
            gated: gatedOf(info),
            license: licenseOf(info),
            files,
            truncated: context.budget.truncated,
            sizing,
        };
    } catch (error) {
        throw lookupError(error, signal);
    }
}
