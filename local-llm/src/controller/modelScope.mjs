// ModelScope (DS002): a registry of kind `modelscope` serves `huggingface` and `hf-snapshot` sources from its own API.
// A revision is resolved to a 40-hex commit by `/commits`, the files of that commit come from one recursive listing
// (each blob with its size and sha256, which is what a pin records), and a file is read from
// `/repo?Revision=<commit>&FilePath=<path>`, which may redirect to a signed CDN URL. The credential is a Bearer token
// sent to the registry's own origin only; every request goes through the fetch the caller gives (the guarded
// transport), asks for identity encoding, and reads a bounded body. The licence and gating the OpenAPI reports are
// advisory: they are shown, never enforced. Nothing here defaults `fetchImpl`.

import { LocalLlmError } from '../errors.mjs';
import { isSnapshotFile, splitGgufFiles } from './catalog.mjs';
import {
    DownloadError,
    METADATA_TIMEOUT_MS,
    assertFile,
    assertRepo,
    assertRevision,
    fetchJson,
} from './downloader.mjs';
import { ggufSizingOf, byFile, CONFIG_MAX_BYTES, LOOKUP_TIMEOUT_MS, ggufRows, licenseOf, lookupError, readConfig, REGISTRY_TOKEN_HINT, sizingFromConfig } from './modelLookup.mjs';
import { originGuardedFetch } from './netGuard.mjs';
import { normalizeLookupInput } from './registryLookup.mjs';

// The most entries one listing answers; the API gives no flag, so a listing of this many may be cut short.
export const LISTING_LIMIT = 3000;
export const LISTING_MAX_BYTES = 4 * 1024 * 1024;
export const SMALL_MAX_BYTES = 1024 * 1024;
const MAX_FILES = 500;
const COMMIT_RE = /^[a-f0-9]{40}$/;
const SHA256_RE = /^[a-f0-9]{64}$/i;
// Without it a gzip answer would arrive unreadable (the transport does not decode).
const IDENTITY = Object.freeze({ 'Accept-Encoding': 'identity' });

const encodePath = (value) => value.split('/').map(encodeURIComponent).join('/');
const apiBase = (baseUrl, repo) => `${baseUrl}/api/v1/models/${encodePath(repo)}`;

function needFetch(fetchImpl) {
    if (typeof fetchImpl !== 'function') throw new TypeError('fetchImpl is required');
}

const resolveFailed = (message, reason = undefined) => new DownloadError('RESOLVE_FAILED', message, reason ? { details: { reason } } : {});
const notFound = (message) => new DownloadError('NOT_FOUND', message);
const listingIncomplete = (repo, commit) => new DownloadError('RESOLVE_FAILED',
    `ModelScope's file list reached its ${LISTING_LIMIT}-entry limit, so it may be incomplete`, { details: { reason: 'listing-incomplete', repo, commit } });

/** The URL a file is read from; `file` is the repo-relative path, encoded as one query value. */
export function modelScopeFileUrl({ baseUrl, repo, commit, file }) {
    return `${apiBase(baseUrl, repo)}/repo?Revision=${commit}&FilePath=${encodeURIComponent(file)}`;
}

// The `Data` of a ModelScope answer. A refusal carries its own HTTP status, which `fetchJson` reports; an HTTP 200 whose
// envelope says otherwise is not a usable answer.
async function dataOf(url, { token, fetchImpl, timeoutMs, maxBytes, signal }) {
    const { json } = await fetchJson(url, { token, fetchImpl, timeoutMs, maxBytes, signal, headers: IDENTITY });
    if (json === null || typeof json !== 'object' || Array.isArray(json)) throw resolveFailed('ModelScope answered with something that is not an object', 'nonconforming-registry');
    if (json.Success === false) throw resolveFailed('ModelScope reported that the request failed');
    if (json.Success !== true || json.Data === null || typeof json.Data !== 'object' || Array.isArray(json.Data)) {
        throw resolveFailed('ModelScope answered without the expected envelope', 'nonconforming-registry');
    }
    return json.Data;
}

/**
 * The 40-hex commit a branch, tag or commit names. `/commits` is asked even for a 40-hex revision, which it must
 * answer with that commit; a revision it does not know is `Commit: null` (not found).
 */
export async function resolveModelScopeCommit({ repo, revision, token = '', fetchImpl, baseUrl, timeoutMs = METADATA_TIMEOUT_MS, maxBytes = SMALL_MAX_BYTES, signal = undefined }) {
    needFetch(fetchImpl);
    assertRepo(repo);
    assertRevision(revision);
    const url = `${apiBase(baseUrl, repo)}/commits?Ref=${encodeURIComponent(revision)}&PageNumber=1&PageSize=1`;
    const data = await dataOf(url, { token, fetchImpl, timeoutMs, maxBytes, signal });
    if (data.Commit === null || (Array.isArray(data.Commit) && data.Commit.length === 0)) {
        throw notFound(`ModelScope has no revision ${revision} of ${repo}`);
    }
    const id = Array.isArray(data.Commit) ? data.Commit[0]?.Id : undefined;
    const commit = typeof id === 'string' ? id.toLowerCase() : '';
    if (!COMMIT_RE.test(commit)) throw resolveFailed('ModelScope gave no commit id for the revision', 'nonconforming-registry');
    if (/^[a-f0-9]{40}$/i.test(revision) && commit !== revision.toLowerCase()) {
        throw resolveFailed('ModelScope answered another commit than the one asked for', 'nonconforming-registry');
    }
    return commit;
}

// A repo-relative path with no empty or dot segment, no absolute or backslash form and no control character.
function safePath(value) {
    if (typeof value !== 'string' || !value || value.length > 512 || /[\\\u0000-\u001f\u007f]/.test(value)) return false;
    return value.split('/').every((segment) => segment !== '' && segment !== '.' && segment !== '..');
}

/**
 * The blobs of a commit with their size and sha256, from one recursive listing. Trees, unsafe or duplicated paths
 * (every copy is dropped, since the listing could otherwise choose the digest), a size that is not a positive integer
 * and a Sha256 that is not 64 hex are skipped. `incomplete` is true for a listing of 3000 entries or more, which the
 * API cuts without a flag; `Files: null` or empty is not found.
 */
export async function listModelScopeFiles({ repo, commit, token = '', fetchImpl, baseUrl, timeoutMs = METADATA_TIMEOUT_MS, maxBytes = LISTING_MAX_BYTES, signal = undefined }) {
    needFetch(fetchImpl);
    assertRepo(repo);
    const data = await dataOf(`${apiBase(baseUrl, repo)}/repo/files?Revision=${commit}&Recursive=true`, { token, fetchImpl, timeoutMs, maxBytes, signal });
    const entries = data.Files;
    if (entries === null || (Array.isArray(entries) && entries.length === 0)) throw notFound(`ModelScope lists no files for ${repo} at ${String(commit).slice(0, 12)}`);
    if (!Array.isArray(entries)) throw resolveFailed('ModelScope\'s file list is not a list', 'nonconforming-registry');
    const seen = new Map();
    for (const entry of entries) {
        if (typeof entry?.Path === 'string') seen.set(entry.Path, (seen.get(entry.Path) ?? 0) + 1);
    }
    const files = [];
    for (const entry of entries) {
        if (entry === null || typeof entry !== 'object' || entry.Type !== 'blob' || !safePath(entry.Path) || seen.get(entry.Path) !== 1) continue;
        if (!Number.isSafeInteger(entry.Size) || entry.Size <= 0 || typeof entry.Sha256 !== 'string' || !SHA256_RE.test(entry.Sha256)) continue;
        files.push({ path: entry.Path, size: entry.Size, sha256: entry.Sha256.toLowerCase() });
    }
    return { files, incomplete: entries.length >= LISTING_LIMIT };
}

async function pinnedListing(options) {
    const commit = await resolveModelScopeCommit(options);
    const listing = await listModelScopeFiles({ ...options, commit });
    if (listing.incomplete) throw listingIncomplete(options.repo, commit);
    return { commit, files: listing.files };
}

/** Pin a GGUF file, or every shard of a split GGUF, at one commit: `{ commit, size, sha256 }` or `{ commit, size, shards }`. */
export async function resolveModelScopeGguf({ file, revision = 'master', ...options }) {
    needFetch(options.fetchImpl);
    assertFile(file);
    const { commit, files } = await pinnedListing({ ...options, revision });
    const byPath = new Map(files.map((entry) => [entry.path, entry]));
    const names = splitGgufFiles(file);
    const found = (names ?? [file]).map((name) => byPath.get(name));
    if (found.includes(undefined)) throw notFound('Model file not found with a sha256 at the pinned commit');
    if (!names) return Object.freeze({ commit, size: found[0].size, sha256: found[0].sha256 });
    const shards = found.map((entry) => ({ file: entry.path, size: entry.size, sha256: entry.sha256 }));
    return Object.freeze({ commit, size: shards.reduce((sum, shard) => sum + shard.size, 0), shards });
}

/** Pin a snapshot: the top-level model files at one commit, each with size and sha256, at least one `.safetensors`. */
export async function resolveModelScopeSnapshot({ repo, revision = 'master', ...options }) {
    needFetch(options.fetchImpl);
    const { commit, files } = await pinnedListing({ ...options, repo, revision });
    const chosen = files.filter((entry) => isSnapshotFile(entry.path)).sort((left, right) => left.path.localeCompare(right.path));
    if (!chosen.some((entry) => entry.path.endsWith('.safetensors'))) throw notFound('No safetensors files at the pinned commit');
    return Object.freeze({ repo, revision, commit, files: chosen });
}

// Gating and licence from the OpenAPI, advisory: an answer that is missing, refused, too large, off-origin or not JSON
// leaves them unknown. A request that did not complete (a refused address, a network error, a timeout) and a stop of
// the agent are not the answer's failure and propagate.
async function advisoryOf(context) {
    const { baseUrl, repo, token, getMetadata, timeoutMs, signal } = context;
    try {
        const { json } = await fetchJson(`${baseUrl}/openapi/v1/models/${encodePath(repo)}`,
            { token, fetchImpl: getMetadata, timeoutMs, maxBytes: SMALL_MAX_BYTES, signal, headers: IDENTITY });
        const info = json?.success === true && json.data !== null && typeof json.data === 'object' ? json.data : {};
        return { gated: info.gated === false ? false : info.gated === true ? 'manual' : null, license: licenseOf({ cardData: { license: info.license } }) };
    } catch (error) {
        const answered = error instanceof DownloadError && error.code === 'RESOLVE_FAILED'
            && (error.details === undefined || Number.isInteger(error.details.status) || ['too-large', 'redirect'].includes(error.details.reason));
        if (!answered || signal?.aborted) throw error;
        return { gated: null, license: null };
    }
}

const rowOf = (entry) => ({ file: entry.path, size: entry.size, sha256: entry.sha256, gitOid: null, quantization: null, shards: null });

/**
 * Look a model up on a ModelScope registry: the commit, gating and licence (advisory), the files of the format and the
 * sizing from one GGUF header or a snapshot's `config.json`. `revision` defaults to `master`. The answer has the shape
 * of the Hugging Face lookup; `registry` is added by the caller.
 */
export async function lookupModelScope(args, {
    token = '', fetchImpl, baseUrl, timeoutMs = METADATA_TIMEOUT_MS, totalMs = LOOKUP_TIMEOUT_MS, signal = undefined, subject = 'ModelScope',
} = {}) {
    needFetch(fetchImpl);
    const input = normalizeLookupInput(args);
    const revision = args.revision ?? 'master';
    const who = { subject, hint: REGISTRY_TOKEN_HINT };
    const deadline = AbortSignal.any([AbortSignal.timeout(totalMs), ...(signal ? [signal] : [])]);
    try {
        const get = originGuardedFetch({ fetchImpl, baseUrl, deadline });
        const getMetadata = originGuardedFetch({ fetchImpl, baseUrl, deadline, sameOriginOnly: true });
        const { repo, format } = input;
        const base = { repo, token, baseUrl, timeoutMs, signal: deadline };
        const commit = await resolveModelScopeCommit({ ...base, revision, fetchImpl: getMetadata });
        const listing = await listModelScopeFiles({ ...base, commit, fetchImpl: getMetadata });
        const advisory = await advisoryOf({ ...base, getMetadata });
        let truncated = listing.incomplete;
        const context = { get, baseUrl, repo, commit, token, who, fileUrl: (file) => modelScopeFileUrl({ baseUrl, repo, commit, file }) };
        let files;
        if (format === 'gguf') {
            const found = new Map();
            for (const entry of [...listing.files].sort((left, right) => (left.path < right.path ? -1 : 1))) {
                if (!entry.path.endsWith('.gguf')) continue;
                try {
                    assertFile(entry.path);
                } catch {
                    continue;
                }
                if (found.size >= MAX_FILES) truncated = true;
                else found.set(entry.path, { size: entry.size, sha256: entry.sha256 });
            }
            files = ggufRows(found, repo, commit);
        } else {
            const rows = listing.files.filter((entry) => isSnapshotFile(entry.path)).map(rowOf);
            if (rows.length > MAX_FILES) truncated = true;
            files = rows.sort(byFile).slice(0, MAX_FILES);
        }
        let sizing = null;
        if (format === 'gguf' && input.file !== undefined) {
            if (!files.some((row) => row.file === input.file)) {
                throw new LocalLlmError('not_found', `${input.file} is not a GGUF file of ${repo} at ${commit.slice(0, 12)} that can be added (a file with a sha256, or the first shard of a complete split set).`);
            }
            sizing = await ggufSizingOf(context, input.file);
        } else if (format !== 'gguf') {
            const row = files.find((entry) => entry.file === 'config.json');
            if (row && row.size <= CONFIG_MAX_BYTES) {
                const { config, readBytes } = await readConfig(context, row);
                sizing = { ...sizingFromConfig(config), readBytes };
            }
        }
        return { provider: 'modelscope', repo, revision, commit, format, ...advisory, files, truncated, sizing };
    } catch (error) {
        throw lookupError(error, signal, token, who);
    }
}
