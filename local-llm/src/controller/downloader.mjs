// Weight downloader for the local-llm agent.
//
// A download is pinned to an immutable identity (repo, file, commit, size,
// sha256) resolved from the Hugging Face API before any byte is fetched. Bytes
// land in `<file>.partial`, next to a `<file>.partial.json` identity record
// that is written before the first byte, so a later call can resume only a
// partial that belongs to the same identity. The running sha256 is rebuilt
// from the bytes on disk on every call and continued while streaming; the
// partial is renamed into place only after the digest matches. Every attempt
// starts again from the resolve URL, so an expired signed CDN redirect is
// replaced by a fresh one. The module never logs and never puts the token in
// an error, a return value, or a file.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

import { isSnapshotFile } from './catalog.mjs';
import { DownloadError, abortedError, throwIfAborted } from './downloadError.mjs';
import {
    CHUNK_BYTES,
    MAX_CANDIDATES_PER_FILE,
    anchorRoot,
    anchorStore,
    assertRealDirs,
    copyToStaging,
    createDigest,
    ensureDirUnder,
    expectedDigest,
    freedBytes,
    hashBound,
    hashPrefix,
    openBound,
    openPartialBound,
    orderCandidates,
    publishVerified,
    regularFileUnder,
    sameStat,
    statIdentity,
    walkShared,
} from './workspaceReuse.mjs';

export { DownloadError };

const REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const FILE_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
const REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;
const COMMIT_RE = /^[a-f0-9]{40}$/;
const COMMIT_ANY_CASE_RE = /^[A-Fa-f0-9]{40}$/;
const SHA256_RE = /^[a-f0-9]{64}$/;
const GIT_OID_RE = /^[a-f0-9]{40}$/;
const MAX_FILE_SEGMENTS = 4;
const MAX_TREE_PAGES = 50;
const SPACE_MARGIN = 1.05;
const RATE_WINDOW_MS = 5000;
const IDENTITY_KEYS = Object.freeze(['repo', 'file', 'commit', 'size', 'sha256']);
// A file named by a fixed URL (a runner's lock entry) is identified by that URL.
const FILE_IDENTITY_KEYS = Object.freeze(['url', 'size', 'sha256']);
// A file of a Hugging Face snapshot: an LFS file by sha256, a small file kept
// in git by its git blob oid.
const SNAPSHOT_IDENTITY_KEYS = Object.freeze(['repo', 'commit', 'file', 'size', 'sha256', 'gitOid']);

// Marks a local filesystem failure so the transfer loop never mistakes it for
// a network error and retries into a full or broken disk.
class LocalWriteError extends Error {
    constructor(cause) {
        super('local write failed');
        this.localCause = cause;
    }
}

function invalidSource(message) {
    return new DownloadError('INVALID_SOURCE', message);
}

function assertRepo(repo) {
    if (typeof repo !== 'string' || !REPO_RE.test(repo) || repo.includes('..')) {
        throw invalidSource('Invalid Hugging Face repository name');
    }
}

function assertFile(file) {
    if (typeof file !== 'string' || file.includes('..') || !file.endsWith('.gguf')) {
        throw invalidSource('Invalid model file path');
    }
    const segments = file.split('/');
    if (segments.length > MAX_FILE_SEGMENTS || !segments.every((segment) => FILE_SEGMENT_RE.test(segment))) {
        throw invalidSource('Invalid model file path');
    }
}

function assertRevision(revision) {
    if (typeof revision !== 'string' || revision.includes('..')) {
        throw invalidSource('Invalid revision');
    }
    if (!COMMIT_ANY_CASE_RE.test(revision) && !REVISION_RE.test(revision)) {
        throw invalidSource('Invalid revision');
    }
}

function assertArtifact(artifact) {
    if (!artifact || typeof artifact !== 'object') {
        throw invalidSource('Missing artifact');
    }
    assertRepo(artifact.repo);
    assertFile(artifact.file);
    if (typeof artifact.commit !== 'string' || !COMMIT_RE.test(artifact.commit)) {
        throw invalidSource('Invalid artifact commit');
    }
    if (!Number.isSafeInteger(artifact.size) || artifact.size <= 0) {
        throw invalidSource('Invalid artifact size');
    }
    if (typeof artifact.sha256 !== 'string' || !SHA256_RE.test(artifact.sha256)) {
        throw invalidSource('Invalid artifact sha256');
    }
}

function encodeSegments(value) {
    return value.split('/').map(encodeURIComponent).join('/');
}

function authHeaders(token) {
    return token ? { Authorization: `Bearer ${token}` } : {};
}

function isRetryableStatus(status) {
    return status === 429 || status >= 500;
}

async function discardBody(response) {
    try {
        await response.body?.cancel();
    } catch {
        // The connection is being dropped anyway.
    }
}

// Metadata requests run while an admin waits for Add or Update; each one has
// its own deadline, which also covers reading the body.
export const METADATA_TIMEOUT_MS = 15_000;

async function fetchJson(url, { token, fetchImpl, timeoutMs = METADATA_TIMEOUT_MS }) {
    const signal = AbortSignal.timeout(timeoutMs);
    let response;
    try {
        response = await fetchImpl(url, { headers: { Accept: 'application/json', ...authHeaders(token) }, signal });
    } catch (err) {
        // The fetch error is not attached as a cause: only a sanitized code.
        throw new DownloadError('RESOLVE_FAILED', 'Hugging Face metadata request failed', {
            retryable: true,
            details: { reason: networkReason(err) },
        });
    }
    if (!response.ok) {
        await discardBody(response);
        throw new DownloadError('RESOLVE_FAILED', `Hugging Face metadata request returned HTTP ${response.status}`, {
            retryable: isRetryableStatus(response.status),
            details: { status: response.status },
        });
    }
    try {
        return { json: await response.json(), link: response.headers.get('link') };
    } catch (err) {
        if (signal.aborted) {
            throw new DownloadError('RESOLVE_FAILED', 'Hugging Face metadata request timed out', {
                retryable: true,
                details: { reason: 'timeout' },
            });
        }
        throw new DownloadError('RESOLVE_FAILED', 'Hugging Face metadata response is not valid JSON');
    }
}

function networkReason(err) {
    if (err?.name === 'TimeoutError') {
        return 'timeout';
    }
    const code = err?.cause?.code ?? err?.code;
    if (typeof code === 'string' && /^[A-Z0-9_]{1,40}$/.test(code)) {
        return code;
    }
    return err?.name === 'TypeError' ? 'fetch-failed' : 'unknown';
}

async function resolveCommit({ repo, revision, token, fetchImpl, baseUrl, timeoutMs }) {
    if (COMMIT_RE.test(revision)) {
        return revision;
    }
    const url = `${baseUrl}/api/models/${encodeSegments(repo)}/revision/${encodeURIComponent(revision)}`;
    const { json } = await fetchJson(url, { token, fetchImpl, timeoutMs });
    const sha = typeof json?.sha === 'string' ? json.sha.toLowerCase() : '';
    if (!COMMIT_RE.test(sha)) {
        throw new DownloadError('RESOLVE_FAILED', 'Hugging Face revision response has no commit sha');
    }
    return sha;
}

// The tree API pages large directories through a Link header. Only same-origin
// next links are followed, so the token is never sent anywhere else.
function nextTreePage(link, baseUrl) {
    const match = /<([^>]+)>\s*;\s*rel="?next"?/.exec(link ?? '');
    if (!match) {
        return null;
    }
    try {
        const next = new URL(match[1], baseUrl);
        return next.origin === new URL(baseUrl).origin ? next.href : null;
    } catch {
        return null;
    }
}

async function findTreeEntry({ repo, file, commit, token, fetchImpl, baseUrl, timeoutMs }) {
    const dirname = path.posix.dirname(file);
    let url = `${baseUrl}/api/models/${encodeSegments(repo)}/tree/${commit}`;
    if (dirname !== '.') {
        url += `/${encodeSegments(dirname)}`;
    }
    for (let page = 0; url && page < MAX_TREE_PAGES; page += 1) {
        const { json, link } = await fetchJson(url, { token, fetchImpl, timeoutMs });
        if (!Array.isArray(json)) {
            throw new DownloadError('RESOLVE_FAILED', 'Hugging Face tree response is not a list');
        }
        const entry = json.find((item) => item?.path === file);
        if (entry) {
            return entry;
        }
        url = nextTreePage(link, baseUrl);
    }
    return null;
}

function lfsIdentity(entry) {
    const oid = typeof entry?.lfs?.oid === 'string' ? entry.lfs.oid.toLowerCase() : '';
    const size = entry?.lfs?.size;
    if (!SHA256_RE.test(oid) || !Number.isSafeInteger(size) || size <= 0 || size !== entry.size) {
        return null;
    }
    return { sha256: oid, size };
}

export async function resolveHuggingFaceArtifact({
    repo,
    file,
    revision = 'main',
    token = '',
    fetchImpl = globalThis.fetch,
    baseUrl = 'https://huggingface.co',
    timeoutMs = METADATA_TIMEOUT_MS,
}) {
    assertRepo(repo);
    assertFile(file);
    assertRevision(revision);
    const commit = await resolveCommit({ repo, revision, token, fetchImpl, baseUrl, timeoutMs });
    const entry = await findTreeEntry({ repo, file, commit, token, fetchImpl, baseUrl, timeoutMs });
    const identity = entry ? lfsIdentity(entry) : null;
    if (!identity) {
        throw new DownloadError('NOT_FOUND', 'Model file not found as an LFS object at the pinned commit', {
            details: { repo, file, commit },
        });
    }
    return Object.freeze({ source: 'huggingface', repo, file, revision, commit, ...identity });
}

function assertInside(root, candidate) {
    const relative = path.relative(root, candidate);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
        throw invalidSource('Artifact path escapes the weights root');
    }
}

export function artifactPaths({ root, artifact }) {
    if (typeof root !== 'string' || !root) {
        throw invalidSource('Missing weights root');
    }
    assertArtifact(artifact);
    const base = path.resolve(root);
    const dir = path.join(base, ...artifact.repo.split('/'), artifact.commit);
    // The whole relative path, not its basename: two files with the same name
    // in different folders of one commit must not share a location.
    const file = path.join(dir, ...artifact.file.split('/'));
    assertInside(base, dir);
    assertInside(dir, file);
    const partial = `${file}.partial`;
    return { root: base, dir, file, partial, identity: `${partial}.json`, meta: `${file}.json` };
}

function identityOf(artifact, keys = IDENTITY_KEYS) {
    return Object.fromEntries(keys.map((key) => [key, artifact[key]]));
}

function sameIdentity(record, artifact, keys = IDENTITY_KEYS) {
    return Boolean(record) && keys.every((key) => record[key] === artifact[key]);
}

async function readJson(fsApi, filePath) {
    try {
        return JSON.parse(await fsApi.promises.readFile(filePath, 'utf8'));
    } catch {
        return null;
    }
}

async function writeJsonAtomic(fsApi, target, value) {
    const temp = `${target}.tmp-${process.pid}-${crypto.randomBytes(4).toString('hex')}`;
    try {
        await fsApi.promises.writeFile(temp, `${JSON.stringify(value, null, 2)}\n`);
        await fsApi.promises.rename(temp, target);
    } catch (err) {
        await fsApi.promises.rm(temp, { force: true }).catch(() => {});
        throw err;
    }
}

// The bytes of a partial this artifact can resume from, or null: a regular
// file whose identity record matches the artifact and that is not longer
// than it. The overview, the plan and the transfer all use this one rule.
async function partialBytes(fsApi, paths, artifact, keys) {
    const partial = await regularFileUnder(fsApi, paths.root, paths.partial);
    if (!partial || partial.size > artifact.size || !sameIdentity(await readJson(fsApi, paths.identity), artifact, keys)) return null;
    return partial.size;
}

// A symbolic link anywhere in the store counts as absent here (C12, DS002).
async function inspectWith(fsApi, paths, artifact, keys = IDENTITY_KEYS) {
    const file = await regularFileUnder(fsApi, paths.root, paths.file);
    if (file?.size === artifact.size && sameIdentity(await readJson(fsApi, paths.meta), artifact, keys)) {
        return { state: 'complete', bytes: file.size };
    }
    const partial = await partialBytes(fsApi, paths, artifact, keys);
    if (partial !== null) return { state: 'partial', bytes: partial };
    return { state: 'absent', bytes: 0 };
}

export async function inspectArtifact({ root, artifact }) {
    return inspectWith(fs, artifactPaths({ root, artifact }), artifact);
}

/**
 * How one pinned file would be obtained (C12, R4), decided without hashing:
 * `owned` (verified in the store; a changed stat is checked again at Run),
 * `in-place` (a regular file of the pinned size already at its store path),
 * `copy` from a /shared candidate of the pinned size, `copy-or-resume` when
 * there is also a partial to resume, or `download`. `bytesNeeded` is the new
 * disk space that way needs; admission adds the reserve. A candidate is not
 * checked here, so beside a partial it cannot raise the need: the transfer
 * copies only when the whole copy fits and otherwise resumes the partial, so
 * the need is the resume's (`copyBytesNeeded` is the copy's).
 */
async function planWith(fsApi, paths, artifact, keys, candidates, inspected) {
    const file = artifact.file;
    const current = inspected ?? await inspectWith(fsApi, paths, artifact, keys);
    if (current.state === 'complete') return { file, method: 'owned', bytesNeeded: 0 };
    const existing = await regularFileUnder(fsApi, paths.root, paths.file);
    if (existing?.size === artifact.size) return { file, method: 'in-place', bytesNeeded: 0 };
    const have = current.state === 'partial' ? current.bytes : 0;
    if (candidates?.length && have > 0) {
        return { file, method: 'copy-or-resume', bytesNeeded: artifact.size - have, copyBytesNeeded: artifact.size, source: candidates[0].path };
    }
    if (candidates?.length) return { file, method: 'copy', bytesNeeded: artifact.size, source: candidates[0].path };
    return { file, method: 'download', bytesNeeded: artifact.size - have };
}

export async function artifactAcquisition({ root, artifact, candidates = [], inspected, fsApi = fs }) {
    return planWith(fsApi, artifactPaths({ root, artifact }), artifact, IDENTITY_KEYS, candidates, inspected);
}

export async function snapshotFileAcquisition({ root, artifact, candidates = [], inspected, fsApi = fs }) {
    return planWith(fsApi, snapshotFilePaths({ root, artifact }), artifact, SNAPSHOT_IDENTITY_KEYS, candidates, inspected);
}

async function removePartial(fsApi, paths) {
    await fsApi.promises.rm(paths.partial, { force: true });
    await fsApi.promises.rm(paths.identity, { force: true });
}

// Copies left behind by a crash; each attempt stages under a fresh name.
async function stagingFiles(fsApi, paths) {
    const prefix = `${path.basename(paths.partial)}.copy-`;
    try {
        const names = await fsApi.promises.readdir(path.dirname(paths.partial));
        return names.filter((name) => name.startsWith(prefix)).map((name) => path.join(path.dirname(paths.partial), name));
    } catch {
        return [];
    }
}

// A partial is kept only when it is a regular file whose identity record
// matches this artifact and it is not longer than the artifact; anything
// else (a link included, which is removed and never followed) restarts from zero.
async function reconcilePartial(fsApi, paths, artifact, keys = IDENTITY_KEYS) {
    const bytes = await partialBytes(fsApi, paths, artifact, keys);
    if (bytes === null) {
        await removePartial(fsApi, paths);
        return 0;
    }
    return bytes;
}

async function nearestExistingAncestor(fsApi, dir) {
    let current = dir;
    for (;;) {
        try {
            await fsApi.promises.stat(current);
            return current;
        } catch {
            const parent = path.dirname(current);
            if (parent === current) {
                return current;
            }
            current = parent;
        }
    }
}

async function freeSpace({ fsApi, statfs, dir, remaining }) {
    const stats = await statfs(await nearestExistingAncestor(fsApi, dir));
    const available = Number(stats.bavail) * Number(stats.bsize);
    const required = Math.ceil(remaining * SPACE_MARGIN);
    return { available, required, fits: available >= required };
}

function insufficientSpace({ required, available }, what) {
    return new DownloadError('INSUFFICIENT_SPACE', `Not enough free space for ${what}`, {
        details: { required, available },
    });
}

async function assertFreeSpace({ fsApi, statfs, dir, remaining, what = 'the model weights' }) {
    const space = await freeSpace({ fsApi, statfs, dir, remaining });
    if (!space.fits) {
        throw insufficientSpace(space, what);
    }
}

function downloadSourceOf(ctx) {
    return ctx.url || resolveUrl(ctx.baseUrl, ctx.artifact);
}

function provenanceOf(meta, ctx) {
    const recorded = meta?.provenance;
    if (recorded && typeof recorded.source === 'string' && typeof recorded.method === 'string') return recorded;
    return { file: ctx.label, source: downloadSourceOf(ctx), method: 'download', bytes: ctx.artifact.size };
}

/**
 * The agent's own file (C12, steps 1 and 2): a verified file whose stat is
 * unchanged is used as it is; a changed one is hashed again; an unrecorded
 * regular file of the pinned size at the store path is hashed and adopted in
 * place. Always through one bound descriptor (R1, R2). Returns null when the
 * file must be obtained another way.
 */
async function useOwnFile(ctx) {
    const { fsApi, paths, artifact, identityKeys, label } = ctx;
    const meta = await readJson(fsApi, paths.meta);
    const recorded = sameIdentity(meta, artifact, identityKeys);
    const opened = await openBound(fsApi, ctx.anchor, paths.file, { signal: ctx.signal });
    if (!opened.handle) {
        if (opened.rejected !== 'missing') {
            ctx.notes.push(`${label}: the store path is ${opened.rejected === 'symlink' ? 'a symbolic link' : opened.rejected}; treated as absent`);
        }
        if (recorded) await fsApi.promises.rm(paths.meta, { force: true });
        return null;
    }
    const { handle, stat } = opened;
    try {
        if (stat.size !== artifact.size) {
            if (recorded) await fsApi.promises.rm(paths.meta, { force: true });
            return null;
        }
        if (recorded && sameStat(meta.stat, statIdentity(stat))) {
            return { provenance: provenanceOf(meta, ctx), current: true };
        }
        // Changed since it was verified, verified before stats were recorded, or never verified here.
        const checked = await hashBound(handle, artifact, {
            signal: ctx.signal,
            chunkBytes: ctx.chunkBytes,
            onBytes: (bytes) => ctx.progress.tick(bytes, 0, 'verifying'),
        });
        if (!checked.stable) {
            throw new DownloadError('CHANGED_WHILE_VERIFYING', `${label} changed while it was being verified; run again`, { retryable: true });
        }
        if (!checked.matches) {
            if (recorded) {
                // The agent's own file lost its bytes: remove it before fetching it again.
                await fsApi.promises.rm(paths.file, { force: true });
                await fsApi.promises.rm(paths.meta, { force: true });
            } else {
                ctx.notes.push(`${label}: the file at the store path does not match the pinned digest; it will be replaced`);
            }
            return null;
        }
        const provenance = recorded ? provenanceOf(meta, ctx) : { file: label, source: paths.file, method: 'in-place', bytes: 0 };
        // A snapshot keeps its records in a separate .state tree, which a canonical file placed by the operator lacks (F3).
        await ensureDirUnder(fsApi, paths.root, path.dirname(paths.meta));
        await writeJsonAtomic(fsApi, paths.meta, {
            ...identityOf(artifact, identityKeys),
            verifiedAt: new Date().toISOString(),
            provenance,
            stat: checked.stat,
        });
        if (!recorded && stat.nlink > 1) {
            ctx.notes.push(`${label} was adopted in place and has ${stat.nlink - 1} other hard link(s); `
                + 'do not keep a link to it under /shared, where every agent can write');
        }
        return recorded ? { provenance, current: true, reverified: true } : { provenance };
    } finally {
        await handle.close().catch(() => {});
    }
}

/**
 * Step 3 (C12): a private, verified copy of a /shared candidate. At most
 * MAX_CANDIDATES_PER_FILE are hashed. Each gets its own staging file; an
 * existing partial and its identity record stay untouched until a verified
 * copy is published (R5). A copy that does not fit beside a partial that can
 * be resumed is not made: the partial is resumed instead, as the plan says
 * (F2). One free-space measurement per candidate decides between the two, and
 * gives a refusal its figures, so two readings cannot disagree. Otherwise
 * space, write and read failures are errors (R6).
 */
async function adoptCopy(ctx, candidates) {
    const { fsApi, paths, artifact, label } = ctx;
    let hashed = 0;
    for (const candidate of candidates) {
        if (hashed >= MAX_CANDIDATES_PER_FILE) break;
        throwIfAborted(ctx.signal);
        const opened = await openBound(fsApi, candidate.anchor, candidate.path, { signal: ctx.signal });
        if (!opened.handle) {
            ctx.notes.push(`${label}: skipped ${candidate.path} (${opened.rejected === 'symlink' ? 'a symbolic link' : opened.rejected})`);
            continue;
        }
        let staging = null;
        let published = null;
        try {
            if (opened.stat.size !== artifact.size) continue;
            hashed += 1;
            const space = await freeSpace({ fsApi, statfs: ctx.statfs, dir: path.dirname(paths.partial), remaining: artifact.size });
            if (!space.fits) {
                const have = await partialBytes(fsApi, paths, artifact, ctx.identityKeys);
                if (have) {
                    ctx.notes.push(`${label}: a private copy of ${candidate.path} does not fit on the disk; `
                        + `resuming the partial download (${have} of ${artifact.size} bytes) instead`);
                    return null;
                }
                throw insufficientSpace(space, 'a private copy of the model file');
            }
            await ensureDirUnder(fsApi, paths.root, path.dirname(paths.partial));
            staging = `${paths.partial}.copy-${crypto.randomBytes(6).toString('hex')}`;
            const copied = await copyToStaging(fsApi, {
                source: candidate.path,
                opened,
                anchor: ctx.anchor,
                staging,
                artifact,
                signal: ctx.signal,
                chunkBytes: ctx.chunkBytes,
                onBytes: (bytes) => ctx.progress.tick(bytes, 0, 'copying'),
                onVerifyBytes: (bytes) => ctx.progress.tick(bytes, 0, 'verifying'),
            });
            if (!copied.staged) {
                ctx.notes.push(`${label}: ${candidate.path} not used; ${copied.reason}`);
                continue;
            }
            published = copied.handle;
            const provenance = { file: label, source: candidate.path, method: 'copy', bytes: artifact.size };
            await publishVerified(fsApi, {
                root: paths.root,
                staged: copied.staged,
                handle: copied.handle,
                from: staging,
                to: paths.file,
                meta: paths.meta,
                record: { ...identityOf(artifact, ctx.identityKeys), verifiedAt: new Date().toISOString(), provenance },
                writeJson: (target, value) => writeJsonAtomic(fsApi, target, value),
            });
            staging = null;
            // Only now is an earlier partial of this file obsolete.
            await removePartial(fsApi, paths);
            return { provenance };
        } finally {
            await published?.close().catch(() => {});
            await opened.handle.close().catch(() => {});
            if (staging) await fsApi.promises.rm(staging, { force: true }).catch(() => {});
        }
    }
    return null;
}

// Progress carries its phase: `downloading`, `copying` (a private copy from
// /shared) or `verifying` (hashing bytes already on disk).
function createProgress({ total, onProgress, intervalMs }) {
    const samples = [];
    let lastEmit = -Infinity;
    const emit = (bytes, transferred, now, phase) => {
        samples.push({ time: now, transferred });
        while (samples.length > 2 && samples[0].time < now - RATE_WINDOW_MS) {
            samples.shift();
        }
        const first = samples[0];
        const seconds = (now - first.time) / 1000;
        const rate = seconds > 0 ? (transferred - first.transferred) / seconds : 0;
        const etaSeconds = rate > 0 ? Math.ceil((total - bytes) / rate) : null;
        lastEmit = now;
        try {
            onProgress({ bytes, total, rate, etaSeconds, transferred, phase });
        } catch {
            // A failing progress consumer must not corrupt the transfer.
        }
    };
    return {
        tick(bytes, transferred, phase = 'downloading') {
            const now = performance.now();
            if (now - lastEmit >= intervalMs) {
                emit(bytes, transferred, now, phase);
            }
        },
        finish(bytes, transferred, phase = 'downloading') {
            emit(bytes, transferred, performance.now(), phase);
        },
    };
}

async function writeFully(handle, chunk) {
    let offset = 0;
    while (offset < chunk.length) {
        const { bytesWritten } = await handle.write(chunk, offset, chunk.length - offset);
        offset += bytesWritten;
    }
}

async function syncQuietly(handle) {
    try {
        await handle.sync();
    } catch {
        // Best effort: the caller is already reporting a failure.
    }
}

function resolveUrl(baseUrl, artifact) {
    return `${baseUrl}/${encodeSegments(artifact.repo)}/resolve/${artifact.commit}/${encodeSegments(artifact.file)}`;
}

async function resetPartial(ctx) {
    try {
        await ctx.handle.truncate(0);
    } catch (err) {
        throw new LocalWriteError(err);
    }
    ctx.have = 0;
    ctx.hash = createDigest(ctx.artifact);
}

async function appendChunk(ctx, chunk) {
    try {
        await writeFully(ctx.handle, chunk);
    } catch (err) {
        throw new LocalWriteError(err);
    }
    ctx.hash.update(chunk);
    ctx.have += chunk.length;
    ctx.transferred += chunk.length;
    ctx.progress.tick(ctx.have, ctx.transferred);
}

// Streams one response body into the partial. Returns a retry reason when the
// body ended early or overran the expected size, or null when it is complete.
async function streamBody(ctx, response) {
    const { signal, artifact } = ctx;
    for await (const chunk of response.body) {
        throwIfAborted(signal);
        if (ctx.have + chunk.length > artifact.size) {
            await resetPartial(ctx);
            return 'overflow';
        }
        await appendChunk(ctx, chunk);
        throwIfAborted(signal);
    }
    return ctx.have === artifact.size ? null : 'short-body';
}

async function handleResponse(ctx, response) {
    const { artifact } = ctx;
    const { status } = response;
    if (status === 206) {
        const expected = `bytes ${ctx.have}-${artifact.size - 1}/${artifact.size}`;
        if (response.headers.get('content-range') !== expected) {
            await discardBody(response);
            await resetPartial(ctx);
            return { retry: 'invalid-content-range', status };
        }
        return { retry: await streamBody(ctx, response), status };
    }
    if (status === 200) {
        const length = response.headers.get('content-length');
        if (length !== null && Number(length) !== artifact.size && ctx.have > 0) {
            // A 200 to a Range request that carries only part of the file
            // (LM Studio's download host does this): a 200 does not say where
            // its body starts, so start again from zero without a Range.
            await discardBody(response);
            await resetPartial(ctx);
            return { retry: 'partial-200', status };
        }
        if (length !== null && Number(length) !== artifact.size) {
            await discardBody(response);
            throw new DownloadError('HTTP_ERROR', 'Server reported a size that differs from the pinned artifact', {
                details: { status, contentLength: Number(length), size: artifact.size },
            });
        }
        // The server ignored Range (or none was sent): the body starts at 0.
        if (ctx.have > 0) {
            await resetPartial(ctx);
        }
        return { retry: await streamBody(ctx, response), status };
    }
    await discardBody(response);
    if (status === 416) {
        if (ctx.have === artifact.size) {
            return { retry: null, status };
        }
        await resetPartial(ctx);
        return { retry: 'range-not-satisfiable', status };
    }
    if (status === 403 || status === 410) {
        return { retry: 'expired-redirect', status };
    }
    if (status === 404) {
        throw new DownloadError('NOT_FOUND', ctx.url ? 'File not found at its pinned URL' : 'Model file not found at the pinned commit',
            { details: { status } });
    }
    if (isRetryableStatus(status)) {
        return { retry: 'server-error', status };
    }
    throw new DownloadError('HTTP_ERROR', `Download returned HTTP ${status}`, { details: { status } });
}

async function attemptOnce(ctx) {
    const { signal, artifact } = ctx;
    const attemptController = new AbortController();
    const signals = signal ? [signal, attemptController.signal] : [attemptController.signal];
    // Identity encoding keeps Content-Length and byte ranges in raw bytes.
    const headers = { 'Accept-Encoding': 'identity', ...authHeaders(ctx.token) };
    if (ctx.have > 0) {
        headers.Range = `bytes=${ctx.have}-`;
    }
    try {
        const response = await ctx.fetchImpl(ctx.url || resolveUrl(ctx.baseUrl, artifact), {
            headers,
            redirect: 'follow',
            signal: AbortSignal.any(signals),
        });
        return await handleResponse(ctx, response);
    } catch (err) {
        if (err instanceof DownloadError) {
            throw err;
        }
        if (err instanceof LocalWriteError) {
            throw err;
        }
        if (signal?.aborted) {
            throw abortedError();
        }
        return { retry: 'network', reason: networkReason(err) };
    } finally {
        attemptController.abort();
    }
}

async function backOff(ctx, failures) {
    const ms = ctx.backoffMs(failures - 1);
    if (ctx.sleep) {
        await ctx.sleep(ms, ctx.signal);
    } else {
        try {
            await delay(ms, undefined, ctx.signal ? { signal: ctx.signal } : {});
        } catch {
            throw abortedError();
        }
    }
    throwIfAborted(ctx.signal);
}

// Retries until the partial holds `size` bytes. The failure counter resets
// only when an attempt left more bytes on disk than it started with, so a
// server that restarts from zero and then drops the connection cannot loop
// forever.
async function transfer(ctx) {
    let failures = 0;
    while (ctx.have < ctx.artifact.size) {
        throwIfAborted(ctx.signal);
        const before = ctx.have;
        const outcome = await attemptOnce(ctx);
        if (outcome.retry === null) {
            return;
        }
        await syncQuietly(ctx.handle);
        failures = ctx.have > before ? 1 : failures + 1;
        if (failures >= ctx.maxAttempts) {
            throw new DownloadError('NETWORK', 'Download failed after repeated attempts; the partial file is kept', {
                retryable: true,
                details: { attempts: failures, lastReason: outcome.retry, lastStatus: outcome.status ?? null },
            });
        }
        await backOff(ctx, failures);
    }
}

function mismatchError(ctx, actual, why = 'Downloaded bytes do not match') {
    return new DownloadError('SHA256_MISMATCH', `${why} the pinned ${ctx.artifact.sha256 === undefined ? 'git oid' : 'sha256'}`, {
        details: { expected: expectedDigest(ctx.artifact), actual },
    });
}

/**
 * The finished partial, verified through its own descriptor with an identical
 * fstat before and after (F4): the streamed digest only proves the bytes
 * received, not the bytes now on disk, including a resumed prefix.
 */
async function verifyPartial(ctx) {
    const streamed = ctx.hash.digest('hex');
    if (streamed !== expectedDigest(ctx.artifact)) {
        await removePartial(ctx.fsApi, ctx.paths);
        throw mismatchError(ctx, streamed);
    }
    const checked = await hashBound(ctx.handle, ctx.artifact, {
        signal: ctx.signal,
        chunkBytes: ctx.chunkBytes,
        onBytes: (bytes) => ctx.progress.tick(bytes, ctx.transferred, 'verifying'),
    });
    if (!checked.stable) {
        throw new DownloadError('CHANGED_WHILE_VERIFYING', `${ctx.label} changed while it was being verified; run again`, { retryable: true });
    }
    if (!checked.matches) {
        await removePartial(ctx.fsApi, ctx.paths);
        throw mismatchError(ctx, null, 'The bytes on disk changed after they were received; they do not match');
    }
    // The object whose every byte was just hashed through this descriptor.
    ctx.written = checked.stat;
}

async function finalize(ctx) {
    const { paths } = ctx;
    // The record goes first: a crash before the rename leaves a verifiable
    // partial, never a final file without its identity record.
    await publishVerified(ctx.fsApi, {
        root: paths.root,
        staged: ctx.written,
        from: paths.partial,
        to: paths.file,
        meta: paths.meta,
        record: {
            ...identityOf(ctx.artifact, ctx.identityKeys),
            verifiedAt: new Date().toISOString(),
            provenance: { file: ctx.label, source: downloadSourceOf(ctx), method: 'download', bytes: ctx.transferred },
        },
        writeJson: (target, value) => writeJsonAtomic(ctx.fsApi, target, value),
    });
    await ctx.fsApi.promises.rm(paths.identity, { force: true });
}

async function runTransfer(ctx) {
    try {
        await transfer(ctx);
        try {
            await ctx.handle.sync();
        } catch (err) {
            throw new LocalWriteError(err);
        }
        ctx.progress.finish(ctx.have, ctx.transferred);
        await verifyPartial(ctx);
    } catch (err) {
        await syncQuietly(ctx.handle);
        if (err instanceof LocalWriteError) {
            if (err.localCause?.code === 'ENOSPC') {
                throw new DownloadError('PAUSED_ENOSPC', 'Disk is full; the partial file is kept for resume', {
                    retryable: true,
                    details: { bytes: ctx.have },
                });
            }
            throw err.localCause;
        }
        throw err;
    } finally {
        await ctx.handle.close().catch(() => {});
    }
    await finalize(ctx);
}

export async function downloadArtifact({
    artifact,
    root,
    token = '',
    fetchImpl = globalThis.fetch,
    baseUrl = 'https://huggingface.co',
    ...options
}) {
    const paths = artifactPaths({ root, artifact });
    return downloadInto({ paths, artifact, identityKeys: IDENTITY_KEYS, url: null, token, fetchImpl, baseUrl, ...options });
}

/**
 * Obtain one pinned file, in the order of C12 (DS002): the agent's own
 * verified file; a file already at the store path, adopted in place; a
 * private copy of a /shared candidate; otherwise a download or its resume.
 * Returns the path, the bytes transferred, `provenance` ({ file, source,
 * method, bytes }) and `notes` for the deployment log.
 */
async function downloadInto({
    paths,
    artifact,
    identityKeys,
    url,
    token = '',
    fetchImpl = globalThis.fetch,
    baseUrl = 'https://huggingface.co',
    onProgress = () => {},
    signal,
    statfs = (p) => fs.promises.statfs(p),
    fsApi = fs,
    progressIntervalMs = 500,
    maxAttempts = 6,
    backoffMs = (attempt) => Math.min(30_000, 1000 * 2 ** attempt),
    sleep,
    // Directories whose regular files may be copied instead of downloaded (C12).
    adoptFrom = [],
    // Or the candidates already found by the caller's walk: a list, or an async function returning one.
    candidates = null,
    chunkBytes = CHUNK_BYTES,
}) {
    throwIfAborted(signal);
    const ctx = {
        paths, artifact, identityKeys, url, token, fetchImpl, baseUrl, signal, statfs, fsApi, maxAttempts, backoffMs, sleep,
        chunkBytes,
        label: artifact.file || path.basename(paths.file),
        notes: [],
        progress: createProgress({ total: artifact.size, onProgress, intervalMs: progressIntervalMs }),
    };
    const done = (result, bytesTransferred = 0) => ({ status: 'complete', path: paths.file, bytesTransferred, notes: ctx.notes, ...result });
    // The store root is anchored once for this operation (F1).
    ctx.anchor = await anchorStore(fsApi, paths.root);
    // No symbolic link on the way to either directory (R2); they are created only when something is placed (R3).
    await assertRealDirs(fsApi, paths.root, path.dirname(paths.file));
    await assertRealDirs(fsApi, paths.root, path.dirname(paths.partial));
    for (const stale of await stagingFiles(fsApi, paths)) await fsApi.promises.rm(stale, { force: true });
    const own = await useOwnFile(ctx);
    if (own) {
        await removePartial(fsApi, paths);
        return done(own);
    }
    let list = typeof candidates === 'function' ? await candidates() : candidates;
    if (!list && adoptFrom.length) {
        list = orderCandidates((await walkShared(fsApi, adoptFrom, [artifact.size], signal)).get(artifact.size), ctx.label);
    }
    const copied = list?.length ? await adoptCopy(ctx, list) : null;
    if (copied) return done(copied);
    const have = await reconcilePartial(fsApi, paths, artifact, identityKeys);
    await assertFreeSpace({ fsApi, statfs, dir: paths.dir, remaining: artifact.size - have });
    await ensureDirUnder(fsApi, paths.root, path.dirname(paths.file));
    await ensureDirUnder(fsApi, paths.root, path.dirname(paths.partial));
    await writeJsonAtomic(fsApi, paths.identity, identityOf(artifact, identityKeys));
    const handle = await openPartialBound(fsApi, ctx.anchor, paths.partial);
    const hash = createDigest(artifact);
    try {
        await hashPrefix(handle, have, hash, { signal, chunkBytes, onBytes: (bytes) => ctx.progress.tick(bytes, 0, 'verifying') });
    } catch (error) {
        await handle.close().catch(() => {});
        throw error;
    }
    Object.assign(ctx, { have, transferred: 0, hash, handle });
    await runTransfer(ctx);
    return done({ provenance: { file: ctx.label, source: downloadSourceOf(ctx), method: 'download', bytes: ctx.transferred } },
        ctx.transferred);
}

/**
 * The last check before a runner loads the file (C12): the verified file's
 * stat must be unchanged, or its bytes are hashed again. Throws when the file
 * is no longer the verified one.
 */
async function verifyInto({ paths, artifact, identityKeys, url = null, baseUrl = 'https://huggingface.co', signal, onProgress = () => {},
    fsApi = fs, progressIntervalMs = 500, chunkBytes = CHUNK_BYTES }) {
    const ctx = {
        paths, artifact, identityKeys, url, baseUrl, signal, fsApi, chunkBytes,
        label: artifact.file || path.basename(paths.file),
        notes: [],
        progress: createProgress({ total: artifact.size, onProgress, intervalMs: progressIntervalMs }),
    };
    ctx.anchor = await anchorRoot(fsApi, paths.root);
    const meta = await readJson(fsApi, paths.meta);
    const own = ctx.anchor && sameIdentity(meta, artifact, identityKeys) ? await useOwnFile(ctx) : null;
    if (!own) {
        throw new DownloadError('CHANGED_AFTER_VERIFY', `${ctx.label} is no longer the verified file; run again to fetch it`);
    }
    return { reverified: Boolean(own.reverified), notes: ctx.notes };
}

export async function verifyArtifact({ root, artifact, ...options }) {
    return verifyInto({ paths: artifactPaths({ root, artifact }), artifact, identityKeys: IDENTITY_KEYS, ...options });
}

function fileSource({ url, size, sha256, target, allowHttp = false }) {
    let parsed;
    try {
        parsed = new URL(url);
    } catch {
        throw invalidSource('Invalid file URL');
    }
    if (parsed.protocol !== 'https:' && !(allowHttp && parsed.protocol === 'http:')) {
        throw invalidSource('Files are fetched over https only');
    }
    if (parsed.username || parsed.password) {
        throw invalidSource('A file URL must not carry credentials');
    }
    if (!Number.isSafeInteger(size) || size <= 0) {
        throw invalidSource('Invalid file size');
    }
    if (typeof sha256 !== 'string' || !SHA256_RE.test(sha256)) {
        throw invalidSource('Invalid file sha256');
    }
    if (typeof target !== 'string' || !path.isAbsolute(target) || target.includes('\0')) {
        throw invalidSource('Invalid file target');
    }
    const file = path.resolve(target);
    return {
        artifact: Object.freeze({ url: parsed.href, size, sha256 }),
        paths: { root: path.dirname(file), dir: path.dirname(file), file, partial: `${file}.partial`, identity: `${file}.partial.json`, meta: `${file}.json` },
    };
}

/**
 * One file pinned by URL, size and sha256 (a runner lock entry), downloaded
 * to `target` with the same resume, free-space and verification rules as
 * model weights. `<target>.json` records the verified identity. Plain http is
 * refused unless `allowHttp` (tests only).
 */
export async function downloadFile({ url, size, sha256, target, allowHttp = false, ...options }) {
    const { artifact, paths } = fileSource({ url, size, sha256, target, allowHttp });
    return downloadInto({ paths, artifact, identityKeys: FILE_IDENTITY_KEYS, url: artifact.url, ...options });
}

export async function inspectFile({ url, size, sha256, target, allowHttp = true }) {
    const { artifact, paths } = fileSource({ url, size, sha256, target, allowHttp });
    return inspectWith(fs, paths, artifact, FILE_IDENTITY_KEYS);
}

async function removeEmptyParents(base, dir) {
    let current = dir;
    while (current !== base && !path.relative(base, current).startsWith('..')) {
        try {
            await fs.promises.rmdir(current);
        } catch (err) {
            if (err.code !== 'ENOENT') {
                return;
            }
        }
        current = path.dirname(current);
    }
}

// Freed bytes count each inode whose last link this removal takes (R8).
export async function removeArtifact({ root, artifact, fsApi = fs }) {
    const paths = artifactPaths({ root, artifact });
    const targets = [paths.file, paths.partial, paths.identity, paths.meta, ...await stagingFiles(fsApi, paths)];
    const stats = [];
    for (const target of targets) {
        try {
            stats.push(await fsApi.promises.lstat(target));
        } catch (err) {
            if (err.code !== 'ENOENT') throw err;
        }
    }
    const freed = freedBytes(stats);
    for (const target of targets) await fsApi.promises.rm(target, { force: true });
    await removeEmptyParents(path.resolve(root), path.dirname(paths.file));
    return freed;
}

/**
 * The directories of one Hugging Face snapshot: `dir` holds only the verified
 * files a runner loads; `stateDir` holds the partial files and identity
 * records, so a runner never sees them.
 */
export function snapshotPaths({ root, repo, commit }) {
    if (typeof root !== 'string' || !root) {
        throw invalidSource('Missing weights root');
    }
    assertRepo(repo);
    if (typeof commit !== 'string' || !COMMIT_RE.test(commit)) {
        throw invalidSource('Invalid snapshot commit');
    }
    const base = path.resolve(root);
    const dir = path.join(base, ...repo.split('/'), commit);
    const stateDir = path.join(base, '.state', ...repo.split('/'), commit);
    assertInside(base, dir);
    assertInside(base, stateDir);
    return { dir, stateDir };
}

function assertSnapshotArtifact(artifact) {
    if (!artifact || typeof artifact !== 'object') {
        throw invalidSource('Missing artifact');
    }
    if (!isSnapshotFile(artifact.file)) {
        throw invalidSource('Invalid snapshot file name');
    }
    if (!Number.isSafeInteger(artifact.size) || artifact.size <= 0) {
        throw invalidSource('Invalid artifact size');
    }
    const hasSha = artifact.sha256 !== undefined;
    if (hasSha ? !SHA256_RE.test(String(artifact.sha256)) : !GIT_OID_RE.test(String(artifact.gitOid || ''))) {
        throw invalidSource('A snapshot file needs its sha256 or git oid');
    }
}

function snapshotFilePaths({ root, artifact }) {
    assertSnapshotArtifact(artifact);
    const { dir, stateDir } = snapshotPaths({ root, repo: artifact.repo, commit: artifact.commit });
    const file = path.join(dir, artifact.file);
    const partial = path.join(stateDir, `${artifact.file}.partial`);
    assertInside(dir, file);
    assertInside(stateDir, partial);
    return { root: path.resolve(root), dir, file, partial, identity: `${partial}.json`, meta: path.join(stateDir, `${artifact.file}.json`) };
}

/** One file of a pinned snapshot, resumed, verified and moved into the snapshot directory. */
export async function downloadSnapshotFile({
    artifact,
    root,
    token = '',
    fetchImpl = globalThis.fetch,
    baseUrl = 'https://huggingface.co',
    ...options
}) {
    const paths = snapshotFilePaths({ root, artifact });
    return downloadInto({ paths, artifact, identityKeys: SNAPSHOT_IDENTITY_KEYS, url: null, token, fetchImpl, baseUrl, ...options });
}

export async function inspectSnapshotFile({ root, artifact }) {
    return inspectWith(fs, snapshotFilePaths({ root, artifact }), artifact, SNAPSHOT_IDENTITY_KEYS);
}

export async function verifySnapshotFile({ root, artifact, ...options }) {
    return verifyInto({ paths: snapshotFilePaths({ root, artifact }), artifact, identityKeys: SNAPSHOT_IDENTITY_KEYS, ...options });
}

/**
 * Pin a snapshot: the revision's commit, and every top-level model file at
 * that commit with its size and digest (LFS sha256, or the git blob oid for a
 * file kept in git). Reads metadata only.
 */
export async function resolveHuggingFaceSnapshot({
    repo,
    revision = 'main',
    token = '',
    fetchImpl = globalThis.fetch,
    baseUrl = 'https://huggingface.co',
    timeoutMs = METADATA_TIMEOUT_MS,
}) {
    assertRepo(repo);
    assertRevision(revision);
    const commit = await resolveCommit({ repo, revision, token, fetchImpl, baseUrl, timeoutMs });
    let url = `${baseUrl}/api/models/${encodeSegments(repo)}/tree/${commit}`;
    const files = [];
    for (let page = 0; url && page < MAX_TREE_PAGES; page += 1) {
        const { json, link } = await fetchJson(url, { token, fetchImpl, timeoutMs });
        if (!Array.isArray(json)) {
            throw new DownloadError('RESOLVE_FAILED', 'Hugging Face tree response is not a list');
        }
        for (const entry of json) {
            if (entry?.type !== 'file' || !isSnapshotFile(entry.path)) continue;
            if (entry.lfs !== undefined && entry.lfs !== null) {
                // Never fall back to the pointer's git oid for an LFS file.
                const lfs = lfsIdentity(entry);
                if (!lfs) throw new DownloadError('RESOLVE_FAILED', `Inconsistent LFS metadata for ${entry.path}`, { details: { repo, commit } });
                files.push({ path: entry.path, size: lfs.size, sha256: lfs.sha256 });
                continue;
            }
            const oid = typeof entry.oid === 'string' ? entry.oid.toLowerCase() : '';
            if (GIT_OID_RE.test(oid) && Number.isSafeInteger(entry.size) && entry.size > 0) files.push({ path: entry.path, size: entry.size, gitOid: oid });
        }
        url = nextTreePage(link, baseUrl);
    }
    if (url) {
        // A listing cut short would pin an incomplete snapshot.
        throw new DownloadError('RESOLVE_FAILED', 'The Hugging Face file list is too long', { details: { repo, commit } });
    }
    if (!files.length) {
        throw new DownloadError('NOT_FOUND', 'No model files at the pinned commit', { details: { repo, commit } });
    }
    files.sort((left, right) => left.path.localeCompare(right.path));
    return Object.freeze({ repo, revision, commit, files });
}
