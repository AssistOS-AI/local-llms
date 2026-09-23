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

const REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const FILE_SEGMENT_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$/;
const REVISION_RE = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;
const COMMIT_RE = /^[a-f0-9]{40}$/;
const COMMIT_ANY_CASE_RE = /^[A-Fa-f0-9]{40}$/;
const SHA256_RE = /^[a-f0-9]{64}$/;
const MAX_FILE_SEGMENTS = 4;
const MAX_TREE_PAGES = 50;
const SPACE_MARGIN = 1.05;
const RATE_WINDOW_MS = 5000;
const IDENTITY_KEYS = Object.freeze(['repo', 'file', 'commit', 'size', 'sha256']);

export class DownloadError extends Error {
    constructor(code, message = code, { retryable = false, details } = {}) {
        super(message);
        this.name = 'DownloadError';
        this.code = code;
        this.retryable = retryable;
        if (details !== undefined) {
            this.details = details;
        }
    }
}

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

async function fetchJson(url, { token, fetchImpl }) {
    let response;
    try {
        response = await fetchImpl(url, { headers: { Accept: 'application/json', ...authHeaders(token) } });
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
    } catch {
        throw new DownloadError('RESOLVE_FAILED', 'Hugging Face metadata response is not valid JSON');
    }
}

function networkReason(err) {
    const code = err?.cause?.code ?? err?.code;
    if (typeof code === 'string' && /^[A-Z0-9_]{1,40}$/.test(code)) {
        return code;
    }
    return err?.name === 'TypeError' ? 'fetch-failed' : 'unknown';
}

async function resolveCommit({ repo, revision, token, fetchImpl, baseUrl }) {
    if (COMMIT_RE.test(revision)) {
        return revision;
    }
    const url = `${baseUrl}/api/models/${encodeSegments(repo)}/revision/${encodeURIComponent(revision)}`;
    const { json } = await fetchJson(url, { token, fetchImpl });
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

async function findTreeEntry({ repo, file, commit, token, fetchImpl, baseUrl }) {
    const dirname = path.posix.dirname(file);
    let url = `${baseUrl}/api/models/${encodeSegments(repo)}/tree/${commit}`;
    if (dirname !== '.') {
        url += `/${encodeSegments(dirname)}`;
    }
    for (let page = 0; url && page < MAX_TREE_PAGES; page += 1) {
        const { json, link } = await fetchJson(url, { token, fetchImpl });
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
}) {
    assertRepo(repo);
    assertFile(file);
    assertRevision(revision);
    const commit = await resolveCommit({ repo, revision, token, fetchImpl, baseUrl });
    const entry = await findTreeEntry({ repo, file, commit, token, fetchImpl, baseUrl });
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
    const file = path.join(dir, path.posix.basename(artifact.file));
    assertInside(base, dir);
    assertInside(dir, file);
    const partial = `${file}.partial`;
    return { dir, file, partial, identity: `${partial}.json`, meta: `${file}.json` };
}

function identityOf(artifact) {
    return Object.fromEntries(IDENTITY_KEYS.map((key) => [key, artifact[key]]));
}

function sameIdentity(record, artifact) {
    return Boolean(record) && IDENTITY_KEYS.every((key) => record[key] === artifact[key]);
}

async function readJson(fsApi, filePath) {
    try {
        return JSON.parse(await fsApi.promises.readFile(filePath, 'utf8'));
    } catch {
        return null;
    }
}

async function fileSize(fsApi, filePath) {
    try {
        const stats = await fsApi.promises.stat(filePath);
        return stats.isFile() ? stats.size : null;
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

async function inspectWith(fsApi, paths, artifact) {
    const size = await fileSize(fsApi, paths.file);
    if (size === artifact.size && sameIdentity(await readJson(fsApi, paths.meta), artifact)) {
        return { state: 'complete', bytes: size };
    }
    const partialSize = await fileSize(fsApi, paths.partial);
    if (partialSize !== null && partialSize <= artifact.size
        && sameIdentity(await readJson(fsApi, paths.identity), artifact)) {
        return { state: 'partial', bytes: partialSize };
    }
    return { state: 'absent', bytes: 0 };
}

export async function inspectArtifact({ root, artifact }) {
    return inspectWith(fs, artifactPaths({ root, artifact }), artifact);
}

async function removePartial(fsApi, paths) {
    await fsApi.promises.rm(paths.partial, { force: true });
    await fsApi.promises.rm(paths.identity, { force: true });
}

// A partial is kept only when its identity record matches this artifact and
// it is not longer than the artifact; anything else restarts from zero.
async function reconcilePartial(fsApi, paths, artifact) {
    const have = await fileSize(fsApi, paths.partial);
    if (have === null) {
        await fsApi.promises.rm(paths.identity, { force: true });
        return 0;
    }
    if (have > artifact.size || !sameIdentity(await readJson(fsApi, paths.identity), artifact)) {
        await removePartial(fsApi, paths);
        return 0;
    }
    return have;
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

async function assertFreeSpace({ fsApi, statfs, dir, remaining }) {
    const stats = await statfs(await nearestExistingAncestor(fsApi, dir));
    const available = Number(stats.bavail) * Number(stats.bsize);
    const required = Math.ceil(remaining * SPACE_MARGIN);
    if (available < required) {
        throw new DownloadError('INSUFFICIENT_SPACE', 'Not enough free space for the model weights', {
            details: { required, available },
        });
    }
}

// Re-hashing a 17 GB partial takes a while, so a Stop is honoured per chunk.
async function hashExisting(fsApi, partial, have, signal) {
    const hash = crypto.createHash('sha256');
    if (have === 0) {
        return hash;
    }
    const stream = fsApi.createReadStream(partial, { start: 0, end: have - 1 });
    for await (const chunk of stream) {
        throwIfAborted(signal);
        hash.update(chunk);
    }
    return hash;
}

function abortedError() {
    return new DownloadError('ABORTED', 'Download stopped; the partial file is kept for resume', { retryable: true });
}

function throwIfAborted(signal) {
    if (signal?.aborted) {
        throw abortedError();
    }
}

function createProgress({ total, onProgress, intervalMs }) {
    const samples = [];
    let lastEmit = -Infinity;
    const emit = (bytes, transferred, now) => {
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
            onProgress({ bytes, total, rate, etaSeconds, transferred });
        } catch {
            // A failing progress consumer must not corrupt the transfer.
        }
    };
    return {
        tick(bytes, transferred) {
            const now = performance.now();
            if (now - lastEmit >= intervalMs) {
                emit(bytes, transferred, now);
            }
        },
        finish(bytes, transferred) {
            emit(bytes, transferred, performance.now());
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
    ctx.hash = crypto.createHash('sha256');
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
        throw new DownloadError('NOT_FOUND', 'Model file not found at the pinned commit', { details: { status } });
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
        const response = await ctx.fetchImpl(resolveUrl(ctx.baseUrl, artifact), {
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

async function finalize(ctx, paths) {
    const digest = ctx.hash.digest('hex');
    if (digest !== ctx.artifact.sha256) {
        await removePartial(ctx.fsApi, paths);
        throw new DownloadError('SHA256_MISMATCH', 'Downloaded bytes do not match the pinned sha256', {
            details: { expected: ctx.artifact.sha256, actual: digest },
        });
    }
    // Meta goes first: a crash before the rename leaves a verifiable partial,
    // never a final file without its identity record.
    await writeJsonAtomic(ctx.fsApi, paths.meta, {
        ...identityOf(ctx.artifact),
        verifiedAt: new Date().toISOString(),
    });
    await ctx.fsApi.promises.rename(paths.partial, paths.file);
    await ctx.fsApi.promises.rm(paths.identity, { force: true });
}

async function runTransfer(ctx, paths) {
    try {
        await transfer(ctx);
        try {
            await ctx.handle.sync();
        } catch (err) {
            throw new LocalWriteError(err);
        }
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
    ctx.progress.finish(ctx.have, ctx.transferred);
    await finalize(ctx, paths);
}

export async function downloadArtifact({
    artifact,
    root,
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
}) {
    const paths = artifactPaths({ root, artifact });
    throwIfAborted(signal);
    const current = await inspectWith(fsApi, paths, artifact);
    if (current.state === 'complete') {
        await fsApi.promises.rm(paths.identity, { force: true });
        return { status: 'complete', path: paths.file, bytesTransferred: 0 };
    }
    const have = await reconcilePartial(fsApi, paths, artifact);
    await assertFreeSpace({ fsApi, statfs, dir: paths.dir, remaining: artifact.size - have });
    await fsApi.promises.mkdir(paths.dir, { recursive: true });
    await writeJsonAtomic(fsApi, paths.identity, identityOf(artifact));
    const ctx = {
        artifact,
        token,
        fetchImpl,
        baseUrl,
        signal,
        fsApi,
        maxAttempts,
        backoffMs,
        sleep,
        have,
        transferred: 0,
        hash: await hashExisting(fsApi, paths.partial, have, signal),
        handle: await fsApi.promises.open(paths.partial, 'a'),
        progress: createProgress({ total: artifact.size, onProgress, intervalMs: progressIntervalMs }),
    };
    await runTransfer(ctx, paths);
    return { status: 'complete', path: paths.file, bytesTransferred: ctx.transferred };
}

async function removeFile(filePath) {
    try {
        const stats = await fs.promises.lstat(filePath);
        await fs.promises.rm(filePath, { force: true });
        return stats.isFile() ? stats.size : 0;
    } catch (err) {
        if (err.code === 'ENOENT') {
            return 0;
        }
        throw err;
    }
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

export async function removeArtifact({ root, artifact }) {
    const paths = artifactPaths({ root, artifact });
    let freed = 0;
    for (const target of [paths.file, paths.partial, paths.identity, paths.meta]) {
        freed += await removeFile(target);
    }
    await removeEmptyParents(path.resolve(root), paths.dir);
    return freed;
}
