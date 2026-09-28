// Verified model files and their reuse from the workspace (C12, DS002).
//
// Trust model: the agent's store under /data is written only by the agent and
// the operator, and its records are a cache, not authentication. /shared is
// writable by every agent, so everything in it is untrusted input: a candidate
// found there is used only through a private copy that was hashed as it was
// written. Hashing proves the bytes it read; the stat identity recorded next to
// a verified file only detects accidental change afterwards.
//
// Every root is anchored once per operation: its resolved path and directory
// identity (an untrusted root must be a real directory, not a link). Every
// file is opened without following a final symbolic link and without
// blocking (a FIFO or device cannot hold the open), must be a regular file,
// and must be, according to /proc/self/fd, exactly the path that was asked
// for under the anchored root: a symbolic link anywhere below the root makes
// it unusable. Below an untrusted root the open starts from a descriptor of
// the root whose own identity was checked, one directory at a time, so a
// root swapped after that check cannot supply the file. The bytes are then
// read through that descriptor.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { DownloadError, throwIfAborted } from './downloadError.mjs';

export const CHUNK_BYTES = 8 * 1024 * 1024;
// Candidates hashed per file from /shared/models; names only decide the order.
export const MAX_CANDIDATES_PER_FILE = 3;
const WALK_MAX_DEPTH = 8;
const WALK_MAX_ENTRIES = 20_000;
const GIT_OID_RE = /^[a-f0-9]{40}$/;
const { O_RDONLY, O_RDWR, O_CREAT, O_EXCL, O_APPEND, O_NOFOLLOW, O_NONBLOCK, O_DIRECTORY } = fs.constants;
// An open that fails with one of these means there is no usable file there.
const MISSING_CODES = ['ENOENT', 'ENOTDIR', 'EACCES', 'EPERM', 'ENXIO', 'EISDIR', 'EWOULDBLOCK', 'EAGAIN'];

// The running digest a file is verified with: sha256, or the git blob oid
// (sha1 over "blob <size>\0" and the bytes) for a small file kept in git.
export function createDigest(artifact) {
    if (artifact.sha256 === undefined && GIT_OID_RE.test(String(artifact.gitOid || ''))) {
        return crypto.createHash('sha1').update(`blob ${artifact.size}\0`);
    }
    return crypto.createHash('sha256');
}

export function expectedDigest(artifact) {
    return artifact.sha256 ?? artifact.gitOid;
}

export function statIdentity(stats) {
    return { dev: stats.dev, ino: stats.ino, size: stats.size, mtimeMs: stats.mtimeMs, ctimeMs: stats.ctimeMs };
}

export function sameStat(recorded, current) {
    return Boolean(recorded) && ['dev', 'ino', 'size', 'mtimeMs', 'ctimeMs'].every((key) => recorded[key] === current[key]);
}

// The same object with the same content: everything but ctime, which the
// agent's own rename changes.
function sameObject(before, after) {
    return ['dev', 'ino', 'size', 'mtimeMs'].every((key) => before[key] === after[key]);
}

function unsafePath(message) {
    return new DownloadError('UNSAFE_PATH', message);
}

/**
 * Anchor a root for one operation (F1): its resolved path and directory
 * identity, taken once and never re-resolved. An untrusted root (/shared)
 * must itself be a directory, not a symbolic link; the agent's own store root
 * is trusted and may be a link the operator configured. Returns null when
 * there is no usable root.
 */
export async function anchorRoot(fsApi, root, { untrusted = false } = {}) {
    let own;
    try {
        own = await fsApi.promises.lstat(root);
    } catch (error) {
        if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return null;
        throw error;
    }
    if (untrusted && !own.isDirectory()) return null;
    const real = await fsApi.promises.realpath(root);
    const target = await fsApi.promises.stat(real);
    if (!target.isDirectory()) return null;
    if (untrusted && (target.dev !== own.dev || target.ino !== own.ino)) return null;
    return Object.freeze({ path: path.resolve(root), real, dev: target.dev, ino: target.ino, untrusted });
}

// The trusted store root, created if needed, anchored for this operation.
export async function anchorStore(fsApi, root) {
    await fsApi.promises.mkdir(root, { recursive: true });
    const anchor = await anchorRoot(fsApi, root);
    if (!anchor) throw unsafePath('The model store root is not a directory');
    return anchor;
}

// Where the kernel must say an opened descriptor points: the path itself,
// under the anchored real root.
function boundPath(anchor, file) {
    const relative = path.relative(anchor.path, path.resolve(file));
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
        throw unsafePath('A model file path escapes its directory');
    }
    return path.join(anchor.real, relative);
}

async function confirmBound(fsApi, handle, expected) {
    let actual;
    try {
        actual = await fsApi.promises.readlink(`/proc/self/fd/${handle.fd}`);
    } catch {
        // Without /proc the opened object cannot be tied to its path: fail closed.
        throw unsafePath('Cannot confirm which file was opened (/proc/self/fd is not readable)');
    }
    return actual === expected;
}

// `name` inside an opened directory, the way openat(2) names it: through the
// kernel's link to that directory, so no rename or swap of a path above it
// changes which directory the name is looked up in.
function inDirectory(dir, name) {
    return `/proc/self/fd/${dir.fd}/${name}`;
}

/**
 * A descriptor of an untrusted root that is the anchored directory itself
 * (C4): opened without following a link, its dev and inode checked on the
 * descriptor, and /proc/self/fd showing it at its anchored real path. Returns
 * null when the root is no longer that directory there.
 */
async function openAnchoredRoot(fsApi, anchor) {
    let handle;
    try {
        handle = await fsApi.promises.open(anchor.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK);
    } catch (error) {
        if (error.code === 'ELOOP' || MISSING_CODES.includes(error.code)) return null;
        throw error;
    }
    try {
        const stat = await handle.stat();
        // confirmBound also fails closed when /proc/self/fd cannot be read, before any name is looked up through it.
        if (stat.isDirectory() && stat.dev === anchor.dev && stat.ino === anchor.ino && await confirmBound(fsApi, handle, anchor.real)) {
            return handle;
        }
    } catch (error) {
        await handle.close().catch(() => {});
        throw error;
    }
    await handle.close().catch(() => {});
    return null;
}

/**
 * Open `relative` below an untrusted root's checked descriptor, one directory
 * at a time, none of them through a symbolic link. Returns { handle } or
 * { rejected }.
 */
async function openBelowRoot(fsApi, anchor, relative, signal) {
    const root = await openAnchoredRoot(fsApi, anchor);
    if (!root) return { rejected: 'its root directory changed' };
    const parts = relative.split(path.sep);
    let dir = root;
    try {
        for (const part of parts.slice(0, -1)) {
            throwIfAborted(signal);
            let next;
            try {
                next = await fsApi.promises.open(inDirectory(dir, part), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_NONBLOCK);
            } catch (error) {
                // Linux reports a link opened with O_DIRECTORY | O_NOFOLLOW as ENOTDIR; lstat only names the reason.
                const link = error.code === 'ENOTDIR'
                    && await fsApi.promises.lstat(inDirectory(dir, part)).then((stats) => stats.isSymbolicLink(), () => false);
                if (error.code === 'ELOOP' || link) return { rejected: 'reached through a symbolic link' };
                if (MISSING_CODES.includes(error.code)) return { rejected: 'missing' };
                throw error;
            }
            if (dir !== root) await dir.close().catch(() => {});
            dir = next;
        }
        throwIfAborted(signal);
        try {
            return { handle: await fsApi.promises.open(inDirectory(dir, parts.at(-1)), O_RDONLY | O_NOFOLLOW | O_NONBLOCK) };
        } catch (error) {
            if (error.code === 'ELOOP') return { rejected: 'symlink' };
            if (MISSING_CODES.includes(error.code)) return { rejected: 'missing' };
            throw error;
        }
    } finally {
        if (dir !== root) await dir.close().catch(() => {});
        await root.close().catch(() => {});
    }
}

async function openPath(fsApi, file) {
    try {
        return { handle: await fsApi.promises.open(file, O_RDONLY | O_NOFOLLOW | O_NONBLOCK) };
    } catch (error) {
        if (error.code === 'ELOOP') return { rejected: 'symlink' };
        if (MISSING_CODES.includes(error.code)) return { rejected: 'missing' };
        throw error;
    }
}

/**
 * Open an existing regular file for reading, bound to `file` under the
 * anchored root, without following a final link or blocking on a special
 * file (F2). Below an untrusted root the file is looked up from the root's
 * checked descriptor (C4). Returns { handle, stat } or { rejected } when the
 * path is missing, a symbolic link, not a regular file, reached through a
 * symbolic link, or its untrusted root changed.
 */
export async function openBound(fsApi, anchor, file, { signal } = {}) {
    throwIfAborted(signal);
    const expected = boundPath(anchor, file);
    const opened = anchor.untrusted
        ? await openBelowRoot(fsApi, anchor, path.relative(anchor.path, path.resolve(file)), signal)
        : await openPath(fsApi, file);
    if (!opened.handle) return opened;
    const { handle } = opened;
    try {
        throwIfAborted(signal);
        const stat = await handle.stat();
        if (!stat.isFile()) {
            await handle.close();
            return { rejected: 'not a regular file' };
        }
        if (!(await confirmBound(fsApi, handle, expected))) {
            await handle.close();
            return { rejected: anchor.untrusted ? 'it or its root directory moved while it was opened' : 'reached through a symbolic link' };
        }
        return { handle, stat };
    } catch (error) {
        await handle.close().catch(() => {});
        throw error;
    }
}

/**
 * Create a new file (never an existing one or a link), bound to `file` under
 * the anchored store, open for writing and for reading back what was written.
 * The descriptor is the caller's only when it is returned; on any failure,
 * including an unreadable /proc/self/fd, it is closed here.
 */
export async function createBound(fsApi, anchor, file) {
    const expected = boundPath(anchor, file);
    const handle = await fsApi.promises.open(file, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_NONBLOCK, 0o644);
    try {
        if (!(await confirmBound(fsApi, handle, expected))) {
            throw unsafePath('A model staging file was created through a symbolic link');
        }
        return handle;
    } catch (error) {
        await handle.close().catch(() => {});
        throw error;
    }
}

/** A download's partial file: appended to and read back through one bound descriptor. */
export async function openPartialBound(fsApi, anchor, file) {
    const expected = boundPath(anchor, file);
    const handle = await fsApi.promises.open(file, O_RDWR | O_APPEND | O_CREAT | O_NOFOLLOW | O_NONBLOCK, 0o644);
    try {
        const stat = await handle.stat();
        if (!stat.isFile() || !(await confirmBound(fsApi, handle, expected))) {
            throw unsafePath('A partial model file is not a regular file in the model store');
        }
        return handle;
    } catch (error) {
        await handle.close().catch(() => {});
        throw error;
    }
}

/**
 * Create `dir` and every directory between `root` and it, refusing any that
 * is a symbolic link or not a directory, so a staged or final file cannot be
 * placed outside the store (R3).
 */
export async function ensureDirUnder(fsApi, root, dir) {
    await fsApi.promises.mkdir(root, { recursive: true });
    const relative = path.relative(root, dir);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        throw unsafePath('A model directory escapes the model store');
    }
    let current = root;
    for (const part of relative ? relative.split(path.sep) : []) {
        current = path.join(current, part);
        try {
            await fsApi.promises.mkdir(current);
        } catch (error) {
            if (error.code !== 'EEXIST') throw error;
        }
        const stats = await fsApi.promises.lstat(current);
        if (!stats.isDirectory()) {
            throw unsafePath(`${path.relative(root, current)} in the model store is a symbolic link or not a directory; remove it`);
        }
    }
}

/**
 * Refuse a symbolic link, or anything but a directory, among the existing
 * directories between `root` and `dir`; creates nothing (R2).
 */
export async function assertRealDirs(fsApi, root, dir) {
    const relative = path.relative(root, dir);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        throw unsafePath('A model directory escapes the model store');
    }
    let current = root;
    for (const part of relative ? relative.split(path.sep) : []) {
        current = path.join(current, part);
        let stats;
        try {
            stats = await fsApi.promises.lstat(current);
        } catch (error) {
            if (error.code === 'ENOENT') return;
            throw error;
        }
        if (!stats.isDirectory()) {
            throw unsafePath(`${path.relative(root, current)} in the model store is a symbolic link or not a directory; remove it`);
        }
    }
}

/**
 * lstat of a regular file whose every directory below `root` is a real
 * directory; null for anything else (missing, a link, a link on the way).
 * For the state shown to the user; loading always goes through openBound.
 */
export async function regularFileUnder(fsApi, root, file) {
    const relative = path.relative(root, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return null;
    const parts = relative.split(path.sep);
    let current = root;
    try {
        for (let index = 0; index < parts.length; index += 1) {
            current = path.join(current, parts[index]);
            const stats = await fsApi.promises.lstat(current);
            const last = index === parts.length - 1;
            if (last ? !stats.isFile() : !stats.isDirectory()) return null;
            if (last) return stats;
        }
    } catch {
        return null;
    }
    return null;
}

/** Continue `hash` with the first `length` bytes read through `handle`. */
export async function hashPrefix(handle, length, hash, { signal, chunkBytes = CHUNK_BYTES, onBytes } = {}) {
    if (length <= 0) return 0;
    const buffer = Buffer.allocUnsafe(Math.min(chunkBytes, length));
    let position = 0;
    while (position < length) {
        throwIfAborted(signal);
        const { bytesRead } = await handle.read(buffer, 0, Math.min(buffer.length, length - position), position);
        if (bytesRead === 0) break;
        hash.update(buffer.subarray(0, bytesRead));
        position += bytesRead;
        onBytes?.(position);
    }
    return position;
}

/**
 * Hash a whole file through its descriptor. The digest counts only when the
 * file had the pinned size and an identical fstat before and after (R1).
 */
export async function hashBound(handle, artifact, options = {}) {
    const before = statIdentity(await handle.stat());
    const hash = createDigest(artifact);
    const read = await hashPrefix(handle, artifact.size, hash, options);
    throwIfAborted(options.signal);
    const after = statIdentity(await handle.stat());
    const stable = before.size === artifact.size && read === artifact.size && sameStat(before, after);
    return { stable, matches: stable && hash.digest('hex') === expectedDigest(artifact), stat: after };
}

async function writeFully(handle, chunk) {
    let offset = 0;
    while (offset < chunk.length) {
        const { bytesWritten } = await handle.write(chunk, offset, chunk.length - offset);
        offset += bytesWritten;
    }
}

function copyWriteError(error) {
    if (error?.code === 'ENOSPC') {
        return new DownloadError('PAUSED_ENOSPC', 'Disk is full while copying the model file; the copy was removed '
            + 'and any partial download is kept', { retryable: true });
    }
    return new DownloadError('COPY_FAILED', `Writing the private copy failed (${error?.code || 'error'})`, {
        details: { code: error?.code ?? null },
    });
}

/**
 * Copy an opened /shared candidate into a new staging file under the store,
 * in abortable chunks, hashing the bytes as they are read (R6). The source's
 * digest proves only what was read from it: the staging file is then hashed
 * again through its own descriptor, with an identical fstat before and after,
 * so what is published is the bytes on disk (C3). Returns { staged, handle }
 * once the staging file matches (the caller closes `handle` after publishing
 * through it), or { reason } when this candidate is not the pinned file.
 * Local write and read failures, and a staging file that does not hold what
 * was written, are errors, never "no candidate". The caller removes the
 * staging file on anything but success.
 */
export async function copyToStaging(fsApi, { source, opened, anchor, staging, artifact, signal, chunkBytes = CHUNK_BYTES, onBytes, onVerifyBytes }) {
    const { handle: src, stat } = opened;
    const before = statIdentity(stat);
    const dst = await createBound(fsApi, anchor, staging);
    let keep = false;
    try {
        const hash = createDigest(artifact);
        const buffer = Buffer.allocUnsafe(Math.min(chunkBytes, artifact.size));
        let position = 0;
        while (position < artifact.size) {
            throwIfAborted(signal);
            let bytesRead;
            try {
                ({ bytesRead } = await src.read(buffer, 0, Math.min(buffer.length, artifact.size - position), position));
            } catch (error) {
                throw new DownloadError('COPY_FAILED', `Reading ${source} failed (${error.code || 'error'})`, {
                    details: { code: error.code ?? null },
                });
            }
            if (bytesRead === 0) return { reason: 'it became shorter while it was copied' };
            const chunk = buffer.subarray(0, bytesRead);
            try {
                await writeFully(dst, chunk);
            } catch (error) {
                throw copyWriteError(error);
            }
            hash.update(chunk);
            position += bytesRead;
            onBytes?.(position);
        }
        throwIfAborted(signal);
        // A source that changed while it was read is not used, even if the copy happens to match.
        if (!sameStat(before, statIdentity(await src.stat()))) return { reason: 'it changed while it was copied' };
        if (hash.digest('hex') !== expectedDigest(artifact)) return { reason: 'its bytes do not match the pinned digest' };
        try {
            await dst.sync();
        } catch (error) {
            throw copyWriteError(error);
        }
        let checked;
        try {
            checked = await hashBound(dst, artifact, { signal, chunkBytes, onBytes: onVerifyBytes });
        } catch (error) {
            if (error instanceof DownloadError) throw error;
            throw new DownloadError('COPY_FAILED', `Reading back the private copy failed (${error?.code || 'error'})`, {
                details: { code: error?.code ?? null },
            });
        }
        if (!checked.stable) {
            throw new DownloadError('CHANGED_WHILE_VERIFYING', 'The private copy changed while it was being verified; '
                + 'the copy was removed and any partial download is kept; run again', { retryable: true });
        }
        if (!checked.matches) {
            throw new DownloadError('SHA256_MISMATCH', 'The private copy on disk does not match the pinned digest that its '
                + 'source matched; the copy was removed and any partial download is kept', {
                details: { expected: expectedDigest(artifact), actual: null },
            });
        }
        keep = true;
        return { staged: checked.stat, handle: dst };
    } finally {
        if (!keep) await dst.close().catch(() => {});
    }
}

/**
 * Move a verified staging (or partial) file to its final path and record the
 * identity of the object that was hashed. The record is written without a
 * stat first, so a crash leaves a file that is checked again; the stat is
 * added only when the final path holds that same object (only its ctime,
 * changed by the rename, may differ). With `handle`, the descriptor the bytes
 * were hashed through, nothing is recorded unless its fstat is still exactly
 * the one that was verified.
 */
export async function publishVerified(fsApi, { root, staged, from, to, meta, record, writeJson, handle = null }) {
    await ensureDirUnder(fsApi, root, path.dirname(to));
    await ensureDirUnder(fsApi, root, path.dirname(meta));
    if (handle && !sameStat(staged, statIdentity(await handle.stat()))) {
        throw new DownloadError('CHANGED_WHILE_VERIFYING', 'The model file changed while it was being verified; run again', {
            retryable: true,
        });
    }
    await writeJson(meta, record);
    await fsApi.promises.rename(from, to);
    const placed = await fsApi.promises.lstat(to);
    const now = statIdentity(placed);
    if (!placed.isFile() || !sameObject(staged, now)) {
        await fsApi.promises.rm(meta, { force: true });
        throw new DownloadError('CHANGED_WHILE_VERIFYING', 'The model file changed while it was being verified; run again', {
            retryable: true,
        });
    }
    await writeJson(meta, { ...record, stat: now });
}

/**
 * One streaming, cancellable walk of the shared roots that collects regular
 * files of every wanted size (C12). Each root is anchored first (F1); a root
 * that is a link, or missing, is skipped. Directory links are never
 * followed; a directory or root swapped for a link during the walk is caught
 * when the candidate's descriptor does not match its anchored path. The
 * signal is checked before every directory is opened and at every entry.
 * Returns a Map from size to [{ path, anchor }].
 */
export async function walkShared(fsApi, roots, sizes, signal) {
    const found = new Map([...new Set(sizes)].map((size) => [size, []]));
    let seen = 0;
    const walk = async (anchor, dir, depth) => {
        if (depth > WALK_MAX_DEPTH || seen >= WALK_MAX_ENTRIES) return;
        throwIfAborted(signal);
        let handle;
        try {
            handle = await fsApi.promises.opendir(dir);
        } catch {
            return;
        }
        try {
            for await (const entry of handle) {
                throwIfAborted(signal);
                if (++seen > WALK_MAX_ENTRIES) return;
                const full = path.join(dir, entry.name);
                if (entry.isDirectory()) {
                    await walk(anchor, full, depth + 1);
                } else if (entry.isFile()) {
                    let stats;
                    try {
                        stats = await fsApi.promises.lstat(full);
                    } catch {
                        continue;
                    }
                    if (stats.isFile() && found.has(stats.size)) found.get(stats.size).push({ path: full, anchor });
                }
            }
        } finally {
            // Returning early from the loop leaves the directory open.
            await handle.close().catch(() => {});
        }
    };
    for (const root of roots) {
        throwIfAborted(signal);
        const anchor = await anchorRoot(fsApi, root, { untrusted: true });
        if (anchor) await walk(anchor, anchor.path, 0);
    }
    return found;
}

/** Candidates for one file: those with the pinned file's name first. */
export function orderCandidates(candidates, file) {
    const name = path.posix.basename(file);
    const list = candidates || [];
    return [...list.filter((entry) => path.basename(entry.path) === name), ...list.filter((entry) => path.basename(entry.path) !== name)];
}

/**
 * The logical bytes a deletion frees (R8): the size of each distinct inode
 * whose every link is in `stats` (the lstat of each regular file about to be
 * removed). An inode with a link elsewhere frees nothing.
 */
export function freedBytes(stats) {
    const inodes = new Map();
    for (const entry of stats) {
        if (!entry?.isFile()) continue;
        const key = `${entry.dev}:${entry.ino}`;
        const seen = inodes.get(key);
        if (seen) seen.links += 1;
        else inodes.set(key, { links: 1, nlink: entry.nlink, size: entry.size });
    }
    let total = 0;
    for (const inode of inodes.values()) {
        if (inode.links >= inode.nlink) total += inode.size;
    }
    return total;
}

/** lstat of every regular file under `dir` (links are not followed). */
export async function regularFilesIn(fsApi, dir) {
    const out = [];
    let entries;
    try {
        entries = await fsApi.promises.readdir(dir, { withFileTypes: true });
    } catch {
        return out;
    }
    for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) out.push(...await regularFilesIn(fsApi, full));
        else if (entry.isFile()) {
            try {
                out.push(await fsApi.promises.lstat(full));
            } catch {}
        }
    }
    return out;
}
