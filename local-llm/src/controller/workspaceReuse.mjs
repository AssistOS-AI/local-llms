// Verified model files and their reuse from the workspace (C12, DS002).
//
// Trust model: the agent's store under /data is written only by the agent and
// the operator, and its records are a cache, not authentication. /shared is
// writable by every agent, so everything in it is untrusted input: a candidate
// found there is used only through a private copy that was hashed as it was
// written. Hashing proves the bytes it read; the stat identity recorded next to
// a verified file only detects accidental change afterwards.
//
// Every file is opened without following a final symbolic link, must be a
// regular file, and must be, according to /proc/self/fd, exactly the path
// that was asked for under its root: a symbolic link anywhere below the root
// makes it unusable. The bytes are then read through that descriptor.

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
const { O_RDONLY, O_WRONLY, O_RDWR, O_CREAT, O_EXCL, O_APPEND, O_NOFOLLOW } = fs.constants;

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

// Where the kernel must say an opened descriptor points: the path itself,
// with the root resolved once (the root may be a configured link).
async function boundPath(fsApi, root, file) {
    const relative = path.relative(root, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
        throw unsafePath('A model file path escapes its directory');
    }
    return path.join(await fsApi.promises.realpath(root), relative);
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

/**
 * Open an existing regular file for reading, bound to `file` under `root`.
 * Returns { handle, stat } or { rejected } when the path is missing, a
 * symbolic link, not a regular file, or reached through a symbolic link.
 */
export async function openBound(fsApi, root, file, flags = O_RDONLY) {
    let expected;
    try {
        expected = await boundPath(fsApi, root, file);
    } catch (error) {
        // No store yet, so nothing in it.
        if (error.code === 'ENOENT') return { rejected: 'missing' };
        throw error;
    }
    let handle;
    try {
        handle = await fsApi.promises.open(file, flags | O_NOFOLLOW);
    } catch (error) {
        if (error.code === 'ELOOP') return { rejected: 'symlink' };
        if (['ENOENT', 'ENOTDIR', 'EACCES', 'EPERM', 'ENXIO', 'EISDIR'].includes(error.code)) return { rejected: 'missing' };
        throw error;
    }
    try {
        const stat = await handle.stat();
        if (!stat.isFile()) {
            await handle.close();
            return { rejected: 'not a regular file' };
        }
        if (!(await confirmBound(fsApi, handle, expected))) {
            await handle.close();
            return { rejected: 'reached through a symbolic link' };
        }
        return { handle, stat };
    } catch (error) {
        await handle.close().catch(() => {});
        throw error;
    }
}

/** Create a new file (never an existing one or a link), bound to `file` under `root`. */
export async function createBound(fsApi, root, file) {
    const expected = await boundPath(fsApi, root, file);
    const handle = await fsApi.promises.open(file, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o644);
    if (!(await confirmBound(fsApi, handle, expected))) {
        await handle.close().catch(() => {});
        throw unsafePath('A model staging file was created through a symbolic link');
    }
    return handle;
}

/** A download's partial file: appended to and read back through one bound descriptor. */
export async function openPartialBound(fsApi, root, file) {
    const expected = await boundPath(fsApi, root, file);
    const handle = await fsApi.promises.open(file, O_RDWR | O_APPEND | O_CREAT | O_NOFOLLOW, 0o644);
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
 * in abortable chunks, hashing the bytes as they are written (R6). Returns
 * { staged } with the staging file's stat once it matches, or { reason } when
 * this candidate is not the pinned file. Local write and read failures are
 * errors, never "no candidate". The caller removes the staging file on
 * anything but success.
 */
export async function copyToStaging(fsApi, { source, opened, root, staging, artifact, signal, chunkBytes = CHUNK_BYTES, onBytes }) {
    const { handle: src, stat } = opened;
    const before = statIdentity(stat);
    const dst = await createBound(fsApi, root, staging);
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
        return { staged: statIdentity(await dst.stat()) };
    } finally {
        await dst.close().catch(() => {});
    }
}

/**
 * Move a verified staging (or partial) file to its final path and record the
 * identity of the object that was hashed. The record is written without a
 * stat first, so a crash leaves a file that is checked again; the stat is
 * added only when the final path holds that same object (only its ctime,
 * changed by the rename, may differ).
 */
export async function publishVerified(fsApi, { root, staged, from, to, meta, record, writeJson }) {
    await ensureDirUnder(fsApi, root, path.dirname(to));
    await ensureDirUnder(fsApi, root, path.dirname(meta));
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
 * files of every wanted size (C12). Directory links are never followed; a
 * directory swapped for a link during the walk is caught later, when the
 * candidate's descriptor does not match its path. Returns a Map from size to
 * [{ path, root }].
 */
export async function walkShared(fsApi, roots, sizes, signal) {
    const found = new Map([...new Set(sizes)].map((size) => [size, []]));
    let seen = 0;
    const walk = async (root, dir, depth) => {
        if (depth > WALK_MAX_DEPTH || seen >= WALK_MAX_ENTRIES) return;
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
                    await walk(root, full, depth + 1);
                } else if (entry.isFile()) {
                    let stats;
                    try {
                        stats = await fsApi.promises.lstat(full);
                    } catch {
                        continue;
                    }
                    if (stats.isFile() && found.has(stats.size)) found.get(stats.size).push({ path: full, root });
                }
            }
        } finally {
            // Returning early from the loop leaves the directory open.
            await handle.close().catch(() => {});
        }
    };
    for (const root of roots) {
        throwIfAborted(signal);
        let stats;
        try {
            stats = await fsApi.promises.lstat(root);
        } catch {
            continue;
        }
        if (stats.isDirectory()) await walk(root, root, 0);
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
