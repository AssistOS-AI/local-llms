// On-demand runner installs (runners plan §5.2).
//
//   cache     /data/runners/<id>/<version>/files/<name>   verified downloads;
//             /data/runners/<id>/<version>/installed.json  the install record.
//             The only part that persists.
//   runnable  /opt/runners/<id>/<version>/                in the container's own
//             filesystem, rebuilt from the cache without network access before
//             the first launch in a container; .ready.json is written last.
//
// Before every rebuild each cached file is copied into container-local
// staging and hashed as it is copied; a file that differs from the lock blocks
// the rebuild, and the build reads only the staged copies, so a file swapped
// in /data after the check never reaches the runnable copy. Code changed in
// the runnable copy lasts only as long as the container, like baked-in code.

import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import zlib from 'node:zlib';

import { LocalLlmError } from '../errors.mjs';
import { DownloadError, downloadFile, inspectFile } from './downloader.mjs';
import { archiveCompressionFor } from './runnerLock.mjs';

export const DEFAULT_CACHE_ROOT = '/data/runners';
export const DEFAULT_RUN_ROOT = '/opt/runners';
export const DEFAULT_UV = '/usr/local/bin/uv';
export const DEFAULT_PYTHON = '/usr/bin/python3';
const SPACE_MARGIN = 1.05;
const TOOL_OUTPUT_KEPT = 64 * 1024;
// A .tar.zst may unpack to this many times its own size, and to at least the floor: a pinned file is trusted
// bytes, so the cap only keeps a bad one from filling the container's disk (Ollama's is about 2.7 times).
const ZSTD_MAX_RATIO = 32;
const ZSTD_MIN_LIMIT = 1024 * 1024;
const TAR_BLOCK = 512;

const WHEEL_RE = /^([A-Za-z0-9](?:[A-Za-z0-9._]*[A-Za-z0-9])?)-([A-Za-z0-9.!+_]+)(?:-\d[^-]*)?-[^-]+-[^-]+-[^-]+\.whl$/;

/** `name==version --hash=sha256:…` for one wheel, from its file name. */
export function wheelRequirement(file) {
    const match = WHEEL_RE.exec(file.name);
    if (!match) throw new LocalLlmError('invalid_runner_lock', `${file.name} is not a wheel file name`);
    return `${match[1]}==${match[2]} --hash=sha256:${file.sha256}`;
}

// Copies a file and returns the sha256 of exactly the bytes written.
async function copyHashed(source, destination, signal) {
    const hash = crypto.createHash('sha256');
    await pipeline(
        fs.createReadStream(source),
        async function* hashing(stream) {
            for await (const chunk of stream) {
                if (signal?.aborted) throw new DownloadError('ABORTED', 'Stopped while verifying the runner cache', { retryable: true });
                hash.update(chunk);
                yield chunk;
            }
        },
        fs.createWriteStream(destination, { flags: 'wx', mode: 0o600 }),
    );
    return hash.digest('hex');
}

// Limits on what a tar's headers may claim, since the .tar comes from a decoder and its content is not yet trusted.
export const TAR_MAX_MEMBERS = 1_000_000;
export const TAR_MAX_PAX_BYTES = 1024 * 1024;
const PAX_TYPES = new Set(['x', 'g']);
const GNU_LONG_NAME_TYPES = new Set(['L', 'K']);
// Members that carry no data, whatever their size field says (GNU tar reads them so): links, devices, directories, fifos.
const NO_DATA_TYPES = new Set(['1', '2', '3', '4', '5', '6']);

const roundUp = (bytes) => Math.ceil(bytes / TAR_BLOCK) * TAR_BLOCK;

// A tar header number: octal text (padded with NULs or spaces), or GNU's base 256 (high bit of the first byte set). null if it is neither.
function tarNumber(field) {
    if (field[0] & 0x80) {
        if (field[0] & 0x40) return null;
        let value = BigInt(field[0] & 0x3f);
        for (let index = 1; index < field.length; index += 1) value = (value << 8n) | BigInt(field[index]);
        return value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : null;
    }
    const text = field.toString('latin1').replace(/\0.*$/s, '').trim();
    return /^[0-7]+$/.test(text) ? Number.parseInt(text, 8) : null;
}

// What is wrong with a header block, or null: the ustar magic (POSIX, GNU and pax all carry it) and the checksum (unsigned or signed sum).
function headerFault(block) {
    if (block.subarray(257, 262).toString('latin1') !== 'ustar') return 'the ustar magic is missing';
    const stored = tarNumber(block.subarray(148, 156));
    let unsigned = 0;
    let signed = 0;
    for (let index = 0; index < TAR_BLOCK; index += 1) {
        const byte = index >= 148 && index < 156 ? 0x20 : block[index];
        unsigned += byte;
        signed += byte > 127 ? byte - 256 : byte;
    }
    return stored !== null && (stored === unsigned || stored === signed) ? null : 'its checksum is wrong';
}

// The `size` a pax extended header sets for the member that follows, if it sets one: { size } (null if none), or { fault }.
function paxSize(data) {
    let size = null;
    for (let at = 0; at < data.length;) {
        const space = data.indexOf(0x20, at);
        const length = space < 0 ? NaN : Number.parseInt(data.subarray(at, space).toString('latin1'), 10);
        if (!Number.isInteger(length) || length <= space - at + 1 || at + length > data.length || data[at + length - 1] !== 0x0a) return { fault: 'a pax header is malformed' };
        const record = data.subarray(space + 1, at + length - 1).toString('utf8');
        const equals = record.indexOf('=');
        if (record.slice(0, equals) === 'size') {
            if (!/^[0-9]{1,15}$/.test(record.slice(equals + 1))) return { fault: 'a pax size is not a number' };
            size = Number(record.slice(equals + 1));
        }
        at += length;
    }
    return { size };
}

/**
 * Whether a file is a whole tar, by walking its headers from the start and never reading a member's data
 * (it seeks past). Each header must be a ustar header (magic, checksum) and its member is skipped by its
 * size: the size field (octal, or GNU base 256), or the `size=` record of a pax extended header (`x` for the
 * next member, `g` from there on); a GNU long name or link header (`L`, `K`) is skipped by its size too.
 * The tar is whole only when a block of zeros is found exactly where a header is due, followed by a second
 * one. The end of the file where a header is due, a member that runs past the end, one block of zeros, a
 * header that is not a header: each is `{ cut }` with what was found. More than `maxMembers` members or a
 * pax header over `maxPax` bytes is `{ limit }`. A whole tar is null. The tail of a tar is no test: a member
 * whose data ends in zeros, cut at its end, looks like the end of the archive.
 */
export async function tarProblem(file, { maxMembers = TAR_MAX_MEMBERS, maxPax = TAR_MAX_PAX_BYTES } = {}) {
    const { size } = await fs.promises.stat(file);
    const handle = await fs.promises.open(file, 'r');
    try {
        const block = Buffer.alloc(TAR_BLOCK);
        const readAt = (at, into, length = into.length) => handle.read(into, 0, length, at);
        let at = 0;
        let members = 0;
        let local = null;
        let global = null;
        for (;;) {
            if (at + TAR_BLOCK > size) return { cut: 'no end-of-archive blocks' };
            await readAt(at, block);
            if (block.every((byte) => byte === 0)) {
                if (at + 2 * TAR_BLOCK > size) return { cut: 'only one end-of-archive block' };
                const second = Buffer.alloc(TAR_BLOCK);
                await readAt(at + TAR_BLOCK, second);
                return second.every((byte) => byte === 0) ? null : { cut: 'a block of zeros is not followed by a second one' };
            }
            const fault = headerFault(block);
            if (fault) return { cut: `the header at byte ${at}: ${fault}` };
            members += 1;
            if (members > maxMembers) return { limit: `it lists more than ${maxMembers} members` };
            const type = String.fromCharCode(block[156]);
            const stated = tarNumber(block.subarray(124, 136));
            if (stated === null) return { cut: `the header at byte ${at}: its size is not a number` };
            if (PAX_TYPES.has(type)) {
                if (stated > maxPax) return { limit: `a pax header is larger than ${maxPax} bytes` };
                if (at + TAR_BLOCK + roundUp(stated) > size) return { cut: 'a pax header runs past the end of the file' };
                const data = Buffer.alloc(stated);
                await readAt(at + TAR_BLOCK, data);
                const pax = paxSize(data);
                if (pax.fault) return { cut: pax.fault };
                if (pax.size !== null) { if (type === 'x') local = pax.size; else global = pax.size; }
                at += TAR_BLOCK + roundUp(stated);
                continue;
            }
            const dataSize = NO_DATA_TYPES.has(type) ? 0 : (GNU_LONG_NAME_TYPES.has(type) ? stated : (local ?? global ?? stated));
            if (!GNU_LONG_NAME_TYPES.has(type)) local = null;
            if (dataSize > size) return { cut: 'a member runs past the end of the file' };
            at += TAR_BLOCK + roundUp(dataSize);
            if (at > size) return { cut: 'a member runs past the end of the file' };
        }
    } finally {
        await handle.close();
    }
}

/**
 * Decompress a pinned .tar.zst into a plain .tar beside it. The image has no
 * zstd tool and runTool gives a process no stdin, so Node's zlib does it and
 * tar then reads the .tar. A stream that is not zstd, or that unpacks to more
 * than ZSTD_MAX_RATIO times its size, is an `install_failed` with the partial
 * .tar removed; a stop ends it as ABORTED.
 *
 * A stream that is cut short is handled two ways, by Node's version. Node 24 reports it as a zlib
 * error (`Z_BUF_ERROR`, "unexpected end of file"), which is an `install_failed` like the rest. Node
 * 25.8 does not: its decoder ends quietly with whatever it decoded, down to an empty .tar, and it
 * decodes only the first frame of a stream of several. Tar takes a .tar that ends on a member's end for
 * a whole one, and the tail of a tar is no test (a last member may end in zeros). So the decompressed
 * .tar is walked header by header (tarProblem) and must end where a tar ends: two blocks of zeros where
 * a header is due. The sha256 pin covers the bytes on disk, and the executable check after the unpack
 * (assertLaunchExecutable) covers a wrong strip.
 */
async function decompressZstd(source, destination, { signal, label, decoder }) {
    const limit = Math.max(ZSTD_MIN_LIMIT, ZSTD_MAX_RATIO * (await fs.promises.stat(source)).size);
    let written = 0;
    const bounded = new Transform({
        transform(chunk, _encoding, done) {
            written += chunk.length;
            if (written > limit) done(Object.assign(new Error(`it unpacks to more than ${limit} bytes (${ZSTD_MAX_RATIO} times its size)`), { code: 'unpack_limit' }));
            else done(null, chunk);
        },
    });
    try {
        await pipeline(fs.createReadStream(source), decoder(), bounded, fs.createWriteStream(destination, { flags: 'wx', mode: 0o600 }), { signal });
    } catch (error) {
        await fs.promises.rm(destination, { force: true });
        if (signal?.aborted) throw new DownloadError('ABORTED', `Stopped while decompressing ${label}`, { retryable: true });
        const why = error?.code === 'unpack_limit' ? error.message
            : (error?.code && error?.message && error.code !== error.message ? `${error.code} (${error.message})` : (error?.code || error?.message || error));
        throw new LocalLlmError('install_failed', `decompressing ${label} failed: ${why}`);
    }
    // Quietly truncated output (Node 25) is not an error to the decoder, so the .tar is walked to its end.
    const problem = await tarProblem(destination);
    if (problem) {
        await fs.promises.rm(destination, { force: true });
        // A stream of several zstd frames, which Node 25.8 decodes only the first of, looks like a cut one: the message says both.
        throw new LocalLlmError('install_failed', `decompressing ${label} failed: ${problem.limit
            ?? `it was cut short (${problem.cut}), or has more zstd frames than this Node decodes`}`);
    }
}

/**
 * After an entry is unpacked: the file its adapter launches (`check.executable`, a path inside the
 * runnable copy) must be there, a regular file that is executable, and must not lead outside the copy
 * through a link. A wrong `strip` or a truncated archive gives a copy without it, and nothing else in
 * the unpack says so. An entry without the field (the image's lock may omit it) is not checked.
 * Resolves with the file's path, or null when nothing is declared.
 */
export async function assertLaunchExecutable(entry, runDir) {
    const relative = entry.check?.executable;
    if (!relative) return null;
    const file = path.join(runDir, relative);
    const fail = (why) => new LocalLlmError('install_failed', `The unpacked ${entry.id} ${entry.version} has no executable ${relative}: ${why}.`);
    let real;
    try {
        real = await fs.promises.realpath(file);
    } catch {
        throw fail('it is not in the runnable copy (a wrong strip, or an archive that was cut short)');
    }
    const root = await fs.promises.realpath(runDir);
    if (real !== root && !real.startsWith(`${root}${path.sep}`)) throw fail('it leads outside the runnable copy');
    const stats = await fs.promises.stat(real);
    if (!stats.isFile()) throw fail('it is not a regular file');
    // The mode says some execute bit is set; access() says the user that runs the agent can use it (mode 0645 has a bit and is no use to its owner).
    try {
        if ((stats.mode & 0o111) === 0) throw new Error('no execute bit');
        await fs.promises.access(real, fs.constants.X_OK);
    } catch {
        throw fail('it is not executable');
    }
    return file;
}

async function readJson(file) {
    try {
        return JSON.parse(await fs.promises.readFile(file, 'utf8'));
    } catch {
        return null;
    }
}

async function writeJson(file, value) {
    const temp = `${file}.tmp-${process.pid}`;
    await fs.promises.writeFile(temp, `${JSON.stringify(value, null, 2)}\n`);
    await fs.promises.rename(temp, file);
}

async function treeBytes(dir) {
    let total = 0;
    let entries;
    try {
        entries = await fs.promises.readdir(dir, { withFileTypes: true });
    } catch {
        return 0;
    }
    for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) total += await treeBytes(full);
        else if (entry.isFile()) total += (await fs.promises.lstat(full)).size;
    }
    return total;
}

/**
 * Run one install step (uv, tar) in its own process group, so an abort (Stop,
 * a drain) ends the whole step. Resolves with the exit code and output tail.
 */
export function runTool({ command, args, env, cwd = '/', signal, killGraceMs = 3000 }) {
    return new Promise((resolve) => {
        if (signal?.aborted) {
            resolve({ code: null, signal: 'SIGTERM', output: '', aborted: true });
            return;
        }
        let output = '';
        const child = spawn(command, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], detached: true });
        const keep = (chunk) => { output = (output + chunk.toString()).slice(-TOOL_OUTPUT_KEPT); };
        child.stdout.on('data', keep);
        child.stderr.on('data', keep);
        const group = (sig) => { try { process.kill(-child.pid, sig); } catch { try { child.kill(sig); } catch {} } };
        let killer = null;
        const onAbort = () => {
            group('SIGTERM');
            killer = setTimeout(() => group('SIGKILL'), killGraceMs);
        };
        signal?.addEventListener('abort', onAbort, { once: true });
        child.once('error', (error) => {
            signal?.removeEventListener('abort', onAbort);
            resolve({ code: null, signal: null, output: `${output}${error.message}`, aborted: false });
        });
        child.once('exit', (code, exitSignal) => {
            signal?.removeEventListener('abort', onAbort);
            clearTimeout(killer);
            group('SIGKILL');
            resolve({ code, signal: exitSignal, output, aborted: Boolean(signal?.aborted) });
        });
    });
}

export function createRunnerInstaller({
    lock,
    cacheRoot = DEFAULT_CACHE_ROOT,
    runRoot = DEFAULT_RUN_ROOT,
    uv = DEFAULT_UV,
    python = DEFAULT_PYTHON,
    download = downloadFile,
    inspect = inspectFile,
    statfs = (target) => fs.promises.statfs(target),
    run = runTool,
    // Removes a path; tests make it fail. `onWarning` hears of a cleanup that could not be done.
    remove = (target) => fs.promises.rm(target, { recursive: true, force: true }),
    onWarning = () => {},
    // The zstd decoder; tests stand in for how a Node version ends a stream that was cut short.
    zstdDecoder = () => zlib.createZstdDecompress(),
    // Tests only: the lock allows https, a local test server speaks http.
    allowHttp = false,
    now = () => new Date(),
} = {}) {
    function entryFor(id) {
        if (typeof id !== 'string' || !Object.hasOwn(lock.runners, id)) {
            throw new LocalLlmError('not_installable', `No installable runner '${String(id)}' in this image's runner lock.`);
        }
        return lock.runners[id];
    }

    function pathsFor(entry) {
        const cacheDir = path.join(cacheRoot, entry.id, entry.version);
        const runDir = path.join(runRoot, entry.id, entry.version);
        return {
            idCache: path.join(cacheRoot, entry.id),
            idRun: path.join(runRoot, entry.id),
            cacheDir,
            filesDir: path.join(cacheDir, 'files'),
            record: path.join(cacheDir, 'installed.json'),
            runDir,
            marker: path.join(runDir, '.ready.json'),
            tmpDir: path.join(runRoot, `.tmp-${entry.id}`),
            stageDir: path.join(runRoot, `.stage-${entry.id}`),
        };
    }

    function target(entry, file) {
        return path.join(pathsFor(entry).filesDir, file.name);
    }

    /** Download state of the cache: complete only when every file verified and the record exists. */
    async function cacheState(entry) {
        let bytes = 0;
        let complete = true;
        let partial = false;
        for (const file of entry.files) {
            const state = await inspect({ url: file.url, size: file.size, sha256: file.sha256, target: target(entry, file), allowHttp: true });
            bytes += state.bytes;
            if (state.state !== 'complete') complete = false;
            if (state.state !== 'absent') partial = true;
        }
        const record = await readJson(pathsFor(entry).record);
        const installed = complete && record?.digest === entry.digest;
        return {
            state: installed ? 'complete' : (partial ? 'partial' : 'absent'),
            bytes,
            total: entry.totalBytes,
            record: installed ? record : null,
        };
    }

    async function assertSpace(entry, remaining) {
        let dir = cacheRoot;
        while (!fs.existsSync(dir)) dir = path.dirname(dir);
        const stats = await statfs(dir);
        const available = Number(stats.bavail) * Number(stats.bsize);
        const required = Math.ceil(remaining * SPACE_MARGIN);
        if (available < required) {
            throw new DownloadError('INSUFFICIENT_SPACE', `Installing ${entry.id} needs ${required} bytes of free disk; ${available} are free.`, {
                details: { required, available },
            });
        }
    }

    /**
     * Download every lock file into the cache (resuming partial files), then
     * write the install record. Only URLs from the lock are ever fetched.
     */
    async function fetchAll(entry, { signal, onProgress = () => {}, licence = null } = {}) {
        const before = await cacheState(entry);
        await assertSpace(entry, entry.totalBytes - before.bytes);
        const paths = pathsFor(entry);
        await fs.promises.mkdir(paths.filesDir, { recursive: true });
        let done = 0;
        let transferred = 0;
        const started = Date.now();
        for (const file of entry.files) {
            const result = await download({
                url: file.url, size: file.size, sha256: file.sha256, target: target(entry, file), signal, statfs, allowHttp,
                onProgress: (progress) => {
                    const elapsed = Math.max(1, (Date.now() - started) / 1000);
                    onProgress({ bytes: done + progress.bytes, total: entry.totalBytes, rate: (transferred + progress.transferred) / elapsed,
                        etaSeconds: null, transferred: transferred + progress.transferred });
                },
            });
            done += file.size;
            transferred += result.bytesTransferred;
        }
        onProgress({ bytes: entry.totalBytes, total: entry.totalBytes, rate: 0, etaSeconds: 0, transferred });
        const record = {
            id: entry.id,
            version: entry.version,
            digest: entry.digest,
            files: entry.files.length,
            totalBytes: entry.totalBytes,
            installedAt: now().toISOString(),
            licence: licence ? { name: entry.licence.name, acceptedBy: licence.acceptedBy, acceptedAt: licence.acceptedAt } : null,
        };
        await writeJson(paths.record, record);
        return { transferred, record };
    }

    /**
     * Copy every cached file into container-local staging, hashing the bytes
     * as they are copied; any difference from the lock blocks the rebuild.
     * The build then reads only the staged copies, never /data.
     */
    async function stageVerified(entry, paths, { signal } = {}) {
        await fs.promises.rm(paths.stageDir, { recursive: true, force: true });
        await fs.promises.mkdir(paths.stageDir, { recursive: true, mode: 0o700 });
        const changed = [];
        for (const file of entry.files) {
            let digest = null;
            try { digest = await copyHashed(target(entry, file), path.join(paths.stageDir, file.name), signal); } catch (error) {
                if (error instanceof DownloadError) throw error;
            }
            if (digest !== file.sha256) changed.push(file.name);
        }
        if (changed.length) {
            await fs.promises.rm(paths.stageDir, { recursive: true, force: true });
            throw new LocalLlmError('cache_changed', `The cached files of ${entry.id} ${entry.version} no longer match the lock `
                + `(${changed.slice(0, 3).join(', ')}${changed.length > 3 ? ', …' : ''}); uninstall and install it again.`, { changed });
        }
        return paths.stageDir;
    }

    async function step(label, command, args, { signal, env }) {
        const result = await run({ command, args, env, signal });
        if (result.aborted) throw new DownloadError('ABORTED', `Stopped while ${label}`, { retryable: true });
        if (result.code !== 0) {
            throw new LocalLlmError('install_failed', `${label} failed (exit ${result.code ?? result.signal}): ${result.output.trim().split('\n').slice(-5).join(' | ')}`);
        }
        return result;
    }

    async function build(entry, paths, { signal, sourceDir }) {
        await fs.promises.mkdir(paths.runDir, { recursive: true });
        await fs.promises.mkdir(paths.tmpDir, { recursive: true });
        const env = {
            PATH: '/usr/local/bin:/usr/bin:/bin',
            HOME: paths.tmpDir,
            TMPDIR: paths.tmpDir,
            LANG: 'C.UTF-8',
            UV_NO_CACHE: '1',
            UV_OFFLINE: '1',
            UV_PYTHON_DOWNLOADS: 'never',
            UV_NO_PROGRESS: '1',
        };
        if (entry.kind === 'python') {
            const venv = path.join(paths.runDir, 'venv');
            await step('creating the Python environment', uv, ['venv', '--python', python, '--no-project', venv], { signal, env });
            const wheels = entry.files.filter((file) => file.name.endsWith('.whl'));
            const requirements = path.join(paths.runDir, 'requirements.txt');
            await fs.promises.writeFile(requirements, `${wheels.map(wheelRequirement).join('\n')}\n`);
            await step('installing the wheels', uv, [
                'pip', 'install', '--python', path.join(venv, 'bin', 'python'), '--offline', '--no-index',
                '--find-links', sourceDir, '--no-deps', '--require-hashes', '--link-mode', 'copy', '--no-cache',
                '-r', requirements,
            ], { signal, env });
        }
        // File times are not restored (--touch): on the container's
        // fuse-overlayfs, setting a directory's time fails with EPERM.
        for (const file of entry.files.filter((candidate) => archiveCompressionFor(entry.kind, candidate.name))) {
            const into = path.join(paths.runDir, file.extract || '.');
            await fs.promises.mkdir(into, { recursive: true });
            const flags = ['-C', into, `--strip-components=${file.strip ?? 1}`, '--no-same-owner', '--touch'];
            if (archiveCompressionFor(entry.kind, file.name) === 'zst') {
                // <stage>/<name>.tar; the staged .zst and then the .tar go as soon as they are read, since the
                // container's own filesystem holds the compressed file, the .tar and the unpacked files at once.
                const staged = path.join(sourceDir, file.name);
                const tarFile = path.join(sourceDir, file.name.slice(0, -'.zst'.length));
                await decompressZstd(staged, tarFile, { signal, label: file.name, decoder: zstdDecoder });
                await fs.promises.rm(staged, { force: true });
                await step(`unpacking ${file.name}`, 'tar', ['-xf', tarFile, ...flags], { signal, env });
                await fs.promises.rm(tarFile, { force: true });
            } else {
                await step(`unpacking ${file.name}`, 'tar', ['-xzf', path.join(sourceDir, file.name), ...flags], { signal, env });
            }
        }
        // Data files the runner reads at run time, copied as verified.
        for (const file of entry.files.filter((candidate) => candidate.into)) {
            const dir = path.join(paths.runDir, file.into);
            await fs.promises.mkdir(dir, { recursive: true });
            await fs.promises.copyFile(path.join(sourceDir, file.name), path.join(dir, file.name), fs.constants.COPYFILE_EXCL);
            await fs.promises.chmod(path.join(dir, file.name), 0o444);
        }
        await fs.promises.rm(paths.tmpDir, { recursive: true, force: true });
    }

    async function rebuild(entry, { signal }) {
        const paths = pathsFor(entry);
        const marker = await readJson(paths.marker);
        if (marker?.digest === entry.digest) return { rebuilt: false, seconds: 0, bytes: marker.bytes ?? null };
        const state = await cacheState(entry);
        if (state.state !== 'complete') {
            throw new LocalLlmError('not_installed', `${entry.id} ${entry.version} is not installed; install it first.`);
        }
        const started = Date.now();
        try {
            const sourceDir = await stageVerified(entry, paths, { signal });
            await fs.promises.rm(paths.runDir, { recursive: true, force: true });
            await build(entry, paths, { signal, sourceDir });
            // The marker says the copy can run: prove the binary is in it first.
            await assertLaunchExecutable(entry, paths.runDir);
        } catch (error) {
            // A failed build (the unpack, the size cap, a stop, a missing binary) leaves no partial copy and no scratch
            // directory. Both removals are tried, and one that fails is reported, never thrown over the error that led here.
            for (const leftover of [paths.runDir, paths.tmpDir]) {
                try {
                    await remove(leftover);
                } catch (cleanup) {
                    onWarning(`could not remove ${leftover} after the build of ${entry.id} ${entry.version} failed (${error?.code ?? 'error'}): ${cleanup?.code || cleanup?.message || cleanup}`);
                }
            }
            throw error;
        } finally {
            await fs.promises.rm(paths.stageDir, { recursive: true, force: true });
        }
        const seconds = (Date.now() - started) / 1000;
        const bytes = await treeBytes(paths.runDir);
        await writeJson(paths.marker, { digest: entry.digest, builtAt: now().toISOString(), seconds, bytes });
        return { rebuilt: true, seconds, bytes };
    }

    /**
     * Make the runnable copy exist in this container. Once built (its marker
     * names this exact lock entry) it is reused until the container is
     * recreated; otherwise the cache is re-verified and the copy rebuilt from
     * an empty directory, the marker written last. Callers that arrive while
     * a runner's copy is being built share that build.
     */
    const building = new Map();
    function ensureRunnable(id, { signal } = {}) {
        const entry = entryFor(id);
        if (!building.has(id)) {
            building.set(id, rebuild(entry, { signal }).finally(() => building.delete(id)));
        }
        return building.get(id);
    }

    /** Remove the cache and runnable copies of every other version of this runner. */
    async function pruneOtherVersions(id) {
        const entry = entryFor(id);
        const paths = pathsFor(entry);
        for (const root of [paths.idCache, paths.idRun]) {
            let versions = [];
            try { versions = await fs.promises.readdir(root); } catch {}
            for (const version of versions) {
                if (version !== entry.version) await fs.promises.rm(path.join(root, version), { recursive: true, force: true });
            }
        }
    }

    async function uninstall(id) {
        const entry = entryFor(id);
        const paths = pathsFor(entry);
        const freed = await treeBytes(paths.idCache);
        await fs.promises.rm(paths.idCache, { recursive: true, force: true });
        await fs.promises.rm(paths.idRun, { recursive: true, force: true });
        await fs.promises.rm(paths.tmpDir, { recursive: true, force: true });
        await fs.promises.rm(paths.stageDir, { recursive: true, force: true });
        return { freedBytes: freed };
    }

    /** What the overview shows about one installable runner. */
    async function describe(id) {
        const entry = entryFor(id);
        const cache = await cacheState(entry);
        const marker = await readJson(pathsFor(entry).marker);
        return {
            version: entry.version,
            totalBytes: entry.totalBytes,
            files: entry.files.length,
            licence: entry.licence,
            cache: { state: cache.state, bytes: cache.bytes, total: cache.total },
            installed: cache.state === 'complete',
            record: cache.record,
            runnable: marker?.digest === entry.digest,
        };
    }

    return Object.freeze({
        lock,
        entryFor,
        pathsFor,
        cacheState,
        fetchAll,
        ensureRunnable,
        pruneOtherVersions,
        uninstall,
        describe,
        installable: (id) => typeof id === 'string' && Object.hasOwn(lock.runners, id),
    });
}
