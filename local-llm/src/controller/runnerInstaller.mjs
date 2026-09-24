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
import { pipeline } from 'node:stream/promises';

import { LocalLlmError } from '../errors.mjs';
import { DownloadError, downloadFile, inspectFile } from './downloader.mjs';

export const DEFAULT_CACHE_ROOT = '/data/runners';
export const DEFAULT_RUN_ROOT = '/opt/runners';
export const DEFAULT_UV = '/usr/local/bin/uv';
export const DEFAULT_PYTHON = '/usr/bin/python3';
const SPACE_MARGIN = 1.05;
const TOOL_OUTPUT_KEPT = 64 * 1024;

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
        for (const file of entry.files.filter((candidate) => /\.(tar\.gz|tgz)$/.test(candidate.name))) {
            const into = path.join(paths.runDir, file.extract || '.');
            await fs.promises.mkdir(into, { recursive: true });
            await step(`unpacking ${file.name}`, 'tar', ['-xzf', path.join(sourceDir, file.name), '-C', into, '--strip-components=1', '--no-same-owner'], { signal, env });
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
