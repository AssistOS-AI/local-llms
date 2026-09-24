// On-demand runner installs: the lock is the only source of URLs, every file
// is verified, downloads resume, the runnable copy is rebuilt once per
// container from a re-verified cache, and uninstall frees everything.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { DownloadError, downloadFile } from '../src/controller/downloader.mjs';
import { ALLOWED_HOSTS, loadRunnerLock, validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { createRunnerInstaller, runTool, wheelRequirement } from '../src/controller/runnerInstaller.mjs';

const SHA = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');

function tempDir(t, name) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-${name}-`));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

// A .tar.gz whose top directory holds bin/runner and VERSION.
function makeArchive(t, version) {
    const dir = tempDir(t, 'archive-src');
    const top = path.join(dir, `runner-${version}`);
    fs.mkdirSync(path.join(top, 'bin'), { recursive: true });
    fs.writeFileSync(path.join(top, 'bin', 'runner'), '#!/bin/sh\necho runner\n', { mode: 0o755 });
    fs.writeFileSync(path.join(top, 'VERSION'), `${version}\n`);
    fs.writeFileSync(path.join(top, 'padding.bin'), crypto.randomBytes(1024 * 1024));
    const archive = path.join(dir, `runner-${version}.tar.gz`);
    execFileSync('tar', ['-czf', archive, '-C', dir, `runner-${version}`]);
    return fs.readFileSync(archive);
}

// Serves files by name; `cutFirst` drops the first response after that many bytes.
async function serve(t, files, { cutFirst = null } = {}) {
    const requests = [];
    const server = http.createServer((req, res) => {
        const name = decodeURIComponent(req.url.split('/').pop());
        requests.push({ name, range: req.headers.range ?? null });
        const body = files[name];
        if (!body) { res.writeHead(404); res.end(); return; }
        const range = /^bytes=(\d+)-$/.exec(req.headers.range ?? '');
        const start = range ? Number(range[1]) : 0;
        const headers = { 'Content-Length': body.length - start };
        if (range) headers['Content-Range'] = `bytes ${start}-${body.length - 1}/${body.length}`;
        res.writeHead(range ? 206 : 200, headers);
        if (cutFirst !== null && requests.length === 1) {
            res.write(body.subarray(start, start + cutFirst), () => res.destroy());
            return;
        }
        res.end(body.subarray(start));
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => server.close());
    return { base: `http://127.0.0.1:${server.address().port}`, requests };
}

function archiveLock(version, bytes, { sha256 = SHA(bytes), requiresAcceptance = false } = {}) {
    return validateRunnerLock({
        schema: 'local-llm.runners-lock/v1',
        runners: {
            testrunner: {
                version,
                kind: 'archive',
                licence: { name: 'MIT', url: 'https://github.com/example/runner/blob/main/LICENSE', requiresAcceptance },
                files: [{ name: `runner-${version}.tar.gz`, url: `https://github.com/example/runner/releases/download/${version}/runner-${version}.tar.gz`, size: bytes.length, sha256 }],
            },
        },
    });
}

// Every lock URL is https on an allowed host; tests answer its requests from the local server.
function redirectTo(base, calls = []) {
    return (options) => {
        calls.push(options.url);
        const fetchImpl = (url, init) => fetch(`${base}/${path.basename(new URL(url).pathname)}`, init);
        return downloadFile({ ...options, fetchImpl });
    };
}

function installer(t, lock, base, extra = {}) {
    const root = tempDir(t, 'installer');
    return {
        root,
        cacheRoot: path.join(root, 'data', 'runners'),
        runRoot: path.join(root, 'opt', 'runners'),
        installer: createRunnerInstaller({
            lock, cacheRoot: path.join(root, 'data', 'runners'), runRoot: path.join(root, 'opt', 'runners'),
            download: redirectTo(base, extra.calls), statfs: async () => ({ bavail: 1e12, bsize: 1 }), ...extra,
        }),
    };
}

test('the lock accepts only pinned https files on allowed hosts', () => {
    const good = { name: 'r.tar.gz', url: 'https://github.com/o/r/releases/download/v1/r.tar.gz', size: 1, sha256: 'a'.repeat(64) };
    const lock = (file, kind = 'archive') => ({
        schema: 'local-llm.runners-lock/v1',
        runners: { r: { version: '1', kind, licence: { name: 'MIT', url: 'https://example.org' }, files: [file] } },
    });
    assert.equal(validateRunnerLock(lock(good)).runners.r.totalBytes, 1);
    for (const [file, message] of [
        [{ ...good, url: 'http://github.com/o/r.tar.gz' }, /plain https URL/],
        [{ ...good, url: 'https://evil.example.com/r.tar.gz' }, /not on an allowed host/],
        [{ ...good, url: 'https://user:pw@github.com/r.tar.gz' }, /plain https URL/],
        [{ ...good, url: 'https://github.com:8443/r.tar.gz' }, /plain https URL/],
        [{ ...good, sha256: 'abc' }, /sha256/],
        [{ ...good, name: '../r.tar.gz' }, /plain file name/],
        [{ ...good, size: 0 }, /size/],
        [{ ...good, script: 'curl | sh' }, /unsupported field 'script'/],
    ]) {
        assert.throws(() => validateRunnerLock(lock(file)), message);
    }
    assert.throws(() => validateRunnerLock(lock({ ...good, name: 'r.zip' })), /\.tar\.gz archive/);
    assert.throws(() => validateRunnerLock(lock({ ...good, name: 'r.tar.gz' }, 'python')), /wheel/);
    assert.throws(() => validateRunnerLock({ schema: 'x', runners: {} }), /schema/);
    assert.ok(ALLOWED_HOSTS.includes('files.pythonhosted.org'));
    // An image without a lock offers nothing to install.
    assert.deepEqual(loadRunnerLock('/nonexistent/runners.lock.json').runners, {});
});

test('install downloads only the lock URL, resumes a cut transfer, and records the install', async (t) => {
    const bytes = makeArchive(t, '1.0.0');
    const { base, requests } = await serve(t, { 'runner-1.0.0.tar.gz': bytes }, { cutFirst: 64 * 1024 });
    const calls = [];
    const { installer: inst, cacheRoot } = installer(t, archiveLock('1.0.0', bytes), base, { calls });
    const entry = inst.entryFor('testrunner');
    const { record } = await inst.fetchAll(entry, { licence: null });
    assert.deepEqual(calls, ['https://github.com/example/runner/releases/download/1.0.0/runner-1.0.0.tar.gz']);
    assert.equal(requests[0].range, null);
    assert.match(requests.at(-1).range, /^bytes=\d+-$/, 'the second request resumed');
    assert.equal(record.digest, entry.digest);
    assert.equal(SHA(fs.readFileSync(path.join(cacheRoot, 'testrunner', '1.0.0', 'files', 'runner-1.0.0.tar.gz'))), SHA(bytes));
    assert.equal((await inst.describe('testrunner')).installed, true);
    assert.throws(() => inst.entryFor('https://evil.example.com/x'), { code: 'not_installable' });
});

test('bytes that differ from the lock are refused and nothing is kept', async (t) => {
    const bytes = makeArchive(t, '1.0.0');
    const { base } = await serve(t, { 'runner-1.0.0.tar.gz': bytes });
    const { installer: inst, cacheRoot } = installer(t, archiveLock('1.0.0', bytes, { sha256: 'b'.repeat(64) }), base);
    await assert.rejects(() => inst.fetchAll(inst.entryFor('testrunner')),
        (error) => error instanceof DownloadError && error.code === 'SHA256_MISMATCH');
    const files = path.join(cacheRoot, 'testrunner', '1.0.0', 'files');
    assert.deepEqual(fs.existsSync(files) ? fs.readdirSync(files) : [], []);
    assert.equal((await inst.describe('testrunner')).installed, false);
});

test('the runnable copy is built once per container and rebuilt after the container is recreated', async (t) => {
    const bytes = makeArchive(t, '1.0.0');
    const { base } = await serve(t, { 'runner-1.0.0.tar.gz': bytes });
    const { installer: inst, runRoot } = installer(t, archiveLock('1.0.0', bytes), base);
    const entry = inst.entryFor('testrunner');
    await assert.rejects(() => inst.ensureRunnable('testrunner'), { code: 'not_installed' });
    await inst.fetchAll(entry);
    const first = await inst.ensureRunnable('testrunner');
    assert.equal(first.rebuilt, true);
    const runDir = path.join(runRoot, 'testrunner', '1.0.0');
    assert.equal(fs.readFileSync(path.join(runDir, 'VERSION'), 'utf8'), '1.0.0\n');
    // A process restart in the same container reuses the copy, tampered or not.
    fs.writeFileSync(path.join(runDir, 'bin', 'runner'), '#!/bin/sh\necho tampered\n');
    assert.equal((await inst.ensureRunnable('testrunner')).rebuilt, false);
    // A new container starts without the copy: it is rebuilt from the cache and the change is gone.
    fs.rmSync(runRoot, { recursive: true, force: true });
    assert.equal((await inst.ensureRunnable('testrunner')).rebuilt, true);
    assert.equal(fs.readFileSync(path.join(runDir, 'bin', 'runner'), 'utf8'), '#!/bin/sh\necho runner\n');
});

test('a cached file changed on disk blocks the rebuild', async (t) => {
    const bytes = makeArchive(t, '1.0.0');
    const { base } = await serve(t, { 'runner-1.0.0.tar.gz': bytes });
    const { installer: inst, cacheRoot, runRoot } = installer(t, archiveLock('1.0.0', bytes), base);
    await inst.fetchAll(inst.entryFor('testrunner'));
    const cached = path.join(cacheRoot, 'testrunner', '1.0.0', 'files', 'runner-1.0.0.tar.gz');
    const changed = Buffer.from(fs.readFileSync(cached));
    changed[changed.length - 10] ^= 0xff;
    fs.writeFileSync(cached, changed);
    await assert.rejects(() => inst.ensureRunnable('testrunner'), { code: 'cache_changed' });
    assert.equal(fs.existsSync(path.join(runRoot, 'testrunner', '1.0.0')), false);
});

test('an update keeps the old version until the new one is in place, and uninstall frees everything', async (t) => {
    const v1 = makeArchive(t, '1.0.0');
    const v2 = makeArchive(t, '2.0.0');
    const { base } = await serve(t, { 'runner-1.0.0.tar.gz': v1, 'runner-2.0.0.tar.gz': v2 });
    const shared = tempDir(t, 'shared');
    const make = (lock) => createRunnerInstaller({
        lock, cacheRoot: path.join(shared, 'data'), runRoot: path.join(shared, 'opt'),
        download: redirectTo(base), statfs: async () => ({ bavail: 1e12, bsize: 1 }),
    });
    const old = make(archiveLock('1.0.0', v1));
    await old.fetchAll(old.entryFor('testrunner'));
    await old.ensureRunnable('testrunner');
    // A failed new version leaves the old one alone.
    const broken = make(archiveLock('2.0.0', v2, { sha256: 'c'.repeat(64) }));
    await assert.rejects(() => broken.fetchAll(broken.entryFor('testrunner')));
    assert.ok(fs.existsSync(path.join(shared, 'data', 'testrunner', '1.0.0', 'installed.json')));
    const next = make(archiveLock('2.0.0', v2));
    await next.fetchAll(next.entryFor('testrunner'));
    await next.ensureRunnable('testrunner');
    await next.pruneOtherVersions('testrunner');
    assert.deepEqual(fs.readdirSync(path.join(shared, 'data', 'testrunner')), ['2.0.0']);
    assert.deepEqual(fs.readdirSync(path.join(shared, 'opt', 'testrunner')), ['2.0.0']);
    const { freedBytes } = await next.uninstall('testrunner');
    assert.ok(freedBytes >= v2.length);
    assert.equal(fs.existsSync(path.join(shared, 'data', 'testrunner')), false);
    assert.equal(fs.existsSync(path.join(shared, 'opt', 'testrunner')), false);
    assert.equal((await next.describe('testrunner')).installed, false);
});

test('a Python runner is installed by uv from the cached wheels only: offline, pinned by hash, no dependency resolution', async (t) => {
    const wheel = crypto.randomBytes(4096);
    const name = 'tiny_runner-1.2.3-py3-none-any.whl';
    const { base } = await serve(t, { [name]: wheel });
    const lock = validateRunnerLock({
        schema: 'local-llm.runners-lock/v1',
        runners: { tiny: { version: '1.2.3', kind: 'python', licence: { name: 'MIT', url: 'https://example.org' },
            files: [{ name, url: `https://files.pythonhosted.org/packages/aa/bb/${name}`, size: wheel.length, sha256: SHA(wheel) }] } },
    });
    const calls = [];
    const fakeRun = async ({ command, args, env }) => {
        calls.push({ command, args, env });
        if (args[0] === 'venv') fs.mkdirSync(path.join(args.at(-1), 'bin'), { recursive: true });
        return { code: 0, signal: null, output: '', aborted: false };
    };
    const { installer: inst, runRoot } = installer(t, lock, base, { run: fakeRun, uv: '/usr/local/bin/uv', python: '/usr/bin/python3' });
    await inst.fetchAll(inst.entryFor('tiny'));
    await inst.ensureRunnable('tiny');
    const runDir = path.join(runRoot, 'tiny', '1.2.3');
    assert.deepEqual(calls[0].args, ['venv', '--python', '/usr/bin/python3', '--no-project', path.join(runDir, 'venv')]);
    const install = calls[1].args;
    for (const flag of ['--offline', '--no-index', '--no-deps', '--require-hashes', '--no-cache']) assert.ok(install.includes(flag), flag);
    // uv reads the verified staging copies in the container, not the cache in /data.
    assert.equal(install[install.indexOf('--find-links') + 1], path.join(runRoot, '.stage-tiny'));
    assert.equal(fs.readFileSync(path.join(runDir, 'requirements.txt'), 'utf8'), `tiny_runner==1.2.3 --hash=sha256:${SHA(wheel)}\n`);
    assert.equal(calls[1].env.UV_OFFLINE, '1');
    assert.equal(calls[1].env.UV_PYTHON_DOWNLOADS, 'never');
    assert.equal(Object.keys(calls[1].env).some((key) => /token|secret|proxy/i.test(key)), false);
    assert.equal(wheelRequirement({ name: 'torch-2.13.0-cp313-cp313-manylinux_2_28_x86_64.whl', sha256: 'f'.repeat(64) }),
        `torch==2.13.0 --hash=sha256:${'f'.repeat(64)}`);
});

test('an install step runs in its own process group and an abort stops all of it', async () => {
    const controller = new AbortController();
    const marker = path.join(os.tmpdir(), `local-llm-step-${process.pid}-${Date.now()}`);
    const pending = runTool({ command: '/bin/sh', args: ['-c', `sleep 60 & echo $! > ${marker}; wait`], env: { PATH: process.env.PATH }, signal: controller.signal });
    let helper = null;
    for (let index = 0; index < 100 && !helper; index += 1) {
        await new Promise((resolve) => setTimeout(resolve, 20));
        try { helper = Number(fs.readFileSync(marker, 'utf8')); } catch {}
    }
    controller.abort();
    const result = await pending;
    fs.rmSync(marker, { force: true });
    assert.equal(result.aborted, true);
    let alive = true;
    for (let index = 0; index < 100 && alive; index += 1) {
        try {
            const stat = fs.readFileSync(`/proc/${helper}/stat`, 'utf8');
            alive = stat.slice(stat.lastIndexOf(')') + 2, stat.lastIndexOf(')') + 3) !== 'Z';
        } catch { alive = false; }
        if (alive) await new Promise((resolve) => setTimeout(resolve, 20));
    }
    assert.equal(alive, false);
});

test('a cached file swapped after it was verified never reaches the runnable copy', async (t) => {
    const bytes = makeArchive(t, '1.0.0');
    const evil = makeArchive(t, '6.6.6');
    const { base } = await serve(t, { 'runner-1.0.0.tar.gz': bytes });
    const steps = [];
    let cached = null;
    // Another writer of /data replaces the cached file before every build step.
    const swapThenRun = async (options) => {
        fs.writeFileSync(cached, evil);
        steps.push(options.args.map(String));
        return runTool(options);
    };
    const { installer: inst, cacheRoot, runRoot } = installer(t, archiveLock('1.0.0', bytes), base, { run: swapThenRun });
    cached = path.join(cacheRoot, 'testrunner', '1.0.0', 'files', 'runner-1.0.0.tar.gz');
    await inst.fetchAll(inst.entryFor('testrunner'));
    assert.equal((await inst.ensureRunnable('testrunner')).rebuilt, true);
    const runDir = path.join(runRoot, 'testrunner', '1.0.0');
    assert.equal(fs.readFileSync(path.join(runDir, 'VERSION'), 'utf8'), '1.0.0\n');
    // The build read a container-local copy, hashed as it was made, never a cache path.
    assert.ok(steps.length > 0);
    assert.ok(steps.every((args) => args.every((arg) => !arg.startsWith(cacheRoot))), JSON.stringify(steps));
    // The staging copy is gone once the runnable copy is in place.
    assert.deepEqual(fs.readdirSync(runRoot).filter((name) => name.startsWith('.')), []);
    // The swapped cache itself is refused on the next rebuild: as changed, or,
    // when its size differs, as an incomplete install.
    fs.rmSync(runDir, { recursive: true, force: true });
    await assert.rejects(() => inst.ensureRunnable('testrunner'), (error) => ['cache_changed', 'not_installed'].includes(error.code));
});

test('two rebuilds of one runner at the same time share one build', async (t) => {
    const bytes = makeArchive(t, '1.0.0');
    const { base } = await serve(t, { 'runner-1.0.0.tar.gz': bytes });
    let steps = 0;
    const slow = async (options) => {
        steps += 1;
        await new Promise((resolve) => setTimeout(resolve, 50));
        return runTool(options);
    };
    const { installer: inst } = installer(t, archiveLock('1.0.0', bytes), base, { run: slow });
    await inst.fetchAll(inst.entryFor('testrunner'));
    const [first, second] = await Promise.all([inst.ensureRunnable('testrunner'), inst.ensureRunnable('testrunner')]);
    assert.equal(steps, 1);
    assert.deepEqual(second, first);
    assert.equal((await inst.ensureRunnable('testrunner')).rebuilt, false);
});
