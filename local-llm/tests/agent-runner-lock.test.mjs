// The agent's own runner lock (DS004): the lock of the platform, merged with the
// image's; .tar.zst archives; the CI check that gates entries; and the runners
// that come from it at run time.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import zlib from 'node:zlib';

import { createController } from '../src/controller/deployments.mjs';
import { DownloadError, downloadFile } from '../src/controller/downloader.mjs';
import { createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { agentRunnerLockFile, archiveCompression, loadRunnerLock, loadRunnerLocks, validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';

const SHA = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');

function tempDir(t, name) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-${name}-`));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

const lockDocument = (runners) => ({ schema: 'local-llm.runners-lock/v1', runners });
const lockFile = (name, extra = {}) => ({
    name, url: `https://github.com/example/r/releases/download/v1/${name}`, size: 10, sha256: 'a'.repeat(64), ...extra,
});
const lockRunner = (name, extra = {}) => ({
    version: '1.0.0', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org/LICENSE' }, files: [lockFile(name)], ...extra,
});

const REPOSITORY = new URL('../../', import.meta.url);
const WORKFLOW = new URL('.github/workflows/runner-lock-check.yml', REPOSITORY);

// The keys of a top-level block of the workflow (two-space indentation), without a YAML parser.
function blockKeys(text, name, indent = 2) {
    const lines = text.split('\n');
    const start = lines.findIndex((line) => line === `${name}:`);
    if (start < 0) return null;
    const keys = [];
    for (const line of lines.slice(start + 1)) {
        if (/^\S/.test(line)) break;
        const match = new RegExp(`^ {${indent}}([A-Za-z_][\\w-]*):`).exec(line);
        if (match) keys.push(match[1]);
    }
    return keys;
}

// The repository's .github directory is not part of a mount of local-llm alone.
const repositoryRoot = fs.existsSync(new URL('AGENTS.md', REPOSITORY));

test('the runner-lock check runs for pull requests and on demand only, on both architectures, inside the published image', {
    skip: !repositoryRoot && 'the repository root is not mounted, so its .github directory is not here',
}, () => {
    const text = fs.readFileSync(WORKFLOW, 'utf8');
    // Never a push: pushing the feature branch must not start CI.
    assert.deepEqual(blockKeys(text, 'on'), ['pull_request', 'workflow_dispatch']);
    assert.doesNotMatch(text, /^\s{2}(push|schedule|workflow_run|pull_request_target):/m);
    // A pull request starts it only when a lock or the installer code changes.
    const paths = text.slice(text.indexOf('paths:'), text.indexOf('workflow_dispatch:'));
    for (const file of [
        'local-llm/catalog/runners.lock.linux-*.json',
        'local-llm/src/controller/runnerInstaller.mjs',
        'local-llm/src/controller/runnerLock.mjs',
        'local-llm/tools/runner_install_check.mjs',
        '.github/workflows/runner-lock-check.yml',
    ]) {
        assert.ok(paths.includes(`'${file}'`), file);
    }
    assert.match(text, /^permissions:\n {2}contents: read$/m);
    // One job per architecture, each on its own runner.
    assert.match(text, /- arch: amd64\n\s+runs-on: ubuntu-24\.04\n/);
    assert.match(text, /- arch: arm64\n\s+runs-on: ubuntu-24\.04-arm\n/);
    // The published image and the repository's own installer code and lock, mounted read-only.
    assert.match(text, /IMAGE: docker\.io\/assistos\/local-llm:latest/);
    assert.match(text, /-v "\$PWD\/local-llm:\/code:ro"/);
    assert.match(text, /node \/code\/tools\/runner_install_check\.mjs "\$runner"/);
    assert.match(text, /--lock "\/code\/catalog\/runners\.lock\.linux-\$\{ARCH\}\.json"/);
    // Every entry of the platform's lock is installed, and a proprietary one is never downloaded.
    assert.match(text, /Object\.keys\(require\(process\.argv\[1\]\)\.runners\)/);
    assert.match(text, /--validate-only/);
    assert.match(text, /--network=none/);
});

// ---------------------------------------------------------------- the two locks

test('the agent lock for this platform is merged with the image lock; an id in both keeps the image entry', async (t) => {
    const dir = tempDir(t, 'locks');
    const write = (name, runners) => {
        const file = path.join(dir, name);
        fs.writeFileSync(file, JSON.stringify(lockDocument(runners)));
        return file;
    };
    const imageFile = write('image.json', { both: lockRunner('both-image.tar.gz', { version: '1.0.0' }), imageonly: lockRunner('imageonly.tar.gz') });
    const agentFile = write('agent.json', { both: lockRunner('both-agent.tar.zst', { version: '9.9.9' }), agentonly: lockRunner('agentonly.tar.zst') });
    const merged = loadRunnerLocks({ image: imageFile, agent: agentFile });
    assert.deepEqual(Object.keys(merged.runners).sort(), ['agentonly', 'both', 'imageonly']);
    // The image's entry wins, whole: its version, its files and its digest.
    const image = loadRunnerLock(imageFile);
    assert.equal(merged.runners.both.version, '1.0.0');
    assert.deepEqual(merged.runners.both.files.map((file) => file.name), ['both-image.tar.gz']);
    assert.equal(merged.runners.both.digest, image.runners.both.digest);
    assert.deepEqual(merged.origin, { both: 'image', imageonly: 'image', agentonly: 'agent' });
    assert.deepEqual(merged.clashes, ['both']);
    assert.deepEqual(merged.ignored, []);
    // The merged lock is what the installer offers, with the image's entry for the id in both.
    const installer = createRunnerInstaller({ lock: merged });
    assert.equal(installer.entryFor('both').version, '1.0.0');
    assert.equal(installer.installable('agentonly'), true);
    assert.equal(installer.installable('missing'), false);
    // No agent lock (another architecture, or none shipped yet) leaves the image's lock as it was.
    for (const agent of [null, path.join(dir, 'nonexistent.json')]) {
        const alone = loadRunnerLocks({ image: imageFile, agent });
        assert.deepEqual(Object.keys(alone.runners), Object.keys(image.runners));
        assert.deepEqual([alone.clashes, alone.ignored], [[], []]);
    }
    // An agent lock that is not valid is left out, with its reason; it never takes the image's runners down with it.
    const broken = write('broken.json', { bad: lockRunner('bad.tar.gz', { files: [lockFile('bad.tar.gz', { url: 'https://evil.example.com/bad.tar.gz' })] }) });
    const tolerated = loadRunnerLocks({ image: imageFile, agent: broken });
    assert.deepEqual(Object.keys(tolerated.runners).sort(), ['both', 'imageonly']);
    assert.equal(tolerated.ignored.length, 1);
    assert.equal(tolerated.ignored[0].file, broken);
    assert.match(tolerated.ignored[0].reason, /not on an allowed host/);
    const garbled = path.join(dir, 'garbled.json');
    fs.writeFileSync(garbled, '{ not json');
    assert.equal(loadRunnerLocks({ image: imageFile, agent: garbled }).ignored.length, 1);
    // An invalid image lock still throws.
    assert.throws(() => loadRunnerLocks({ image: garbled, agent: agentFile }));
    // The agent's lock is chosen by the CPU architecture, as data.
    assert.equal(path.basename(agentRunnerLockFile('arm64')), 'runners.lock.linux-arm64.json');
    assert.equal(path.basename(agentRunnerLockFile('x64')), 'runners.lock.linux-amd64.json');
    assert.equal(path.dirname(agentRunnerLockFile('x64')), path.resolve(import.meta.dirname, '..', 'catalog'));
    for (const arch of ['riscv64', 'ia32', 'constructor', '__proto__', 'toString']) assert.equal(agentRunnerLockFile(arch), null, arch);
    // The controller builds its installer from the merged lock and says where the two met.
    const data = tempDir(t, 'locks-data');
    const controller = createController({
        dataDir: data, env: { PATH: '/usr/bin' }, seedCatalog: [], stateStore: createStateStore({ dataDir: data }), runnerLocks: merged,
        snapshot: async () => ({ gpu: { available: false, state: 'absent', reason: 'none' }, memory: {}, disk: {}, cpus: 1, cores: 1 }),
        detectRunner: () => ({ installed: false, version: null, reason: null }),
    });
    t.after(() => controller.drain());
    const lines = (await controller.status()).logs.map((entry) => entry.line);
    assert.ok(lines.some((line) => line.includes("runner lock: both is in the image's lock and in the agent's lock; the image's entry is used")), lines.join('\n'));
    const warned = createController({
        dataDir: data, env: { PATH: '/usr/bin' }, seedCatalog: [], stateStore: createStateStore({ dataDir: data }), runnerLocks: tolerated,
        snapshot: async () => ({ gpu: { available: false, state: 'absent', reason: 'none' }, memory: {}, disk: {}, cpus: 1, cores: 1 }),
        detectRunner: () => ({ installed: false, version: null, reason: null }),
    });
    t.after(() => warned.drain());
    const ignoredLines = (await warned.status()).logs.map((entry) => entry.line);
    assert.ok(ignoredLines.some((line) => line.includes("runner lock: the agent's lock") && line.includes('is ignored')), ignoredLines.join('\n'));
});

// ---------------------------------------------------------------- .tar.zst

test('a .tar.zst archive is accepted with strip 0 and validated like a .tar.gz', () => {
    const zst = lockFile('ollama-linux-arm64.tar.zst', { strip: 0 });
    const entry = validateRunnerLock(lockDocument({ ollama: lockRunner('x', { files: [zst] }) })).runners.ollama;
    assert.equal(entry.files[0].name, 'ollama-linux-arm64.tar.zst');
    assert.equal(entry.files[0].strip, 0);
    assert.equal(entry.totalBytes, 10);
    assert.equal(archiveCompression('a.tar.zst'), 'zst');
    assert.deepEqual(['a.tar.gz', 'a.tgz'].map(archiveCompression), ['gz', 'gz']);
    for (const name of ['a.tar', 'a.zst', 'a.tar.zstd', 'a.tar.zst.gz', 'a.zip', 'atar.zst']) assert.equal(archiveCompression(name), null, name);
    const one = (file, kind = 'archive') => () => validateRunnerLock(lockDocument({ r: lockRunner('x', { kind, files: [file] }) }));
    // strip is 0 or 1, exactly as for a .tar.gz, and the file needs the same pins.
    assert.equal(one(lockFile('a.tar.zst', { strip: 1 }))().runners.r.files[0].strip, 1);
    assert.equal(one(lockFile('a.tar.zst'))().runners.r.files[0].strip, undefined);
    for (const strip of [2, -1, '0', 0.5]) assert.throws(one(lockFile('a.tar.zst', { strip })), /strip must be 0 or 1/);
    assert.throws(one(lockFile('a.tar.zst', { size: 0 })), /size/);
    assert.throws(one(lockFile('a.tar.zst', { sha256: 'abc' })), /sha256/);
    assert.throws(one(lockFile('a.tar.zst', { url: 'http://github.com/o/r/a.tar.zst' })), /plain https URL/);
    assert.throws(one(lockFile('a.tar.zst', { url: 'https://evil.example.com/a.tar.zst' })), /not on an allowed host/);
    assert.throws(one(lockFile('a.tar.zst', { url: 'https://github.com:8443/a.tar.zst' })), /plain https URL/);
    assert.throws(one(lockFile('../a.tar.zst')), /plain file name/);
    assert.throws(one(lockFile('a.tar.zst', { script: 'curl | sh' })), /unsupported field 'script'/);
    assert.throws(one(lockFile('a.tar.zst', { into: 'data' })), /into is only for data files/);
    // Not an archive this lock unpacks: the message names both kinds; .tar.gz is still accepted.
    assert.throws(one(lockFile('a.zip')), /\.tar\.gz archive or a \.tar\.zst archive/);
    assert.equal(one(lockFile('a.tar.gz', { strip: 0 }))().runners.r.files[0].strip, 0);
    // A python runner's source archives stay .tar.gz: .tar.zst is for archive runners.
    assert.throws(one(lockFile('a.tar.zst', { extract: 'src' }), 'python'), /wheel, a \.tar\.gz archive with extract, or a data file with into/);
    assert.throws(one(lockFile('a.tar.zst', { strip: 0 }), 'python'), /strip is only for archives/);
});

// A tar of a small runner: bin/runner and VERSION, entries at its root (strip 0).
function makeTar(t) {
    const src = tempDir(t, 'zst-src');
    fs.mkdirSync(path.join(src, 'bin'));
    fs.writeFileSync(path.join(src, 'bin', 'runner'), '#!/bin/sh\necho runner\n', { mode: 0o755 });
    fs.writeFileSync(path.join(src, 'VERSION'), '1.0.0\n');
    fs.writeFileSync(path.join(src, 'padding.bin'), crypto.randomBytes(256 * 1024));
    const archive = path.join(tempDir(t, 'zst-out'), 'runner.tar');
    execFileSync('tar', ['-cf', archive, '-C', src, 'bin', 'VERSION', 'padding.bin']);
    return fs.readFileSync(archive);
}

const ZST_NAME = 'runner-1.0.0.tar.zst';

function zstLock(bytes, { sha256 = SHA(bytes), strip = 0 } = {}) {
    return validateRunnerLock(lockDocument({
        testrunner: {
            version: '1.0.0', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org/LICENSE' },
            files: [{ name: ZST_NAME, url: `https://github.com/example/runner/releases/download/1.0.0/${ZST_NAME}`, size: bytes.length, sha256, strip }],
        },
    }));
}

// An installer over a cache that already holds the pinned file: nothing is downloaded, and
// `run` (the tar step) is the test's own. `bytes` are what the cache holds; the lock pins `pinned`.
function cachedInstaller(t, bytes, { pinned = bytes, run } = {}) {
    const root = tempDir(t, 'zst-installer');
    const lock = zstLock(pinned);
    const cacheRoot = path.join(root, 'data', 'runners');
    const runRoot = path.join(root, 'opt', 'runners');
    const installer = createRunnerInstaller({
        lock, cacheRoot, runRoot, run,
        inspect: async ({ target }) => ({ state: 'complete', bytes: fs.statSync(target).size }),
    });
    const entry = installer.entryFor('testrunner');
    const files = path.join(cacheRoot, 'testrunner', '1.0.0', 'files');
    fs.mkdirSync(files, { recursive: true });
    fs.writeFileSync(path.join(files, ZST_NAME), bytes);
    fs.writeFileSync(path.join(cacheRoot, 'testrunner', '1.0.0', 'installed.json'), JSON.stringify({ id: 'testrunner', version: '1.0.0', digest: entry.digest }));
    return { installer, runRoot, stageDir: path.join(runRoot, '.stage-testrunner'), runDir: path.join(runRoot, 'testrunner', '1.0.0') };
}

test('a .tar.zst is decompressed by Node into a staged .tar that tar then reads, and the staging is left empty', async (t) => {
    const tar = makeTar(t);
    const calls = [];
    let staged = null;
    const h = cachedInstaller(t, zlib.zstdCompressSync(tar), {
        run: async (options) => {
            // While tar runs, the staging holds the .tar beside nothing else: the staged .zst went once it was read.
            calls.push({ command: options.command, args: options.args });
            staged = { names: fs.readdirSync(h.stageDir), tar: fs.readFileSync(options.args[1]) };
            return { code: 0, signal: null, output: '', aborted: false };
        },
    });
    const built = await h.installer.ensureRunnable('testrunner');
    assert.equal(built.rebuilt, true);
    assert.deepEqual(calls, [{
        command: 'tar',
        args: ['-xf', path.join(h.stageDir, 'runner-1.0.0.tar'), '-C', h.runDir, '--strip-components=0', '--no-same-owner', '--touch'],
    }]);
    assert.deepEqual(staged.names, ['runner-1.0.0.tar']);
    assert.ok(staged.tar.equals(tar), 'tar read exactly the decompressed bytes');
    // Afterwards: no staging, and the marker is written last.
    assert.equal(fs.existsSync(h.stageDir), false);
    assert.equal(JSON.parse(fs.readFileSync(path.join(h.runDir, '.ready.json'), 'utf8')).digest, h.installer.entryFor('testrunner').digest);
    // Reused while the copy exists: no second decompression.
    assert.equal((await h.installer.ensureRunnable('testrunner')).rebuilt, false);
    assert.equal(calls.length, 1);
});

test('a malformed zstd stream is a coded error, leaves no staging and never reaches tar', async (t) => {
    const calls = [];
    const garbage = Buffer.from('this is not a zstd stream, only text that happens to be pinned');
    const h = cachedInstaller(t, garbage, { run: async (options) => { calls.push(options.args); return { code: 0, output: '' }; } });
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'),
        (error) => error.code === 'install_failed' && /decompressing runner-1\.0\.0\.tar\.zst failed: ZSTD_error_prefix_unknown/.test(error.message));
    assert.deepEqual(calls, [], 'tar never ran');
    assert.equal(fs.existsSync(h.stageDir), false, 'the staging is removed');
    assert.equal(fs.existsSync(path.join(h.runDir, '.ready.json')), false, 'no marker: the next Run rebuilds');
    // The next attempt starts clean and fails the same way, not with leftovers.
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'), { code: 'install_failed' });
});

test('a cached .tar.zst that no longer matches its pinned sha256 blocks the rebuild before anything is unpacked', async (t) => {
    const good = zlib.zstdCompressSync(makeTar(t));
    const changed = Buffer.from(good);
    changed[changed.length - 10] ^= 0xff;
    const calls = [];
    const h = cachedInstaller(t, changed, { pinned: good, run: async (options) => { calls.push(options.args); return { code: 0, output: '' }; } });
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'), { code: 'cache_changed' });
    assert.deepEqual(calls, []);
    assert.equal(fs.existsSync(h.stageDir), false);
    assert.equal(fs.existsSync(h.runDir), false, 'nothing was unpacked');
});

// The two tests below run the real downloader and GNU tar, as the agent does in its container.
const LINUX_ONLY = process.platform !== 'linux' && 'needs Linux: the downloader confirms files through /proc/self/fd, and the unpack uses GNU tar (--touch)';

async function serve(t, files) {
    const server = http.createServer((req, res) => {
        const body = files[decodeURIComponent(req.url.split('/').pop())];
        if (!body) { res.writeHead(404); res.end(); return; }
        res.writeHead(200, { 'Content-Length': body.length });
        res.end(body);
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => server.close());
    return `http://127.0.0.1:${server.address().port}`;
}

function onlineInstaller(t, lock, base) {
    const root = tempDir(t, 'zst-online');
    return {
        installer: createRunnerInstaller({
            lock, cacheRoot: path.join(root, 'data', 'runners'), runRoot: path.join(root, 'opt', 'runners'),
            // The lock's https URL is answered by the local server.
            download: (options) => downloadFile({ ...options, fetchImpl: (url, init) => fetch(`${base}/${path.basename(new URL(url).pathname)}`, init) }),
            statfs: async () => ({ bavail: 1e12, bsize: 1 }),
        }),
        runDir: path.join(root, 'opt', 'runners', 'testrunner', '1.0.0'),
        cacheFiles: path.join(root, 'data', 'runners', 'testrunner', '1.0.0', 'files'),
        stageDir: path.join(root, 'opt', 'runners', '.stage-testrunner'),
    };
}

test('a .tar.zst is downloaded, verified, decompressed and unpacked with strip 0 (Linux)', { skip: LINUX_ONLY }, async (t) => {
    const tar = makeTar(t);
    const bytes = zlib.zstdCompressSync(tar);
    const base = await serve(t, { [ZST_NAME]: bytes });
    const h = onlineInstaller(t, zstLock(bytes), base);
    await h.installer.fetchAll(h.installer.entryFor('testrunner'));
    await h.installer.ensureRunnable('testrunner');
    assert.equal(fs.readFileSync(path.join(h.runDir, 'VERSION'), 'utf8'), '1.0.0\n');
    assert.equal(fs.readFileSync(path.join(h.runDir, 'bin', 'runner'), 'utf8'), '#!/bin/sh\necho runner\n');
    assert.equal(fs.existsSync(h.stageDir), false);
});

test('a .tar.zst whose bytes do not match the pinned sha256 is refused with SHA256_MISMATCH and nothing is unpacked (Linux)', { skip: LINUX_ONLY }, async (t) => {
    const bytes = zlib.zstdCompressSync(makeTar(t));
    const base = await serve(t, { [ZST_NAME]: bytes });
    const h = onlineInstaller(t, zstLock(bytes, { sha256: 'b'.repeat(64) }), base);
    await assert.rejects(() => h.installer.fetchAll(h.installer.entryFor('testrunner')),
        (error) => error instanceof DownloadError && error.code === 'SHA256_MISMATCH');
    assert.deepEqual(fs.existsSync(h.cacheFiles) ? fs.readdirSync(h.cacheFiles) : [], [], 'nothing is kept');
    assert.equal((await h.installer.describe('testrunner')).installed, false);
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'), { code: 'not_installed' });
    assert.equal(fs.existsSync(h.runDir), false, 'nothing was unpacked');
    // The same file with a wrong size is refused as well.
    const sized = validateRunnerLock(lockDocument({ testrunner: { ...lockRunner(ZST_NAME), files: [{ ...lockFile(ZST_NAME), size: bytes.length + 1, sha256: SHA(bytes), strip: 0 }] } }));
    const again = onlineInstaller(t, sized, base);
    await assert.rejects(() => again.installer.fetchAll(again.installer.entryFor('testrunner')), (error) => error instanceof DownloadError);
    assert.equal(fs.existsSync(again.runDir), false);
});
