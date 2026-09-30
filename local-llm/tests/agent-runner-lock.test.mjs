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

import { installMessage, runnersPanelHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';
import { validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { DownloadError, downloadFile } from '../src/controller/downloader.mjs';
import { fetchOllamaRegistryManifest, manifestPath, readOllamaManifest } from '../src/controller/ollamaStore.mjs';
import { createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { agentRunnerLockFile, archiveCompression, loadRunnerLock, loadRunnerLocks, validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { RUNNERS } from '../src/runners/index.mjs';
import { manifestBytes, registry, sha256 } from './ollama-fixture.mjs';

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
    // The agent's lock must say which file each archive runner launches (check.executable); the image's need not.
    const launches = { check: { executable: 'bin/launch' } };
    const agentFile = write('agent.json', { both: lockRunner('both-agent.tar.zst', { version: '9.9.9', ...launches }), agentonly: lockRunner('agentonly.tar.zst', launches) });
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
    const data = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-locks-data-'));
    // Both controllers write under the one directory, and it goes after both have drained.
    const controllers = [];
    t.after(async () => {
        await Promise.all(controllers.map((each) => each.drain()));
        fs.rmSync(data, { recursive: true, force: true });
    });
    const controller = createController({
        dataDir: data, env: { PATH: '/usr/bin' }, seedCatalog: [], stateStore: createStateStore({ dataDir: data }), runnerLocks: merged,
        snapshot: async () => ({ gpu: { available: false, state: 'absent', reason: 'none' }, memory: {}, disk: {}, cpus: 1, cores: 1 }),
        detectRunner: () => ({ installed: false, version: null, reason: null }),
    });
    controllers.push(controller);
    const lines = (await controller.status()).logs.map((entry) => entry.line);
    assert.ok(lines.some((line) => line.includes("runner lock: both is in the image's lock and in the agent's lock; the image's entry is used")), lines.join('\n'));
    const warned = createController({
        dataDir: data, env: { PATH: '/usr/bin' }, seedCatalog: [], stateStore: createStateStore({ dataDir: data }), runnerLocks: tolerated,
        snapshot: async () => ({ gpu: { available: false, state: 'absent', reason: 'none' }, memory: {}, disk: {}, cpus: 1, cores: 1 }),
        detectRunner: () => ({ installed: false, version: null, reason: null }),
    });
    controllers.push(warned);
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
    for (const extra of [{ extract: 'src' }, { strip: 0 }, { into: 'data' }]) {
        assert.throws(one(lockFile('a.tar.zst', extra), 'python'), /is a \.tar\.zst archive, which only an archive runner may list/, JSON.stringify(extra));
    }
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

// ---------------------------------------------------------------- Ollama on the cpu profile

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const M1_TOTAL = 6036128 * 1024;
const M1_AVAILABLE = 3122576 * 1024;
const ABSENT_GPU = { available: false, state: 'absent', reason: 'No GPU is attached to this agent: GPU not applied to this Box yet' };
const GB10 = {
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified', totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
};
const ARM64_CONTRACT = Object.freeze({ architecture: 'arm64', llama_cpp: 'b11159', gpu_compute_capabilities: '12.1' });
const IN_IMAGE = '/opt/llama.cpp/llama-server';
const TAG = 'qwen2.5:0.5b';

const snapshotOf = (gpu) => ({
    gpu: structuredClone(gpu), memory: { totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE },
    disk: { freeBytes: 100_000 * MIB, totalBytes: 400_000 * MIB }, cpus: 4, cores: 4,
});

async function until(predicate, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

// A complete Ollama store entry for the tag: its manifest and one blob, as a finished pull leaves them.
function writeOllamaModel(dataDir, tag, bytes = 16) {
    const modelsDir = path.join(dataDir, 'models', 'ollama');
    const hex = 'b'.repeat(64);
    fs.mkdirSync(path.join(modelsDir, 'blobs'), { recursive: true });
    fs.writeFileSync(path.join(modelsDir, 'blobs', `sha256-${hex}`), Buffer.alloc(bytes));
    const target = manifestPath(modelsDir, tag);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, JSON.stringify({
        schemaVersion: 2, layers: [{ digest: `sha256:${hex}`, size: bytes, mediaType: 'application/vnd.ollama.image.model' }],
    }));
    return readOllamaManifest(modelsDir, tag);
}

function ollamaModel(source, extra = {}) {
    return validateModel({ id: 'olla-cpu', displayName: 'Qwen 0.5B', profiles: ['cpu', 'dedicated', 'unified'], contextLength: 32768,
        memory: { layers: 24, kvBytesPerToken: 12288 }, sources: { ollama: { type: 'ollama', tag: TAG, ...source } }, ...extra }, { seed: true });
}

// A controller on an injected snapshot and runner process. `installer` and `runnerLocks` are the test's own; detection is the adapters'.
function controllerOn(t, { snap = () => snapshotOf(ABSENT_GPU), seed, imageContract = ARM64_CONTRACT, fileExists = (file) => file === IN_IMAGE,
    fetchImpl, installer, runnerLocks, dataDir = null, runners, resolveOllama } = {}) {
    const dir = dataDir ?? fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-ollama-controller-'));
    const started = [];
    const calls = [];
    const downloads = [];
    const weights = path.join(dir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const controller = createController({
        dataDir: dir,
        env: { PATH: '/usr/bin' },
        seedCatalog: seed,
        stateStore: createStateStore({ dataDir: dir }),
        snapshot: async () => snap(),
        // Model weights: nothing is downloaded, and a Run that got as far as asking is on record.
        download: async ({ artifact }) => { downloads.push(artifact); return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        startRunner({ command, args, env }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 9000 + started.length, command, args, env,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                async kill() { running = false; resolveExit({ code: null, signal: 'SIGKILL', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: fetchImpl || (async (url, options = {}) => {
            const pathname = new URL(String(url)).pathname;
            calls.push([pathname, options.method || 'GET', options.body ? JSON.parse(options.body) : null]);
            if (pathname === '/api/ps') {
                return { ok: true, status: 200, json: async () => ({ models: [{ name: TAG, size: 16, size_vram: 0, context_length: 4096 }] }) };
            }
            return { ok: true, status: 200, json: async () => ({}) };
        }),
        apiKeyFactory: () => KEY,
        pollMs: 2,
        stopGraceMs: 50,
        imageContract,
        fileExists,
        sharedModelsRoot: null,
        readMemory: () => ({ totalBytes: M1_TOTAL, availableBytes: M1_AVAILABLE }),
        readPressure: () => 0,
        unifiedGuardMs: 5,
        dropCache: () => true,
        ...(installer ? { installer } : {}),
        ...(runnerLocks ? { runnerLocks } : {}),
        ...(runners ? { runners } : {}),
        ...(resolveOllama ? { resolveOllama } : {}),
    });
    // The drain saves state and writes the log, so the directory (which this harness owns, given or not) goes only after it.
    t.after(async () => {
        await controller.drain();
        fs.rmSync(dir, { recursive: true, force: true });
    });
    return { controller, started, calls, downloads, dataDir: dir };
}

// The agent's lock with Ollama in it (a small .tar.zst stands in for the 1.5 GB archive), an installer over it whose
// download and unpack are stand-ins, and what they were asked to do.
function ollamaInstaller(t, { image = {} } = {}) {
    const root = tempDir(t, 'ollama-install');
    const archive = zlib.zstdCompressSync(Buffer.from('not a tar: the unpack below is a stand-in for GNU tar'));
    const agent = path.join(root, 'agent.json');
    fs.writeFileSync(agent, JSON.stringify(lockDocument({
        ollama: { version: '0.34.4', kind: 'archive', licence: { name: 'MIT', url: 'https://github.com/ollama/ollama/blob/v0.34.4/LICENSE' },
            files: [{ name: 'ollama-linux-arm64.tar.zst', url: 'https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-linux-arm64.tar.zst',
                size: archive.length, sha256: SHA(archive), strip: 0 }],
            check: { executable: 'bin/ollama' } },
    })));
    const imageFile = path.join(root, 'image.json');
    fs.writeFileSync(imageFile, JSON.stringify(lockDocument(image)));
    const locks = loadRunnerLocks({ image: imageFile, agent });
    const downloads = [];
    const unpacked = [];
    const runRoot = path.join(root, 'opt', 'runners');
    const installer = createRunnerInstaller({
        lock: locks, cacheRoot: path.join(root, 'data', 'runners'), runRoot,
        download: async ({ url, target }) => {
            downloads.push(url);
            fs.mkdirSync(path.dirname(target), { recursive: true });
            fs.writeFileSync(target, archive);
            return { bytesTransferred: archive.length };
        },
        inspect: async ({ target }) => (fs.existsSync(target) ? { state: 'complete', bytes: fs.statSync(target).size } : { state: 'absent', bytes: 0 }),
        statfs: async () => ({ bavail: 1e12, bsize: 1 }),
        run: async ({ args }) => {
            unpacked.push(args);
            const into = args[args.indexOf('-C') + 1];
            fs.mkdirSync(path.join(into, 'bin'), { recursive: true });
            fs.writeFileSync(path.join(into, 'bin', 'ollama'), '#!/bin/sh\n', { mode: 0o755 });
            return { code: 0, output: '' };
        },
    });
    return { installer, locks, downloads, unpacked, runDir: path.join(runRoot, 'ollama', '0.34.4') };
}

test('Ollama installed on demand runs from its runnable copy, with GPUs hidden and num_gpu 0 on the cpu profile', async (t) => {
    const lock = ollamaInstaller(t);
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-ollama-controller-'));
    // The manifest a finished pull leaves, and a tag pinned to exactly it.
    const pulled = writeOllamaModel(dataDir, TAG);
    const model = ollamaModel({ manifestDigest: pulled.manifestDigest, size: pulled.size });
    const { controller, started, calls } = controllerOn(t, { installer: lock.installer, seed: [model], dataDir });
    // Before the install: offered, with its parameters for the cpu profile and an Install.
    const before = (await controller.overview()).runners.find((runner) => runner.id === 'ollama');
    assert.equal(before.supported, true);
    assert.equal(before.installed, false);
    assert.deepEqual(Object.keys(before.paramSchema.properties), ['numCtx', 'numThread', 'keepAlive']);
    assert.equal('profileUnsupportedReason' in before, false);
    assert.equal(before.install.version, '0.34.4');
    assert.equal(before.install.installed, false);
    // Run before the install is refused, before anything starts.
    await assert.rejects(() => controller.run({ modelId: 'olla-cpu', runnerId: 'ollama', requestId: 'request-before1' }), { code: 'runner_not_installed' });
    // Install: only the lock's file is fetched, decompressed and unpacked with strip 0.
    const accepted = await controller.installRunner({ runnerId: 'ollama' });
    assert.equal(accepted.accepted, true);
    await until(() => controller.state.runnerInstalls?.ollama?.phase === 'installed');
    assert.deepEqual(lock.downloads, ['https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-linux-arm64.tar.zst']);
    assert.equal(lock.unpacked.length, 1);
    assert.ok(lock.unpacked[0].includes('--strip-components=0'));
    const after = (await controller.overview()).runners.find((runner) => runner.id === 'ollama');
    assert.deepEqual([after.installed, after.version], [true, '0.34.4']);
    // The model's row on cpu: admitted from the pinned size.
    const row = (await controller.overview()).models.find((entry) => entry.id === 'olla-cpu').runners.ollama;
    assert.equal(row.admission.status, 'ok', row.admission.reason);
    assert.equal(row.admission.estimate.weightsBytes, 16);
    // Run: the launch is the runnable copy's binary with no driver path and no device visible.
    await controller.run({ modelId: 'olla-cpu', runnerId: 'ollama', requestId: 'request-ollama1' });
    await until(() => controller.state.deployment?.phase === 'ready');
    const [process] = started;
    assert.equal(process.command, path.join(lock.runDir, 'bin', 'ollama'));
    assert.deepEqual(process.args, ['serve']);
    assert.equal(process.env.CUDA_VISIBLE_DEVICES, '');
    assert.equal(Object.hasOwn(process.env, 'LD_LIBRARY_PATH'), false);
    assert.equal(process.env.OLLAMA_KV_CACHE_TYPE, 'f16');
    assert.equal(process.env.OLLAMA_CONTEXT_LENGTH, '4096');
    assert.equal(Object.hasOwn(process.env, 'OLLAMA_FLASH_ATTENTION'), false);
    // The model is loaded with no layer on a GPU. The chat target carries the same options, but the chat responder does not
    // forward them (Ollama's /v1 endpoint takes no `options`): what keeps a chat request off the GPU is the hidden device.
    const load = calls.find(([pathname]) => pathname === '/api/generate');
    assert.deepEqual(load[2].options, { num_ctx: 4096, num_gpu: 0 });
    assert.deepEqual(controller.state.deployment.params, { numCtx: 4096, numThread: null, keepAlive: '30m' });
    assert.deepEqual(controller.chatTarget(), {
        runnerId: 'ollama', modelId: 'olla-cpu', baseUrl: 'http://127.0.0.1:18434', apiKey: null, model: TAG,
        requestOptions: { options: { num_ctx: 4096, num_gpu: 0 }, keep_alive: '30m' }, profile: 'cpu',
    });
    await controller.stop();
});

test('an entry only the agent lock has is offered where the runner has a policy, and the image\'s own executable wins', async (t) => {
    const lock = ollamaInstaller(t);
    const ollamaOf = async (h) => (await h.controller.overview()).runners.find((runner) => runner.id === 'ollama');
    // The cpu profile, on an image without Ollama: offered, with an Install.
    const cpu = controllerOn(t, { installer: lock.installer, seed: [] });
    const onCpu = await ollamaOf(cpu);
    assert.equal(onCpu.supported, true);
    assert.ok(onCpu.install);
    // Unified memory (DGX Spark) is unchanged: not available on this platform, no Install, refused before any download.
    const unified = controllerOn(t, { installer: lock.installer, seed: [], snap: () => snapshotOf(GB10) });
    const onUnified = await ollamaOf(unified);
    assert.equal((await unified.controller.overview()).profile, 'unified');
    assert.equal(onUnified.supported, false);
    assert.match(onUnified.unsupportedReason, /^Ollama is not available on this platform: this image does not include it\.$/);
    assert.equal('install' in onUnified, false);
    await assert.rejects(() => unified.controller.installRunner({ runnerId: 'ollama' }),
        (error) => error.code === 'runner_unavailable' && /not available on this platform/.test(error.message));
    // While the profile is undecided (a GPU that cannot be read yet) it stays as it was: not offered.
    const cold = controllerOn(t, { installer: lock.installer, seed: [], snap: () => snapshotOf({ available: false, state: 'unreadable', reason: 'nvidia-smi failed: timed out' }) });
    const onCold = await ollamaOf(cold);
    assert.equal((await cold.controller.overview()).profile, null);
    assert.equal(onCold.supported, false);
    assert.equal('install' in onCold, false);
    assert.deepEqual(lock.downloads, [], 'nothing was downloaded');
    // The image holds the executable: the image wins, so there is no install, and the image's binary runs.
    const present = controllerOn(t, { installer: lock.installer, seed: [], fileExists: (file) => [IN_IMAGE, '/opt/ollama/bin/ollama'].includes(file) });
    const inImage = await ollamaOf(present);
    assert.equal(inImage.supported, true);
    assert.equal('install' in inImage, false);
    // The controller says so at start, once; with no such binary it says nothing.
    const said = (h) => h.controller.status().then((status) => status.logs.filter((entry) => /runner lock: ollama is in the agent's lock/.test(entry.line)).map((entry) => entry.line));
    assert.deepEqual(await said(present), ["runner lock: ollama is in the agent's lock, but the image holds /opt/ollama/bin/ollama; the image's binary is used"]);
    assert.deepEqual(await said(cpu), []);
    await assert.rejects(() => present.controller.installRunner({ runnerId: 'ollama' }),
        (error) => error.code === 'runner_unavailable' && /Ollama is part of this image and needs no install/.test(error.message));
    // An id no lock lists is not installable at all; Uninstall still works for one a lock lists, under any profile.
    await assert.rejects(() => unified.controller.installRunner({ runnerId: 'nope' }), { code: 'not_installable' });
    await assert.rejects(() => unified.controller.uninstallRunner({ runnerId: 'nope' }), { code: 'not_installable' });
    assert.deepEqual(await unified.controller.uninstallRunner({ runnerId: 'ollama' }), { freedBytes: 0 });
    assert.deepEqual(lock.downloads, []);
});

test('an Ollama pull whose stored manifest differs from the pinned digest fails with identity_changed and nothing is loaded', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-ollama-identity-'));
    const pulled = writeOllamaModel(dataDir, TAG);
    const wrong = `sha256:${'f'.repeat(64)}`;
    const model = ollamaModel({ manifestDigest: wrong, size: pulled.size });
    // The image holds Ollama, so this is the pull's own check, on any profile.
    const { controller, calls } = controllerOn(t, { seed: [model], dataDir, imageContract: null, fileExists: () => true });
    await controller.run({ modelId: 'olla-cpu', runnerId: 'ollama', requestId: 'request-ident01' });
    await until(() => controller.state.deployment?.phase === 'error');
    assert.equal(controller.state.deployment.error, `The Ollama tag ${TAG} now resolves to ${pulled.manifestDigest}, not the pinned ${wrong}; update the model entry to accept it.`);
    assert.equal(calls.some(([pathname]) => pathname === '/api/generate'), false, 'the model was never loaded');
    assert.equal(controller.state.deployment.runner, null);
});

test('an Ollama tag is pinned at add with the manifest digest and the size of its layers', async (t) => {
    const manifest = manifestBytes({ config: 500, layers: [1000, 2000], spacing: 2 });
    const reg = await registry(t, { '/v2/library/qwen2.5/manifests/0.5b': { body: manifest } });
    const { controller, dataDir } = controllerOn(t, { seed: [], resolveOllama: (tag, options) => fetchOllamaRegistryManifest(tag, { baseUrl: reg.base, ...options }) });
    // The pin belongs to the cpu profile, which the dashboard's first overview commits.
    assert.equal((await controller.overview()).profile, 'cpu');
    const added = await controller.addModel({ id: 'user-olla', sources: { ollama: { type: 'ollama', tag: TAG } } });
    // The digest of the exact bytes the registry sent, and config plus layers.
    const pin = { type: 'ollama', tag: TAG, manifestDigest: `sha256:${sha256(manifest)}`, size: 3500 };
    assert.deepEqual(added.model.sources.ollama, pin);
    assert.deepEqual(controller.state.registry[0].sources.ollama, pin);
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(dataDir, 'state', 'controller.json'), 'utf8')).registry[0].sources.ollama, pin);
    // One manifest request, asking for the Docker manifest, with no credential.
    assert.deepEqual(reg.requests, [{ url: '/v2/library/qwen2.5/manifests/0.5b', method: 'GET',
        accept: 'application/vnd.docker.distribution.manifest.v2+json', authorization: null }]);
    // The same bytes in the Ollama store read back as the same identity: the pin uses the store's own formula.
    const target = manifestPath(path.join(dataDir, 'models', 'ollama'), TAG);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, manifest);
    const stored = readOllamaManifest(path.join(dataDir, 'models', 'ollama'), TAG);
    assert.deepEqual([stored.manifestDigest, stored.size], [pin.manifestDigest, pin.size]);
    // Other bytes, same content: another digest (it is the digest of the bytes, not of the JSON value).
    const reformatted = manifestBytes({ config: 500, layers: [1000, 2000], spacing: 0 });
    assert.notEqual(sha256(reformatted), sha256(manifest));
    const other = await registry(t, { '/v2/library/qwen2.5/manifests/0.5b': { body: reformatted } });
    assert.equal((await fetchOllamaRegistryManifest(TAG, { baseUrl: other.base })).manifestDigest, `sha256:${sha256(reformatted)}`);
});

// ---------------------------------------------------------------- llama.cpp's CPU build

const X86 = {
    available: true, name: 'NVIDIA GeForce RTX 3060 Laptop GPU', driverVersion: '595.91.07', memoryModel: 'dedicated',
    totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [],
    device: { pciDeviceId: '0x252010DE', computeCapability: '8.6', addressingMode: 'None' },
};
const AMD64_CONTRACT = Object.freeze({ architecture: 'amd64', llama_cpp: 'b11159', ik_llama_cpp: '20f7a72' });
const GGUF_SOURCE = { type: 'huggingface', repo: 'acme/models', file: 'small.gguf', revision: 'main', commit: 'a'.repeat(40), size: 400 * MIB, sha256: 'b'.repeat(64) };
const SMALL = validateModel({ id: 'small-cpu', displayName: 'Small', profiles: ['cpu', 'dedicated', 'unified'], contextLength: 32768,
    memory: { layers: 24, kvBytesPerToken: 12288 }, sources: { gguf: GGUF_SOURCE } }, { seed: true });

// The agent's lock with llama.cpp's CPU build in it (a small .tar.gz stands in for the release asset), an
// installer over it whose download and unpack are stand-ins, and what they were asked to do.
function cpuBuildInstaller(t) {
    const root = tempDir(t, 'cpu-build');
    const archive = Buffer.from('a release archive, in spirit');
    const agent = path.join(root, 'agent.json');
    fs.writeFileSync(agent, JSON.stringify(lockDocument({
        'llama.cpp-cpu': { version: 'b11295', kind: 'archive', licence: { name: 'MIT', url: 'https://github.com/ggml-org/llama.cpp/blob/b11295/LICENSE' },
            files: [{ name: 'llama-b11295-bin-ubuntu-arm64.tar.gz', url: 'https://github.com/ggml-org/llama.cpp/releases/download/b11295/llama-b11295-bin-ubuntu-arm64.tar.gz',
                size: archive.length, sha256: SHA(archive) }],
            check: { executable: 'llama-server' } },
    })));
    const imageFile = path.join(root, 'image.json');
    fs.writeFileSync(imageFile, JSON.stringify(lockDocument({})));
    const downloads = [];
    const unpacked = [];
    const runRoot = path.join(root, 'opt', 'runners');
    const installer = createRunnerInstaller({
        lock: loadRunnerLocks({ image: imageFile, agent }), cacheRoot: path.join(root, 'data', 'runners'), runRoot,
        download: async ({ url, target }) => {
            downloads.push(url);
            fs.mkdirSync(path.dirname(target), { recursive: true });
            fs.writeFileSync(target, archive);
            return { bytesTransferred: archive.length };
        },
        inspect: async ({ target }) => (fs.existsSync(target) ? { state: 'complete', bytes: fs.statSync(target).size } : { state: 'absent', bytes: 0 }),
        statfs: async () => ({ bavail: 1e12, bsize: 1 }),
        run: async ({ args }) => {
            unpacked.push(args);
            const into = args[args.indexOf('-C') + 1];
            fs.mkdirSync(into, { recursive: true });
            fs.writeFileSync(path.join(into, 'llama-server'), '#!/bin/sh\n', { mode: 0o755 });
            return { code: 0, output: '' };
        },
    });
    return { installer, downloads, unpacked, runDir: path.join(runRoot, 'llama.cpp-cpu', 'b11295') };
}

test('llama.cpp-cpu is offered only on the cpu profile and runs from its runnable copy', async (t) => {
    const lock = cpuBuildInstaller(t);
    const cpu = controllerOn(t, { installer: lock.installer, seed: [SMALL] });
    const { controller, started } = cpu;
    // On cpu: a runner of its own, with the CPU parameters, an Install, and a column of rows beside llama.cpp's.
    const before = await controller.overview();
    const build = before.runners.find((runner) => runner.id === 'llama.cpp-cpu');
    assert.deepEqual(before.runners.map((runner) => runner.id), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio', 'llama.cpp-cpu']);
    assert.deepEqual([build.displayName, build.pinnedVersion, build.supported, build.installed], ['llama.cpp (CPU build)', 'b11295', true, false]);
    assert.deepEqual(Object.keys(build.paramSchema.properties), ['ctxSize', 'parallel', 'loadMode', 'threads', 'chatTemplateKwargs']);
    assert.equal('profileUnsupportedReason' in build, false);
    assert.equal(build.install.version, 'b11295');
    assert.equal(build.install.licence.name, 'MIT');
    assert.deepEqual(before.models[0].weights.gguf.runners, ['llama.cpp', 'llama.cpp-cpu']);
    assert.equal(before.models[0].runners['llama.cpp-cpu'].admission.status, 'ok');
    assert.equal(before.models[0].runners['llama.cpp-cpu'].params.ctxSize, 4096);
    // Not installed yet: refused before anything is downloaded.
    await assert.rejects(() => controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp-cpu', requestId: 'request-before1' }), { code: 'runner_not_installed' });
    assert.deepEqual(cpu.downloads, []);
    await controller.installRunner({ runnerId: 'llama.cpp-cpu' });
    await until(() => controller.state.runnerInstalls?.['llama.cpp-cpu']?.phase === 'installed');
    assert.deepEqual(lock.downloads, ['https://github.com/ggml-org/llama.cpp/releases/download/b11295/llama-b11295-bin-ubuntu-arm64.tar.gz']);
    assert.ok(lock.unpacked[0].includes('--strip-components=1'), 'the default strip of one leading directory');
    assert.equal((await controller.overview()).runners.find((runner) => runner.id === 'llama.cpp-cpu').installed, true);
    // Run: the runnable copy's llama-server, on its own port, with the CPU flags, no driver path and no device visible.
    await controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp-cpu', requestId: 'request-cpubuild1' });
    await until(() => controller.state.deployment?.phase === 'ready');
    const [process] = started;
    assert.equal(process.command, path.join(lock.runDir, 'llama-server'));
    const flags = process.args.join(' ');
    for (const expected of ['--port 18085', '--device none', '--n-gpu-layers 0', '--fit off', '--flash-attn on', '--ctx-size 4096', '--load-mode mmap']) {
        assert.ok(flags.includes(expected), expected);
    }
    assert.equal(process.env.CUDA_VISIBLE_DEVICES, '');
    assert.equal(Object.hasOwn(process.env, 'LD_LIBRARY_PATH'), false);
    assert.deepEqual([controller.state.deployment.profile, controller.state.deployment.runnerId], ['cpu', 'llama.cpp-cpu']);
    assert.equal(controller.chatTarget().baseUrl, 'http://127.0.0.1:18085');
    assert.equal(controller.chatTarget().profile, 'cpu');
    await controller.stop();
    // On the GPU profiles it is not offered at all, and Run, preview and Install refuse it before anything downloads.
    for (const [label, gpu, contract] of [['dedicated', X86, null], ['dedicated', X86, AMD64_CONTRACT], ['unified', GB10, ARM64_CONTRACT], ['unified', GB10, null]]) {
        const h = controllerOn(t, { installer: lock.installer, seed: [SMALL], snap: () => snapshotOf(gpu), imageContract: contract });
        const overview = await h.controller.overview();
        assert.equal(overview.profile, label);
        assert.equal(overview.runners.some((runner) => runner.id === 'llama.cpp-cpu'), false, label);
        assert.equal('llama.cpp-cpu' in overview.models[0].runners, false, label);
        assert.equal(overview.models[0].weights.gguf.runners.includes('llama.cpp-cpu'), false, label);
        const why = new RegExp(`^llama\\.cpp \\(CPU build\\) runs only on the cpu profile; this machine uses the ${label} profile\\.$`);
        await assert.rejects(() => h.controller.run({ modelId: 'small-cpu', runnerId: 'llama.cpp-cpu', requestId: `request-${label}-${contract ? 'c' : 'n'}` }),
            (error) => error.code === 'runner_unavailable' && why.test(error.message), label);
        await assert.rejects(() => h.controller.installRunner({ runnerId: 'llama.cpp-cpu' }),
            (error) => error.code === 'runner_unavailable' && why.test(error.message), label);
        const preview = (await h.controller.overview({ preview: { modelId: 'small-cpu', runnerId: 'llama.cpp-cpu' } })).preview;
        assert.equal(preview.admission.status, 'incompatible', label);
        assert.match(preview.admission.reason, why, label);
        assert.deepEqual(h.downloads, [], label);
        assert.equal(h.controller.state.deployment, null, label);
    }
    assert.deepEqual(lock.downloads.length, 1, 'only the install on the cpu profile fetched anything');
});

// ---------------------------------------------------------------- the shipped locks

const CATALOG = new URL('../catalog/', import.meta.url);

// What the release metadata on GitHub gave for each asset (gh release view --json assets), 2026-09-30.
const PINS = Object.freeze({
    arm64: {
        ollama: {
            version: '0.34.4', licence: 'https://github.com/ollama/ollama/blob/v0.34.4/LICENSE', executable: 'bin/ollama',
            file: { name: 'ollama-linux-arm64.tar.zst', url: 'https://github.com/ollama/ollama/releases/download/v0.34.4/ollama-linux-arm64.tar.zst',
                size: 1549684612, sha256: '96f50a1192133028cf4e010d8c333f8af14b1505db6be7b2034c11487e7fd7e6', strip: 0 },
        },
        'llama.cpp-cpu': {
            version: 'b11295', licence: 'https://github.com/ggml-org/llama.cpp/blob/b11295/LICENSE', executable: 'llama-server',
            file: { name: 'llama-b11295-bin-ubuntu-arm64.tar.gz', url: 'https://github.com/ggml-org/llama.cpp/releases/download/b11295/llama-b11295-bin-ubuntu-arm64.tar.gz',
                size: 13577862, sha256: '439dff1fdb5d223f1d1e1adacd6845abff12aa518b746e2a2f4f01d55d57aae9' },
        },
    },
    amd64: {
        'llama.cpp-cpu': {
            version: 'b11295', licence: 'https://github.com/ggml-org/llama.cpp/blob/b11295/LICENSE', executable: 'llama-server',
            file: { name: 'llama-b11295-bin-ubuntu-x64.tar.gz', url: 'https://github.com/ggml-org/llama.cpp/releases/download/b11295/llama-b11295-bin-ubuntu-x64.tar.gz',
                size: 17531126, sha256: 'cd54d7dcf8818acc69c54dfc559b36da5f693d30b223e8dde87b4da4570664ed' },
        },
    },
});

test('the shipped agent locks pin exactly what the release metadata gave: Ollama on arm64 and the CPU build on both', () => {
    for (const [arch, expected] of Object.entries(PINS)) {
        const file = new URL(`runners.lock.linux-${arch}.json`, CATALOG);
        const text = fs.readFileSync(file, 'utf8');
        // JSON in two-space indentation (DS006), and a lock the loader accepts as it is.
        assert.equal(text, `${JSON.stringify(JSON.parse(text), null, 2)}\n`, arch);
        const lock = loadRunnerLock(file.pathname, { requireExecutable: true });
        assert.deepEqual(Object.keys(lock.runners).sort(), Object.keys(expected).sort(), arch);
        for (const [id, pin] of Object.entries(expected)) {
            const entry = lock.runners[id];
            assert.deepEqual([entry.version, entry.kind, entry.licence.url, entry.licence.name], [pin.version, 'archive', pin.licence, 'MIT'], `${arch} ${id}`);
            assert.deepEqual([entry.licence.requiresAcceptance, entry.licence.proprietary], [false, false], `${arch} ${id}`);
            assert.equal(entry.files.length, 1);
            assert.deepEqual({ ...entry.files[0] }, pin.file, `${arch} ${id}`);
            assert.equal(entry.totalBytes, pin.file.size);
            // The file the install proves, and the adapters launch: the agent's lock names it.
            assert.equal(entry.check.executable, pin.executable, `${arch} ${id}`);
            // Every file is on an allowed host, is the asset of the tag it names, and the runner has an adapter that pins the same version.
            assert.equal(new URL(entry.files[0].url).hostname, 'github.com');
            assert.ok(entry.files[0].url.includes(`/download/${entry.version === '0.34.4' ? 'v0.34.4' : entry.version}/`), `${arch} ${id}`);
            assert.ok(Object.hasOwn(RUNNERS, id), `${arch} ${id} has an adapter`);
            assert.equal(RUNNERS[id].pinnedVersion, entry.version, `${arch} ${id}`);
        }
    }
    // The amd64 image has Ollama in it, so its agent lock does not list it: the image's binary is the one that runs.
    assert.equal(Object.hasOwn(loadRunnerLock(new URL('runners.lock.linux-amd64.json', CATALOG).pathname).runners, 'ollama'), false);
    // Each of the workflow's architectures has its lock, and every lock file has its job.
    if (repositoryRoot) {
        const workflow = fs.readFileSync(WORKFLOW, 'utf8');
        const arches = [...workflow.matchAll(/- arch: (\w+)/g)].map((match) => match[1]).sort();
        const locks = fs.readdirSync(CATALOG).map((name) => /^runners\.lock\.linux-(\w+)\.json$/.exec(name)?.[1]).filter(Boolean).sort();
        assert.deepEqual(arches, locks);
    }
});

test('a licence link must be https, and every string an agent lock or a registry supplies is escaped in the dashboard', () => {
    const one = (licence) => () => validateRunnerLock(lockDocument({ r: { ...lockRunner('r.tar.gz'), licence: { name: 'MIT', ...licence } } }));
    assert.ok(one({ url: 'https://example.org/LICENSE', source: 'https://example.org/src' })());
    for (const url of ['javascript:alert(1)', 'data:text/html,<script>', 'http://example.org/LICENSE', 'example.org/LICENSE', '//example.org', 'file:///etc/passwd']) {
        assert.throws(one({ url }), /licence\.url must be an https URL/, url);
        assert.throws(one({ url: 'https://example.org/LICENSE', source: url }), /licence\.source must be an https URL/, url);
    }
    // What a lock, a registry or a manifest puts in a runner's name, version, reason and licence reaches the page only escaped.
    const hostile = '<img src=x onerror=alert(1)>"\'&';
    const runner = {
        id: 'llama.cpp-cpu', displayName: hostile, supported: true, installed: false, version: hostile, reason: hostile, enabled: true,
        install: {
            version: hostile, totalBytes: 10, installed: false, installing: false, files: 1,
            licence: { name: hostile, url: `https://example.org/${hostile}`, source: `https://example.org/s/${hostile}`, notice: hostile, requiresAcceptance: false },
            state: { phase: 'error', error: hostile, pausedReason: hostile, licence: { acceptedBy: hostile, acceptedAt: hostile } },
        },
    };
    const lacking = { id: 'ollama', displayName: hostile, supported: false, unsupportedReason: hostile, profileUnsupportedReason: hostile };
    const html = runnersPanelHtml([runner, lacking]);
    assert.doesNotMatch(html, /<img/);
    assert.doesNotMatch(html, /onerror=alert\(1\)>/);
    assert.doesNotMatch(html, /"><img|'><img/);
    assert.match(html, /&lt;img src=x onerror=alert\(1\)&gt;&quot;&#39;&amp;/);
    // The dialog text is plain text, not markup, so it carries the strings as they are.
    assert.ok(installMessage(runner).includes(hostile));
});
