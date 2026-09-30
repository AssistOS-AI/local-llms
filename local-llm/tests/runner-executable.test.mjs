// The unpacked runnable copy must hold the binary the adapter launches (DS004): the lock says where
// (`check.executable`), the installer refuses a copy without it and writes no marker, the install check
// proves it (ldd, --version under the CPU environment), and an archive that decompresses to less than it
// should, or to far more, is refused. Node's zstd decoder does not report a stream that was cut short, so
// these checks are what catch it.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import zlib from 'node:zlib';

import { createRunnerInstaller, runTool } from '../src/controller/runnerInstaller.mjs';
import { agentRunnerLockFile, archiveCompressionFor, loadRunnerLock, loadRunnerLocks, validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { RUNNERS } from '../src/runners/index.mjs';
import { probeExecutable } from '../tools/runner_install_check.mjs';

const SHA = (buffer) => crypto.createHash('sha256').update(buffer).digest('hex');

function tempDir(t, name) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-${name}-`));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

const lockOf = (runners) => ({ schema: 'local-llm.runners-lock/v1', runners });
const archiveEntry = (name, extra = {}) => ({
    version: '1.0.0', kind: 'archive', licence: { name: 'MIT', url: 'https://example.org/LICENSE' },
    files: [{ name, url: `https://github.com/example/r/releases/download/v1/${name}`, size: 10, sha256: 'a'.repeat(64) }], ...extra,
});

// ---------------------------------------------------------------- the lock says where the binary is

test('check.executable is a plain relative path, required in the agent lock for archive runners and optional in the image lock', (t) => {
    const withExecutable = (executable, options) => () => validateRunnerLock(lockOf({ r: archiveEntry('r.tar.gz', { check: { executable } }) }), options);
    for (const good of ['llama-server', 'bin/ollama', 'a/b.c/d_e-f+g']) assert.equal(withExecutable(good)().runners.r.check.executable, good, good);
    for (const bad of ['', '/bin/ollama', '../x', 'a/../x', 'a/./x', './x', 'a//b', 'bin/', 'a b', 'a\\b', 'a/b/c/d/e/f/g/h/i', 'x'.repeat(81), 7, null, ['x']]) {
        assert.throws(withExecutable(bad), /check\.executable must be a relative path inside the runnable copy/, JSON.stringify(bad));
    }
    // The image's lock may omit it (vLLM, TabbyAPI and LM Studio have no path the check knows); the agent's may not.
    const without = (kind = 'archive', options) => () => validateRunnerLock(lockOf({
        r: kind === 'archive' ? archiveEntry('r.tar.gz') : { ...archiveEntry('w-1-py3-none-any.whl'), kind: 'python' },
    }), options);
    assert.equal('executable' in without()().runners.r.check, false);
    assert.throws(without('archive', { requireExecutable: true }), /check\.executable is required: the install must prove the file the adapter launches/);
    assert.throws(() => validateRunnerLock(lockOf({ r: archiveEntry('r.tar.gz', { check: {} }) }), { requireExecutable: true }), /check\.executable is required/);
    assert.ok(without('python', { requireExecutable: true })(), 'a python runner has no single binary to name');
    // The agent's lock is loaded that way: an entry without it leaves the agent lock out, with the reason.
    const dir = tempDir(t, 'exe-locks');
    const write = (name, runners) => { const file = path.join(dir, name); fs.writeFileSync(file, JSON.stringify(lockOf(runners))); return file; };
    const image = write('image.json', { keep: archiveEntry('keep.tar.gz') });
    const missing = loadRunnerLocks({ image, agent: write('agent-missing.json', { 'llama.cpp-cpu': archiveEntry('x.tar.gz') }) });
    assert.deepEqual(Object.keys(missing.runners), ['keep']);
    assert.match(missing.ignored[0].reason, /check\.executable is required/);
    const given = loadRunnerLocks({ image, agent: write('agent-given.json', { 'llama.cpp-cpu': archiveEntry('x.tar.gz', { check: { executable: 'llama-server' } }) }) });
    assert.deepEqual(Object.keys(given.runners).sort(), ['keep', 'llama.cpp-cpu']);
    assert.deepEqual(given.ignored, []);
});

test('the shipped agent locks name the file each adapter launches, and the adapters launch exactly that', () => {
    for (const arch of ['arm64', 'x64']) {
        const lock = loadRunnerLock(agentRunnerLockFile(arch), { requireExecutable: true });
        for (const [id, entry] of Object.entries(lock.runners)) {
            assert.ok(entry.check.executable, `${arch} ${id}`);
            const runnerDir = '/opt/runners/probe';
            const launch = RUNNERS[id].buildLaunch(id === 'ollama'
                ? { params: {}, port: 18434, dataDir: '/data/local-llm', runnerDir }
                : { artifactPath: '/m.gguf', params: {}, port: 18085, apiKey: 'k'.repeat(43), model: { id: 'm', contextLength: 32768 }, profile: 'cpu', runnerDir });
            assert.equal(launch.command, path.join(runnerDir, entry.check.executable), `${arch} ${id}`);
        }
    }
});

test('a python-kind entry cannot list a .tar.zst, as a source archive or as a data file, and the installer and validator agree on what an archive is', () => {
    const python = (file) => () => validateRunnerLock(lockOf({ r: { ...archiveEntry('w-1-py3-none-any.whl'), kind: 'python', files: [file] } }));
    const zst = { name: 'data.tar.zst', url: 'https://github.com/example/r/releases/download/v1/data.tar.zst', size: 10, sha256: 'a'.repeat(64) };
    for (const extra of [{}, { into: 'data' }, { extract: 'src' }, { strip: 0 }]) {
        assert.throws(python({ ...zst, ...extra }), /is a \.tar\.zst archive, which only an archive runner may list/, JSON.stringify(extra));
    }
    // The one predicate both ask: a .tar.zst is an archive for an archive runner only, a .tar.gz for either.
    assert.deepEqual([['archive', 'a.tar.zst'], ['python', 'a.tar.zst'], ['archive', 'a.tar.gz'], ['python', 'a.tgz'], ['archive', 'a.zip'], ['python', 'a.whl']]
        .map(([kind, name]) => archiveCompressionFor(kind, name)), ['zst', null, 'gz', 'gz', null, null]);
    // A .tar.gz source archive with extract is still fine for python, a data file with into is, and a .tar.gz data file is not.
    const gz = { ...zst, name: 'src.tar.gz' };
    assert.ok(python({ ...gz, extract: 'src' })());
    assert.throws(python({ ...gz, into: 'data' }), /into is only for data files, not wheels or archives/);
    assert.ok(python({ ...zst, name: 'o200k_base.tiktoken', into: 'tiktoken' })());
});

// ---------------------------------------------------------------- the installer proves the binary before the marker

// GNU tar's --touch (the agent's flag: no file times on the container's overlay) is unknown to macOS's bsdtar; the rest is the same.
const tarRun = (options) => runTool({ ...options, args: process.platform === 'linux' ? options.args : options.args.filter((arg) => arg !== '--touch') });

// A tar of a few files (name -> { content, mode }), archived from `top` (its entries sit under `top/` unless `top` is '.').
function tarOf(t, files, top = '.') {
    const src = tempDir(t, 'exe-src');
    for (const [name, { content = '#!/bin/sh\necho hi\n', mode = 0o755 }] of Object.entries(files)) {
        const target = path.join(src, name);
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, content, { mode });
    }
    const archive = path.join(tempDir(t, 'exe-out'), 'x.tar');
    execFileSync('tar', ['-cf', archive, '-C', src, ...(top === '.' ? fs.readdirSync(src) : [top])]);
    return fs.readFileSync(archive);
}

const NAME = 'runner-1.0.0.tar.zst';

// An installer over a cache that already holds `bytes`, pinned as `pinned`; the entry declares `executable` and `strip`.
function installerOver(t, bytes, { executable = 'llama-server', strip = 0, run = tarRun, pinned = bytes } = {}) {
    const root = tempDir(t, 'exe-installer');
    const lock = validateRunnerLock(lockOf({
        testrunner: archiveEntry(NAME, {
            files: [{ name: NAME, url: `https://github.com/example/r/releases/download/1.0.0/${NAME}`, size: pinned.length, sha256: SHA(pinned), strip }],
            ...(executable ? { check: { executable } } : {}),
        }),
    }));
    const cacheRoot = path.join(root, 'data', 'runners');
    const runRoot = path.join(root, 'opt', 'runners');
    const installer = createRunnerInstaller({
        lock, cacheRoot, runRoot, run,
        inspect: async ({ target }) => ({ state: 'complete', bytes: fs.statSync(target).size }),
    });
    const entry = installer.entryFor('testrunner');
    const files = path.join(cacheRoot, 'testrunner', '1.0.0', 'files');
    fs.mkdirSync(files, { recursive: true });
    fs.writeFileSync(path.join(files, NAME), bytes);
    fs.writeFileSync(path.join(cacheRoot, 'testrunner', '1.0.0', 'installed.json'), JSON.stringify({ id: 'testrunner', version: '1.0.0', digest: entry.digest }));
    const runDir = path.join(runRoot, 'testrunner', '1.0.0');
    return { installer, runDir, marker: path.join(runDir, '.ready.json'), stageDir: path.join(runRoot, '.stage-testrunner') };
}

const missingExecutable = (relative) => (error) => error.code === 'install_failed' && error.message.startsWith(`The unpacked testrunner 1.0.0 has no executable ${relative}: `);

test('an unpacked copy that holds its binary installs; one that is empty, mis-stripped or not executable leaves no marker and no copy', async (t) => {
    // The right archive and the right strip: installed, marker written.
    const good = installerOver(t, zlib.zstdCompressSync(tarOf(t, { 'llama-server': {}, 'libggml.so': { mode: 0o644 } })));
    assert.equal((await good.installer.ensureRunnable('testrunner')).rebuilt, true);
    assert.ok(fs.existsSync(good.marker));
    assert.ok(fs.statSync(path.join(good.runDir, 'llama-server')).mode & 0o111);
    // An archive of nothing: a valid empty tar, which unpacks to an empty copy.
    const empty = installerOver(t, zlib.zstdCompressSync(Buffer.alloc(1024)));
    await assert.rejects(() => empty.installer.ensureRunnable('testrunner'), missingExecutable('llama-server'));
    assert.equal(fs.existsSync(empty.marker), false, 'no marker');
    assert.equal(fs.existsSync(empty.runDir), false, 'no copy left');
    assert.equal(fs.existsSync(empty.stageDir), false, 'no staging');
    assert.equal((await empty.installer.describe('testrunner')).runnable, false);
    // A wrong strip, both ways: the entries sit under a directory and the lock says strip 0 ...
    const top = zlib.zstdCompressSync(tarOf(t, { 'runner-1.0.0/llama-server': {} }, 'runner-1.0.0'));
    const tooDeep = installerOver(t, top, { strip: 0 });
    await assert.rejects(() => tooDeep.installer.ensureRunnable('testrunner'), missingExecutable('llama-server'));
    assert.equal(fs.existsSync(tooDeep.marker), false);
    // ... and they sit at the root and the lock strips one directory.
    const flat = zlib.zstdCompressSync(tarOf(t, { 'llama-server': {} }));
    const tooShallow = installerOver(t, flat, { strip: 1 });
    await assert.rejects(() => tooShallow.installer.ensureRunnable('testrunner'), missingExecutable('llama-server'));
    assert.equal(fs.existsSync(tooShallow.marker), false);
    // With the strip that matches each archive, both install.
    assert.equal((await installerOver(t, top, { strip: 1 }).installer.ensureRunnable('testrunner')).rebuilt, true);
    assert.equal((await installerOver(t, flat, { strip: 0 }).installer.ensureRunnable('testrunner')).rebuilt, true);
    // Not executable, a directory where the file should be, and a link out of the copy.
    const plain = installerOver(t, zlib.zstdCompressSync(tarOf(t, { 'llama-server': { mode: 0o644 } })));
    await assert.rejects(() => plain.installer.ensureRunnable('testrunner'), (error) => missingExecutable('llama-server')(error) && /it is not executable\.$/.test(error.message));
    const directory = installerOver(t, zlib.zstdCompressSync(tarOf(t, { 'llama-server/inside': {} })));
    await assert.rejects(() => directory.installer.ensureRunnable('testrunner'), (error) => /it is not a regular file\.$/.test(error.message));
    const outside = tempDir(t, 'exe-outside');
    fs.writeFileSync(path.join(outside, 'elsewhere'), '#!/bin/sh\n', { mode: 0o755 });
    const linked = installerOver(t, zlib.zstdCompressSync(tarOf(t, { placeholder: {} })), {
        run: async (options) => {
            const into = options.args[options.args.indexOf('-C') + 1];
            fs.mkdirSync(into, { recursive: true });
            fs.symlinkSync(path.join(outside, 'elsewhere'), path.join(into, 'llama-server'));
            return { code: 0, output: '' };
        },
    });
    await assert.rejects(() => linked.installer.ensureRunnable('testrunner'), (error) => /it leads outside the runnable copy\.$/.test(error.message));
    assert.equal(fs.existsSync(linked.marker), false);
});

test('an archive cut short so that it decompresses to an empty .tar does not install', async (t) => {
    const whole = zlib.zstdCompressSync(Buffer.concat([tarOf(t, { 'llama-server': {} }), Buffer.alloc(1024)]));
    // Node's decoder ends quietly on a truncated stream: 12 bytes are a header and no data.
    const cut = whole.subarray(0, 12);
    const decoded = zlib.createZstdDecompress();
    const sizes = [];
    await new Promise((resolve, reject) => { decoded.on('data', (chunk) => sizes.push(chunk.length)).on('end', resolve).on('error', reject).end(cut); });
    assert.deepEqual(sizes, [], 'the premise: no error, and nothing decoded');
    // GNU tar takes an empty file for an empty archive, so the stand-in here does too; anything else is unpacked for real.
    const tar = (options) => (fs.statSync(options.args[1]).size === 0 ? Promise.resolve({ code: 0, output: '' }) : tarRun(options));
    const h = installerOver(t, cut, { run: tar });
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'), missingExecutable('llama-server'));
    assert.equal(fs.existsSync(h.marker), false);
    assert.equal(fs.existsSync(h.runDir), false);
    // The same entry over the whole archive installs.
    assert.equal((await installerOver(t, whole, { run: tar }).installer.ensureRunnable('testrunner')).rebuilt, true);
});

test('a .tar.zst that unpacks to far more than its size is refused, with no staging and no marker', async (t) => {
    // 4 MiB of zeros compress to a few dozen bytes: more than 32 times that, and more than the 1 MiB floor.
    const bomb = zlib.zstdCompressSync(Buffer.alloc(4 * 1024 * 1024));
    assert.ok(bomb.length < 1024);
    const calls = [];
    const h = installerOver(t, bomb, { run: async (options) => { calls.push(options.args); return { code: 0, output: '' }; } });
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'),
        (error) => error.code === 'install_failed' && /^decompressing runner-1\.0\.0\.tar\.zst failed: it unpacks to more than 1048576 bytes \(32 times its size\)$/.test(error.message));
    assert.deepEqual(calls, [], 'tar never ran');
    assert.equal(fs.existsSync(h.stageDir), false);
    assert.equal(fs.existsSync(h.marker), false);
    // The cap is a multiple of the compressed size above the floor: a stream that unpacks to 3 times its size is nowhere near it.
    const random = crypto.randomBytes(2 * 1024 * 1024);
    const fine = installerOver(t, zlib.zstdCompressSync(Buffer.concat([random, random, random])), { executable: null, run: async () => ({ code: 0, output: '' }) });
    assert.equal((await fine.installer.ensureRunnable('testrunner')).rebuilt, true);
});

// ---------------------------------------------------------------- the install check proves the binary too

function copyWith(t, script, { mode = 0o755 } = {}) {
    const runDir = tempDir(t, 'exe-copy');
    if (script !== null) fs.writeFileSync(path.join(runDir, 'llama-server'), script, { mode });
    return runDir;
}
const entryOf = (extra = {}) => validateRunnerLock(lockOf({ 'llama.cpp-cpu': archiveEntry('x.tar.gz', { version: 'b11295', check: { executable: 'llama-server', ...extra } }) })).runners['llama.cpp-cpu'];
const noLibraries = () => [];

test('the install check finds the executable, runs ldd on it and runs --version under the CPU environment', async (t) => {
    // The CPU environment: no driver library path, no device visible, and the version it prints is the pinned one.
    const runDir = copyWith(t, '#!/bin/sh\necho "version: 11295 (abcdef) cuda=${CUDA_VISIBLE_DEVICES-unset} lib=${LD_LIBRARY_PATH-unset}"\n');
    const ok = await probeExecutable({ entry: entryOf(), runDir, ldd: noLibraries });
    assert.deepEqual(ok.problems, []);
    assert.equal(ok.report.found, true);
    assert.equal(ok.report.version.output, 'version: 11295 (abcdef) cuda= lib=unset');
    assert.equal(ok.report.version.expected, '11295');
    // ldd runs on that file, and a library nobody provides fails it; the driver's, a shipped one and a listed optional one do not.
    const asked = [];
    const libs = (file) => { asked.push(file); return ['libmissing.so.1', 'libcuda.so.1', 'libshipped.so', 'libmpi.so.40']; };
    const linked = await probeExecutable({ entry: entryOf({ optionalLibraries: { 'libmpi.so*': 'multi-node only' } }), runDir, ldd: libs, provided: new Set(['libshipped.so']) });
    assert.deepEqual(asked, [path.join(runDir, 'llama-server')]);
    assert.equal(linked.problems.length, 1);
    assert.match(linked.problems[0], /llama-server needs libraries that are neither in the copy, nor supplied by the driver grant, nor optional: libmissing\.so\.1$/);
    assert.deepEqual(linked.report.libraries.providedByEnvironment, ['libshipped.so']);
    assert.deepEqual(Object.keys(linked.report.libraries.optional), ['libmpi.so.40']);
    // Not there, not executable, a --version that fails or names another version, or one that never ends.
    const gone = await probeExecutable({ entry: entryOf(), runDir: copyWith(t, null), ldd: noLibraries });
    assert.match(gone.problems[0], /has no executable llama-server: it is not in the runnable copy/);
    assert.equal(gone.report.found, false);
    const plain = await probeExecutable({ entry: entryOf(), runDir: copyWith(t, '#!/bin/sh\n', { mode: 0o644 }), ldd: noLibraries });
    assert.match(plain.problems[0], /it is not executable/);
    const failing = await probeExecutable({ entry: entryOf(), runDir: copyWith(t, '#!/bin/sh\necho "no GPU" >&2\nexit 3\n'), ldd: noLibraries });
    assert.match(failing.problems[0], /llama-server --version failed \(exit 3\): no GPU/);
    const other = await probeExecutable({ entry: entryOf(), runDir: copyWith(t, '#!/bin/sh\necho "version: 11159 (abc)"\n'), ldd: noLibraries });
    assert.match(other.problems[0], /llama-server --version does not name 11295: version: 11159/);
    // An Ollama-style version line names the pinned version as it is; the tag's 'b' is dropped only before digits.
    const ollama = validateRunnerLock(lockOf({ ollama: archiveEntry('o.tar.gz', { version: '0.34.4', check: { executable: 'llama-server' } }) })).runners.ollama;
    const client = await probeExecutable({ entry: ollama, runDir: copyWith(t, '#!/bin/sh\necho "Warning: could not connect to a running Ollama instance"\necho "Warning: client version is 0.34.4"\n'), ldd: noLibraries });
    assert.deepEqual(client.problems, []);
});
