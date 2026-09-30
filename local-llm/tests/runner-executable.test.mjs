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
import { Duplex, Transform } from 'node:stream';
import zlib from 'node:zlib';

import { createController } from '../src/controller/deployments.mjs';
import { createRunnerInstaller, runTool } from '../src/controller/runnerInstaller.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { agentRunnerLockFile, archiveCompressionFor, loadRunnerLock, loadRunnerLocks, validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { RUNNERS } from '../src/runners/index.mjs';
import { IMAGE_LOCK } from './overview-scenarios.mjs';
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
    // Members go in the order the files were named, so a test can cut after the first.
    const names = [...new Set(Object.keys(files).map((name) => name.split('/')[0]))];
    // No AppleDouble members from macOS's tar.
    execFileSync('tar', ['-cf', archive, '-C', src, ...(top === '.' ? names : [top])], { env: { ...process.env, COPYFILE_DISABLE: '1' } });
    return fs.readFileSync(archive);
}

const NAME = 'runner-1.0.0.tar.zst';

// An installer over a cache that already holds `bytes`, pinned as `pinned`; the entry declares `executable` and `strip`.
function installerOver(t, bytes, { executable = 'llama-server', strip = 0, run = tarRun, pinned = bytes, zstdDecoder } = {}) {
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
        lock, cacheRoot, runRoot, run, ...(zstdDecoder ? { zstdDecoder } : {}),
        inspect: async ({ target }) => ({ state: 'complete', bytes: fs.statSync(target).size }),
    });
    const entry = installer.entryFor('testrunner');
    const files = path.join(cacheRoot, 'testrunner', '1.0.0', 'files');
    fs.mkdirSync(files, { recursive: true });
    fs.writeFileSync(path.join(files, NAME), bytes);
    fs.writeFileSync(path.join(cacheRoot, 'testrunner', '1.0.0', 'installed.json'), JSON.stringify({ id: 'testrunner', version: '1.0.0', digest: entry.digest }));
    const runDir = path.join(runRoot, 'testrunner', '1.0.0');
    return { installer, runDir, marker: path.join(runDir, '.ready.json'), stageDir: path.join(runRoot, '.stage-testrunner'), tmpDir: path.join(runRoot, '.tmp-testrunner'), runRoot };
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

const TAR_BLOCK = 512;

// Where the member called `name` ends in a tar: its header block and its data padded to whole blocks, found by walking the headers
// (bsdtar on a Mac puts AppleDouble and pax members in front of it).
function memberEnd(tar, name) {
    for (let at = 0; at + TAR_BLOCK <= tar.length;) {
        const header = tar.subarray(at, at + TAR_BLOCK);
        const size = parseInt(header.subarray(124, 135).toString().replace(/\0/g, '').trim() || '0', 8);
        const end = at + TAR_BLOCK + Math.ceil(size / TAR_BLOCK) * TAR_BLOCK;
        if (header.subarray(0, 100).toString().replace(/\0.*/s, '') === name) return end;
        at = end;
    }
    throw new Error(`no member ${name}`);
}

// Nothing is unpacked from a .tar that is not whole; a stand-in for GNU tar that takes an empty file for an empty archive, as it does.
const tarOrNothing = (options) => (fs.statSync(options.args[1]).size === 0 ? Promise.resolve({ code: 0, output: '' }) : tarRun(options));

const refusedCutShort = (label) => (error) => error.code === 'install_failed'
    && new RegExp(`^decompressing ${label.replaceAll('.', '\\.')} failed: (Z_BUF_ERROR|ZSTD_error_[a-z_]+|it was cut short \\(no end-of-archive blocks\\))`).test(error.message);

test('an archive cut short does not install, whether the Node version reports the cut or ends quietly', async (t) => {
    const tar = tarOf(t, { 'llama-server': { content: '#!/bin/sh\n'.padEnd(1000, '#') }, 'lib/libggml-cpu.so': { mode: 0o644, content: 'x'.repeat(3000) } });
    const whole = zlib.zstdCompressSync(tar);
    // Cut at many places: Node 24 says so (Z_BUF_ERROR), Node 25.8 ends quietly with what it had, down to an empty .tar.
    // Either way the .tar is not whole, nothing is unpacked, and there is no marker and no copy.
    for (const at of [4, 12, 20, 100, Math.floor(whole.length / 2), whole.length - 1]) {
        const calls = [];
        const h = installerOver(t, whole.subarray(0, at), { run: async (options) => { calls.push(options.args); return tarOrNothing(options); } });
        await assert.rejects(() => h.installer.ensureRunnable('testrunner'), refusedCutShort('runner-1.0.0.tar.zst'), `cut at ${at}`);
        assert.deepEqual(calls, [], `cut at ${at}: tar never ran`);
        assert.equal(fs.existsSync(h.marker), false, `cut at ${at}`);
        assert.equal(fs.existsSync(h.runDir), false, `cut at ${at}`);
        assert.equal(fs.existsSync(h.stageDir), false, `cut at ${at}`);
    }
    // The whole archive installs, every member in place.
    const ok = installerOver(t, whole);
    assert.equal((await ok.installer.ensureRunnable('testrunner')).rebuilt, true);
    assert.ok(fs.existsSync(path.join(ok.runDir, 'lib', 'libggml-cpu.so')));
});

test('a tar cut inside a member of zeros is refused, though its last blocks are zeros: its size is not a multiple of 512', async (t) => {
    // The cut keeps a whole launch file and 2,500 bytes of a member that is all zeros, so the end looks like the marker and only the size gives it away.
    const tar = tarOf(t, { 'llama-server': {}, 'zeros.bin': { mode: 0o644, content: Buffer.alloc(10_000) } });
    const dataStart = memberEnd(tar, 'zeros.bin') - Math.ceil(10_000 / TAR_BLOCK) * TAR_BLOCK;
    const cut = tar.subarray(0, dataStart + 2500);
    assert.notEqual(cut.length % TAR_BLOCK, 0);
    assert.ok(cut.subarray(-2 * TAR_BLOCK).every((byte) => byte === 0), 'the premise: the last two blocks are zeros');
    const h = installerOver(t, zlib.zstdCompressSync(cut));
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'), refusedCutShort('runner-1.0.0.tar.zst'));
    assert.equal(fs.existsSync(h.marker), false);
    assert.equal(fs.existsSync(h.runDir), false);
    // The whole archive is fine.
    assert.equal((await installerOver(t, zlib.zstdCompressSync(tar)).installer.ensureRunnable('testrunner')).rebuilt, true);
});

test('a decoder that reports the cut (Node 24) and one that ends quietly (Node 25.8) both leave no marker, no copy and no staging', async (t) => {
    const tar = tarOf(t, { 'llama-server': {}, 'lib/libggml-cpu.so': { mode: 0o644, content: 'x'.repeat(3000) } });
    const bytes = zlib.zstdCompressSync(tar);
    // A decoder that hands on the first `keep` bytes it decoded, then either fails as Node 24 does or just ends.
    const cutting = (keep, fail) => () => {
        let seen = 0;
        const real = zlib.createZstdDecompress();
        const out = new Transform({
            transform(chunk, _encoding, done) { done(); },
        });
        real.on('data', (chunk) => { const room = keep - seen; if (room > 0) out.push(chunk.subarray(0, room)); seen += chunk.length; });
        real.on('error', (error) => out.destroy(error));
        real.on('end', () => (fail ? out.destroy(Object.assign(new Error('unexpected end of file'), { code: 'Z_BUF_ERROR', errno: -5 })) : out.push(null)));
        return Duplex.from({ writable: real, readable: out });
    };
    const calls = [];
    const run = async (options) => { calls.push(options.args); return tarRun(options); };
    // Node 24: a zlib error, coded, wrapped as install_failed.
    const reporting = installerOver(t, bytes, { run, zstdDecoder: cutting(1536, true) });
    await assert.rejects(() => reporting.installer.ensureRunnable('testrunner'),
        (error) => error.code === 'install_failed' && error.message === 'decompressing runner-1.0.0.tar.zst failed: Z_BUF_ERROR (unexpected end of file)');
    // Node 25.8: no error, and a .tar that stops on the end of its first member (no end-of-archive blocks), or empty, or in mid-member.
    for (const keep of [0, 700, memberEnd(tar, 'llama-server')]) {
        const quiet = installerOver(t, bytes, { run, zstdDecoder: cutting(keep, false) });
        await assert.rejects(() => quiet.installer.ensureRunnable('testrunner'),
            (error) => error.code === 'install_failed' && error.message === 'decompressing runner-1.0.0.tar.zst failed: it was cut short (no end-of-archive blocks)', `keep ${keep}`);
        assert.equal(fs.existsSync(quiet.marker), false, `keep ${keep}`);
        assert.equal(fs.existsSync(quiet.runDir), false, `keep ${keep}`);
        assert.equal(fs.existsSync(quiet.stageDir), false, `keep ${keep}`);
    }
    for (const h of [reporting]) {
        assert.equal(fs.existsSync(h.marker), false);
        assert.equal(fs.existsSync(h.runDir), false);
        assert.equal(fs.existsSync(h.stageDir), false);
    }
    assert.deepEqual(calls, [], 'tar never ran');
});

test('a tar cut exactly where a member ends, after the launch file, is refused: it has no end-of-archive blocks', async (t) => {
    // llama-server ends on a block boundary, and lib/libggml-cpu.so is what follows; the cut keeps only the first.
    const tar = tarOf(t, { 'llama-server': { content: '#!/bin/sh\n'.padEnd(1000, '#') }, 'lib/libggml-cpu.so': { mode: 0o644, content: 'x'.repeat(3000) } });
    const end = memberEnd(tar, 'llama-server');
    assert.equal(end % TAR_BLOCK, 0);
    // The premise: something follows, so what is cut off is real: the library, and the marker after it.
    assert.ok(tar.length - end > 3000 + TAR_BLOCK, 'the library member follows');
    assert.ok(tar.subarray(end, end + TAR_BLOCK).some((byte) => byte !== 0), 'and starts right there');
    const cut = tar.subarray(0, end);
    const calls = [];
    const h = installerOver(t, zlib.zstdCompressSync(cut), { run: async (options) => { calls.push(options.args); return tarRun(options); } });
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'),
        (error) => error.code === 'install_failed' && /^decompressing runner-1\.0\.0\.tar\.zst failed: it was cut short \(no end-of-archive blocks\)$/.test(error.message));
    assert.deepEqual(calls, []);
    assert.equal(fs.existsSync(h.marker), false);
    assert.equal(fs.existsSync(h.runDir), false);
    // One block of zeros after it is not the marker either; two are, whatever else follows.
    const oneBlock = installerOver(t, zlib.zstdCompressSync(Buffer.concat([cut, Buffer.alloc(TAR_BLOCK)])));
    await assert.rejects(() => oneBlock.installer.ensureRunnable('testrunner'), refusedCutShort('runner-1.0.0.tar.zst'));
    const twoBlocks = installerOver(t, zlib.zstdCompressSync(Buffer.concat([cut, Buffer.alloc(2 * TAR_BLOCK)])));
    assert.equal((await twoBlocks.installer.ensureRunnable('testrunner')).rebuilt, true);
    assert.ok(fs.existsSync(path.join(twoBlocks.runDir, 'llama-server')));
    // A whole archive written with the usual padding to 10 KiB installs, and so does an empty one.
    const padded = Buffer.concat([tar, Buffer.alloc(10240 - (tar.length % 10240))]);
    assert.equal((await installerOver(t, zlib.zstdCompressSync(padded)).installer.ensureRunnable('testrunner')).rebuilt, true);
    assert.equal((await installerOver(t, zlib.zstdCompressSync(Buffer.alloc(1024))).installer.ensureRunnable('testrunner').catch((error) => error.code)), 'install_failed', 'an empty tar has no launch file');
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
    const fine = installerOver(t, zlib.zstdCompressSync(Buffer.concat([random, random, random, Buffer.alloc(1024)])), { executable: null, run: async () => ({ code: 0, output: '' }) });
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

// ---------------------------------------------------------------- the entry digest

test('the entry digest covers check.executable only when it is named: image-lock digests are what they were, and a copy built before the proof is built again', async (t) => {
    // What the image lock's entries (vLLM, TabbyAPI, LM Studio) have always had: sha256 of the id, version, kind and files.
    // These literals were taken from 899f40b, before the launch file existed; an entry that names none must keep them, or every
    // runner installed on a GPU host would be rebuilt.
    const image = validateRunnerLock(IMAGE_LOCK).runners;
    assert.equal(image.vllm.digest, 'eeeaaf2f12ff58b4c99ae4b85b45ee7701c148089b53d6d8167ae8030aaf60a7');
    assert.equal(image.lmstudio.digest, '703b86cd1c04f1199f15b421b0d94ef53b4e2391cc75ff0266866cfc8bd3fd46');
    const old = (entry) => crypto.createHash('sha256').update(JSON.stringify({ id: entry.id, version: entry.version, kind: entry.kind, files: entry.files })).digest('hex');
    for (const entry of Object.values(image)) assert.equal(entry.digest, old(entry), entry.id);
    // A check that names no launch file, and every other field of it, leave the digest alone.
    const digestOf = (check) => validateRunnerLock(lockOf({ r: archiveEntry('r.tar.gz', check === undefined ? {} : { check }) })).runners.r.digest;
    const plain = digestOf(undefined);
    for (const check of [{}, { optionalLibraries: { 'libx.so*': 'never used' } }, { imports: [], distributions: { a: '1' } }]) assert.equal(digestOf(check), plain, JSON.stringify(check));
    // Naming one changes it, and so does naming another; the same one gives the same digest.
    const named = digestOf({ executable: 'bin/ollama' });
    assert.notEqual(named, plain);
    assert.notEqual(digestOf({ executable: 'llama-server' }), named);
    assert.equal(digestOf({ executable: 'bin/ollama', optionalLibraries: { 'libx.so*': 'never used' } }), named);
    const entry = validateRunnerLock(lockOf({ r: archiveEntry('r.tar.gz', { check: { executable: 'bin/ollama' } }) })).runners.r;
    assert.equal(entry.digest, crypto.createHash('sha256').update(JSON.stringify({ id: 'r', version: '1.0.0', kind: 'archive', files: entry.files, check: { executable: 'bin/ollama' } })).digest('hex'));
    // A copy marked ready before the lock named the launch file is not reused: the entry is installed again (its files are still
    // in the cache and are checked, not fetched), the copy is built again, and the proof now applies to it.
    const tar = zlib.zstdCompressSync(tarOf(t, { 'runner-1.0.0/not-the-binary': {} }, 'runner-1.0.0'));
    const before = installerOver(t, tar, { executable: null, strip: 1 });
    assert.equal((await before.installer.ensureRunnable('testrunner')).rebuilt, true);
    const oldMarker = JSON.parse(fs.readFileSync(before.marker, 'utf8')).digest;
    const cacheRoot = path.join(before.runRoot, '..', '..', 'data', 'runners');
    const lockNaming = validateRunnerLock(lockOf({ testrunner: archiveEntry(NAME, {
        files: [{ name: NAME, url: `https://github.com/example/r/releases/download/1.0.0/${NAME}`, size: tar.length, sha256: SHA(tar), strip: 1 }],
        check: { executable: 'llama-server' },
    }) }));
    assert.notEqual(lockNaming.runners.testrunner.digest, oldMarker);
    const after = createRunnerInstaller({
        lock: lockNaming, cacheRoot, runRoot: before.runRoot, run: tarRun,
        inspect: async ({ target }) => ({ state: 'complete', bytes: fs.statSync(target).size }),
        download: async () => ({ bytesTransferred: 0 }), statfs: async () => ({ bavail: 1e12, bsize: 1 }),
    });
    assert.equal((await after.describe('testrunner')).runnable, false, 'the old marker names another entry');
    await after.fetchAll(after.entryFor('testrunner'));
    await assert.rejects(() => after.ensureRunnable('testrunner'), missingExecutable('llama-server'));
    assert.equal(fs.existsSync(before.marker), false, 'the copy built without the proof is gone, not reused');
});

// ---------------------------------------------------------------- the lock file is a file of the repository

test('an agent lock that is a symbolic link is left out and logged, like any invalid agent lock', async (t) => {
    const dir = tempDir(t, 'lock-links');
    const good = { 'llama.cpp-cpu': archiveEntry('x.tar.gz', { check: { executable: 'llama-server' } }) };
    const real = path.join(dir, 'real.json');
    fs.writeFileSync(real, JSON.stringify(lockOf(good)));
    const image = path.join(dir, 'image.json');
    fs.writeFileSync(image, JSON.stringify(lockOf({ keep: archiveEntry('keep.tar.gz') })));
    // A regular file is taken.
    assert.deepEqual(Object.keys(loadRunnerLocks({ image, agent: real }).runners).sort(), ['keep', 'llama.cpp-cpu']);
    // A link to that very file is not, whatever it points at, and a dangling one is not either.
    const link = path.join(dir, 'linked.json');
    fs.symlinkSync(real, link);
    const dangling = path.join(dir, 'dangling.json');
    fs.symlinkSync(path.join(dir, 'nowhere.json'), dangling);
    for (const agent of [link, dangling]) {
        const locks = loadRunnerLocks({ image, agent });
        assert.deepEqual(Object.keys(locks.runners), ['keep'], agent);
        assert.equal(locks.ignored.length, 1);
        assert.equal(locks.ignored[0].file, agent);
        assert.equal(locks.ignored[0].reason, 'it is a symbolic link, not a file of the repository');
    }
    // A link in a directory above it is fine: the workspace mounts make /code/catalog one.
    const directoryLink = path.join(dir, 'catalog-link');
    fs.symlinkSync(dir, directoryLink);
    assert.deepEqual(Object.keys(loadRunnerLocks({ image, agent: path.join(directoryLink, 'real.json') }).runners).sort(), ['keep', 'llama.cpp-cpu']);
    // The image's lock is read as it always was, and a missing agent lock is still just empty.
    assert.deepEqual(loadRunnerLocks({ image, agent: path.join(dir, 'absent.json') }).ignored, []);
    // The controller says so in its log.
    const data = tempDir(t, 'lock-links-data');
    const controller = createController({
        dataDir: data, env: { PATH: '/usr/bin' }, seedCatalog: [], stateStore: createStateStore({ dataDir: data }), runnerLocks: loadRunnerLocks({ image, agent: link }),
        snapshot: async () => ({ gpu: { available: false, state: 'absent', reason: 'none' }, memory: {}, disk: {}, cpus: 1, cores: 1 }),
        detectRunner: () => ({ installed: false, version: null, reason: null }),
    });
    t.after(async () => { await controller.drain(); fs.rmSync(data, { recursive: true, force: true }); });
    const lines = (await controller.status()).logs.map((entry) => entry.line);
    assert.ok(lines.some((line) => line.includes(`the agent's lock ${link} is ignored: it is a symbolic link, not a file of the repository`)), lines.join('\n'));
    // The locks the repository ships are files.
    for (const arch of ['arm64', 'x64']) assert.equal(fs.lstatSync(agentRunnerLockFile(arch)).isFile(), true, arch);
});

// ---------------------------------------------------------------- smaller checks of the binary and of a failed build

test('--version names the pinned version as a whole token: not one inside a longer number or a longer dotted version', async (t) => {
    const says = (line, version) => probeExecutable({
        entry: validateRunnerLock(lockOf({ r: archiveEntry('x.tar.gz', { version, check: { executable: 'llama-server' } }) })).runners.r,
        runDir: copyWith(t, `#!/bin/sh\necho "${line}"\n`), ldd: noLibraries,
    }).then((result) => result.problems.length === 0);
    for (const [line, version] of [
        ['ollama version is 0.34.4', '0.34.4'], ['Warning: client version is 0.34.4', '0.34.4'], ['ollama 0.34.4', '0.34.4'], ['v0.34.4', '0.34.4'],
        ['0.34.4 (release)', '0.34.4'], ['version is 0.34.4.', '0.34.4'], ['(0.34.4)', '0.34.4'],
        ['version: 11295 (abcdef)', 'b11295'], ['build b11295 commit abcdef', 'b11295'], ['11295', 'b11295'],
    ]) assert.equal(await says(line, version), true, `${line} names ${version}`);
    for (const [line, version] of [
        ['ollama version is 0.34.40', '0.34.4'], ['0.34.4.1', '0.34.4'], ['version 10.34.4', '0.34.4'], ['version 0.34.45', '0.34.4'], ['x.0.34.4', '0.34.4'],
        ['version: 112950 (abcdef)', 'b11295'], ['version: 211295', 'b11295'], ['11295.1', 'b11295'], ['version: 11296', 'b11295'],
    ]) assert.equal(await says(line, version), false, `${line} does not name ${version}`);
});

test('an executable the owner cannot run is refused, even with an execute bit set', {
    // root may run a file that has any execute bit; there is nothing to refuse for it.
    skip: process.getuid?.() === 0 && 'root can execute any file with an execute bit',
}, async (t) => {
    const h = installerOver(t, zlib.zstdCompressSync(tarOf(t, { 'llama-server': { mode: 0o645 } })));
    await assert.rejects(() => h.installer.ensureRunnable('testrunner'), (error) => missingExecutable('llama-server')(error) && /it is not executable\.$/.test(error.message));
    assert.equal(fs.existsSync(h.marker), false);
    assert.equal(fs.existsSync(h.runDir), false);
    const fine = installerOver(t, zlib.zstdCompressSync(tarOf(t, { 'llama-server': { mode: 0o705 } })));
    assert.equal((await fine.installer.ensureRunnable('testrunner')).rebuilt, true);
});

test('a failed unpack or a refused size leaves no partial runnable copy and no scratch directory', async (t) => {
    const nothingLeft = (h, label) => {
        assert.equal(fs.existsSync(h.runDir), false, `${label}: the runnable copy`);
        assert.equal(fs.existsSync(h.tmpDir), false, `${label}: the scratch directory`);
        assert.equal(fs.existsSync(h.stageDir), false, `${label}: the staging`);
        assert.equal(fs.existsSync(h.marker), false, `${label}: the marker`);
        assert.deepEqual(fs.existsSync(path.dirname(h.runDir)) ? fs.readdirSync(path.dirname(h.runDir)) : [], [], `${label}: nothing under the runner's directory`);
    };
    // tar fails part-way: something is already in the copy when it does.
    const tar = zlib.zstdCompressSync(tarOf(t, { 'llama-server': {} }));
    const failing = installerOver(t, tar, { run: async (options) => {
        const into = options.args[options.args.indexOf('-C') + 1];
        fs.writeFileSync(path.join(into, 'half-unpacked'), 'partial');
        return { code: 2, signal: null, output: 'tar: Unexpected EOF in archive\n', aborted: false };
    } });
    await assert.rejects(() => failing.installer.ensureRunnable('testrunner'), (error) => error.code === 'install_failed' && /unpacking runner-1\.0\.0\.tar\.zst failed \(exit 2\): tar: Unexpected EOF in archive$/.test(error.message));
    nothingLeft(failing, 'tar failed');
    // The size cap.
    const bomb = installerOver(t, zlib.zstdCompressSync(Buffer.alloc(4 * 1024 * 1024)));
    await assert.rejects(() => bomb.installer.ensureRunnable('testrunner'), { code: 'install_failed' });
    nothingLeft(bomb, 'size cap');
    // A stop in the middle of the build.
    const stopped = new AbortController();
    const stopping = installerOver(t, tar, { run: async (options) => { stopped.abort(); return runTool({ ...options, signal: stopped.signal }); } });
    await assert.rejects(() => stopping.installer.ensureRunnable('testrunner', { signal: stopped.signal }), { code: 'ABORTED' });
    nothingLeft(stopping, 'stopped');
    // The next attempt starts clean and installs.
    const again = installerOver(t, tar, { run: tarRun });
    assert.equal((await again.installer.ensureRunnable('testrunner')).rebuilt, true);
});
