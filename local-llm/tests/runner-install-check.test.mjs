// The CI install check's library-link rule (runners plan §5.2, "Library
// links"): a library missing under ldd fails the check unless the driver
// grant supplies it (libcuda.so.1), another wheel in the same environment
// ships it (it is loaded at run time, as torch loads its CUDA libraries), or
// the lock lists it as optional for a feature this runner does not use, with
// the reason.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { checkDataFiles, classifyMissingLibraries } from '../tools/runner_install_check.mjs';
import { validateRunnerLock } from '../src/controller/runnerLock.mjs';

const TOOL = new URL('../tools/runner_install_check.mjs', import.meta.url).pathname;

test('missing libraries are explained by the driver, the environment or the lock, or they fail the check', () => {
    const result = classifyMissingLibraries({
        missing: {
            'venv/torchcodec/libtorchcodec_core7.so': ['libtorch.so', 'libavutil.so.58', 'libcuda.so.1'],
            'venv/nvshmem/nvshmem_bootstrap_mpi.so.3': ['libmpi.so.40'],
            'venv/foo/_ext.so': ['libssl.so.3'],
        },
        provided: new Set(['libtorch.so', 'libc10.so']),
        optional: { 'libav*.so.*': 'torchcodec video decoding, unused for text generation', 'libmpi.so.*': 'NVSHMEM multi-node bootstrap' },
    });
    assert.deepEqual(result.unresolved, { 'venv/foo/_ext.so': ['libssl.so.3'] });
    assert.deepEqual(result.optional, { 'libavutil.so.58': 'torchcodec video decoding, unused for text generation', 'libmpi.so.40': 'NVSHMEM multi-node bootstrap' });
    assert.deepEqual(result.providedByEnvironment, ['libtorch.so']);
    // A pattern matches a whole name, not a substring.
    const exact = classifyMissingLibraries({ missing: { 'x.so': ['libmpich.so.12'] }, provided: new Set(), optional: { 'libmpi.so.*': 'r' } });
    assert.deepEqual(exact.unresolved, { 'x.so': ['libmpich.so.12'] });
});

test('the lock names optional libraries with a reason; anything else is refused', () => {
    const file = { name: 'x-1-py3-none-any.whl', url: 'https://files.pythonhosted.org/packages/x-1-py3-none-any.whl', size: 1, sha256: 'a'.repeat(64) };
    const lock = (check) => ({ schema: 'local-llm.runners-lock/v1',
        runners: { r: { version: '1', kind: 'python', licence: { name: 'MIT', url: 'https://example.org' }, files: [file], check } } });
    const ok = validateRunnerLock(lock({ optionalLibraries: { 'libmpi.so.*': 'NVSHMEM multi-node bootstrap' } }));
    assert.deepEqual(ok.runners.r.check.optionalLibraries, { 'libmpi.so.*': 'NVSHMEM multi-node bootstrap' });
    assert.deepEqual(validateRunnerLock(lock({})).runners.r.check.optionalLibraries, {});
    for (const bad of [{ '*': 'everything' }, { 'lib/../x.so': 'r' }, { 'libx.so': '' }, { 'libx.so': 3 }, ['libx.so']]) {
        assert.throws(() => validateRunnerLock(lock({ optionalLibraries: bad })), /optionalLibraries/);
    }
});

test('every pinned data file must sit in its directory of the runnable copy with the pinned bytes', (t) => {
    const runDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-datacheck-'));
    t.after(() => fs.rmSync(runDir, { recursive: true, force: true }));
    const vocab = crypto.randomBytes(2048);
    const sha256 = crypto.createHash('sha256').update(vocab).digest('hex');
    const entry = { files: [
        { name: 'x-1-py3-none-any.whl', sha256: 'a'.repeat(64) },
        { name: 'o200k_base.tiktoken', sha256, into: 'tiktoken' },
    ] };
    assert.deepEqual(checkDataFiles(entry, runDir), { placed: { 'tiktoken/o200k_base.tiktoken': 'missing' }, problems: ['tiktoken/o200k_base.tiktoken is missing'] });
    fs.mkdirSync(path.join(runDir, 'tiktoken'));
    fs.writeFileSync(path.join(runDir, 'tiktoken', 'o200k_base.tiktoken'), Buffer.concat([vocab, Buffer.from('x')]));
    assert.deepEqual(checkDataFiles(entry, runDir).problems, ['tiktoken/o200k_base.tiktoken differs from the lock']);
    fs.writeFileSync(path.join(runDir, 'tiktoken', 'o200k_base.tiktoken'), vocab);
    assert.deepEqual(checkDataFiles(entry, runDir), { placed: { 'tiktoken/o200k_base.tiktoken': 'ok' }, problems: [] });
});

test('the check prints its whole report before it exits, however long', async () => {
    // An unknown runner id fails fast; the report must still arrive complete on a pipe.
    const child = spawn(process.execPath, [TOOL, 'no-such-runner', '--lock', '/nonexistent/lock.json'], { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    child.stdout.on('data', (chunk) => { out += chunk; });
    const code = await new Promise((resolve) => child.on('exit', resolve));
    assert.equal(code, 1);
    assert.equal(JSON.parse(out).error, 'not_installable');
});
