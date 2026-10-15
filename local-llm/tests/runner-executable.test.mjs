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

import { loadRunnerLock, loadRunnerLocks } from 'pworker/local-llm/internal/controller/runnerLock.mjs';
import { createStateStore } from 'pworker/local-llm/internal/controller/stateStore.mjs';

import { agentRunnerLockFile } from '../src/agentHost.mjs';
import { RUNNERS, createAgentController as createController } from '../src/engineOptions.mjs';

import { IMAGE_LOCK } from './overview-scenarios.mjs';

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

