// The controller side of on-demand runners: Install and Uninstall go through
// the command queue, the licence acceptance is recorded with who and when,
// detection follows without an agent restart, a drain pauses an install and
// the next Install resumes it, and a Run rebuilds a missing runnable copy.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { downloadFile } from '../src/controller/downloader.mjs';
import { validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { RUNNERS } from '../src/runners/index.mjs';
import { deepFreeze, normalizeWith } from '../src/runners/params.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();
const SNAPSHOT = Object.freeze({
    gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
    memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
    disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
});

function tempDir(t, name) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-${name}-`));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

function makeArchive(t) {
    const dir = tempDir(t, 'src');
    fs.mkdirSync(path.join(dir, 'runner', 'bin'), { recursive: true });
    fs.writeFileSync(path.join(dir, 'runner', 'bin', 'serve'), '#!/bin/sh\n', { mode: 0o755 });
    fs.writeFileSync(path.join(dir, 'runner', 'padding.bin'), crypto.randomBytes(512 * 1024));
    execFileSync('tar', ['-czf', path.join(dir, 'runner.tar.gz'), '-C', dir, 'runner']);
    return fs.readFileSync(path.join(dir, 'runner.tar.gz'));
}

// `hold` keeps every response open after its first chunk until released.
async function serve(t, body, { hold = false } = {}) {
    const requests = [];
    let release = () => {};
    const released = new Promise((resolve) => { release = resolve; });
    const server = http.createServer(async (req, res) => {
        const range = /^bytes=(\d+)-$/.exec(req.headers.range ?? '');
        requests.push(req.headers.range ?? null);
        const start = range ? Number(range[1]) : 0;
        const headers = { 'Content-Length': body.length - start };
        if (range) headers['Content-Range'] = `bytes ${start}-${body.length - 1}/${body.length}`;
        res.writeHead(range ? 206 : 200, headers);
        if (hold && requests.length === 1) {
            res.write(body.subarray(start, start + 64 * 1024));
            await released;
            res.destroy();
            return;
        }
        res.end(body.subarray(start));
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => { release(); server.close(); });
    return { base: `http://127.0.0.1:${server.address().port}`, requests, release };
}

function lockFor(bytes, { requiresAcceptance = false } = {}) {
    return validateRunnerLock({
        schema: 'local-llm.runners-lock/v1',
        runners: {
            'fake-installable': {
                version: '1.0.0',
                kind: 'archive',
                licence: { name: 'AGPL-3.0', url: 'https://github.com/example/runner/blob/main/LICENSE',
                    source: 'https://github.com/example/runner', notice: 'Installed from upstream, not redistributed.', requiresAcceptance },
                files: [{ name: 'runner.tar.gz', url: 'https://github.com/example/runner/releases/download/v1/runner.tar.gz',
                    size: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') }],
            },
        },
    });
}

// An installed-on-demand GGUF runner: detection reads the installer, and
// the start pipeline runs from the runnable copy it is given.
function installableRunner(record) {
    const paramSchema = deepFreeze({ type: 'object', additionalProperties: false, properties: {
        contextTokens: { type: 'integer', minimum: 512, maximum: 65536, default: 4096, title: 'Context', description: 'Context tokens.' },
    } });
    return Object.freeze({
        id: 'fake-installable',
        displayName: 'Fake installable',
        weightFormat: 'gguf',
        pinnedVersion: '1.0.0',
        supported: true,
        installable: true,
        executable: null,
        port: 18998,
        apiKey: false,
        paramSchema,
        basicParams: Object.freeze([]),
        normalizeParams: (params, { model } = {}) => normalizeWith(paramSchema, 'fake-installable', params, model),
        describeContext: ({ contextTokens }) => ({ totalContext: contextTokens, perRequestContext: contextTokens, parallel: 1, kvUnified: false }),
        detect: async ({ installer }) => {
            const info = await installer.describe('fake-installable');
            return { installed: info.installed, version: info.installed ? info.version : null, reason: info.installed ? null : 'Not installed' };
        },
        async start(ctx) {
            record.runnerDir = ctx.runnerDir;
            const process = ctx.launch({ command: path.join(ctx.runnerDir, 'bin', 'serve'), args: [], env: {} });
            await ctx.waitForHttp(`http://127.0.0.1:${ctx.port}/health`, { process });
            return {};
        },
        chatModel: (deployment) => deployment.modelId,
        admit: () => ({ status: 'ok', reason: null, estimate: { isEstimate: true }, warnings: [] }),
    });
}

function harness(t, { lock, base }) {
    const dataDir = tempDir(t, 'data');
    const runRoot = path.join(tempDir(t, 'opt'), 'runners');
    const record = {};
    const runner = installableRunner(record);
    const started = [];
    const installer = createRunnerInstaller({
        lock,
        cacheRoot: path.join(dataDir, 'runners'),
        runRoot,
        statfs: async () => ({ bavail: 1e12, bsize: 1 }),
        download: (options) => downloadFile({
            ...options,
            fetchImpl: (url, init) => fetch(`${base}/${path.basename(new URL(url).pathname)}`, init),
            sleep: async () => {},
        }),
    });
    const stateStore = createStateStore({ dataDir });
    const make = () => createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: SEED,
        stateStore,
        runners: { ...RUNNERS, [runner.id]: runner },
        installer,
        snapshot: async () => structuredClone(SNAPSHOT),
        download: async ({ artifact }) => ({ status: 'complete', path: `/data/models/gguf/${artifact.file}`, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'complete', bytes: 1 }),
        startRunner({ command, args, env }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 9000 + started.length, command, args, env,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        detectRunner: (definition) => (definition.installable ? definition.detect({ installer })
            : { installed: definition.supported, version: definition.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
    });
    return { make, controller: make(), dataDir, runRoot, record, started, installer };
}

async function until(predicate, timeoutMs = 5000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 10));
    }
    assert.fail('condition not reached');
}

const installState = (controller) => controller.state.runnerInstalls?.['fake-installable'];

test('install refuses without the licence acceptance it needs, then records who accepted it and when', async (t) => {
    const bytes = makeArchive(t);
    const { base } = await serve(t, bytes);
    const h = harness(t, { lock: lockFor(bytes, { requiresAcceptance: true }), base });
    await assert.rejects(() => h.controller.installRunner({ runnerId: 'fake-installable', acceptedBy: 'admin@example.com' }),
        (error) => error.code === 'licence_required' && error.details.licence.name === 'AGPL-3.0');
    const accepted = await h.controller.installRunner({ runnerId: 'fake-installable', acceptLicence: true, acceptedBy: 'admin@example.com' });
    assert.equal(accepted.accepted, true);
    await until(() => installState(h.controller)?.phase === 'installed');
    const record = installState(h.controller);
    assert.equal(record.licence.acceptedBy, 'admin@example.com');
    assert.match(record.licence.acceptedAt, /^\d{4}-\d\d-\d\dT/);
    assert.ok(record.rebuild.seconds >= 0);
    // A runner that is not in the lock cannot be installed, whatever the input.
    await assert.rejects(() => h.controller.installRunner({ runnerId: 'llama.cpp' }), { code: 'not_installable' });
    await assert.rejects(() => h.controller.installRunner({ runnerId: 'fake-installable', url: 'https://evil.example.com/x' }),
        { code: 'invalid_request' });
});

test('detection follows an install and an uninstall without an agent restart', async (t) => {
    const bytes = makeArchive(t);
    const { base } = await serve(t, bytes);
    const h = harness(t, { lock: lockFor(bytes), base });
    const byId = async () => Object.fromEntries((await h.controller.overview()).runners.map((runner) => [runner.id, runner]));
    const before = (await byId())['fake-installable'];
    assert.equal(before.installed, false);
    assert.equal(before.install.version, '1.0.0');
    assert.equal(before.install.totalBytes, bytes.length);
    assert.equal(before.install.licence.name, 'AGPL-3.0');
    await h.controller.installRunner({ runnerId: 'fake-installable', acceptedBy: 'admin@example.com' });
    await until(() => installState(h.controller)?.phase === 'installed');
    const after = (await byId())['fake-installable'];
    assert.equal(after.installed, true);
    assert.equal(after.version, '1.0.0');
    assert.equal(after.install.runnable, true);
    const removed = await h.controller.uninstallRunner({ runnerId: 'fake-installable' });
    assert.ok(removed.freedBytes >= bytes.length);
    assert.equal((await byId())['fake-installable'].installed, false);
    assert.equal(installState(h.controller), undefined);
});

test('a drain pauses an install; after a restart, Install resumes it from the partial file', async (t) => {
    const bytes = makeArchive(t);
    const server = await serve(t, bytes, { hold: true });
    const h = harness(t, { lock: lockFor(bytes), base: server.base });
    await h.controller.installRunner({ runnerId: 'fake-installable', acceptedBy: 'admin@example.com' });
    await until(() => installState(h.controller)?.download?.bytes > 0);
    await h.controller.drain();
    server.release();
    assert.equal(installState(h.controller).phase, 'paused');
    // A new controller (the agent restarted) keeps the partial and resumes it.
    const again = h.make();
    assert.equal(again.state.runnerInstalls['fake-installable'].phase, 'paused');
    await again.installRunner({ runnerId: 'fake-installable', acceptedBy: 'admin@example.com' });
    await until(() => again.state.runnerInstalls['fake-installable']?.phase === 'installed');
    assert.match(server.requests.at(-1), /^bytes=\d+-$/);
});

test('a Run rebuilds a missing runnable copy before launch; Uninstall is refused while the runner runs', async (t) => {
    const bytes = makeArchive(t);
    const { base } = await serve(t, bytes);
    const h = harness(t, { lock: lockFor(bytes), base });
    await h.controller.installRunner({ runnerId: 'fake-installable', acceptedBy: 'admin@example.com' });
    await until(() => installState(h.controller)?.phase === 'installed');
    // The container was recreated: its filesystem has no runnable copy.
    fs.rmSync(h.runRoot, { recursive: true, force: true });
    await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'fake-installable', requestId: 'request-install-1' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.equal(h.record.runnerDir, path.join(h.runRoot, 'fake-installable', '1.0.0'));
    assert.ok(fs.existsSync(path.join(h.record.runnerDir, 'bin', 'serve')));
    assert.equal(h.started[0].command, path.join(h.record.runnerDir, 'bin', 'serve'));
    await assert.rejects(() => h.controller.uninstallRunner({ runnerId: 'fake-installable' }), { code: 'in_use' });
    await h.controller.stop();
    await h.controller.uninstallRunner({ runnerId: 'fake-installable' });
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'fake-installable', requestId: 'request-install-2' }),
        { code: 'runner_not_installed' });
});

test('a lock entry without a runner adapter cannot be installed or seen through the controller', async (t) => {
    const bytes = makeArchive(t);
    const { base } = await serve(t, bytes);
    const entry = (id) => ({
        version: '1.0.0', kind: 'archive',
        licence: { name: 'AGPL-3.0', url: 'https://github.com/example/runner/blob/main/LICENSE', requiresAcceptance: false },
        files: [{ name: 'runner.tar.gz', url: `https://github.com/example/${id}/releases/download/v1/runner.tar.gz`,
            size: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') }],
    });
    const lock = validateRunnerLock({ schema: 'local-llm.runners-lock/v1', runners: { 'fake-installable': entry('a'), orphan: entry('b') } });
    const h = harness(t, { lock, base });
    await assert.rejects(() => h.controller.installRunner({ runnerId: 'orphan', acceptedBy: 'admin@example.com' }), { code: 'not_installable' });
    await assert.rejects(() => h.controller.uninstallRunner({ runnerId: 'orphan' }), { code: 'not_installable' });
    assert.equal((await h.controller.overview()).runners.some((runner) => runner.id === 'orphan'), false);
    assert.equal(h.controller.state.runnerInstalls?.orphan, undefined);
});

test('a Run on a runner that is being installed is refused as busy', async (t) => {
    const bytes = makeArchive(t);
    const server = await serve(t, bytes, { hold: true });
    const h = harness(t, { lock: lockFor(bytes), base: server.base });
    await h.controller.installRunner({ runnerId: 'fake-installable', acceptedBy: 'admin@example.com' });
    await until(() => installState(h.controller)?.download?.bytes > 0);
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'fake-installable', requestId: 'request-install-3' }),
        { code: 'busy' });
    await h.controller.drain();
    server.release();
});
