import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { callController } from '../src/controlSocket.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { startRunnerProcess } from '../src/controller/runnerProcess.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { RUNNERS } from '../src/runners/index.mjs';

const MAIN = new URL('../src/main.mjs', import.meta.url).pathname;
const MIB = 1024 * 1024;
const PAYLOAD = crypto.randomBytes(4 * MIB);
const SHA256 = crypto.createHash('sha256').update(PAYLOAD).digest('hex');
const COMMIT = 'e'.repeat(40);

function fixture(t, { crashAgentServer = false } = {}) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-main-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const agentDir = path.join(root, 'Agent', 'server');
    fs.mkdirSync(agentDir, { recursive: true });
    // A stand-in AgentServer: alive until SIGTERM, then exits 0; or crashes.
    // For the test only, it records the control channel main handed it.
    const agentEnv = path.join(root, 'agent-env.json');
    fs.writeFileSync(path.join(agentDir, 'AgentServer.mjs'), crashAgentServer
        ? 'setTimeout(() => process.exit(3), 300);\n'
        : `import fs from 'node:fs';
fs.writeFileSync(${JSON.stringify(agentEnv)}, JSON.stringify({
    socketPath: process.env.LOCAL_LLM_SOCKET, token: process.env.LOCAL_LLM_CONTROL_TOKEN }));
process.on('SIGTERM', () => process.exit(0)); setInterval(() => {}, 1000);\n`);
    const smi = path.join(root, 'nvidia-smi');
    fs.writeFileSync(smi, '#!/bin/sh\ncase "$1" in --query-gpu=*) echo "Test GPU, 6144, 13, 6000, 595.91.07";; esac\n', { mode: 0o755 });
    const catalog = path.join(root, 'catalog.json');
    fs.writeFileSync(catalog, JSON.stringify({
        schema: 'local-llm.catalog/v2',
        models: [{
            id: 'tiny',
            architecture: 'dense',
            sources: { gguf: { type: 'huggingface', repo: 'test/tiny', file: 'tiny.gguf', revision: COMMIT,
                commit: COMMIT, size: PAYLOAD.length, sha256: SHA256 } },
            memory: { layers: 4, nonExpertBytes: PAYLOAD.length, expertBytesPerLayer: 0, kvBytesPerToken: 1024 },
        }],
    }));
    const socket = `@local-llm-test-${crypto.randomBytes(8).toString('hex')}`;
    return { root, dataDir: path.join(root, 'data'), agentLib: path.join(root, 'Agent'), smi, catalog, socket, agentEnv };
}

// Serves the artifact slowly (64 KiB every 40 ms) so a download is in flight.
async function slowServer(t) {
    const server = http.createServer((req, res) => {
        const start = Number(/bytes=(\d+)-/.exec(req.headers.range || '')?.[1] || 0);
        res.writeHead(start ? 206 : 200, {
            'content-length': PAYLOAD.length - start,
            ...(start ? { 'content-range': `bytes ${start}-${PAYLOAD.length - 1}/${PAYLOAD.length}` } : {}),
        });
        let offset = start;
        const timer = setInterval(() => {
            if (offset >= PAYLOAD.length) { clearInterval(timer); res.end(); return; }
            res.write(PAYLOAD.subarray(offset, offset + 64 * 1024));
            offset += 64 * 1024;
        }, 40);
        req.on('close', () => clearInterval(timer));
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => server.close());
    return `http://127.0.0.1:${server.address().port}`;
}

function startMain(f, extraEnv = {}) {
    const child = spawn(process.execPath, [MAIN], {
        env: {
            PATH: process.env.PATH,
            LOCAL_LLM_DATA_DIR: f.dataDir,
            LOCAL_LLM_SOCKET: f.socket,
            LOCAL_LLM_CATALOG_FILE: f.catalog,
            LOCAL_LLM_NVIDIA_SMI: f.smi,
            PLOINKY_AGENT_LIB_DIR: f.agentLib,
            ...extraEnv,
        },
        stdio: ['ignore', 'pipe', 'pipe'],
    });
    let output = '';
    child.stdout.on('data', (chunk) => { output += chunk; });
    child.stderr.on('data', (chunk) => { output += chunk; });
    const exited = new Promise((resolve) => child.once('exit', (code, signal) => resolve({ code, signal, output })));
    return { child, exited, output: () => output };
}

async function until(predicate, timeoutMs = 10_000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try { if (await predicate()) return; } catch {}
        await new Promise((resolve) => setTimeout(resolve, 50));
    }
    assert.fail('condition not reached');
}

// The socket and token main handed its AgentServer, once the controller is up.
async function channel(f) {
    await until(() => fs.existsSync(f.agentEnv));
    return JSON.parse(fs.readFileSync(f.agentEnv, 'utf8'));
}

test('main hands a fresh per-start token and its abstract socket only to AgentServer, and refuses calls without it', async (t) => {
    const f = fixture(t);
    const preset = 'p'.repeat(43);
    const main = startMain(f, { LOCAL_LLM_CONTROL_TOKEN: preset });
    const opts = await channel(f);
    assert.equal(opts.socketPath, f.socket);
    assert.match(opts.token, /^[A-Za-z0-9_-]{43}$/);
    // A token in main's own environment is ignored, never reused.
    assert.notEqual(opts.token, preset);
    const listed = fs.readFileSync('/proc/net/unix', 'utf8').split('\n').map((line) => line.trim().split(/\s+/)[7]);
    assert.ok(listed.includes(f.socket));
    assert.equal((await callController('status', {}, opts)).phase, 'idle');
    for (const token of [undefined, preset]) {
        await assert.rejects(() => callController('status', {}, { socketPath: opts.socketPath, token }), { code: 'unauthorized' });
    }
    main.child.kill('SIGTERM');
    const result = await main.exited;
    assert.equal(result.code, 0, result.output);
    assert.ok(!result.output.includes(opts.token), 'the token never reaches the log');
    assert.match(result.output, new RegExp(`controller ready on ${f.socket}`));
});

test('SIGTERM during a download drains and exits 0, keeping the partial and its identity', async (t) => {
    const f = fixture(t);
    const base = await slowServer(t);
    const main = startMain(f, { LOCAL_LLM_HF_BASE_URL: base });
    let opts = await channel(f);
    await callController('run', { requestId: 'drain-000001', modelId: 'tiny', runnerId: 'llama.cpp' }, opts);
    await until(async () => (await callController('status', {}, opts)).deployment.download.bytes > 256 * 1024);
    const started = Date.now();
    main.child.kill('SIGTERM');
    const result = await main.exited;
    assert.equal(result.code, 0, result.output);
    assert.ok(Date.now() - started < 30_000);
    const dir = path.join(f.dataDir, 'models', 'gguf', 'test', 'tiny', COMMIT);
    const partial = fs.statSync(path.join(dir, 'tiny.gguf.partial'));
    assert.ok(partial.size > 0 && partial.size < PAYLOAD.length);
    assert.equal(JSON.parse(fs.readFileSync(path.join(dir, 'tiny.gguf.partial.json'), 'utf8')).sha256, SHA256);
    const state = JSON.parse(fs.readFileSync(path.join(f.dataDir, 'state', 'controller.json'), 'utf8'));
    assert.equal(state.deployment.phase, 'paused');
    assert.match(state.deployment.pausedReason, /restarted during the download/);

    // After the restart nothing resumes by itself; the next Run resumes with Range.
    fs.rmSync(f.agentEnv);
    const again = startMain(f, { LOCAL_LLM_HF_BASE_URL: base });
    opts = await channel(f);
    assert.equal((await callController('status', {}, opts)).phase, 'paused');
    await new Promise((resolve) => setTimeout(resolve, 300));
    assert.equal(fs.statSync(path.join(dir, 'tiny.gguf.partial')).size, partial.size);
    again.child.kill('SIGTERM');
    assert.equal((await again.exited).code, 0);
});

test('an AgentServer crash makes the controller exit non-zero', async (t) => {
    const f = fixture(t, { crashAgentServer: true });
    const main = startMain(f);
    const result = await main.exited;
    assert.notEqual(result.code, 0);
    assert.match(result.output, /AgentServer exited unexpectedly/);
});

test('a drain stops and reaps a real runner process', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-reap-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const fakeRunner = {
        ...RUNNERS['llama.cpp'],
        buildLaunch: () => ({ command: process.execPath, args: ['-e', 'setInterval(() => {}, 1000)'], env: {} }),
    };
    let pid = null;
    const controller = createController({
        dataDir,
        env: { PATH: process.env.PATH },
        stateStore: createStateStore({ dataDir }),
        runners: { ...RUNNERS, 'llama.cpp': fakeRunner },
        snapshot: async () => ({
            gpu: { available: true, totalBytes: 6144 * MIB, freeBytes: 6000 * MIB, processes: [] },
            memory: { totalBytes: 32 * 1024 * MIB, availableBytes: 24 * 1024 * MIB },
            disk: { freeBytes: 300 * 1024 * MIB },
        }),
        download: async () => ({ status: 'complete', path: '/data/models/gpt.gguf', bytesTransferred: 0 }),
        inspect: async () => ({ state: 'complete', bytes: 1 }),
        startRunner: (options) => {
            const handle = startRunnerProcess(options);
            pid = handle.pid;
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200 }),
        detectRunner: () => ({ installed: true, version: 'b11125', reason: null }),
        pollMs: 5,
        stopGraceMs: 2000,
    });
    await controller.run({ requestId: 'reap-0000001', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' });
    await until(() => controller.state.deployment.phase === 'ready');
    assert.doesNotThrow(() => process.kill(pid, 0));
    await controller.drain();
    assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' });
    assert.equal(controller.state.deployment.phase, 'idle');
});
