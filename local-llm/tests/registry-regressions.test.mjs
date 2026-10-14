// Reproductions of three defects of the Hugging Face path that the network guard fixes (DS002). Each fails at
// 7fdbff5 on an assertion, never on a missing module: the test transport comes from a guarded dynamic import that is
// undefined there, so the controller falls back to its old fetch. Exactly three top-level tests, no hooks, and every
// server and controller is closed in `finally`.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createController } from '../src/controller/deployments.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';

const MIB = 1024 * 1024;
const REPO = 'acme/models';
const COMMIT = 'c'.repeat(40);
const ARM64_CONTRACT = Object.freeze({ architecture: 'arm64', llama_cpp: 'b11159', gpu_compute_capabilities: '12.1' });
const CONFIG = Buffer.from('{"num_hidden_layers":2}\n');
const gitOid = (bytes) => crypto.createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const lfs = (file, size, char) => ({ type: 'file', path: file, size, oid: 'f'.repeat(40), lfs: { oid: char.repeat(64), size, pointerSize: 134 } });

const netFixture = async () => (await import('./net-fixture.mjs').catch(() => ({})));

// A server that records every request and answers with `handler`; it never throws into the test.
async function listen(handler) {
    const requests = [];
    const server = http.createServer((req, res) => {
        requests.push({ method: req.method, url: req.url });
        try {
            handler(req, res, new URL(req.url, 'http://placeholder.invalid'));
        } catch {
            res.writeHead(500);
            res.end();
        }
    });
    server.on('clientError', (error, socket) => socket.destroy());
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = server.address().port;
    const close = async () => {
        try {
            server.closeAllConnections();
            await new Promise((resolve) => server.close(() => resolve()));
        } catch {
            // Already closed.
        }
    };
    return { origin: `http://127.0.0.1:${port}`, port, requests, close };
}

function sendJson(res, status, body, headers = {}) {
    const text = JSON.stringify(body);
    res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text), ...headers });
    res.end(text);
}

// A controller on the cpu profile with its hardware and runners injected; only the Hugging Face side is real.
function controllerFor(dir, { hfBaseUrl, registryTransport }) {
    return createController({
        dataDir: dir,
        env: { PATH: '/usr/bin' },
        seedCatalog: [],
        hfBaseUrl,
        ...(registryTransport ? { registryTransport } : {}),
        stateStore: createStateStore({ dataDir: dir }),
        snapshot: async () => ({
            gpu: { available: false, state: 'absent', reason: 'No GPU is attached to this agent' },
            memory: { totalBytes: 8 * 1024 * MIB, availableBytes: 4 * 1024 * MIB },
            cgroupMemory: null, disk: { freeBytes: 100_000 * MIB, totalBytes: 400_000 * MIB }, cpus: 4, cores: 4,
        }),
        download: async () => assert.fail('nothing is downloaded here'),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        imageContract: ARM64_CONTRACT,
        fileExists: (candidate) => candidate === '/opt/llama.cpp/llama-server',
        sharedModelsRoot: null,
        readMemory: () => ({ totalBytes: 8 * 1024 * MIB, availableBytes: 4 * 1024 * MIB }),
        readPressure: () => 0,
        dropCache: () => true,
    });
}

async function settle(controller, dir, servers) {
    try {
        await controller?.drain();
    } catch {
        // The test's own assertions decide.
    }
    for (const server of servers) await server.close();
    fs.rmSync(dir, { recursive: true, force: true });
}

const outcomeOf = (promise) => promise.then(() => null, (error) => error);

test('R1 pin follows a cross-origin metadata redirect', { timeout: 60_000 }, async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-r1-'));
    // Origin B would answer the revision with a plausible commit.
    const b = await listen((req, res, url) => (url.pathname.endsWith('/revision/main') ? sendJson(res, 200, { sha: COMMIT }) : sendJson(res, 404, {})));
    const a = await listen((req, res, url) => {
        if (url.pathname === `/api/models/${REPO}/revision/main`) {
            res.writeHead(302, { Location: `${b.origin}/api/models/${REPO}/revision/main`, 'Content-Length': 0 });
            res.end();
            return;
        }
        if (url.pathname === `/api/models/${REPO}/tree/${COMMIT}`) {
            sendJson(res, 200, [lfs('m-Q4_K_M.gguf', 4 * MIB, 'a')]);
            return;
        }
        sendJson(res, 404, {});
    });
    let controller;
    try {
        const registryTransport = (await netFixture())?.createTestTransport?.();
        controller = controllerFor(dir, { hfBaseUrl: a.origin, registryTransport });
        const outcome = await outcomeOf(controller.addModel({
            id: 'r1-model',
            sources: { gguf: { type: 'huggingface', repo: REPO, file: 'm-Q4_K_M.gguf', revision: 'main' } },
        }));
        assert.ok(a.requests.length >= 1, 'R1: origin A saw at least one request (the pin ran)');
        assert.equal(outcome?.code, 'RESOLVE_FAILED', 'R1: Add rejects with RESOLVE_FAILED');
        assert.equal(outcome?.details?.reason, 'redirect', 'R1: Add rejects with reason redirect');
        assert.equal(b.requests.length, 0, 'R1: origin B saw 0 requests');
    } finally {
        await settle(controller, dir, [a, b]);
    }
});

test('R2 a dropped next page pins an incomplete snapshot', { timeout: 60_000 }, async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-r2-'));
    const c = await listen((req, res) => sendJson(res, 200, [lfs('model-00002-of-00002.safetensors', MIB, 'b')]));
    const a = await listen((req, res, url) => {
        if (url.pathname === `/api/models/${REPO}/revision/main`) {
            sendJson(res, 200, { sha: COMMIT });
            return;
        }
        if (url.pathname === `/api/models/${REPO}/tree/${COMMIT}`) {
            // Page 1 alone is a valid snapshot; page 2 lives on origin C.
            sendJson(res, 200, [
                { type: 'file', path: 'config.json', size: CONFIG.length, oid: gitOid(CONFIG) },
                lfs('model-00001-of-00002.safetensors', MIB, 'a'),
            ], { Link: `<${c.origin}/api/models/${REPO}/tree/${COMMIT}?cursor=1>; rel="next"` });
            return;
        }
        sendJson(res, 404, {});
    });
    let controller;
    try {
        const registryTransport = (await netFixture())?.createTestTransport?.();
        controller = controllerFor(dir, { hfBaseUrl: a.origin, registryTransport });
        const outcome = await outcomeOf(controller.addModel({ id: 'r2-model', sources: { hf: { type: 'hf-snapshot', repo: REPO, revision: 'main' } } }));
        assert.ok(a.requests.some((request) => request.url.includes('/tree/')), 'R2: origin A served tree page 1');
        assert.equal(outcome?.code, 'RESOLVE_FAILED', 'R2: Add rejects with RESOLVE_FAILED');
        assert.equal(outcome?.details?.reason, 'listing-incomplete', 'R2: Add rejects with reason listing-incomplete');
        assert.deepEqual(controller.state.registry, [], 'R2: state.registry is unchanged');
        assert.equal(c.requests.length, 0, 'R2: origin C saw 0 requests');
    } finally {
        await settle(controller, dir, [a, c]);
    }
});

test('R3 a host resolving to loopback is contacted', { timeout: 60_000 }, async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-r3-'));
    const hf = await listen((req, res, url) => {
        if (url.pathname === `/api/models/${REPO}`) {
            sendJson(res, 200, { id: REPO, gated: false });
            return;
        }
        if (url.pathname === `/api/models/${REPO}/revision/main`) {
            sendJson(res, 200, { sha: COMMIT });
            return;
        }
        if (url.pathname === `/api/models/${REPO}/tree/${COMMIT}`) {
            sendJson(res, 200, [lfs('m-Q4_K_M.gguf', 4 * MIB, 'a')]);
            return;
        }
        sendJson(res, 404, {});
    });
    let strict;
    let allowed;
    try {
        const fixture = await netFixture();
        const hfBaseUrl = `http://localhost:${hf.port}`;
        // Production classification; plain http is allowed only so the fake can answer.
        strict = controllerFor(dir, { hfBaseUrl, registryTransport: fixture?.createTestTransport?.({ allowHttp: true, extraAllowed: null }) });
        const outcome = await outcomeOf(strict.lookupModel({ repo: REPO }));
        assert.equal(outcome?.code, 'address_refused', 'R3: the lookup is refused with address_refused');
        assert.equal(hf.requests.length, 0, 'R3: the server saw 0 requests');
        // Positive control: the same lookup with loopback allowed reaches the server.
        allowed = controllerFor(dir, { hfBaseUrl, registryTransport: fixture?.createTestTransport?.({ allowHttp: true, extraAllowed: fixture?.LOOPBACK }) });
        const looked = await allowed.lookupModel({ repo: REPO });
        assert.equal(looked.commit, COMMIT, 'R3: with loopback allowed the lookup reads the commit');
        assert.ok(hf.requests.length >= 1, 'R3: with loopback allowed the server is reached');
    } finally {
        try {
            await allowed?.drain();
        } catch {
            // The test's own assertions decide.
        }
        await settle(strict, dir, [hf]);
    }
});
