// Characterization of the llama.cpp and Ollama runners as seen from the
// controller: the exact process each start launches, the probes it makes in
// order, the chat target it hands out and the overview entry it reports.
// The runner framework must not change any of it.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { manifestPath } from '../src/controller/ollamaStore.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();
const KEY = 'k'.repeat(43);

const SNAPSHOT = Object.freeze({
    gpu: {
        available: true, name: 'Test GPU', driverVersion: '595.91.07',
        totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [],
    },
    memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
    disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
    cpus: 20,
});

function harness(t, { fetchImpl, download } = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-characterize-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const started = [];
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: SEED,
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => structuredClone(SNAPSHOT),
        download: download || (async () => ({ status: 'complete', path: '/data/models/gguf/gpt.gguf', bytesTransferred: 0 })),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        startRunner({ command, args, env }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 5000 + started.length, command, args, env,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl,
        apiKeyFactory: () => KEY,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
    });
    return { controller, started, dataDir };
}

async function until(predicate, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

function writeOllamaModel(dataDir, tag) {
    const modelsDir = path.join(dataDir, 'models', 'ollama');
    const hex = 'b'.repeat(64);
    fs.mkdirSync(path.join(modelsDir, 'blobs'), { recursive: true });
    fs.writeFileSync(path.join(modelsDir, 'blobs', `sha256-${hex}`), Buffer.alloc(16));
    const target = manifestPath(modelsDir, tag);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, JSON.stringify({
        schemaVersion: 2, layers: [{ digest: `sha256:${hex}`, size: 16, mediaType: 'application/vnd.ollama.image.model' }],
    }));
}

test('llama.cpp: the exact launch, probes and chat target', async (t) => {
    const probes = [];
    const h = harness(t, {
        fetchImpl: async (url, options = {}) => {
            probes.push([String(url), options.headers?.authorization ?? null]);
            return { ok: true, status: 200, json: async () => ({}) };
        },
    });
    await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp', requestId: 'request-0001' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const [process] = h.started;
    assert.equal(process.command, '/opt/llama.cpp/llama-server');
    assert.deepEqual(process.args, [
        '-m', '/data/models/gguf/gpt.gguf', '--host', '127.0.0.1', '--port', '18080', '--api-key', KEY,
        '--no-webui', '-lv', '4', '--alias', 'gpt-oss-20b', '--ctx-size', '16384', '--n-gpu-layers', '99',
        '--n-cpu-moe', '17', '--flash-attn', 'auto', '--cache-type-k', 'f16', '--cache-type-v', 'f16', '-np', '1',
        '--batch-size', '256', '--ubatch-size', '256', '--chat-template-kwargs', '{"reasoning_effort":"low"}', '--jinja',
    ]);
    assert.deepEqual(process.env, {
        PATH: '/usr/bin', HOME: path.join(h.dataDir, 'home'), LANG: 'C.UTF-8', LD_LIBRARY_PATH: '/usr/local/nvidia/lib64',
    });
    assert.deepEqual(probes, [
        ['http://127.0.0.1:18080/health', null],
        ['http://127.0.0.1:18080/v1/models', `Bearer ${KEY}`],
    ]);
    assert.deepEqual(h.controller.state.deployment.runner, {
        pid: 5000, port: 18080, startedAt: h.controller.state.deployment.runner.startedAt,
    });
    assert.deepEqual(h.controller.chatTarget(), {
        runnerId: 'llama.cpp', modelId: 'gpt-oss-20b', baseUrl: 'http://127.0.0.1:18080',
        apiKey: KEY, model: 'gpt-oss-20b', requestOptions: null,
    });
    await h.controller.stop();
});

test('Ollama: the exact launch, pull, load, probes and chat target', async (t) => {
    const calls = [];
    const encoder = new TextEncoder();
    let h;
    h = harness(t, {
        fetchImpl: async (url, options = {}) => {
            const pathname = new URL(String(url)).pathname;
            calls.push([String(url), options.method || 'GET', options.body ? JSON.parse(options.body) : null]);
            if (pathname === '/api/pull') {
                writeOllamaModel(h.dataDir, 'gpt-oss:20b');
                return { ok: true, status: 200, body: [encoder.encode(`${JSON.stringify({ status: 'success' })}\n`)] };
            }
            if (pathname === '/api/ps') {
                return { ok: true, status: 200, json: async () => ({ models: [{ name: 'gpt-oss:20b', size: 16, size_vram: 8, context_length: 4096 }] }) };
            }
            return { ok: true, status: 200, json: async () => ({}) };
        },
    });
    // The seed pins a manifest digest this fake store does not produce, so
    // characterize with an unpinned user entry for the same tag.
    await h.controller.addModel({ id: 'olla-user', sources: { ollama: { type: 'ollama', tag: 'gpt-oss:20b' } } });
    const accepted = await h.controller.run({ modelId: 'olla-user', runnerId: 'ollama', requestId: 'request-0001' });
    assert.equal(accepted.deployment.phase, 'starting');
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const [process] = h.started;
    assert.equal(process.command, '/opt/ollama/bin/ollama');
    assert.deepEqual(process.args, ['serve']);
    assert.deepEqual(process.env, {
        PATH: '/usr/bin', HOME: `${h.dataDir}/home`, LANG: 'C.UTF-8',
        OLLAMA_MODELS: `${h.dataDir}/models/ollama`, OLLAMA_HOST: '127.0.0.1:18434', OLLAMA_NUM_PARALLEL: '1',
        OLLAMA_MAX_LOADED_MODELS: '1', OLLAMA_KV_CACHE_TYPE: 'f16', OLLAMA_CONTEXT_LENGTH: '4096',
        OLLAMA_KEEP_ALIVE: '30m', OLLAMA_NO_CLOUD: '1', LD_LIBRARY_PATH: '/usr/local/nvidia/lib64',
    });
    assert.deepEqual(calls, [
        ['http://127.0.0.1:18434/api/version', 'GET', null],
        ['http://127.0.0.1:18434/api/pull', 'POST', { model: 'gpt-oss:20b', stream: true }],
        ['http://127.0.0.1:18434/api/generate', 'POST', {
            model: 'gpt-oss:20b', prompt: '', stream: false, options: { num_ctx: 4096 }, keep_alive: '30m',
        }],
        ['http://127.0.0.1:18434/api/ps', 'GET', null],
    ]);
    const deployment = h.controller.state.deployment;
    assert.deepEqual(deployment.runner.ollama, { sizeBytes: 16, sizeVramBytes: 8, contextLength: 4096 });
    assert.deepEqual(deployment.resolved.blobs, [`sha256:${'b'.repeat(64)}`]);
    assert.deepEqual(h.controller.chatTarget(), {
        runnerId: 'ollama', modelId: 'olla-user', baseUrl: 'http://127.0.0.1:18434', apiKey: null,
        model: 'gpt-oss:20b', requestOptions: { options: { num_ctx: 4096 }, keep_alive: '30m' },
    });
    await h.controller.stop();
});

test('overview: the llama.cpp and Ollama runner entries and the seed model entries', async (t) => {
    const h = harness(t, { fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }) });
    const overview = await h.controller.overview();
    const byId = Object.fromEntries(overview.runners.map((runner) => [runner.id, runner]));
    for (const [id, displayName, weightFormat, pinnedVersion] of [
        ['llama.cpp', 'llama.cpp', 'gguf', 'b11125'], ['ollama', 'Ollama', 'ollama', '0.34.3'],
    ]) {
        // The runner framework adds the form metadata (basicParams, moeParams);
        // every field the entry had before is unchanged.
        const { paramSchema, basicParams: _basic, moeParams: _moe, ...rest } = byId[id];
        assert.deepEqual(rest, { id, displayName, weightFormat, pinnedVersion, supported: true, installed: true, version: pinnedVersion, reason: null }, id);
        assert.equal(typeof paramSchema.properties, 'object', id);
    }
    const gpt = overview.models.find((model) => model.id === 'gpt-oss-20b');
    const llama = gpt.runners['llama.cpp'];
    assert.equal(llama.size, 12109566624);
    assert.deepEqual(llama.download, { state: 'absent', bytes: 0, total: 12109566624 });
    assert.deepEqual(llama.params, {
        ctxSize: 16384, nGpuLayers: 99, nCpuMoe: 17, flashAttn: 'auto', cacheTypeK: 'f16', cacheTypeV: 'f16',
        threads: null, parallel: 1, batchSize: 256, ubatchSize: 256, mlock: false, noMmap: false,
        chatTemplateKwargs: { reasoning_effort: 'low' },
    });
    assert.deepEqual(llama.context, { totalContext: 16384, perRequestContext: 16384, parallel: 1, kvUnified: false });
    assert.equal(llama.admission.status, 'ok');
    assert.deepEqual(llama.admission.estimate, {
        weightsBytes: 12109566624, gpuWeightsBytes: 4271164000, cpuWeightsBytes: 7838402624, kvBytes: 421527552,
        computeBytes: 184507433, gpuBytes: 5086914185, ramBytes: 8643708992, basis: 'measured memory profile', isEstimate: true,
    });
    const ollama = gpt.runners.ollama;
    assert.equal(ollama.size, 13793441244);
    assert.deepEqual(ollama.download, { state: 'absent', bytes: 0, total: 13793441244 });
    assert.deepEqual(ollama.params, { numCtx: 4096, numGpu: null, numThread: null, flashAttention: null, kvCacheType: 'f16', keepAlive: '30m' });
    assert.equal(ollama.admission.status, 'ok');
    assert.equal(ollama.admission.reason, null);
    assert.ok(ollama.admission.estimate.ramBytes > 8 * GIB);
});
