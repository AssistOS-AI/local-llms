// The runner framework: every runner is an adapter that brings its own
// port, key, start-up pipeline, chat model name, admission policy and form
// metadata. A new runner is an adapter plus data; the controller has no
// per-runner branches.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admissionResult } from '../src/controller/admission.mjs';
import { WEIGHT_FORMATS, loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { RUNNERS, defaultPorts, runnerSummaries } from '../src/runners/index.mjs';
import { deepFreeze, normalizeWith } from '../src/runners/params.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();
const KEY = 'f'.repeat(43);

const SNAPSHOT = Object.freeze({
    gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
    memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
    disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
});

// A third GGUF runner the controller has never heard of.
function fakeRunner(record) {
    const paramSchema = deepFreeze({
        type: 'object',
        additionalProperties: false,
        properties: {
            contextTokens: { type: 'integer', minimum: 512, maximum: 65536, default: 4096, title: 'Context', description: 'Context tokens.' },
        },
    });
    return Object.freeze({
        id: 'fake-gguf',
        displayName: 'Fake GGUF runner',
        weightFormat: 'gguf',
        pinnedVersion: 'v1',
        supported: true,
        executable: '/opt/fake/fake-server',
        port: 18999,
        apiKey: true,
        paramSchema,
        basicParams: Object.freeze(['contextTokens']),
        normalizeParams: (params, { model } = {}) => normalizeWith(paramSchema, 'fake-gguf', params, model),
        describeContext: ({ contextTokens }) => ({ totalContext: contextTokens, perRequestContext: contextTokens, parallel: 1, kvUnified: false }),
        detect: () => ({ installed: true, version: 'v1', reason: null }),
        async start(ctx) {
            record.ctx = { port: ctx.port, apiKey: ctx.apiKey, weights: ctx.weights, params: ctx.params };
            const process = ctx.launch({ command: '/opt/fake/fake-server', args: ['--model', ctx.weights.path, '--port', String(ctx.port)], env: { FAKE: '1' } });
            await ctx.waitForHttp(`http://127.0.0.1:${ctx.port}/ready`, { process, headers: { authorization: `Bearer ${ctx.apiKey}` } });
            return { fake: { served: true } };
        },
        chatModel: (deployment) => `fake/${deployment.modelId}`,
        admit({ model, source, params }) {
            record.admitted = { model: model.id, file: source.file, params };
            return admissionResult('ok', null, { gpuBytes: 1 * GIB, basis: 'fake policy' });
        },
        parseReport: () => ({ device: 'CUDA0 (fake)', totalMiB: 1 }),
    });
}

function harness(t, { runners, detectRunner } = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-framework-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const started = [];
    const probes = [];
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        seedCatalog: SEED,
        stateStore: createStateStore({ dataDir }),
        runners,
        snapshot: async () => structuredClone(SNAPSHOT),
        download: async ({ artifact }) => ({ status: 'complete', path: `/data/models/gguf/${artifact.file}`, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'complete', bytes: 1 }),
        remove: async () => 1,
        startRunner({ command, args, env }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 7000 + started.length, command, args, env,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async (url, options = {}) => {
            probes.push([String(url), options.headers?.authorization ?? null]);
            return { ok: true, status: 200, json: async () => ({}) };
        },
        apiKeyFactory: () => KEY,
        detectRunner: detectRunner || ((runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null })),
        pollMs: 2,
        stopGraceMs: 50,
    });
    return { controller, started, probes };
}

async function until(predicate, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

test('a third runner registered as an adapter runs through the controller without controller changes', async (t) => {
    const record = {};
    const fake = fakeRunner(record);
    const h = harness(t, { runners: { ...RUNNERS, [fake.id]: fake } });

    const overview = await h.controller.overview();
    assert.ok(overview.runners.some((runner) => runner.id === 'fake-gguf' && runner.displayName === 'Fake GGUF runner'));
    const entry = overview.models.find((model) => model.id === 'gpt-oss-20b').runners['fake-gguf'];
    assert.equal(entry.admission.status, 'ok');
    assert.equal(entry.admission.estimate.basis, 'fake policy');
    // It reads the same GGUF as llama.cpp: one artifact, one download state.
    assert.deepEqual(entry.download, overview.models[0].runners['llama.cpp'].download);
    assert.equal(record.admitted.file, 'gpt-oss-20b-MXFP4.gguf');

    const accepted = await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'fake-gguf', requestId: 'request-fake-01', params: { contextTokens: 8192 } });
    assert.equal(accepted.deployment.phase, 'downloading');
    await until(() => h.controller.state.deployment?.phase === 'ready');
    assert.deepEqual(record.ctx, {
        port: 18999, apiKey: KEY, weights: { path: '/data/models/gguf/gpt-oss-20b-MXFP4.gguf' }, params: { contextTokens: 8192 },
    });
    const [process] = h.started;
    assert.equal(process.command, '/opt/fake/fake-server');
    assert.equal(process.env.FAKE, '1');
    assert.equal(process.env.HF_TOKEN, undefined);
    assert.deepEqual(h.probes, [['http://127.0.0.1:18999/ready', `Bearer ${KEY}`]]);
    assert.deepEqual(h.controller.state.deployment.runner.fake, { served: true });
    assert.equal(h.controller.state.deployment.runner.port, 18999);
    assert.deepEqual(h.controller.chatTarget(), {
        runnerId: 'fake-gguf', modelId: 'gpt-oss-20b', baseUrl: 'http://127.0.0.1:18999',
        apiKey: KEY, model: 'fake/gpt-oss-20b', requestOptions: null,
    });
    const status = await h.controller.status();
    assert.equal(status.runnerReport.device, 'CUDA0 (fake)');

    // The GGUF is in use by the fake runner, so no runner of that format may delete it.
    await assert.rejects(() => h.controller.deleteWeights({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' }), { code: 'in_use' });
    await assert.rejects(() => h.controller.deleteWeights({ modelId: 'gpt-oss-20b', format: 'gguf' }), { code: 'in_use' });
    await h.controller.stop();
    assert.equal(process.stopped ?? !process.running, true);
    assert.equal((await h.controller.deleteWeights({ modelId: 'gpt-oss-20b', format: 'gguf' })).freedBytes, 1);
    await assert.rejects(() => h.controller.deleteWeights({ modelId: 'gpt-oss-20b', format: 'bogus' }), { code: 'invalid_request' });
    await assert.rejects(() => h.controller.deleteWeights({ modelId: 'gpt-oss-20b', format: 'toString' }), { code: 'invalid_request' });
});

test('a runner that dies right after its readiness probes ends in error, never in ready', async (t) => {
    const record = {};
    const base = fakeRunner(record);
    const dying = Object.freeze({
        ...base,
        async start(ctx) {
            const process = ctx.launch({ command: '/opt/fake/fake-server', args: [], env: {} });
            await ctx.waitForHttp(`http://127.0.0.1:${ctx.port}/ready`, { process });
            await process.stop();
            return {};
        },
    });
    const h = harness(t, { runners: { ...RUNNERS, [dying.id]: dying } });
    await h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'fake-gguf', requestId: 'request-dying-1' });
    await until(() => ['ready', 'error'].includes(h.controller.state.deployment?.phase));
    assert.equal(h.controller.state.deployment.phase, 'error');
    await assert.rejects(async () => h.controller.chatTarget(), { code: 'not_ready' });
});

test('ports come from the adapters, one per supported runner, all distinct', () => {
    assert.deepEqual(defaultPorts(), { 'llama.cpp': 18080, 'ik_llama.cpp': 18081, ollama: 18434 });
    assert.throws(() => defaultPorts({ a: { id: 'a', supported: true, port: 18080 }, b: { id: 'b', supported: true, port: 18080 } }),
        /port 18080/);
});

test('the registry has llama.cpp, ik_llama.cpp, Ollama and the vLLM placeholder; LM Studio is gone', () => {
    assert.deepEqual(Object.keys(RUNNERS), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm']);
    assert.equal(fs.existsSync(new URL('../src/runners/lmStudio.mjs', import.meta.url)), false);
    const summaries = runnerSummaries();
    assert.deepEqual(summaries.find((summary) => summary.id === 'llama.cpp').basicParams, ['ctxSize', 'nCpuMoe']);
    assert.deepEqual(summaries.find((summary) => summary.id === 'llama.cpp').moeParams, ['nCpuMoe']);
    assert.deepEqual(summaries.find((summary) => summary.id === 'ollama').basicParams, ['numCtx']);
});

test('every runnerId enum in mcp-config.json lists exactly the registry, and the format enum the weight formats', () => {
    const config = JSON.parse(fs.readFileSync(new URL('../mcp-config.json', import.meta.url), 'utf8'));
    const enums = [];
    const visit = (node) => {
        if (!node || typeof node !== 'object') return;
        if (node.runnerId?.enum) enums.push(node.runnerId.enum);
        for (const value of Object.values(node)) visit(value);
    };
    visit(config);
    assert.equal(enums.length, 3);
    for (const values of enums) assert.deepEqual(values, Object.keys(RUNNERS));
    const remove = config.tools.find((tool) => tool.name === 'local_llm_weights_delete');
    assert.deepEqual(remove.inputSchema.properties.format.enum, Object.keys(WEIGHT_FORMATS));
});

test('runner detection is cached, and re-detection picks up an install or removal without a restart', async (t) => {
    let calls = 0;
    let installed = false;
    const h = harness(t, {
        detectRunner: (runner) => {
            calls += 1;
            return runner.id === 'llama.cpp' ? { installed, version: installed ? 'b11125' : null, reason: installed ? null : 'missing' }
                : { installed: runner.supported, version: runner.pinnedVersion, reason: null };
        },
    });
    const first = await h.controller.overview();
    assert.equal(first.runners[0].installed, false);
    const afterFirst = calls;
    await h.controller.overview();
    assert.equal(calls, afterFirst, 'cached between overviews');
    installed = true;
    h.controller.redetectRunners();
    const second = await h.controller.overview();
    assert.equal(second.runners[0].installed, true);
});

test('an unsupported runner is refused with the reason its adapter gives, before anything is downloaded', async (t) => {
    const h = harness(t);
    const overview = await h.controller.overview();
    const vllm = overview.runners.find((runner) => runner.id === 'vllm');
    assert.equal(vllm.supported, false);
    // vLLM reads Hugging Face snapshots; the seed has none, so it has no entry for gpt-oss-20b.
    assert.equal(overview.models[0].runners.vllm, undefined);
    await assert.rejects(() => h.controller.run({ modelId: 'gpt-oss-20b', runnerId: 'vllm', requestId: 'request-vllm-01' }),
        { code: 'runner_unsupported', message: /Not supported/ });
});
