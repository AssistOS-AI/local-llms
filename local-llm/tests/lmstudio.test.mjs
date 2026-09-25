// LM Studio (runners plan §5.7, Phase R7; LOCAL_LLM_LMSTUDIO_HANDOFF.md): its
// headless daemon, llmster, installed on demand after an admin accepts LM
// Studio's Terms, for internal use only, behind an operator switch. It runs
// from the verified runnable copy with a container-local home, reads the
// agent's own GGUF files by symbolic link, and loads through LM Studio's MIT
// SDK with every memory flag set and checked in the engine's command line.
import assert from 'node:assert/strict';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { RUNNERS, defaultPorts } from '../src/runners/index.mjs';
import {
    LMSTUDIO_SWITCH,
    createOutputFilter,
    effectiveFlags,
    engineFlagProblems,
    expectedEngineFlags,
    lmStudioRunner,
    loadConfig,
    strayProcesses,
} from '../src/runners/lmStudio.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();
const GPT = SEED.find((model) => model.id === 'gpt-oss-20b');
const RUN_DIR = '/opt/runners/lmstudio/0.0.25-1';

function snapshot({ freeMiB = 6000, availableGiB = 24 } = {}) {
    return {
        gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: (6144 - freeMiB) * MIB, freeBytes: freeMiB * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: availableGiB * GIB },
        disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
    };
}

test('LM Studio is a GGUF runner on its own loopback port, with no per-start key (none exists headless)', () => {
    assert.equal(RUNNERS.lmstudio, lmStudioRunner);
    assert.equal(lmStudioRunner.id, 'lmstudio');
    assert.equal(lmStudioRunner.weightFormat, 'gguf');
    assert.equal(lmStudioRunner.pinnedVersion, '0.0.25-1');
    assert.equal(lmStudioRunner.supported, true);
    assert.equal(lmStudioRunner.apiKey, false);
    assert.equal(defaultPorts().lmstudio, 18084);
    assert.equal(new Set(Object.values(defaultPorts())).size, Object.keys(defaultPorts()).length);
    assert.match(lmStudioRunner.displayName, /LM Studio/);
});

test('LM Studio is off unless the operator sets the switch to internal-use', () => {
    assert.equal(LMSTUDIO_SWITCH, 'LOCAL_LLM_LMSTUDIO');
    for (const env of [{}, { LOCAL_LLM_LMSTUDIO: '' }, { LOCAL_LLM_LMSTUDIO: 'yes' }, { LOCAL_LLM_LMSTUDIO: 'true' }, { LOCAL_LLM_LMSTUDIO: 'INTERNAL-USE' }]) {
        const gate = lmStudioRunner.enabled(env);
        assert.equal(gate.enabled, false, JSON.stringify(env));
        assert.match(gate.reason, /not enabled on this deployment \(internal use only\)/i);
        assert.match(gate.reason, /LOCAL_LLM_LMSTUDIO/);
    }
    assert.deepEqual(lmStudioRunner.enabled({ LOCAL_LLM_LMSTUDIO: 'internal-use' }), { enabled: true, reason: null });
});

test('the load config sets every memory flag, and the exact layer counts and threads go through the argument override', () => {
    const values = lmStudioRunner.normalizeParams({}, { model: GPT });
    assert.deepEqual(values, {
        ctxSize: 16384, nGpuLayers: 99, nCpuMoe: 17, flashAttn: 'on', cacheTypeK: 'f16', cacheTypeV: 'f16',
        threads: null, parallel: 1, batchSize: 256, ubatchSize: 256, noMmap: false,
    });
    const config = loadConfig(values, { threads: 12 });
    assert.deepEqual(config, {
        contextLength: 16384,
        gpu: { ratio: 1, numCpuExpertLayersRatio: 0 },
        maxParallelPredictions: 1,
        useUnifiedKvCache: false,
        evalBatchSize: 256,
        physicalBatchSize: 256,
        flashAttention: true,
        llamaKCacheQuantizationType: false,
        llamaVCacheQuantizationType: false,
        keepModelInMemory: false,
        tryMmap: true,
        contextCheckpoints: 0,
        llamaCppArgumentsOverride: {
            enabled: true,
            disabledParameters: [],
            overrideParameters: [
                { key: '--n-gpu-layers', value: '99' },
                { key: '--n-cpu-moe', value: '17' },
                { key: '--threads', value: '12' },
            ],
            excludeAllConfig: false,
        },
    });
    const other = loadConfig({ ...values, parallel: 2, cacheTypeK: 'q8_0', cacheTypeV: 'q4_0', flashAttn: 'off', noMmap: true, nCpuMoe: 0 }, { threads: 4 });
    assert.equal(other.useUnifiedKvCache, true);
    assert.equal(other.llamaKCacheQuantizationType, 'q8_0');
    assert.equal(other.llamaVCacheQuantizationType, 'q4_0');
    assert.equal(other.flashAttention, false);
    assert.equal(other.tryMmap, false);
    assert.deepEqual(other.llamaCppArgumentsOverride.overrideParameters.map((p) => p.key), ['--n-gpu-layers', '--n-cpu-moe', '--threads']);
    // Parameters LM Studio cannot honour are refused, like any unknown field.
    assert.throws(() => lmStudioRunner.normalizeParams({ mlock: true }, { model: GPT }), /unknown parameter/);
    assert.throws(() => lmStudioRunner.normalizeParams({ flashAttn: 'auto' }, { model: GPT }), /must be one of on, off/);
    assert.throws(() => lmStudioRunner.normalizeParams({ ubatchSize: 512, batchSize: 256 }, { model: GPT }), /ubatchSize/);
});

test('the engine command line is checked flag by flag, the last value of a repeated flag counting', () => {
    // As observed on 2026-09-25: LM Studio's own flags, then the override appended.
    const argv = ['/opt/runners/lmstudio/0.0.25-1/home/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.41.0/llama-server',
        '--model', '/x.gguf', '--host', '127.0.0.1', '--port', '43351', '--api-key', 'secret', '--ctx-size', '16384',
        '--n-gpu-layers', '999999', '--n-cpu-moe', '0', '--ctx-checkpoints', '0', '--batch-size', '256', '--ubatch-size', '256',
        '--threads', '7', '--parallel', '1', '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--flash-attn', 'on', '--kv-offload',
        '--no-kv-unified', '--load-mode', 'mmap', '--n-gpu-layers', '99', '--n-cpu-moe', '17', '--threads', '12'];
    const flags = effectiveFlags(argv);
    assert.equal(flags['--n-cpu-moe'], '17');
    assert.equal(flags['--threads'], '12');
    assert.equal(flags['--no-kv-unified'], true);
    const values = lmStudioRunner.normalizeParams({}, { model: GPT });
    const expected = expectedEngineFlags(values, { threads: 12 });
    assert.deepEqual(engineFlagProblems(argv, expected), []);
    // Anything that differs from what admission sized is reported.
    const drifted = argv.map((arg) => (arg === '17' ? '16' : arg));
    assert.deepEqual(engineFlagProblems(drifted, expected), ['--n-cpu-moe is 16, expected 17']);
    const unified = argv.map((arg) => (arg === '--no-kv-unified' ? '--kv-unified' : arg));
    assert.match(engineFlagProblems(unified, expected).join(' '), /kv-unified/);
    const mlock = argv.map((arg) => (arg === 'mmap' ? 'mmap+mlock' : arg));
    assert.deepEqual(engineFlagProblems(mlock, expected), ['--load-mode is mmap+mlock, expected mmap']);
    // The pinned engine only (L7): an engine from another directory is refused.
    const updated = [argv[0].replace('2.41.0', '2.44.0'), ...argv.slice(1)];
    assert.match(engineFlagProblems(updated, expected).join(' '), /not the pinned engine/);
});

test('leftover LM Studio processes are found by executable path, never by name', () => {
    const procs = [
        { pid: 10, exe: '/opt/runners/lmstudio/0.0.25-1/llmster' },
        { pid: 11, exe: '/opt/runners/lmstudio/0.0.25-1/home/.lmstudio/.internal/utils/node' },
        { pid: 12, exe: '/opt/runners/lmstudio/0.0.25-1/home/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.41.0/llama-server' },
        // llama.cpp's and ik_llama.cpp's servers have the same name and are never touched.
        { pid: 13, exe: '/opt/llama.cpp/llama-server' },
        { pid: 14, exe: '/opt/ik_llama.cpp/llama-server' },
        { pid: 15, exe: '/usr/local/bin/node' },
        { pid: 16, exe: '/opt/runners/lmstudio/0.0.25-10/llmster' },
        { pid: 17, exe: null },
    ];
    assert.deepEqual(strayProcesses(procs, RUN_DIR), [10, 11, 12]);
    assert.deepEqual(strayProcesses(procs, RUN_DIR, { keep: [10] }), [11, 12]);
});

test('llmster output keeps lifecycle lines and never passes request or response bodies to the runner log', () => {
    // Lines as llmster printed them on 2026-09-25 (L0 evidence), with a marker prompt.
    const lines = [
        '[APIServerProvider] API Server started on port: 41343',
        'llmster started successfully! ',
        '[2026-09-25 11:13:43][DEBUG] Received request: POST to /v1/chat/completions with body {',
        '  "model": "qwen3-0.6b",',
        '  "messages": [',
        '    {',
        '      "role": "user",',
        '      "content": "/no_think The code word is ZEBRAMARKER1790334823. Reply with exactly one word: PONG"',
        '    }',
        '  ],',
        '  "max_tokens": 64',
        '}',
        '[2026-09-25 11:13:43][INFO][qwen3-0.6b] Running chat completion on conversation with 1 messages.',
        // Observed live in L3: the input of an embeddings request, on one bracketed line.
        '[2026-09-25 12:30:36][INFO] Received request to embed: ZEBRAL31790339435C',
        '[2026-09-25 12:30:00][INFO] Server started.',
        '[2026-09-25 12:30:00][INFO] Just-in-time model loading active.',
        '[2026-09-25 12:30:00][INFO][LM STUDIO SERVER]    ->  POST http://localhost:18084/v1/chat/completions',
        '[2026-09-25 12:30:00][INFO][LMSAuthenticator][Client=lms-cli][Endpoint=listLoaded] Listing loaded models',
        '[2026-09-25 11:13:43][INFO][qwen3-0.6b] Generated prediction:  {',
        '  "id": "chatcmpl-werjlvazzxsod4s2xriqh",',
        '        "content": "PONG ZEBRAMARKER1790334823",',
        '}',
        '[2026-09-25 11:13:43][INFO] Returning {',
        '  "data": [ { "id": "ZEBRAMARKER-model" } ]',
        '}',
        '[2026-09-25 11:13:44][ERROR] Something failed: {"messages":[{"content":"ZEBRAMARKER inline"}]}',
        '[LLMProxyObject] Engine protocol runtime exited unexpectedly. exitCode=1, signal=null',
        'App is quitting',
    ];
    const filter = createOutputFilter();
    const kept = lines.map((line) => filter(line)).filter((line) => line !== null);
    assert.equal(kept.some((line) => line.includes('ZEBRAMARKER')), false, kept.join('\n'));
    assert.equal(kept.some((line) => line.includes('PONG')), false);
    for (const expected of ['API Server started on port: 41343', 'llmster started successfully!', 'Server started.',
        'Just-in-time model loading active.', 'POST http://localhost:18084/v1/chat/completions', '[Endpoint=listLoaded] Listing loaded models',
        'Engine protocol runtime exited unexpectedly', 'App is quitting']) {
        assert.ok(kept.some((line) => line.includes(expected)), expected);
    }
    // Server-log lines about requests are dropped whatever their shape ...
    assert.equal(kept.some((line) => /Received request|Running chat completion|Returning/.test(line)), false, kept.join('\n'));
    // ... and an error keeps only what precedes its first colon.
    assert.ok(kept.includes('[2026-09-25 11:13:44][ERROR] Something failed: …'), kept.join('\n'));
});

test('admission is the llama-server memory model plus LM Studio\'s own RAM, and the engine flags match what it sized', () => {
    const params = lmStudioRunner.normalizeParams({}, { model: GPT });
    const lms = admit({ runner: lmStudioRunner, model: GPT, source: GPT.sources.gguf, params, snapshot: snapshot() });
    const llama = admit({ runner: RUNNERS['llama.cpp'], model: GPT, source: GPT.sources.gguf,
        params: RUNNERS['llama.cpp'].normalizeParams({}, { model: GPT }), snapshot: snapshot() });
    assert.equal(lms.status, 'ok');
    assert.equal(lms.estimate.gpuBytes, llama.estimate.gpuBytes);
    assert.equal(lms.estimate.ramBytes, llama.estimate.ramBytes + 768 * MIB);
    assert.match(lms.estimate.basis, /LM Studio/);
    // Too little RAM now counts LM Studio's daemon as well.
    const tight = admit({ runner: lmStudioRunner, model: GPT, source: GPT.sources.gguf, params,
        snapshot: snapshot({ availableGiB: (llama.estimate.ramBytes + 1 * GIB + 256 * MIB) / GIB }) });
    assert.equal(tight.status, 'insufficient-now');
});

// ---- the helper, against a stand-in for the SDK (same surface, no LM Studio)

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { serverLogsDir } from '../src/runners/lmStudio.mjs';

const HELPER = new URL('../src/runners/lmStudioLoad.mjs', import.meta.url).pathname;

function tempDir(t, name) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-lms-${name}-`));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

// A fake @lmstudio/sdk whose state lives in a JSON file, so each helper run sees the last one's.
function fakeSdk(t, state) {
    const sdk = tempDir(t, 'sdk');
    const pkg = path.join(sdk, 'node_modules', '@lmstudio', 'sdk');
    fs.mkdirSync(pkg, { recursive: true });
    fs.writeFileSync(path.join(sdk, 'state.json'), JSON.stringify(state));
    fs.writeFileSync(path.join(pkg, 'package.json'), JSON.stringify({ name: '@lmstudio/sdk', version: '2.0.0', main: 'index.js' }));
    fs.writeFileSync(path.join(pkg, 'index.js'), `
const fs = require('node:fs');
const file = ${JSON.stringify(path.join(sdk, 'state.json'))};
const read = () => JSON.parse(fs.readFileSync(file, 'utf8'));
const write = (s) => fs.writeFileSync(file, JSON.stringify(s));
const handle = (entry) => ({ identifier: entry.identifier, async unload() { const s = read(); s.loaded = s.loaded.filter((e) => e.identifier !== entry.identifier); s.unloads.push(entry.identifier); write(s); } });
class LMStudioClient {
  constructor({ baseUrl }) { const s = read(); s.baseUrls.push(baseUrl); write(s); if (s.failConnects-- > 0) { write(s); this.down = true; } }
  get system() { const self = this; return {
    async getLMStudioVersion() { if (self.down) throw new Error('ECONNREFUSED'); return { version: '0.0.25', build: 1 }; },
    async listDownloadedModels() { return read().models; },
    async startHttpServer(opts) { const s = read(); s.server = opts; write(s); },
  }; }
  get llm() { return {
    async listLoaded() { return read().loaded.filter((e) => e.type === 'llm').map(handle); },
    async load(key, opts) { const s = read(); s.loads.push({ key, ...opts }); s.loaded.push({ type: 'llm', identifier: opts.identifier }); write(s); },
  }; }
  get embedding() { return { async listLoaded() { return read().loaded.filter((e) => e.type === 'embedding').map(handle); } }; }
}
module.exports = { LMStudioClient };
`);
    return { sdk, state: () => JSON.parse(fs.readFileSync(path.join(sdk, 'state.json'), 'utf8')) };
}

function helper(step, options) {
    const result = spawnSync(process.execPath, [HELPER, step, JSON.stringify(options)], { encoding: 'utf8', timeout: 20000 });
    return { code: result.status, out: result.stdout.trim(), err: result.stderr.trim() };
}

// LM Studio lets an SDK (guest) client load a model but not start the server
// (observed 2026-09-25: "does not have the required permission:
// httpServer.start"), so the helper only loads; `lms` does the rest.
test('the helper loads the file we imported by LM Studio\'s own key, with the given config, and nothing else', (t) => {
    const fake = fakeSdk(t, {
        baseUrls: [], failConnects: 2, unloads: [], loads: [], server: null,
        models: [
            { modelKey: 'qwen3-0.6b', path: 'someone/else/Qwen3-0.6B-Q8_0.gguf' },
            { modelKey: 'qwen3-0.6b@q8', path: 'local-llm/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf' },
        ],
        loaded: [{ type: 'embedding', identifier: 'text-embedding-nomic-embed-text-v1.5' }],
    });
    const config = { contextLength: 4096 };
    const loaded = helper('load', { daemon: 'ws://127.0.0.1:41343', sdk: fake.sdk, path: 'local-llm/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf', identifier: 'qwen3-0.6b', config });
    assert.equal(loaded.code, 0, loaded.err);
    assert.deepEqual(JSON.parse(loaded.out), { modelKey: 'qwen3-0.6b@q8', identifier: 'qwen3-0.6b' });
    const state = fake.state();
    // It retried until llmster answered, always at the daemon's address.
    assert.ok(state.baseUrls.length >= 3 && state.baseUrls.every((url) => url === 'ws://127.0.0.1:41343'));
    assert.deepEqual(state.loads, [{ key: 'qwen3-0.6b@q8', identifier: 'qwen3-0.6b', config, verbose: false }]);
    // The server and the unloads are left to the privileged lms CLI.
    assert.deepEqual(state.unloads, []);
    assert.equal(state.server, null);
    // A file LM Studio did not index is not guessed at.
    const other = fakeSdk(t, { baseUrls: [], failConnects: 0, unloads: [], loads: [], server: null, models: [{ modelKey: 'x', path: 'someone/else/x.gguf' }], loaded: [] });
    assert.equal(helper('guard', { sdk: other.sdk }).code, 1, 'guard is gone from the helper');

    // An unknown step fails with a reason.
    const nope = helper('nope', { sdk: fake.sdk });
    assert.equal(nope.code, 1);
    assert.match(nope.err, /unknown step nope/);
});

// ---- start() and afterExit, with a fake /proc

function fakeProc(t, entries) {
    const dir = tempDir(t, 'proc');
    for (const { pid, exe, argv } of entries) {
        fs.mkdirSync(path.join(dir, String(pid)));
        if (exe) fs.symlinkSync(exe, path.join(dir, String(pid), 'exe'));
        if (argv) fs.writeFileSync(path.join(dir, String(pid), 'cmdline'), `${argv.join('\0')}\0`);
    }
    return dir;
}

function startContext(t, runDir, { port = 18084 } = {}) {
    const calls = { launched: [], exec: [], waited: [] };
    return {
        calls,
        ctx: {
            runnerDir: runDir,
            model: GPT,
            params: {},
            port,
            weights: { path: '/data/models/gguf/ggml-org/gpt-oss-20b-GGUF/ef9b12f/gpt-oss-20b-MXFP4.gguf' },
            launch(spec) { calls.launched.push(spec); return { running: true }; },
            async exec(step) {
                calls.exec.push(step);
                if (step.args[1] === 'load') return { modelKey: 'gpt-oss-20b', identifier: 'gpt-oss-20b' };
                if (step.args[0] === 'ps') return calls.listed ?? [{ identifier: 'gpt-oss-20b', type: 'llm' }];
                return {};
            },
            async waitForHttp(url) { calls.waited.push(url); },
        },
    };
}

const ENGINE = `${RUN_DIR}/home/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.41.0/llama-server`;
const ENGINE_ARGS = ['--ctx-size', '16384', '--n-gpu-layers', '999999', '--n-cpu-moe', '0', '--ctx-checkpoints', '0', '--batch-size', '256',
    '--ubatch-size', '256', '--threads', '7', '--parallel', '1', '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--flash-attn', 'on',
    '--no-kv-unified', '--load-mode', 'mmap', '--n-gpu-layers', '99', '--n-cpu-moe', '17', '--threads', '12'];

test('start runs llmster from its copy, imports by symbolic link, loads through the SDK and checks the engine it got', async (t) => {
    const root = tempDir(t, 'run');
    const runDir = path.join(root, 'lmstudio', '0.0.25-1');
    const home = path.join(runDir, 'home', '.lmstudio');
    fs.mkdirSync(path.join(home, 'server-logs', '2026-09'), { recursive: true });
    fs.writeFileSync(path.join(home, 'server-logs', '2026-09', '2026-09-25.1.log'), 'a prompt');
    fs.mkdirSync(path.join(home, 'models', 'local-llm', 'old-model'), { recursive: true });
    const engine = ENGINE.replace(RUN_DIR, runDir);
    const kills = [];
    const procDir = fakeProc(t, [
        { pid: 100, exe: path.join(runDir, 'llmster'), argv: ['llmster'] },
        { pid: 101, exe: engine, argv: [engine, ...ENGINE_ARGS] },
        { pid: 102, exe: '/opt/llama.cpp/llama-server', argv: ['/opt/llama.cpp/llama-server'] },
    ]);
    const { ctx, calls } = startContext(t, runDir);
    // Nothing of ours runs yet; a stray from an earlier start would be killed first.
    const details = await lmStudioRunner.start(ctx, { procDir, kill: (pid, signal) => kills.push([pid, signal]), node: '/usr/local/bin/node' });
    assert.deepEqual(kills, [[100, 'SIGKILL'], [101, 'SIGKILL']], 'leftovers under the copy, never llama.cpp');
    assert.equal(fs.existsSync(path.join(home, 'server-logs')), false, 'the old server log is gone before the start');
    assert.equal(fs.existsSync(path.join(home, 'models', 'local-llm')), false, 'only our own earlier imports are cleared');
    const [launch] = calls.launched;
    assert.equal(launch.command, path.join(runDir, 'llmster'));
    assert.deepEqual(launch.args, []);
    assert.equal(launch.env.HOME, path.join(runDir, 'home'));
    assert.equal(typeof launch.outputFilter, 'function');
    assert.deepEqual(calls.exec.map((step) => (step.args[0] === HELPER ? step.args.slice(0, 2) : step.args)), [
        [HELPER, 'wait'],
        ['import', ctx.weights.path, '--symbolic-link', '-y', '--user-repo', 'local-llm/gpt-oss-20b'],
        [HELPER, 'load'],
        ['server', 'start', '--port', '18084', '--bind', '127.0.0.1'],
        ['ps', '--json'],
    ]);
    const lms = path.join(runDir, '.bundle', 'lms');
    assert.deepEqual(calls.exec.map((step) => step.command), ['/usr/local/bin/node', lms, '/usr/local/bin/node', lms, lms]);
    const load = JSON.parse(calls.exec[2].args[2]);
    assert.equal(load.path, 'local-llm/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf');
    assert.equal(load.identifier, 'gpt-oss-20b');
    assert.equal(load.sdk, '/opt/local-llm/lmstudio-sdk');
    assert.equal(load.config.llamaCppArgumentsOverride.overrideParameters[1].value, '17');
    assert.ok(calls.exec.every((step) => !Object.keys(step.env).some((key) => /token|secret|key/i.test(key))));
    assert.deepEqual(calls.waited, ['http://127.0.0.1:18084/v1/models']);
    assert.deepEqual(details, { lmstudio: { modelKey: 'gpt-oss-20b', engine: { pid: 101, dir: 'llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.41.0' } } });
});

test('start refuses an engine whose flags differ from what admission sized, and an update llmster staged', async (t) => {
    const root = tempDir(t, 'run');
    const runDir = path.join(root, 'lmstudio', '0.0.25-1');
    const engine = ENGINE.replace(RUN_DIR, runDir);
    const drifted = ENGINE_ARGS.map((arg) => (arg === '17' ? '16' : arg));
    const procDir = fakeProc(t, [{ pid: 201, exe: engine, argv: [engine, ...drifted] }]);
    const { ctx } = startContext(t, runDir);
    const kill = () => {};
    await assert.rejects(() => lmStudioRunner.start(ctx, { procDir, kill, node: 'node' }),
        (error) => error.code === 'runner_flags' && /--n-cpu-moe is 16, expected 17/.test(error.message));
    const staged = path.join(runDir, 'home', '.lmstudio', '.internal', 'staged-updates-app');
    fs.mkdirSync(staged, { recursive: true });
    fs.writeFileSync(path.join(staged, 'llmster-0.0.26'), 'x');
    const again = startContext(t, runDir);
    await assert.rejects(() => lmStudioRunner.start(again.ctx, { procDir: fakeProc(t, []), kill, node: 'node' }), { code: 'runner_changed' });
    assert.equal(again.calls.launched.length, 0, 'nothing was launched');
});

test('after llmster exits, its server log is deleted and anything left in its copy is killed', (t) => {
    const root = tempDir(t, 'run');
    const runDir = path.join(root, 'lmstudio', '0.0.25-1');
    fs.mkdirSync(path.join(serverLogsDir(runDir), '2026-09'), { recursive: true });
    fs.writeFileSync(path.join(serverLogsDir(runDir), '2026-09', '2026-09-25.1.log'), 'ZEBRAMARKER');
    const engine = ENGINE.replace(RUN_DIR, runDir);
    const kills = [];
    const lines = [];
    lmStudioRunner.afterExit({ runnerDir: runDir, log: (line) => lines.push(line) },
        { procDir: fakeProc(t, [{ pid: 301, exe: engine }, { pid: 302, exe: '/opt/ik_llama.cpp/llama-server' }]), kill: (pid, signal) => kills.push([pid, signal]) });
    assert.deepEqual(kills, [[301, 'SIGKILL']]);
    assert.deepEqual(lines, ['killed a leftover LM Studio process (301)']);
    assert.equal(fs.existsSync(serverLogsDir(runDir)), false);
    // A runner outside the installer has no copy to clean.
    assert.doesNotThrow(() => lmStudioRunner.afterExit({ runnerDir: null }));
    // The watchdog runs every 30 s.
    assert.equal(lmStudioRunner.watchdog.intervalMs, 30_000);
});

// ---- the output filter in the real runner process (review findings, 2026-09-25)

import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';

import { createLogBuffer, startRunnerProcess } from '../src/controller/runnerProcess.mjs';

function fakeChild() {
    const child = new EventEmitter();
    child.pid = 4343;
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = () => {};
    return child;
}

async function runThroughFilter(write) {
    const log = createLogBuffer();
    const child = fakeChild();
    const launch = lmStudioRunner.buildLaunch({ runnerDir: RUN_DIR });
    startRunnerProcess({ command: launch.command, args: launch.args, env: {}, log, filter: launch.outputFilter,
        spawnImpl: () => child, killImpl: () => {} });
    await write(child.stdout);
    child.stdout.end();
    await new Promise((resolve) => setImmediate(resolve));
    return log.all().map((entry) => entry.line);
}

test('the adapter\'s output filter works in the real runner process: one fresh filter per stream', async () => {
    const lines = await runThroughFilter(async (out) => {
        out.write('[APIServerProvider] API Server started on port: 41343\n');
        out.write('[2026-09-25 11:13:43][DEBUG] Received request: POST to /v1/chat/completions with body {\n  "messages": [ "ZEBRA" ]\n}\n');
        out.write('llmster started successfully! \n');
    });
    // The request line and its body are dropped; the next line is logged again.
    assert.deepEqual(lines, ['[APIServerProvider] API Server started on port: 41343', 'llmster started successfully!']);
});

test('a body line longer than the log\'s line bound never leaks, inline or pretty-printed', async () => {
    const big = (fill) => fill.repeat(300 * 1024);
    for (const [shape, text] of [
        ['inline', `[2026-09-25][ERROR] failed: {"content":"${big('ZEBRA ')}"}\nApp is quitting\n`],
        ['pretty', `[2026-09-25][DEBUG] Received request: POST to /v1/chat/completions with body {\n  "content": "${big('a]b}')}"\n}\nApp is quitting\n`],
        ['bare', `${big('ZEBRA ')}\nApp is quitting\n`],
    ]) {
        const lines = await runThroughFilter(async (out) => {
            for (let at = 0; at < text.length; at += 50_000) out.write(text.slice(at, at + 50_000));
        });
        const leaked = lines.filter((line) => /ZEBRA|a\]b\}/.test(line));
        assert.deepEqual(leaked.map((line) => line.slice(0, 60)), [], shape);
        assert.ok(lines.includes('App is quitting'), `${shape}: logging resumes after the long line`);
    }
});

test('a header that opens a JSON block after inline JSON keeps nothing past the first brace', () => {
    const filter = createOutputFilter();
    assert.equal(filter('[E] x: {"content":"ZEBRA","a":['), '[E] x: {…}');
    assert.equal(filter('  "b"'), null);
    assert.equal(filter(']'), null);
    assert.equal(filter('[I] next'), '[I] next');
});

test('the watchdog unloads what a request loaded, keeps ours, and is not logged every 30 s when it finds nothing', async () => {
    const run = async (listed) => {
        const steps = [];
        const note = await lmStudioRunner.watchdog.check({ exec: async (step) => { steps.push(step); return step.args[0] === 'ps' ? listed : {}; },
            modelId: 'qwen3-0.6b', runnerDir: RUN_DIR });
        return { note, steps };
    };
    const quiet = await run([{ identifier: 'qwen3-0.6b', type: 'llm' }]);
    assert.equal(quiet.note, null);
    assert.deepEqual(quiet.steps.map((step) => [step.args, step.quiet]), [[['ps', '--json'], true]]);
    assert.equal(quiet.steps[0].command, `${RUN_DIR}/.bundle/lms`);
    const busy = await run([{ identifier: 'qwen3-0.6b', type: 'llm' }, { identifier: 'text-embedding-nomic-embed-text-v1.5', type: 'embedding' }]);
    assert.deepEqual(busy.steps.map((step) => step.args), [['ps', '--json'], ['unload', 'text-embedding-nomic-embed-text-v1.5']]);
    assert.equal(busy.note, 'unloaded text-embedding-nomic-embed-text-v1.5, which a request had loaded');
});

test('start unloads anything else LM Studio holds after the load, and fails when our model is not among them', async (t) => {
    const root = tempDir(t, 'run');
    const runDir = path.join(root, 'lmstudio', '0.0.25-1');
    const engine = ENGINE.replace(RUN_DIR, runDir);
    const procDir = fakeProc(t, [{ pid: 401, exe: engine, argv: [engine, ...ENGINE_ARGS] }]);
    const extra = startContext(t, runDir);
    extra.calls.listed = [{ identifier: 'gpt-oss-20b', type: 'llm' }, { identifier: 'text-embedding-nomic-embed-text-v1.5', type: 'embedding' }];
    await lmStudioRunner.start(extra.ctx, { procDir, kill: () => {}, node: 'node' });
    assert.deepEqual(extra.calls.exec.at(-1).args, ['unload', 'text-embedding-nomic-embed-text-v1.5']);
    const missing = startContext(t, runDir);
    missing.calls.listed = [];
    await assert.rejects(() => lmStudioRunner.start(missing.ctx, { procDir, kill: () => {}, node: 'node' }),
        (error) => error.code === 'runner_step_failed' && /gpt-oss-20b is not loaded/.test(error.message));
});
