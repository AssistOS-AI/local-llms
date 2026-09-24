// vLLM (runners plan §5.5, Phase R5): installed on demand (DS004), reads a
// Hugging Face snapshot (hf), runs from its runnable copy on loopback with a
// per-start key, and is admitted only when weights and KV cache fit the GPU,
// unless the admin explicitly offloads weights to RAM.
import assert from 'node:assert/strict';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { defaultPorts } from '../src/runners/index.mjs';
import { ParamError } from '../src/runners/params.mjs';
import { vllmRunner } from '../src/runners/vllm.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const SEED = loadSeedCatalog();
const QWEN = SEED.find((model) => model.id === 'qwen3-4b-awq');
const GPT = SEED.find((model) => model.id === 'gpt-oss-20b');
const RUN_DIR = '/opt/runners/vllm/0.30.0';
const SNAPSHOT = '/data/models/hf/Qwen/Qwen3-4B-AWQ/74d4bd2bd4bff9cafc9345221320bffb08b406a3';
const CACHE = '/opt/runners/.cache/vllm';

function snapshot({ freeMiB = 6000, availableGiB = 24, processes = [] } = {}) {
    return {
        gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: (6144 - freeMiB) * MIB, freeBytes: freeMiB * MIB, processes },
        memory: { totalBytes: 31 * GIB, availableBytes: availableGiB * GIB },
        disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
    };
}

function launch(params = {}, extra = {}) {
    return vllmRunner.buildLaunch({
        runnerDir: RUN_DIR, artifactPath: SNAPSHOT, params, port: 18082, apiKey: KEY, model: QWEN,
        gpuMemoryUtilization: 0.85, cacheDir: CACHE, ...extra,
    });
}

const valueOf = (args, flag) => args[args.indexOf(flag) + 1];

test('the seed catalog pins Qwen3-4B-AWQ and gpt-oss-20b as Hugging Face snapshots for vLLM', () => {
    assert.ok(QWEN, 'qwen3-4b-awq is a seed model');
    assert.equal(QWEN.sources.hf.repo, 'Qwen/Qwen3-4B-AWQ');
    assert.equal(QWEN.sources.hf.commit, '74d4bd2bd4bff9cafc9345221320bffb08b406a3');
    assert.equal(QWEN.sources.hf.files.length, 7);
    assert.equal(QWEN.sources.hf.size, 2681909887);
    assert.equal(GPT.sources.hf.repo, 'openai/gpt-oss-20b');
    assert.equal(GPT.sources.hf.commit, '6cee5e81ee83917806bbde320786a8fb61efebee');
    assert.equal(GPT.sources.hf.files.length, 10);
    assert.equal(GPT.sources.hf.size, 13789244452);
    assert.ok(QWEN.memory.kvBytesPerToken > 0);
});

test('vLLM is supported, reads hf snapshots and has its own loopback port', () => {
    assert.equal(vllmRunner.supported, true);
    assert.equal(vllmRunner.weightFormat, 'hf');
    assert.equal(vllmRunner.pinnedVersion, '0.30.0');
    assert.equal(defaultPorts().vllm, 18082);
    assert.equal(new Set(Object.values(defaultPorts())).size, Object.keys(defaultPorts()).length);
});

test('vLLM runs from its runnable copy on loopback, with the key in its environment, never in its arguments', () => {
    const { command, args, env } = launch();
    assert.equal(command, `${RUN_DIR}/venv/bin/python`);
    assert.deepEqual(args.slice(0, 2), ['-m', 'vllm.entrypoints.openai.api_server']);
    assert.ok(args.every((arg) => typeof arg === 'string'));
    assert.equal(valueOf(args, '--model'), SNAPSHOT);
    assert.equal(valueOf(args, '--served-model-name'), 'qwen3-4b-awq');
    assert.equal(valueOf(args, '--host'), '127.0.0.1');
    assert.equal(valueOf(args, '--port'), '18082');
    assert.equal(valueOf(args, '--max-model-len'), String(QWEN.recommended.vllm.maxModelLen));
    assert.equal(valueOf(args, '--gpu-memory-utilization'), '0.85');
    assert.equal(valueOf(args, '--max-num-seqs'), '1');
    assert.ok(args.includes('--enforce-eager'), 'eager by default for a faster start');
    assert.equal(args.includes('--api-key'), false);
    assert.equal(args.join(' ').includes(KEY), false);
    assert.equal(args.includes('--cpu-offload-gb'), false);
    assert.equal(env.VLLM_API_KEY, KEY);
    assert.equal(env.LD_LIBRARY_PATH, '/usr/local/nvidia/lib64');
    // Triton links the granted driver library by its versioned name.
    assert.equal(env.TRITON_LIBCUDA_PATH, '/usr/local/nvidia/lib64');
    assert.ok(env.TRITON_CACHE_DIR.startsWith(`${CACHE}/`));
    // ZMQ and distributed-init sockets fail on the container's fuse-overlayfs
    // /tmp; the agent's own /dev/shm (a private tmpfs since R4) works.
    assert.ok(env.VLLM_RPC_BASE_PATH.startsWith('/dev/shm/'));
    assert.equal(env.TMPDIR, env.VLLM_RPC_BASE_PATH);
    // torch.distributed's Gloo and NCCL groups otherwise listen on the
    // container's Box-network address, where other agents could reach them.
    assert.equal(env.GLOO_SOCKET_IFNAME, 'lo');
    assert.equal(env.NCCL_SOCKET_IFNAME, 'lo');
    assert.equal(env.VLLM_HOST_IP, '127.0.0.1');
    for (const [name, value] of Object.entries({ HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', VLLM_USE_FLASHINFER_SAMPLER: '0', VLLM_NO_USAGE_STATS: '1', DO_NOT_TRACK: '1' })) {
        assert.equal(env[name], value, name);
    }
});

test('explicit parameters win; offload, quantization and KV cache type are passed only when set', () => {
    const { args } = launch({ gpuMemoryUtilization: 0.7, maxModelLen: 2048, cpuOffloadGb: 4, kvCacheDtype: 'fp8', enforceEager: false, quantization: 'awq_marlin', maxNumSeqs: 2 });
    assert.equal(valueOf(args, '--gpu-memory-utilization'), '0.7');
    assert.equal(valueOf(args, '--max-model-len'), '2048');
    assert.equal(valueOf(args, '--cpu-offload-gb'), '4');
    assert.equal(valueOf(args, '--kv-cache-dtype'), 'fp8');
    assert.equal(valueOf(args, '--quantization'), 'awq_marlin');
    assert.equal(valueOf(args, '--max-num-seqs'), '2');
    assert.equal(args.includes('--enforce-eager'), false);
    assert.throws(() => launch({ gpuMemoryUtilization: 1.2 }), ParamError);
    assert.throws(() => launch({ tensorParallelSize: 2 }), ParamError);
    assert.throws(() => launch({ quantization: 'awq; rm -rf /' }), ParamError);
    assert.throws(() => launch({}, { runnerDir: 'relative' }));
    assert.throws(() => launch({}, { artifactPath: 'relative' }));
});

test('admission: Qwen3-4B-AWQ fits a 6 GB GPU and gets a GPU memory share from what is free', () => {
    const params = vllmRunner.normalizeParams({}, { model: QWEN });
    const result = admit({ runner: vllmRunner, model: QWEN, source: QWEN.sources.hf, params, snapshot: snapshot() });
    assert.equal(result.status, 'ok', result.reason);
    assert.equal(result.estimate.weightsBytes, QWEN.sources.hf.size);
    assert.equal(result.estimate.kvBytes, QWEN.memory.kvBytesPerToken * params.maxModelLen);
    assert.ok(result.estimate.gpuMemoryUtilization >= 0.8 && result.estimate.gpuMemoryUtilization <= 0.9, String(result.estimate.gpuMemoryUtilization));
    assert.equal(result.estimate.isEstimate, true);
    // Another process holds most of the GPU now.
    const busy = admit({ runner: vllmRunner, model: QWEN, source: QWEN.sources.hf, params,
        snapshot: snapshot({ freeMiB: 2048, processes: [{ name: 'ollama', usedBytes: 4000 * MIB }] }) });
    assert.equal(busy.status, 'insufficient-now');
    assert.match(busy.reason, /free now.*ollama/);
    // An admin-set share that more than the free memory is refused before launch.
    const tooMuch = admit({ runner: vllmRunner, model: QWEN, source: QWEN.sources.hf, params: { ...params, gpuMemoryUtilization: 0.95 },
        snapshot: snapshot({ freeMiB: 5000 }) });
    assert.equal(tooMuch.status, 'insufficient-now');
});

test('admission refuses gpt-oss-20b on vLLM without offload, with the reason and the way out; offload is explicit and warned', () => {
    const params = vllmRunner.normalizeParams({}, { model: GPT });
    const refused = admit({ runner: vllmRunner, model: GPT, source: GPT.sources.hf, params, snapshot: snapshot() });
    assert.equal(refused.status, 'incompatible');
    assert.match(refused.reason, /GPU memory/);
    assert.match(refused.reason, /cpuOffloadGb/);
    const offloaded = admit({ runner: vllmRunner, model: GPT, source: GPT.sources.hf,
        params: vllmRunner.normalizeParams({ cpuOffloadGb: 11 }, { model: GPT }), snapshot: snapshot() });
    assert.equal(offloaded.status, 'ok', offloaded.reason);
    assert.ok(offloaded.warnings.some((warning) => /offload/i.test(warning) && /slower/.test(warning)));
    assert.equal(offloaded.estimate.cpuWeightsBytes, 11 * GIB);
    // Offloading more than the RAM that is free is refused.
    const noRam = admit({ runner: vllmRunner, model: GPT, source: GPT.sources.hf,
        params: vllmRunner.normalizeParams({ cpuOffloadGb: 11 }, { model: GPT }), snapshot: snapshot({ availableGiB: 6 }) });
    assert.equal(noRam.status, 'insufficient-now');
});

test('the report parser reads the model and KV cache sizes vLLM logs', () => {
    const report = vllmRunner.parseReport([
        { line: '(EngineCore pid=1162) INFO 09-24 20:37:24 [model_runner.py:428] Model loading took 2.5 GiB memory and 2.154850 seconds' },
        { line: '(EngineCore pid=1162) INFO 09-24 20:37:27 [gpu_worker.py:640] Available KV cache memory: 2.05 GiB' },
        { line: '(EngineCore pid=1162) INFO 09-24 20:37:27 [kv_cache_utils.py:2395] GPU KV cache size: 14,896 tokens, Maximum concurrency for 4,096 tokens per request: 3.64x' },
    ]);
    assert.equal(report.modelMiB, 2560);
    assert.equal(report.kvMiB, 2099);
    assert.equal(report.kvTokens, 14896);
    assert.equal(report.totalMiB, 4659);
});

test('admission: an admin-set GPU share that can never fit is incompatible, not busy-now; a share below 0.1 is busy-now', () => {
    const params = vllmRunner.normalizeParams({ gpuMemoryUtilization: 0.3 }, { model: QWEN });
    const small = admit({ runner: vllmRunner, model: QWEN, source: QWEN.sources.hf, params, snapshot: snapshot() });
    assert.equal(small.status, 'incompatible');
    assert.match(small.reason, /gpuMemoryUtilization/);
    // A large GPU with little free memory would leave vLLM a share below its minimum.
    const big = { ...snapshot(), gpu: { available: true, name: 'Big GPU', totalBytes: 80 * GIB, usedBytes: 72 * GIB, freeBytes: 8 * GIB, processes: [] } };
    const busy = admit({ runner: vllmRunner, model: QWEN, source: QWEN.sources.hf, params: vllmRunner.normalizeParams({}, { model: QWEN }), snapshot: big });
    assert.equal(busy.status, 'insufficient-now');
    assert.ok(busy.estimate.gpuMemoryUtilization < 0.1);
});

test('a vLLM Run reaches ready through the controller, launched with the GPU share admission computed', async (t) => {
    const { createController } = await import('../src/controller/deployments.mjs');
    const { createStateStore } = await import('../src/controller/stateStore.mjs');
    const fs = await import('node:fs');
    const os = await import('node:os');
    const path = await import('node:path');
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-vllm-run-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const runDir = path.join(root, 'opt', 'runners', 'vllm', '0.30.0');
    const started = [];
    const probes = [];
    const installer = {
        installable: (id) => id === 'vllm',
        describe: async () => ({ installed: true, runnable: true, version: '0.30.0', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
        ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }),
        entryFor: () => ({ id: 'vllm', version: '0.30.0' }),
        pathsFor: () => ({ runDir }),
    };
    const controller = createController({
        dataDir: path.join(root, 'data'),
        env: { PATH: '/usr/bin' },
        seedCatalog: SEED,
        stateStore: createStateStore({ dataDir: path.join(root, 'data') }),
        snapshot: async () => snapshot(),
        installer,
        detectRunner: async () => ({ installed: true, version: '0.30.0', reason: null }),
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }),
        downloadSnapshot: async () => ({ status: 'complete', bytesTransferred: 0 }),
        shmDir: path.join(root, 'shm'),
        startRunner({ command, args, env, cwd }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 4242, command, args, env, cwd,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async (url, options = {}) => {
            probes.push([String(url), options.headers?.authorization ? 'Bearer ***' : null]);
            return { ok: true, status: 200, json: async () => ({}) };
        },
        pollMs: 2,
        stopGraceMs: 50,
    });
    await controller.run({ modelId: 'qwen3-4b-awq', runnerId: 'vllm', requestId: 'request-vllm-ready' });
    for (let index = 0; index < 500 && controller.state.deployment?.phase !== 'ready'; index += 1) {
        assert.notEqual(controller.state.deployment?.phase, 'error', controller.state.deployment?.error);
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.equal(controller.state.deployment.phase, 'ready');
    const [launched] = started;
    assert.equal(launched.command, `${runDir}/venv/bin/python`);
    const share = controller.state.deployment.admission.estimate.gpuMemoryUtilization;
    assert.equal(valueOf(launched.args, '--gpu-memory-utilization'), String(share));
    assert.ok(launched.env.VLLM_RPC_BASE_PATH.startsWith(path.join(root, 'shm')));
    assert.ok(fs.existsSync(path.join(root, 'opt', 'runners', '.cache', 'vllm')));
    assert.deepEqual(probes.map(([url, auth]) => [new URL(url).pathname, auth]), [['/health', null], ['/v1/models', 'Bearer ***']]);
    await controller.stop();
});
