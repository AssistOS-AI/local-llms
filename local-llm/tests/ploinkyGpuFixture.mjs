import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { validateRunnerLock } from '../src/controller/runnerLock.mjs';
import { vllmRunnerLockDigest } from '../src/controller/vllmMpsQualification.mjs';
import { RUNNERS } from '../src/runners/index.mjs';
import { IMAGE_LOCK } from './overview-scenarios.mjs';

export const GIB = 1024 ** 3;
export const MIB = 1024 ** 2;
export const GPU_TEST_LOCK = validateRunnerLock(IMAGE_LOCK);
export const MPS_ENV = Object.freeze({ CUDA_MPS_PIPE_DIRECTORY: '/run/ploinky-mps-pipe', CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: '50', CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=3072M' });
export const TEST_GPU = Object.freeze({ available: true, name: 'Synthetic test GPU', memoryModel: 'dedicated', driverVersion: '999.1',
    totalBytes: 6 * GIB, freeBytes: 3 * GIB, usedBytes: 3 * GIB, processes: [], device: { pciDeviceId: '0x123410DE', computeCapability: '9.9', addressingMode: 'None' } });
const SEED = loadSeedCatalog();
const awq = SEED.find((entry) => entry.id === 'qwen3-4b-awq');
const raw = structuredClone(Object.fromEntries(Object.entries(awq).filter(([key]) => !['seed', 'recommended'].includes(key))));
// Exactly 2 GiB with 512 tokens: 1216 MiB pinned fixture weights + 64 MiB
// KV + the existing 768 MiB vLLM overhead. This is test data, not a live model.
raw.id = 'budget-fixture-awq';
raw.sources.hf.files = [{ path: 'model.safetensors', size: 1216 * MIB, sha256: 'c'.repeat(64) }];
delete raw.sources.hf.size;
raw.memory.kvBytesPerToken = 128 * 1024;
export const GPU_MODEL = validateModel(raw, { seed: true });

export function testQualification(gpu = TEST_GPU) {
    return { runnerLockDigest: vllmRunnerLockDigest(GPU_TEST_LOCK.runners.vllm), driverVersion: gpu.driverVersion,
        gpuPciDeviceId: gpu.device.pciDeviceId, computeCapability: gpu.device.computeCapability, deviceTotalBytes: gpu.totalBytes,
        denominator: 'physical-device', evidenceDigest: 'e'.repeat(64) };
}

export function gpuHarness(t, { gpu = TEST_GPU, env = MPS_ENV, qualificationDataProvider = () => [testQualification(gpu)], profile = null, seedCatalog = null } = {}) {
    const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-gpu-budget-')));
    const dataDir = path.join(root, 'data');
    const weights = path.join(root, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const runDir = path.join(root, 'opt', 'runners', 'run');
    fs.mkdirSync(path.join(runDir, 'tabbyAPI'), { recursive: true });
    const procDir = path.join(root, 'proc');
    fs.mkdirSync(procDir);
    const runners = { ...RUNNERS, lmstudio: { ...RUNNERS.lmstudio, start: (ctx) => RUNNERS.lmstudio.start(ctx, { procDir }) } };
    const started = [];
    let currentGpu = structuredClone(gpu);
    const installer = {
        installable: () => true, describe: async () => ({ installed: true, runnable: true, version: 'test', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
        ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }),
        entryFor: (id) => GPU_TEST_LOCK.runners[id] || { id, version: 'test' }, pathsFor: () => ({ runDir }),
    };
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin', LOCAL_LLM_VLLM_UNIFIED: 'experimental', LOCAL_LLM_LMSTUDIO: 'internal-use', LOCAL_LLM_RUN_ROOT: path.join(root, 'opt', 'runners'), PRIVATE_AGENT_SECRET: 'do-not-forward', ...env },
        seedCatalog: seedCatalog || [...SEED, GPU_MODEL], runners, runnerLocks: GPU_TEST_LOCK, qualificationDataProvider, installer,
        stateStore: createStateStore({ dataDir }), profile,
        snapshot: async () => ({ gpu: structuredClone(currentGpu), memory: { totalBytes: 128 * GIB, availableBytes: 120 * GIB }, disk: { freeBytes: 1024 * GIB, totalBytes: 2048 * GIB }, cpus: 20, cores: 10, cgroupMemory: null }),
        download: async () => ({ status: 'complete', path: weights, bytesTransferred: 0 }), inspect: async () => ({ state: 'absent', bytes: 0 }), remove: async () => 0,
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }), downloadSnapshot: async () => ({ status: 'complete', bytesTransferred: 0 }),
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '{}' }), apiKeyFactory: () => 'k'.repeat(43),
        detectRunner: (definition) => ({ installed: definition.supported, version: definition.pinnedVersion || 'test', reason: null }), imageContract: null,
        fileExists: () => true, sharedModelsRoot: null, shmDir: path.join(root, 'shm'), readMemory: () => ({ totalBytes: 128 * GIB, availableBytes: 120 * GIB }),
        readPressure: () => 0, unifiedGuardMs: 60_000, memoryGuardLoadMs: 60_000, memoryGuardReadyMs: 60_000, dropCache: () => true, pollMs: 2, stopGraceMs: 20,
        startRunner({ command, args, env: runnerEnv }) {
            let running = true;
            let finish;
            const handle = { pid: 6000 + started.length, command, args, env: runnerEnv, exited: new Promise((resolve) => { finish = resolve; }), get running() { return running; },
                async stop() { running = false; finish({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                async kill() { running = false; finish({ code: null, signal: 'SIGKILL', error: null }); return handle.exited; } };
            started.push(handle);
            return handle;
        },
    });
    t.after(async () => { await controller.stop(); await fs.promises.rm(root, { recursive: true, force: true }); });
    return { controller, started, setGpu(next) { currentGpu = structuredClone(next); },
        run: (params = {}, { runnerId = 'vllm', modelId = GPU_MODEL.id, requestId = 'gpu-budget-test-request' } = {}) => controller.run({ modelId, runnerId, params, requestId }) };
}

export async function waitForLaunch(harness) {
    const deadline = Date.now() + 3000;
    while (!harness.started.length && Date.now() < deadline && harness.controller.state.deployment?.phase !== 'error') await new Promise((resolve) => setTimeout(resolve, 2));
    assert.ok(harness.started.length, `runner launch reached: ${harness.controller.state.deployment?.error || harness.controller.state.deployment?.phase}`);
    return harness.started[0];
}
