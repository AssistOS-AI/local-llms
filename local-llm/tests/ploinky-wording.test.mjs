// The Ploinky wording (V6): every message the agent hands the engine, and every sentence its hardware limits emit, equals the
// literal it replaced, byte for byte. The expected texts are copied from local-llms 34f8f498 (the tree before the engine moved);
// the base file and line stand above each. The engine's own defaults are neutral, so a key that is not bound here would show
// as different wording.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { MESSAGE_KEYS } from 'pworker/local-llm';

import { createAgentController } from '../src/engineOptions.mjs';
import { APU_BUDGET_REASON, createAgentLimits } from '../src/hardwareLimits.mjs';
import { PLOINKY_MESSAGES } from '../src/ploinkyMessages.mjs';

const GIB = 1024 ** 3;
const MIB = 1024 ** 2;
const BASE_MESSAGES = [
    // controller/deployments.mjs:981
    ['notReady', [], 'No local model is ready. Run one from Settings > Agents > Local LLMs.'],
    // controller/hardware.mjs:198
    ['gpuAccessReported', ['driver said no'], 'Ploinky reports: driver said no'],
    // controller/hardware.mjs:198
    ['gpuAccessReported', [null], 'Ploinky reports: GPU access is unavailable; the cause is unknown. Check host `ploinky gpu status`.'],
    // controller/hardware.mjs:200-202
    ['gpuAbsent', [], 'No GPU is available to this agent. On the host, `ploinky gpu status` shows why: revoked: run `ploinky gpu grant --agent local-llms/local-llm`; not applied yet: run `ploinky start`; or the host has no usable GPU.'],
    // controller/modelLookup.mjs:49
    ['hfTokenHint', [], 'set a token with `ploinky var HF_TOKEN <token>` and restart local-llm'],
    // controller/hardware.mjs:496
    ['vulkanRuntimeMissing', [], 'A GPU render node is attached, but this image lacks a supported declared Vulkan runtime; update the local-llm image.'],
    // controller/vulkan.mjs:95
    ['apuSwitchOff', ['Radeon X', 'LOCAL_LLM_AMD_APU', 'experimental'], 'An AMD integrated GPU (Radeon X) is attached. APU use is off: the operator enables it with ploinky var LOCAL_LLM_AMD_APU experimental and a restart of local-llm.'],
    // runners/vllm.mjs:132-133
    ['vllmUnifiedSwitchHint', ['LOCAL_LLM_VLLM_UNIFIED', 'experimental'], 'The operator can turn it on with ploinky var LOCAL_LLM_VLLM_UNIFIED experimental; it has not been measured on this hardware.'],
    // runners/lmStudio.mjs:258
    ['lmStudioEnableHint', ['LOCAL_LLM_LMSTUDIO', 'internal-use'], 'This deployment\'s operator enables it with "ploinky var LOCAL_LLM_LMSTUDIO internal-use" and a restart of local-llm.'],
    // runners/lmStudio.mjs:338
    ['lmStudioStagedUpdate', [], 'LM Studio has staged an update in its home; restart local-llm to rebuild LM Studio from its verified download.'],
    // controller/computeDiagnostics.mjs:17
    ['profileRestartGpuRecovered', [], 'GPU access is available now; restart local-llm to select it. Stopping and starting the model does not change the compute profile.'],
    // controller/computeDiagnostics.mjs:18
    ['profileRestartChanged', [], 'After changing GPU access, machine or provider, restart the agent or Box through Ploinky to select a new compute profile.'],
    // controller/admission.mjs:420
    ['restartToUseDevice', ['vulkan'], 'This agent started with the vulkan profile; restart local-llm to use the current device or run on the CPU.'],
    // controller/admission.mjs:435
    ['restartToRunOnCpu', ['dedicated'], 'This agent started with the dedicated profile; restart local-llm to run on the CPU.'],
    // controller/admission.mjs:441-442
    ['restartForMemoryModel', ['unified', 'dedicated'], 'The GPU now reports unified memory, but this agent started with the dedicated profile; restart local-llm to use it.'],
];

test('every message key is bound, and each one says what its base literal said', () => {
    assert.deepEqual(Object.keys(PLOINKY_MESSAGES).sort(), [...MESSAGE_KEYS].sort());
    assert.deepEqual([...new Set(BASE_MESSAGES.map(([key]) => key))].sort(), [...MESSAGE_KEYS].sort(), 'a key without a base literal here is not covered');
    for (const [key, args, expected] of BASE_MESSAGES) assert.equal(PLOINKY_MESSAGES[key](...args), expected, `${key}(${args.join(', ')})`);
});

test('the controller built for the agent words its refusals with them', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-wording-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const controller = createAgentController({ env: { LOCAL_LLM_DATA_DIR: dataDir } });
    assert.throws(() => controller.chatTarget(), (error) => error.code === 'not_ready'
        && error.message === 'No local model is ready. Run one from Settings > Agents > Local LLMs.');
    await controller.drain();
});

// What the hardware limits emit. `refuse` and `result` stand in for the engine's builders and keep what they are given.
const refuse = (...parts) => parts;
const result = (status, reason, estimate, warnings, reasonCode) => ({ status, reason, estimate, warnings, reasonCode });
const gib = (bytes) => `${(bytes / GIB).toFixed(2)} GiB`;
const amount = (bytes) => (bytes >= GIB ? `${(bytes / GIB).toFixed(2)} GiB` : `${Math.round(bytes / MIB)} MiB`);
const MPS = Object.freeze({ CUDA_MPS_PIPE_DIRECTORY: '/run/ploinky-mps-pipe', CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: '50', CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=3072M' });
const limits = createAgentLimits();

test('a pinned layer count that does not fit the GPU share says so in the share\'s words (admission.mjs:200)', () => {
    const refusal = limits.refuseResourceRequest({ policy: 'ollama', gpu: { budgetBytes: 3 * GIB, freeBytes: 2 * GIB }, estimate: { gpuBytes: 5 * GIB }, marginBytes: 256 * MIB, gib, refuse });
    assert.deepEqual(refusal, ['insufficient-now', 'Pinned GPU layers need about 5.00 GiB; only 2.00 GiB of the Ploinky GPU budget is free now.']);
});

test('an explicit vLLM utilization above the share says so in the share\'s words (admission.mjs:271)', () => {
    const refusal = limits.refuseResourceRequest({ policy: 'vllm', gpu: { budgetBytes: 3 * GIB, totalBytes: 3 * GIB }, params: { gpuMemoryUtilization: 0.9 }, gpuMemoryUtilization: 0.9, denominator: 6 * GIB, gib, refuse });
    assert.deepEqual(refusal, ['incompatible', 'gpuMemoryUtilization 0.9 allocates 5.40 GiB against physical GPU memory, exceeding the 3.00 GiB Ploinky GPU share. Lower gpuMemoryUtilization or clear the share.']);
});

test('a GPU share is refused on a Vulkan GPU, on unified memory and when the telemetry cannot be read, in the base words (admission.mjs:415, :446, :449)', () => {
    const gate = { snapshot: {}, env: MPS, result };
    assert.deepEqual(limits.admit({ stage: 'vulkan', ...gate }), result('incompatible', 'GPU shares need NVIDIA MPS; Vulkan GPUs cannot use this share.', {}, [], 'gpu_share_unsupported'));
    assert.deepEqual(limits.admit({ stage: 'gpu', selected: 'unified', gpu: { memoryModel: 'unified', totalBytes: 8 * GIB, freeBytes: 4 * GIB }, ...gate }).refusal,
        result('incompatible', 'Ploinky GPU sharing requires a verified dedicated-memory GPU. Unified or unknown memory-model GPUs cannot use this MPS share; clear it through Ploinky.', {}, [], 'gpu_share_unsupported'));
    assert.deepEqual(limits.admit({ stage: 'gpu', selected: 'dedicated', gpu: { memoryModel: 'dedicated', totalBytes: Number.NaN, freeBytes: 4 * GIB }, ...gate }).refusal,
        result('insufficient-now', 'The physical GPU capacity or current free memory cannot be read under this Ploinky share. Retry after repairing GPU telemetry.', {}, [], 'gpu_budget_unreadable'));
});

test('the container-limit sentences: the APU reason, the pool warning, the APU refusal and the guard\'s two stops (vulkan.mjs:14, profiles.mjs:533 and :618, deployments.mjs:1190-1191 and :1208)', () => {
    assert.equal(APU_BUDGET_REASON, 'APU GPU use is not available while this agent has a finite or unknown container memory limit; such a limit is not verified to cap shared GPU memory.');
    // The APU reason reaches the snapshot's GPU under a finite limit.
    const fsApi = { readFileSync(file) { if (file.endsWith('memory.max')) return `${2 * GIB}`; if (file.endsWith('memory.current')) return '0'; throw Object.assign(new Error('absent'), { code: 'ENOENT' }); } };
    const shared = limits.decorateSnapshot({ snapshot: { gpu: { available: true, memoryModel: 'shared' }, memory: {} }, fsApi, env: {}, observation: limits.observeSnapshot({ fsApi }) });
    assert.equal(shared.gpu.reason, APU_BUDGET_REASON);
    assert.equal(limits.memoryPool({ totalBytes: 8 * GIB, availableBytes: 6 * GIB }, shared, { amount }).warning, 'A container memory limit of 2.00 GiB applies.');
    assert.deepEqual(limits.refuseProfile({ stage: 'policy', profile: 'apu', snapshot: shared, refuse }), ['incompatible', 'C3', 'APU shared GPU memory cannot be used under a container memory limit.']);
    assert.deepEqual(limits.refuseProfile({ stage: 'admission', profile: 'apu', snapshot: shared, refuse }), ['incompatible', 'C3', APU_BUDGET_REASON]);
    assert.deepEqual(limits.guardBreach({ memoryReadState: 'known', finiteMemoryBytes: GIB }, { check: 'watchable', profile: 'apu' }),
        { message: 'stopped: APU shared GPU memory cannot be watched under a finite or unknown container memory limit' });
    assert.deepEqual(limits.guardBreach({ memoryReadState: 'unknown', finiteMemoryBytes: GIB }, { check: 'headroom', memory: {}, available: 1 }),
        { unreadable: 'stopped: budget_unreadable: a required host or container memory reading could not be read twice in a row, so the runner cannot be watched' });
});
