import assert from 'node:assert/strict';
import { MPS_VARIABLES } from '../src/controller/ploinkyBudget.mjs';
import { GIB, GPU_MODEL, MPS_ENV, TEST_GPU, gpuHarness, testQualification, waitForLaunch } from './ploinkyGpuFixture.mjs';

export const QUALIFICATION_MISMATCHES = Object.freeze([
    ['runnerLockDigest', 'b'.repeat(64)], ['driverVersion', '999.2'], ['gpuPciDeviceId', '0x987610DE'],
    ['computeCapability', '9.8'], ['deviceTotalBytes', 7 * GIB],
]);
export const GPU_ADAPTERS = Object.freeze([
    ['llama.cpp', 'qwen2.5-0.5b-instruct-q4_k_m'], ['ik_llama.cpp', 'qwen2.5-0.5b-instruct-q4_k_m'],
    ['lmstudio', 'qwen2.5-0.5b-instruct-q4_k_m'], ['ollama', 'gpt-oss-20b'], ['vllm', GPU_MODEL.id], ['tabbyapi', 'qwen3-8b-exl3'],
]);
export const CPU_ADAPTERS = Object.freeze([
    ['llama.cpp', 'qwen2.5-0.5b-instruct-q4_k_m'], ['llama.cpp-cpu', 'qwen2.5-0.5b-instruct-q4_k_m'], ['ollama', 'gpt-oss-20b'],
]);
export const INVALID_MPS_VALUES = Object.freeze([
    { CUDA_MPS_PIPE_DIRECTORY: MPS_ENV.CUDA_MPS_PIPE_DIRECTORY },
    { ...MPS_ENV, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: undefined },
    { ...MPS_ENV, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: '0' },
    { ...MPS_ENV, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: '101' },
    { ...MPS_ENV, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '1=3072M' },
    { ...MPS_ENV, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=3072M,1=1M' },
    { ...MPS_ENV, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=9007199254740991M' },
    { ...MPS_ENV, CUDA_MPS_PIPE_DIRECTORY: '/secret-path-do-not-echo' },
]);

// Each aggregate iteration owns its cleanup instead of leaving several live
// runners until the aggregate test ends. Scenario leaves use the same checks.
async function scenario(t, label, options, check) {
    const cleanups = [];
    try {
        const h = gpuHarness({ after: (callback) => cleanups.push(callback) }, options);
        await check(h);
    } finally {
        for (const cleanup of cleanups.reverse()) await cleanup();
    }
    t.diagnostic(`${label}: assertions and owned cleanup passed`);
}

export async function checkQualificationMismatch(t, [field, value]) {
    await scenario(t, `qualification ${field}`, { qualificationDataProvider: () => [{ ...testQualification(), [field]: value }] }, async (h) => {
        await assert.rejects(h.run({ maxModelLen: 512 }), (error) => error.details?.admission?.reasonCode === 'vllm_mps_unqualified');
        assert.equal(h.started.length, 0);
    });
}

export async function checkGpuAdapter(t, [runnerId, modelId]) {
    const gpu = { ...TEST_GPU, totalBytes: 24 * GIB, freeBytes: 24 * GIB, usedBytes: 0 };
    const env = { ...MPS_ENV, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=16384M' };
    await scenario(t, `GPU ${runnerId}`, { gpu, env }, async (h) => {
        const params = runnerId === 'vllm' ? { maxModelLen: 512 } : {};
        assert.equal((await h.run(params, { runnerId, modelId })).accepted, true);
        const launch = await waitForLaunch(h);
        assert.deepEqual(Object.fromEntries(MPS_VARIABLES.map((key) => [key, launch.env[key]])), env);
        assert.equal(Object.hasOwn(launch.env, 'PRIVATE_AGENT_SECRET'), false);
        assert.equal(Object.keys(launch.env).filter((key) => key.startsWith('CUDA_MPS_')).length, 3);
    });
}

export async function checkCpuAdapter(t, [runnerId, modelId]) {
    await scenario(t, `CPU ${runnerId}`, { gpu: { available: false, state: 'absent', reason: 'No GPU.' }, profile: 'cpu', qualificationDataProvider: null }, async (h) => {
        assert.equal((await h.run({}, { runnerId, modelId })).accepted, true);
        const launch = await waitForLaunch(h);
        assert.equal(Object.keys(launch.env).some((key) => key.startsWith('CUDA_MPS_')), false);
    });
}

export async function checkInvalidMps(t, [index, env]) {
    await scenario(t, `invalid MPS vector ${index}`, { env }, async (h) => {
        await assert.rejects(h.run({ maxModelLen: 512 }), (error) => error.details?.admission?.reasonCode === 'gpu_budget_invalid' && !error.message.includes('secret-path'));
        assert.equal(h.started.length, 0);
    });
}
