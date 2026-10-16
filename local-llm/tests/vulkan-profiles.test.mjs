import assert from 'node:assert/strict';
import childProcess from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { VENUS } from 'pworker/local-llm/internal/controller/profiles.mjs';

import { cpuFallbackExplanation, getRunner, loadSeedCatalog, readVulkanGpu, vulkanContractOf } from '../src/engineOptions.mjs';
import { admit, readSnapshot } from '../src/hardwareLimits.mjs';

import { mpsRunnerEnvironment } from '../src/controller/ploinkyBudget.mjs';
import { hardwareCardsHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';
import { cpuFs, VULKAN_KEYS } from './overview-cpu-scenarios.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const contract = vulkanContractOf(VULKAN_KEYS);
const runner = getRunner('llama.cpp-vulkan');
const fixture = (name) => fs.readFileSync(new URL(`./fixtures/vulkan/${name}`, import.meta.url), 'utf8');
const seed = loadSeedCatalog();
const small = seed.find((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m');

const enoent = () => Object.assign(new Error('absent'), { code: 'ENOENT' });

function probe({ summary = fixture('venus-summary.txt'), devices = fixture('venus-devices.txt'), failure = null, listFailure = null } = {}) {
    const calls = [];
    const execFileImpl = (command, args, options, callback) => {
        calls.push({ command, args, options });
        if (command.endsWith('/nvidia-smi')) callback(enoent(), '', '');
        else if (command === contract.vulkaninfo) callback(failure, failure ? '' : summary, failure?.stderr || '');
        else callback(listFailure, listFailure ? '' : devices, '');
        return null;
    };
    return { calls, execFileImpl };
}
function renderFs() {
    return { ...cpuFs(), readdirSync: () => ['renderD128'], lstatSync: () => ({ isCharacterDevice: () => true }) };
}
async function snapshot(options = {}, vulkan = contract, fsApi = renderFs()) {
    const host = probe(options);
    const snap = await readSnapshot({ dataDir: '/tmp', vulkan, fsApi, execFileImpl: host.execFileImpl,
        apu: options.apu === true,
        env: { PLOINKY_GPU_STATUS: 'attached' }, statfs: async () => ({ bsize: 4096, blocks: 1000000, bavail: 999000 }) });
    return { snap, calls: host.calls };
}
async function gpu(driver = 'venus') {
    return readVulkanGpu({ contract, execFileImpl: probe(driver === 'radv' ? { summary: fixture('radv-summary.txt'), devices: fixture('radv-devices.txt') } : {}).execFileImpl });
}
function roomy(gpu) { return { gpu, memory: { totalBytes: 6 * GIB, availableBytes: 5 * GIB }, disk: { freeBytes: 100 * GIB } }; }
async function verdict(profile, { model = small, params = {}, snap = null, gpuBudget = { state: 'absent' } } = {}) {
    return admit({ runner, model, source: model.sources.gguf, params: runner.normalizeParams(params, { model, profile }),
        snapshot: snap || roomy(await gpu(profile === 'vulkan' ? 'radv' : 'venus')), profile, gpuBudget });
}

test('admission: GPU share is refused on vulkan and venus', async () => {
    for (const profile of ['vulkan', 'venus']) {
        const result = await verdict(profile, { gpuBudget: { state: 'known', environment: {} } });
        assert.equal(result.reasonCode, 'gpu_share_unsupported'); assert.equal(result.status, 'incompatible');
        assert.deepEqual(mpsRunnerEnvironment({ CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: 'broken' }, profile), {});
    }
});

test('dashboard: Vulkan and Venus cards; CPU wording names only observed backends', async () => {
    const hardware = roomy(await gpu());
    assert.match(hardwareCardsHtml(hardware, null, { profile: 'venus', limits: { deviceCapBytes: VENUS.deviceCapBytes } }), /Headroom unknown/);
    assert.match(hardwareCardsHtml(hardware, null, { profile: 'vulkan' }), /unverified on AMD hardware/);
    assert.match(hardwareCardsHtml(hardware, null, { profile: 'cpu', decision: { explanation: cpuFallbackExplanation({ cause: 'absent' }) } }), /Runs on the CPU: no usable GPU was observed/);
    const fallback = hardwareCardsHtml(hardware, null, { profile: 'cpu', decision: { explanation: cpuFallbackExplanation({ cause: 'unusable', gpuBackend: 'vulkan' }) } });
    assert.match(fallback, /the Vulkan GPU cannot be used/); assert.doesNotMatch(fallback, /NVIDIA/);
});

