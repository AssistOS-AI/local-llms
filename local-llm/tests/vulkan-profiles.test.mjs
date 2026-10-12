import assert from 'node:assert/strict';
import childProcess from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { readGpu, readSnapshot, runHardwareQuery } from '../src/controller/hardware.mjs';
import { mpsRunnerEnvironment } from '../src/controller/ploinkyBudget.mjs';
import { decideProfile, VENUS } from '../src/controller/profiles.mjs';
import { parseRunnerReport } from '../src/controller/runnerProcess.mjs';
import { hasRenderNode, parseListDevices, readVulkanGpu, VULKAN_ALL_ICD_FILES, vulkanContractOf, vulkanProbeEnv } from '../src/controller/vulkan.mjs';
import { getRunner, RUNNERS, runnerSummary } from '../src/runners/index.mjs';
import { hardwareCardsHtml } from '../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard-view.js';
import { cpuFs, VULKAN_KEYS } from './overview-cpu-scenarios.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const contract = vulkanContractOf(VULKAN_KEYS);
const runner = getRunner('llama.cpp-vulkan');
const fixture = (name) => fs.readFileSync(new URL(`./fixtures/vulkan/${name}`, import.meta.url), 'utf8');
const seed = loadSeedCatalog();
const small = seed.find((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const large = seed.find((model) => model.id === 'qwen3-4b-instruct-2507-q4_k_m');
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

test('contract: valid; missing; wrong version; traversal; relative; server path mismatch', () => {
    assert.ok(Object.isFrozen(contract));
    assert.equal(vulkanContractOf(null), null);
    for (const change of [{ vulkan_contract: '2' }, { vulkan_llama_cpp: 'latest' }, { vulkan_mesa: '' },
        { vulkan_icd_radv: '/opt/local-llm/vulkan/icd.d/../../../etc/x.json' }, { vulkan_icd_venus: 'relative.json' },
        { vulkan_llama_cpp_server: '/opt/llama.cpp-vulkan/../llama-server' }]) assert.equal(vulkanContractOf({ ...VULKAN_KEYS, ...change }), null);
    for (const key of Object.keys(VULKAN_KEYS)) {
        const missing = { ...VULKAN_KEYS }; delete missing[key]; assert.equal(vulkanContractOf(missing), null, key);
    }
});
test('snapshot: no render node keeps the absent GPU and runs nvidia-smi only', async () => {
    const { snap, calls } = await snapshot({}, contract, cpuFs());
    assert.equal(snap.gpu.state, 'absent'); assert.equal(calls.length, 1);
    assert.equal('backend' in snap.gpu, false);
    assert.equal(hasRenderNode({ fsApi: { readdirSync: () => ['renderD128'], lstatSync: () => ({ isCharacterDevice: () => false }) } }), false);
});
test('snapshot: render node without a Vulkan contract is unusable', async () => {
    const { snap, calls } = await snapshot({}, null);
    assert.equal(snap.gpu.state, 'unusable'); assert.match(snap.gpu.reason, /update the local-llm image/); assert.equal(calls.length, 1);
});
test('snapshot: Venus fixture decides venus', async () => {
    const { snap } = await snapshot();
    assert.equal(snap.gpu.driver, 'venus'); assert.equal(snap.gpu.icdFile, contract.vulkan_icd_venus);
    assert.equal(snap.gpu.name, 'Virtio-GPU Venus (Apple M1)');
    assert.equal(snap.gpu.totalBytes, 16384 * MIB); assert.equal(snap.gpu.freeBytes, 16384 * MIB);
    assert.equal(snap.gpu.vendorId, '0x106b'); assert.equal(snap.gpu.deviceType, 'PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU');
    assert.equal(decideProfile({ snapshot: snap }).profile, 'venus');
});
test('snapshot: RADV discrete decides vulkan; RADV integrated is unusable', async () => {
    const { snap } = await snapshot({ summary: fixture('radv-summary.txt'), devices: fixture('radv-devices.txt') });
    assert.equal(decideProfile({ snapshot: snap }).profile, 'vulkan');
    assert.equal(snap.gpu.name, 'AMD Radeon (TM) RX 7900 XTX (RADV NAVI31)');
    const integrated = await snapshot({ summary: fixture('radv-integrated-summary.txt') });
    assert.equal(integrated.snap.gpu.state, 'unusable');
});
test('snapshot: two accepted devices, llvmpipe present, (none) are unusable', async () => {
    const software = fixture('venus-summary.txt') + '\nGPU1:\n deviceName = llvmpipe\n deviceType = PHYSICAL_DEVICE_TYPE_CPU\n vendorID = 0x10005\n driverID = DRIVER_ID_MESA_LLVMPIPE\n';
    for (const options of [{ summary: fixture('two-summary.txt') }, { summary: software }, { devices: fixture('none-devices.txt') },
        { summary: fixture('venus-summary.txt').replace('DRIVER_ID_MESA_VENUS', 'DRIVER_ID_INTEL_OPEN_SOURCE_MESA') }]) {
        assert.equal((await snapshot(options)).snap.gpu.state, 'unusable');
    }
});
test('snapshot: incompatible driver is unusable; timeout and malformed are unreadable then cpu at 60 s', async () => {
    const failure = Object.assign(new Error('failed'), { stderr: fixture('incompatible-driver.txt') });
    assert.equal((await snapshot({ failure })).snap.gpu.state, 'unusable');
    for (const options of [{ failure: Object.assign(new Error('timeout'), { killed: true }) }, { summary: fixture('malformed-summary.txt') },
        { listFailure: new Error('list failed') }]) {
        const { snap } = await snapshot(options);
        assert.equal(snap.gpu.state, 'unreadable');
        const first = decideProfile({ snapshot: snap, nowMs: 0 });
        assert.equal(first.profile, null);
        assert.equal(decideProfile({ snapshot: snap, nowMs: 60000, unreadable: first.unreadable }).profile, 'cpu');
    }
});
test('snapshot: list-devices name mismatch, zero total and oversize values', async () => {
    for (const [devices, state] of [
        [fixture('venus-devices.txt').replace('Apple M1', 'Other M1'), 'unusable'],
        ['Available devices:\n  Vulkan0: Virtio-GPU Venus (Apple M1) (0 MiB, 0 MiB free)\n', 'unusable'],
        ['Available devices:\n  Vulkan0: Virtio-GPU Venus (Apple M1) (9007199254740992 MiB, 0 MiB free)\n', 'unreadable'],
    ]) assert.equal((await snapshot({ devices })).snap.gpu.state, state);
    assert.throws(() => parseListDevices('Vulkan0: no header'), /header/);
    assert.throws(() => parseListDevices('Available devices:\n bad row'), /Malformed/);
});
test('probe exec options: explicit env without ambient Vulkan variables; nvidia-smi options unchanged', async () => {
    const { calls } = await snapshot();
    assert.deepEqual(calls[0].options, { timeout: 10000, killSignal: 'SIGKILL', encoding: 'utf8' });
    assert.deepEqual(calls[1].options.env, vulkanProbeEnv(VULKAN_ALL_ICD_FILES));
    assert.deepEqual(calls[2].options.env, vulkanProbeEnv(contract.vulkan_icd_venus));
    assert.deepEqual(Object.keys(calls[1].options.env).sort(), ['PATH', 'HOME', 'LANG', 'VK_DRIVER_FILES', 'VK_LOADER_LAYERS_DISABLE', 'MESA_SHADER_CACHE_DISABLE'].sort());
});
test('probe env: vulkanProbeEnv rejects an empty or relative ICD list; detect uses VULKAN_ALL_ICD_FILES', () => {
    for (const value of ['', 'relative.json', '/a.so', '/a.json:', ':']) assert.throws(() => vulkanProbeEnv(value));
    let seen;
    runner.detect({ spawnSync(command, args, options) { seen = { command, args, options }; return { status: 0, stdout: 'build 11159,', stderr: '' }; } });
    assert.equal(seen.command, '/opt/llama.cpp-vulkan/llama-server');
    assert.deepEqual(seen.options.env, vulkanProbeEnv(VULKAN_ALL_ICD_FILES));
});
test('probe cancellation kills and reaps the child, returning AbortError', async () => {
    const stop = new AbortController();
    const child = new EventEmitter(); Object.assign(child, { pid: 123, exitCode: null, signalCode: null });
    let callback; let killed;
    child.kill = (signal) => { killed = signal; child.signalCode = signal; queueMicrotask(() => { child.emit('exit'); callback(new Error('killed'), '', ''); }); };
    const pending = runHardwareQuery((command, args, options, cb) => { callback = cb; return child; }, '/probe', [], { signal: stop.signal });
    stop.abort(); const result = await pending;
    assert.equal(killed, 'SIGKILL'); assert.equal(result.aborted, true); assert.equal(result.error.name, 'AbortError');
});
test('decide: arm64 capability contract does not block vulkan or venus', async () => {
    for (const driver of ['radv', 'venus']) assert.equal(decideProfile({ snapshot: roomy(await gpu(driver)), capabilities: ['12.1'] }).profile, driver === 'radv' ? 'vulkan' : 'venus');
});
test('admission: llama.cpp, ik_llama.cpp, llama.cpp-cpu, ollama, vllm, tabbyapi, lmstudio are incompatible on vulkan and venus', async () => {
    for (const profile of ['vulkan', 'venus']) for (const definition of Object.values(RUNNERS).filter((item) => item !== runner)) {
        const result = admit({ runner: definition, model: small, source: small.sources.gguf, params: {}, snapshot: roomy(await gpu(profile === 'vulkan' ? 'radv' : 'venus')), profile, gpuBudget: { state: 'absent' } });
        assert.equal(result.status, 'incompatible', definition.id); assert.match(result.reason, /use llama.cpp \(Vulkan build\)/);
    }
});
test('admission: GPU share is refused on vulkan and venus', async () => {
    for (const profile of ['vulkan', 'venus']) {
        const result = await verdict(profile, { gpuBudget: { state: 'known', environment: {} } });
        assert.equal(result.reasonCode, 'gpu_share_unsupported'); assert.equal(result.status, 'incompatible');
        assert.deepEqual(mpsRunnerEnvironment({ CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: 'broken' }, profile), {});
    }
});
test('venus: qwen3-4b is refused by the cap; qwen2.5-0.5b is admitted; free below need is insufficient-now; VM reserve applies', async () => {
    assert.match((await verdict('venus', { model: large })).reason, /Venus admits at most/);
    const accepted = await verdict('venus'); assert.equal(accepted.status, 'ok'); assert.equal(accepted.estimate.deviceCapBytes, VENUS.deviceCapBytes);
    const device = await gpu();
    assert.equal((await verdict('venus', { snap: roomy({ ...device, freeBytes: 0 }) })).status, 'insufficient-now');
    const snap = roomy(device); snap.memory = { totalBytes: 2 * GIB, availableBytes: 2 * GIB };
    assert.equal((await verdict('venus', { snap })).status, 'incompatible');
    assert.equal((await verdict('venus', { snap: { ...roomy(device), cgroupMemory: { maxBytes: 2 * GIB, currentBytes: 0 } } })).status, 'incompatible');
});
test('venus: ctxSize 131072 with parallel 4 on qwen2.5-0.5b is refused by the cap, not by normalization', async () => {
    const result = await verdict('venus', { params: { ctxSize: 131072, parallel: 4 } });
    assert.equal(result.status, 'incompatible'); assert.match(result.reason, /Venus admits at most/);
});
test('summaries: profileUnsupportedReason on vulkan and venus', () => {
    for (const profile of ['vulkan', 'venus']) for (const definition of Object.values(RUNNERS)) {
        const summary = runnerSummary(definition, profile);
        if (definition === runner) assert.equal('profileUnsupportedReason' in summary, false);
        else assert.match(summary.profileUnsupportedReason, /Vulkan build/);
    }
});
test('report: Vulkan0 buffers, device name with parentheses, offload count; CUDA report unchanged', () => {
    const lines = fixture('runner.log').trim().split('\n').map((line) => ({ line }));
    for (const profile of ['vulkan', 'venus']) {
        const report = parseRunnerReport(lines, { profile });
        assert.deepEqual(report, { modelMiB: 325.02, kvMiB: 48, computeMiB: 215, offloaded: { layers: 25, of: 25 },
            device: 'Vulkan0 (AMD Radeon (TM) RX 7900 XTX (RADV NAVI31))', totalMiB: 588 });
    }
    const cuda = [{ line: 'using device CUDA0 (NVIDIA GPU)' }, { line: 'CUDA0 model buffer size = 325.02 MiB' }];
    assert.equal(parseRunnerReport(cuda).device, 'CUDA0 (NVIDIA GPU)');
    assert.equal(parseRunnerReport(cuda).modelMiB, 325.02);
    assert.equal(parseRunnerReport(lines).modelMiB, null);
});
test('dashboard: Vulkan and Venus cards; CPU card text unchanged without gpuBackend', async () => {
    const hardware = roomy(await gpu());
    assert.match(hardwareCardsHtml(hardware, null, { profile: 'venus', limits: { deviceCapBytes: VENUS.deviceCapBytes } }), /Headroom unknown/);
    assert.match(hardwareCardsHtml(hardware, null, { profile: 'vulkan' }), /unverified on AMD hardware/);
    assert.match(hardwareCardsHtml(hardware, null, { profile: 'cpu', decision: { cause: 'absent' } }), /No NVIDIA GPU is used: no NVIDIA GPU is attached/);
    const fallback = hardwareCardsHtml(hardware, null, { profile: 'cpu', decision: { cause: 'unusable', gpuBackend: 'vulkan' } });
    assert.match(fallback, /the Vulkan GPU cannot be used/); assert.doesNotMatch(fallback, /NVIDIA/);
});

function controllerHarness(t, { driver = 'venus', defaultReader = null, imageContract = { architecture: 'arm64', gpu_compute_capabilities: '12.1', ...VULKAN_KEYS }, env = {} } = {}) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-vulkan-test-'));
    const handles = []; const memory = { totalBytes: 6 * GIB, availableBytes: 5 * GIB }; let snaps;
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin', ...env }, seedCatalog: seed, imageContract, hostArch: 'arm64', fileExists: () => true,
        sharedModelsRoot: null, detectRunner: () => ({ installed: true, version: 'b11159', reason: null }),
        ...(defaultReader ? { readHardwareSnapshot: defaultReader } : { snapshot: async () => (snaps || roomy(await gpu(driver))) }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async () => ({ status: 'complete', path: path.join(dataDir, 'fixture.gguf'), bytesTransferred: 0 }),
        startRunner({ command, args, env }) {
            let running = true; let resolveExit;
            const handle = { command, args, env, pid: 4242, exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM' }); return handle.exited; },
                async kill() { return handle.stop(); } };
            handles.push(handle); return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }), pollMs: 1,
        readMemory: () => memory, readPressure: () => 0, readBudget: () => null, unifiedGuardMs: 5,
    });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { recursive: true, force: true }); });
    return { controller, handles, dataDir, memory, setSnapshot(value) { snaps = value; } };
}
async function waitFor(check) {
    for (let i = 0; i < 500; i += 1) { if (await check()) return; await new Promise((resolve) => setTimeout(resolve, 5)); }
    assert.fail('Timed out waiting for controller state');
}
test('controller: production default snapshot (none injected) passes vulkanContractOf(imageContract) to readHardwareSnapshot', async (t) => {
    const calls = [];
    const h = controllerHarness(t, { defaultReader: async (options) => { calls.push(options); return roomy(await gpu()); } });
    const view = await h.controller.overview();
    assert.equal(view.profile, 'venus'); assert.deepEqual(calls[0].vulkan, contract);
    assert.equal(calls[0].dataDir, h.dataDir); assert.ok(calls[0].signal instanceof AbortSignal); assert.equal(calls[0].env.PATH, '/usr/bin');
    assert.equal(view.profileDecision.gpuBackend, 'vulkan'); assert.equal(view.limits.deviceCapBytes, VENUS.deviceCapBytes);
    assert.ok(view.models.some((model) => model.id === small.id));
    const old = controllerHarness(t, { imageContract: { architecture: 'arm64' }, defaultReader: async (options) => {
        assert.equal(options.vulkan, null); return (await snapshot({}, options.vulkan)).snap;
    } });
    assert.equal((await old.controller.overview()).profile, 'cpu');
});
test('controller: production default snapshot uses fs and execFile defaults when fsApi and execFileImpl are undefined', async (t) => {
    const host = probe();
    const read = fs.readFileSync; const readdir = fs.readdirSync; const lstat = fs.lstatSync;
    const facts = renderFs();
    t.mock.method(fs, 'readFileSync', (file, ...args) => String(file).startsWith('/proc/') || String(file).startsWith('/sys/') ? facts.readFileSync(file) : read(file, ...args));
    t.mock.method(fs, 'readdirSync', (file, ...args) => file === '/dev/dri' ? facts.readdirSync(file) : readdir(file, ...args));
    t.mock.method(fs, 'lstatSync', (file, ...args) => file === '/dev/dri/renderD128' ? facts.lstatSync(file) : lstat(file, ...args));
    t.mock.method(childProcess, 'execFile', host.execFileImpl); syncBuiltinESMExports();
    t.after(() => { t.mock.restoreAll(); syncBuiltinESMExports(); });
    const h = controllerHarness(t, { defaultReader: async (options) => readSnapshot({ ...options, fsApi: undefined, execFileImpl: undefined }) });
    assert.equal((await h.controller.overview()).profile, 'venus'); assert.equal(host.calls.length, 3);
});
test('controller: accepted Venus Run reaches launch with the Venus ICD and starts the pooled guard', async (t) => {
    const h = controllerHarness(t);
    const accepted = await h.controller.run({ modelId: small.id, runnerId: runner.id, requestId: 'venus-run-0001' });
    assert.equal(accepted.accepted, true);
    await waitFor(() => h.controller.state.deployment?.phase === 'ready');
    assert.equal(h.handles[0].env.VK_DRIVER_FILES, contract.vulkan_icd_venus);
    await waitFor(async () => (await h.controller.status()).memoryGuard?.samples > 0);
    const duplicate = await h.controller.run({ modelId: small.id, runnerId: runner.id, requestId: 'venus-run-0001' });
    assert.equal(duplicate.duplicate, true);
    h.memory.availableBytes = 0;
    await waitFor(() => h.controller.state.deployment?.phase === 'error');
    assert.equal(h.handles[0].running, false);
});
test('controller: accepted Vulkan Run with the arm64 image contract', async (t) => {
    const h = controllerHarness(t, { driver: 'radv' });
    const accepted = await h.controller.run({ modelId: small.id, runnerId: runner.id, requestId: 'vulkan-run-0001' });
    assert.equal(accepted.accepted, true);
    await waitFor(() => h.controller.state.deployment?.phase === 'ready');
    assert.equal(h.controller.state.deployment.profile, 'vulkan'); assert.equal(h.handles[0].env.VK_DRIVER_FILES, contract.vulkan_icd_radv);
    assert.ok(h.handles[0].args.includes('Vulkan0'));
    assert.match(h.controller.state.deployment.admission.warnings.join(' '), /AMD support is unit-tested only/);
    h.setSnapshot({ ...roomy(await gpu('radv')), gpu: { available: false, state: 'absent', reason: 'render node gone' } });
    await h.controller.stop();
    await assert.rejects(h.controller.run({ modelId: small.id, runnerId: runner.id, requestId: 'vulkan-gone-0001' }), /restart local-llm/);
});
test('runner env: exact keys, Mesa cache outside /data, no LD_LIBRARY_PATH or GGML_VK_VISIBLE_DEVICES', async (t) => {
    const h = controllerHarness(t, { env: { SECRET_SENTINEL: 'test-only', VK_DRIVER_FILES: '/evil.json', GGML_VK_VISIBLE_DEVICES: 'all', LOCAL_LLM_RUN_ROOT: '/opt/test-runners' } });
    await h.controller.run({ modelId: small.id, runnerId: runner.id, requestId: 'venus-env-0001' });
    await waitFor(() => h.handles.length === 1);
    assert.deepEqual(Object.keys(h.handles[0].env).sort(), ['PATH', 'HOME', 'LANG', 'CUDA_CACHE_PATH', 'VK_DRIVER_FILES', 'VK_LOADER_LAYERS_DISABLE', 'MESA_SHADER_CACHE_DIR'].sort());
    assert.equal(h.handles[0].env.MESA_SHADER_CACHE_DIR, '/opt/test-runners/.mesa-cache');
    assert.equal(h.handles[0].env.VK_DRIVER_FILES, contract.vulkan_icd_venus);
    const args = h.handles[0].args;
    for (const [flag, value] of [['--device', 'Vulkan0'], ['-ngl', '999'], ['--flash-attn', 'auto'], ['-ctk', 'f16'], ['-ctv', 'f16'], ['-b', '512'], ['-ub', '512'], ['--cache-ram', '256']]) assert.equal(args[args.indexOf(flag) + 1], value, flag);
});
test('compat: old agent readGpu with an attached render node commits cpu absent', async () => {
    const result = await readGpu({ execFileImpl: probe().execFileImpl, env: { PLOINKY_GPU_STATUS: 'attached' } });
    const decision = decideProfile({ snapshot: { gpu: result } });
    assert.equal(decision.profile, 'cpu'); assert.equal(decision.cause, 'absent');
});
test('compat: new agent with an old image is unusable then cpu', async () => {
    const { snap } = await snapshot({}, vulkanContractOf({ architecture: 'arm64' }));
    assert.equal(snap.gpu.state, 'unusable'); assert.equal(decideProfile({ snapshot: snap }).profile, 'cpu');
});
test('compat: no render node with a new image keeps cpu absent', async () => {
    const { snap } = await snapshot({}, contract, cpuFs());
    assert.equal(decideProfile({ snapshot: snap }).profile, 'cpu'); assert.equal(decideProfile({ snapshot: snap }).cause, 'absent');
});
