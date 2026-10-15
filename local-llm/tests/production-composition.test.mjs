// V24: the agent's controller, built the way main.mjs builds it, reproduces what the container's limits did before the engine
// moved. The only stand-ins are for the machine (a file system, an nvidia-smi, the environment) and for the runner process and
// the downloads, which are not limits. Nothing replaces how the controller reads the container's limits: no `readBudget`, no
// `snapshot`, no `readHardwareSnapshot` and no qualification data are given.
//
// The negative control: with LOCAL_LLM_TEST_IDENTITY_LIMITS=1 the same controller is built over the engine's identity limits
// (a deployment that wired no limits at all), and every test below must fail. Run the file both ways.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { IDENTITY_LIMITS } from 'pworker/local-llm';

import { createAgentController } from '../src/engineOptions.mjs';
import { GIB, GPU_MODEL, GPU_TEST_LOCK, MIB, MPS_ENV } from './ploinkyGpuFixture.mjs';

const IDENTITY = process.env.LOCAL_LLM_TEST_IDENTITY_LIMITS === '1';
const KEY = 'k'.repeat(43);
const SMALL = 'qwen2.5-0.5b-instruct-q4_k_m';
const RTX_LINE = 'NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 4539, 1433, 595.91.07\n';
const RTX_DEVICE = '0x252010DE, 8.6, None, [N/A], [N/A], 44\n';

const enoent = (file) => Object.assign(new Error(`ENOENT: ${file}`), { code: 'ENOENT' });
const meminfo = (totalGiB, availableGiB) => `MemTotal: ${totalGiB * 1024 * 1024} kB\nMemAvailable: ${availableGiB * 1024 * 1024} kB\nSwapFree: 0 kB\nMemFree: 1 kB\nCached: 1 kB\n`;

// A machine: a file system whose files a test can change while the controller runs, and an nvidia-smi.
function machine({ files = {}, gpu = false } = {}) {
    const state = { files: { '/proc/meminfo': meminfo(64, 48), '/proc/self/status': 'Cpus_allowed_list:\t0-3\n', ...files }, smiCalls: 0, failedReads: {} };
    const fsApi = {
        readFileSync(file) {
            if (!Object.hasOwn(state.files, file)) throw enoent(file);
            const value = typeof state.files[file] === 'function' ? state.files[file]() : state.files[file];
            if (value instanceof Error) {
                state.failedReads[file] = (state.failedReads[file] ?? 0) + 1;
                throw value;
            }
            return value;
        },
        readdirSync(dir) { throw enoent(dir); },
        lstatSync(file) { throw enoent(file); },
    };
    const answers = gpu ? {
        '--query-gpu=name,memory.total,memory.used,memory.free,driver_version': RTX_LINE,
        '--query-gpu=pci.device_id,compute_cap,addressing_mode,utilization.gpu,power.draw,temperature.gpu': RTX_DEVICE,
        '--query-compute-apps=pid,process_name,used_memory': '',
    } : {};
    const execFileImpl = (command, args, options, callback) => {
        state.smiCalls += 1;
        const answer = answers[args[0]];
        if (answer === undefined) callback(Object.assign(new Error('absent'), { code: 'ENOENT' }), '', '');
        else callback(null, answer, '');
        return null;
    };
    return { state, fsApi, execFileImpl };
}

function box(t, { files, gpu = false, env = {}, extra = {} } = {}) {
    const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-production-')));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const weights = path.join(root, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const world = machine({ files, gpu });
    const started = [];
    const controller = createAgentController({
        env: { PATH: '/usr/bin', LOCAL_LLM_DATA_DIR: path.join(root, 'data'), LOCAL_LLM_RUN_ROOT: path.join(root, 'opt'), ...env },
        fsApi: world.fsApi,
        execFileImpl: world.execFileImpl,
        ...(IDENTITY ? { limits: IDENTITY_LIMITS } : {}),
        // Not limits: the downloads, the runner process and the clocks of a run.
        download: async () => ({ status: 'complete', path: weights, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }),
        downloadSnapshot: async () => ({ status: 'complete', bytesTransferred: 0 }),
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '{}' }),
        apiKeyFactory: () => KEY,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion || 'test', reason: null }),
        fileExists: () => true,
        sharedModelsRoot: null,
        dropCache: () => true,
        pollMs: 2,
        stopGraceMs: 20,
        unifiedGuardMs: 5,
        startRunner({ command, args, env: runnerEnv }) {
            let running = true;
            let finish;
            const handle = { pid: 7000 + started.length, command, args, env: runnerEnv, exited: new Promise((resolve) => { finish = resolve; }), get running() { return running; },
                async stop() { running = false; handle.stopped = 'SIGTERM'; finish({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                async kill() { running = false; handle.stopped = 'SIGKILL'; finish({ code: null, signal: 'SIGKILL', error: null }); return handle.exited; } };
            started.push(handle);
            return handle;
        },
        ...extra,
    });
    t.after(async () => { await controller.stop().catch(() => {}); await controller.drain().catch(() => {}); });
    return { controller, started, world };
}

async function until(predicate, label, timeoutMs = 4000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 2));
    }
    assert.fail(`not reached: ${label}`);
}

const cgroup = (maxBytes, currentBytes, extra = {}) => ({ '/sys/fs/cgroup/memory.max': `${maxBytes}\n`, '/sys/fs/cgroup/memory.current': `${currentBytes}\n`, ...extra });

test('V24.1 a finite memory.max refuses the model the container cannot hold, and says it is the container\'s limit', async (t) => {
    const limited = box(t, { files: cgroup(1 * GIB, 100 * MIB) });
    await assert.rejects(limited.controller.run({ modelId: SMALL, runnerId: 'llama.cpp', requestId: 'limit-refused-1' }), (error) => {
        const admission = error.details?.admission;
        return admission?.status === 'incompatible' && admission.warnings.includes('A container memory limit of 1.00 GiB applies.');
    });
    assert.equal(limited.started.length, 0, 'nothing was launched');
    // The same machine without the limit admits it.
    const open = box(t, { files: cgroup('max', 0) });
    assert.equal((await open.controller.run({ modelId: SMALL, runnerId: 'llama.cpp', requestId: 'limit-open-0001' })).accepted, true);
});

test('V24.2 the overview publishes the container budget: exact CPUs, the memory limit and who set them', async (t) => {
    const { controller } = box(t, { files: cgroup(8 * GIB, 1 * GIB, { '/sys/fs/cgroup/cpu.max': '150000 100000\n' }) });
    const overview = await controller.overview();
    assert.deepEqual(overview.limits.budget, { cpus: 1.5, memoryBytes: 8 * GIB, source: 'ploinky' });
    const plain = box(t, { files: cgroup('max', 0) });
    assert.equal(JSON.stringify(await plain.controller.overview()).includes('"budget"'), false, 'an unlimited container publishes none');
});

test('V24.3 the memory guard stops the runner after two unreadable samples of the container\'s memory use, and not after one', async (t) => {
    const current = '/sys/fs/cgroup/memory.current';
    const unreadable = () => Object.assign(new Error('unreadable'), { code: 'EIO' });
    // One unreadable sample, then readable again: the guard counts it and goes on.
    const once = box(t, { files: cgroup(16 * GIB, 2 * GIB) });
    await once.controller.run({ modelId: SMALL, runnerId: 'llama.cpp', requestId: 'guard-single-0001' });
    await until(() => once.started.length === 1 && once.controller.state.deployment?.phase === 'ready', 'the runner is ready');
    let reads = 0;
    once.world.state.files[current] = () => { reads += 1; return reads === 1 ? unreadable() : `${2 * GIB}\n`; };
    await until(() => reads >= 6, 'the guard went on sampling');
    assert.equal(once.world.state.failedReads[current], 1);
    assert.equal(once.started[0].stopped, undefined, 'one unreadable sample does not stop it');
    // Unreadable from then on (not absent): the guard cannot see the headroom and stops the process group.
    const h = box(t, { files: cgroup(16 * GIB, 2 * GIB) });
    await h.controller.run({ modelId: SMALL, runnerId: 'llama.cpp', requestId: 'guard-unreadable-1' });
    await until(() => h.started.length === 1 && h.controller.state.deployment?.phase === 'ready', 'the runner is ready');
    assert.equal(h.started[0].stopped, undefined, 'a readable container is not stopped');
    h.world.state.files[current] = unreadable;
    await until(() => h.started[0].stopped, 'the runner is stopped');
    assert.equal(h.started[0].stopped, 'SIGKILL', 'the process group is killed');
    assert.ok(h.world.state.failedReads[current] >= 2, `stopped after ${h.world.state.failedReads[current]} unreadable samples`);
    await until(() => h.controller.state.deployment?.phase === 'error', 'the deployment failed');
    assert.match(h.controller.state.deployment.error, /^stopped: budget_unreadable: /);
});

test('V24.4 a CPU quota caps the threads the runner is launched with', { skip: os.availableParallelism() < 4 && 'needs at least 4 CPUs for the cap to show' }, async (t) => {
    const capped = box(t, { files: cgroup('max', 0, { '/sys/fs/cgroup/cpu.max': '100000 100000\n' }) });
    await capped.controller.run({ modelId: SMALL, runnerId: 'llama.cpp', requestId: 'threads-capped-1' });
    await until(() => capped.started.length === 1, 'the runner is launched');
    const threads = (handle) => handle.args[handle.args.indexOf('--threads') + 1];
    assert.equal(threads(capped.started[0]), '1', 'one CPU of quota is one thread');
    const open = box(t, { files: cgroup('max', 0, { '/sys/fs/cgroup/cpu.max': 'max 100000\n' }) });
    await open.controller.run({ modelId: SMALL, runnerId: 'llama.cpp', requestId: 'threads-open-0001' });
    await until(() => open.started.length === 1, 'the runner is launched');
    assert.notEqual(threads(open.started[0]), '1', 'without a quota the threads follow the machine');
});

test('V24.5 vLLM under a GPU share is refused unless the reviewed qualification data holds its tuple', async (t) => {
    const share = { ...MPS_ENV, CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: '0=4096M' };
    const h = box(t, { gpu: true, files: cgroup('max', 0), env: { ...share, LOCAL_LLM_RUN_ROOT: undefined },
        extra: { seedCatalog: [GPU_MODEL], runnerLocks: GPU_TEST_LOCK, installer: { installable: () => true, describe: async () => ({ installed: true, runnable: true, version: 'test', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
            ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }), entryFor: (id) => GPU_TEST_LOCK.runners[id] || { id, version: 'test' }, pathsFor: () => ({ runDir: '/nonexistent/vllm' }) } } });
    await assert.rejects(h.controller.run({ modelId: GPU_MODEL.id, runnerId: 'vllm', params: { maxModelLen: 512 }, requestId: 'vllm-share-0001' }),
        (error) => error.details?.admission?.reasonCode === 'vllm_mps_unqualified');
    assert.equal(h.started.length, 0);
});
