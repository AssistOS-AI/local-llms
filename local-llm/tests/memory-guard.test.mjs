// The controller's host-memory guard (DS003): a deployment whose admission
// sets a RAM floor (vLLM with CPU offload) is watched while it loads and runs,
// and stopped with a clear error when available memory falls below the floor.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createController } from '../src/controller/deployments.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { DRAIN_RUNNER_GRACE_MS } from '../src/drainBudget.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SEED = loadSeedCatalog();

function harness(t, { ready = true, stopDelayMs = 0 } = {}) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-memguard-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const runDir = path.join(root, 'opt', 'runners', 'vllm', '0.30.0');
    const memory = { availableBytes: 40 * GIB, reads: 0 };
    const stops = [];
    const handles = [];
    const controller = createController({
        dataDir: path.join(root, 'data'),
        env: { PATH: '/usr/bin' },
        seedCatalog: SEED,
        stateStore: createStateStore({ dataDir: path.join(root, 'data') }),
        // Admission sees a roomy machine: 64 GiB of RAM, so the floor is 6.4 GiB.
        snapshot: async () => ({
            gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
            memory: { totalBytes: 64 * GIB, availableBytes: 48 * GIB },
            disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
        }),
        readMemory: () => { memory.reads += 1; return { totalBytes: 64 * GIB, availableBytes: memory.availableBytes }; },
        memoryGuardLoadMs: 5,
        memoryGuardReadyMs: 10,
        installer: {
            installable: (id) => id === 'vllm',
            describe: async () => ({ installed: true, runnable: true, version: '0.30.0', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
            ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }),
            entryFor: () => ({ id: 'vllm', version: '0.30.0' }),
            pathsFor: () => ({ runDir }),
        },
        detectRunner: async () => ({ installed: true, version: '0.30.0', reason: null }),
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }),
        downloadSnapshot: async () => ({ status: 'complete', bytesTransferred: 0 }),
        shmDir: path.join(root, 'shm'),
        startRunner({ command, args, env }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 4242 + handles.length, command, args, env,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop(options = {}) {
                    stops.push(options);
                    // A runner that takes a while to exit after SIGTERM.
                    if (stopDelayMs) await new Promise((resolve) => setTimeout(resolve, stopDelayMs));
                    running = false;
                    resolveExit({ code: null, signal: 'SIGTERM', error: null });
                    return handle.exited;
                },
            };
            handles.push(handle);
            return handle;
        },
        fetchImpl: async () => (ready ? { ok: true, status: 200, json: async () => ({}) } : { ok: false, status: 503, json: async () => ({}) }),
        pollMs: 2,
        stopGraceMs: 10_000,
    });
    t.after(() => controller.stop().catch(() => {}));
    return { controller, memory, stops, handles };
}

async function until(check, what) {
    for (let index = 0; index < 1000; index += 1) {
        if (check()) return;
        await new Promise((resolve) => setTimeout(resolve, 2));
    }
    assert.fail(`timed out waiting for ${what}`);
}

const offloadRun = (controller, requestId) => controller.run({ modelId: 'gpt-oss-20b', runnerId: 'vllm', requestId, params: { cpuOffloadGb: 10 } });

test('an offload deployment is stopped when host memory falls below its floor while it loads', async (t) => {
    const { controller, memory, stops, handles } = harness(t, { ready: false });
    await offloadRun(controller, 'request-guard-load');
    assert.equal(controller.state.deployment.admission.estimate.ramFloorBytes, Math.round(6.4 * GIB));
    await until(() => handles.length === 1, 'the runner launch');
    memory.availableBytes = 2 * GIB;
    await until(() => controller.state.deployment.phase === 'error', 'the guard stop');
    assert.equal(controller.state.deployment.error, 'stopped: host memory below the floor (2.0 GiB available, 6.4 GiB required)');
    assert.equal(handles[0].running, false);
    // A quick stop: SIGTERM, then SIGKILL within the drain's grace, not the 10 s Stop grace.
    assert.ok(stops.length >= 1 && stops.every((options) => options.graceMs <= DRAIN_RUNNER_GRACE_MS), JSON.stringify(stops));
    assert.equal(controller.state.deployment.runner, null);
});

test('a ready offload deployment is stopped when host memory falls below its floor', async (t) => {
    const { controller, memory, stops, handles } = harness(t);
    await offloadRun(controller, 'request-guard-ready');
    await until(() => controller.state.deployment.phase === 'ready', 'ready');
    const readsWhenReady = memory.reads;
    await until(() => memory.reads > readsWhenReady + 1, 'sampling while ready');
    memory.availableBytes = 5 * GIB;
    await until(() => controller.state.deployment.phase === 'error', 'the guard stop');
    assert.equal(controller.state.deployment.error, 'stopped: host memory below the floor (5.0 GiB available, 6.4 GiB required)');
    assert.equal(handles[0].running, false);
    assert.ok(stops.every((options) => options.graceMs <= DRAIN_RUNNER_GRACE_MS));
});

test('memory above the floor leaves the deployment alone, and sampling ends with the runner', async (t) => {
    const { controller, memory, handles } = harness(t);
    await offloadRun(controller, 'request-guard-fine');
    await until(() => controller.state.deployment.phase === 'ready', 'ready');
    await until(() => memory.reads >= 5, 'several samples');
    assert.equal(controller.state.deployment.phase, 'ready');
    await controller.stop();
    assert.equal(handles[0].running, false);
    const after = memory.reads;
    await new Promise((resolve) => setTimeout(resolve, 60));
    assert.ok(memory.reads <= after + 1, `sampling continued after Stop (${after} -> ${memory.reads})`);
});

test('a deployment without a RAM floor is not sampled', async (t) => {
    const { controller, memory } = harness(t);
    await controller.run({ modelId: 'qwen3-4b-awq', runnerId: 'vllm', requestId: 'request-guard-none' });
    await until(() => controller.state.deployment.phase === 'ready', 'ready');
    await new Promise((resolve) => setTimeout(resolve, 40));
    assert.equal(controller.state.deployment.admission.estimate.ramFloorBytes, undefined);
    assert.equal(memory.reads, 0);
});

test('a drain that begins while the guard is stopping the runner finishes cleanly, with no unhandled rejection', async (t) => {
    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    t.after(() => process.off('unhandledRejection', onUnhandled));
    const { controller, memory, stops } = harness(t, { stopDelayMs: 60 });
    await offloadRun(controller, 'request-guard-drain');
    await until(() => controller.state.deployment.phase === 'ready', 'ready');
    memory.availableBytes = 1 * GIB;
    await until(() => stops.length >= 1, 'the guard stop');
    // SIGTERM for the agent arrives while the runner is still exiting.
    await controller.drain();
    await new Promise((resolve) => setTimeout(resolve, 120));
    assert.deepEqual(unhandled, []);
    assert.equal(controller.state.deployment.phase, 'idle');
});

test('a Stop pressed while the guard is stopping the runner leaves the deployment idle', async (t) => {
    const { controller, memory, stops } = harness(t, { stopDelayMs: 60 });
    await offloadRun(controller, 'request-guard-user-stop');
    await until(() => controller.state.deployment.phase === 'ready', 'ready');
    memory.availableBytes = 1 * GIB;
    await until(() => stops.length >= 1, 'the guard stop');
    await controller.stop();
    await new Promise((resolve) => setTimeout(resolve, 120));
    assert.equal(controller.state.deployment.phase, 'idle');
    assert.equal(controller.state.deployment.error, null);
});
