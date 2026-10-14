// Test seams for the registry service and its credentials (DS002): a controller on a dedicated 6 GiB GPU with its
// hardware, downloads and runners injected, the admin tools driven through the agent's own handler map, and the
// constants the registry tests share. Nothing here is used by the agent itself.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { controllerHandlers } from '../src/controlHandlers.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { DownloadError } from '../src/controller/downloader.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { handleTool } from '../tools/local_llm_tool.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
export const SEED = loadSeedCatalog();
export const RUN = Object.freeze({ modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' });
// Two valid credentials keys (64 lowercase hex), standing in for two workspaces' generated secrets.
export const KEY = 'a1'.repeat(32);
export const OTHER_KEY = '5e'.repeat(32);
export const ADMIN = Object.freeze({ user: Object.freeze({ roles: Object.freeze(['admin']), email: 'admin@example.com' }) });

export function deferred() {
    let resolve;
    let reject;
    const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
    return { promise, resolve, reject };
}

export async function until(predicate, timeoutMs = 2000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
}

export function tempDir(t, prefix = 'local-llm-registries-') {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

function snapshot() {
    return {
        gpu: { available: true, name: 'Test GPU', driverVersion: '595.91.07', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
        disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
        cpus: 20,
    };
}

function fakeRunners() {
    const started = [];
    return {
        started,
        startRunner({ command, args, env, log }) {
            const exit = deferred();
            let running = true;
            const handle = {
                pid: 4242 + started.length, command, args, env, exited: exit.promise,
                get running() { return running; },
                async stop() {
                    if (running) { running = false; exit.resolve({ code: 0, signal: 'SIGTERM', error: null }); }
                    return exit.promise;
                },
            };
            log.append('stdout', 'load_tensors: offloaded 25/25 layers to GPU');
            started.push(handle);
            return handle;
        },
    };
}

/**
 * A controller over `dataDir` (a new one when absent). `credentialsKey` is the explicit option agentProcess.mjs
 * passes; `env` is the controller's environment; `resolveHf` may hold a pin open; `extra` overrides any option.
 */
export function registryHarness(t, {
    dataDir = tempDir(t),
    credentialsKey = KEY,
    env = { PATH: '/usr/bin' },
    initialState = null,
    resolveHf = null,
    downloads = ['complete'],
    extra = {},
} = {}) {
    const stateStore = createStateStore({ dataDir });
    if (initialState) stateStore.save(initialState);
    const runners = fakeRunners();
    const calls = { download: [], remove: [] };
    const controller = createController({
        dataDir,
        env,
        credentialsKey,
        seedCatalog: SEED,
        stateStore,
        snapshot: async () => snapshot(),
        download: ({ artifact, signal }) => {
            const entry = deferred();
            calls.download.push({ artifact, entry });
            signal?.addEventListener('abort', () => entry.reject(new DownloadError('ABORTED', 'aborted')), { once: true });
            // As the real downloader does, a completed download says where its bytes came from (logged as provenance).
            if (downloads.shift() === 'complete') {
                entry.resolve({ status: 'complete', path: '/data/models/x.gguf', bytesTransferred: 12,
                    provenance: { file: artifact.file, source: `https://huggingface.co/${artifact.repo}`, method: 'download', bytes: 12 } });
            }
            return entry.promise;
        },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async (options) => { calls.remove.push(options); return 0; },
        startRunner: runners.startRunner,
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
        ...(resolveHf ? { resolveHf } : {}),
        ...extra,
    });
    const handlers = controllerHandlers(controller);
    // As the control socket carries it: JSON both ways.
    const call = async (op, args) => JSON.parse(JSON.stringify(await handlers[op](JSON.parse(JSON.stringify(args)))));
    const tool = (name, input, authInfo = ADMIN) => handleTool(name, input, { authInfo, call });
    return { controller, dataDir, runners, calls, tool, stateStore };
}

/** Every regular file under `dir`, recursively. */
export function filesUnder(dir) {
    const found = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) found.push(...filesUnder(full));
        else if (entry.isFile()) found.push(full);
    }
    return found;
}

/** The code of a rejected promise, or a failure naming what resolved. */
export async function codeOf(promise) {
    try {
        await promise;
    } catch (error) {
        return error?.code;
    }
    return 'resolved';
}

export const credentialsFile = (dataDir) => path.join(dataDir, 'state', 'registry-credentials.json');
export const registriesFile = (dataDir) => path.join(dataDir, 'state', 'registries.json');
export const readJson = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));
