// Runtime models on unified memory without benchmark gates (owner, 2026-09-29),
// the Qwen MTP default with MTP as a declared model capability, experimental
// vLLM behind the operator's switch, and complete installed-distribution
// validation (DS002, DS004, DS005).
//
// Hardware, downloads, verification, runners and the runner installer are
// injected: no /proc, GPU, network or real model is used. Every number here is
// synthetic; no envelope, margin or constant is approved by these tests.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createStateStore, validateModel } from 'pworker/local-llm';

import { createAgentController as createController, loadSeedCatalog } from '../src/engineOptions.mjs';
import { controllerHandlers } from '../src/controlHandlers.mjs';
import { TOOL_OPERATIONS, handleTool } from '../tools/local_llm_tool.mjs';
import { modelEntryFromForm } from '../IDE-plugins/local-llm-settings/local-llm-settings-model.js';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
const TOTAL = 120 * GIB;
const SEED = loadSeedCatalog();

const GB10 = Object.freeze({
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
});

function unifiedSnap(available = 100 * GIB, total = TOTAL) {
    return { gpu: structuredClone(GB10), memory: { totalBytes: total, availableBytes: available }, disk: { freeBytes: 400 * GIB }, cpus: 20 };
}

async function until(predicate, label, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 3));
    }
    assert.fail(`timed out waiting for ${label}`);
}

/** The Run form's preview, through the overview as the dashboard asks for it. */
async function previewOf(h, request) {
    return (await h.controller.overview({ preview: request })).preview;
}

/** A unified-memory controller; `dataDir` lets a second controller reload the first one's state. */
// `hostArch` omitted leaves the controller's own default (process.arch).
function harness(t, { seed = SEED, available = () => 100 * GIB, env = {}, dataDir, installer, resolveHf, snapshot = null, imageContract = null,
    hostArch } = {}) {
    const dir = dataDir || fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-runtime-models-'));
    const weights = path.join(dir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const started = [];
    const calls = { download: 0, resolveHf: 0 };
    const controller = createController({
        dataDir: dir,
        env: { PATH: '/usr/bin', ...env },
        seedCatalog: seed,
        stateStore: createStateStore({ dataDir: dir }),
        snapshot: async () => (snapshot ? snapshot() : unifiedSnap(available())),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async () => { calls.download += 1; return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        verify: async () => ({ notes: [] }),
        remove: async () => 0,
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }),
        downloadSnapshot: async () => ({ status: 'complete', bytesTransferred: 0 }),
        resolveHf: resolveHf || (async () => { calls.resolveHf += 1; return { commit: 'c'.repeat(40), size: 20 * GIB, sha256: 'd'.repeat(64) }; }),
        sharedModelsRoot: null,
        imageContract,
        hostArch,
        installer: installer || { installable: () => false },
        shmDir: path.join(dir, 'shm'),
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner({ command, args, env: runnerEnv }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 7100 + started.length, command, args, env: runnerEnv,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                async kill() { running = false; resolveExit({ code: null, signal: 'SIGKILL', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        pollMs: 1,
        stopGraceMs: 20,
        readMemory: () => ({ totalBytes: TOTAL, availableBytes: available() }),
        readPressure: () => 0,
        unifiedGuardMs: 5,
        dropCache: () => true,
    });
    cleanupAfter(t, controller, dataDir ? null : dir);
    return { controller, started, calls, dataDir: dir };
}

// One cleanup per test: every controller of the test drains (and saves its
// state) before any of the test's directories is removed, so no drain can
// recreate a directory that was already removed.
const cleanups = new WeakMap();
function cleanupAfter(t, controller, dir) {
    let entry = cleanups.get(t);
    if (!entry) {
        entry = { controllers: [], dirs: [] };
        cleanups.set(t, entry);
        t.after(async () => {
            for (const each of entry.controllers) await each.drain().catch(() => {});
            for (const each of entry.dirs) fs.rmSync(each, { recursive: true, force: true });
        });
    }
    entry.controllers.push(controller);
    if (dir) entry.dirs.push(dir);
}

const valueOf = (args, flag) => args[args.indexOf(flag) + 1];

// ------------------------------------------------ bundled models by estimate

// ------------------------------------------------------ models added at run time

const ADDED = Object.freeze({
    id: 'user-moe', displayName: 'User MoE', architecture: 'moe', contextLength: 65536, profiles: ['unified'],
    memory: { layers: 40, kvBytesPerToken: 40960 },
    sources: { gguf: { type: 'huggingface', repo: 'user/Model-GGUF', file: 'model-Q4_K_M.gguf', revision: 'main' } },
});

test('the admin tools add a model at run time and plan and start its Run through the operations the agent serves', async (t) => {
    const h = harness(t);
    // main.mjs serves controllerHandlers(controller) on the control socket; each tool maps its input onto one operation.
    const handlers = controllerHandlers(h.controller);
    for (const { op } of Object.values(TOOL_OPERATIONS)) assert.equal(typeof handlers[op], 'function', op);
    const ops = [];
    // The socket carries JSON both ways; so does this stand-in.
    const call = async (op, args) => {
        ops.push(op);
        return JSON.parse(JSON.stringify(await handlers[op](JSON.parse(JSON.stringify(args)))));
    };
    const admin = { user: { roles: ['admin'], email: 'admin@example.com' } };
    const tool = (name, input, authInfo = admin) => handleTool(name, input, { authInfo, call });
    await assert.rejects(tool('local_llm_model_add', { model: structuredClone(ADDED) }, { user: { roles: ['user'] } }),
        (error) => error.code === 'admin_required');
    assert.deepEqual(ops, [], 'refused before any operation');
    const added = await tool('local_llm_model_add', { model: structuredClone(ADDED) });
    assert.equal(added.model.id, 'user-moe');
    assert.equal(added.model.sources.gguf.commit, 'c'.repeat(40));
    const listed = (await tool('local_llm_overview', {})).models.find((entry) => entry.id === 'user-moe');
    assert.equal(listed.runners['llama.cpp'].admission.status, 'ok');
    assert.equal(listed.runners['llama.cpp'].admission.estimate.measured, false);
    const { preview } = await tool('local_llm_overview', { preview: { modelId: 'user-moe', runnerId: 'llama.cpp', params: { ctxSize: 16384 } } });
    assert.equal(preview.params.ctxSize, 16384, 'explicit parameters win');
    assert.equal(preview.admission.status, 'ok');
    const run = await tool('local_llm_run', { requestId: 'request-tool-01', modelId: 'user-moe', runnerId: 'llama.cpp', params: { ctxSize: 16384 } });
    assert.equal(run.accepted, true);
    await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
    assert.equal(valueOf(h.started[0].args, '--ctx-size'), '16384');
    const status = await tool('local_llm_status', {});
    assert.equal(status.deployment.modelId, 'user-moe');
    await tool('local_llm_stop', {});
    assert.deepEqual(ops, ['addModel', 'overview', 'overview', 'run', 'status', 'stop']);
});

test('the Add model form carries the optional sizing data and the MTP head; malformed numbers reach validation', () => {
    // What the dashboard's Add model form sends: its inputs' names, and the MTP checkbox's value when it is checked.
    const html = fs.readFileSync(new URL('../IDE-plugins/local-llm-tool-button/components/local-llm-dashboard/local-llm-dashboard.html',
        import.meta.url), 'utf8');
    const inputs = [...html.matchAll(/<input\b[^>]*>/g)].map(([tag]) => ({
        name: /\bname="([^"]+)"/.exec(tag)?.[1], type: /\btype="([^"]+)"/.exec(tag)?.[1], value: /\bvalue="([^"]*)"/.exec(tag)?.[1] }));
    for (const name of ['contextLength', 'layers', 'kvBytesPerToken']) {
        assert.equal(inputs.filter((input) => input.name === name && input.type === 'number').length, 1, name);
    }
    const mtpInputs = inputs.filter((input) => input.name === 'mtp');
    assert.deepEqual(mtpInputs, [{ name: 'mtp', type: 'checkbox', value: 'true' }]);
    const base = { id: 'Form-Model', sourceKind: 'huggingface', repo: 'u/m', file: 'm.gguf' };
    const full = modelEntryFromForm({ ...base, contextLength: '65536', layers: '40', kvBytesPerToken: '40960', mtp: mtpInputs[0].value });
    assert.equal(full.contextLength, 65536);
    assert.deepEqual(full.memory, { layers: 40, kvBytesPerToken: 40960 });
    assert.equal(full.mtp, true);
    const empty = modelEntryFromForm({ ...base, contextLength: '', layers: ' ', kvBytesPerToken: '' });
    assert.equal(Object.hasOwn(empty, 'contextLength'), false);
    assert.equal(Object.hasOwn(empty, 'memory'), false);
    assert.equal(Object.hasOwn(empty, 'mtp'), false);
    const typo = modelEntryFromForm({ ...base, contextLength: '64k' });
    assert.equal(typo.contextLength, '64k');
    assert.throws(() => validateModel(typo), (error) => error.details.field === 'contextLength');
});

// ------------------------------------------------- experimental vLLM, unified

// ------------------------------------------------ the tool entry's operations

test('the operations main.mjs serves are exactly those the tools and the chat responder call', () => {
    const called = new Set(Object.values(TOOL_OPERATIONS).map(({ op }) => op));
    for (const file of ['../src/chatResponder.mjs', '../src/testPrompt.mjs']) {
        const source = fs.readFileSync(new URL(file, import.meta.url), 'utf8');
        for (const [, op] of source.matchAll(/\b(?:call|observedCall)\('([A-Za-z]+)'/g)) called.add(op);
    }
    assert.ok(called.has('chatTarget') && called.has('recordCompletion'));
    const served = Object.keys(controllerHandlers({}));
    assert.deepEqual(served.sort(), [...called].sort());
    const main = fs.readFileSync(new URL('../src/main.mjs', import.meta.url), 'utf8');
    assert.match(main, /handlers: controllerHandlers\(controller\)/);
});

