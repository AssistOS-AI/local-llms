import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createObservationClock, validateModel } from 'pworker/local-llm';
import { createPromptObservations } from 'pworker/local-llm/internal/promptObservations.mjs';

import { createAgentController as createController, loadSeedCatalog } from '../src/engineOptions.mjs';
import { readPromptStream } from '../src/promptStream.mjs';
import { runTestPrompt } from '../src/testPrompt.mjs';
import { controllerHandlers } from '../src/controlHandlers.mjs';
import { VULKAN_KEYS } from './overview-cpu-scenarios.mjs';

function observationHarness() {
    let now = 0;
    let current = { deploymentId: 'deployment', runnerIdentity: { runnerStartId: 'runner', pid: 42 },
        target: { modelId: 'model', runnerId: 'runner', baseUrl: 'http://127.0.0.1:1', apiKey: 'private' } };
    const events = createObservationClock({ clock: () => now });
    const observations = createPromptObservations({ events, current: () => current });
    return { observations, events, tick: (ms) => { now += ms; }, replace: () => { current = null; } };
}
const credentials = (begun) => ({ observationId: begun.observation.id, lease: begun.lease });

function response(chunks, onCancel = () => {}) {
    let index = 0;
    return { ok: true, body: new ReadableStream({ pull(controller) {
        if (index < chunks.length) controller.enqueue(new TextEncoder().encode(chunks[index++])); else controller.close();
    }, cancel: onCancel }) };
}
const frame = (value) => `data: ${JSON.stringify(value)}\n\n`;
const delta = (value) => frame({ choices: [{ delta: value }] });

test('SSE only acknowledges nonempty content or reasoning, bounds retained output and requires completion', async () => {
    let first = 0;
    const result = await readPromptStream(response([delta({ role: 'assistant' }), delta({ content: '' }), delta({ reasoning_content: 'think' }), delta({ content: 'a'.repeat(17000) }), 'data: [DONE]\n\n']), { onFirstToken: async () => { first++; } });
    assert.equal(first, 1); assert.equal(result.text.length, 16000); assert.equal(result.reasoningChars, 5); assert.equal(result.truncated, true);
    for (const chunks of [[], ['data: {}'], [delta({ role: 'assistant' })], ['data: {broken}\n\n'], [delta({ content: 12 })]]) {
        await assert.rejects(() => readPromptStream(response(chunks), { onFirstToken: () => assert.fail('no token') }));
    }
    await assert.rejects(() => readPromptStream(response([delta({ content: 'a'.repeat(66000) })]), { onFirstToken: () => assert.fail('too large') }), { code: 'prompt_stream_overflow' });
    await assert.rejects(() => readPromptStream(response(Array(18).fill(':'.padEnd(64000, 'x') + '\n\n')), { onFirstToken: () => assert.fail() }), { code: 'prompt_stream_overflow' });
    await assert.rejects(() => readPromptStream(response([delta({ content: 'x' })]), { onFirstToken: () => { throw new Error('lost acknowledgement'); } }));
});

test('a stalled stream is cancelled by its enclosing deadline', async () => {
    let cancelled = false; const abort = new AbortController();
    const pending = readPromptStream({ body: new ReadableStream({ cancel() { cancelled = true; } }) }, { signal: abort.signal, onFirstToken: () => {} });
    abort.abort(); await assert.rejects(() => pending, { code: 'prompt_timeout' }); assert.equal(cancelled, true);
});

test('observed admin tool uses only the leased target and requires event and terminal acknowledgements', async () => {
    for (const lose of [null, 'recordPromptObservation', 'finishPromptObservation']) {
        const h = observationHarness(); const calls = [];
        const call = async (op, args) => {
            calls.push(op);
            if (op === lose) throw new Error('control reply lost');
            return h.observations[{ beginPromptObservation: 'begin', recordPromptObservation: 'record', observePromptStop: 'observeStop', finishPromptObservation: 'finish' }[op]](args);
        };
        const promise = runTestPrompt({ prompt: 'Name one colour.', observationId: 'prompt_01', call,
            fetchImpl: async (url, init) => { assert.equal(url, 'http://127.0.0.1:1/v1/chat/completions'); assert.equal(JSON.parse(init.body).stream, true);
                return response([delta({ content: 'Blue' }), frame({ choices: [{ delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 4, completion_tokens: 1 } }), 'data: [DONE]\n\n']); } });
        if (lose) await assert.rejects(() => promise); else { const result = await promise; assert.equal(result.text, 'Blue'); assert.equal(result.observation.outcome, 'success'); }
        assert.equal(calls[0], 'beginPromptObservation');
        assert.ok(calls.includes('finishPromptObservation'));
    }
});

function toolHarness({ intercept = async (op, args, invoke) => invoke(), fetchImpl, clock } = {}) {
    const h = observationHarness(); const calls = []; let signal; let stream; let cancelled = 0;
    const stop = () => h.observations.stop({ deploymentId: 'deployment', runnerIdentity: { runnerStartId: 'runner' }, sequence: h.events.next() });
    const call = async (op, args, options) => {
        calls.push({ op, options });
        const invoke = () => h.observations[{ beginPromptObservation: 'begin', recordPromptObservation: 'record', observePromptStop: 'observeStop', finishPromptObservation: 'finish' }[op]](args);
        return intercept(op, args, invoke, { h, stop, calls });
    };
    const pending = runTestPrompt({ prompt: 'Name one colour.', observationId: 'prompt_01', call, ...(clock ? { clock } : {}),
        fetchImpl: fetchImpl || (async (url, init) => { signal = init.signal;
            return { ok: true, body: new ReadableStream({ start(controller) { stream = controller; controller.enqueue(new TextEncoder().encode(delta({ content: 'Blue' }))); }, cancel() { cancelled++; } }) }; }) });
    // Observe rejection immediately so a deliberately delayed assertion never creates an unhandled rejection.
    const result = pending.then((value) => ({ value }), (error) => ({ error }));
    return { h, calls, stop, result, signal: () => signal, cancelled: () => cancelled, stream: () => stream };
}

test('acknowledged Stop notification explicitly aborts pending body I/O and produces exact terminal proof', async () => {
    const tool = toolHarness({ intercept: async (op, args, invoke, { stop }) => { const result = invoke(); if (op === 'recordPromptObservation') stop(); return result; } });
    const result = await tool.result; assert.equal(result.error.code, 'prompt_cancelled_by_stop');
    assert.equal(tool.signal().aborted, true); assert.equal(tool.signal().reason.code, 'prompt_cancelled_by_stop'); assert.equal(tool.cancelled(), 1);
    const [record] = tool.h.observations.snapshot();
    assert.equal(record.outcome, 'interrupted-after-stop'); assert.equal(record.errorCode, result.error.code);
    assert.deepEqual(record.cancellation, { mechanism: 'controller-stop-abort-v1', stopSequence: record.stopSequence, notificationSequence: record.stopNotificationSequence });
    assert.ok(record.firstTokenSequence < record.stopSequence && record.stopSequence < record.stopNotificationSequence && record.stopNotificationSequence < record.terminalSequence);
    assert.equal(tool.calls.filter(({ op }) => op === 'finishPromptObservation').length, 1);
    assert.ok(tool.calls.filter(({ op }) => op === 'observePromptStop').every(({ options }) => options.timeoutMs > 0 && options.timeoutMs <= 1000));
});

test('lost first-token acknowledgement cannot become cancellation proof when Stop notification arrives', async () => {
    const tool = toolHarness({ intercept: async (op, args, invoke, { stop }) => {
        const result = invoke();
        if (op === 'recordPromptObservation') { stop(); return new Promise(() => {}); }
        return result;
    } });
    assert.equal((await tool.result).error.code, 'observation_incomplete');
    const [record] = tool.h.observations.snapshot(); assert.equal(record.firstTokenPresent, true);
    assert.equal(record.cancellation, null); assert.notEqual(record.errorCode, 'prompt_cancelled_by_stop');
    assert.equal(tool.calls.filter(({ op }) => op === 'finishPromptObservation').length, 1);
});

test('lost notification and terminal acknowledgements independently refuse cancellation certification', async () => {
    for (const lost of ['observePromptStop', 'finishPromptObservation']) {
        const tool = toolHarness({ intercept: async (op, args, invoke, { stop, calls }) => {
            const result = invoke(); if (op === 'recordPromptObservation') stop();
            if (op === lost && (op !== 'observePromptStop' || calls.filter((entry) => entry.op === op).length > 1)) throw new Error('reply lost');
            return result;
        } });
        const result = await tool.result;
        assert.equal(result.error.code, lost === 'observePromptStop' ? 'observation_control_failed' : 'observation_incomplete');
        assert.equal(tool.calls.filter(({ op }) => op === 'finishPromptObservation').length, 1);
        if (lost === 'finishPromptObservation') assert.equal(tool.h.observations.snapshot()[0].errorCode, 'prompt_cancelled_by_stop', 'committed controller proof cannot substitute for lost tool acknowledgement');
    }
});

test('foreign or malformed notification replies fail before inference', async () => {
    for (const mutate of [
        (reply) => { reply.recorded = false; }, (reply) => { reply.observation.id = 'foreign_01'; },
        (reply) => { reply.observation.controllerEpoch = 'foreign'; }, (reply) => { reply.observation.runnerIdentity.runnerStartId = 'foreign'; },
        (reply) => { reply.observation.stopSequence = 1.1; }, (reply) => { reply.observation.stopSequence = Number.MAX_SAFE_INTEGER + 1; },
        (reply) => { delete reply.observation.stopNotificationSequence; },
    ]) {
        let fetched = false;
        const tool = toolHarness({ intercept: async (op, args, invoke) => { const result = invoke(); if (op === 'observePromptStop') mutate(result); return result; },
            fetchImpl: async () => { fetched = true; throw new Error('must not fetch'); } });
        assert.equal((await tool.result).error.code, 'observation_incomplete'); assert.equal(fetched, false);
    }
});

test('protocol, UTF-8, transport, overflow, truncation and HTTP failure never emit cancellation proof', async () => {
    const cases = [
        ['prompt_stream_invalid', async () => response(['data: {bad}\n\n'])],
        ['prompt_stream_invalid', async () => ({ ok: true, body: new ReadableStream({ start(controller) { controller.enqueue(Uint8Array.of(0xff)); controller.close(); } }) })],
        ['prompt_transport_failed', async () => { throw new TypeError('fetch failed'); }],
        ['prompt_transport_failed', async () => ({ ok: true, body: new ReadableStream({ pull(controller) { controller.error(new TypeError('body failed')); } }) })],
        ['prompt_stream_overflow', async () => response([delta({ content: 'a'.repeat(66000) })])],
        ['prompt_stream_truncated', async () => response([delta({ role: 'assistant' })])],
        ['runner_error', async () => ({ ok: false, status: 503 })],
    ];
    for (const [code, fetchImpl] of cases) {
        const tool = toolHarness({ fetchImpl }); const result = await tool.result;
        assert.equal(result.error.code, code); assert.equal(tool.h.observations.snapshot()[0].cancellation, null);
        assert.equal(tool.calls.filter(({ op }) => op === 'finishPromptObservation').length, 1);
    }
});

test('protocol and transport recognized after recorded Stop but before notification retain distinct failures', async () => {
    for (const transport of [false, true]) {
        const tool = toolHarness({ intercept: async (op, args, invoke, { stop }) => { const result = invoke(); if (op === 'recordPromptObservation') stop(); return result; },
            fetchImpl: async () => { let pulled = false; return { ok: true, body: new ReadableStream({ pull(controller) {
                if (!pulled) { pulled = true; controller.enqueue(new TextEncoder().encode(delta({ content: 'Blue' }))); }
                else if (transport) controller.error(new TypeError('unrelated failure'));
                else { controller.enqueue(new TextEncoder().encode('data: {bad}\n\n')); controller.close(); }
            } }) }; } });
        assert.equal((await tool.result).error.code, transport ? 'prompt_transport_failed' : 'prompt_stream_invalid');
        assert.equal(tool.h.observations.snapshot()[0].cancellation, null);
    }
});

test('raw stream success wins before a late notification or control failure and is finished once', async () => {
    for (const rejectPoll of [false, true]) {
        let settlePoll;
        const tool = toolHarness({ intercept: async (op, args, invoke, { stop, calls }) => {
            if (op === 'observePromptStop' && calls.filter((entry) => entry.op === op).length > 1) {
                stop(); return new Promise((resolve, reject) => { settlePoll = () => rejectPoll ? reject(new Error('late failure')) : resolve(invoke()); });
            }
            return invoke();
        } });
        await until(() => settlePoll);
        tool.stream().enqueue(new TextEncoder().encode('data: [DONE]\n\n')); tool.stream().close();
        await new Promise((resolve) => setTimeout(resolve, 5)); settlePoll();
        const result = await tool.result; assert.equal(result.value.observation.outcome, 'success'); assert.equal(result.value.observation.cancellation, null);
        assert.equal(tool.calls.filter(({ op }) => op === 'finishPromptObservation').length, 1);
    }
});

test('deadline precedes a later Stop notification without extending inference time', async () => {
    let now = 0;
    const tool = toolHarness({ clock: () => now, intercept: async (op, args, invoke, { calls, stop }) => {
        if (op === 'observePromptStop' && calls.filter((entry) => entry.op === op).length > 1) { now = 240000; stop(); }
        return invoke();
    } });
    assert.equal((await tool.result).error.code, 'prompt_timeout'); assert.equal(tool.h.observations.snapshot()[0].cancellation, null);
});

test('notification-selected abort wins over a consequent transport rejection without relabelling earlier failure', async () => {
    let reads = 0; let cancelled = 0; let rejectRead;
    const tool = toolHarness({ intercept: async (op, args, invoke, { stop }) => { const result = invoke(); if (op === 'recordPromptObservation') stop(); return result; },
        fetchImpl: async (url, { signal }) => ({ ok: true, body: { getReader: () => ({
            read() {
                if (!reads++) return Promise.resolve({ value: new TextEncoder().encode(delta({ content: 'Blue' })), done: false });
                return new Promise((resolve, reject) => { rejectRead = reject; signal.addEventListener('abort', () => reject(new TypeError('closed after explicit abort')), { once: true }); });
            }, cancel() { cancelled++; rejectRead?.(new TypeError('cancelled')); return Promise.resolve(); }, releaseLock() {},
        }) } }) });
    assert.equal((await tool.result).error.code, 'prompt_cancelled_by_stop'); assert.equal(cancelled, 1);
    assert.equal(tool.h.observations.snapshot()[0].cancellation.mechanism, 'controller-stop-abort-v1');
});

test('terminal acknowledgement must match exact cancellation proof and lost success never triggers a second finish', async () => {
    for (const mutate of [(record) => { record.cancellation.notificationSequence++; }, (record) => { record.runnerIdentity.pid++; },
        (record) => { record.cancellation.extra = true; }, (record) => { record.terminalSequence = record.stopSequence; }]) {
        const tool = toolHarness({ intercept: async (op, args, invoke, { stop }) => { const result = invoke();
            if (op === 'recordPromptObservation') stop(); if (op === 'finishPromptObservation') mutate(result.observation); return result; } });
        assert.equal((await tool.result).error.code, 'observation_incomplete'); assert.equal(tool.calls.filter(({ op }) => op === 'finishPromptObservation').length, 1);
    }
    const success = toolHarness({ fetchImpl: async () => response([delta({ content: 'Blue' }), 'data: [DONE]\n\n']),
        intercept: async (op, args, invoke) => { const result = invoke(); if (op === 'finishPromptObservation') throw new Error('lost success reply'); return result; } });
    assert.equal((await success.result).error.code, 'observation_incomplete'); assert.equal(success.h.observations.snapshot()[0].outcome, 'success');
    assert.equal(success.calls.filter(({ op }) => op === 'finishPromptObservation').length, 1);
});

test('joining a poll and finishing share one two-second budget', async () => {
    let now = 0; let releasePoll; let finishTimeout;
    const tool = toolHarness({ clock: () => now, intercept: async (op, args, invoke, { calls }) => {
        if (op === 'observePromptStop' && calls.filter((entry) => entry.op === op).length > 1) {
            return new Promise((resolve) => { releasePoll = () => { now = 1700; resolve(invoke()); }; });
        }
        if (op === 'finishPromptObservation') finishTimeout = calls.at(-1).options.timeoutMs;
        return invoke();
    } });
    await until(() => releasePoll); tool.stream().enqueue(new TextEncoder().encode('data: [DONE]\n\n')); tool.stream().close();
    await new Promise((resolve) => setTimeout(resolve, 5)); releasePoll();
    assert.equal((await tool.result).value.observation.outcome, 'success'); assert.equal(finishTimeout, 300);
});

test('stalled poll is single-flight, bounded, joined and leaves no scheduled polling after settlement', async () => {
    let inFlight = 0; let maximum = 0; let stalledResolve;
    const tool = toolHarness({ intercept: async (op, args, invoke, { calls }) => {
        if (op !== 'observePromptStop') return invoke();
        inFlight++; maximum = Math.max(maximum, inFlight);
        if (calls.filter((entry) => entry.op === op).length > 1) {
            try { return await new Promise((resolve) => { stalledResolve = () => resolve(invoke()); }); } finally { inFlight--; }
        }
        inFlight--; return invoke();
    } });
    await until(() => stalledResolve); const before = performance.now();
    tool.stream().enqueue(new TextEncoder().encode('data: [DONE]\n\n')); tool.stream().close();
    const result = await tool.result; assert.equal(result.value.observation.outcome, 'success');
    assert.ok(performance.now() - before < 2000); assert.equal(maximum, 1);
    const polls = tool.calls.filter(({ op }) => op === 'observePromptStop').length;
    await new Promise((resolve) => setTimeout(resolve, 120)); assert.equal(tool.calls.filter(({ op }) => op === 'observePromptStop').length, polls);
    // The underlying test promise has no real socket. Its late terminal lookup is observed by the bounded call.
    assert.throws(stalledResolve, { code: 'observation_stale' });
});

async function until(predicate) {
    const deadline = performance.now() + 2000;
    while (performance.now() < deadline) { if (await predicate()) return; await new Promise((resolve) => setTimeout(resolve, 2)); }
    assert.fail('observation not reached');
}

function controllerHarness(t, { profile = 'cpu', health = 503, holdModels = false, drop = () => true, files = null } = {}) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'llm-observe-'));
    const weight = path.join(dataDir, 'weight.gguf'); fs.writeFileSync(weight, 'offline');
    const memory = { totalBytes: 8 * 2 ** 30, availableBytes: 7 * 2 ** 30 }; const handles = []; let modelRequests = 0; let downloads = 0; let drops = 0;
    const seed = loadSeedCatalog().filter((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m').map((model) => validateModel({ ...model, recommended: { venus: { 'llama.cpp-vulkan': { ctxSize: 2048, parallel: 1, loadMode: 'none' } } } }, { seed: true }));
    if (files) {
        const original = seed[0].sources.gguf;
        const shards = [1, 2, 3].map((index) => ({ file: `fixture-0000${index}-of-00003.gguf`, size: 100000000, sha256: original.sha256 }));
        seed[0] = validateModel({ ...seed[0], sources: { gguf: { ...original, file: shards[0].file, size: 300000000, sha256: undefined, shards } } }, { seed: true });
    }
    const controller = createController({ dataDir, seedCatalog: seed, sharedModelsRoot: null, imageContract: { ...VULKAN_KEYS },
        env: { PATH: '/usr/bin' }, fileExists: () => true, installer: { installable: () => false }, detectRunner: () => ({ installed: true }),
        snapshot: async () => ({ memory, disk: { freeBytes: 1e12 }, gpu: profile === 'cpu' ? { available: false, state: 'absent' }
            : { available: true, backend: 'vulkan', driver: 'venus', memoryModel: 'venus', name: 'fixture', totalBytes: 8 * 2 ** 30, freeBytes: 8 * 2 ** 30, icdFile: VULKAN_KEYS.vulkan_icd_venus } }),
        readMemory: () => memory, readPressure: () => ({ some: { avg10: 0 }, full: { avg10: 0 } }), readBudget: () => ({ state: 'absent' }),
        inspect: async () => ({ state: 'complete', bytes: 491400032, total: 491400032 }),
        download: async () => { const target = files ? files(weight, dataDir)[downloads % 3] : weight; downloads++; return { status: 'complete', path: target, bytesTransferred: 0 }; },
        dropCache: (file) => { drops++; return drop(file); }, pollMs: 2, stopGraceMs: 20,
        startRunner() { let resolve; const handle = { pid: 9000 + handles.length, running: true, exited: new Promise((done) => { resolve = done; }),
            async stop() { handle.running = false; resolve({ code: 0, signal: 'SIGTERM' }); }, async kill() { return handle.stop(); } }; handles.push(handle); return handle; },
        fetchImpl: async (url, { signal }) => {
            if (url.endsWith('/health')) { const status = typeof health === 'function' ? health() : health; if (status === 'refused') throw new Error('connection refused'); return { ok: status === 200, status }; }
            modelRequests++;
            if (holdModels) return new Promise((resolve, reject) => { signal.addEventListener('abort', () => reject(new Error('stopped')), { once: true }); });
            return { ok: true, status: 200 };
        },
    });
    t.after(async () => { await controller.drain(); fs.rmSync(dataDir, { force: true, recursive: true }); });
    return { controller, handles, count: () => ({ downloads, drops, modelRequests }),
        run: (requestId, params = { ctxSize: 2048, parallel: 1, loadMode: 'none' }) => controller.run({ requestId, modelId: seed[0].id, runnerId: profile === 'cpu' ? 'llama.cpp' : 'llama.cpp-vulkan', params }) };
}

test('cached runs each perform standard per-file cache preparation and retain separate bounded outcomes', async (t) => {
    for (const [drop, success, failure, errors] of [[() => true, 1, 0, 0], [() => false, 0, 1, 0], [() => { throw new Error('fixture'); }, 0, 0, 1]]) {
        const h = controllerHarness(t, { health: 200, drop, files: (weight, dir) => [weight, dir, path.join(dir, 'missing')] });
        const observations = [];
        for (const id of ['request_first', 'request_reload']) {
            await h.run(id); await until(async () => (await h.controller.status()).phase === 'ready');
            const status = await h.controller.status(); observations.push(status.startupObservation);
            const counts = status.startupObservation.pageCachePreparation;
            assert.deepEqual(counts, { policy: 'existing-best-effort-per-file-drop', phase: 'complete', filesConsidered: 3, attempts: 1,
                reportedSuccesses: success, reportedFailures: failure, skippedNonRegular: 1, observationErrors: errors + 1 });
            const handlers = controllerHandlers(h.controller); const began = handlers.beginPromptObservation({ observationId: id });
            handlers.recordPromptObservation({ ...credentials(began), event: 'first-token' });
            await h.controller.stop(); handlers.finishPromptObservation({ ...credentials(began), outcome: 'failed', errorCode: 'runner_unreachable' });
        }
        assert.notEqual(observations[0].deploymentId, observations[1].deploymentId);
        assert.notEqual(observations[0].runnerIdentity.runnerStartId, observations[1].runnerIdentity.runnerStartId);
        assert.deepEqual(h.count(), { downloads: 6, drops: 2, modelRequests: 2 });
    }
});

