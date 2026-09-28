// Stop and Cancel invalidate every Run submitted before them (DS001): the Run
// whose command is running and the Runs still waiting behind it in the command
// queue. A Run submitted after the Stop or Cancel is new intent and goes ahead.
// Duplicates, busy, replace and the order of consecutive Stops and Cancels keep
// their meaning. Hardware, downloads and runners are injected; no /proc is used.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';

const GIB = 1024 ** 3;
const KEY = 'k'.repeat(43);
const MODEL = 'gpt-oss-20b';
// Delete weights of another model's GGUF (a store that removes through the injected `remove`) holds the queue.
const OTHER_WEIGHTS = Object.freeze({ modelId: 'gpt-oss-120b', format: 'gguf' });

function deferred() {
    let resolve;
    const promise = new Promise((ok) => { resolve = ok; });
    return { promise, resolve };
}

async function until(predicate, label, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 3));
    }
    assert.fail(`timed out waiting for ${label}`);
}

const settle = (promise) => promise.then((value) => ({ value }), (error) => ({ error: error.code }));
const outcome = (result) => result.error ?? (result.value?.duplicate ? 'duplicate' : result.value?.accepted ? 'accepted' : 'ok');

/**
 * A controller whose snapshot `holdRead` (1-based) waits until released and
 * ends when its signal aborts, and whose weight removal can be held, so a
 * Run or a Delete weights command occupies the queue while others wait.
 */
function harness(t, { holdRead = null } = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-run-cancel-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    const weights = path.join(dataDir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const held = deferred();
    const removal = { gate: null, entered: deferred() };
    let reads = 0;
    const downloads = [];
    const started = [];
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        stateStore: createStateStore({ dataDir }),
        seedCatalog: loadSeedCatalog(),
        snapshot: ({ signal } = {}) => {
            reads += 1;
            const snap = {
                gpu: { available: true, name: 'RTX', memoryModel: 'dedicated', totalBytes: 6 * GIB, freeBytes: 6 * GIB, usedBytes: 0, processes: [] },
                memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB },
                disk: { freeBytes: 400 * GIB },
            };
            if (reads !== holdRead) return Promise.resolve(snap);
            return new Promise((resolve, reject) => {
                signal?.addEventListener('abort', () => reject(Object.assign(new Error('stopped'), { name: 'AbortError' })), { once: true });
                held.promise.then(() => resolve(snap));
            });
        },
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async ({ artifact }) => { downloads.push(artifact.file); return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        remove: async () => {
            removal.entered.resolve();
            if (removal.gate) await removal.gate.promise;
            return 0;
        },
        sharedModelsRoot: null,
        imageContract: null,
        installer: { installable: () => false },
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner({ args }) {
            const exit = deferred();
            const handle = { pid: 9200 + started.length, args, exited: exit.promise, running: true,
                async stop() { handle.running = false; handle.stopped = true; exit.resolve({ code: 0, signal: 'SIGTERM' }); return exit.promise; },
                async kill() { return handle.stop(); } };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        pollMs: 1,
        stopGraceMs: 20,
    });
    t.after(() => controller.drain());
    return { controller, downloads, started, release: () => held.promise && held.resolve(), reads: () => reads, removal };
}

const run = (h, requestId, extra = {}) => settle(h.controller.run({ requestId, modelId: MODEL, runnerId: 'llama.cpp', ...extra }));

// A Run occupies the queue in its admission snapshot (read 2: read 1 decides the profile).
// Returned wrapped: awaiting a bare promise here would wait for the held Run itself.
async function holdRunInAdmission(h, requestId) {
    const pending = run(h, requestId);
    await until(() => h.reads() === 2, 'the admission snapshot to be held');
    return { pending };
}

function assertNothingRecordedBut(h, requestIds) {
    assert.deepEqual(Object.keys(h.controller.state.requests).sort(), [...requestIds].sort());
}

for (const op of ['stop', 'cancelDownload']) {
    test(`${op} cancels the running Run and every Run queued before it; a Run submitted afterwards goes ahead`, async (t) => {
        const h = harness(t, { holdRead: 2 });
        const { pending: a } = await holdRunInAdmission(h, 'request-before-a');
        const b = run(h, 'request-before-b');
        const c = run(h, 'request-before-c', { replace: true });
        const stopping = settle(h.controller[op]());
        // New intent: submitted after the Stop or Cancel, while the earlier Runs still hold the queue.
        const d = run(h, 'request-after-d');
        h.release();
        const [ra, rb, rc, rs, rd] = [await a, await b, await c, await stopping, await d];
        assert.deepEqual([outcome(ra), outcome(rb), outcome(rc)], ['cancelled', 'cancelled', 'cancelled']);
        assert.equal(rs.error, undefined, `${op} succeeds`);
        assert.equal(outcome(rd), 'accepted');
        await until(() => h.controller.state.deployment?.phase === 'ready', 'the later Run to be ready');
        // Only the later Run recorded anything, downloaded or started.
        assertNothingRecordedBut(h, ['request-after-d']);
        assert.equal(h.controller.state.deployment.requestId, 'request-after-d');
        assert.equal(h.downloads.length, 1);
        assert.equal(h.started.length, 1);
        await h.controller.stop();
    });
}

test('a cancelled queued Run leaves no request id, parameters or deployment, so the same request can be sent again', async (t) => {
    const h = harness(t, { holdRead: 2 });
    const { pending: a } = await holdRunInAdmission(h, 'request-retry-a');
    const b = run(h, 'request-retry-b', { params: { ctxSize: 8192 } });
    const stopping = settle(h.controller.stop());
    h.release();
    assert.equal(outcome(await a), 'cancelled');
    assert.equal(outcome(await b), 'cancelled');
    await stopping;
    assert.equal(h.controller.state.deployment, null);
    assert.deepEqual(h.controller.state.requests, {});
    assert.deepEqual(h.controller.state.params, {});
    assert.equal(h.downloads.length, 0);
    assert.equal(h.started.length, 0);
    // Sent again after the Stop, the same request is a new Run.
    const again = await run(h, 'request-retry-b', { params: { ctxSize: 8192 } });
    assert.equal(outcome(again), 'accepted');
    await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
    assert.equal(h.controller.state.deployment.params.ctxSize, 8192);
    await h.controller.stop();
});

test('a Cancel while a Run waits behind another command cancels it and succeeds; without the Cancel the Run goes ahead', async (t) => {
    // Control: a Run queued behind a held Delete weights is accepted once the queue moves.
    const control = harness(t);
    control.removal.gate = deferred();
    const deleting = settle(control.controller.deleteWeights(OTHER_WEIGHTS));
    await control.removal.entered.promise;
    const queued = run(control, 'request-queued-ok');
    control.removal.gate.resolve();
    await deleting;
    assert.equal(outcome(await queued), 'accepted');
    await control.controller.stop();

    const h = harness(t);
    h.removal.gate = deferred();
    const deletingToo = settle(h.controller.deleteWeights(OTHER_WEIGHTS));
    await h.removal.entered.promise;
    const waiting = run(h, 'request-queued-cancel');
    const cancelling = settle(h.controller.cancelDownload());
    h.removal.gate.resolve();
    assert.equal((await deletingToo).error, undefined, 'the other command is not affected');
    assert.equal(outcome(await waiting), 'cancelled');
    const cancelled = await cancelling;
    assert.equal(cancelled.error, undefined, 'the Cancel stopped a queued Run, so it succeeds');
    assert.equal(cancelled.value.deployment, null);
    assert.equal(h.downloads.length, 0);
    assert.deepEqual(h.controller.state.requests, {});
    // With nothing submitted and nothing downloading, a Cancel is still not_downloading.
    assert.equal((await settle(h.controller.cancelDownload())).error, 'not_downloading');
});

test('consecutive Stop and Cancel keep their order: the first cancels the earlier Runs, the second finds nothing left', async (t) => {
    const h = harness(t, { holdRead: 2 });
    const { pending: a } = await holdRunInAdmission(h, 'request-order-a');
    const b = run(h, 'request-order-b');
    const stopping = settle(h.controller.stop());
    const cancelling = settle(h.controller.cancelDownload());
    h.release();
    assert.deepEqual([outcome(await a), outcome(await b)], ['cancelled', 'cancelled']);
    assert.equal((await stopping).error, undefined);
    assert.equal((await cancelling).error, 'not_downloading', 'the Stop already cancelled them');

    const g = harness(t, { holdRead: 2 });
    const { pending: c } = await holdRunInAdmission(g, 'request-order-c');
    const cancelFirst = settle(g.controller.cancelDownload());
    const stopSecond = settle(g.controller.stop());
    g.release();
    assert.equal(outcome(await c), 'cancelled');
    assert.equal((await cancelFirst).error, undefined, 'the Cancel cancelled the pending Run');
    assert.equal((await stopSecond).error, undefined);
    // After both, new intent goes ahead.
    assert.equal(outcome(await run(g, 'request-order-d')), 'accepted');
    await until(() => g.controller.state.deployment?.phase === 'ready', 'ready');
    assert.equal(g.downloads.length, 1);
    await g.controller.stop();
});

test('duplicates, busy and replace keep their meaning across a Stop', async (t) => {
    const h = harness(t);
    assert.equal(outcome(await run(h, 'request-first-x')), 'accepted');
    await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
    // Busy: another Run without replace while one is active.
    assert.equal((await run(h, 'request-other-y')).error, 'busy');
    // Replace: accepted, and the active runner is stopped first.
    assert.equal(outcome(await run(h, 'request-replace-z', { replace: true })), 'accepted');
    await until(() => h.controller.state.deployment?.requestId === 'request-replace-z' && h.controller.state.deployment.phase === 'ready', 'the replacement');
    assert.equal(h.started[0].stopped, true);
    // A duplicate of an accepted request stays a no-op, before or after a Stop.
    h.removal.gate = deferred();
    const deleting = settle(h.controller.deleteWeights(OTHER_WEIGHTS));
    await h.removal.entered.promise;
    const queuedDuplicate = run(h, 'request-first-x');
    const stopping = settle(h.controller.stop());
    h.removal.gate.resolve();
    await deleting;
    assert.equal(outcome(await queuedDuplicate), 'duplicate', 'a duplicate is answered as one, not cancelled');
    await stopping;
    assert.equal(outcome(await run(h, 'request-first-x')), 'duplicate');
    assert.equal(h.downloads.length, 2, 'no download for duplicates');
    assert.equal(h.started.length, 2);
});
