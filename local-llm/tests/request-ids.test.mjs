// Accepted Run requests are keyed by request id, and every id the request
// pattern allows is an ordinary key: `constructor`, `__proto__`, `toString` and
// `hasOwnProperty` are fresh ids until accepted, are recorded and persisted like
// any other, and never read or change a prototype. Hardware, downloads and
// runners are injected; no /proc is used.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { acceptedRequest, createStateStore, recordRequest, requestMap } from '../src/controller/stateStore.mjs';

const GIB = 1024 ** 3;
const KEY = 'k'.repeat(43);
const SPECIAL = Object.freeze(['constructor', '__proto__', 'toString', 'hasOwnProperty']);
const ORDINARY = 'fresh-ordinary-0001';
const REQUEST_ID_RE = /^[A-Za-z0-9_-]{8,128}$/;
// Object.prototype as it was before any test ran; checked again at the end.
const PROTOTYPE_BEFORE = Object.getOwnPropertyNames(Object.prototype).sort();

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

function assertPrototypesUntouched() {
    assert.deepEqual(Object.getOwnPropertyNames(Object.prototype).sort(), PROTOTYPE_BEFORE);
    assert.equal(Object.getPrototypeOf({}), Object.prototype);
    assert.equal(({}).deploymentId, undefined);
}

function harness(t, { dataDir = null } = {}) {
    const dir = dataDir || fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-request-ids-'));
    if (!dataDir) t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const weights = path.join(dir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const removal = { gate: null, entered: deferred() };
    const downloads = [];
    const started = [];
    const controller = createController({
        dataDir: dir,
        env: { PATH: '/usr/bin' },
        stateStore: createStateStore({ dataDir: dir }),
        seedCatalog: loadSeedCatalog(),
        snapshot: async () => ({
            gpu: { available: true, name: 'RTX', memoryModel: 'dedicated', totalBytes: 6 * GIB, freeBytes: 6 * GIB, usedBytes: 0, processes: [] },
            memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB },
            disk: { freeBytes: 400 * GIB },
        }),
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
            const handle = { pid: 9300 + started.length, args, exited: exit.promise, running: true,
                async stop() { handle.running = false; exit.resolve({ code: 0, signal: 'SIGTERM' }); return exit.promise; },
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
    return { controller, downloads, started, removal, dataDir: dir };
}

const run = (h, requestId, modelId = 'gpt-oss-20b') => settle(h.controller.run({ requestId, modelId, runnerId: 'llama.cpp' }));

test('the special ids are valid request ids; the map of accepted requests starts empty, as an ordinary object', (t) => {
    const h = harness(t);
    for (const requestId of [...SPECIAL, ORDINARY]) assert.match(requestId, REQUEST_ID_RE);
    assert.equal(Object.getPrototypeOf(h.controller.state.requests), Object.prototype);
    assert.deepEqual(h.controller.state.requests, {});
    // Inherited members such as constructor or toString exist on it, and are never taken for records.
    for (const requestId of SPECIAL) assert.equal(acceptedRequest(h.controller.state.requests, requestId), null);
});

for (const requestId of [...SPECIAL, ORDINARY]) {
    test(`a fresh ${requestId} is not a duplicate: it reaches normal validation, and nothing is recorded`, async (t) => {
        const h = harness(t);
        assert.equal(outcome(await run(h, requestId, 'unknown-model')), 'unknown_model');
        assert.deepEqual(Object.keys(h.controller.state.requests), []);
        assert.equal(h.controller.state.deployment, null);
        assert.equal(h.downloads.length, 0);
        assertPrototypesUntouched();
    });

    test(`${requestId} is accepted once, its retry is a duplicate, and it survives a restart`, async (t) => {
        const h = harness(t);
        assert.equal(outcome(await run(h, requestId)), 'accepted');
        await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
        assert.equal(h.controller.state.deployment.requestId, requestId);
        assert.ok(Object.hasOwn(h.controller.state.requests, requestId), 'recorded as an own entry');
        assert.equal(Object.getPrototypeOf(h.controller.state.requests), Object.prototype, 'recording did not set a prototype');
        assert.deepEqual(Object.keys(h.controller.state.requests), [requestId]);
        assert.equal(outcome(await run(h, requestId)), 'duplicate');
        assert.equal(h.downloads.length, 1);
        assertPrototypesUntouched();
        // Persisted as a key of the state file, and read back as one after a restart.
        await h.controller.drain();
        const file = JSON.parse(fs.readFileSync(path.join(h.dataDir, 'state', 'controller.json'), 'utf8'));
        assert.ok(Object.hasOwn(file.requests, requestId));
        assert.deepEqual(Object.keys(file.requests), [requestId]);
        const again = harness(t, { dataDir: h.dataDir });
        assert.equal(Object.getPrototypeOf(again.controller.state.requests), Object.prototype);
        assert.ok(Object.hasOwn(again.controller.state.requests, requestId));
        assert.equal(outcome(await run(again, requestId)), 'duplicate');
        // Another special id is still fresh after the restart.
        const other = SPECIAL.find((id) => id !== requestId);
        assert.equal(outcome(await run(again, other, 'unknown-model')), 'unknown_model');
        assertPrototypesUntouched();
    });
}

test('a Run with a special id queued before a Cancel is new work: it is cancelled and the Cancel succeeds', async (t) => {
    for (const requestId of [...SPECIAL, ORDINARY]) {
        const h = harness(t);
        h.removal.gate = deferred();
        const deleting = settle(h.controller.deleteWeights({ modelId: 'gpt-oss-120b', format: 'gguf' }));
        await h.removal.entered.promise;
        const queued = run(h, requestId);
        const cancelling = settle(h.controller.cancelDownload());
        h.removal.gate.resolve();
        await deleting;
        assert.equal(outcome(await queued), 'cancelled', requestId);
        const cancelled = await cancelling;
        assert.equal(cancelled.error, undefined, `${requestId}: the Cancel cancelled new work`);
        assert.deepEqual(Object.keys(h.controller.state.requests), []);
        assert.equal(h.downloads.length, 0);
        // Sent again after the Cancel, it is accepted.
        assert.equal(outcome(await run(h, requestId)), 'accepted', requestId);
        await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
        await h.controller.stop();
    }
    assertPrototypesUntouched();
});

test('a state file that names special ids is read as own entries, and only those ids are duplicates', async (t) => {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-request-ids-'));
    t.after(() => fs.rmSync(dataDir, { recursive: true, force: true }));
    fs.mkdirSync(path.join(dataDir, 'state'), { recursive: true });
    // Written as text: `"__proto__"` is an ordinary key of the JSON document.
    fs.writeFileSync(path.join(dataDir, 'state', 'controller.json'), `{
  "version": 1, "deployment": null, "params": {}, "registry": [],
  "requests": {
    "__proto__": { "deploymentId": "d-proto", "at": "2026-09-29T00:00:00.000Z" },
    "constructor": { "deploymentId": "d-ctor", "at": "2026-09-29T00:00:01.000Z" }
  }
}
`);
    const h = harness(t, { dataDir });
    assert.equal(Object.getPrototypeOf(h.controller.state.requests), Object.prototype, 'loading did not set a prototype');
    assert.deepEqual(Object.keys(h.controller.state.requests).sort(), ['__proto__', 'constructor']);
    assert.equal(acceptedRequest(h.controller.state.requests, '__proto__').deploymentId, 'd-proto');
    assert.equal(outcome(await run(h, '__proto__')), 'duplicate');
    assert.equal(outcome(await run(h, 'constructor')), 'duplicate');
    assert.equal(outcome(await run(h, 'toString', 'unknown-model')), 'unknown_model');
    // Saved again, both stay keys of the file.
    await h.controller.drain();
    const file = JSON.parse(fs.readFileSync(path.join(dataDir, 'state', 'controller.json'), 'utf8'));
    assert.deepEqual(Object.keys(file.requests).sort(), ['__proto__', 'constructor']);
    assertPrototypesUntouched();
});

test('the request map helpers: own entries only, data properties only, from any source object', () => {
    const plain = {};
    // An inherited member is never a record, even in an ordinary object.
    for (const requestId of SPECIAL) assert.equal(acceptedRequest(plain, requestId), null);
    const map = requestMap();
    for (const requestId of SPECIAL) recordRequest(map, requestId, { deploymentId: `d-${requestId}`, at: 'x' });
    assert.equal(Object.getPrototypeOf(map), Object.prototype, 'recording __proto__ did not set a prototype');
    assert.deepEqual(Object.keys(map).sort(), [...SPECIAL].sort());
    for (const requestId of SPECIAL) {
        const descriptor = Object.getOwnPropertyDescriptor(map, requestId);
        assert.equal(descriptor.value.deploymentId, `d-${requestId}`);
        assert.equal(typeof descriptor.get, 'undefined');
    }
    // Copying a parsed document keeps a `__proto__` key as data.
    const copied = requestMap(JSON.parse('{"__proto__":{"deploymentId":"d"},"ordinary-id-01":{"deploymentId":"e"}}'));
    assert.deepEqual(Object.keys(copied).sort(), ['__proto__', 'ordinary-id-01']);
    assert.equal(Object.getPrototypeOf(copied), Object.prototype);
    // Recording into an ordinary object defines data too: its prototype does not change.
    recordRequest(plain, '__proto__', { deploymentId: 'd' });
    assert.equal(Object.getPrototypeOf(plain), Object.prototype);
    assert.ok(Object.hasOwn(plain, '__proto__'));
    assertPrototypesUntouched();
});
