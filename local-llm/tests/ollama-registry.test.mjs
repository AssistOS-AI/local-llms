// The pin of an Ollama tag at Add (DS002): what is read from the registry, what is refused, and how a
// pinned entry is kept. The registry is a local server: nothing here reaches registry.ollama.ai.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { createController } from '../src/controller/deployments.mjs';
import { fetchOllamaRegistryManifest } from '../src/controller/ollamaStore.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { manifestBytes, registry, sha256 } from './ollama-fixture.mjs';

const fetchFrom = (base, options = {}) => (tag) => fetchOllamaRegistryManifest(tag, { baseUrl: base, ...options });
const refused = (code, pattern) => (error) => error.code === code && (!pattern || pattern.test(error.message));

test('the manifest request is for the tag the entry names: its namespace, name and version', async (t) => {
    const bytes = manifestBytes();
    const reg = await registry(t, {
        '/v2/library/qwen2.5/manifests/0.5b': { body: bytes },
        '/v2/library/llama3/manifests/latest': { body: bytes },
        '/v2/acme/tiny/manifests/q4_K_M': { body: bytes },
    });
    for (const tag of ['qwen2.5:0.5b', 'llama3', 'acme/tiny:q4_K_M']) {
        assert.deepEqual(await fetchFrom(reg.base)(tag), { manifestDigest: `sha256:${sha256(bytes)}`, size: 3500 }, tag);
    }
    assert.deepEqual(reg.requests.map((request) => request.url), [
        '/v2/library/qwen2.5/manifests/0.5b', '/v2/library/llama3/manifests/latest', '/v2/acme/tiny/manifests/q4_K_M',
    ]);
    for (const request of reg.requests) {
        assert.equal(request.accept, 'application/vnd.docker.distribution.manifest.v2+json');
        assert.equal(request.authorization, null);
    }
});

test('a missing tag, a failing registry and an unreachable one refuse the pin with a coded error', async (t) => {
    const reg = await registry(t, { '/v2/library/boom/manifests/1': { status: 500, body: 'oops' } });
    await assert.rejects(() => fetchFrom(reg.base)('nosuch:1'), refused('not_found', /has no tag nosuch:1/));
    await assert.rejects(() => fetchFrom(reg.base)('boom:1'), refused('pin_failed', /answered HTTP 500 for the manifest of boom:1/));
    // A port nothing listens on.
    const closed = await registry(t, {});
    const dead = closed.base;
    await assert.rejects(() => fetchOllamaRegistryManifest('a:1', { baseUrl: dead.replace(/:\d+$/, ':1') }), refused('pin_failed', /Could not read the manifest of a:1/));
    // A registry that never answers ends at the deadline.
    const hung = await registry(t, () => ({ hang: true }));
    await assert.rejects(() => fetchFrom(hung.base, { timeoutMs: 50 })('slow:1'), refused('pin_failed', /timed out/));
});

test('a redirect is followed only within the registry\'s own origin, and never for long', async (t) => {
    const bytes = manifestBytes({ layers: [10] });
    const elsewhere = await registry(t, { '/v2/library/m/manifests/1': { body: bytes } });
    const reg = await registry(t, (req) => ({
        '/v2/library/away/manifests/1': { status: 302, headers: { Location: `${elsewhere.base}/v2/library/m/manifests/1` } },
        '/v2/library/near/manifests/1': { status: 307, headers: { Location: '/v2/library/m/manifests/1' } },
        '/v2/library/m/manifests/1': { body: bytes },
        '/v2/library/loop/manifests/1': { status: 302, headers: { Location: '/v2/library/loop/manifests/1' } },
        '/v2/library/nolocation/manifests/1': { status: 302 },
    })[req.url]);
    assert.equal((await fetchFrom(reg.base)('near:1')).size, 510);
    await assert.rejects(() => fetchFrom(reg.base)('away:1'), refused('pin_failed', /redirected the manifest request of away:1 away from .*which is not followed/));
    assert.deepEqual(elsewhere.requests, [], 'the other origin was never contacted');
    await assert.rejects(() => fetchFrom(reg.base)('loop:1'), refused('pin_failed', /redirected/));
    await assert.rejects(() => fetchFrom(reg.base)('nolocation:1'), refused('pin_failed', /redirected/));
});

test('a manifest that is too large, not JSON, or not a usable version 2 manifest is not pinned', async (t) => {
    const good = JSON.parse(manifestBytes().toString());
    const serve = (value) => Buffer.from(typeof value === 'string' ? value : JSON.stringify(value));
    const digest = (char) => `sha256:${char.repeat(64)}`;
    const bodies = {
        huge: Buffer.alloc(1024 * 1024 + 1, 0x20),
        text: serve('this is not json'),
        version1: serve({ ...good, schemaVersion: 1 }),
        nolayers: serve({ ...good, layers: [] }),
        layersnotalist: serve({ ...good, layers: 'x' }),
        noconfig: serve({ schemaVersion: 2, layers: good.layers }),
        toomany: serve({ ...good, layers: Array.from({ length: 513 }, () => ({ digest: digest('1'), size: 1 })) }),
        baddigest: serve({ ...good, layers: [{ digest: 'sha256:../../x', size: 5 }] }),
        upperdigest: serve({ ...good, layers: [{ digest: `sha256:${'A'.repeat(64)}`, size: 5 }] }),
        zerosize: serve({ ...good, layers: [{ digest: digest('1'), size: 0 }] }),
        negativesize: serve({ ...good, layers: [{ digest: digest('1'), size: -4 }] }),
        fractional: serve({ ...good, layers: [{ digest: digest('1'), size: 1.5 }] }),
        stringsize: serve({ ...good, layers: [{ digest: digest('1'), size: '5' }] }),
        overflow: serve({ ...good, config: { digest: digest('c'), size: Number.MAX_SAFE_INTEGER }, layers: [{ digest: digest('1'), size: Number.MAX_SAFE_INTEGER }] }),
    };
    const reg = await registry(t, Object.fromEntries(Object.entries(bodies).map(([name, body]) => [`/v2/library/${name}/manifests/1`, { body }])));
    await assert.rejects(() => fetchFrom(reg.base)('huge:1'), refused('pin_failed', /larger than 1048576 bytes/));
    for (const name of Object.keys(bodies).filter((key) => key !== 'huge')) {
        await assert.rejects(() => fetchFrom(reg.base)(`${name}:1`), refused('pin_failed', /is not usable/), name);
    }
});

function controllerOf(t, { resolveOllama, registryEntries = [] } = {}) {
    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-ollama-pin-'));
    if (registryEntries.length) {
        fs.mkdirSync(path.join(dataDir, 'state'), { recursive: true });
        fs.writeFileSync(path.join(dataDir, 'state', 'controller.json'),
            JSON.stringify({ version: 1, deployment: null, params: {}, requests: {}, registry: registryEntries, cpuProfileMigration: '2026-09-30' }));
    }
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin' }, seedCatalog: [], stateStore: createStateStore({ dataDir }), resolveOllama,
        snapshot: async () => ({ gpu: { available: false, state: 'absent', reason: 'none' }, memory: {}, disk: {}, cpus: 1, cores: 1 }),
        detectRunner: () => ({ installed: false, version: null, reason: null }),
    });
    // The drain saves state and writes the log, so the directory goes only after it.
    t.after(async () => {
        await controller.drain();
        fs.rmSync(dataDir, { recursive: true, force: true });
    });
    return controller;
}

const DIGEST = `sha256:${'a'.repeat(64)}`;
const entry = (source, id = 'user-olla') => ({ id, sources: { ollama: { type: 'ollama', tag: 'qwen2.5:0.5b', ...source } } });

test('Add keeps a pin that is whole, completes one that is not, refuses one the registry contradicts, and fails closed', async (t) => {
    const asked = [];
    const resolveOllama = async (tag) => { asked.push(tag); return { manifestDigest: DIGEST, size: 3500 }; };
    const controller = controllerOf(t, { resolveOllama });
    // A whole pin is kept as given, and the registry is not asked.
    const whole = await controller.addModel(entry({ manifestDigest: `sha256:${'d'.repeat(64)}`, size: 42 }, 'whole'));
    assert.deepEqual(whole.model.sources.ollama.size, 42);
    assert.deepEqual(asked, []);
    // A digest alone gets its size, and only if the registry agrees with it.
    assert.deepEqual((await controller.addModel(entry({ manifestDigest: DIGEST }, 'digest-only'))).model.sources.ollama,
        { type: 'ollama', tag: 'qwen2.5:0.5b', manifestDigest: DIGEST, size: 3500 });
    // A size alone gets its digest, and only if the registry agrees with it.
    assert.deepEqual((await controller.addModel(entry({ size: 3500 }, 'size-only'))).model.sources.ollama.manifestDigest, DIGEST);
    // A tag that moved since it was pinned is never re-pinned silently.
    await assert.rejects(() => controller.addModel(entry({ manifestDigest: `sha256:${'e'.repeat(64)}` }, 'moved')),
        refused('identity_changed', /now resolves to sha256:a+, not the pinned sha256:e+; update the model entry without its manifestDigest and size to accept it/));
    await assert.rejects(() => controller.addModel(entry({ size: 99 }, 'resized')), refused('identity_changed', /is now 3500 bytes, not the pinned 99/));
    assert.deepEqual(controller.state.registry.map((model) => model.id), ['whole', 'digest-only', 'size-only']);
    // Registry down: the Add fails, and nothing is stored.
    const down = controllerOf(t, { resolveOllama: async () => { throw Object.assign(new Error('registry.ollama.ai is down'), { code: 'pin_failed' }); } });
    await assert.rejects(() => down.addModel(entry({}, 'unreachable')), { code: 'pin_failed' });
    assert.deepEqual(down.state.registry, []);
});

test('an Update that names a bare tag pins it afresh, which is how a legacy entry or a moved tag is accepted', async (t) => {
    const legacy = entry({}, 'legacy');
    const controller = controllerOf(t, { registryEntries: [legacy], resolveOllama: async () => ({ manifestDigest: DIGEST, size: 3500 }) });
    const updated = await controller.updateModel(entry({}, 'legacy'));
    assert.deepEqual(updated.model.sources.ollama, { type: 'ollama', tag: 'qwen2.5:0.5b', manifestDigest: DIGEST, size: 3500 });
    assert.deepEqual(controller.state.registry[0].sources.ollama, updated.model.sources.ollama);
    // A model with no Ollama source never asks the registry.
    const asked = [];
    const hf = controllerOf(t, { resolveOllama: async (tag) => { asked.push(tag); return { manifestDigest: DIGEST, size: 1 }; } });
    await hf.addModel({ id: 'hf-only', sources: { gguf: { type: 'huggingface', repo: 'acme/models', file: 'm.gguf', revision: 'main', commit: 'a'.repeat(40), size: 10, sha256: 'b'.repeat(64) } } });
    assert.deepEqual(asked, []);
});
