// The registry list and its admin tools (DS002): validation (A18), the key's path into the agent (A12, N4),
// snapshots and the in-queue re-check of Add and Update (A26, N5), and missing or disabled registries (A28).
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { startAgent } from '../src/agentProcess.mjs';
import { LocalLlmError } from '../src/errors.mjs';
import { createCommandQueue } from '../src/controller/commandQueue.mjs';
import { createRegistryService, normalizeRegistryUrl, MAX_ADMIN_REGISTRIES, REGISTRY_ID_RE } from '../src/controller/registries.mjs';
import {
    KEY, RUN, codeOf, deferred, readJson, registriesFile, registryHarness, tempDir, until,
} from './registry-fixture.mjs';

const MIRROR = Object.freeze({ id: 'mirror', kind: 'huggingface', url: 'https://mirror.example/hf' });
const COMMIT = 'c'.repeat(40);
const pinnedGguf = (extra = {}) => ({ type: 'huggingface', repo: 'acme/model', file: 'model.gguf', revision: 'main', ...extra });
const userEntry = (id, sources) => ({ id, displayName: id, sources });

// A service with its operations on a queue of its own, for the rules that need no controller.
function service(t, { dataDir = tempDir(t), credentialsKey = KEY, references = () => ({ entries: [], active: null }) } = {}) {
    const registries = createRegistryService({ dataDir, credentialsKey });
    const logged = [];
    const ops = registries.operations({ queue: createCommandQueue(), log: (line) => logged.push(line), references });
    return { registries, ops, logged, dataDir };
}

// ------------------------------------------------------------------ A18

test('A18: a registry url must be https with a lowercase DNS host, no userinfo, query, fragment or IP literal', async (t) => {
    const { ops } = service(t);
    const refused = [
        ['http', 'http://mirror.example'],
        ['userinfo', 'https://user:pass@mirror.example'],
        ['user only', 'https://user@mirror.example'],
        ['query', 'https://mirror.example/hf?a=1'],
        ['empty query', 'https://mirror.example/hf?'],
        ['fragment', 'https://mirror.example/hf#x'],
        ['IPv4 literal', 'https://10.0.0.1'],
        ['IPv6 literal', 'https://[::1]'],
        ['public IPv4 literal', 'https://8.8.8.8'],
        ['decimal 127.0.0.1', 'https://2130706433'],
        ['hex 127.0.0.1', 'https://0x7f.1'],
        ['short 127.0.0.1', 'https://127.1'],
        ['uppercase host', 'https://Mirror.example'],
        ['unicode host', 'https://bücher.example'],
        ['dot segment', 'https://mirror.example/a/../b'],
        ['empty segment', 'https://mirror.example/a//b'],
        ['encoded segment', 'https://mirror.example/a/%2e%2e/b'],
        ['prefix of 257 characters', `https://mirror.example/${'p'.repeat(256)}`],
        ['not a URL', 'mirror.example'],
        ['default port written out', 'https://mirror.example:443'],
    ];
    for (const [what, url] of refused) {
        assert.equal(await codeOf(ops.saveRegistry({ registry: { ...MIRROR, url } })), 'invalid_registry', what);
    }
    for (const [kind, url] of [['oci', 'https://registry.example/v2'], ['modelscope', 'https://www.modelscope.ai/models']]) {
        assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'prefixed', kind, url } })), 'invalid_registry', `a prefix on ${kind}`);
    }
    // Accepted: localhost with a port (the live negative control), an IDN in its DNS form, a 256-character prefix.
    const accepted = [
        ['local', 'oci', 'https://localhost:8443'],
        ['idn', 'huggingface', 'https://xn--bcher-kva.example'],
        ['long', 'huggingface', `https://mirror.example/${'p'.repeat(255)}`],
        ['slash', 'huggingface', 'https://mirror.example/hf/'],
    ];
    for (const [id, kind, url] of accepted) {
        const saved = await ops.saveRegistry({ registry: { id, kind, url } });
        assert.equal(saved.registry.url, url.replace(/\/$/, ''), id);
    }
    assert.equal(normalizeRegistryUrl('https://localhost:8443'), 'https://localhost:8443');
    assert.equal((await ops.listRegistries()).registries.find((entry) => entry.id === 'local').authOrigins[0], 'https://localhost:8443');
});

test('A18: ids, kinds, unknown keys, the 32-entry cap and the built-ins are enforced', async (t) => {
    const { ops } = service(t);
    for (const id of ['a', 'x'.repeat(33), 'Hub', 'x_y', '-ab', 'ab.c', 42]) {
        assert.equal(await codeOf(ops.saveRegistry({ registry: { ...MIRROR, id } })), 'invalid_registry', String(id));
    }
    assert.ok(REGISTRY_ID_RE.test('x'.repeat(32)) && REGISTRY_ID_RE.test('ab'));
    assert.equal(await codeOf(ops.saveRegistry({ registry: { ...MIRROR, extra: 1 } })), 'invalid_registry', 'unknown key');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { ...MIRROR, kind: 'ngc' } })), 'invalid_registry', 'unknown kind');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'nokind', url: 'https://a.example' } })), 'invalid_registry');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { ...MIRROR, label: 'x'.repeat(61) } })), 'invalid_registry', 'label');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { ...MIRROR, authOrigins: ['https://a.example'] } })), 'invalid_registry', 'authOrigins off oci');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'o', kind: 'oci', url: 'https://r.example', authOrigins: Array(5).fill('https://a.example') } })),
        'invalid_registry', 'five auth origins');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'oo', kind: 'oci', url: 'https://r.example', authOrigins: ['http://a.example'] } })),
        'invalid_registry', 'an http auth origin');
    assert.equal(await codeOf(ops.saveRegistry({ registry: null })), 'invalid_registry');
    // Built-ins: huggingface is read-only; the others are reserved ids that may only be enabled or disabled.
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'huggingface', enabled: false } })), 'read_only');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'dockerhub', kind: 'oci', url: 'https://registry.example' } })), 'invalid_registry');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'modelscope', url: 'https://www.modelscope.ai' } })), 'invalid_registry', 'not re-pointable');
    for (const id of ['huggingface', 'dockerhub', 'ghcr', 'modelscope']) {
        assert.equal(await codeOf(ops.removeRegistry({ id })), 'read_only', `remove ${id}`);
    }
    assert.equal((await ops.saveRegistry({ registry: { id: 'dockerhub', enabled: false } })).registry.enabled, false);
    // A kind change is refused; the url of an unreferenced registry may change.
    await ops.saveRegistry({ registry: MIRROR });
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'mirror', kind: 'modelscope' } })), 'invalid_registry', 'kind change');
    assert.equal((await ops.saveRegistry({ registry: { id: 'mirror', url: 'https://mirror.example/v2' } })).registry.url, 'https://mirror.example/v2');
    // 32 administrator entries at most.
    for (let index = (await ops.listRegistries()).registries.filter((entry) => !entry.builtIn).length; index < MAX_ADMIN_REGISTRIES; index += 1) {
        await ops.saveRegistry({ registry: { id: `reg-${index}`, kind: 'oci', url: `https://r${index}.example` } });
    }
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'one-more', kind: 'oci', url: 'https://more.example' } })), 'invalid_registry');
    assert.equal((await ops.listRegistries()).registries.filter((entry) => !entry.builtIn).length, MAX_ADMIN_REGISTRIES);
});

test('A18: a url or kind change and a remove are refused while a stored entry or the running job uses the registry', async (t) => {
    // A raw stored entry naming `mirror` (this catalog cannot read it yet, so it is kept raw) and a running job using `jobreg`.
    const entries = [userEntry('mirrored', { gguf: pinnedGguf({ registry: 'mirror', commit: COMMIT }) })];
    let active = null;
    const { ops } = service(t, { references: () => ({ entries, active }) });
    await ops.saveRegistry({ registry: MIRROR });
    await ops.saveRegistry({ registry: { id: 'jobreg', kind: 'oci', url: 'https://job.example' } });
    active = { modelId: 'running-model', artifact: { type: 'oci', registry: 'jobreg' } };
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'mirror', url: 'https://other.example' } })), 'registry_in_use');
    assert.equal(await codeOf(ops.saveRegistry({ registry: { id: 'mirror', kind: 'oci' } })), 'registry_in_use');
    await assert.rejects(ops.removeRegistry({ id: 'mirror' }), (error) => error.code === 'registry_in_use' && /mirrored/.test(error.message)
        && error.details.models.includes('mirrored'));
    assert.equal(await codeOf(ops.removeRegistry({ id: 'jobreg' })), 'registry_in_use');
    // Allowed while referenced: label, disable.
    assert.equal((await ops.saveRegistry({ registry: { id: 'mirror', label: 'Mirror', enabled: false } })).registry.enabled, false);
    const list = await ops.listRegistries();
    assert.deepEqual(list.registries.find((entry) => entry.id === 'mirror').usedBy, ['mirrored']);
    assert.deepEqual(list.registries.find((entry) => entry.id === 'jobreg').usedBy, ['running-model']);
    // The job ends: jobreg is free again.
    active = null;
    assert.deepEqual(await ops.removeRegistry({ id: 'jobreg' }), { removed: 'jobreg' });
    // Orphan (probe): removing an unknown registry, and a credential for it, is invalid_registry.
    assert.equal(await codeOf(ops.removeRegistry({ id: 'nope' })), 'invalid_registry');
    assert.equal(await codeOf(ops.setRegistryCredential({ id: 'nope', token: 'x' })), 'invalid_registry');
});

test('A18: through the controller, a stored entry naming a registry blocks its url change and removal until the entry is removed', async (t) => {
    // Raw, as this catalog keeps an entry it cannot read yet; the controller passes its state.registry to the service.
    const entry = userEntry('mirrored', { gguf: pinnedGguf({ registry: 'mirror', commit: COMMIT, size: 1000, sha256: 'b'.repeat(64) }) });
    const h = registryHarness(t, { initialState: { version: 1, deployment: null, params: {}, requests: {}, registry: [entry] } });
    await h.controller.saveRegistry({ registry: MIRROR });
    assert.deepEqual((await h.controller.listRegistries()).registries.find((registry) => registry.id === 'mirror').usedBy, ['mirrored']);
    assert.equal(await codeOf(h.controller.saveRegistry({ registry: { id: 'mirror', url: 'https://other.example/hf' } })), 'registry_in_use');
    assert.equal(await codeOf(h.controller.removeRegistry({ id: 'mirror' })), 'registry_in_use');
    // Twin: once the entry is gone, both succeed.
    await h.controller.removeModel({ modelId: 'mirrored' });
    assert.deepEqual((await h.controller.listRegistries()).registries.find((registry) => registry.id === 'mirror').usedBy, []);
    assert.equal((await h.controller.saveRegistry({ registry: { id: 'mirror', url: 'https://other.example/hf' } })).registry.url, 'https://other.example/hf');
    assert.deepEqual(await h.controller.removeRegistry({ id: 'mirror' }), { removed: 'mirror' });
});

test('the list shows the built-ins, kinds, formats and credential states, and registries.json is rewritten, never deleted', async (t) => {
    const { ops, dataDir } = service(t);
    let list = await ops.listRegistries();
    assert.deepEqual(list.registries.map((entry) => [entry.id, entry.kind, entry.url, entry.enabled, entry.builtIn, entry.editable]), [
        ['huggingface', 'huggingface', 'https://huggingface.co', true, true, false],
        ['dockerhub', 'oci', 'https://registry-1.docker.io', true, true, false],
        ['ghcr', 'oci', 'https://ghcr.io', true, true, false],
        ['modelscope', 'modelscope', 'https://www.modelscope.cn', true, true, false],
    ]);
    assert.deepEqual(list.registries.find((entry) => entry.id === 'dockerhub').authOrigins, ['https://auth.docker.io']);
    assert.deepEqual(list.registries.find((entry) => entry.id === 'dockerhub').formats, ['gguf', 'hf']);
    assert.deepEqual(list.registries.find((entry) => entry.id === 'modelscope').formats, ['gguf', 'hf', 'exl3']);
    assert.deepEqual(list.credentialStorage, { available: true, reason: null });
    assert.equal(fs.existsSync(registriesFile(dataDir)), false, 'a missing file means the defaults');
    await ops.saveRegistry({ registry: MIRROR });
    await ops.saveRegistry({ registry: { id: 'ghcr', enabled: false } });
    assert.deepEqual(readJson(registriesFile(dataDir)).builtins, { dockerhub: { enabled: true }, ghcr: { enabled: false }, modelscope: { enabled: true } });
    // Back to the defaults: the file stays, holding the defaults.
    await ops.removeRegistry({ id: 'mirror' });
    await ops.saveRegistry({ registry: { id: 'ghcr', enabled: true } });
    assert.equal(fs.existsSync(registriesFile(dataDir)), true);
    assert.deepEqual(readJson(registriesFile(dataDir)), {
        version: 1, registries: [], builtins: { dockerhub: { enabled: true }, ghcr: { enabled: true }, modelscope: { enabled: true } },
    });
    // A restart reads the same list.
    list = await service(t, { dataDir }).ops.listRegistries();
    assert.deepEqual(list.registries.map((entry) => entry.id), ['huggingface', 'dockerhub', 'ghcr', 'modelscope']);
});

test('a hand-edited registries.json keeps its valid entries and logs the rest; a corrupt one is moved aside', async (t) => {
    const dataDir = tempDir(t);
    fs.mkdirSync(path.join(dataDir, 'state'), { recursive: true });
    fs.writeFileSync(registriesFile(dataDir), JSON.stringify({ version: 1, registries: [
        MIRROR, { ...MIRROR, url: 'https://dup.example' }, { id: 'ip', kind: 'oci', url: 'https://10.0.0.1' }, { id: 'dockerhub', kind: 'oci', url: 'https://x.example' },
    ], builtins: { modelscope: { enabled: false } } }));
    const loaded = service(t, { dataDir });
    const list = await loaded.ops.listRegistries();
    assert.deepEqual(list.registries.filter((entry) => !entry.builtIn).map((entry) => [entry.id, entry.url]), [['mirror', 'https://mirror.example/hf']]);
    assert.equal(list.registries.find((entry) => entry.id === 'modelscope').enabled, false);
    assert.equal(loaded.logged.filter((line) => /registries\.json entry is not used/.test(line)).length, 3);
    fs.writeFileSync(registriesFile(dataDir), '{ not json');
    const corrupt = service(t, { dataDir });
    assert.deepEqual((await corrupt.ops.listRegistries()).registries.map((entry) => entry.id), ['huggingface', 'dockerhub', 'ghcr', 'modelscope']);
    assert.ok(fs.readdirSync(path.join(dataDir, 'state')).some((name) => name.startsWith('registries.json.corrupt-')));
    assert.ok(corrupt.logged.some((line) => /moved aside/.test(line)));
});

// ------------------------------------------------------------------ A12 (N4)

test('A12: startAgent takes the credentials key out of the environment and hands it only to the controller and the registry service', async (t) => {
    const saved = process.env.LOCAL_LLM_CREDENTIALS_KEY;
    const listeners = Object.fromEntries(['SIGTERM', 'SIGINT', 'SIGHUP'].map((signal) => [signal, process.listeners(signal)]));
    t.after(() => {
        if (saved === undefined) delete process.env.LOCAL_LLM_CREDENTIALS_KEY;
        else process.env.LOCAL_LLM_CREDENTIALS_KEY = saved;
        for (const [signal, before] of Object.entries(listeners)) {
            for (const listener of process.listeners(signal)) if (!before.includes(listener)) process.removeListener(signal, listener);
        }
    });
    const dataDir = tempDir(t);
    process.env.LOCAL_LLM_CREDENTIALS_KEY = KEY;
    const seen = {};
    const registryService = { stub: 'registries' };
    startAgent({
        controllerOptions: { dataDir },
        makeRegistryService: (options) => { seen.registry = options; return registryService; },
        makeController: (options) => { seen.controller = options; return { drain: async () => {} }; },
        startServer: async () => ({ close: async () => {} }),
        spawnProcess: (command, args, options) => {
            seen.spawn = { command, args, env: options.env };
            const child = new EventEmitter();
            child.kill = () => {};
            return child;
        },
    });
    assert.equal(process.env.LOCAL_LLM_CREDENTIALS_KEY, undefined, 'deleted from process.env at once');
    await until(() => seen.spawn);
    assert.equal(seen.registry.credentialsKey === KEY, true, 'createRegistryService received the key');
    assert.equal(seen.registry.dataDir, dataDir);
    assert.equal(seen.controller.credentialsKey === KEY, true, 'createController received the key');
    assert.equal(seen.controller.registryService, registryService);
    assert.equal(Object.hasOwn(seen.spawn.env, 'LOCAL_LLM_CREDENTIALS_KEY'), false, 'AgentServer\'s environment');
    assert.equal(Object.values(seen.spawn.env).includes(KEY), false, 'no variable holds the key');
    assert.ok(seen.spawn.env.LOCAL_LLM_CONTROL_TOKEN, 'AgentServer still gets its control token');
    // Twin: without the variable, both receive null.
    delete seen.spawn;
    startAgent({
        controllerOptions: { dataDir },
        makeRegistryService: (options) => { seen.registry = options; return registryService; },
        makeController: (options) => { seen.controller = options; return { drain: async () => {} }; },
        startServer: async () => ({ close: async () => {} }),
        spawnProcess: (command, args, options) => { seen.spawn = { env: options.env }; const child = new EventEmitter(); child.kill = () => {}; return child; },
    });
    await until(() => seen.spawn);
    assert.equal(seen.registry.credentialsKey, null);
    assert.equal(seen.controller.credentialsKey, null);
});

test('A12: only agentProcess.mjs reads LOCAL_LLM_CREDENTIALS_KEY from an environment', () => {
    const root = new URL('../src/', import.meta.url);
    const sources = [];
    const walk = (dir) => {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) walk(full);
            else if (entry.name.endsWith('.mjs')) sources.push(full);
        }
    };
    walk(root.pathname);
    const readers = sources.filter((file) => /env\s*(?:\?\.\s*|\.\s*|\[\s*['"`])LOCAL_LLM_CREDENTIALS_KEY/.test(fs.readFileSync(file, 'utf8')));
    assert.deepEqual(readers.map((file) => path.relative(root.pathname, file)), ['agentProcess.mjs']);
});

test('A12: the runner environment has neither the credentials key nor HF_TOKEN', async (t) => {
    const h = registryHarness(t, { env: { PATH: '/usr/bin', HF_TOKEN: 'hf_runner_check', LOCAL_LLM_CREDENTIALS_KEY: KEY } });
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const [runner] = h.runners.started;
    assert.equal(Object.hasOwn(runner.env, 'LOCAL_LLM_CREDENTIALS_KEY'), false);
    assert.equal(Object.hasOwn(runner.env, 'HF_TOKEN'), false);
    assert.equal(Object.values(runner.env).some((value) => String(value).includes(KEY) || String(value).includes('hf_runner_check')), false);
    await h.controller.stop();
});

// ------------------------------------------------------------------ A26 and N5

test('A26: a snapshot keeps its record and credential while the registry is disabled and its credential cleared; the next one sees the change', async (t) => {
    const { registries, ops } = service(t);
    await ops.saveRegistry({ registry: MIRROR });
    await ops.setRegistryCredential({ id: 'mirror', token: 'snapshot-token' });
    const taken = registries.snapshot('mirror');
    await ops.clearRegistryCredential({ id: 'mirror' });
    await ops.saveRegistry({ registry: { id: 'mirror', enabled: false } });
    assert.equal(taken.credential.token === 'snapshot-token', true, 'the snapshot keeps its decrypted credential');
    assert.equal(Object.isFrozen(taken), true);
    assert.equal(JSON.stringify(taken).includes('snapshot-token'), false, 'the credential is not serialized with the record');
    assert.equal(await codeOf((async () => registries.snapshot('mirror'))()), 'registry_disabled');
    await ops.saveRegistry({ registry: { id: 'mirror', enabled: true } });
    assert.equal(registries.snapshot('mirror').credential, null, 'the next snapshot sees the cleared credential');
});

test('N5: a registry removed and re-created with another url or kind, removed, or disabled during a pin fails the queue re-check', async (t) => {
    const { registries, ops } = service(t);
    const source = { gguf: pinnedGguf({ registry: 'mirror' }) };
    const pin = async () => {
        await ops.saveRegistry({ registry: MIRROR });
        return registries.snapshotSources(source);
    };
    // Removed and re-created with another url.
    let taken = await pin();
    await ops.removeRegistry({ id: 'mirror' });
    await ops.saveRegistry({ registry: { ...MIRROR, url: 'https://evil.example/hf' } });
    assert.equal(await codeOf((async () => registries.recheck(taken))()), 'registry_changed');
    await ops.removeRegistry({ id: 'mirror' });
    // Removed and re-created with another kind (a bare origin, so modelscope accepts it).
    taken = await pin();
    await ops.removeRegistry({ id: 'mirror' });
    await ops.saveRegistry({ registry: { id: 'mirror', kind: 'modelscope', url: 'https://mirror.example' } });
    assert.equal(await codeOf((async () => registries.recheck(taken))()), 'registry_changed');
    await ops.removeRegistry({ id: 'mirror' });
    // Removed and not re-created.
    taken = await pin();
    await ops.removeRegistry({ id: 'mirror' });
    assert.equal(await codeOf((async () => registries.recheck(taken))()), 'registry_unknown');
    // Disabled.
    taken = await pin();
    await ops.saveRegistry({ registry: { id: 'mirror', enabled: false } });
    assert.equal(await codeOf((async () => registries.recheck(taken))()), 'registry_disabled');
    // Twin: re-created identically, the re-check passes.
    await ops.removeRegistry({ id: 'mirror' });
    taken = await pin();
    await ops.removeRegistry({ id: 'mirror' });
    await ops.saveRegistry({ registry: MIRROR });
    assert.doesNotThrow(() => registries.recheck(taken));
    assert.deepEqual(taken.map((entry) => [entry.format, entry.registry.id, entry.registry.kind, entry.registry.url]),
        [['gguf', 'mirror', 'huggingface', 'https://mirror.example/hf']]);
});

test('N5: a registry whose kind cannot serve the source is invalid_model at Add and Update, and registry_changed at Run', async (t) => {
    const { registries } = service(t);
    for (const [type, registry] of [['oci', 'modelscope'], ['oci-snapshot', 'modelscope'], ['huggingface', 'dockerhub'], ['hf-snapshot', 'ghcr']]) {
        const source = { type, registry };
        assert.equal(await codeOf((async () => registries.snapshotSources({ gguf: source }))()), 'invalid_model', `${type} on ${registry}`);
        assert.equal(await codeOf((async () => registries.assertRunnable(source))()), 'registry_changed', `${type} on ${registry} at Run`);
    }
    // Sources without a registry are Hugging Face's; Ollama tags never name one.
    assert.deepEqual(registries.snapshotSources({ gguf: pinnedGguf(), ollama: { type: 'ollama', tag: 'x:1' } }).map((entry) => entry.registry.id), ['huggingface']);
    assert.doesNotThrow(() => registries.assertRunnable({ type: 'ollama', tag: 'x:1' }));
});

test('A28: a missing registry is registry_unknown and a disabled one registry_disabled for pins and Runs; stores never ask', async (t) => {
    const { registries, ops } = service(t);
    const gone = { type: 'hf-snapshot', registry: 'gone', repo: 'acme/model', commit: COMMIT };
    assert.equal(await codeOf((async () => registries.snapshotSources({ hf: gone }))()), 'registry_unknown');
    assert.equal(await codeOf((async () => registries.assertRunnable(gone))()), 'registry_unknown');
    await ops.saveRegistry({ registry: { id: 'modelscope', enabled: false } });
    const disabled = { ...gone, registry: 'modelscope' };
    assert.equal(await codeOf((async () => registries.snapshotSources({ hf: disabled }))()), 'registry_disabled');
    assert.equal(await codeOf((async () => registries.assertRunnable(disabled))()), 'registry_disabled');
});

test('A28: a stored entry naming a missing registry still loads in the overview and can be removed', async (t) => {
    const entry = userEntry('orphaned', { gguf: pinnedGguf({ registry: 'gone', commit: COMMIT, size: 1000, sha256: 'b'.repeat(64) }) });
    const h = registryHarness(t, { initialState: { version: 1, deployment: null, params: {}, requests: {}, registry: [entry] } });
    const overview = await h.controller.overview();
    assert.ok(overview.unsupportedModels.some((model) => model.id === 'orphaned'), 'listed, raw, while this catalog cannot read it');
    assert.deepEqual((await h.controller.listRegistries()).registries.find((registry) => registry.id === 'huggingface').usedBy, []);
    assert.deepEqual(await h.controller.removeModel({ modelId: 'orphaned' }), {
        removed: 'orphaned', note: 'The entry was not supported by this catalog; downloaded weights, if any, stay on disk.',
    });
});

test('N5: the Add and Update queue blocks re-check the snapshot taken before the pin, and a failed re-check saves nothing', async (t) => {
    const pinGate = deferred();
    const events = [];
    let changed = false;
    const dataDir = tempDir(t);
    const real = createRegistryService({ dataDir, credentialsKey: KEY });
    const registryService = {
        ...real,
        snapshotSources: (sources) => { events.push('snapshot'); return real.snapshotSources(sources); },
        recheck: (taken) => {
            events.push('recheck');
            if (changed) throw new LocalLlmError('registry_changed', 'Registry changed while the model was being pinned.');
            return real.recheck(taken);
        },
    };
    const resolveHf = async () => {
        events.push('pin');
        await pinGate.promise;
        return { commit: COMMIT, size: 1000, sha256: 'b'.repeat(64) };
    };
    const h = registryHarness(t, { dataDir, resolveHf, extra: { registryService } });
    const draft = userEntry('pinned-model', { gguf: pinnedGguf() });
    const adding = h.controller.addModel(draft);
    await until(() => events.includes('pin'));
    // An edit is not held up by the pin: it runs through the queue at once.
    await h.controller.saveRegistry({ registry: MIRROR });
    changed = true;
    pinGate.resolve();
    assert.equal(await codeOf(adding), 'registry_changed');
    assert.deepEqual(events, ['snapshot', 'pin', 'recheck']);
    assert.equal(h.controller.state.registry.length, 0, 'state.registry unchanged');
    assert.equal(readJson(path.join(dataDir, 'state', 'controller.json')).registry.length, 0);
    // Twin: the same Add with an unchanged registry is saved.
    changed = false;
    events.length = 0;
    assert.equal((await h.controller.addModel(draft)).model.id, 'pinned-model');
    assert.deepEqual(events, ['snapshot', 'pin', 'recheck']);
    // Update: the same rule.
    changed = true;
    events.length = 0;
    const before = JSON.stringify(h.controller.state.registry);
    assert.equal(await codeOf(h.controller.updateModel({ ...draft, displayName: 'Changed', sources: { gguf: pinnedGguf({ revision: 'other' }) } })), 'registry_changed');
    assert.deepEqual(events, ['snapshot', 'pin', 'recheck']);
    assert.equal(JSON.stringify(h.controller.state.registry), before, 'state.registry unchanged');
});
