// OCI weight stores (DS002): paths and keys under _oci/<id> from the source alone, the N6 leaf that keeps one model's
// directory out of another's (A17), downloads through the registry session with the bearer at home only, blob digests
// (A13), the header re-read after a Run, and what a running job, a drain and the N5 re-check do (A26). The fakes are
// in oci-fixture.mjs; oci-registry.test.mjs holds the session, lookups and pins.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { createCommandQueue } from '../src/controller/commandQueue.mjs';
import { validateModel } from '../src/controller/catalog.mjs';
import { downloadArtifact, inspectArtifact, removeArtifact, resolveHuggingFaceArtifact } from '../src/controller/downloader.mjs';
import { OCI_IDENTITY_KEYS, ociGgufPaths, ociSnapshotPaths } from '../src/controller/ociStore.mjs';
import { createRegistryService } from '../src/controller/registries.mjs';
import { createWeightStores } from '../src/controller/weightStores.mjs';
import {
    BASIC, CDN_SIGNATURE, CONFIG_JSON, GGUF, REG, REPO, SAFETENSORS, SECRET, TAG, USERNAME,
    addReg, authOf, cncfGguf, createOciWorld, digestOf, entryOf, ociController, ociSnapshotSource, ociSource, scratch, sha256Of, shardBytes,
} from './oci-fixture.mjs';
import { KEY, codeOf, deferred, until } from './registry-fixture.mjs';

const H = digestOf(Buffer.from('manifest one'));
const H2 = digestOf(Buffer.from('manifest two'));
const hexOf = (digest) => digest.slice('sha256:'.length);
const SHARDS = [shardBytes(0, 2), shardBytes(1, 2)];
const pinnedGguf = (extra = {}) => ({ type: 'oci', registry: 'reg', repository: REPO, tag: TAG, manifestDigest: H, layout: 'cncf', file: 'm-Q4_K_M.gguf', size: GGUF.length, sha256: sha256Of(GGUF), ...extra });
const pinnedSplit = (extra = {}) => pinnedGguf({
    file: 'm-00001-of-00002.gguf', sha256: undefined, size: SHARDS[0].length + SHARDS[1].length,
    shards: SHARDS.map((bytes, index) => ({ file: `m-0000${index + 1}-of-00002.gguf`, size: bytes.length, sha256: sha256Of(bytes) })), ...extra,
});
const SNAPSHOT_FILES = [['config.json', CONFIG_JSON], ['model.safetensors', SAFETENSORS]];
const pinnedSnapshot = (extra = {}) => ({
    type: 'oci-snapshot', registry: 'reg', repository: REPO, tag: TAG, manifestDigest: H,
    files: SNAPSHOT_FILES.map(([name, bytes]) => ({ path: name, size: bytes.length, sha256: sha256Of(bytes) })), size: CONFIG_JSON.length + SAFETENSORS.length, ...extra,
});

// A store over the real downloader and a real registry service; `registries` may be replaced to prove it is never asked.
function storeOn(t, world, { disk = scratch(t), registries = undefined } = {}) {
    const service = createRegistryService({ dataDir: disk.dir, credentialsKey: KEY, hfBaseUrl: 'https://hf.test' });
    const ops = service.operations({ queue: createCommandQueue(), references: () => ({ entries: [], active: null }) });
    const stores = createWeightStores({
        dataDir: disk.dir, env: {}, hfBaseUrl: 'https://hf.test',
        inspect: inspectArtifact, download: downloadArtifact, remove: removeArtifact, resolveHf: resolveHuggingFaceArtifact,
        state: () => ({}), save: () => {}, activeArtifact: () => null,
        registryTransport: world?.transport ?? null, registries: registries ?? service,
        downloadOci: undefined,
    });
    return { ops, registries: service, stores, dir: disk.dir };
}
async function storeWithRegistry(t, world) {
    const made = storeOn(t, world);
    await made.ops.saveRegistry({ registry: REG });
    await made.ops.setRegistryCredential({ id: 'reg', token: SECRET, tokenUsername: USERNAME });
    for (const bytes of [GGUF, ...SHARDS, ...SNAPSHOT_FILES.map(([, value]) => value)]) world.blobs.set(digestOf(bytes), bytes);
    return made;
}

// ------------------------------------------------------------------ paths and keys (A17, N6)

test('A17: OCI paths and keys are under _oci/<id> with the @sha256-<hex> leaf, from the source alone (the registry list is never read)', async (t) => {
    const never = new Proxy({}, { get: (_, name) => () => assert.fail(`the registry service was asked for ${String(name)}`) });
    const { stores, dir } = storeOn(t, null, { registries: never });
    const leaf = `smollm2@sha256-${hexOf(H)}`;
    assert.equal(stores.oci.key(pinnedGguf()), `gguf:_oci/reg/${REPO}@${H}/m-Q4_K_M.gguf`);
    assert.equal(stores.oci.key(pinnedSplit()), `gguf:_oci/reg/${REPO}@${H}/m-00001-of-00002.gguf#2`);
    assert.equal(stores['oci-snapshot'].key(pinnedSnapshot()), `hf:_oci/reg/${REPO}@${H}`);
    assert.equal(stores.oci.paths(pinnedGguf()).file, path.join(dir, 'models', 'gguf', '_oci', 'reg', 'ai', leaf, 'm-Q4_K_M.gguf'));
    assert.equal(stores.oci.paths(pinnedGguf()).partial, path.join(dir, 'models', 'gguf', '_oci', 'reg', 'ai', leaf, 'm-Q4_K_M.gguf.partial'));
    assert.deepEqual(ociSnapshotPaths({ root: path.join(dir, 'models', 'hf'), source: pinnedSnapshot() }), {
        dir: path.join(dir, 'models', 'hf', '_oci', 'reg', 'ai', leaf),
        stateDir: path.join(dir, 'models', 'hf', '.state', '_oci', 'reg', 'ai', leaf),
    });
    assert.deepEqual([...OCI_IDENTITY_KEYS], ['registry', 'repository', 'manifestDigest', 'file', 'size', 'sha256']);
    for (const [store, source] of [[stores.oci, pinnedGguf()], [stores.oci, pinnedSplit()], [stores['oci-snapshot'], pinnedSnapshot()]]) {
        assert.equal((await store.state(source)).state, 'absent');
        assert.equal((await store.plan(source)).bytesNeeded, source.size);
        assert.equal(await store.remove(source), 0);
        assert.deepEqual(store.sizes(source).reduce((sum, size) => sum + size, 0), source.size);
    }
    assert.equal((await stores.oci.state({ ...pinnedGguf(), manifestDigest: undefined })).state, 'unpinned');
    assert.throws(() => ociGgufPaths({ root: dir, source: pinnedGguf(), file: '../x.gguf' }), { code: 'invalid_model' });
});

test('N6: ns/m at H and ns/m/<H> at H2 share no prefix for the GGUF and snapshot directories; deleting one leaves the other byte-identical', async (t) => {
    const world = await createOciWorld(t, { repo: null });
    const { stores, dir } = await storeWithRegistry(t, world);
    const first = { repository: 'ns/m', manifestDigest: H };
    const others = [{ repository: `ns/m/sha256-${hexOf(H)}`, manifestDigest: H2 }, { repository: `ns/m/${hexOf(H)}`, manifestDigest: H2 }];
    const ggufRoot = path.join(dir, 'models', 'gguf');
    const hfRoot = path.join(dir, 'models', 'hf');
    const apart = (one, two) => one !== two && !two.startsWith(`${one}${path.sep}`) && !one.startsWith(`${two}${path.sep}`);
    for (const other of others) {
        const [a, b] = [first, other].map((names) => ({ ...pinnedGguf(names), file: 'm-Q4_K_M.gguf' }));
        assert.ok(apart(ociGgufPaths({ root: ggufRoot, source: a, file: a.file }).dir, ociGgufPaths({ root: ggufRoot, source: b, file: b.file }).dir), other.repository);
        const [sa, sb] = [first, other].map((names) => ociSnapshotPaths({ root: hfRoot, source: pinnedSnapshot(names) }));
        assert.ok(apart(sa.dir, sb.dir) && apart(sa.stateDir, sb.stateDir), other.repository);
    }
    // Both models on disk, then the first one's weights deleted: every file of the others is still there, byte for byte.
    const all = [first, ...others];
    for (const names of all) {
        await stores.oci.fetch({ artifact: pinnedGguf(names) });
        await stores['oci-snapshot'].fetch({ artifact: pinnedSnapshot(names) });
    }
    const filesOf = (names) => [stores.oci.paths(pinnedGguf(names)).file, ...SNAPSHOT_FILES.map(([name]) => path.join(ociSnapshotPaths({ root: hfRoot, source: pinnedSnapshot(names) }).dir, name))];
    const expected = [GGUF, ...SNAPSHOT_FILES.map(([, bytes]) => bytes)];
    assert.ok(await stores.oci.remove(pinnedGguf(first)) > 0);
    assert.ok(await stores['oci-snapshot'].remove(pinnedSnapshot(first)) > 0);
    assert.ok(filesOf(first).every((file) => !fs.existsSync(file)));
    for (const names of others) {
        filesOf(names).forEach((file, index) => assert.ok(fs.readFileSync(file).equals(expected[index]), file));
        assert.equal((await stores.oci.state(pinnedGguf(names))).state, 'complete');
        assert.equal((await stores['oci-snapshot'].state(pinnedSnapshot(names))).state, 'complete');
    }
});

test('the catalog takes oci sources for gguf and oci-snapshot sources for hf, pinned or not, and refuses what does not fit', () => {
    const model = (sources) => validateModel({ id: 'oci-model', sources });
    assert.equal(model({ gguf: ociSource() }).sources.gguf.type, 'oci');
    assert.equal(model({ gguf: pinnedSplit() }).sources.gguf.shards.length, 2);
    assert.equal(model({ hf: pinnedSnapshot() }).sources.hf.files.length, 2);
    assert.equal(validateModel({ id: 'oci-seed', sources: { gguf: pinnedGguf({ layout: 'docker', file: 'model.gguf' }) } }, { seed: true }).sources.gguf.layout, 'docker');
    const refused = [
        [{ gguf: ociSource({ registry: undefined }) }, 'sources.gguf.registry'],
        [{ gguf: ociSource({ registry: 'huggingface' }) }, 'sources.gguf.registry'],
        [{ gguf: ociSource({ repository: 'smollm2' }) }, 'sources.gguf.repository'],
        [{ gguf: ociSource({ repository: 'AI/smollm2' }) }, 'sources.gguf.repository'],
        [{ gguf: ociSource({ repository: 'a/b/c/d/e' }) }, 'sources.gguf.repository'],
        [{ gguf: ociSource({ tag: 'a/b' }) }, 'sources.gguf.tag'],
        [{ gguf: ociSource({ manifestDigest: 'sha256:abc' }) }, 'sources.gguf.manifestDigest'],
        [{ gguf: ociSource({ file: '../m.gguf' }) }, 'sources.gguf.file'],
        [{ gguf: ociSource({ file: 'm-00002-of-00002.gguf' }) }, 'sources.gguf.file'],
        [{ gguf: pinnedGguf({ manifestDigest: undefined }) }, 'sources.gguf.manifestDigest'],
        [{ gguf: pinnedGguf({ layout: 'docker' }) }, 'sources.gguf.file'],
        [{ gguf: pinnedGguf({ layout: 'zip' }) }, 'sources.gguf.layout'],
        [{ gguf: pinnedGguf({ sha256: 'x' }) }, 'sources.gguf.sha256'],
        [{ gguf: pinnedGguf({ extra: 1 }) }, 'sources.gguf.extra'],
        [{ gguf: ociSnapshotSource() }, 'sources.gguf.type'],
        [{ hf: ociSource() }, 'sources.hf.type'],
        [{ exl3: ociSnapshotSource() }, 'sources.exl3.type'],
        [{ hf: pinnedSnapshot({ files: [{ path: 'config.json', size: 1, sha256: 'a'.repeat(64) }], size: undefined }) }, 'sources.hf.files'],
        [{ hf: pinnedSnapshot({ files: [{ path: 'sub/model.safetensors', size: 1, sha256: 'a'.repeat(64) }], size: undefined }) }, 'sources.hf.files[0].path'],
        [{ hf: pinnedSnapshot({ manifestDigest: undefined }) }, 'sources.hf.manifestDigest'],
        [{ hf: ociSnapshotSource({ size: 3 }) }, 'sources.hf.size'],
    ];
    for (const [sources, field] of refused) {
        assert.throws(() => model(sources), { code: 'invalid_model', details: { field } }, field);
    }
    assert.throws(() => model({ gguf: { type: 'ollama', tag: 'x:1' } }), /sources\.gguf\.type must be huggingface or oci/);
    assert.throws(() => validateModel({ id: 'unpinned-seed', sources: { gguf: ociSource() } }, { seed: true }), { code: 'invalid_model' });
});

// ------------------------------------------------------------------ downloads

test('A13: a blob of the right size with the wrong bytes is SHA256_MISMATCH; the partial is removed and no final file exists', async (t) => {
    const world = await createOciWorld(t);
    const { stores } = await storeWithRegistry(t, world);
    const wrong = Buffer.from(GGUF);
    wrong[wrong.length - 1] ^= 0xff;
    world.cdnBytes = new Map([...world.blobs, [digestOf(GGUF), wrong]]);
    await assert.rejects(stores.oci.fetch({ artifact: pinnedGguf() }), { code: 'SHA256_MISMATCH' });
    const paths = stores.oci.paths(pinnedGguf());
    assert.ok(!fs.existsSync(paths.partial) && !fs.existsSync(paths.file) && !fs.existsSync(paths.meta));
    // Twin: the right bytes verify and are published.
    world.cdnBytes = null;
    await stores.oci.fetch({ artifact: pinnedGguf() });
    assert.ok(fs.readFileSync(paths.file).equals(GGUF));
});

test('Run downloads an OCI GGUF into _oci/<id>: the bearer at the registry only, provenance the blob URL only, the header read again', async (t) => {
    const world = await createOciWorld(t);
    const made = cncfGguf();
    const digest = world.publish(TAG, made);
    const h = ociController(t, world);
    await addReg(h);
    // The entry says its sizing came from the header, with values the header does not give: the Run reads it again.
    await h.controller.addModel(entryOf('oci-run', { gguf: ociSource() }, { sizingSource: 'gguf-header' }));
    world.reg.requests.length = 0;
    world.cdn.requests.length = 0;
    await h.controller.run({ modelId: 'oci-run', runnerId: 'llama.cpp', requestId: 'request-oci-run' });
    await until(() => ['ready', 'error'].includes(h.controller.state.deployment?.phase), 5000);
    assert.equal(h.controller.state.deployment.phase, 'ready', h.controller.state.deployment.error);
    const file = path.join(h.disk.dir, 'models', 'gguf', '_oci', 'reg', 'ai', `smollm2@sha256-${hexOf(digest)}`, 'SmolLM2-135M-Instruct-Q4_K_M.gguf');
    assert.ok(fs.readFileSync(file).equals(GGUF));
    const blobUrl = `/v2/${REPO}/blobs/${digestOf(GGUF)}`;
    const atRegistry = world.reg.requests.filter((request) => request.url === blobUrl);
    assert.ok(atRegistry.length >= 1 && atRegistry.at(-1).headers.authorization?.startsWith('Bearer issued-token-'), JSON.stringify(authOf(world.reg)));
    assert.ok(authOf(world.reg).every((value) => value !== BASIC));
    assert.ok(world.cdn.requests.length >= 1 && authOf(world.cdn).every((value) => value === undefined));
    const [entry] = h.controller.state.deployment.provenance;
    assert.equal(entry.source, `https://reg.test${blobUrl}`);
    const text = JSON.stringify(h.controller.state.deployment);
    assert.ok(!text.includes(SECRET) && !text.includes(CDN_SIGNATURE) && !text.includes('issued-token') && !text.includes('cdn.test'));
    const log = fs.readFileSync(path.join(h.disk.dir, 'logs', 'runner.log'), 'utf8');
    assert.match(log, /oci-run: the GGUF header of the verified file differs from the sizing stored when the model was added/);
    await h.controller.stop();
    // Delete weights twice: the file and its records go, then there is nothing to free.
    assert.ok((await h.controller.deleteWeights({ modelId: 'oci-run', runnerId: 'llama.cpp' })).freedBytes >= GGUF.length);
    assert.equal((await h.controller.deleteWeights({ modelId: 'oci-run', runnerId: 'llama.cpp' })).freedBytes, 0);
    assert.ok(!fs.existsSync(file));
});

test('a split GGUF and a snapshot download blob by blob into _oci/<id>; an update keeps the pin while registry, repository, tag and file stay', async (t) => {
    const world = await createOciWorld(t);
    const { stores, dir } = await storeWithRegistry(t, world);
    const split = await stores.oci.fetch({ artifact: pinnedSplit() });
    assert.deepEqual(split.files.map((file) => path.basename(file)), ['m-00001-of-00002.gguf', 'm-00002-of-00002.gguf']);
    split.files.forEach((file, index) => assert.ok(fs.readFileSync(file).equals(SHARDS[index])));
    const snapshot = await stores['oci-snapshot'].fetch({ artifact: pinnedSnapshot() });
    assert.equal(snapshot.path, path.join(dir, 'models', 'hf', '_oci', 'reg', 'ai', `smollm2@sha256-${hexOf(H)}`));
    assert.deepEqual(fs.readdirSync(snapshot.path).sort(), ['config.json', 'model.safetensors']);
    assert.equal((await stores['oci-snapshot'].recheck({ artifact: pinnedSnapshot() })).notes.length, 0);
    assert.equal((await stores.oci.state(pinnedSplit())).state, 'complete');
    assert.ok(world.cdn.requests.length === 4 && authOf(world.cdn).every((value) => value === undefined));
    assert.equal(world.auth_.requests.filter((request) => request.headers.authorization === BASIC).length, world.auth_.requests.length);
    // carryPin: an unchanged source keeps its pin; another tag or file, or another digest, is pinned again.
    const unpinned = ociSource({ file: 'm-00001-of-00002.gguf' });
    assert.equal(stores.oci.carryPin(unpinned, pinnedSplit()).manifestDigest, H);
    assert.equal(stores.oci.carryPin({ ...unpinned, tag: 'other' }, pinnedSplit()).manifestDigest, undefined);
    assert.equal(stores.oci.carryPin({ ...unpinned, file: 'x.gguf' }, pinnedSplit()).manifestDigest, undefined);
    assert.equal(stores.oci.carryPin({ ...unpinned, manifestDigest: H2 }, pinnedSplit()).shards, undefined);
    assert.equal(stores['oci-snapshot'].carryPin(ociSnapshotSource(), pinnedSnapshot()).files.length, 2);
    assert.equal(stores['oci-snapshot'].carryPin(ociSnapshotSource({ tag: 'other' }), pinnedSnapshot()).files, undefined);
});

// ------------------------------------------------------------------ jobs, drains and the N5 re-check (A26)

test('A26: a credential cleared and a registry disabled during a running download do not change its requests, though its token expires', async (t) => {
    const world = await createOciWorld(t, { expiresIn: 10 });
    const { stores, ops, registries } = await storeWithRegistry(t, world);
    const hold = deferred();
    const arrived = deferred();
    let first = true;
    world.cdnHook = (req, res, bytes) => {
        if (!first) return false;
        first = false;
        arrived.resolve();
        hold.promise.then(() => res.destroy());
        return true;
    };
    const job = stores.oci.fetch({ artifact: pinnedSplit() });
    await arrived.promise;
    const before = world.auth_.requests.length;
    await ops.clearRegistryCredential({ id: 'reg' });
    await ops.saveRegistry({ registry: { id: 'reg', enabled: false } });
    hold.resolve();
    assert.equal((await job).files.length, 2);
    // Each token lived 0 s, so every later request was challenged again and the job answered with its own credential.
    assert.ok(world.auth_.requests.length > before, 'the token was asked for again during the job');
    assert.deepEqual(new Set(authOf(world.auth_)), new Set([BASIC]));
    // The next operation sees the change.
    const { lookupInRegistry } = await import('../src/controller/registryLookup.mjs');
    world.publish(TAG, cncfGguf());
    await assert.rejects(lookupInRegistry({ registry: 'reg', repo: REPO, revision: TAG }, { fetchImpl: world.transport, registries }), { code: 'registry_disabled' });
    await ops.saveRegistry({ registry: { id: 'reg', enabled: true } });
    world.auth_.requests.length = 0;
    await lookupInRegistry({ registry: 'reg', repo: REPO, revision: TAG }, { fetchImpl: world.transport, registries });
    assert.ok(world.auth_.requests.length >= 1 && authOf(world.auth_).every((value) => value === undefined), 'the cleared credential is not sent');
});

test('A26: a drain ends an OCI pin, a lookup and a token fetch with shutting_down; the third concurrent lookup is busy', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, cncfGguf());
    const h = ociController(t, world);
    await addReg(h);
    const outcome = (promise) => promise.then(() => 'resolved', (error) => error.code);
    world.gate = deferred();
    t.after(() => world.gate.resolve());
    const first = outcome(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG }));
    const second = outcome(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG }));
    await until(() => world.reg.requests.length === 2);
    await assert.rejects(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG }), { code: 'busy' });
    const adding = outcome(h.controller.addModel(entryOf('draining', { gguf: ociSource() })));
    await until(() => world.reg.requests.length === 3);
    const drain = h.controller.drain();
    assert.deepEqual(await Promise.all([first, second, adding]), ['shutting_down', 'shutting_down', 'shutting_down']);
    await drain;

    // The same while the token is being fetched.
    const other = await createOciWorld(t);
    other.publish(TAG, cncfGguf());
    const g = ociController(t, other);
    await addReg(g);
    other.tokenGate = deferred();
    t.after(() => other.tokenGate.resolve());
    const looking = outcome(g.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG }));
    const pinning = outcome(g.controller.addModel(entryOf('token-drain', { gguf: ociSource() })));
    await until(() => other.auth_.requests.length === 2);
    const drained = g.controller.drain();
    assert.deepEqual(await Promise.all([looking, pinning]), ['shutting_down', 'shutting_down']);
    await drained;
    assert.deepEqual(g.controller.state.registry, []);
});

test('A26 (N5): an OCI registry re-created elsewhere, removed or disabled during an Add\'s pin fails the Add; a kind that cannot serve the source is invalid_model', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, cncfGguf());
    const h = ociController(t, world);
    await addReg(h);
    let count = 0;
    const during = async (change, source = ociSource()) => {
        world.gate = deferred();
        const arrivals = world.reg.requests.length;
        const running = codeOf(h.controller.addModel(entryOf(`during-${count += 1}`, { gguf: source })));
        await until(() => world.reg.requests.length > arrivals);
        await change();
        world.gate.resolve();
        return running;
    };
    const recreate = (changed) => async () => {
        await h.controller.removeRegistry({ id: 'reg' });
        await h.controller.saveRegistry({ registry: { ...REG, ...changed } });
    };
    assert.equal(await during(recreate({ url: 'https://elsewhere.test' })), 'registry_changed');
    await recreate({})();
    assert.equal(await during(recreate({ kind: 'huggingface', authOrigins: undefined })), 'registry_changed');
    await recreate({})();
    assert.equal(h.controller.state.registry.length, 0);
    assert.equal(await during(recreate({})), 'resolved', 'twin: created again as it was');
    const before = JSON.stringify(h.controller.state.registry);
    await h.controller.saveRegistry({ registry: { ...REG, id: 'reg-two' } });
    assert.equal(await during(() => h.controller.removeRegistry({ id: 'reg-two' }), ociSource({ registry: 'reg-two' })), 'registry_unknown');
    await h.controller.saveRegistry({ registry: { ...REG, id: 'reg-two' } });
    assert.equal(await during(() => h.controller.saveRegistry({ registry: { id: 'reg-two', enabled: false } }), ociSource({ registry: 'reg-two' })), 'registry_disabled');
    assert.equal(JSON.stringify(h.controller.state.registry), before, 'state.registry unchanged');
    // Kinds: an oci source on a ModelScope registry, an oci-snapshot on a mirror, a Hugging Face snapshot on an OCI registry.
    await h.controller.saveRegistry({ registry: { id: 'mirror', kind: 'huggingface', url: 'https://hf.test' } });
    for (const sources of [{ gguf: ociSource({ registry: 'modelscope' }) }, { hf: ociSnapshotSource({ registry: 'mirror' }) },
        { hf: { type: 'hf-snapshot', registry: 'reg', repo: 'acme/model', revision: 'main' } }]) {
        const arrivals = world.reg.requests.length + world.hf.requests.length;
        assert.equal(await codeOf(h.controller.addModel(entryOf('kind-mismatch', sources))), 'invalid_model', JSON.stringify(sources));
        assert.equal(world.reg.requests.length + world.hf.requests.length, arrivals);
    }
    assert.equal(JSON.stringify(h.controller.state.registry), before);
});
