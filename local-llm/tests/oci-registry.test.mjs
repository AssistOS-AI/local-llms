// OCI registries (DS002), the session, lookup and pins: the token realm and the credential boundary (A7), manifest
// digests (A13), refusals before any blob (A29), the CNCF safetensors mapping (A30), the dialects and the lookup input.
// The fake registry, realm and CDN are loopback servers behind a transport that maps their https names onto them
// (oci-fixture.mjs); oci-store.test.mjs holds paths, downloads and concurrency, net-guard.test.mjs the A5 rows.
import assert from 'node:assert/strict';
import test from 'node:test';

import { downloadPinned } from '../src/controller/downloader.mjs';
import { createOciSession, lookupOci, parseChallenge, readOciManifest } from '../src/controller/ociRegistry.mjs';
import { sanitizeOciFilepath } from '../src/controller/ociSource.mjs';
import { createRegistryService } from '../src/controller/registries.mjs';
import { lookupInRegistry } from '../src/controller/registryLookup.mjs';
import { countingListener } from './net-fixture.mjs';
import {
    AUTH_URL, BASIC, CDN_SIGNATURE, CONFIG_JSON, GGUF, MEDIA, REG, REG_URL, REPO, SECRET, TAG, USERNAME,
    addReg, authOf, blobRequests, cncfGguf, createOciWorld, digestOf, entryOf, image, json, legacyGguf, legacySplit,
    ociController, ociSnapshotSource, ociSource, scratch, sha256Of, shardBytes, snapshotOf,
} from './oci-fixture.mjs';
import { KEY, codeOf } from './registry-fixture.mjs';

const MIB = 1024 * 1024;
const GGUF_NAME = 'SmolLM2-135M-Instruct-Q4_K_M.gguf';
const headOf = (session) => session.request(session.manifestUrl(TAG), { method: 'HEAD' });
const sessionOn = (world, snapshot = snapshotOf(), now = Date.now) => createOciSession({ registry: snapshot, repository: REPO, fetchImpl: world.transport, now });
const queryOf = (request) => new URL(request.url, 'http://x').searchParams;

// ------------------------------------------------------------------ the credential boundary (A7)

test('A7: the bearer comes from a realm in authOrigins, which alone gets Basic; the registry sees the bearer and the CDN nothing', async (t) => {
    const world = await createOciWorld(t);
    const digest = world.publish(TAG, cncfGguf());
    const h = ociController(t, world);
    await addReg(h);
    const found = await h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG, file: GGUF_NAME });
    assert.deepEqual(found.registry, { id: 'reg', kind: 'oci', label: 'reg' });
    assert.deepEqual([found.provider, found.revision, found.commit, found.manifestDigest, found.layout, found.license, found.gated, found.truncated],
        ['oci', TAG, null, digest, 'cncf', 'Apache-2.0', null, false]);
    assert.deepEqual(found.files, [{ file: GGUF_NAME, size: GGUF.length, sha256: sha256Of(GGUF), gitOid: null, quantization: 'Q4_K_M', shards: null }]);
    assert.deepEqual([found.sizing.source, found.sizing.memory.layers], ['gguf-header', 24]);
    // One token request, at the realm, with the scope computed here and the challenge's service.
    assert.equal(world.auth_.requests.length, 1);
    const query = queryOf(world.auth_.requests[0]);
    assert.deepEqual([query.get('service'), query.get('scope')], ['reg.test', `repository:${REPO}:pull`]);
    assert.equal(world.auth_.requests[0].headers.authorization, BASIC, 'Basic, at the realm');
    // The registry: anonymous first, then the bearer only; never Basic, never the secret.
    const atRegistry = authOf(world.reg);
    assert.equal(atRegistry[0], undefined);
    assert.ok(atRegistry.slice(1).every((value) => value === 'Bearer issued-token-1'), JSON.stringify(atRegistry));
    assert.ok(world.cdn.requests.length >= 2 && authOf(world.cdn).every((value) => value === undefined), 'the CDN gets no authorization');
    assert.equal(world.other.requests.length + world.hf.requests.length, 0);
    const text = JSON.stringify(found);
    assert.ok(!text.includes(SECRET) && !text.includes(CDN_SIGNATURE) && !text.includes('issued-token'));
});

test('A7: a realm off authOrigins, over http, or malformed is never asked; Basic goes nowhere but an authOrigins realm', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, cncfGguf());
    const challenged = async (challenge, snapshot = snapshotOf()) => {
        world.challenge = challenge;
        return headOf(sessionOn(world, snapshot));
    };
    await assert.rejects(challenged('Bearer realm="https://other.test/token",service="reg.test"'), { code: 'ACCESS_DENIED', message: /token realm not allowed/ });
    await assert.rejects(challenged(`Bearer realm="http://auth.test/token"`), { code: 'ACCESS_DENIED' });
    await assert.rejects(challenged(`Bearer realm="https://user:pw@auth.test/token"`), { code: 'ACCESS_DENIED' });
    for (const malformed of ['Bearer service="reg.test"', 'Bearer realm=', `Bearer realm="${AUTH_URL}/token" junk`, 'Negotiate abc', `Bearer realm="${AUTH_URL}/token",realm="x"`]) {
        assert.equal((await challenged(malformed)).status, 401, malformed);
    }
    assert.equal(world.other.requests.length + world.auth_.requests.length, 0, 'no realm was asked');
    // Twin: other.test in authOrigins is asked, with Basic.
    await codeOf(challenged('Bearer realm="https://other.test/token"', snapshotOf({ authOrigins: [AUTH_URL, 'https://other.test'] })));
    assert.deepEqual(authOf(world.other), [BASIC]);

    // A realm at the registry origin when only auth.test may receive the credential (Docker Hub's shape): asked without it.
    world.registryHook = (req, res, url) => {
        if (url.pathname !== '/token') return false;
        world.tokens.add('home-token');
        const body = json({ token: 'home-token' });
        res.writeHead(200, { 'Content-Length': body.length });
        res.end(body);
        return true;
    };
    assert.equal((await challenged(`Bearer realm="${REG_URL}/token"`)).status, 200);
    const realmAtHome = () => world.reg.requests.filter((request) => request.url.startsWith('/token'));
    assert.deepEqual(realmAtHome().map((request) => request.headers.authorization), [undefined]);
    // Twin: with the registry origin in authOrigins, it gets Basic.
    assert.equal((await challenged(`Bearer realm="${REG_URL}/token"`, snapshotOf({ authOrigins: [REG_URL] }))).status, 200);
    assert.deepEqual(realmAtHome().map((request) => request.headers.authorization), [undefined, BASIC]);
    assert.ok(world.reg.requests.filter((request) => request.url.startsWith('/v2/')).every((request) => request.headers.authorization !== BASIC));

    // A Basic challenge: the credential goes to the registry only when its own origin is in authOrigins.
    world.auth = 'basic';
    world.challenge = 'Basic realm="reg"';
    world.reg.requests.length = 0;
    assert.equal((await headOf(sessionOn(world))).status, 401);
    assert.deepEqual(authOf(world.reg), [undefined], 'Basic is not sent to a registry outside authOrigins');
    assert.equal((await headOf(sessionOn(world, snapshotOf({ authOrigins: [REG_URL, AUTH_URL] })))).status, 200);
    assert.deepEqual(authOf(world.reg), [undefined, undefined, BASIC]);
    // Without a credential nothing is sent and the 401 stands.
    assert.equal((await headOf(sessionOn(world, snapshotOf({ authOrigins: [REG_URL], credential: null })))).status, 401);
});

test('the realm URL: its own query is kept, service and scope are set here; a realm at localhost meets the guard', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, cncfGguf());
    world.challenge = `Bearer realm="${AUTH_URL}/token?a=1&scope=repository:other/repo:push,pull",service="reg.test"`;
    assert.equal((await headOf(sessionOn(world))).status, 200);
    const query = queryOf(world.auth_.requests[0]);
    assert.deepEqual([query.get('a'), query.get('service'), query.getAll('scope')], ['1', 'reg.test', [`repository:${REPO}:pull`]]);
    // localhost in authOrigins: the production guard refuses it, and nothing connects; off authOrigins it is not even tried.
    const listener = await countingListener();
    t.after(() => listener.close());
    const realm = `https://localhost:${listener.port}`;
    world.challenge = `Bearer realm="${realm}/token"`;
    await assert.rejects(headOf(sessionOn(world, snapshotOf({ authOrigins: [realm] }))), { code: 'ADDRESS_REFUSED' });
    await assert.rejects(headOf(sessionOn(world)), { code: 'ACCESS_DENIED' });
    assert.equal(listener.stats.connections, 0);
    assert.equal(world.calls.filter((call) => call.url.startsWith(realm)).length, 1, 'only the authOrigins realm reached the transport');
});

test('the token cache: reused while fresh, asked again once expired, and a 401 with a cached token refetches once', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, cncfGguf());
    let clock = 1_000_000;
    const session = sessionOn(world, snapshotOf(), () => clock);
    assert.equal((await headOf(session)).status, 200);
    assert.equal((await headOf(session)).status, 200);
    assert.equal(world.issued, 1, 'one token for two requests');
    clock += 289_000;
    assert.equal((await headOf(session)).status, 200);
    assert.equal(world.issued, 1);
    clock += 2_000; // past expires_in (300 s) less 10 s
    assert.equal((await headOf(session)).status, 200);
    assert.equal(world.issued, 2);
    world.tokens.clear(); // revoked: the cached token gets a 401
    assert.equal((await headOf(session)).status, 200);
    assert.equal(world.issued, 3, 'refetched once');
    // A registry that never accepts: one refetch, then its 401 stands.
    world.tokenHook = (req, res) => {
        world.issued += 1;
        const body = json({ token: `rejected-${world.issued}` });
        res.writeHead(200, { 'Content-Length': body.length });
        res.end(body);
        return true;
    };
    world.tokens.clear();
    const before = world.issued;
    assert.equal((await headOf(session)).status, 401);
    assert.equal(world.issued, before + 1);
    // expires_in under 10 s: the fresh token is still used for its own retry.
    world.tokenHook = null;
    world.expiresIn = 5;
    assert.equal((await headOf(sessionOn(world))).status, 200);
});

test('the token answer: 64 KiB at most, a printable token of at most 16 KiB, token or access_token', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, cncfGguf());
    const answer = (body) => {
        world.tokenHook = (req, res) => {
            const bytes = typeof body === 'string' ? Buffer.from(body) : json(body);
            res.writeHead(200, { 'Content-Length': bytes.length });
            res.end(bytes);
            return true;
        };
        return headOf(sessionOn(world));
    };
    const accepted = (token) => { world.tokens.add(token); return token; };
    assert.equal((await answer({ access_token: accepted('access-only') })).status, 200);
    // A 16 KiB token is used (the fake registry's own header limit may then drop the connection); 16 KiB + 1 is not.
    const largest = 't'.repeat(16 * 1024);
    await answer({ token: largest }).catch(() => {});
    assert.ok(world.calls.some((call) => call.authorization === `Bearer ${largest}`));
    await assert.rejects(answer({ token: 't'.repeat(16 * 1024 + 1) }), { code: 'RESOLVE_FAILED', details: { reason: 'token' } });
    assert.ok(!world.calls.some((call) => call.authorization === `Bearer ${largest}t`));
    for (const bad of [{ token: 'a b' }, { token: 'a\u0000b' }, { token: 7 }, {}, 'not json', []]) {
        await assert.rejects(answer(bad), { code: 'RESOLVE_FAILED', details: { reason: 'token' } }, JSON.stringify(bad));
    }
    const padded = (size) => { const body = JSON.stringify({ token: accepted('padded'), pad: '' }); return body.replace('""', `"${'p'.repeat(size - body.length)}"`); };
    assert.equal((await answer(padded(64 * 1024))).status, 200);
    await assert.rejects(answer(padded(64 * 1024 + 1)), { code: 'RESOLVE_FAILED', details: { reason: 'too-large' } });
});

test('parseChallenge reads one Bearer or Basic challenge and refuses anything else', () => {
    assert.deepEqual({ ...parseChallenge('Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:ai/smollm2:pull"') },
        { scheme: 'bearer', realm: 'https://auth.docker.io/token', service: 'registry.docker.io' });
    assert.deepEqual({ ...parseChallenge('Basic realm="r"') }, { scheme: 'basic', realm: 'r', service: null });
    assert.deepEqual({ ...parseChallenge('bearer realm=https://a.example/t') }, { scheme: 'bearer', realm: 'https://a.example/t', service: null });
    for (const bad of [null, '', 'Bearer', 'Bearer x', 'Bearer realm="a', 'Bearer realm="a" ,, ', 'Digest realm="a"', `Bearer realm="${'a'.repeat(5000)}"`]) {
        assert.equal(parseChallenge(bad), null, String(bad).slice(0, 40));
    }
});

// ------------------------------------------------------------------ manifest digests (A13)

test('A13: a manifest that does not hash to its digest refuses the pin and the lookup, and reads no blob', async (t) => {
    const world = await createOciWorld(t);
    const real = cncfGguf();
    const tampered = json({ ...real.manifest, annotations: { changed: 'yes' } });
    world.publish(TAG, real);
    world.digestHeader = digestOf(real.bytes);
    world.manifests.set(digestOf(real.bytes), tampered);
    const h = ociController(t, world);
    await addReg(h);
    await assert.rejects(h.controller.addModel(entryOf('tampered', { gguf: ociSource() })), { code: 'SHA256_MISMATCH' });
    await assert.rejects(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG }), { code: 'lookup_failed', details: { reason: 'digest-mismatch' } });
    // A digest the admin typed is held to the same rule.
    await assert.rejects(h.controller.addModel(entryOf('typed', { gguf: ociSource({ manifestDigest: digestOf(real.bytes) }) })), { code: 'SHA256_MISMATCH' });
    assert.deepEqual(h.controller.state.registry, []);
    assert.equal(blobRequests(world).length, 0);
    // Twin: the real body pins; without a digest header the GET by tag gives the body's own digest.
    world.manifests.set(digestOf(real.bytes), real.bytes);
    const pinned = (await h.controller.addModel(entryOf('real', { gguf: ociSource() }))).model.sources.gguf;
    assert.equal(pinned.manifestDigest, digestOf(real.bytes));
    world.digestHeader = false;
    world.reg.requests.length = 0;
    const byTag = await h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG });
    assert.equal(byTag.manifestDigest, digestOf(real.bytes));
    assert.ok(world.reg.requests.some((request) => request.method === 'GET' && request.url === `/v2/${REPO}/manifests/${TAG}`));
});

// ------------------------------------------------------------------ refusals (A29)

const at = (size, base) => {
    const made = base();
    const length = json({ ...made.manifest, annotations: { pad: '' } }).length;
    return { ...made, bytes: json({ ...made.manifest, annotations: { pad: 'x'.repeat(size - length) } }) };
};
const ggufLayers = (count) => image({ files: Array.from({ length: count }, (_, index) => [MEDIA.weight, Buffer.from(`gguf file ${index}`), `f-${index}.gguf`]) });

test('A29: refused artifacts are unsupported_artifact (or too-large) for the lookup and the pin, with no blob read', async (t) => {
    const world = await createOciWorld(t);
    const h = ociController(t, world);
    await addReg(h);
    const legacyConfig = json({ config: { format: 'safetensors' } });
    const cases = [
        ['an index', { manifest: { schemaVersion: 2, mediaType: MEDIA.index, manifests: [] } }],
        ['KitOps', image({ files: [[MEDIA.weight, GGUF, 'm.gguf']], extra: { artifactType: 'application/vnd.kitops.modelkit.manifest.v1+json' } })],
        ['an Ollama layer', image({ layout: 'docker', files: [['application/vnd.ollama.image.model', GGUF]] })],
        ['an Ollama manifest', { manifest: { schemaVersion: 2, mediaType: 'application/vnd.docker.distribution.manifest.v2+json',
            config: { mediaType: 'application/vnd.docker.container.image.v1+json', digest: digestOf(GGUF), size: 1 }, layers: [] } }],
        ['legacy safetensors', image({ layout: 'docker', config: legacyConfig, files: [['application/vnd.docker.ai.safetensors', GGUF], ['application/vnd.docker.ai.vllm.config.tar', CONFIG_JSON]] })],
        ['vllm.config.tar', image({ layout: 'docker', files: [[MEDIA.legacyGguf, GGUF], ['application/vnd.docker.ai.vllm.config.tar', CONFIG_JSON]] })],
        ['dir.tar', image({ layout: 'docker', files: [[MEDIA.legacyGguf, GGUF], ['application/vnd.docker.ai.dir.tar', CONFIG_JSON]] })],
        ['weight.v1.tar', image({ files: [['application/vnd.cncf.model.weight.v1.tar', GGUF, 'm.gguf']] })],
        ['config v0.2', { manifest: { ...cncfGguf().manifest, artifactType: undefined, config: { ...cncfGguf().manifest.config, mediaType: 'application/vnd.docker.ai.model.config.v0.2+json' } } }],
        ['a traversing path', image({ files: [[MEDIA.weight, GGUF, '../m.gguf']] })],
        ['an absolute path', image({ files: [[MEDIA.weight, GGUF, '/m.gguf']] })],
        ['a backslash', image({ files: [[MEDIA.weight, GGUF, 'a\\m.gguf']] })],
        ['a control character', image({ files: [[MEDIA.weight, GGUF, 'm\u0001.gguf']] })],
        ['a duplicated path', image({ files: [[MEDIA.weight, GGUF, 'm.gguf'], [MEDIA.weight, CONFIG_JSON, './m.gguf']] })],
        ['65 GGUF layers', ggufLayers(65)],
        ['no GGUF file (unicode name only)', image({ files: [[MEDIA.weight, GGUF, 'modèle.gguf']] })],
    ];
    for (const [name, made] of cases) {
        const tag = `case-${cases.findIndex(([other]) => other === name)}`;
        world.publish(tag, { bytes: made.bytes ?? json(made.manifest), blobs: made.blobs ?? new Map() });
        for (const act of [
            () => h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: tag, ...(name.includes('safetensors') ? { format: 'hf' } : {}) }),
            () => h.controller.addModel(entryOf(`m-${tag}`, name.includes('safetensors') ? { hf: ociSnapshotSource({ tag }) } : { gguf: ociSource({ tag }) })),
        ]) assert.equal(await codeOf(act()), 'unsupported_artifact', name);
    }
    // The manifest cap: 1 MiB is read, 1 MiB + 1 is not.
    world.publish('at-cap', at(MIB, cncfGguf));
    world.publish('over-cap', at(MIB + 1, cncfGguf));
    await assert.rejects(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: 'over-cap' }), { code: 'lookup_failed', details: { reason: 'too-large' } });
    await assert.rejects(h.controller.addModel(entryOf('over', { gguf: ociSource({ tag: 'over-cap' }) })), { code: 'RESOLVE_FAILED', details: { reason: 'too-large' } });
    assert.equal(blobRequests(world).length, 0, 'no refusal read a blob');
    assert.deepEqual(h.controller.state.registry, []);
    // Twins: 1 MiB and 64 GGUF layers are read.
    assert.equal((await h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: 'at-cap' })).files.length, 1);
    world.publish('sixty-four', ggufLayers(64));
    assert.equal((await h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: 'sixty-four' })).files.length, 64);
    // A CNCF config whose format is neither gguf nor safetensors is refused once it is read.
    world.publish('onnx', image({ config: json({ config: { format: 'onnx' } }), files: [[MEDIA.weight, GGUF, 'm.gguf']] }));
    assert.equal(await codeOf(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: 'onnx' })), 'unsupported_artifact');
});

test('sanitizeOciFilepath: dot and empty segments drop; absolute, backslash, control characters and .. are unsafe', () => {
    assert.deepEqual(['a/b.gguf', './a//b.gguf', 'a/./b.gguf', 'modèle.gguf'].map(sanitizeOciFilepath), ['a/b.gguf', 'a/b.gguf', 'a/b.gguf', 'modèle.gguf']);
    for (const unsafe of ['', '/', '.', '../a', 'a/../b', '/abs', 'a\\b', 'a\u0000b', 'a\u007fb', 'x'.repeat(513), 7, null]) {
        assert.equal(sanitizeOciFilepath(unsafe), null, String(unsafe).slice(0, 20));
    }
});

// ------------------------------------------------------------------ CNCF safetensors (A30)

// The layers of Docker Hub's ai/smollm2:135m-safetensors (q3-smollm2-135m-safetensors.manifest.json), in order.
const EVIDENCE_LAYERS = Object.freeze([
    [MEDIA.doc, 'README.md'], [MEDIA.weightConfig, 'all_results.json'], [MEDIA.weightConfig, 'config.json'],
    [MEDIA.weightConfig, 'eval_results.json'], [MEDIA.weightConfig, 'generation_config.json'], [MEDIA.weightConfig, 'merges.txt'],
    [MEDIA.weight, 'model.safetensors'], [MEDIA.weightConfig, 'special_tokens_map.json'], [MEDIA.weightConfig, 'tokenizer.json'],
    [MEDIA.weightConfig, 'tokenizer_config.json'], [MEDIA.weightConfig, 'train_results.json'], [MEDIA.weightConfig, 'trainer_state.json'],
    [MEDIA.weight, 'training_args.bin'], [MEDIA.weightConfig, 'vocab.json'],
]);
const evidenceImage = ({ without = null, extra = [] } = {}) => image({
    config: json({ config: { format: 'safetensors' }, descriptor: { licenses: ['Apache-2.0'] } }),
    files: [...EVIDENCE_LAYERS, ...extra].filter(([, name]) => name !== without)
        .map(([media, name]) => [media, name === 'config.json' ? CONFIG_JSON : Buffer.from(`bytes of ${name}\n`), name]),
});

test('A30: the evidence layer list pins 12 files; README.md, training_args.bin and a subdirectory drop; no safetensors is refused', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, evidenceImage({ extra: [[MEDIA.weight, 'original/model.safetensors']] }));
    world.publish('no-weights', evidenceImage({ without: 'model.safetensors' }));
    const h = ociController(t, world);
    await addReg(h);
    const { model } = await h.controller.addModel(entryOf('smol', { hf: ociSnapshotSource() }));
    const names = model.sources.hf.files.map((file) => file.path);
    assert.equal(names.length, 12);
    assert.deepEqual(names, EVIDENCE_LAYERS.map(([, name]) => name).filter((name) => !['README.md', 'training_args.bin'].includes(name)));
    assert.equal(model.sources.hf.size, model.sources.hf.files.reduce((sum, file) => sum + file.size, 0));
    assert.ok(model.sources.hf.files.every((file) => file.sha256 === sha256Of(file.path === 'config.json' ? CONFIG_JSON : Buffer.from(`bytes of ${file.path}\n`))));
    assert.equal(await codeOf(h.controller.addModel(entryOf('none', { hf: ociSnapshotSource({ tag: 'no-weights' }) }))), 'unsupported_artifact');
    // The lookup sizes from config.json, read by its digest; a CDN that serves other bytes of that size fails it.
    const found = await h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG, format: 'hf' });
    assert.deepEqual([found.files.length, found.sizing.source, found.sizing.memory.layers, found.sizing.contextLength], [12, 'config.json', 2, 2048]);
    world.cdnBytes = new Map([...world.blobs, [digestOf(CONFIG_JSON), Buffer.from(CONFIG_JSON.toString().replace('2048', '4096'))]]);
    await assert.rejects(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG, format: 'hf' }), { code: 'lookup_failed', details: { reason: 'digest-mismatch' } });
    // exl3 is not an OCI format; a GGUF lookup of a safetensors image finds nothing to add.
    await assert.rejects(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG, format: 'exl3' }), { code: 'invalid_request' });
    await assert.rejects(h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: TAG }), { code: 'unsupported_artifact' });
});

// ------------------------------------------------------------------ GGUF dialects

test('GGUF pins: the legacy name, a file derived when unique, mmproj never offered, split headers and order checked', async (t) => {
    const world = await createOciWorld(t);
    const h = ociController(t, world);
    await addReg(h);
    let made = 0;
    const pin = async (tag, extra = {}) => (await h.controller.addModel(entryOf(`m-${made += 1}`, { gguf: ociSource({ tag, ...extra }) }))).model.sources.gguf;
    world.publish('legacy', legacyGguf());
    assert.deepEqual((({ layout, file, size, sha256 }) => ({ layout, file, size, sha256 }))(await pin('legacy')),
        { layout: 'docker', file: 'model.gguf', size: GGUF.length, sha256: sha256Of(GGUF) });
    world.publish('two', image({ files: [[MEDIA.weight, GGUF, 'a-Q4_K_M.gguf'], [MEDIA.weight, CONFIG_JSON, 'b-Q8_0.gguf'], [MEDIA.weight, Buffer.from('proj'), 'mmproj-F16.gguf']] }));
    assert.deepEqual(await codeOf(pin('two')), 'invalid_model', 'two files: name one');
    assert.equal((await pin('two', { file: 'b-Q8_0.gguf' })).sha256, sha256Of(CONFIG_JSON));
    assert.equal(await codeOf(pin('two', { file: 'mmproj-F16.gguf' })), 'NOT_FOUND');
    assert.deepEqual((await h.controller.lookupModel({ registry: 'reg', repo: REPO, revision: 'two' })).files.map((row) => row.file), ['a-Q4_K_M.gguf', 'b-Q8_0.gguf']);
    // A legacy split: each shard's header must say k-1 of N in layer order.
    const shards = [shardBytes(0, 2), shardBytes(1, 2)];
    world.publish('split', legacySplit(shards));
    const split = await pin('split');
    assert.deepEqual(split.shards.map((shard) => [shard.file, shard.sha256]), [['model-00001-of-00002.gguf', sha256Of(shards[0])], ['model-00002-of-00002.gguf', sha256Of(shards[1])]]);
    world.publish('swapped', legacySplit([shards[1], shards[0]]));
    assert.equal(await codeOf(pin('swapped')), 'unsupported_artifact');
    world.publish('unsplit', legacySplit([GGUF, shardBytes(1, 2)]));
    assert.equal(await codeOf(pin('unsplit')), 'unsupported_artifact');
    // CNCF shards are named; out of layer order they are refused, not reordered.
    const named = (order) => image({ files: order.map((k) => [MEDIA.weight, shards[k - 1], `q-0000${k}-of-00002.gguf`]) });
    world.publish('cncf-split', named([1, 2]));
    assert.equal((await pin('cncf-split')).shards.length, 2);
    world.publish('cncf-swapped', named([2, 1]));
    assert.equal(await codeOf(pin('cncf-swapped')), 'unsupported_artifact');
});

// ------------------------------------------------------------------ input and the transport rule

test('the lookup input of an OCI registry: repository and tag rules, latest by default, refused before any request', async (t) => {
    const world = await createOciWorld(t);
    world.publish('latest', cncfGguf());
    const h = ociController(t, world);
    await addReg(h);
    assert.equal((await h.controller.lookupModel({ registry: 'reg', repo: REPO })).revision, 'latest');
    world.reg.requests.length = 0;
    for (const args of [{ repo: 'AI/smollm2' }, { repo: 'a/b/c/d/e' }, { repo: 'smollm2/' }, { repo: REPO, revision: 'a/b' }, { repo: REPO, revision: '-x' },
        { repo: REPO, format: 'exl3' }, { repo: REPO, file: '../m.gguf' }, { repo: REPO, file: 'm-00002-of-00002.gguf' }]) {
        assert.equal(await codeOf(h.controller.lookupModel({ registry: 'reg', ...args })), 'invalid_request', JSON.stringify(args));
        assert.equal(await codeOf(h.controller.lookupModel({ registry: 'dockerhub', ...args })), 'invalid_request', `dockerhub ${JSON.stringify(args)}`);
    }
    assert.equal(world.reg.requests.length, 0);
    // An OCI repository is not a Hugging Face one: a/b/c is fine here and refused for huggingface and modelscope.
    assert.equal(await codeOf(h.controller.lookupModel({ registry: 'reg', repo: 'a/b/c' })), 'not_found');
    assert.equal(await codeOf(h.controller.lookupModel({ registry: 'modelscope', repo: 'a/b/c' })), 'invalid_request');
});

test('a named registry is never looked up without the guarded transport (internal_error, no request); the transport twin works', async (t) => {
    const world = await createOciWorld(t);
    world.publish(TAG, cncfGguf());
    const disk = scratch(t);
    const registries = createRegistryService({ dataDir: disk.dir, credentialsKey: KEY });
    const ops = registries.operations({ queue: { run: (fn) => fn() } });
    await ops.saveRegistry({ registry: REG });
    await ops.saveRegistry({ registry: { id: 'mirror', kind: 'huggingface', url: 'https://hf.test' } });
    await ops.setRegistryCredential({ id: 'reg', token: SECRET, tokenUsername: USERNAME });
    let calls = 0;
    const plainFetch = async () => { calls += 1; throw new Error('a plain fetch is not a model path'); };
    for (const registry of ['reg', 'mirror', 'modelscope', 'dockerhub']) {
        await assert.rejects(lookupInRegistry({ registry, repo: registry === 'reg' || registry === 'dockerhub' ? REPO : 'acme/model' }, { fetchImpl: plainFetch, registries }),
            { code: 'internal_error' }, registry);
        await assert.rejects(lookupInRegistry({ registry, repo: 'acme/model' }, { registries }), { code: 'internal_error' }, `${registry} without fetchImpl`);
    }
    assert.equal(calls, 0);
    assert.equal(world.calls.length, 0);
    const found = await lookupInRegistry({ registry: 'reg', repo: REPO, revision: TAG }, { fetchImpl: world.transport, registries });
    assert.equal(found.registry.kind, 'oci');
    assert.ok(world.calls.length >= 3);
    // Every new adapter function needs its fetch.
    assert.throws(() => createOciSession({ registry: snapshotOf(), repository: REPO }), { name: 'TypeError', message: 'fetchImpl is required' });
    await assert.rejects(lookupOci({ repo: REPO }, { registry: snapshotOf() }), { name: 'TypeError', message: 'fetchImpl is required' });
    await assert.rejects(downloadPinned({ paths: {}, artifact: {}, identityKeys: [], url: `${REG_URL}/x` }), { name: 'TypeError', message: 'fetchImpl is required' });
    await assert.rejects(readOciManifest(sessionOn(world), { tag: 'missing' }), { code: 'NOT_FOUND' });
});
