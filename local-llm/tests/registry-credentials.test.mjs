// Registry credentials (DS002): write-only, encrypted at rest under the agent's generated key, bound to their
// registry's definition, and never in a result, an error, a log, the state files or provenance (A9, A10).
// Assertions on text that may hold a credential only test `includes`, so a failure never prints the value.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { serializeError } from '../src/errors.mjs';
import { CREDENTIALS_UNAVAILABLE, createCredentialStore, credentialAad, deriveCredentialKeys } from '../src/controller/registryCredentials.mjs';
import { createRegistryService } from '../src/controller/registries.mjs';
import { fakeServer, createTestTransport } from './net-fixture.mjs';
import {
    KEY, OTHER_KEY, RUN, codeOf, credentialsFile, filesUnder, readJson, registriesFile, registryHarness, tempDir, until,
} from './registry-fixture.mjs';

const TOKEN = 'sekret-registry-value-1';
const MIRROR = Object.freeze({ id: 'mirror', kind: 'huggingface', url: 'https://mirror.example/hf' });
const OCI = Object.freeze({ id: 'oci-reg', kind: 'oci', url: 'https://registry.example' });

const hasToken = (value) => (typeof value === 'string' ? value : JSON.stringify(value) ?? '').includes(TOKEN);

async function errorText(promise) {
    try {
        await promise;
    } catch (error) {
        // What the tool process writes for a failed call (tools/local_llm_tool.mjs), plus the error's own fields.
        return { code: error.code, text: `${JSON.stringify(serializeError(error))}\n${error.stack ?? ''}\n${String(error)}` };
    }
    return { code: 'resolved', text: '' };
}

const credentialOf = (list, id) => list.registries.find((registry) => registry.id === id)?.credential;

test('A9: a credential set through the tool never appears in a result, an error, the list, the overview, the log, provenance or any file, and its file is 0600', async (t) => {
    const h = registryHarness(t);
    await h.tool('local_llm_registry_save', { registry: MIRROR });
    const set = await h.tool('local_llm_registry_credential_set', { id: 'mirror', token: TOKEN });
    assert.deepEqual(set, { id: 'mirror', credential: 'set' });
    assert.equal(hasToken(set), false, 'the set result');
    // Error results: an unknown registry, a token that is too long, an unexpected field, huggingface.
    for (const [input, code] of [
        [{ id: 'nope', token: TOKEN }, 'invalid_registry'],
        [{ id: 'mirror', token: `${TOKEN}${'x'.repeat(4096)}` }, 'invalid_registry'],
        [{ id: 'mirror', token: TOKEN, tokenUsername: `${TOKEN}\n` }, 'invalid_registry'],
        [{ id: 'huggingface', token: TOKEN }, 'read_only'],
    ]) {
        const failed = await errorText(h.tool('local_llm_registry_credential_set', input));
        assert.equal(failed.code, code, `error code for ${input.id}`);
        assert.equal(failed.text.includes(TOKEN), false, `the ${code} error for ${input.id}`);
    }
    // The key absent: another controller over a copy of nothing, with no key.
    const keyless = registryHarness(t, { credentialsKey: null });
    await keyless.tool('local_llm_registry_save', { registry: MIRROR });
    const unavailable = await errorText(keyless.tool('local_llm_registry_credential_set', { id: 'mirror', token: TOKEN }));
    assert.equal(unavailable.code, 'credentials_unavailable');
    assert.equal(unavailable.text.includes(TOKEN), false, 'the credentials_unavailable error');

    const list = await h.tool('local_llm_registry_list', {});
    assert.equal(credentialOf(list, 'mirror'), 'set');
    assert.equal(hasToken(list), false, 'registry_list');
    // A Run writes provenance lines to the log; the overview and status are read after it.
    await h.controller.run({ ...RUN, requestId: 'request-0001' });
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const overview = await h.tool('local_llm_overview', {});
    assert.equal(hasToken(overview), false, 'the overview');
    const status = await h.tool('local_llm_status', { sinceSeq: 0 });
    // The leak checks below can fail: the log, the deployment's provenance and the overview are all non-empty.
    assert.ok(Array.isArray(status.logs) && status.logs.length > 0, 'status() has log lines');
    assert.ok(status.logs.some((line) => JSON.stringify(line).includes('downloaded from https://huggingface.co/')), 'provenance was logged');
    assert.ok(h.controller.state.deployment.provenance.length > 0, 'the deployment records provenance');
    assert.equal(hasToken(h.controller.state.deployment.provenance), false, 'provenance');
    assert.equal(hasToken(status.logs), false, 'status().logs');
    assert.equal(hasToken(status), false, 'status()');
    await h.controller.stop();
    // Every file the agent wrote: controller.json, registries.json, registry-credentials.json (raw bytes), runner.log.
    const files = filesUnder(h.dataDir);
    for (const name of ['controller.json', 'registries.json', 'registry-credentials.json', 'runner.log']) {
        assert.ok(files.some((file) => file.endsWith(`/${name}`)), `${name} was written`);
    }
    for (const file of files) {
        assert.equal(fs.readFileSync(file).includes(Buffer.from(TOKEN)), false, `raw bytes of ${file.slice(h.dataDir.length)}`);
    }
    assert.equal(fs.statSync(credentialsFile(h.dataDir)).mode & 0o777, 0o600);
    assert.equal(fs.statSync(registriesFile(h.dataDir)).mode & 0o777, 0o600);
    assert.equal(fs.statSync(`${h.dataDir}/state`).mode & 0o777, 0o700);
});

test('A10: with the key absent or malformed, credentials cannot be set, the list says why, and a lookup still works anonymously', async (t) => {
    const hf = await fakeServer((req, res) => { res.writeHead(404, { 'Content-Type': 'application/json' }); res.end('{}'); });
    t.after(() => hf.close());
    for (const credentialsKey of [null, '', KEY.slice(1), KEY.toUpperCase(), `${KEY}0`, 'z'.repeat(64)]) {
        const h = registryHarness(t, { credentialsKey, extra: { hfBaseUrl: hf.origin, registryTransport: createTestTransport() } });
        const what = `key of length ${credentialsKey?.length ?? 'null'}`;
        await h.tool('local_llm_registry_save', { registry: MIRROR });
        const failed = await errorText(h.tool('local_llm_registry_credential_set', { id: 'mirror', token: TOKEN }));
        assert.equal(failed.code, 'credentials_unavailable', what);
        const list = await h.tool('local_llm_registry_list', {});
        assert.equal(list.credentialStorage.available, false, what);
        assert.match(list.credentialStorage.reason, /LOCAL_LLM_CREDENTIALS_KEY/, what);
        assert.equal(credentialOf(list, 'mirror'), 'none', what);
        assert.equal(fs.existsSync(credentialsFile(h.dataDir)), false, `${what}: no credentials file`);
        // The Hugging Face lookup is unaffected and anonymous: it reaches the server, which answers not found.
        const before = hf.requests.length;
        assert.equal(await codeOf(h.controller.lookupModel({ repo: 'acme/model' })), 'not_found', what);
        assert.ok(hf.requests.length > before, `${what}: the lookup reached the server`);
        assert.ok(hf.requests.slice(before).every((request) => request.headers.authorization === undefined), `${what}: anonymous`);
    }
    // The absent key gives the documented message.
    const keyless = registryHarness(t, { credentialsKey: null });
    assert.equal((await keyless.tool('local_llm_registry_list', {})).credentialStorage.reason, CREDENTIALS_UNAVAILABLE);
});

test('A10: a stored credential read after a restart with another key, or with none, is unreadable, anonymous and logged once', async (t) => {
    const dataDir = tempDir(t);
    const first = registryHarness(t, { dataDir });
    await first.tool('local_llm_registry_save', { registry: MIRROR });
    await first.tool('local_llm_registry_credential_set', { id: 'mirror', token: TOKEN });
    assert.equal(credentialOf(await first.tool('local_llm_registry_list', {}), 'mirror'), 'set');

    for (const credentialsKey of [OTHER_KEY, null]) {
        const restarted = registryHarness(t, { dataDir, credentialsKey });
        const list = await restarted.tool('local_llm_registry_list', {});
        assert.equal(credentialOf(list, 'mirror'), 'unreadable', `key ${credentialsKey ? 'changed' : 'absent'}`);
        assert.equal(list.credentialStorage.available, credentialsKey !== null);
        await restarted.tool('local_llm_registry_list', {});
        const status = await restarted.controller.status({ sinceSeq: 0 });
        assert.ok(Array.isArray(status.logs) && status.logs.length > 0, 'status() has log lines');
        const lines = status.logs.filter((line) => JSON.stringify(line).includes('registry mirror') && JSON.stringify(line).includes('re-enter it'));
        assert.equal(lines.length, 1, 'one log line per registry per start');
        assert.equal(hasToken(status), false, 'the log');
        // A lookup, pin or job through it is anonymous: its snapshot carries no credential.
        const service = createRegistryService({ dataDir, credentialsKey });
        assert.equal(service.snapshot('mirror').credential, null);
    }
    // Setting it again under the new key overwrites the unreadable entry.
    const rekeyed = registryHarness(t, { dataDir, credentialsKey: OTHER_KEY });
    await rekeyed.tool('local_llm_registry_credential_set', { id: 'mirror', token: 'replacement-token-2' });
    assert.equal(credentialOf(await rekeyed.tool('local_llm_registry_list', {}), 'mirror'), 'set');
    assert.equal(readJson(credentialsFile(dataDir)).keyId, deriveCredentialKeys(OTHER_KEY).keyId);
    assert.equal(createRegistryService({ dataDir, credentialsKey: OTHER_KEY }).snapshot('mirror').credential.token === 'replacement-token-2', true);
    assert.equal(createRegistryService({ dataDir, credentialsKey: KEY }).snapshot('mirror').credential, null, 'the old key reads nothing now');
});

test('A10: a credential cannot be moved to another registry or survive a URL change, and a corrupt tag is unreadable', async (t) => {
    const dataDir = tempDir(t);
    const h = registryHarness(t, { dataDir });
    await h.tool('local_llm_registry_save', { registry: MIRROR });
    await h.tool('local_llm_registry_save', { registry: { ...MIRROR, id: 'mirror-two' } });
    await h.tool('local_llm_registry_credential_set', { id: 'mirror', token: 'moved-token' });
    const sealed = readJson(credentialsFile(dataDir));
    // Moved by hand to another id with the same kind and URL: the AAD names the id, so GCM fails.
    fs.writeFileSync(credentialsFile(dataDir), JSON.stringify({ ...sealed, entries: { ...sealed.entries, 'mirror-two': sealed.entries.mirror } }));
    let list = await registryHarness(t, { dataDir }).tool('local_llm_registry_list', {});
    assert.equal(credentialOf(list, 'mirror'), 'set');
    assert.equal(credentialOf(list, 'mirror-two'), 'unreadable');
    // The URL changed by a manual edit of registries.json (no explicit delete): the AAD names the URL.
    const registries = readJson(registriesFile(dataDir));
    registries.registries = registries.registries.map((entry) => (entry.id === 'mirror' ? { ...entry, url: 'https://elsewhere.example/hf' } : entry));
    fs.writeFileSync(registriesFile(dataDir), JSON.stringify(registries));
    list = await registryHarness(t, { dataDir }).tool('local_llm_registry_list', {});
    assert.equal(credentialOf(list, 'mirror'), 'unreadable');
    // A corrupt tag (probe): unreadable, never set, and the store never throws.
    const storeDir = tempDir(t);
    const store = createCredentialStore({ dataDir: storeDir, credentialsKey: KEY });
    const record = { id: 'reg', kind: 'huggingface', url: 'https://reg.example', authOrigins: [] };
    store.seal(record, { token: 'tag-token' });
    assert.equal(store.state(record), 'set');
    const raw = readJson(store.file);
    const tag = Buffer.from(raw.entries.reg.tag, 'base64');
    tag[0] ^= 1;
    raw.entries.reg.tag = tag.toString('base64');
    fs.writeFileSync(store.file, JSON.stringify(raw));
    const reopened = createCredentialStore({ dataDir: storeDir, credentialsKey: KEY });
    assert.equal(reopened.state(record), 'unreadable');
    assert.equal(reopened.open(record), null);
    // The AAD itself: id, kind, URL and the sorted auth origins, one per line.
    assert.equal(credentialAad({ id: 'r', kind: 'oci', url: 'https://r.example', authOrigins: ['https://b.example', 'https://a.example'] }).toString(),
        'local-llm/registry-credential/v1\nr\noci\nhttps://r.example\nhttps://a.example https://b.example');
});

test('A10: a save that changes url or authOrigins deletes the credential; one that changes only the label keeps it', async (t) => {
    const h = registryHarness(t);
    await h.tool('local_llm_registry_save', { registry: MIRROR });
    await h.tool('local_llm_registry_save', { registry: OCI });
    const entries = () => Object.keys(readJson(credentialsFile(h.dataDir)).entries).sort();
    await h.tool('local_llm_registry_credential_set', { id: 'mirror', token: 'mirror-token' });
    await h.tool('local_llm_registry_credential_set', { id: 'oci-reg', token: 'oci-token', tokenUsername: 'robot' });
    assert.deepEqual(entries(), ['mirror', 'oci-reg']);
    await h.tool('local_llm_registry_save', { registry: { id: 'mirror', label: 'Mirror' } });
    assert.deepEqual(entries(), ['mirror', 'oci-reg'], 'a label change keeps it');
    await h.tool('local_llm_registry_save', { registry: { id: 'mirror', url: 'https://mirror.example/other' } });
    assert.deepEqual(entries(), ['oci-reg'], 'a url change deletes it');
    await h.tool('local_llm_registry_save', { registry: { id: 'oci-reg', authOrigins: ['https://auth.registry.example'] } });
    assert.deepEqual(entries(), [], 'an authOrigins change deletes it');
    const list = await h.tool('local_llm_registry_list', {});
    assert.equal(credentialOf(list, 'mirror'), 'none');
    assert.equal(credentialOf(list, 'oci-reg'), 'none');
});

test('credential set twice keeps one entry; clear twice answers none; the last clear rewrites the file with no entries', async (t) => {
    const h = registryHarness(t);
    await h.tool('local_llm_registry_save', { registry: OCI });
    await h.tool('local_llm_registry_credential_set', { id: 'oci-reg', token: 'first-token' });
    await h.tool('local_llm_registry_credential_set', { id: 'oci-reg', token: 'second-token', tokenUsername: 'robot' });
    assert.deepEqual(Object.keys(readJson(credentialsFile(h.dataDir)).entries), ['oci-reg']);
    const service = createRegistryService({ dataDir: h.dataDir, credentialsKey: KEY });
    const credential = service.snapshot('oci-reg').credential;
    assert.equal(credential.token === 'second-token' && credential.username === 'robot', true, 'the second set replaced the first');
    // The snapshot's credential is not part of its serialized form.
    assert.equal(JSON.stringify(service.snapshot('oci-reg')).includes('second-token'), false);
    assert.deepEqual(await h.tool('local_llm_registry_credential_clear', { id: 'oci-reg' }), { id: 'oci-reg', credential: 'none' });
    assert.deepEqual(await h.tool('local_llm_registry_credential_clear', { id: 'oci-reg' }), { id: 'oci-reg', credential: 'none' });
    // The file stays, rewritten with no entries; it is never deleted.
    assert.equal(fs.existsSync(credentialsFile(h.dataDir)), true);
    assert.deepEqual(readJson(credentialsFile(h.dataDir)).entries, {});
    assert.equal(fs.statSync(credentialsFile(h.dataDir)).mode & 0o777, 0o600);
    // Clearing works without the key too, and an unknown id is invalid_registry.
    const keyless = registryHarness(t, { dataDir: h.dataDir, credentialsKey: null });
    assert.deepEqual(await keyless.tool('local_llm_registry_credential_clear', { id: 'oci-reg' }), { id: 'oci-reg', credential: 'none' });
    assert.equal(await codeOf(keyless.tool('local_llm_registry_credential_clear', { id: 'nope' })), 'invalid_registry');
    assert.equal(await codeOf(keyless.tool('local_llm_registry_credential_clear', { id: 'huggingface' })), 'read_only');
});

test('token bounds (probe): 4096 characters are accepted, 4097 are not; tokenUsername refuses a newline, a colon and 257 characters', async (t) => {
    const h = registryHarness(t);
    await h.tool('local_llm_registry_save', { registry: OCI });
    assert.deepEqual(await h.tool('local_llm_registry_credential_set', { id: 'oci-reg', token: 't'.repeat(4096) }), { id: 'oci-reg', credential: 'set' });
    assert.deepEqual(await h.tool('local_llm_registry_credential_set', { id: 'oci-reg', token: 'tok', tokenUsername: 'u'.repeat(256) }),
        { id: 'oci-reg', credential: 'set' });
    for (const input of [
        { id: 'oci-reg', token: 't'.repeat(4097) },
        { id: 'oci-reg', token: '' },
        { id: 'oci-reg', token: 'line\nbreak' },
        { id: 'oci-reg', token: 'tok', tokenUsername: 'robot\nname' },
        { id: 'oci-reg', token: 'tok', tokenUsername: 'ro:bot' },
        { id: 'oci-reg', token: 'tok', tokenUsername: 'u'.repeat(257) },
        { id: 'oci-reg', token: 'tok', tokenUsername: '' },
        { id: 'oci-reg', token: 42 },
    ]) {
        assert.equal(await codeOf(h.controller.setRegistryCredential(input)), 'invalid_registry', JSON.stringify(input).slice(0, 80));
    }
    assert.equal(await codeOf(h.controller.setRegistryCredential({ id: 'oci-reg', token: 'tok', secret: 'x' })), 'invalid_registry');
    // Every refusal left the last good credential in place.
    assert.equal(createRegistryService({ dataDir: h.dataDir, credentialsKey: KEY }).snapshot('oci-reg').credential.username === 'u'.repeat(256), true);
});

test('key derivation accepts exactly 64 lowercase hex characters (probe: 63 hex is unavailable)', () => {
    assert.ok(deriveCredentialKeys(KEY));
    for (const key of [KEY.slice(1), `${KEY}a`, KEY.toUpperCase(), ` ${KEY.slice(1)}`, null, undefined, 42]) {
        assert.equal(deriveCredentialKeys(key), null, String(key).length);
    }
    const one = deriveCredentialKeys(KEY);
    const two = deriveCredentialKeys(OTHER_KEY);
    assert.equal(one.encKey.length, 32);
    assert.match(one.keyId, /^[0-9a-f]{16}$/);
    assert.notEqual(one.keyId, two.keyId);
    assert.equal(deriveCredentialKeys(KEY).keyId, one.keyId, 'deterministic');
});
