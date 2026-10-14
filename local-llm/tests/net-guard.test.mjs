// The network guard (DS002): address classes, the guarded transport (DNS, IP literals, the connected address,
// identity encoding, timeouts, null bodies, its own agents) and followRedirects. Every refusal has a twin that must
// connect, so a guard that refused everything, or nothing, fails here. Resolvers are injected; only loopback is used.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { downloadArtifact, downloadSnapshotFile } from '../src/controller/downloader.mjs';
import { DownloadError } from '../src/controller/downloadError.mjs';
import {
    HEADERS_TIMEOUT_MS,
    IDLE_TIMEOUT_MS,
    classifyAddress,
    createGuardedTransport,
    followRedirects,
} from '../src/controller/netGuard.mjs';
import { originGuardedFetch } from '../src/controller/modelLookup.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { allowing, cannedResponse, countingListener, createTestTransport, fakeServer, scriptedResolver, scriptedTransport } from './net-fixture.mjs';

const refusedWith = (reason) => (error) => {
    assert.ok(error instanceof DownloadError, `a DownloadError, not ${error?.name}: ${error?.message}`);
    assert.equal(error.code, 'ADDRESS_REFUSED');
    assert.equal(error.retryable, false);
    if (reason) assert.equal(error.details.reason, reason);
    return true;
};
const settle = (promise) => promise.then((value) => ({ value }), (error) => ({ error }));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test('classifyAddress refuses every non-global class and allows global addresses', () => {
    const refused = {
        '0.0.0.0': 'unspecified', '0.1.2.3': 'unspecified', '10.0.0.5': 'private', '100.64.0.1': 'private', '100.127.255.255': 'private',
        '127.0.0.1': 'loopback', '127.255.0.1': 'loopback', '169.254.169.254': 'metadata', '169.254.1.1': 'link-local',
        '172.16.0.1': 'private', '172.31.255.255': 'private', '192.0.0.8': 'reserved', '192.0.2.1': 'documentation', '192.88.99.1': 'reserved',
        '192.168.1.1': 'private', '198.18.0.1': 'reserved', '198.19.255.255': 'reserved', '198.51.100.7': 'documentation',
        '203.0.113.9': 'documentation', '224.0.0.1': 'multicast', '239.255.255.255': 'multicast', '240.0.0.1': 'reserved',
        '255.255.255.255': 'reserved',
        '::1': 'non-global-ipv6', '::': 'non-global-ipv6', 'fe80::1': 'non-global-ipv6', 'fe80::1%eth0': 'non-global-ipv6',
        'fc00::1': 'non-global-ipv6', 'fd00:ec2::254': 'non-global-ipv6', 'ff02::1': 'non-global-ipv6', '64:ff9b::808:808': 'non-global-ipv6',
        '2001:db8::1': 'documentation', '2001::1': 'reserved', '2001:2::1': 'reserved', '2001:20::1': 'reserved', '2002:c000:204::1': 'reserved',
        // IPv4-mapped forms are judged as the IPv4 address, in both the dotted and the hex spelling.
        '::ffff:127.0.0.1': 'loopback', '::ffff:7f00:1': 'loopback', '[::ffff:7f00:1]': 'loopback', '::ffff:169.254.169.254': 'metadata',
        '::ffff:a9fe:a9fe': 'metadata', '::ffff:10.1.2.3': 'private',
        'registry.test': 'reserved', '': 'reserved',
    };
    for (const [address, reason] of Object.entries(refused)) {
        assert.deepEqual({ ...classifyAddress(address) }, { allowed: false, reason }, address);
    }
    for (const address of ['8.8.8.8', '1.1.1.1', '100.63.255.255', '100.128.0.1', '172.32.0.1', '198.20.0.1', '223.255.255.255',
        '2606:4700:4700::1111', '2a00:1450:4001::200e', '2001:4860::8888', '2001:3::1', '::ffff:8.8.8.8']) {
        assert.deepEqual({ ...classifyAddress(address) }, { allowed: true, reason: 'global' }, address);
    }
});

test('refused after DNS: a name resolving to a private, loopback or metadata address is never connected (with its twin)', async () => {
    const listener = await countingListener();
    try {
        for (const [address, reason] of [['127.0.0.1', 'loopback'], ['10.0.0.5', 'private'], ['169.254.169.254', 'metadata'],
            ['fd00:ec2::254', 'non-global-ipv6'], ['::1', 'non-global-ipv6']]) {
            const resolve = scriptedResolver({ 'registry.test': [address] });
            const transport = createGuardedTransport({ resolve, keepAlive: false });
            const outcome = await settle(transport(`https://registry.test:${listener.port}/api/models/x`, { headers: { Authorization: 'Bearer secret-token' } }));
            assert.ok(outcome.error, address);
            refusedWith(reason)(outcome.error);
            assert.equal(outcome.error.message, `registry.test resolves to a ${reason} address, which local-llm does not contact`);
            assert.ok(!outcome.error.message.includes(address) && !outcome.error.message.includes('secret'), 'no address or token in the message');
            assert.deepEqual(resolve.calls.map((call) => call.hostname), ['registry.test'], 'the resolver was asked');
        }
        assert.equal(listener.stats.connections, 0, 'nothing was connected');
        // Twin: the same request with 127.0.0.1 allowed by the classifier connects exactly once.
        const resolve = scriptedResolver({ 'registry.test': ['127.0.0.1'] });
        const twin = createGuardedTransport({ resolve, classify: allowing(['127.0.0.1'], classifyAddress), keepAlive: false });
        const outcome = await settle(twin(`https://registry.test:${listener.port}/api/models/x`));
        assert.ok(outcome.error && outcome.error.code !== 'ADDRESS_REFUSED', 'the twin got past the guard (the listener drops TLS)');
        assert.equal(listener.stats.connections, 1);
    } finally {
        await listener.close();
    }
});

test('a redirect to a host that resolves to a refused address is refused at that hop (with its twin)', async () => {
    const listener = await countingListener();
    const home = await fakeServer((req, res, url) => {
        const to = url.searchParams.get('to');
        res.writeHead(302, { Location: `https://${to}:${listener.port}/blob`, 'Content-Length': 0 });
        res.end();
    }, { host: '127.0.0.2' });
    try {
        const resolve = scriptedResolver({ 'meta.test': ['169.254.169.254'], 'loop.test': ['127.0.0.1'], 'ok.test': ['127.0.0.1'] });
        // Only the home server's address (127.0.0.2) is allowed besides production's global ones.
        const transport = createGuardedTransport({ resolve, allowHttp: true, classify: allowing(['127.0.0.2'], classifyAddress), keepAlive: false });
        const get = followRedirects(transport, { homeOrigin: home.origin });
        await assert.rejects(get(`${home.origin}/r?to=meta.test`), refusedWith('metadata'));
        await assert.rejects(get(`${home.origin}/r?to=loop.test`), refusedWith('loopback'));
        assert.deepEqual(resolve.calls.map((call) => call.hostname), ['meta.test', 'loop.test'], 'each hop went back through the resolver');
        assert.equal(listener.stats.connections, 0, 'the listener behind the refused names saw no connection');
        // Twin: a redirect to a host the classifier allows reaches its listener once.
        const twin = createGuardedTransport({ resolve, allowHttp: true, classify: allowing(['127.0.0.2', '127.0.0.1'], classifyAddress), keepAlive: false });
        const outcome = await settle(followRedirects(twin, { homeOrigin: home.origin })(`${home.origin}/r?to=ok.test`));
        assert.ok(outcome.error && outcome.error.code !== 'ADDRESS_REFUSED');
        assert.equal(listener.stats.connections, 1);
    } finally {
        await home.close();
        await listener.close();
    }
});

test('IP literals are judged before connecting and the resolver is never asked (with its twin)', async () => {
    const listener = await countingListener();
    try {
        const resolve = scriptedResolver({});
        const transport = createGuardedTransport({ resolve, keepAlive: false });
        for (const [url, reason] of [
            [`https://127.0.0.1:${listener.port}/`, 'loopback'],
            [`https://[::ffff:7f00:1]:${listener.port}/`, 'loopback'],
            [`https://[::ffff:127.0.0.1]:${listener.port}/`, 'loopback'],
            [`https://169.254.169.254:${listener.port}/`, 'metadata'],
            [`https://[::1]:${listener.port}/`, 'non-global-ipv6'],
            // WHATWG URL parsing normalizes these to 127.0.0.1.
            [`https://2130706433:${listener.port}/`, 'loopback'],
            [`https://0x7f.1:${listener.port}/`, 'loopback'],
            [`https://127.1:${listener.port}/`, 'loopback'],
        ]) {
            await assert.rejects(transport(url), refusedWith(reason), url);
        }
        assert.deepEqual(resolve.calls, [], 'the resolver was never asked for a literal');
        assert.equal(listener.stats.connections, 0);
        // Twin: the same literal with the classifier allowing it connects, still without the resolver.
        const twin = createGuardedTransport({ resolve, classify: allowing(['127.0.0.1'], classifyAddress), keepAlive: false });
        const outcome = await settle(twin(`https://127.0.0.1:${listener.port}/`));
        assert.ok(outcome.error && outcome.error.code !== 'ADDRESS_REFUSED');
        assert.equal(listener.stats.connections, 1);
        assert.deepEqual(resolve.calls, []);
    } finally {
        await listener.close();
    }
});

test('an all: true answer keeps only the allowed addresses, in both lookup callback shapes', async () => {
    const server = await fakeServer((req, res) => { res.writeHead(200, { 'Content-Length': 2 }); res.end('ok'); });
    const previous = net.getDefaultAutoSelectFamily();
    try {
        for (const autoSelect of [true, false]) {
            // With autoSelectFamily net asks the hook for every address (all: true); without it, for one.
            net.setDefaultAutoSelectFamily(autoSelect);
            const resolve = scriptedResolver({ 'multi.test': ['10.0.0.5', '127.0.0.1'] });
            const classify = allowing(['127.0.0.1'], classifyAddress);
            const transport = createGuardedTransport({ resolve, classify, allowHttp: true, keepAlive: false });
            const response = await transport(`http://multi.test:${server.port}/`);
            assert.equal(response.status, 200);
            assert.equal(await response.text(), 'ok');
            assert.deepEqual(resolve.calls, [{ hostname: 'multi.test', all: true }], 'the guard always reads every address');
            // Judged at DNS (both), then the connected address (only the allowed one).
            assert.deepEqual(classify.calls, ['10.0.0.5', '127.0.0.1', '127.0.0.1'], `autoSelectFamily ${autoSelect}`);
        }
        assert.equal(server.stats.connections, 2);
        // An answer holding only refused addresses is refused with the first refusal.
        const onlyRefused = createGuardedTransport({ resolve: scriptedResolver({ 'bad.test': ['192.168.1.1', '::ffff:169.254.169.254', '2001:db8::1'] }), allowHttp: true });
        await assert.rejects(onlyRefused(`http://bad.test:${server.port}/`), refusedWith('private'));
        for (const [address, reason] of [['::ffff:169.254.169.254', 'metadata'], ['2001:db8::1', 'documentation']]) {
            const single = createGuardedTransport({ resolve: scriptedResolver({ 'one.test': [address] }), allowHttp: true });
            await assert.rejects(single(`http://one.test:${server.port}/`), refusedWith(reason), address);
        }
        assert.equal(server.stats.connections, 2);
    } finally {
        net.setDefaultAutoSelectFamily(previous);
        await server.close();
    }
});

test('DNS rebinding: with keepAlive off each request resolves again, and a name that turns private is refused', async () => {
    const server = await fakeServer((req, res) => { res.writeHead(200, { 'Content-Length': 2 }); res.end('ok'); });
    try {
        // 127.0.0.1 stands in for a public address here: the classifier allows it.
        const resolve = scriptedResolver({ 'rebind.test': [['127.0.0.1'], ['10.0.0.5']] });
        const transport = createGuardedTransport({ resolve, classify: allowing(['127.0.0.1'], classifyAddress), allowHttp: true, keepAlive: false });
        const first = await transport(`http://rebind.test:${server.port}/one`);
        assert.equal(await first.text(), 'ok');
        await assert.rejects(transport(`http://rebind.test:${server.port}/two`), refusedWith('private'));
        assert.equal(resolve.calls.length, 2);
        assert.equal(server.stats.connections, 1);
        assert.deepEqual(server.requests.map((request) => request.url), ['/one']);
    } finally {
        await server.close();
    }
});

test('the connected address is judged again, and a refused one gets no request byte and no credential', async () => {
    const listener = await countingListener({ hold: true });
    try {
        // Allowed at DNS, refused once connected: the classifier changes its answer between the two checks.
        let calls = 0;
        const classify = (address) => { calls += 1; return calls === 1 ? { allowed: true, reason: 'global' } : classifyAddress(address); };
        const transport = createGuardedTransport({ resolve: scriptedResolver({ 'flip.test': ['127.0.0.1'] }), classify, allowHttp: true, keepAlive: false });
        await assert.rejects(transport(`http://flip.test:${listener.port}/secret`, { headers: { Authorization: 'Bearer secret-token' } }), refusedWith('loopback'));
        await sleep(50);
        assert.equal(listener.stats.connections, 1, 'the socket was opened');
        assert.equal(listener.stats.bytes, 0, 'and nothing was written to it');
        assert.equal(calls, 2);
        // Twin: the connected address allowed too, so the request is written (and the counter counts it).
        const twin = createGuardedTransport({ resolve: scriptedResolver({ 'flip.test': ['127.0.0.1'] }), classify: allowing(['127.0.0.1'], classifyAddress),
            allowHttp: true, keepAlive: false, headersTimeoutMs: 200 });
        await assert.rejects(twin(`http://flip.test:${listener.port}/secret`, { headers: { Authorization: 'Bearer secret-token' } }),
            (error) => error.code === 'HEADERS_TIMEOUT');
        assert.equal(listener.stats.connections, 2);
        assert.ok(listener.stats.bytes > 0, 'the allowed connection carried the request');
    } finally {
        await listener.close();
    }
});

test('the scheme is checked before DNS: plain http only when allowed, never another scheme or userinfo', async () => {
    const resolve = scriptedResolver({ 'x.test': ['8.8.8.8'] });
    const strict = createGuardedTransport({ resolve });
    for (const url of ['http://x.test/', 'ftp://x.test/', 'file:///etc/passwd']) {
        await assert.rejects(strict(url), (error) => error instanceof DownloadError && error.code === 'RESOLVE_FAILED' && error.details.reason === 'scheme', url);
    }
    await assert.rejects(strict('https://user:pass@x.test/'), (error) => error.code === 'RESOLVE_FAILED' && error.details.reason === 'credentials');
    assert.deepEqual(resolve.calls, []);
    assert.equal(strict.allowHttp, false);
    assert.equal(createGuardedTransport({ allowHttp: true }).allowHttp, true);
});

test('identity encoding is sent whatever the caller asks, and an encoded answer is refused', async () => {
    const server = await fakeServer((req, res, url) => {
        if (url.pathname === '/gzip') { res.writeHead(200, { 'Content-Encoding': 'gzip', 'Content-Length': 3 }); res.end('abc'); return; }
        res.writeHead(200, { 'Content-Encoding': 'identity', 'Content-Length': 2 });
        res.end('ok');
    });
    try {
        const transport = createTestTransport();
        const plain = await transport(`${server.origin}/plain`, { headers: { 'accept-encoding': 'gzip, br', Range: 'bytes=0-' } });
        assert.equal(await plain.text(), 'ok');
        assert.equal(server.requests[0].headers['accept-encoding'], 'identity');
        assert.equal(server.requests[0].headers.range, 'bytes=0-');
        await assert.rejects(transport(`${server.origin}/gzip`), (error) => error instanceof DownloadError && error.code === 'HTTP_ERROR'
            && error.details.reason === 'encoding');
        transport.close();
    } finally {
        await server.close();
    }
});

test('204, 205, 304 and a HEAD answer have a null body and read as empty without throwing', async () => {
    const server = await fakeServer((req, res, url) => {
        const status = Number(url.pathname.slice(1)) || 200;
        res.writeHead(status, status === 200 ? { 'Content-Length': 5 } : {});
        res.end(status === 200 ? 'hello' : undefined);
    });
    try {
        const transport = createTestTransport();
        for (const status of [204, 205, 304]) {
            const response = await transport(`${server.origin}/${status}`);
            assert.equal(response.status, status);
            assert.equal(response.body, null, String(status));
            assert.equal(await response.text(), '');
        }
        const head = await transport(`${server.origin}/200`, { method: 'HEAD' });
        assert.equal(head.status, 200);
        assert.equal(head.ok, true);
        assert.equal(head.body, null);
        assert.equal(head.headers.get('Content-Length'), '5');
        const get = await transport(`${server.origin}/200`);
        assert.equal(await get.text(), 'hello');
        // A pooled socket served all of them.
        assert.equal(server.stats.connections, 1);
        transport.close();
    } finally {
        await server.close();
    }
});

test('headers that do not come in time abort the request; the defaults are 300 s', async () => {
    assert.equal(HEADERS_TIMEOUT_MS, 300_000);
    assert.equal(IDLE_TIMEOUT_MS, 300_000);
    const server = await fakeServer((req, res, url) => {
        if (url.pathname === '/slow') { setTimeout(() => { res.writeHead(200, { 'Content-Length': 2 }); res.end('ok'); }, 1000); return; }
        res.writeHead(200, { 'Content-Length': 4 });
        res.write('ab');
        // The rest never comes.
    });
    try {
        const transport = createTestTransport({ headersTimeoutMs: 100, idleTimeoutMs: 100 });
        const started = Date.now();
        await assert.rejects(transport(`${server.origin}/slow`), (error) => error.name === 'TimeoutError' && error.code === 'HEADERS_TIMEOUT');
        assert.ok(Date.now() - started < 900, 'aborted at the headers timeout, not when the server answered');
        const stalled = await transport(`${server.origin}/stall`);
        await assert.rejects(stalled.text(), (error) => error.name === 'TimeoutError' && error.code === 'IDLE_TIMEOUT');
        transport.close();
    } finally {
        await server.close();
    }
});

test('an abort before the answer rejects with the signal\'s reason, and one during the body ends the body', async () => {
    const server = await fakeServer((req, res, url) => {
        if (url.pathname === '/never') return;
        res.writeHead(200, { 'Content-Length': 4 });
        res.write('ab');
    });
    try {
        const transport = createTestTransport();
        await assert.rejects(transport(`${server.origin}/x`, { signal: AbortSignal.abort() }), (error) => error.name === 'AbortError');
        const timed = AbortSignal.timeout(100);
        await assert.rejects(transport(`${server.origin}/never`, { signal: timed }), (error) => error.name === 'TimeoutError');
        const stop = new AbortController();
        const response = await transport(`${server.origin}/body`, { signal: stop.signal });
        const reading = settle((async () => { for await (const chunk of response.body) void chunk; })());
        await sleep(50);
        stop.abort();
        const outcome = await reading;
        assert.equal(outcome.error?.name, 'AbortError');
        transport.close();
    } finally {
        await server.close();
    }
});

test('each transport has its own agents and pools its own sockets; the global agents are never used', async () => {
    const server = await fakeServer((req, res) => { res.writeHead(200, { 'Content-Length': 2 }); res.end('ok'); });
    const globalCreate = http.globalAgent.createConnection;
    let globalUses = 0;
    http.globalAgent.createConnection = function counted(...args) { globalUses += 1; return globalCreate.apply(this, args); };
    try {
        const one = createTestTransport();
        const two = createTestTransport();
        for (let index = 0; index < 3; index += 1) assert.equal(await (await one(`${server.origin}/a`)).text(), 'ok');
        assert.equal(server.stats.connections, 1, 'one transport reuses its pooled socket');
        assert.equal(await (await two(`${server.origin}/b`)).text(), 'ok');
        assert.equal(server.stats.connections, 2, 'another transport never reuses it');
        const noPool = createTestTransport({ keepAlive: false });
        await (await noPool(`${server.origin}/c`)).text();
        await (await noPool(`${server.origin}/d`)).text();
        assert.equal(server.stats.connections, 4, 'without keepAlive every request connects');
        assert.equal(globalUses, 0);
        one.close();
        two.close();
    } finally {
        http.globalAgent.createConnection = globalCreate;
        await server.close();
    }
});

test('followRedirects: five hops at most, a Location is required, no step down to http, credentials stay home', async () => {
    const other = await fakeServer((req, res) => { res.writeHead(200, { 'Content-Length': 2 }); res.end('ok'); }, { host: '127.0.0.2' });
    const home = await fakeServer((req, res, url) => {
        const hops = Number(url.searchParams.get('n') ?? 0);
        if (url.pathname === '/loop') { res.writeHead(302, { Location: `/loop?n=${hops + 1}`, 'Content-Length': 0 }); res.end(); return; }
        if (url.pathname === '/nowhere') { res.writeHead(307, { 'Content-Length': 0 }); res.end(); return; }
        if (url.pathname === '/away') { res.writeHead(302, { Location: `${other.origin}/cdn`, 'Content-Length': 0 }); res.end(); return; }
        if (url.pathname === '/self') { res.writeHead(301, { Location: '/final', 'Content-Length': 0 }); res.end(); return; }
        if (url.pathname === '/ftp') { res.writeHead(302, { Location: 'ftp://127.0.0.1/x', 'Content-Length': 0 }); res.end(); return; }
        res.writeHead(200, { 'Content-Length': 2 });
        res.end('ok');
    });
    try {
        const transport = createTestTransport();
        const get = followRedirects(transport, { homeOrigin: home.origin });
        const redirect = (error) => error instanceof DownloadError && error.code === 'RESOLVE_FAILED' && error.details.reason === 'redirect';
        await assert.rejects(get(`${home.origin}/loop`), redirect);
        assert.equal(home.requests.filter((request) => request.url.startsWith('/loop')).length, 6, 'the request and five redirects');
        await assert.rejects(get(`${home.origin}/nowhere`), redirect);
        await assert.rejects(get(`${home.origin}/ftp`), redirect);
        const credentials = { Authorization: 'Bearer t', Cookie: 'c=1', 'Proxy-Authorization': 'Basic p', Range: 'bytes=5-', 'X-Other': 'kept' };
        assert.equal(await (await get(`${home.origin}/self`, { headers: credentials })).text(), 'ok');
        const final = home.requests.find((request) => request.url === '/final');
        assert.equal(final.headers.authorization, 'Bearer t', 'a same-origin hop keeps the credential');
        assert.equal(await (await get(`${home.origin}/away`, { headers: credentials })).text(), 'ok');
        const cdn = other.requests[0].headers;
        assert.equal(cdn.authorization, undefined);
        assert.equal(cdn.cookie, undefined);
        assert.equal(cdn['proxy-authorization'], undefined);
        assert.equal(cdn.range, 'bytes=5-', 'Range survives the hop');
        assert.equal(cdn['x-other'], 'kept');
        // Metadata calls refuse a hop to another origin before it is made.
        const before = other.requests.length;
        await assert.rejects(followRedirects(transport, { homeOrigin: home.origin, sameOriginOnly: true })(`${home.origin}/away`), redirect);
        assert.equal(other.requests.length, before);
        // A transport without http follows no http hop, and https never steps down to http.
        const httpsOnly = Object.assign(async () => ({ status: 302, headers: { get: () => 'http://example.test/x' }, body: null }), { allowHttp: false });
        await assert.rejects(followRedirects(httpsOnly, { homeOrigin: 'https://example.test' })('https://example.test/'), redirect);
        // The lookup's originGuardedFetch is the same follower over a fetch implementation, with its old messages.
        const viaFetch = originGuardedFetch({ fetchImpl: fetch, baseUrl: home.origin, deadline: AbortSignal.timeout(10_000), sameOriginOnly: true });
        await assert.rejects(viaFetch(`${home.origin}/away`), (error) => redirect(error) && /^Hugging Face redirected a metadata request to another origin/.test(error.message));
        transport.close();
    } finally {
        await home.close();
        await other.close();
    }
});

// ------------------------------------------------------------ every model path (A5)

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const REPO = 'acme/models';
const COMMIT = 'c'.repeat(40);
const CONFIG = Buffer.from('{"num_hidden_layers":2}\n');
const gitOid = (bytes) => crypto.createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const lfsEntry = (file, size, char) => ({ type: 'file', path: file, size, oid: 'f'.repeat(40), lfs: { oid: char.repeat(64), size, pointerSize: 134 } });
const SHARD_1 = 'm-Q4_K_M-00001-of-00002.gguf';
const SHARD_2 = 'm-Q4_K_M-00002-of-00002.gguf';

// The seeds the Run rows use: a pinned GGUF file, a pinned split GGUF and a pinned snapshot.
function runSeeds() {
    const seeds = loadSeedCatalog();
    const gguf = seeds.find((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m');
    const snapshot = seeds.find((model) => model.id === 'qwen3-4b-awq');
    const half = Math.floor(gguf.sources.gguf.size / 2);
    const split = validateModel({
        ...gguf, id: 'a5-split', sources: { gguf: {
            type: 'huggingface', repo: REPO, file: SHARD_1, revision: COMMIT, commit: COMMIT,
            shards: [{ file: SHARD_1, size: half, sha256: 'a'.repeat(64) }, { file: SHARD_2, size: gguf.sources.gguf.size - half, sha256: 'b'.repeat(64) }],
        } },
    }, { seed: true });
    // The same three on the admin registry `mirror`.
    const onMirror = (model, id, format) => validateModel({
        ...model, id, sources: { ...model.sources, [format]: { ...model.sources[format], registry: 'mirror' } },
    }, { seed: true });
    const onMs = (model, id, format) => validateModel({
        ...model, id, sources: { ...model.sources, [format]: { ...model.sources[format], registry: 'ms' } },
    }, { seed: true });
    const oci = { registry: 'reg', repository: OCI_REPO, tag: 'latest', manifestDigest: OCI.gguf.digest };
    const onOci = [
        validateModel({ ...gguf, id: 'a5o-gguf', sources: { gguf: { type: 'oci', ...oci, layout: 'cncf', file: 'm.gguf', size: gguf.sources.gguf.size, sha256: 'e'.repeat(64) } } }, { seed: true }),
        validateModel({ ...snapshot, id: 'a5o-snapshot', sources: { hf: { type: 'oci-snapshot', ...oci, files: [{ path: 'config.json', size: CONFIG.length, sha256: 'f'.repeat(64) },
            { path: 'model.safetensors', size: snapshot.sources.hf.size, sha256: '9'.repeat(64) }] } } }, { seed: true }),
    ];
    return [gguf, split, snapshot, onMirror(gguf, 'a5m-gguf', 'gguf'), onMirror(split, 'a5m-split', 'gguf'), onMirror(snapshot, 'a5m-snapshot', 'hf'),
        onMs(gguf, 'a5s-gguf', 'gguf'), onMs(split, 'a5s-split', 'gguf'), onMs(snapshot, 'a5s-snapshot', 'hf'), ...onOci];
}

// Earlier requests of a flow answered in process, so the row's own request is the one that reaches the transport.
// A key starting with "~" is a pattern for a path that is not fixed (the canary's random revision).
const answered = (answers) => {
    const served = new Map();
    return (url) => {
        for (const [key, { body, times = Infinity, status = 200, headers = {} }] of Object.entries(answers)) {
            if (key.startsWith('~') ? !new RegExp(key.slice(1)).test(url.pathname) : url.pathname !== key) continue;
            const count = served.get(key) ?? 0;
            if (count >= times) return null;
            served.set(key, count + 1);
            return cannedResponse(status, body, { 'content-type': 'application/json', ...(typeof headers === 'function' ? headers(url) : headers) });
        }
        return null;
    };
};
const matchesTarget = (target, actual) => (target instanceof RegExp ? target.test(actual) : actual === target);
const includesTarget = (target, actual) => (target instanceof RegExp ? target.test(actual) : actual.includes(target));
const GGUF_TREE = [lfsEntry('m-Q4_K_M.gguf', 4 * MIB, 'a'), lfsEntry(SHARD_1, 2 * MIB, 'b'), lfsEntry(SHARD_2, 2 * MIB, 'c')];
const SNAPSHOT_TREE = [{ type: 'file', path: 'config.json', size: CONFIG.length, oid: gitOid(CONFIG) }, lfsEntry('model.safetensors', MIB, 'd')];
const ggufEntry = (file, registry) => ({ id: 'a5-added', sources: { gguf: { type: 'huggingface', ...(registry ? { registry } : {}), repo: REPO, file, revision: 'main' } } });

// One set of rows per registry: huggingface.co (a literal-address base URL), and the admin mirror `mirror`, whose URL
// `https://localhost:<port>/p` is a name the production transport resolves to loopback. `answers` (null: none) are the
// earlier requests of the flow, and `target` (a path, or a pattern for the canary) is the request the row sends to the
// transport. A mirror's flow asks a canary before it resolves a branch.
function a5Rows({ prefix = '', registry = null } = {}) {
    const api = `${prefix}/api/models/${REPO}`;
    const info = { [api]: { body: { id: REPO, gated: false } } };
    const canaryPath = new RegExp(`^${api}/revision/local-llm-canary-[0-9a-f]{16}$`);
    const canary = registry ? { [`~${canaryPath.source}`]: { body: { error: 'Revision Not Found' }, status: 404 } } : {};
    const revision = { [`${api}/revision/main`]: { body: { sha: COMMIT } } };
    const revisionTarget = registry ? canaryPath : `${api}/revision/main`;
    const treeOf = (entries, times) => ({ [`${api}/tree/${COMMIT}`]: { body: entries, times } });
    const resolve = (file) => `${prefix}/${REPO}/resolve/${COMMIT}/${file}`;
    const named = (name) => (registry ? `${registry}: ${name}` : name);
    const as = registry ? { registry } : {};
    const seedId = (id) => (registry ? id.replace('a5-', 'a5m-') : id);
    return [
        { name: 'lookup: model info', kind: 'lookup', answers: null, target: api, act: (c) => c.lookupModel({ ...as, repo: REPO }) },
        { name: 'lookup: revision', kind: 'lookup', answers: { ...info }, target: revisionTarget, act: (c) => c.lookupModel({ ...as, repo: REPO }) },
        { name: 'lookup: tree', kind: 'lookup', answers: { ...info, ...canary, ...revision }, target: `${api}/tree/${COMMIT}`, act: (c) => c.lookupModel({ ...as, repo: REPO }) },
        { name: 'lookup: GGUF header', kind: 'lookup', answers: { ...info, ...canary, ...revision, ...treeOf(GGUF_TREE) }, target: resolve('m-Q4_K_M.gguf'),
            act: (c) => c.lookupModel({ ...as, repo: REPO, file: 'm-Q4_K_M.gguf' }) },
        { name: 'lookup: config.json', kind: 'lookup', answers: { ...info, ...canary, ...revision, ...treeOf(SNAPSHOT_TREE) }, target: resolve('config.json'),
            act: (c) => c.lookupModel({ ...as, repo: REPO, format: 'hf' }) },
        ...(registry ? [
            { name: 'lookup: revision after the canary', kind: 'lookup', answers: { ...info, ...canary }, target: `${api}/revision/main`, act: (c) => c.lookupModel({ ...as, repo: REPO }) },
            { name: 'GGUF pin: revision after the canary', kind: 'pin', answers: { ...canary }, target: `${api}/revision/main`, act: (c) => c.addModel(ggufEntry('m-Q4_K_M.gguf', registry)) },
        ] : []),
        { name: 'GGUF pin: revision', kind: 'pin', answers: null, target: revisionTarget, act: (c) => c.addModel(ggufEntry('m-Q4_K_M.gguf', registry)) },
        { name: 'GGUF pin: tree', kind: 'pin', answers: { ...canary, ...revision }, target: `${api}/tree/${COMMIT}`, act: (c) => c.addModel(ggufEntry('m-Q4_K_M.gguf', registry)) },
        { name: 'split pin: second shard', kind: 'pin', answers: { ...canary, ...revision, ...treeOf(GGUF_TREE, 1) }, target: `${api}/tree/${COMMIT}`,
            act: (c) => c.addModel(ggufEntry(SHARD_1, registry)) },
        { name: 'snapshot pin: tree', kind: 'pin', answers: { ...canary, ...revision }, target: `${api}/tree/${COMMIT}`,
            act: (c) => c.addModel({ id: 'a5-snapshot', sources: { hf: { type: 'hf-snapshot', ...as, repo: REPO, revision: 'main' } } }) },
        { name: 'GGUF Run', kind: 'run', answers: null, target: '/resolve/', act: (c) => c.run({ modelId: registry ? 'a5m-gguf' : 'qwen2.5-0.5b-instruct-q4_k_m', runnerId: 'llama.cpp', requestId: 'a5-run-gguf-01' }) },
        { name: 'split Run', kind: 'run', answers: null, target: resolve(SHARD_1), act: (c) => c.run({ modelId: seedId('a5-split'), runnerId: 'llama.cpp', requestId: 'a5-run-split-1' }) },
        { name: 'snapshot Run', kind: 'run', answers: null, target: '/resolve/', act: (c) => c.run({ modelId: registry ? 'a5m-snapshot' : 'qwen3-4b-awq', runnerId: 'vllm', requestId: 'a5-run-snap-01' }) },
    ].map((row) => ({ ...row, name: named(row.name), registry }));
}

// The same rows for the ModelScope registry `ms` (`https://localhost:<port>`): its requests are commits, the listing,
// the advisory OpenAPI and the /repo reads (a GGUF header, config.json, a download). `answers` are the earlier requests.
function msRows() {
    const api = `/api/v1/models/${REPO}`;
    const envelope = (Data) => ({ Code: 200, Data, Success: true });
    const blob = (path_, size, char) => ({ Type: 'blob', Path: path_, Size: size, Sha256: char.repeat(64) });
    const commits = { [`${api}/commits`]: { body: envelope({ Commit: [{ Id: COMMIT }] }) } };
    const filesOf = (entries) => ({ [`${api}/repo/files`]: { body: envelope({ Files: entries }) } });
    const openapi = { [`/openapi/v1/models/${REPO}`]: { body: { success: true, data: { gated: false } } } };
    const ggufFiles = [blob('m-Q4_K_M.gguf', 4 * MIB, 'a'), blob(SHARD_1, 2 * MIB, 'b'), blob(SHARD_2, 2 * MIB, 'c')];
    const snapshotFiles = [blob('config.json', CONFIG.length, 'e'), blob('model.safetensors', MIB, 'd')];
    const repoRead = /\/api\/v1\/models\/[^?]+\/repo$/;
    const add = (format, source) => (c) => c.addModel({ id: 'a5-added', sources: { [format]: { registry: 'ms', revision: 'master', repo: REPO, ...source } } });
    const gguf = (file) => add('gguf', { type: 'huggingface', file });
    const lookup = (extra = {}) => (c) => c.lookupModel({ registry: 'ms', repo: REPO, ...extra });
    return [
        { name: 'lookup: commits', kind: 'lookup', answers: null, target: `${api}/commits`, act: lookup() },
        { name: 'lookup: listing', kind: 'lookup', answers: { ...commits }, target: `${api}/repo/files`, act: lookup() },
        { name: 'lookup: openapi', kind: 'lookup', answers: { ...commits, ...filesOf(ggufFiles) }, target: `/openapi/v1/models/${REPO}`, act: lookup() },
        { name: 'lookup: GGUF header', kind: 'lookup', answers: { ...commits, ...filesOf(ggufFiles), ...openapi }, target: repoRead, act: lookup({ file: 'm-Q4_K_M.gguf' }) },
        { name: 'lookup: config.json', kind: 'lookup', answers: { ...commits, ...filesOf(snapshotFiles), ...openapi }, target: repoRead, act: lookup({ format: 'hf' }) },
        { name: 'GGUF pin: commits', kind: 'pin', answers: null, target: `${api}/commits`, act: gguf('m-Q4_K_M.gguf') },
        { name: 'GGUF pin: listing', kind: 'pin', answers: { ...commits }, target: `${api}/repo/files`, act: gguf('m-Q4_K_M.gguf') },
        { name: 'split pin: listing', kind: 'pin', answers: { ...commits }, target: `${api}/repo/files`, act: gguf(SHARD_1) },
        { name: 'snapshot pin: listing', kind: 'pin', answers: { ...commits }, target: `${api}/repo/files`, act: add('hf', { type: 'hf-snapshot' }) },
        { name: 'GGUF Run', kind: 'run', answers: null, target: repoRead, act: (c) => c.run({ modelId: 'a5s-gguf', runnerId: 'llama.cpp', requestId: 'a5-run-gguf-02' }) },
        { name: 'split Run', kind: 'run', answers: null, target: repoRead, act: (c) => c.run({ modelId: 'a5s-split', runnerId: 'llama.cpp', requestId: 'a5-run-split-2' }) },
        { name: 'snapshot Run', kind: 'run', answers: null, target: repoRead, act: (c) => c.run({ modelId: 'a5s-snapshot', runnerId: 'vllm', requestId: 'a5-run-snap-02' }) },
    ].map((row) => ({ ...row, name: `ms: ${row.name}`, registry: 'ms' }));
}
// The same rows for the OCI registry `reg` (`https://localhost:<port>`, its own origin its only auth origin): the manifest
// HEAD and GET, the token realm a 401 names, the config blob, a GGUF header, config.json, the pins and both Runs. A Run
// row's twin answers the retry after the dropped connection with 404, so the Run ends after one backoff.
const OCI_REPO = 'ai/m';
const ociDigest = (bytes) => `sha256:${crypto.createHash('sha256').update(bytes).digest('hex')}`;
function ociImage(format, layers, legacy = false) {
    const config = Buffer.from(JSON.stringify({ config: { format } }));
    const describe = (mediaType, bytes, filepath) => ({ mediaType, digest: ociDigest(bytes), size: bytes.length, ...(filepath ? { annotations: { 'org.cncf.model.filepath': filepath } } : {}) });
    const bytes = Buffer.from(JSON.stringify({
        schemaVersion: 2, mediaType: 'application/vnd.oci.image.manifest.v1+json', ...(legacy ? {} : { artifactType: 'application/vnd.cncf.model.manifest.v1+json' }),
        config: describe(legacy ? 'application/vnd.docker.ai.model.config.v0.1+json' : 'application/vnd.cncf.model.config.v1+json', config),
        layers: layers.map(([mediaType, body, filepath]) => describe(mediaType, body, filepath)),
    }));
    return { bytes, digest: ociDigest(bytes), config, configDigest: ociDigest(config), layers: layers.map(([, body]) => ociDigest(body)) };
}
const OCI = {
    gguf: ociImage('gguf', [['application/vnd.cncf.model.weight.v1.raw', Buffer.from('gguf bytes'), 'm.gguf']]),
    snap: ociImage('safetensors', [['application/vnd.cncf.model.weight.config.v1.raw', CONFIG, 'config.json'], ['application/vnd.cncf.model.weight.v1.raw', Buffer.from('st'), 'model.safetensors']]),
    split: ociImage('gguf', [['application/vnd.docker.ai.gguf.v3', Buffer.from('shard one')], ['application/vnd.docker.ai.gguf.v3', Buffer.from('shard two')]], true),
};
function ociRows() {
    const v2 = `/v2/${OCI_REPO}`;
    const head = (tag, image) => ({ [`${v2}/manifests/${tag}`]: { body: '', headers: { 'docker-content-digest': image.digest } } });
    const get = (image) => ({ [`${v2}/manifests/${image.digest}`]: { body: image.bytes } });
    const config = (image) => ({ [`${v2}/blobs/${image.configDigest}`]: { body: image.config } });
    const gguf = { ...head('latest', OCI.gguf), ...get(OCI.gguf) };
    const snap = { ...head('snap', OCI.snap), ...get(OCI.snap) };
    const challenge = { [`${v2}/manifests/latest`]: { status: 401, body: '', headers: (url) => ({ 'www-authenticate': `Bearer realm="${url.origin}/token",service="reg"` }) } };
    const lookup = (extra = {}) => (c) => c.lookupModel({ registry: 'reg', repo: OCI_REPO, ...extra });
    const add = (sources) => (c) => c.addModel({ id: 'a5-added', sources });
    const source = (type, tag) => ({ type, registry: 'reg', repository: OCI_REPO, tag });
    const blob = /^\/v2\/ai\/m\/blobs\/sha256:[0-9a-f]{64}$/;
    return [
        { name: 'lookup: manifest HEAD', kind: 'lookup', answers: null, target: `${v2}/manifests/latest`, act: lookup() },
        { name: 'lookup: token realm', kind: 'lookup', answers: challenge, target: /^\/token$/, act: lookup() },
        { name: 'lookup: manifest GET', kind: 'lookup', answers: head('latest', OCI.gguf), target: `${v2}/manifests/${OCI.gguf.digest}`, act: lookup() },
        { name: 'lookup: config blob', kind: 'lookup', answers: gguf, target: `${v2}/blobs/${OCI.gguf.configDigest}`, act: lookup() },
        { name: 'lookup: GGUF header', kind: 'lookup', answers: { ...gguf, ...config(OCI.gguf) }, target: `${v2}/blobs/${OCI.gguf.layers[0]}`, act: lookup({ file: 'm.gguf' }) },
        { name: 'lookup: config.json', kind: 'lookup', answers: { ...snap, ...config(OCI.snap) }, target: `${v2}/blobs/${OCI.snap.layers[0]}`, act: lookup({ revision: 'snap', format: 'hf' }) },
        { name: 'oci pin: manifest HEAD', kind: 'pin', answers: null, target: `${v2}/manifests/latest`, act: add({ gguf: source('oci', 'latest') }) },
        { name: 'oci pin: config blob', kind: 'pin', answers: gguf, target: `${v2}/blobs/${OCI.gguf.configDigest}`, act: add({ gguf: source('oci', 'latest') }) },
        { name: 'legacy split pin: shard header', kind: 'pin', answers: { ...head('split', OCI.split), ...get(OCI.split) }, target: `${v2}/blobs/${OCI.split.layers[0]}`,
            act: add({ gguf: source('oci', 'split') }) },
        { name: 'oci-snapshot pin: config blob', kind: 'pin', answers: snap, target: `${v2}/blobs/${OCI.snap.configDigest}`, act: add({ hf: source('oci-snapshot', 'snap') }) },
        { name: 'GGUF Run', kind: 'run', answers: null, once: true, target: blob, act: (c) => c.run({ modelId: 'a5o-gguf', runnerId: 'llama.cpp', requestId: 'a5-run-gguf-03' }) },
        { name: 'snapshot Run', kind: 'run', answers: null, once: true, target: blob, act: (c) => c.run({ modelId: 'a5o-snapshot', runnerId: 'vllm', requestId: 'a5-run-snap-03' }) },
    ].map((row) => ({ ...row, name: `oci: ${row.name}`, registry: 'reg' }));
}
// The twin of a Run row: the first request reaches the listener; its retry is answered 404 in process.
const onceThen404 = () => {
    let sent = false;
    return () => {
        if (!sent) { sent = true; return null; }
        return cannedResponse(404, '');
    };
};
const A5_ROWS = [...a5Rows(), ...a5Rows({ prefix: '/p', registry: 'mirror' }), ...msRows(), ...ociRows()];

// The admin registry list a mirror row's controller starts with: `mirror` at a name that resolves to loopback.
function writeMirrorRegistry(dataDir, port) {
    fs.mkdirSync(path.join(dataDir, 'state'), { recursive: true });
    fs.writeFileSync(path.join(dataDir, 'state', 'registries.json'), JSON.stringify({
        version: 1,
        registries: [{ id: 'mirror', kind: 'huggingface', url: `https://localhost:${port}/p` }, { id: 'ms', kind: 'modelscope', url: `https://localhost:${port}` },
            { id: 'reg', kind: 'oci', url: `https://localhost:${port}` }],
        builtins: {},
    }));
}

// A controller with every model path at its default; the hardware, the runners and their probe are injected.
function a5Controller(dataDir, hfBaseUrl, extra = {}) {
    return createController({
        dataDir,
        env: { PATH: '/usr/bin', HF_TOKEN: 'hf_a5_token' },
        seedCatalog: runSeeds(),
        hfBaseUrl,
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => ({
            gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: 144 * MIB, freeBytes: 6000 * MIB, processes: [] },
            memory: { totalBytes: 64 * GIB, availableBytes: 48 * GIB },
            disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
        }),
        imageContract: null,
        fileExists: () => true,
        sharedModelsRoot: null,
        installer: {
            installable: (id) => id === 'vllm',
            describe: async () => ({ installed: true, runnable: true, version: '0.30.0', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
            ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }),
            entryFor: () => ({ id: 'vllm', version: '0.30.0' }),
            pathsFor: () => ({ runDir: path.join(dataDir, 'run') }),
        },
        detectRunner: async () => ({ installed: true, version: '0.30.0', reason: null }),
        shmDir: path.join(dataDir, 'shm'),
        startRunner: () => assert.fail('no runner starts: every Run stops at its download'),
        // The runner's readiness probe is loopback to the agent's own runner, not a model path.
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        pollMs: 2,
        stopGraceMs: 50,
        ...extra,
    });
}

async function settled(controller) {
    for (let index = 0; index < 2000; index += 1) {
        const phase = controller.state.deployment?.phase;
        if (['error', 'paused', 'ready'].includes(phase)) return controller.state.deployment;
        await sleep(5);
    }
    assert.fail('the Run did not settle');
}

test('every model path goes through the production transport (with its twin)', { timeout: 120_000 }, async () => {
    const listener = await countingListener();
    const hfBaseUrl = `https://127.0.0.1:${listener.port}`;
    const realFetch = globalThis.fetch;
    let fetchCalls = 0;
    globalThis.fetch = async () => {
        fetchCalls += 1;
        throw new Error('globalThis.fetch is not a model path');
    };
    const dirs = [];
    const dataDir = () => {
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-a5-'));
        dirs.push(dir);
        return dir;
    };
    try {
        for (const row of A5_ROWS) {
            // Refused: a default controller (no transport injected) where the row has no earlier answers, else the
            // production transport with those answers in front of it.
            const inner = createGuardedTransport();
            const scripted = row.answers ? scriptedTransport(inner, answered(row.answers)) : null;
            const dir = dataDir();
            if (row.registry) writeMirrorRegistry(dir, listener.port);
            const controller = a5Controller(dir, hfBaseUrl, scripted ? { registryTransport: scripted } : {});
            // huggingface.co's base URL is a loopback literal; a mirror's is `localhost`, which resolves to IPv4 and IPv6 loopback.
            const host = row.registry ? 'localhost' : '127.0.0.1';
            const refusal = row.registry ? /^(loopback|non-global-ipv6)$/ : /^loopback$/;
            try {
                if (row.kind === 'run') {
                    await row.act(controller);
                    const deployment = await settled(controller);
                    assert.equal(deployment.phase, 'error', row.name);
                    assert.match(deployment.error, new RegExp(`^${host} resolves to a (loopback|non-global-ipv6) address, which local-llm does not contact$`), row.name);
                } else {
                    const code = row.kind === 'lookup' ? 'address_refused' : 'ADDRESS_REFUSED';
                    await assert.rejects(row.act(controller), (error) => error.code === code && refusal.test(error.details?.reason), row.name);
                }
                if (scripted) {
                    assert.equal(scripted.delegated.length, 1, `${row.name}: one request reached the transport`);
                    assert.ok(matchesTarget(row.target, scripted.delegated[0]), `${row.name}: the transport saw the row's request, not ${scripted.delegated[0]}`);
                }
                assert.equal(listener.stats.connections, 0, `${row.name}: no connection`);
            } finally {
                await controller.drain();
                inner.close();
            }
        }
        assert.equal(fetchCalls, 0, 'no model path called globalThis.fetch');

        // Twin: the same rows with loopback allowed reach the listener, each through its own request.
        for (const row of A5_ROWS) {
            const inner = createTestTransport({ allowHttp: false });
            const scripted = scriptedTransport(inner, row.once ? onceThen404() : row.answers ? answered(row.answers) : () => null);
            const before = listener.stats.connections;
            const dir = dataDir();
            if (row.registry) writeMirrorRegistry(dir, listener.port);
            const controller = a5Controller(dir, hfBaseUrl, {
                registryTransport: scripted,
                // One attempt, so the dropped connection ends the Run at once (the store still hands over its fetch).
                download: (options) => downloadArtifact({ ...options, maxAttempts: 1, sleep: async () => {} }),
                downloadSnapshot: (options) => downloadSnapshotFile({ ...options, maxAttempts: 1, sleep: async () => {} }),
            });
            try {
                if (row.kind === 'run') {
                    await row.act(controller);
                    const deployment = await settled(controller);
                    assert.ok(['paused', 'error'].includes(deployment.phase), row.name);
                    assert.doesNotMatch(String(deployment.error), /does not contact/, row.name);
                } else {
                    await assert.rejects(row.act(controller), (error) => !['address_refused', 'ADDRESS_REFUSED'].includes(error.code), row.name);
                }
                assert.equal(scripted.delegated.length, 1, `${row.name}: one request reached the transport`);
                assert.ok(includesTarget(row.target, scripted.delegated[0]), `${row.name}: ${scripted.delegated[0]}`);
                assert.ok(listener.stats.connections >= before + 1, `${row.name}: the listener saw a connection`);
            } finally {
                await controller.drain();
                inner.close();
            }
        }
        assert.equal(fetchCalls, 0, 'no model path called globalThis.fetch');
    } finally {
        globalThis.fetch = realFetch;
        await listener.close();
        for (const dir of dirs) fs.rmSync(dir, { recursive: true, force: true });
    }
});
