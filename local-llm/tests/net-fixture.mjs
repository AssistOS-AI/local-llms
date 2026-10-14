// Test seams for the network guard (DS002): a guarded transport that may also reach loopback fakes, listeners that
// count connections and bytes, scripted resolvers, and a transport wrapper that answers chosen requests in process.
// Nothing here is used by the agent itself.
import http from 'node:http';
import net from 'node:net';

import { createGuardedTransport, followRedirects } from '../src/controller/netGuard.mjs';

/** Loopback addresses, for `extraAllowed`: IPv4 127/8, `::1`, and IPv4-mapped loopback. */
export const LOOPBACK = (() => {
    const list = new net.BlockList();
    list.addSubnet('127.0.0.0', 8, 'ipv4');
    list.addAddress('::1', 'ipv6');
    list.addSubnet('::ffff:127.0.0.0', 104, 'ipv6');
    return list;
})();

/**
 * The guarded transport the tests use: production classification plus loopback (`extraAllowed`), and plain http
 * for the loopback fakes. `extraAllowed: null` gives production classification exactly.
 */
export function createTestTransport({ allowHttp = true, extraAllowed = LOOPBACK, ...options } = {}) {
    return createGuardedTransport({ allowHttp, extraAllowed, ...options });
}

/** The fetch an HTTP-level test hands the downloader: undefined (its default fetch) or the guarded test transport. */
export function fetchVia(via, baseUrl, options = {}) {
    if (via === 'fetch') return undefined;
    return followRedirects(options.transport ?? createTestTransport(), { homeOrigin: baseUrl, ...options, transport: undefined });
}

/** The two ways an HTTP-level test runs, and its name for each: the fetch run keeps the original name. */
export const VIAS = Object.freeze(['fetch', 'transport']);
export const named = (name, via) => (via === 'fetch' ? name : `${name} [transport]`);

/** A raw TCP listener that counts connections and the bytes they sent, then drops each one at once. */
export async function countingListener({ host = '127.0.0.1', hold = false } = {}) {
    const sockets = new Set();
    const stats = { connections: 0, bytes: 0 };
    const server = net.createServer((socket) => {
        stats.connections += 1;
        sockets.add(socket);
        socket.on('data', (chunk) => { stats.bytes += chunk.length; });
        socket.on('error', () => {});
        socket.on('close', () => sockets.delete(socket));
        if (!hold) socket.destroy();
    });
    await new Promise((resolve) => server.listen(0, host, resolve));
    return {
        port: server.address().port,
        stats,
        async close() {
            for (const socket of sockets) socket.destroy();
            await new Promise((resolve) => server.close(() => resolve()));
        },
    };
}

/** An HTTP fake on loopback that records requests and connections; `handler(req, res, url)` answers. */
export async function fakeServer(handler, { host = '127.0.0.1' } = {}) {
    const requests = [];
    const stats = { connections: 0 };
    const server = http.createServer((req, res) => {
        const url = new URL(req.url, 'http://placeholder.invalid');
        requests.push({ method: req.method, url: req.url, headers: { ...req.headers } });
        res.on('error', () => {});
        try {
            handler(req, res, url);
        } catch {
            res.writeHead(500);
            res.end();
        }
    });
    server.on('connection', () => { stats.connections += 1; });
    server.on('clientError', (error, socket) => socket.destroy());
    await new Promise((resolve) => server.listen(0, host, resolve));
    const port = server.address().port;
    return {
        port,
        origin: `http://${host.includes(':') ? `[${host}]` : host}:${port}`,
        requests,
        stats,
        async close() {
            server.closeAllConnections();
            await new Promise((resolve) => server.close(() => resolve()));
        },
    };
}

const familyOf = (address) => net.isIP(address);

/**
 * A resolver with dns.lookup's signature. `answers` maps a host name to a list of addresses, or to a list of such
 * lists that successive calls take in turn (the last one repeats). Every call is recorded.
 */
export function scriptedResolver(answers) {
    const calls = [];
    const counts = new Map();
    const resolve = (hostname, options, callback) => {
        calls.push({ hostname, all: Boolean(options?.all) });
        const entry = answers[hostname];
        if (!entry) {
            process.nextTick(callback, Object.assign(new Error(`getaddrinfo ENOTFOUND ${hostname}`), { code: 'ENOTFOUND' }));
            return;
        }
        const index = counts.get(hostname) ?? 0;
        counts.set(hostname, index + 1);
        const sequence = Array.isArray(entry[0]) ? entry : [entry];
        const addresses = sequence[Math.min(index, sequence.length - 1)].map((address) => ({ address, family: familyOf(address) }));
        if (options?.all) process.nextTick(callback, null, addresses);
        else process.nextTick(callback, null, addresses[0].address, addresses[0].family);
    };
    resolve.calls = calls;
    return resolve;
}

/** A classifier that allows exactly `addresses` and otherwise asks `fallback` (production by default). */
export function allowing(addresses, fallback) {
    const allowed = new Set(addresses);
    const calls = [];
    const classify = (address) => {
        calls.push(address);
        return allowed.has(address) ? { allowed: true, reason: 'global' } : fallback(address);
    };
    classify.calls = calls;
    return classify;
}

/** A response in the transport's thin shape, answered in process. */
export function cannedResponse(status, body = '', headers = {}) {
    const bytes = Buffer.isBuffer(body) ? body : Buffer.from(typeof body === 'string' ? body : JSON.stringify(body));
    const lower = Object.fromEntries(Object.entries({ 'content-length': String(bytes.length), ...headers }).map(([name, value]) => [name.toLowerCase(), String(value)]));
    return Object.freeze({
        status,
        ok: status >= 200 && status <= 299,
        headers: Object.freeze({ get: (name) => lower[String(name).toLowerCase()] ?? null }),
        body: Object.freeze({
            async* [Symbol.asyncIterator]() { if (bytes.length) yield bytes; },
            cancel: async () => {},
        }),
        text: async () => bytes.toString('utf8'),
        json: async () => JSON.parse(bytes.toString('utf8')),
    });
}

/**
 * A transport that answers the requests `script(url)` returns a response for, and sends every other request to
 * `inner` (recorded in `delegated`), so a test can reach one later request of a flow through the real transport.
 */
export function scriptedTransport(inner, script) {
    const delegated = [];
    const transport = async (url, init = {}) => {
        const canned = script(new URL(url));
        if (canned) return canned;
        delegated.push(new URL(url).pathname);
        return inner(url, init);
    };
    transport.allowHttp = inner.allowHttp;
    transport.delegated = delegated;
    return transport;
}
