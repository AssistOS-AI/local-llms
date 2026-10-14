// The network guard for every request a model path makes (DS002): Hugging Face lookups, pins and downloads go
// through one transport that contacts only globally routable addresses. The address is checked after DNS, inside
// the `lookup` hook `net` connects with (the DNS-rebinding defence: a socket opens only to an address this hook
// returned), for IP-literal hosts before connecting, and again on the connected socket before a single request byte
// is written. Redirects are never followed by the transport; `followRedirects` follows them by hand, sends every hop
// back through the transport, and drops credentials on hops that leave their origin. Code can inject the resolver
// and the classifier (tests); nothing in the environment or in a request can relax the policy.

import dns from 'node:dns';
import http from 'node:http';
import https from 'node:https';
import net from 'node:net';

import { DownloadError } from './downloadError.mjs';

export const HEADERS_TIMEOUT_MS = 300_000;
export const IDLE_TIMEOUT_MS = 300_000;
export const MAX_REDIRECTS = 5;

const REDIRECTS = new Set([301, 302, 303, 307, 308]);
// No body for these statuses, whatever the server sends (RFC 9110); 1xx never reaches here.
const NULL_BODY_STATUSES = new Set([204, 205, 304]);
const CREDENTIAL_HEADERS = new Set(['authorization', 'cookie', 'proxy-authorization']);
const REASONS = new Set(['loopback', 'private', 'link-local', 'metadata', 'unspecified', 'reserved', 'multicast', 'documentation', 'non-global-ipv6']);

function subnets(type, cidrs) {
    const list = new net.BlockList();
    for (const cidr of cidrs) {
        const [address, prefix] = cidr.split('/');
        list.addSubnet(address, Number(prefix), type);
    }
    return list;
}

// Checked in order: the metadata address before the link-local range that holds it.
const IPV4_REFUSED = Object.freeze([
    ['metadata', subnets('ipv4', ['169.254.169.254/32'])],
    ['unspecified', subnets('ipv4', ['0.0.0.0/8'])],
    ['loopback', subnets('ipv4', ['127.0.0.0/8'])],
    ['private', subnets('ipv4', ['10.0.0.0/8', '100.64.0.0/10', '172.16.0.0/12', '192.168.0.0/16'])],
    ['link-local', subnets('ipv4', ['169.254.0.0/16'])],
    ['documentation', subnets('ipv4', ['192.0.2.0/24', '198.51.100.0/24', '203.0.113.0/24'])],
    ['multicast', subnets('ipv4', ['224.0.0.0/4'])],
    ['reserved', subnets('ipv4', ['192.0.0.0/24', '192.88.99.0/24', '198.18.0.0/15', '240.0.0.0/4'])],
]);
const IPV6_GLOBAL = subnets('ipv6', ['2000::/3']);
const IPV6_REFUSED = Object.freeze([
    ['documentation', subnets('ipv6', ['2001:db8::/32'])],
    ['reserved', subnets('ipv6', ['2001::/32', '2001:2::/48', '2001:20::/28', '2002::/16'])],
]);

// The eight 16-bit words of an IPv6 address (zone dropped, a dotted IPv4 tail folded in), or null.
function ipv6Words(address) {
    let text = address.split('%')[0];
    const tail = /(\d+\.\d+\.\d+\.\d+)$/.exec(text);
    if (tail) {
        if (net.isIPv4(tail[1]) === false) return null;
        const [a, b, c, d] = tail[1].split('.').map(Number);
        text = `${text.slice(0, -tail[1].length)}${((a << 8) | b).toString(16)}:${((c << 8) | d).toString(16)}`;
    }
    const halves = text.split('::');
    if (halves.length > 2) return null;
    const part = (value) => (value ? value.split(':').map((word) => Number.parseInt(word, 16)) : []);
    const head = part(halves[0]);
    const rest = halves.length === 2 ? part(halves[1]) : [];
    const fill = 8 - head.length - rest.length;
    if (halves.length === 1 ? head.length !== 8 : fill < 0) return null;
    const words = [...head, ...Array(halves.length === 2 ? fill : 0).fill(0), ...rest];
    return words.length === 8 && words.every((word) => Number.isInteger(word) && word >= 0 && word <= 0xffff) ? words : null;
}

const refuse = (reason) => Object.freeze({ allowed: false, reason });
const GLOBAL = Object.freeze({ allowed: true, reason: 'global' });

/**
 * Whether local-llm may contact `address` (an IP, with or without IPv6 brackets). Only globally routable addresses
 * are allowed; `::ffff:a.b.c.d` is judged as the IPv4 address it maps.
 */
export function classifyAddress(address) {
    const text = typeof address === 'string' ? address.replace(/^\[(.*)\]$/, '$1') : '';
    const family = net.isIP(text);
    if (family === 0) return refuse('reserved');
    let ipv4 = family === 4 ? text : null;
    let ipv6 = null;
    if (family === 6) {
        const words = ipv6Words(text);
        if (!words) return refuse('reserved');
        if (words.slice(0, 5).every((word) => word === 0) && words[5] === 0xffff) {
            ipv4 = [words[6] >> 8, words[6] & 0xff, words[7] >> 8, words[7] & 0xff].join('.');
        } else {
            ipv6 = words.map((word) => word.toString(16)).join(':');
        }
    }
    if (ipv4 !== null) {
        for (const [reason, list] of IPV4_REFUSED) if (list.check(ipv4, 'ipv4')) return refuse(reason);
        return GLOBAL;
    }
    if (!IPV6_GLOBAL.check(ipv6, 'ipv6')) return refuse('non-global-ipv6');
    for (const [reason, list] of IPV6_REFUSED) if (list.check(ipv6, 'ipv6')) return refuse(reason);
    return GLOBAL;
}

function refusalError(host, reason) {
    return new DownloadError('ADDRESS_REFUSED', `${host} resolves to a ${reason} address, which local-llm does not contact`, {
        details: { host, reason },
        retryable: false,
    });
}

function timeoutError(what) {
    const error = new Error(what === 'headers' ? 'The server sent no response headers in time' : 'The connection was idle for too long');
    error.name = 'TimeoutError';
    error.code = what === 'headers' ? 'HEADERS_TIMEOUT' : 'IDLE_TIMEOUT';
    return error;
}

function headerValue(headers, name) {
    const value = headers[String(name).toLowerCase()];
    if (value === undefined) return null;
    return Array.isArray(value) ? value.join(', ') : String(value);
}

// The thin response the model paths use: status, ok, headers.get, body iteration, cancel, json and text.
function wrapResponse(res, nullBody) {
    const status = res.statusCode;
    const body = nullBody ? null : Object.freeze({
        [Symbol.asyncIterator]: () => res[Symbol.asyncIterator](),
        cancel: async () => { res.destroy(); },
    });
    const readAll = async () => {
        if (!body) return Buffer.alloc(0);
        const parts = [];
        for await (const chunk of res) parts.push(chunk);
        return Buffer.concat(parts);
    };
    return Object.freeze({
        status,
        ok: status >= 200 && status <= 299,
        headers: Object.freeze({ get: (name) => headerValue(res.headers, name) }),
        body,
        text: async () => (await readAll()).toString('utf8'),
        json: async () => JSON.parse((await readAll()).toString('utf8')),
    });
}

function extraAllows(extraAllowed, address) {
    if (!extraAllowed) return false;
    const text = String(address).replace(/^\[(.*)\]$/, '$1');
    const family = net.isIP(text);
    if (family === 0) return false;
    if (typeof extraAllowed === 'function') return extraAllowed(text) === true;
    return typeof extraAllowed.check === 'function' && extraAllowed.check(text, family === 6 ? 'ipv6' : 'ipv4');
}

/**
 * The guarded transport: `transport(url, { method, headers, signal })` resolves to the thin response. `resolve`,
 * `classify`, `allowHttp` and `extraAllowed` are code-only test seams. Each transport owns its own agents, so its
 * pooled sockets were all opened under this one policy and no other code's sockets are shared.
 */
export function createGuardedTransport({
    resolve = dns.lookup,
    classify = classifyAddress,
    allowHttp = false,
    extraAllowed = null,
    keepAlive = true,
    headersTimeoutMs = HEADERS_TIMEOUT_MS,
    idleTimeoutMs = IDLE_TIMEOUT_MS,
} = {}) {
    const agents = Object.freeze({ 'http:': new http.Agent({ keepAlive }), 'https:': new https.Agent({ keepAlive }) });
    const judge = (address) => {
        if (extraAllows(extraAllowed, address)) return GLOBAL;
        const verdict = classify(String(address).replace(/^\[(.*)\]$/, '$1'));
        if (verdict?.allowed === true) return verdict;
        return refuse(REASONS.has(verdict?.reason) ? verdict.reason : 'reserved');
    };

    async function transport(url, { method = 'GET', headers = {}, signal } = {}) {
        const target = new URL(url);
        if (target.protocol !== 'https:' && !(allowHttp && target.protocol === 'http:')) {
            throw new DownloadError('RESOLVE_FAILED', 'local-llm fetches models over https only', { details: { reason: 'scheme' } });
        }
        if (target.username || target.password) {
            throw new DownloadError('RESOLVE_FAILED', 'A registry URL must not carry credentials', { details: { reason: 'credentials' } });
        }
        if (signal?.aborted) throw signal.reason;
        const host = target.hostname.replace(/^\[(.*)\]$/, '$1');
        const literal = net.isIP(host) !== 0;
        // `net` skips the lookup hook for an IP literal, so a literal is judged here, before any connection.
        if (literal) {
            const verdict = judge(host);
            if (!verdict.allowed) throw refusalError(target.hostname, verdict.reason);
        }
        return new Promise((resolvePromise, reject) => {
            let refusal = null;
            let settled = false;
            let response = null;
            const lookup = (hostname, options, callback) => {
                const answer = (error, result, family) => {
                    if (error) {
                        callback(error);
                        return;
                    }
                    const entries = Array.isArray(result) ? result
                        : typeof result === 'string' ? [{ address: result, family: family || net.isIP(result) }] : [];
                    const allowed = [];
                    let refusedReason = null;
                    for (const entry of entries) {
                        const verdict = judge(entry?.address);
                        if (verdict.allowed) allowed.push({ address: entry.address, family: entry.family || net.isIP(entry.address) });
                        else refusedReason ??= verdict.reason;
                    }
                    if (allowed.length === 0) {
                        refusal = refusalError(target.hostname, refusedReason ?? 'reserved');
                        callback(refusal);
                        return;
                    }
                    if (options?.all) callback(null, allowed);
                    else callback(null, allowed[0].address, allowed[0].family);
                };
                try {
                    resolve(hostname, { ...(typeof options === 'object' ? options : {}), all: true }, answer);
                } catch (error) {
                    callback(error);
                }
            };
            const requestHeaders = {};
            for (const [name, value] of Object.entries(headers ?? {})) {
                if (name.toLowerCase() !== 'accept-encoding') requestHeaders[name] = value;
            }
            // Identity encoding keeps Content-Length and byte ranges in raw bytes.
            requestHeaders['Accept-Encoding'] = 'identity';
            const req = (target.protocol === 'https:' ? https : http).request(target, {
                method,
                headers: requestHeaders,
                agent: agents[target.protocol],
                lookup,
                ...(target.protocol === 'https:' && !literal ? { servername: host } : {}),
            });
            const headersTimer = setTimeout(() => fail(timeoutError('headers')), headersTimeoutMs);
            headersTimer.unref?.();
            const onAbort = () => {
                if (settled) response?.destroy(signal.reason);
                else fail(signal.reason);
            };
            function fail(error) {
                if (settled) return;
                settled = true;
                clearTimeout(headersTimer);
                signal?.removeEventListener('abort', onAbort);
                req.destroy();
                reject(error);
            }
            signal?.addEventListener('abort', onAbort, { once: true });
            req.on('error', (error) => fail(refusal ?? error));
            // Nothing is written (no request line, no header, no credential) until the connected address is judged.
            req.on('socket', (socket) => {
                const check = () => {
                    if (settled) return;
                    // A peer that closed at once leaves no address to judge: nothing is written, and it is a network error.
                    if (!socket.remoteAddress) {
                        socket.destroy();
                        fail(Object.assign(new Error('The connection closed before its address could be checked'), { code: 'ECONNRESET' }));
                        return;
                    }
                    const verdict = judge(socket.remoteAddress);
                    if (!verdict.allowed) {
                        refusal ??= refusalError(target.hostname, verdict.reason);
                        socket.destroy(refusal);
                        fail(refusal);
                        return;
                    }
                    req.end();
                };
                // A pooled socket is already connected and emits no `connect`.
                if (socket.connecting) socket.once('connect', check);
                else check();
            });
            req.on('response', (res) => {
                clearTimeout(headersTimer);
                if (settled) {
                    res.destroy();
                    return;
                }
                const encoding = String(res.headers['content-encoding'] ?? '').trim().toLowerCase();
                if (encoding !== '' && encoding !== 'identity') {
                    res.destroy();
                    fail(new DownloadError('HTTP_ERROR', 'The server sent an encoded response, which local-llm does not accept', {
                        details: { status: res.statusCode, reason: 'encoding' },
                    }));
                    return;
                }
                settled = true;
                response = res;
                const socket = res.socket;
                const onIdle = () => res.destroy(timeoutError('idle'));
                socket?.setTimeout(idleTimeoutMs);
                socket?.on('timeout', onIdle);
                res.once('close', () => {
                    socket?.removeListener('timeout', onIdle);
                    signal?.removeEventListener('abort', onAbort);
                });
                const nullBody = method === 'HEAD' || NULL_BODY_STATUSES.has(res.statusCode) || res.statusCode < 200;
                if (nullBody) res.resume();
                resolvePromise(wrapResponse(res, nullBody));
            });
        });
    }
    transport.allowHttp = allowHttp === true;
    // Closes the pooled sockets (tests, and a controller that is replaced).
    transport.close = () => {
        agents['http:'].destroy();
        agents['https:'].destroy();
    };
    return transport;
}

async function discard(response) {
    try {
        await response.body?.cancel();
    } catch {
        // The connection is being dropped anyway.
    }
}

const redirectRefused = (message) => new DownloadError('RESOLVE_FAILED', message, { details: { reason: 'redirect' } });

/**
 * A fetch-like function over `transport` that follows redirects itself, at most `maxHops`, each hop back through the
 * transport. `Authorization`, `Cookie` and `Proxy-Authorization` go only to `credentialOrigins`; `Range` and the
 * other headers are sent on every hop. With `sameOriginOnly` (metadata, whose answers are taken as the registry's own)
 * a hop off `homeOrigin` is refused. https never steps down to http, and http is followed only when the transport
 * allows it. Each request also ends at `deadline`.
 */
export function followRedirects(transport, {
    homeOrigin,
    credentialOrigins = [homeOrigin],
    sameOriginOnly = false,
    maxHops = MAX_REDIRECTS,
    deadline = undefined,
    subject = 'The registry',
} = {}) {
    const home = new URL(homeOrigin).origin;
    const trusted = new Set(credentialOrigins.map((origin) => new URL(origin).origin));
    const schemes = transport.allowHttp === true ? ['https:', 'http:'] : ['https:'];
    return async function fetchLike(url, init = {}) {
        let current = new URL(url);
        for (let hops = 0; ; hops += 1) {
            const headers = {};
            for (const [name, value] of Object.entries(init.headers ?? {})) {
                if (!trusted.has(current.origin) && CREDENTIAL_HEADERS.has(name.toLowerCase())) continue;
                headers[name] = value;
            }
            const signals = [init.signal, deadline].filter(Boolean);
            const signal = signals.length > 1 ? AbortSignal.any(signals) : signals[0];
            const response = await transport(current.href, { method: init.method ?? 'GET', headers, signal });
            if (!REDIRECTS.has(response.status)) return response;
            const location = response.headers?.get?.('location');
            await discard(response);
            if (!location || hops >= maxHops) throw redirectRefused(`${subject} redirected the request too often, or without saying where.`);
            let next;
            try {
                next = new URL(location, current);
            } catch {
                throw redirectRefused(`${subject} redirected the request to an address that is not https.`);
            }
            if (!schemes.includes(next.protocol) || (current.protocol === 'https:' && next.protocol !== 'https:')) {
                throw redirectRefused(`${subject} redirected the request to an address that is not https.`);
            }
            if (sameOriginOnly && next.origin !== home) {
                throw redirectRefused(`${subject} redirected a metadata request to another origin, which is not followed.`);
            }
            current = next;
        }
    };
}

// A fetch implementation seen as a transport: it never follows redirects, and the scheme rules are the old lookup's.
function fetchAsTransport(fetchImpl) {
    if (typeof fetchImpl?.allowHttp === 'boolean') return fetchImpl;
    const adapted = (url, { method, headers, signal }) => fetchImpl(url, { method, headers, signal, redirect: 'manual' });
    adapted.allowHttp = true;
    return adapted;
}

/**
 * The lookup's redirect-following fetch (DS002): `followRedirects` over `fetchImpl`, which is the controller's guarded
 * transport or, for direct callers, a fetch implementation. The token goes only to the Hugging Face origin.
 */
export function originGuardedFetch({ fetchImpl, baseUrl, deadline, sameOriginOnly = false }) {
    return followRedirects(fetchAsTransport(fetchImpl), { homeOrigin: baseUrl, sameOriginOnly, deadline, subject: 'Hugging Face' });
}

/**
 * The body of a metadata response as text, never more than `maxBytes`: a larger Content-Length, or a body that grows
 * past it, ends the read and the connection.
 */
export async function readCapped(response, maxBytes) {
    const declared = Number(response.headers.get('content-length'));
    const tooLarge = () => new DownloadError('RESOLVE_FAILED', `Hugging Face metadata response is larger than ${maxBytes} bytes`,
        { details: { reason: 'too-large' } });
    if (Number.isFinite(declared) && declared > maxBytes) {
        await discard(response);
        throw tooLarge();
    }
    const parts = [];
    let total = 0;
    for await (const chunk of response.body ?? []) {
        total += chunk.length;
        if (total > maxBytes) {
            await discard(response);
            throw tooLarge();
        }
        parts.push(chunk);
    }
    return Buffer.concat(parts, total).toString('utf8');
}
