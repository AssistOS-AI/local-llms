// Container-local control channel between the controller and the processes
// AgentServer spawns (MCP tools, the chat-completions responder): one JSON
// request per connection. Callers are authorized before they get here
// (Router-signed invocation, admin check in the tool); the channel itself
// only makes sure a request comes from one of the controller's own children.
//
// Two locks (runners plan, I5 / Phase S1):
//   - a Linux abstract socket, named per start. It has no file, so it is not
//     in the Box's shared /dev/shm, and it exists only in this agent's
//     network namespace, which other agents do not share;
//   - a per-start token that main.mjs hands only to AgentServer, through its
//     environment, and so to the tools and the chat responder. It is never
//     written to disk or to a log, and runner processes never receive it. It
//     still holds for an agent run with host networking.
// In the environment an abstract name is written with a leading '@' (as ss
// prints it), because an environment value cannot hold the leading NUL.

import crypto from 'node:crypto';
import net from 'node:net';

import { LocalLlmError, serializeError } from './errors.mjs';

const MAX_REQUEST_BYTES = 256 * 1024;
const ABSTRACT_NAME_RE = /^@[A-Za-z0-9._-]{1,100}$/;
const MIN_TOKEN_LENGTH = 32;

/** A fresh abstract socket name and token for one controller start. */
export function newControlChannel() {
    return {
        socketPath: `@local-llm-${crypto.randomBytes(8).toString('hex')}`,
        token: crypto.randomBytes(32).toString('base64url'),
    };
}

function abstractAddress(socketPath) {
    if (typeof socketPath !== 'string' || !ABSTRACT_NAME_RE.test(socketPath)) {
        throw new Error('The control channel must be a Linux abstract socket named "@<name>", not a file path.');
    }
    return `\0${socketPath.slice(1)}`;
}

function digest(value) {
    return crypto.createHash('sha256').update(String(value)).digest();
}

export async function startControlServer({ socketPath, token, handlers }) {
    const address = abstractAddress(socketPath);
    if (typeof token !== 'string' || token.length < MIN_TOKEN_LENGTH) {
        throw new Error(`The control channel needs a per-start token of at least ${MIN_TOKEN_LENGTH} characters.`);
    }
    const expected = digest(token);
    // Compared as fixed-length digests, so neither the length nor a prefix leaks through timing.
    const authorized = (value) => typeof value === 'string' && crypto.timingSafeEqual(digest(value), expected);
    const connections = new Set();
    const server = net.createServer((socket) => {
        connections.add(socket);
        socket.on('close', () => connections.delete(socket));
        let buffer = '';
        socket.setEncoding('utf8');
        socket.on('data', async (chunk) => {
            buffer += chunk;
            if (buffer.length > MAX_REQUEST_BYTES) {
                socket.end(`${JSON.stringify({ ok: false, error: { code: 'too_large', message: 'Request too large.' } })}\n`);
                return;
            }
            const newline = buffer.indexOf('\n');
            if (newline < 0) return;
            const line = buffer.slice(0, newline);
            buffer = '';
            let request;
            try {
                request = JSON.parse(line);
            } catch {
                socket.end(`${JSON.stringify({ ok: false, error: { code: 'bad_request', message: 'Invalid JSON.' } })}\n`);
                return;
            }
            let reply;
            try {
                if (!authorized(request?.token)) {
                    throw new LocalLlmError('unauthorized', 'The request did not carry this controller start\'s token.');
                }
                const handler = Object.hasOwn(handlers, request.op) ? handlers[request.op] : null;
                if (!handler) throw new LocalLlmError('unknown_op', `Unknown operation ${String(request.op)}.`);
                reply = { ok: true, result: await handler(request.args || {}) };
            } catch (error) {
                reply = { ok: false, error: serializeError(error) };
            }
            if (!socket.destroyed) socket.end(`${JSON.stringify(reply)}\n`);
        });
        socket.on('error', () => {});
    });
    await new Promise((resolve, reject) => {
        server.once('error', reject);
        server.listen(address, resolve);
    });
    return Object.freeze({
        socketPath,
        close() {
            return new Promise((resolve) => {
                server.close(() => resolve());
                for (const socket of connections) socket.destroy();
            });
        },
    });
}

export function callController(op, args = {}, {
    socketPath = process.env.LOCAL_LLM_SOCKET,
    token = process.env.LOCAL_LLM_CONTROL_TOKEN,
    timeoutMs = 120_000,
} = {}) {
    return new Promise((resolve, reject) => {
        let address;
        try {
            address = abstractAddress(socketPath);
        } catch {
            reject(new LocalLlmError('controller_unavailable', 'The local-llm controller is not reachable: no control channel in this process\'s environment.'));
            return;
        }
        const socket = net.createConnection(address);
        let buffer = '';
        const timer = setTimeout(() => {
            socket.destroy();
            reject(new LocalLlmError('controller_timeout', `The local-llm controller did not answer ${op} in time.`));
        }, timeoutMs);
        socket.setEncoding('utf8');
        socket.on('connect', () => socket.write(`${JSON.stringify({ op, args, token })}\n`));
        socket.on('data', (chunk) => { buffer += chunk; });
        socket.on('error', (error) => {
            clearTimeout(timer);
            reject(new LocalLlmError('controller_unavailable', `The local-llm controller is not reachable: ${error.code || error.message}`));
        });
        socket.on('end', () => {
            clearTimeout(timer);
            let reply;
            try {
                reply = JSON.parse(buffer);
            } catch {
                reject(new LocalLlmError('controller_error', 'The local-llm controller sent an invalid reply.'));
                return;
            }
            if (reply.ok) resolve(reply.result);
            else reject(new LocalLlmError(reply.error?.code || 'controller_error', reply.error?.message || 'Failed.', reply.error?.details));
        });
    });
}
