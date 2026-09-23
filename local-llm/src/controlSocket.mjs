// Container-local control channel between the controller and the processes
// AgentServer spawns (MCP tools, the chat-completions responder): one JSON
// request per connection on a 0600 Unix socket. Callers are authorized
// before they get here (Router-signed invocation, admin check in the tool).

import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

import { LocalLlmError, serializeError } from './errors.mjs';

export const DEFAULT_SOCKET = '/tmp/local-llm/controller.sock';
const MAX_REQUEST_BYTES = 256 * 1024;

export async function startControlServer({ socketPath = DEFAULT_SOCKET, handlers }) {
    fs.mkdirSync(path.dirname(socketPath), { recursive: true, mode: 0o700 });
    fs.rmSync(socketPath, { force: true });
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
            const handler = Object.hasOwn(handlers, request?.op) ? handlers[request.op] : null;
            let reply;
            try {
                if (!handler) throw new LocalLlmError('unknown_op', `Unknown operation ${String(request?.op)}.`);
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
        server.listen(socketPath, resolve);
    });
    fs.chmodSync(socketPath, 0o600);
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
    socketPath = process.env.LOCAL_LLM_SOCKET || DEFAULT_SOCKET,
    timeoutMs = 120_000,
} = {}) {
    return new Promise((resolve, reject) => {
        const socket = net.createConnection(socketPath);
        let buffer = '';
        const timer = setTimeout(() => {
            socket.destroy();
            reject(new LocalLlmError('controller_timeout', `The local-llm controller did not answer ${op} in time.`));
        }, timeoutMs);
        socket.setEncoding('utf8');
        socket.on('connect', () => socket.write(`${JSON.stringify({ op, args })}\n`));
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
