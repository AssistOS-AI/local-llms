// Container-local control channel between the controller and the processes
// AgentServer spawns (MCP tools, the chat-completions responder): one JSON
// request per connection on a 0600 Unix socket. Callers are authorized
// before they get here (Router-signed invocation, admin check in the tool).

import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

import { LocalLlmError, serializeError } from './errors.mjs';

// /dev/shm, not /tmp: in nested rootless Podman the container's /tmp is
// fuse-overlayfs, which records a new socket node as root-owned 0755, so the
// controller's own user could not connect to it. /dev/shm is a private tmpfs.
export const DEFAULT_SOCKET = '/dev/shm/local-llm/controller.sock';
const MAX_REQUEST_BYTES = 256 * 1024;

// The socket must live in a real directory owned by this user with mode 0700,
// and the socket itself must come out owned by this user with mode 0600.
function preparePrivateDirectory(directory) {
    fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
    fs.chmodSync(directory, 0o700);
    const stats = fs.lstatSync(directory);
    const ownerMatches = typeof process.getuid !== 'function' || stats.uid === process.getuid();
    if (!stats.isDirectory() || !ownerMatches || (stats.mode & 0o077) !== 0) {
        throw new Error(`Refusing control socket directory ${directory}: it must be a 0700 directory owned by this user.`);
    }
}

export async function startControlServer({ socketPath = DEFAULT_SOCKET, handlers }) {
    preparePrivateDirectory(path.dirname(socketPath));
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
    const previousUmask = process.umask(0o177);
    try {
        await new Promise((resolve, reject) => {
            server.once('error', reject);
            server.listen(socketPath, resolve);
        });
    } finally {
        process.umask(previousUmask);
    }
    // Fail at startup, not on the first tool call, on a filesystem that does
    // not keep the owner or mode of a new socket node.
    const stats = fs.lstatSync(socketPath);
    const ownerMatches = typeof process.getuid !== 'function' || stats.uid === process.getuid();
    if (!stats.isSocket() || !ownerMatches || (stats.mode & 0o077) !== 0) {
        await new Promise((resolve) => server.close(() => resolve()));
        throw new Error(`Refusing control socket ${socketPath}: it came out as uid ${stats.uid} mode `
            + `${(stats.mode & 0o777).toString(8)}; use a directory on a filesystem that keeps socket ownership.`);
    }
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
