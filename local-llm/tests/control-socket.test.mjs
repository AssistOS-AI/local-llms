// The control channel between the controller and the processes AgentServer
// spawns (runners plan, I5 / Phase S1): a per-start Linux abstract socket,
// which exists only in this agent's network namespace, and a per-start token
// that only the controller's own children receive. No file in /dev/shm.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import test from 'node:test';

import { callController, newControlChannel, startControlServer } from '../src/controlSocket.mjs';
import { LocalLlmError } from '../src/errors.mjs';

const RESPONDER = new URL('../src/chatResponder.mjs', import.meta.url).pathname;
const TOKEN = crypto.randomBytes(32).toString('base64url');

const socketName = () => `@local-llm-test-${crypto.randomBytes(8).toString('hex')}`;

function abstractSockets() {
    return fs.readFileSync('/proc/net/unix', 'utf8').split('\n').slice(1)
        .map((line) => line.trim().split(/\s+/)[7]).filter((name) => name?.startsWith('@'));
}

async function server(t, handlers) {
    const socketPath = socketName();
    const control = await startControlServer({ socketPath, token: TOKEN, handlers });
    t.after(() => control.close());
    return { socketPath, control };
}

test('the controller listens on an abstract socket, not a file', async (t) => {
    const { socketPath } = await server(t, { ping: () => 'pong' });
    // An abstract socket is listed with a leading '@' and no filesystem path.
    assert.ok(abstractSockets().includes(socketPath), `${socketPath} in /proc/net/unix`);
    assert.equal(await callController('ping', {}, { socketPath, token: TOKEN }), 'pong');
    await assert.rejects(() => callController('nope', {}, { socketPath, token: TOKEN }), { code: 'unknown_op' });
});

test('each start gets a new abstract name and a new 32-byte token', async (t) => {
    const first = newControlChannel();
    const second = newControlChannel();
    assert.match(first.socketPath, /^@local-llm-[0-9a-f]{16}$/);
    assert.match(first.token, /^[A-Za-z0-9_-]{43}$/);
    assert.notEqual(first.socketPath, second.socketPath);
    assert.notEqual(first.token, second.token);
    const control = await startControlServer({ ...first, handlers: { ping: () => 'pong' } });
    t.after(() => control.close());
    assert.equal(await callController('ping', {}, first), 'pong');
});

test('a request without the per-start token, or with another one, is refused before any handler runs', async (t) => {
    let calls = 0;
    const { socketPath } = await server(t, { run: () => { calls += 1; return 'ran'; } });
    for (const token of [undefined, '', 'x', crypto.randomBytes(32).toString('base64url'), `${TOKEN}x`, TOKEN.slice(1)]) {
        await assert.rejects(() => callController('run', {}, { socketPath, token }), (error) => {
            assert.equal(error.code, 'unauthorized');
            assert.ok(!error.message.includes(TOKEN));
            return true;
        });
    }
    // A raw client that sends no token field at all.
    const reply = await new Promise((resolve, reject) => {
        const socket = net.createConnection(`\0${socketPath.slice(1)}`);
        let buffer = '';
        socket.setEncoding('utf8');
        socket.on('connect', () => socket.write(`${JSON.stringify({ op: 'run', args: {} })}\n`));
        socket.on('data', (chunk) => { buffer += chunk; });
        socket.on('end', () => resolve(JSON.parse(buffer)));
        socket.on('error', reject);
    });
    assert.equal(reply.ok, false);
    assert.equal(reply.error.code, 'unauthorized');
    assert.equal(calls, 0);
    assert.equal(await callController('run', {}, { socketPath, token: TOKEN }), 'ran');
    assert.equal(calls, 1);
});

test('a filesystem socket path, or a missing or short token, is refused at start', async () => {
    for (const socketPath of ['/dev/shm/local-llm/controller.sock', 'relative.sock', '@', '@bad name', '']) {
        await assert.rejects(() => startControlServer({ socketPath, token: TOKEN, handlers: {} }), /abstract socket/);
    }
    for (const token of [undefined, '', 'short']) {
        await assert.rejects(() => startControlServer({ socketPath: socketName(), token, handlers: {} }), /token/);
    }
});

test('clients read the socket and the token from their environment', async (t) => {
    const saved = { socket: process.env.LOCAL_LLM_SOCKET, token: process.env.LOCAL_LLM_CONTROL_TOKEN };
    t.after(() => {
        for (const [key, value] of [['LOCAL_LLM_SOCKET', saved.socket], ['LOCAL_LLM_CONTROL_TOKEN', saved.token]]) {
            if (value === undefined) delete process.env[key];
            else process.env[key] = value;
        }
    });
    const { socketPath } = await server(t, { ping: () => 'pong' });
    process.env.LOCAL_LLM_SOCKET = socketPath;
    process.env.LOCAL_LLM_CONTROL_TOKEN = TOKEN;
    assert.equal(await callController('ping'), 'pong');
    delete process.env.LOCAL_LLM_CONTROL_TOKEN;
    await assert.rejects(() => callController('ping'), { code: 'unauthorized' });
    delete process.env.LOCAL_LLM_SOCKET;
    await assert.rejects(() => callController('ping'), { code: 'controller_unavailable' });
});

// The chat responder runs as a separate process, spawned by AgentServer with
// its environment: with the token it reaches the controller, without it not.
function runResponder(env) {
    return new Promise((resolve) => {
        const child = spawn(process.execPath, [RESPONDER], { env: { PATH: process.env.PATH, ...env }, stdio: ['pipe', 'pipe', 'pipe'] });
        let stdout = '';
        child.stdout.on('data', (chunk) => { stdout += chunk; });
        child.on('exit', (code) => resolve({ code, reply: JSON.parse(stdout) }));
        child.stdin.end(JSON.stringify({ request: { messages: [{ role: 'user', content: 'hi' }] } }));
    });
}

test('the chat responder still reaches the controller with the token from its environment', async (t) => {
    let calls = 0;
    const { socketPath } = await server(t, {
        chatTarget: () => { calls += 1; throw new LocalLlmError('not_ready', 'No model is ready.'); },
    });
    const withToken = await runResponder({ LOCAL_LLM_SOCKET: socketPath, LOCAL_LLM_CONTROL_TOKEN: TOKEN });
    assert.equal(withToken.reply.error, 'not_ready');
    assert.equal(withToken.reply.status, 503);
    assert.equal(calls, 1);
    const without = await runResponder({ LOCAL_LLM_SOCKET: socketPath });
    assert.equal(without.reply.error, 'unauthorized');
    assert.equal(calls, 1);
});
