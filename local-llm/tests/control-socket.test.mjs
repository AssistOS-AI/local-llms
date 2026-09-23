import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { callController, startControlServer } from '../src/controlSocket.mjs';

function tempRoot(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-socket-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    return root;
}

test('the socket is private: a 0700 directory and a 0600 socket', async (t) => {
    const socketPath = path.join(tempRoot(t), 'run', 'c.sock');
    const server = await startControlServer({ socketPath, handlers: { ping: () => 'pong' } });
    t.after(() => server.close());
    assert.equal(fs.statSync(path.dirname(socketPath)).mode & 0o777, 0o700);
    assert.equal(fs.statSync(socketPath).mode & 0o777, 0o600);
    assert.equal(await callController('ping', {}, { socketPath }), 'pong');
    await assert.rejects(() => callController('nope', {}, { socketPath }), { code: 'unknown_op' });
});

test('an existing open directory is tightened, and a symlinked directory is refused', async (t) => {
    const root = tempRoot(t);
    const open = path.join(root, 'open');
    fs.mkdirSync(open, { mode: 0o777 });
    fs.chmodSync(open, 0o777);
    const server = await startControlServer({ socketPath: path.join(open, 'c.sock'), handlers: {} });
    await server.close();
    assert.equal(fs.statSync(open).mode & 0o777, 0o700);

    const target = path.join(root, 'elsewhere');
    fs.mkdirSync(target, { mode: 0o700 });
    fs.symlinkSync(target, path.join(root, 'link'));
    await assert.rejects(
        () => startControlServer({ socketPath: path.join(root, 'link', 'c.sock'), handlers: {} }),
        /must be a 0700 directory owned by this user/,
    );
});
