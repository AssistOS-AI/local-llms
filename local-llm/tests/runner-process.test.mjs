import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import { PassThrough } from 'node:stream';
import test from 'node:test';

import { createLogBuffer, parseRunnerReport, startRunnerProcess } from '../src/controller/runnerProcess.mjs';

// A process counts as gone once it no longer exists or is a zombie waiting
// for its reaper.
function alive(pid) {
    try {
        const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
        return stat.slice(stat.lastIndexOf(')') + 2, stat.lastIndexOf(')') + 3) !== 'Z';
    } catch {
        return false;
    }
}

async function until(predicate, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return true;
        await new Promise((resolve) => setTimeout(resolve, 20));
    }
    return false;
}

// A runner that starts a helper process of its own, like a server that
// spawns worker processes, and reports the helper's pid.
async function runnerWithHelper(t, script) {
    const log = createLogBuffer();
    const runner = startRunnerProcess({ command: '/bin/sh', args: ['-c', script], env: { PATH: process.env.PATH }, log });
    let helper = null;
    await until(() => {
        const line = log.all().find((entry) => entry.line.startsWith('helper '));
        helper = line ? Number(line.line.slice(7)) : null;
        return helper !== null;
    });
    t.after(() => { try { process.kill(helper, 'SIGKILL'); } catch {} });
    assert.ok(Number.isInteger(helper) && alive(helper), 'the helper process started');
    return { runner, helper };
}

test('stop signals the runner\'s whole process group, so a helper it started does not outlive it', async (t) => {
    const { runner, helper } = await runnerWithHelper(t, 'sleep 60 & echo "helper $!"; wait');
    await runner.stop({ graceMs: 2000 });
    assert.equal(await until(() => !alive(helper)), true, 'the helper was stopped with the runner');
});

test('a helper left behind by a runner that exits on its own is stopped too', async (t) => {
    const { runner, helper } = await runnerWithHelper(t, 'sleep 60 & echo "helper $!"; sleep 0.5; exit 3');
    const result = await runner.exited;
    assert.equal(result.code, 3);
    assert.equal(await until(() => !alive(helper)), true, 'the helper did not outlive the runner');
});

function fakeChild() {
    const child = new EventEmitter();
    child.pid = 4242;
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = () => {};
    return child;
}

test('runner output split across chunks is logged as whole lines', async () => {
    const log = createLogBuffer();
    const child = fakeChild();
    startRunnerProcess({ command: 'runner', args: [], env: {}, log, spawnImpl: () => child });
    child.stderr.write('load_tensors:        CUDA0 model buffer size =  40');
    child.stderr.write('73.34 MiB\nllama_kv_cache:      CUDA0 KV buffer size =   384.00 MiB\nload_tensors: off');
    child.stderr.write(Buffer.from('loaded 25/25 layers to GPU — done\n', 'utf8').subarray(0, 30));
    child.stderr.write(Buffer.from('loaded 25/25 layers to GPU — done\n', 'utf8').subarray(30));
    child.stderr.end('tail without newline');
    await new Promise((resolve) => setImmediate(resolve));
    const lines = log.all().map((entry) => entry.line);
    assert.deepEqual(lines, [
        'load_tensors:        CUDA0 model buffer size =  4073.34 MiB',
        'llama_kv_cache:      CUDA0 KV buffer size =   384.00 MiB',
        'load_tensors: offloaded 25/25 layers to GPU — done',
        'tail without newline',
    ]);
    const report = parseRunnerReport(log.all());
    assert.equal(report.modelMiB, 4073.34);
    assert.deepEqual(report.offloaded, { layers: 25, of: 25 });
});
