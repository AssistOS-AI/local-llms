import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
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

test('a runner that is not a group leader is still stopped', async () => {
    const log = createLogBuffer();
    const runner = startRunnerProcess({
        command: '/bin/sh', args: ['-c', 'exec sleep 60'], env: { PATH: process.env.PATH }, log,
        spawnImpl: (command, args, options) => spawn(command, args, { ...options, detached: false }),
    });
    const outcome = await Promise.race([
        runner.stop({ graceMs: 500 }).then(() => 'stopped'),
        new Promise((resolve) => setTimeout(resolve, 3000, 'hung')),
    ]);
    if (outcome === 'hung') process.kill(runner.pid, 'SIGKILL');
    assert.equal(outcome, 'stopped');
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

// A runner whose output echoes request bodies (LM Studio's llmster) brings
// a line filter: each stream gets its own, and a line it drops never reaches
// the log, in memory or in /data/logs/runner.log.
test('an output filter sees every line of each stream on its own and can drop or shorten it', async () => {
    const log = createLogBuffer();
    const child = fakeChild();
    const made = [];
    const filter = () => {
        const seen = [];
        made.push(seen);
        return (line) => { seen.push(line); return line.includes('SECRET') ? null : line.toUpperCase(); };
    };
    startRunnerProcess({ command: 'runner', args: [], env: {}, log, spawnImpl: () => child, killImpl: () => {}, filter });
    child.stdout.write('one\ntwo SECRET\nthr');
    child.stdout.end('ee\n');
    child.stderr.end('SECRET on stderr\nfour\n');
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(log.all().map((entry) => `${entry.stream}:${entry.line}`).sort(), ['stderr:FOUR', 'stdout:ONE', 'stdout:THREE']);
    assert.equal(made.length, 2, 'one filter per stream');
    assert.deepEqual(made.flat().sort(), ['SECRET on stderr', 'four', 'one', 'three', 'two SECRET']);
});

test('runner output split across chunks is logged as whole lines', async () => {
    const log = createLogBuffer();
    const child = fakeChild();
    startRunnerProcess({ command: 'runner', args: [], env: {}, log, spawnImpl: () => child, killImpl: () => {} });
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

test('a filter factory that throws stops the start before anything is spawned', () => {
    const log = createLogBuffer();
    let spawned = false;
    assert.throws(() => startRunnerProcess({ command: 'runner', args: [], env: {}, log, killImpl: () => {},
        spawnImpl: () => { spawned = true; return fakeChild(); }, filter: () => { throw new Error('bad filter'); } }), /bad filter/);
    assert.equal(spawned, false);
});
