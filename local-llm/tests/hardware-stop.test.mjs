// Stopping hardware snapshots (DS001 drain, DS003): an nvidia-smi query can
// take up to 10 s and a snapshot runs up to three in turn. A Stop, a Cancel or
// the drain ends the query in progress at once (SIGKILL), waits for it to exit
// (bounded, HARDWARE_QUERY_REAP_MS), records one that does not, and never lets
// a stopped snapshot admit or launch anything.
//
// The nvidia-smi here is a fixture executable (a Node script): no GPU is used.
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { readGpu, readSnapshot, unreapedQueries } from '../src/controller/hardware.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { HARDWARE_QUERY_REAP_MS } from '../src/drainBudget.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);

function tempDir(t, prefix) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

/**
 * A fixture nvidia-smi: answers like a dedicated GPU, records every query's
 * pid in `queries.log`, and waits `delayMs` before answering the query whose
 * 1-based number is in `slowQueries`.
 */
function fixtureSmi(t, { delayMs = 8000, slowQueries = [1] } = {}) {
    const dir = tempDir(t, 'local-llm-smi-');
    const log = path.join(dir, 'queries.log');
    const smi = path.join(dir, 'nvidia-smi');
    fs.writeFileSync(smi, `#!${process.execPath}
import fs from 'node:fs';
const log = ${JSON.stringify(log)};
const arg = process.argv[2] || '';
fs.appendFileSync(log, process.pid + ' ' + arg + '\\n');
const number = fs.readFileSync(log, 'utf8').trim().split('\\n').length;
const out = arg.startsWith('--query-gpu=name,') ? 'RTX, 6144, 0, 6144, 580.1'
    : arg.startsWith('--query-gpu=pci.') ? '0x123410DE, 8.6, HMM, 0, 5, 30' : '';
setTimeout(() => { if (out) console.log(out); }, ${JSON.stringify(slowQueries)}.includes(number) ? ${delayMs} : 0);
`, { mode: 0o755 });
    const queries = () => (fs.existsSync(log) ? fs.readFileSync(log, 'utf8').trim().split('\n').filter(Boolean)
        .map((line) => ({ pid: Number(line.split(' ')[0]), arg: line.split(' ')[1] })) : []);
    return { smi, queries };
}

function alive(pid) {
    try {
        process.kill(pid, 0);
        return true;
    } catch (error) {
        return error.code !== 'ESRCH';
    }
}

async function until(predicate, label, timeoutMs = 5000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail(`timed out waiting for ${label}`);
}

// ------------------------------------------------------- the queries

test('a stopped query is killed at once, has exited when the snapshot answers, and no further query starts', async (t) => {
    const fixture = fixtureSmi(t, { delayMs: 8000, slowQueries: [1] });
    const stop = new AbortController();
    const reading = readGpu({ nvidiaSmi: fixture.smi, env: {}, signal: stop.signal });
    await until(() => fixture.queries().length === 1, 'the first query to start');
    const [query] = fixture.queries();
    assert.ok(alive(query.pid), 'the slow query is running');
    const stoppedAt = Date.now();
    stop.abort();
    await assert.rejects(reading, { name: 'AbortError', code: 'ABORT_ERR' });
    const elapsed = Date.now() - stoppedAt;
    assert.ok(elapsed < HARDWARE_QUERY_REAP_MS, `answered ${elapsed} ms after the stop`);
    assert.equal(alive(query.pid), false, 'the killed query has exited (reaped) before the answer');
    assert.equal(fixture.queries().length, 1, 'the device and process queries never started');
    assert.deepEqual(unreapedQueries(), []);
});

test('a snapshot stopped before it starts runs no query; an unstopped slow snapshot answers normally', async (t) => {
    const fixture = fixtureSmi(t, { delayMs: 300, slowQueries: [1, 2, 3] });
    const dataDir = tempDir(t, 'local-llm-snap-');
    const stopped = new AbortController();
    stopped.abort();
    let spawned = 0;
    const execFileImpl = () => { spawned += 1; throw new Error('no query may start'); };
    await assert.rejects(readSnapshot({ dataDir, signal: stopped.signal, execFileImpl }), { name: 'AbortError' });
    assert.equal(spawned, 0, 'no query was started');
    // Positive control: a signal that is never aborted changes nothing; the snapshot waits for slow queries.
    const gpu = await readGpu({ nvidiaSmi: fixture.smi, env: {}, signal: new AbortController().signal });
    assert.equal(gpu.available, true);
    assert.equal(gpu.name, 'RTX');
    assert.equal(gpu.freeBytes, 6144 * MIB);
    assert.equal(gpu.device.computeCapability, '8.6');
    assert.equal(fixture.queries().length, 3, 'all three queries ran to the end');
});

test('a killed query that does not exit is recorded after the bounded wait, until it exits', async () => {
    // A child that ignores the kill (as one stuck in the driver would).
    const child = Object.assign(new EventEmitter(), { pid: 424242, exitCode: null, signalCode: null, signals: [] });
    child.kill = (signal) => { child.signals.push(signal); return true; };
    const stop = new AbortController();
    const reading = readGpu({ execFileImpl: () => child, nvidiaSmi: '/fixture/nvidia-smi', env: {}, signal: stop.signal });
    const stoppedAt = Date.now();
    stop.abort();
    await assert.rejects(reading, { name: 'AbortError' });
    const elapsed = Date.now() - stoppedAt;
    assert.deepEqual(child.signals, ['SIGKILL']);
    assert.ok(elapsed >= HARDWARE_QUERY_REAP_MS - 20 && elapsed < HARDWARE_QUERY_REAP_MS + 500, `answered after ${elapsed} ms`);
    assert.deepEqual(unreapedQueries(), [424242]);
    child.signalCode = 'SIGKILL';
    child.emit('exit', null, 'SIGKILL');
    assert.deepEqual(unreapedQueries(), []);
});

// ------------------------------------------------- the deployment job

const DEDICATED = Object.freeze({
    gpu: { available: true, name: 'RTX', memoryModel: 'dedicated', totalBytes: 6 * GIB, freeBytes: 6 * GIB, usedBytes: 0, processes: [] },
    memory: { totalBytes: 32 * GIB, availableBytes: 28 * GIB },
    disk: { freeBytes: 400 * GIB },
});

/**
 * A controller whose `slowRead`-th snapshot takes `delayMs`. A snapshot that
 * `honours` the signal rejects when it aborts, as readSnapshot does; one that
 * does not keeps going and answers normally.
 */
function harness(t, { slowRead, delayMs = 8000, honours = true, unreaped = () => [], settle = undefined }) {
    const dataDir = tempDir(t, 'local-llm-stop-');
    const weights = path.join(dataDir, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const reads = [];
    const started = [];
    const downloads = [];
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin' },
        stateStore: createStateStore({ dataDir }),
        seedCatalog: loadSeedCatalog(),
        snapshot: ({ signal } = {}) => {
            const read = { number: reads.length + 1, signal, settled: false };
            reads.push(read);
            const delay = read.number === slowRead ? delayMs : 0;
            return new Promise((resolve, reject) => {
                const timer = setTimeout(() => { read.settled = true; resolve(structuredClone(DEDICATED)); }, delay);
                if (honours) {
                    signal?.addEventListener('abort', () => {
                        clearTimeout(timer);
                        read.settled = true;
                        reject(Object.assign(new Error('stopped'), { name: 'AbortError' }));
                    }, { once: true });
                }
            });
        },
        unreapedQueries: unreaped,
        ...(settle ? { settleStoppedQueries: settle } : {}),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        download: async ({ artifact }) => { downloads.push(artifact); return { status: 'complete', path: weights, bytesTransferred: 0 }; },
        remove: async () => 0,
        sharedModelsRoot: null,
        imageContract: null,
        installer: { installable: () => false },
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        startRunner({ args }) {
            let finish;
            const exited = new Promise((resolve) => { finish = resolve; });
            const handle = { pid: 9100, args, exited, running: true,
                async stop() { handle.running = false; finish({ code: 0, signal: 'SIGTERM' }); return exited; },
                async kill() { return handle.stop(); } };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        pollMs: 1,
        stopGraceMs: 20,
    });
    return { controller, reads, started, downloads, dataDir };
}

// A Run's snapshots: the profile (1), admission (2), after the download (3), and the last before launch (4).
const FINAL = 4;

async function runInto(h, requestId) {
    await h.controller.run({ requestId, modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' });
    await until(() => h.reads.length === FINAL, 'the final snapshot to start');
}

test('a drain during the final snapshot stops it at once, launches nothing and settles the deployment idle', async (t) => {
    const h = harness(t, { slowRead: FINAL });
    await runInto(h, 'request-drain-final');
    const began = Date.now();
    await h.controller.drain();
    const elapsed = Date.now() - began;
    assert.ok(elapsed < 1500, `the drain took ${elapsed} ms`);
    assert.equal(h.reads[FINAL - 1].signal.aborted, true, 'the snapshot was told to stop');
    assert.equal(h.started.length, 0, 'no runner after the stop');
    assert.equal(h.controller.state.deployment.phase, 'idle');
    assert.equal(h.controller.state.deployment.error, null);
});

test('a Stop during the final snapshot stops it at once and launches nothing', async (t) => {
    const h = harness(t, { slowRead: FINAL });
    await runInto(h, 'request-stop-final');
    const began = Date.now();
    await h.controller.stop();
    assert.ok(Date.now() - began < 1500);
    assert.equal(h.reads[FINAL - 1].signal.aborted, true);
    assert.equal(h.started.length, 0);
    assert.equal(h.controller.state.deployment.phase, 'idle');
});

test('a snapshot that ignores the stop and answers afterwards is not used: nothing is launched', async (t) => {
    const h = harness(t, { slowRead: FINAL, delayMs: 200, honours: false });
    await runInto(h, 'request-ignores-stop');
    await h.controller.stop();
    await until(() => h.reads[FINAL - 1].settled, 'the ignoring snapshot to answer');
    await new Promise((resolve) => setTimeout(resolve, 50));
    assert.equal(h.started.length, 0);
    assert.equal(h.controller.state.deployment.phase, 'idle');
});

test('without a stop, a slow final snapshot is waited for and the runner starts', async (t) => {
    const h = harness(t, { slowRead: FINAL, delayMs: 300 });
    await runInto(h, 'request-slow-ok');
    await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
    assert.equal(h.reads[FINAL - 1].signal.aborted, false);
    assert.equal(h.started.length, 1);
    await h.controller.stop();
});

test('a Run command in its slow admission snapshot is stopped when the drain starts, not after the command wait', async (t) => {
    // Snapshot 1 decides the profile; snapshot 2 is the Run's admission, inside the command queue.
    const h = harness(t, { slowRead: 2 });
    const running = h.controller.run({ requestId: 'request-drain-admit', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' });
    await until(() => h.reads.length === 2, 'the admission snapshot to start');
    const began = Date.now();
    await h.controller.drain();
    const elapsed = Date.now() - began;
    // The command wait is 1 s; a query stopped only after it would make the drain take that long.
    assert.ok(elapsed < 500, `the drain took ${elapsed} ms`);
    await assert.rejects(running, { code: 'shutting_down' });
    assert.equal(h.reads[1].signal.aborted, true);
    assert.equal(h.controller.state.deployment, null, 'nothing was recorded');
    assert.equal(h.started.length, 0);
});

test('a drain stops queries outside the job too, and a command that meets the drain refuses as shutting down', async (t) => {
    const h = harness(t, { slowRead: 1, unreaped: () => [424242] });
    const overview = h.controller.overview();
    await until(() => h.reads.length === 1, 'the overview snapshot to start');
    const began = Date.now();
    await h.controller.drain();
    assert.ok(Date.now() - began < 1500);
    await assert.rejects(overview, { code: 'shutting_down' });
    assert.equal(h.reads[0].signal.aborted, true);
    // A query that did not exit after its kill is named in the log.
    const lines = (await h.controller.status()).logs.map((entry) => entry.line);
    assert.ok(lines.some((line) => /nvidia-smi \(pid 424242\) was killed but has not exited/.test(line)));
});

test('the drain waits for stopped queries to settle before it reports the ones that did not exit', async (t) => {
    let settled = false;
    const h = harness(t, {
        slowRead: 1,
        // A query is recorded as not exiting only when its bounded wait ends.
        settle: () => new Promise((resolve) => setTimeout(() => { settled = true; resolve(); }, 200)),
        unreaped: () => (settled ? [515151] : []),
    });
    const overview = h.controller.overview().catch((error) => error);
    await until(() => h.reads.length === 1, 'the overview snapshot to start');
    await h.controller.drain();
    assert.equal((await overview).code, 'shutting_down');
    const lines = (await h.controller.status()).logs.map((entry) => entry.line);
    assert.ok(lines.some((line) => /nvidia-smi \(pid 515151\) was killed but has not exited/.test(line)));
});

// ----------------------------------- a Run that has not started its job yet

// A Run's first two snapshots belong to its command, before anything is recorded:
// the profile (1, while none is committed) and admission (2).
for (const [read, label] of [[1, 'profile'], [2, 'admission']]) {
    for (const op of ['stop', 'cancelDownload']) {
        for (const honours of [true, false]) {
            test(`${op} during the pending Run's ${label} snapshot${honours ? '' : ' (which ignores its signal)'} stops it at once; nothing is recorded, downloaded or started, and the same request can be sent again`, async (t) => {
                const h = harness(t, { slowRead: read, honours });
                const running = h.controller.run({ requestId: 'request-pending-01', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' })
                    .then((value) => ({ value }), (error) => ({ error }));
                await until(() => h.reads.length === read, `snapshot ${read} to start`);
                const began = Date.now();
                const answer = await h.controller[op]();
                assert.ok(Date.now() - began < 500, `${op} took ${Date.now() - began} ms against an 8 s snapshot`);
                assert.equal(h.reads[read - 1].signal.aborted, true, 'the held snapshot was told to stop');
                const outcome = await running;
                assert.equal(outcome.error?.code, 'cancelled', `the Run reports cancelled, not ${outcome.error?.code ?? 'accepted'}`);
                // A Stop or Cancel that stopped a pending Run succeeds, and leaves no deployment.
                assert.equal(answer.deployment, null);
                assert.equal(h.controller.state.deployment, null);
                assert.deepEqual(h.controller.state.requests, {});
                assert.deepEqual(h.controller.state.params, {});
                assert.equal(h.downloads.length, 0);
                assert.equal(h.started.length, 0);
                // Nothing was recorded, so the same request is a new Run, not a duplicate.
                const retried = await h.controller.run({ requestId: 'request-pending-01', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' });
                assert.equal(retried.accepted, true);
                assert.equal(retried.duplicate, undefined);
                await until(() => h.controller.state.deployment?.phase === 'ready', 'the retried Run to be ready');
                assert.equal(h.downloads.length, 1);
                assert.equal(h.started.length, 1);
                await h.controller.stop();
            });
        }
    }
    test(`without a stop, a slow ${label} snapshot is waited for and the Run is accepted (the snapshot is on the Run's path)`, async (t) => {
        const h = harness(t, { slowRead: read, delayMs: 300 });
        const began = Date.now();
        const accepted = await h.controller.run({ requestId: 'request-pending-ok', modelId: 'gpt-oss-20b', runnerId: 'llama.cpp' });
        assert.ok(Date.now() - began >= 290, 'the Run waited for the slow snapshot');
        assert.equal(accepted.accepted, true);
        assert.equal(h.reads[read - 1].signal.aborted, false);
        await until(() => h.controller.state.deployment?.phase === 'ready', 'ready');
        await h.controller.stop();
    });
}

test('a snapshot stopped while only the free-disk read is pending answers at once; the late read is handled', async (t) => {
    const unhandled = [];
    const onUnhandled = (reason) => unhandled.push(reason);
    process.on('unhandledRejection', onUnhandled);
    t.after(() => process.off('unhandledRejection', onUnhandled));
    let release;
    const statfsGate = new Promise((resolve, reject) => { release = reject; });
    // Whatever happens, the held read ends with the test.
    t.after(() => release(new Error('test ended')));
    let queries = 0;
    const answers = {
        '--query-gpu=name,memory.total,memory.used,memory.free,driver_version': 'RTX, 6144, 0, 6144, 580.1\n',
        '--query-gpu=pci.device_id,compute_cap,addressing_mode,utilization.gpu,power.draw,temperature.gpu': '0x123410DE, 8.6, HMM, 0, 5, 30\n',
    };
    const stop = new AbortController();
    const reading = readSnapshot({
        dataDir: '/nowhere',
        signal: stop.signal,
        execFileImpl: (command, args, options, callback) => {
            queries += 1;
            queueMicrotask(() => callback(null, answers[args[0]] ?? '', ''));
            return null;
        },
        statfs: () => statfsGate,
    }).then((value) => ({ value }), (error) => ({ error }));
    await until(() => queries === 3, 'the three GPU queries');
    await new Promise((resolve) => setTimeout(resolve, 10));
    const began = Date.now();
    stop.abort();
    // Bounded: a snapshot that waited for the held read would never answer here.
    const outcome = await Promise.race([reading, new Promise((resolve) => setTimeout(() => resolve({ timedOut: true }), 2000))]);
    assert.notEqual(outcome.timedOut, true, 'the stopped snapshot waited for the pending free-disk read');
    assert.ok(Date.now() - began < 100, 'the stop did not wait for statfs');
    assert.equal(outcome.error?.code, 'ABORT_ERR');
    // The free-disk read fails after the stop: handled, never unhandled.
    release(new Error('late statfs failure'));
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.deepEqual(unhandled, []);
});
