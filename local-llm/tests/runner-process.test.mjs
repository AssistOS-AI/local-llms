import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import test from 'node:test';

import { createLogBuffer, parseRunnerReport, startRunnerProcess } from '../src/controller/runnerProcess.mjs';

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
