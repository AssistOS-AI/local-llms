import assert from 'node:assert/strict';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import { readGpu } from '../src/controller/hardware.mjs';
import { getRunner } from '../src/runners/index.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const [GPT] = loadSeedCatalog();
const SMI = '/usr/local/nvidia/bin/nvidia-smi';
const GPU_QUERY = ['--query-gpu=name,memory.total,memory.used,memory.free,driver_version', '--format=csv,noheader,nounits'];
const APPS_QUERY = ['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'];
// Exact nvidia-smi output of an NVIDIA DGX Spark (GB10, driver 580.159.03) for the query above.
const GB10_LINE = 'NVIDIA GB10, [N/A], [N/A], [N/A], 580.159.03\n';

// An nvidia-smi double that answers each query with the given text and records the calls.
function smi(answers) {
    const calls = [];
    const execFileImpl = (command, args, options, callback) => {
        calls.push([command, ...args]);
        const answer = answers[args[0]];
        if (answer === undefined) callback(Object.assign(new Error('unexpected query'), { code: 2 }), '', 'unexpected');
        else callback(null, answer, '');
    };
    return { calls, execFileImpl };
}

test('a GPU that reports its memory keeps today\'s snapshot, byte for byte', async () => {
    const host = smi({
        [GPU_QUERY[0]]: 'NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 4539, 1433, 595.91.07\n',
        [APPS_QUERY[0]]: '2811, /usr/local/bin/ollama, 4502\n',
    });
    const gpu = await readGpu({ execFileImpl: host.execFileImpl, nvidiaSmi: SMI, env: {} });
    assert.deepEqual(host.calls, [[SMI, ...GPU_QUERY], [SMI, ...APPS_QUERY]]);
    assert.equal(JSON.stringify(gpu), JSON.stringify({
        available: true,
        name: 'NVIDIA GeForce RTX 3060 Laptop GPU',
        driverVersion: '595.91.07',
        totalBytes: 6144 * MIB,
        usedBytes: 4539 * MIB,
        freeBytes: 1433 * MIB,
        processes: [{ pid: 2811, name: '/usr/local/bin/ollama', usedBytes: 4502 * MIB }],
    }));
});

test('a GPU that reports no memory figures (GB10) is unavailable, with the reason', async () => {
    const host = smi({ [GPU_QUERY[0]]: GB10_LINE });
    const gpu = await readGpu({ execFileImpl: host.execFileImpl, nvidiaSmi: SMI, env: {} });
    assert.deepEqual(host.calls, [[SMI, ...GPU_QUERY]]);
    assert.equal(gpu.available, false);
    assert.equal(gpu.name, 'NVIDIA GB10');
    assert.equal(gpu.driverVersion, '580.159.03');
    assert.equal(gpu.reason, 'nvidia-smi reports no memory figures for NVIDIA GB10 (total [N/A], used [N/A], free [N/A]), '
        + 'so this agent cannot size models for it and refuses every Run. '
        + 'GPUs that share system memory, such as the one in NVIDIA DGX Spark, are not supported yet.');
    // Nothing non-finite reaches the snapshot, which the control socket sends as JSON.
    assert.equal(['totalBytes', 'usedBytes', 'freeBytes'].some((key) => key in gpu), false);
    assert.deepEqual(JSON.parse(JSON.stringify(gpu)), gpu);
});

test('any memory figure that is not a number makes the GPU unknown', async () => {
    for (const [total, used, free] of [
        ['[Not Supported]', '[Not Supported]', '[Not Supported]'],
        ['Not Supported', '0', '0'],
        ['6144', '[N/A]', '6000'],
        ['6144', '144', ''],
        ['6144', '144', '-1'],
        ['6144 MiB', '144', '6000'],
        ['NaN', '144', '6000'],
        ['Infinity', '144', '6000'],
    ]) {
        const host = smi({ [GPU_QUERY[0]]: `Some GPU, ${total}, ${used}, ${free}, 580.159.03\n` });
        const gpu = await readGpu({ execFileImpl: host.execFileImpl, nvidiaSmi: SMI, env: {} });
        assert.equal(gpu.available, false, `${total} ${used} ${free}`);
        assert.match(gpu.reason, /^nvidia-smi reports no memory figures for Some GPU/);
    }
});

test('with GB10\'s snapshot every runner refuses every model as incompatible, however large', async () => {
    const host = smi({ [GPU_QUERY[0]]: GB10_LINE });
    const gpu = await readGpu({ execFileImpl: host.execFileImpl, nvidiaSmi: SMI, env: {} });
    const snapshot = {
        gpu,
        memory: { totalBytes: 125442396 * 1024, availableBytes: 106 * GIB },
        disk: { freeBytes: 132 * 1000 ** 3 },
    };
    const huge = {
        ...GPT,
        sources: {
            gguf: { ...GPT.sources.gguf, size: 500 * GIB },
            ollama: { ...GPT.sources.ollama, size: 400 * GIB },
            hf: { ...GPT.sources.hf, size: 300 * GIB },
        },
    };
    for (const runnerId of ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm']) {
        const runner = getRunner(runnerId);
        for (const model of [GPT, huge]) {
            const source = model.sources[runner.weightFormat];
            assert.ok(source, `${runnerId} has a source`);
            const decision = admit({ runner, model, source, params: runner.normalizeParams({}, { model }), snapshot });
            assert.equal(decision.status, 'incompatible', `${runnerId} ${model.sources.gguf.size}`);
            assert.equal(decision.reason, gpu.reason);
        }
    }
});
