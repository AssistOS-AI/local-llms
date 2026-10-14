import assert from 'node:assert/strict';
import test from 'node:test';

import { admit } from '../src/controller/admission.mjs';
import { loadSeedCatalog } from '../src/controller/catalog.mjs';
import {
    DEVICE_QUERY_FIELDS,
    defaultThreads,
    memoryModelOf,
    performanceCoreCount,
    physicalCoreCount,
    readGpu,
    readMemory,
    readMemoryPressure,
} from '../src/controller/hardware.mjs';
import * as hardware from '../src/controller/hardware.mjs';
import { observeMemoryBudget } from '../src/controller/ploinkyBudget.mjs';
import { getRunner } from '../src/runners/index.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const [GPT] = loadSeedCatalog();
const SMI = '/usr/local/nvidia/bin/nvidia-smi';
const GPU_QUERY = ['--query-gpu=name,memory.total,memory.used,memory.free,driver_version', '--format=csv,noheader,nounits'];
const DEVICE_QUERY = [`--query-gpu=${DEVICE_QUERY_FIELDS}`, '--format=csv,noheader,nounits'];
const APPS_QUERY = ['--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'];
// Exact nvidia-smi output of an NVIDIA DGX Spark (GB10, driver 580.159.03) for the two queries.
const GB10_LINE = 'NVIDIA GB10, [N/A], [N/A], [N/A], 580.159.03\n';
const GB10_DEVICE = '0x2E1210DE, 12.1, ATS, 0, 4.52, 47\n';
// Today's dedicated output (the RTX 3060 Laptop GPU behind the dedicated measurements).
const X86_LINE = 'NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 4539, 1433, 595.91.07\n';
const X86_APPS = '2811, /usr/local/bin/ollama, 4502\n';

// An nvidia-smi double that answers each query with the given text and records the calls.
function smi(answers) {
    const calls = [];
    const execFileImpl = (command, args, options, callback) => {
        calls.push([command, ...args]);
        const answer = answers[args[0]];
        if (answer === undefined) callback(Object.assign(new Error('unexpected query'), { code: 2 }), '', 'Field "x" is not a valid field to query.');
        else callback(null, answer, '');
    };
    return { calls, execFileImpl };
}

const read = (answers) => {
    const host = smi(answers);
    return readGpu({ execFileImpl: host.execFileImpl, nvidiaSmi: SMI, env: {} }).then((gpu) => ({ gpu, calls: host.calls }));
};

// Today's fields of a dedicated snapshot, as readGpu returned them before the hardware profiles.
const pickToday = (gpu) => Object.fromEntries(['available', 'name', 'driverVersion', 'totalBytes', 'usedBytes', 'freeBytes', 'processes']
    .map((key) => [key, gpu[key]]));
const TODAY_X86 = {
    available: true,
    name: 'NVIDIA GeForce RTX 3060 Laptop GPU',
    driverVersion: '595.91.07',
    totalBytes: 6144 * MIB,
    usedBytes: 4539 * MIB,
    freeBytes: 1433 * MIB,
    processes: [{ pid: 2811, name: '/usr/local/bin/ollama', usedBytes: 4502 * MIB }],
};

test('golden: today\'s x86 output keeps the same first query and the same values, and is dedicated', async () => {
    const { gpu, calls } = await read({
        [GPU_QUERY[0]]: X86_LINE,
        [DEVICE_QUERY[0]]: '0x252010DE, 8.6, None, 7, 12.50, 41\n',
        [APPS_QUERY[0]]: X86_APPS,
    });
    assert.deepEqual(calls[0], [SMI, ...GPU_QUERY]);
    assert.deepEqual(pickToday(gpu), TODAY_X86);
    assert.equal(gpu.memoryModel, 'dedicated');
    assert.deepEqual(gpu.device, { pciDeviceId: '0x252010DE', computeCapability: '8.6', addressingMode: 'None' });
    assert.deepEqual(gpu.telemetry, { utilizationPercent: 7, powerWatts: 12.5, temperatureC: 41 });
});

test('an older driver that refuses the second query keeps today\'s dedicated snapshot', async () => {
    const { gpu, calls } = await read({ [GPU_QUERY[0]]: X86_LINE, [APPS_QUERY[0]]: X86_APPS });
    assert.deepEqual(calls.map((call) => call[1]), [GPU_QUERY[0], DEVICE_QUERY[0], APPS_QUERY[0]]);
    assert.deepEqual(pickToday(gpu), TODAY_X86);
    assert.equal(gpu.memoryModel, 'dedicated');
    assert.equal('device' in gpu, false);
});

test('partial telemetry (a laptop GPU with power.draw [N/A]) never disables the GPU', async () => {
    const { gpu } = await read({
        [GPU_QUERY[0]]: X86_LINE, [DEVICE_QUERY[0]]: '0x252010DE, 8.6, None, [N/A], [N/A], 44\n', [APPS_QUERY[0]]: '',
    });
    assert.equal(gpu.available, true);
    assert.equal(gpu.memoryModel, 'dedicated');
    assert.deepEqual(gpu.telemetry, { utilizationPercent: null, powerWatts: null, temperatureC: 44 });
});

test('GB10 is unified: no GPU memory figures, the pool is system memory', async () => {
    const { gpu } = await read({
        [GPU_QUERY[0]]: GB10_LINE, [DEVICE_QUERY[0]]: GB10_DEVICE, [APPS_QUERY[0]]: '59, /opt/llama.cpp/llama-server, 11875\n',
    });
    assert.deepEqual(gpu, {
        available: true,
        name: 'NVIDIA GB10',
        driverVersion: '580.159.03',
        memoryModel: 'unified',
        totalBytes: null,
        usedBytes: null,
        freeBytes: null,
        processes: [{ pid: 59, name: '/opt/llama.cpp/llama-server', usedBytes: 11875 * MIB }],
        device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
        telemetry: { utilizationPercent: 0, powerWatts: 4.52, temperatureC: 47 },
    });
    // Nothing non-finite reaches the snapshot, which the control socket sends as JSON.
    assert.deepEqual(JSON.parse(JSON.stringify(gpu)), gpu);
});

test('a GB10 whose driver reports numbers is still unified (known-unified device id)', async () => {
    const { gpu } = await read({
        [GPU_QUERY[0]]: 'NVIDIA GB10, 122502, 11875, 110627, 590.10.01\n', [DEVICE_QUERY[0]]: GB10_DEVICE, [APPS_QUERY[0]]: '',
    });
    assert.equal(gpu.memoryModel, 'unified');
    assert.equal(gpu.totalBytes, null);
});

test('GH200-like (ATS addressing with numeric HBM) is dedicated', async () => {
    const { gpu } = await read({
        [GPU_QUERY[0]]: 'NVIDIA GH200 480GB, 97871, 0, 97280, 580.95.05\n',
        [DEVICE_QUERY[0]]: '0x234210DE, 9.0, ATS, 0, 90.00, 30\n', [APPS_QUERY[0]]: '',
    });
    assert.equal(gpu.memoryModel, 'dedicated');
    assert.equal(gpu.totalBytes, 97871 * MIB);
});

test('Jetson Thor-like: unified when it reports HMM, unknown and refused when it reports no addressing mode', async () => {
    const unified = await read({
        [GPU_QUERY[0]]: 'NVIDIA Thor, [N/A], [N/A], [N/A], 580.00\n', [DEVICE_QUERY[0]]: '0x2B0010DE, 11.0, HMM, 0, [N/A], 40\n', [APPS_QUERY[0]]: '',
    });
    assert.equal(unified.gpu.memoryModel, 'unified');
    assert.equal(unified.gpu.device.computeCapability, '11.0');
    const unknown = await read({
        [GPU_QUERY[0]]: 'NVIDIA Thor, Not Supported, Not Supported, Not Supported, 580.00\n',
        [DEVICE_QUERY[0]]: '0x2B0010DE, 11.0, [N/A], 0, [N/A], 40\n',
    });
    assert.equal(unknown.gpu.available, false);
    assert.match(unknown.gpu.reason, /^nvidia-smi reports no memory figures for NVIDIA Thor/);
});

test('no memory figures and no device facts (the second query failed) is unknown, never unified', async () => {
    const { gpu, calls } = await read({ [GPU_QUERY[0]]: GB10_LINE });
    assert.deepEqual(calls.map((call) => call[1]), [GPU_QUERY[0], DEVICE_QUERY[0]]);
    assert.equal(gpu.available, false);
    assert.equal(gpu.name, 'NVIDIA GB10');
    // Deliberate change (CPU profile, DS005): a failed device query is unreadable, not a verdict on the GPU.
    assert.equal(gpu.state, 'unreadable');
    assert.equal(gpu.reason, 'nvidia-smi gave no memory figures for NVIDIA GB10 and its device query failed, '
        + 'so its memory model is not known yet');
});

test('any memory figure that is not a number, on a GPU not known to share memory, makes it unknown', async () => {
    for (const [total, used, free] of [
        ['[Not Supported]', '[Not Supported]', '[Not Supported]'],
        ['6144', '[N/A]', '6000'],
        ['6144', '144', ''],
        ['6144', '144', '-1'],
        ['6144 MiB', '144', '6000'],
        ['NaN', '144', '6000'],
        ['Infinity', '144', '6000'],
    ]) {
        const { gpu } = await read({
            [GPU_QUERY[0]]: `Some GPU, ${total}, ${used}, ${free}, 580.159.03\n`, [DEVICE_QUERY[0]]: '0x11112222, 8.9, None, 0, 1, 30\n',
        });
        assert.equal(gpu.available, false, `${total} ${used} ${free}`);
        assert.match(gpu.reason, /^nvidia-smi reports no memory figures for Some GPU/);
    }
    assert.equal(memoryModelOf({ memoryNumeric: false, device: { addressingMode: 'None' } }), 'unknown');
    assert.equal(memoryModelOf({ memoryNumeric: false, device: null }), 'unknown');
});

test('an unknown GPU is refused for every runner and every model, however large', async () => {
    const { gpu } = await read({ [GPU_QUERY[0]]: GB10_LINE });
    const snapshot = { gpu, memory: { totalBytes: 125442396 * 1024, availableBytes: 106 * GIB }, disk: { freeBytes: 132 * 1000 ** 3 } };
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
            assert.equal(decision.status, 'incompatible', runnerId);
            assert.equal(decision.reason, gpu.reason);
        }
    }
});

function fakeFs(files) {
    return { readFileSync(file) { if (Object.hasOwn(files, file)) return files[file]; throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' }); } };
}

test('performance cores: the highest-capacity class of the allowed CPUs, only when capacities differ', () => {
    // DGX Spark: A725 at 718-731 on cpu0-4 and 10-14, X925 at 997-1024 on cpu5-9 and 15-19.
    const spark = { '/proc/self/status': 'Name:\tnode\nCpus_allowed_list:\t0-19\n' };
    for (let cpu = 0; cpu < 20; cpu += 1) {
        const big = (cpu >= 5 && cpu <= 9) || cpu >= 15;
        spark[`/sys/devices/system/cpu/cpu${cpu}/cpu_capacity`] = `${big ? 997 + (cpu % 5) * 6 : 718 + (cpu % 5) * 3}\n`;
    }
    assert.equal(performanceCoreCount({ fsApi: fakeFs(spark) }), 10);
    // An affinity to part of the machine counts only the allowed CPUs.
    assert.equal(performanceCoreCount({ fsApi: fakeFs({ ...spark, '/proc/self/status': 'Cpus_allowed_list:\t0-7\n' }) }), 3);
    // A cgroup CPU quota of 4 CPUs caps it.
    assert.equal(performanceCoreCount({ fsApi: fakeFs({ ...spark, '/sys/fs/cgroup/cpu.max': '400000 100000\n' }) }), 4);
    assert.equal(performanceCoreCount({ fsApi: fakeFs({ ...spark, '/sys/fs/cgroup/cpu.max': 'max 100000\n' }) }), 10);
    const uniform = { '/proc/self/status': 'Cpus_allowed_list:\t0-3\n' };
    for (let cpu = 0; cpu < 4; cpu += 1) uniform[`/sys/devices/system/cpu/cpu${cpu}/cpu_capacity`] = '1024\n';
    assert.equal(performanceCoreCount({ fsApi: fakeFs(uniform) }), null);
    // No cpu_capacity (most x86 kernels): null, so the unified default falls back to today's rule.
    assert.equal(performanceCoreCount({ fsApi: fakeFs({ '/proc/self/status': 'Cpus_allowed_list:\t0-3\n' }) }), null);
    // Today's rule is unchanged.
    assert.equal(defaultThreads(20), 18);
    assert.equal(typeof physicalCoreCount, 'function');
});

test('memory: MemFree and Cached are read for display; PSI full avg10 is read when the kernel has it', () => {
    const meminfo = 'MemTotal:       125442396 kB\nMemFree:         2369088 kB\nMemAvailable:   116370624 kB\nCached:         109211296 kB\nSwapFree:              0 kB\n';
    assert.deepEqual(readMemory({ fsApi: fakeFs({ '/proc/meminfo': meminfo }) }), {
        totalBytes: 125442396 * 1024, availableBytes: 116370624 * 1024, swapFreeBytes: 0,
        freeBytes: 2369088 * 1024, cachedBytes: 109211296 * 1024,
    });
    const psi = 'some avg10=16.20 avg60=4.10 avg300=1.00 total=17785584\nfull avg10=14.23 avg60=3.02 avg300=0.80 total=17759002\n';
    assert.equal(readMemoryPressure({ fsApi: fakeFs({ '/proc/pressure/memory': psi }) }), 14.23);
    assert.equal(readMemoryPressure({ fsApi: fakeFs({}) }), null);
});

test('until a runner has a unified-memory policy, a unified GPU refuses it by name, never with a figure of 0', async () => {
    const { gpu } = await read({ [GPU_QUERY[0]]: GB10_LINE, [DEVICE_QUERY[0]]: GB10_DEVICE, [APPS_QUERY[0]]: '' });
    const snapshot = { gpu, memory: { totalBytes: 125442396 * 1024, availableBytes: 106 * GIB }, disk: { freeBytes: 132 * 1000 ** 3 } };
    // vLLM has an experimental unified policy behind the operator's switch
    // (tests/runtime-models-unified.test.mjs); these have none.
    for (const runnerId of ['ik_llama.cpp', 'ollama']) {
        const runner = getRunner(runnerId);
        const source = GPT.sources[runner.weightFormat];
        const decision = admit({ runner, model: GPT, source, params: runner.normalizeParams({}, { model: GPT }), snapshot });
        assert.equal(decision.status, 'incompatible', runnerId);
        assert.equal(decision.reason, `${runner.displayName} is not available on a GPU that shares system memory (NVIDIA GB10) in this release.`);
    }
});

// A-LL-11b (the local-llm half; ploinky pins its own table, A-P-19): the GB10 tuple the GPU-aware budget stop is qualified on.
test('A-LL-11b: the GPU-aware budget stop is qualified for exactly PCI 0x2E1210DE with driver 580.178.04', () => {
    const gpu = (pciDeviceId, driverVersion) => ({ driverVersion, device: { pciDeviceId } });
    assert.deepEqual(hardware.UNIFIED_QUALIFIED_TUPLES.map(({ pciDeviceId, driverVersion }) => [pciDeviceId, driverVersion]),
        [['0x2E1210DE', '580.178.04']]);
    assert.equal(hardware.unifiedTupleQualified(gpu('0x2E1210DE', '580.178.04')), true);
    // The driver is compared after trim; nothing else is normalised.
    assert.equal(hardware.unifiedTupleQualified(gpu('0x2E1210DE', ' 580.178.04\n')), true);
    for (const rejected of [
        gpu('0x2E1210DE', '580.159.03'), gpu('0x2E1210DE', '580.95.05'), gpu('0x2E1210DE', '580.178.040'), gpu('0x2E1210DE', '580.178'),
        gpu('0x2E1210DE', '580'), gpu('0x2E1210DE', undefined), gpu('0x2E1210DE', 580.178), gpu('0x2e1210de', '580.178.04'),
        gpu('0x2B0010DE', '580.178.04'), gpu(null, '580.178.04'), { driverVersion: '580.178.04' }, {}, null, undefined,
    ]) assert.equal(hardware.unifiedTupleQualified(rejected), false, JSON.stringify(rejected));
});

// A-LL-16h and the readGpuProcesses contract (r6 n3): the shared query drops bad rows only in readGpuProcesses.
test('readGpuProcesses: one bounded spawn, rows with a plain PID and memory, never throws', async () => {
    const calls = [];
    const host = (answer) => (command, args, options, callback) => {
        calls.push({ command, args, timeout: options.timeout });
        if (answer instanceof Error) callback(answer, '', 'boom');
        else callback(null, answer, '');
    };
    const rowsOf = (answer) => hardware.readGpuProcesses({ execFileImpl: host(answer), nvidiaSmi: SMI });
    const good = await rowsOf('59, /opt/llama.cpp/llama-server, 11875\n61, python, 512\n');
    assert.equal(good.ok, true);
    assert.deepEqual(good.rows, [
        { pid: 59, name: '/opt/llama.cpp/llama-server', usedBytes: 11875 * MIB },
        { pid: 61, name: 'python', usedBytes: 512 * MIB },
    ]);
    assert.ok(Number.isFinite(good.execMs) && good.execMs >= 0);
    assert.deepEqual(calls, [{ command: SMI, args: APPS_QUERY, timeout: 1500 }], 'one spawn, bounded at 1500 ms');
    const garbage = await rowsOf('garbage\nx, y, 5\n7, p, [N/A]\n8, p, \n, p, 5\n9, p, 1.5\n12, p, 4\n');
    assert.equal(garbage.ok, true);
    assert.deepEqual(garbage.rows, [{ pid: 12, name: 'p', usedBytes: 4 * MIB }], 'a non-numeric PID or memory drops that row');
    const none = await rowsOf('');
    assert.equal(none.ok, true);
    assert.deepEqual(none.rows, []);
    const failed = await rowsOf(Object.assign(new Error('exit 1'), { code: 1 }));
    assert.equal(failed.ok, false);
    assert.match(failed.reason, /^nvidia-smi failed: boom/);
    // A stop is a result, never a throw.
    const controller = new AbortController();
    controller.abort();
    const aborted = await hardware.readGpuProcesses({ execFileImpl: host('1, p, 1'), nvidiaSmi: SMI, signal: controller.signal });
    assert.equal(aborted.ok, false);
    assert.equal(aborted.reason, 'aborted');
});

// A-LL-16h: readGpuProcesses is built on a shared query, and readGpu's own output for a garbage row must not change.
test('A-LL-16h: readGpu keeps NaN and an undefined name for a garbage compute-apps row, as before', async () => {
    const { gpu } = await read({ [GPU_QUERY[0]]: GB10_LINE, [DEVICE_QUERY[0]]: GB10_DEVICE, [APPS_QUERY[0]]: 'garbage\n7, p, [N/A]\n' });
    assert.equal(gpu.processes.length, 2);
    assert.ok(Number.isNaN(gpu.processes[0].pid) && gpu.processes[0].name === undefined && Number.isNaN(gpu.processes[0].usedBytes));
    assert.equal(gpu.processes[1].pid, 7);
    assert.ok(Number.isNaN(gpu.processes[1].usedBytes));
});

// A-LL-16f: the command name is parenthesised and may hold ") ", so the fields are read after the last ")".
test('readProcessGroup parses pgrp after the last ")" of /proc/<pid>/stat, and null for anything else', () => {
    const fsApi = (files) => ({ readFileSync(file) {
        if (!Object.hasOwn(files, file)) throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' });
        return files[file];
    } });
    const api = fsApi({
        '/proc/7000/stat': '7000 (llama-server) S 1 7000 7000 0 -1 4194560 1 0 0\n',
        '/proc/7001/stat': '7001 (ev) S ) 1) S 1 7001 7001 0 -1 4194560\n',
        '/proc/7002/stat': '7002 (child) R 7000 7000 7000 0 -1\n',
        '/proc/7003/stat': 'no parentheses here\n',
        '/proc/7004/stat': '7004 (short) S 1\n',
    });
    assert.equal(hardware.readProcessGroup(7000, { fsApi: api }), 7000);
    assert.equal(hardware.readProcessGroup(7001, { fsApi: api }), 7001, 'the correct pgrp, not a field of the name');
    assert.equal(hardware.readProcessGroup(7002, { fsApi: api }), 7000);
    for (const pid of [7003, 7004, 7999, 0, -1, 1.5, NaN, undefined]) assert.equal(hardware.readProcessGroup(pid, { fsApi: api }), null, String(pid));
});

test('parsePageCacheBytes: active_file + inactive_file only; file and shmem are ignored; either missing is unreadable', () => {
    const stat = (extra) => `anon 100\nfile 5368709120\nshmem 1073741824\n${extra}`;
    assert.equal(hardware.parsePageCacheBytes(stat('active_file 1610612736\ninactive_file 536870912\n')), 2 * 1024 * MIB);
    assert.equal(hardware.parsePageCacheBytes(stat('active_file 0\ninactive_file 0\n')), 0);
    for (const bad of [stat('active_file 5\n'), stat('inactive_file 5\n'), stat(''), stat('active_file x\ninactive_file 1\n'),
        'xactive_file 1\ninactive_file 1\n', null, undefined, 12]) assert.equal(hardware.parsePageCacheBytes(bad), null, String(bad));
});

// rn1: the budget read that gives the limit also gives memory.current, without changing the observation's published shape.
test('observeMemoryBudget carries the same read\'s memory.current as a non-enumerable currentBytes', () => {
    const known = observeMemoryBudget({ maxText: String(12 * GIB), currentText: String(5 * GIB) });
    assert.equal(known.currentBytes, 5 * GIB);
    assert.deepEqual(Object.keys(known), ['memoryReadState', 'finiteMemoryBytes', 'headroomBytes', 'reasonCode']);
    assert.deepEqual(known, { memoryReadState: 'known', finiteMemoryBytes: 12 * GIB, headroomBytes: 7 * GIB, reasonCode: null });
    assert.ok(Object.isFrozen(known));
    assert.equal(Object.getOwnPropertyDescriptor(known, 'currentBytes').enumerable, false);
    assert.equal(JSON.stringify(known).includes('currentBytes'), false);
    // An unlimited or unreadable budget has none.
    for (const unlimited of [observeMemoryBudget({ maxText: 'max', currentText: '1' }), observeMemoryBudget({ maxText: String(GIB), currentText: 'x' })]) {
        assert.equal(unlimited.currentBytes, undefined);
    }
});
