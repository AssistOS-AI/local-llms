// Default CPU threads for the llama-server runners (runners plan, I2):
// physical cores minus 2, counted over the CPUs this process may use, with
// SMT siblings counted once. An admin-set value still wins.
import assert from 'node:assert/strict';
import test from 'node:test';

import { defaultThreads, physicalCoreCount } from '../src/controller/hardware.mjs';
import { ikLlamaCppRunner } from '../src/runners/ikLlamaCpp.mjs';
import { llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { createLlamaServerRunner } from '../src/runners/llamaServer.mjs';

function fakeFs(files) {
    return {
        readFileSync(file) {
            if (Object.hasOwn(files, file)) return files[file];
            throw Object.assign(new Error(`ENOENT: ${file}`), { code: 'ENOENT' });
        },
    };
}

function topology(cores) {
    const files = {};
    cores.forEach(([pkg, core], cpu) => {
        files[`/sys/devices/system/cpu/cpu${cpu}/topology/physical_package_id`] = `${pkg}\n`;
        files[`/sys/devices/system/cpu/cpu${cpu}/topology/core_id`] = `${core}\n`;
    });
    return files;
}

// This machine's shape: 6 performance cores with two threads each, then 8
// efficiency cores with one: 20 logical CPUs, 14 physical cores.
const HYBRID = [
    ...Array.from({ length: 12 }, (_, cpu) => [0, Math.floor(cpu / 2) * 4]),
    ...Array.from({ length: 8 }, (_, index) => [0, 24 + index]),
];

test('physical cores count SMT siblings once, over the allowed CPUs only', () => {
    const files = { '/proc/self/status': 'Name:\tnode\nCpus_allowed_list:\t0-19\n', ...topology(HYBRID) };
    assert.equal(physicalCoreCount({ fsApi: fakeFs(files), availableParallelism: () => 20 }), 14);
    // A cpuset of four logical CPUs on two performance cores and one efficiency core.
    const pinned = { ...files, '/proc/self/status': 'Cpus_allowed_list:\t0-1,4-5,12\n' };
    assert.equal(physicalCoreCount({ fsApi: fakeFs(pinned), availableParallelism: () => 5 }), 3);
    // Two sockets whose core ids repeat are still distinct cores.
    const sockets = { '/proc/self/status': 'Cpus_allowed_list:\t0-3\n', ...topology([[0, 0], [0, 1], [1, 0], [1, 1]]) };
    assert.equal(physicalCoreCount({ fsApi: fakeFs(sockets), availableParallelism: () => 4 }), 4);
});

test('a cgroup CPU quota caps the count; without sysfs, /proc/cpuinfo, then the logical count', () => {
    const files = { '/proc/self/status': 'Cpus_allowed_list:\t0-19\n', ...topology(HYBRID), '/sys/fs/cgroup/cpu.max': '400000 100000\n' };
    assert.equal(physicalCoreCount({ fsApi: fakeFs(files), availableParallelism: () => 20 }), 4);
    assert.equal(physicalCoreCount({ fsApi: fakeFs({ ...files, '/sys/fs/cgroup/cpu.max': 'max 100000\n' }), availableParallelism: () => 20 }), 14);
    const cpuinfo = [0, 1, 2, 3].map((cpu) => `processor\t: ${cpu}\nphysical id\t: 0\ncore id\t\t: ${Math.floor(cpu / 2)}\n`).join('\n');
    assert.equal(physicalCoreCount({ fsApi: fakeFs({ '/proc/self/status': 'Cpus_allowed_list:\t0-3\n', '/proc/cpuinfo': cpuinfo }), availableParallelism: () => 4 }), 2);
    assert.equal(physicalCoreCount({ fsApi: fakeFs({}), availableParallelism: () => 6 }), 6);
});

test('the default is physical cores minus 2, never below 1', () => {
    assert.equal(defaultThreads(14), 12);
    assert.equal(defaultThreads(3), 1);
    assert.equal(defaultThreads(1), 1);
});

test('llama.cpp and ik_llama.cpp pass the default threads unless the admin set a value', () => {
    const dialect = { quietArgs: [], unifiedKv: true, loadArgs: () => [], jinja: () => false, parseVersion: () => null };
    const runner = createLlamaServerRunner({ id: 'llama.cpp', displayName: 'T', executable: '/opt/t/llama-server', pinnedVersion: 'b1', port: 18999, dialect, cpuCores: () => 14 });
    const launch = (params) => runner.buildLaunch({ artifactPath: '/data/models/m.gguf', params, port: 18999, apiKey: 'k'.repeat(43), model: { id: 'm' } }).args;
    const threadsOf = (args) => args[args.indexOf('--threads') + 1];
    assert.equal(threadsOf(launch({})), '12');
    assert.equal(threadsOf(launch({ threads: null })), '12');
    assert.equal(threadsOf(launch({ threads: 8 })), '8');
    // The stored parameters keep "empty": the default follows the machine the agent runs on.
    assert.equal(runner.normalizeParams({}).threads, null);
    // The real adapters use this machine's physical cores.
    const expected = String(defaultThreads(physicalCoreCount()));
    for (const real of [llamaCppRunner, ikLlamaCppRunner]) {
        const args = real.buildLaunch({ artifactPath: '/data/models/m.gguf', params: {}, port: real.port, apiKey: 'k'.repeat(43), model: { id: 'm' } }).args;
        assert.equal(threadsOf(args), expected, real.id);
    }
});
