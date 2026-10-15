// Default CPU threads for the llama-server runners (runners plan, I2):
// physical cores minus 2, counted over the CPUs this process may use, with
// SMT siblings counted once. An admin-set value still wins.
import assert from 'node:assert/strict';
import test from 'node:test';

import { physicalCoreCount } from '../src/hardwareLimits.mjs';

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

test('a cgroup CPU quota caps the count; without sysfs, /proc/cpuinfo, then the logical count', () => {
    const files = { '/proc/self/status': 'Cpus_allowed_list:\t0-19\n', ...topology(HYBRID), '/sys/fs/cgroup/cpu.max': '400000 100000\n' };
    assert.equal(physicalCoreCount({ fsApi: fakeFs(files), availableParallelism: () => 20 }), 4);
    assert.equal(physicalCoreCount({ fsApi: fakeFs({ ...files, '/sys/fs/cgroup/cpu.max': 'max 100000\n' }), availableParallelism: () => 20 }), 14);
    const cpuinfo = [0, 1, 2, 3].map((cpu) => `processor\t: ${cpu}\nphysical id\t: 0\ncore id\t\t: ${Math.floor(cpu / 2)}\n`).join('\n');
    assert.equal(physicalCoreCount({ fsApi: fakeFs({ '/proc/self/status': 'Cpus_allowed_list:\t0-3\n', '/proc/cpuinfo': cpuinfo }), availableParallelism: () => 4 }), 2);
    assert.equal(physicalCoreCount({ fsApi: fakeFs({}), availableParallelism: () => 6 }), 6);
});

