// The finite memory limit this process has seen (hardware.mjs at 34f8f498, `establishedMemoryLimits`): a later failure to read
// memory.max never turns it into unlimited. At base one process-wide map, keyed by the file system reader, served the snapshot
// reader and the memory guard alike. Now a limits value holds it, and its snapshot hooks and its guard sample share it.
import assert from 'node:assert/strict';
import test from 'node:test';

import { readSnapshot } from 'pworker/local-llm';

import { memoryBudgetOf } from '../src/controller/ploinkyBudget.mjs';
import { AGENT_LIMITS, createAgentLimits, readCgroupMemoryObservation, readSnapshot as readAgentSnapshot } from '../src/hardwareLimits.mjs';

const GIB = 1024 ** 3;
const MAX = '/sys/fs/cgroup/memory.max';
const CURRENT = '/sys/fs/cgroup/memory.current';
const enoent = (file) => Object.assign(new Error(`ENOENT: ${file}`), { code: 'ENOENT' });

function world(overrides = {}) {
    const files = {
        '/proc/meminfo': 'MemTotal: 33554432 kB\nMemAvailable: 25165824 kB\nSwapFree: 0 kB\nMemFree: 1 kB\nCached: 1 kB\n',
        '/proc/self/status': 'Cpus_allowed_list:\t0-3\n', [MAX]: `${2 * GIB}\n`, [CURRENT]: `${GIB}\n`, ...overrides,
    };
    const fsApi = {
        readFileSync(file) {
            if (!Object.hasOwn(files, file)) throw enoent(file);
            if (files[file] instanceof Error) throw files[file];
            return files[file];
        },
        readdirSync(dir) { throw enoent(dir); },
    };
    return { files, fsApi };
}

const execFileImpl = (command, args, options, callback) => callback(Object.assign(new Error('absent'), { code: 'ENOENT' }), '', '');
const snapshotOf = (fsApi, limits, extra = {}) => readSnapshot({ dataDir: '/tmp', fsApi, execFileImpl, env: {}, limits, statfs: async () => ({ bsize: 4096, blocks: 1000, bavail: 900 }), ...extra });

test('a snapshot establishes the limit, and a later snapshot and the guard\'s sample both keep it when memory.max disappears', async () => {
    const { files, fsApi } = world();
    const limits = createAgentLimits({ fsApi });
    const first = await snapshotOf(fsApi, limits);
    assert.deepEqual(first.cgroupMemory, { maxBytes: 2 * GIB, currentBytes: GIB });
    assert.equal(memoryBudgetOf(first).memoryReadState, 'known');
    // The key order the snapshot always had (hardware.mjs:505-513 at 34f8f498), and no budget fact in what is serialized.
    assert.deepEqual(Object.keys(first), ['at', 'gpu', 'memory', 'disk', 'cpus', 'cores', 'cgroupMemory']);
    assert.equal(JSON.stringify(first).includes('memoryBudget'), false);
    assert.equal(JSON.stringify(first).includes('gpuBudget'), false);
    // The file is gone. Absent would mean "no limit" for a process that never saw one; this one has seen 2 GiB.
    delete files[MAX];
    const second = await snapshotOf(fsApi, limits);
    assert.equal(second.cgroupMemory, null, 'the raw projection reads what is there');
    const budget = memoryBudgetOf(second);
    assert.deepEqual([budget.memoryReadState, budget.finiteMemoryBytes, budget.reasonCode], ['unknown', 2 * GIB, 'budget_unreadable']);
    // The guard samples the same state: its sample carries the established limit, and it is watched.
    const sample = limits.guardSample();
    assert.deepEqual([sample.memoryReadState, sample.finiteMemoryBytes, sample.watched], ['unknown', 2 * GIB, true]);
});

test('the guard\'s sample establishes the limit for the snapshot, as it did through the one map', async () => {
    const { files, fsApi } = world();
    const limits = createAgentLimits({ fsApi });
    assert.equal(limits.guardSample().finiteMemoryBytes, 2 * GIB);
    delete files[MAX];
    const budget = memoryBudgetOf(await snapshotOf(fsApi, limits));
    assert.deepEqual([budget.memoryReadState, budget.finiteMemoryBytes], ['unknown', 2 * GIB]);
});

test('a limit established before /proc/meminfo failed is still established: the observation is made before the memory read', async () => {
    const { files, fsApi } = world();
    const limits = createAgentLimits({ fsApi });
    files['/proc/meminfo'] = Object.assign(new Error('unreadable'), { code: 'EIO' });
    await assert.rejects(snapshotOf(fsApi, limits), (error) => error.code === 'EIO');
    delete files[MAX];
    assert.deepEqual([limits.guardSample().memoryReadState, limits.guardSample().finiteMemoryBytes], ['unknown', 2 * GIB]);
});

test('without an established limit, an absent memory.max is no limit; two limits values do not share what they saw', async () => {
    const { files, fsApi } = world();
    const seeing = createAgentLimits({ fsApi });
    const other = createAgentLimits({ fsApi });
    await snapshotOf(fsApi, seeing);
    delete files[MAX];
    assert.equal(seeing.guardSample().finiteMemoryBytes, 2 * GIB);
    assert.deepEqual([other.guardSample().memoryReadState, other.guardSample().finiteMemoryBytes, other.guardSample().watched], ['known', null, false]);
});

test('the process-wide readers share one map, keyed by the reader: a snapshot through them and a direct observation agree', async () => {
    const { files, fsApi } = world();
    await readAgentSnapshot({ dataDir: '/tmp', fsApi, execFileImpl, env: {}, statfs: async () => ({ bsize: 4096, blocks: 1000, bavail: 900 }) });
    delete files[MAX];
    const observation = readCgroupMemoryObservation({ fsApi });
    assert.deepEqual([observation.memoryReadState, observation.finiteMemoryBytes], ['unknown', 2 * GIB]);
    // Another reader object has seen nothing.
    const stranger = world({});
    delete stranger.files[MAX];
    assert.deepEqual([readCgroupMemoryObservation({ fsApi: stranger.fsApi }).memoryReadState, readCgroupMemoryObservation({ fsApi: stranger.fsApi }).finiteMemoryBytes], ['known', null]);
    assert.equal(typeof AGENT_LIMITS.guardSample, 'function');
});

test('under a finite limit a shared-memory GPU is replaced in place: the key keeps its second position', async () => {
    const { fsApi } = world();
    const limits = createAgentLimits({ fsApi });
    const snapshot = { at: 'now', gpu: { available: true, memoryModel: 'shared', name: 'APU' }, memory: {}, disk: {}, cpus: 4, cores: 4 };
    const decorated = limits.decorateSnapshot({ snapshot, fsApi, env: {}, observation: limits.observeSnapshot({ fsApi }) });
    assert.deepEqual(Object.keys(decorated), ['at', 'gpu', 'memory', 'disk', 'cpus', 'cores', 'cgroupMemory']);
    assert.deepEqual([decorated.gpu.available, decorated.gpu.state, decorated.gpu.backend], [false, 'unusable', 'vulkan']);
});
