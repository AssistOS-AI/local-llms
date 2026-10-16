import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { createAgentController as createController, loadSeedCatalog, readGpu, vulkanContractOf } from '../src/engineOptions.mjs';
import { readSnapshot } from '../src/hardwareLimits.mjs';
import { collectCpuOverview, cpuFs, VULKAN_KEYS } from './overview-cpu-scenarios.mjs';

const golden = JSON.parse(fs.readFileSync(new URL('./overview-cpu-golden-b2b387f.json', import.meta.url), 'utf8'));
for (const arch of ['x64', 'arm64']) for (const vulkanKeys of [false, true]) {
    const name = `${arch}, base contract${vulkanKeys ? ' plus Vulkan keys' : ''}`;
    test(`CPU host unchanged: ${name}`, async () => {
        assert.deepEqual(await collectCpuOverview({ createController, readSnapshot, loadSeedCatalog, arch, vulkanKeys }), golden.scenarios[name]);
    });
}

function absentQuery(calls) {
    return (command, args, options, callback) => {
        calls.push(command);
        callback(Object.assign(new Error('absent'), { code: 'ENOENT' }), '', '');
    };
}
test('CPU host unchanged: snapshot runs nvidia-smi only and returns the readGpu object', async () => {
    const calls = [];
    const execFileImpl = absentQuery(calls);
    const snap = await readSnapshot({ dataDir: '/tmp', fsApi: cpuFs(), execFileImpl, env: {},
        vulkan: vulkanContractOf(VULKAN_KEYS), statfs: async () => ({ bsize: 1, blocks: 20, bavail: 10 }) });
    assert.equal(calls.length, 1);
    assert.deepEqual(snap.gpu, await readGpu({ execFileImpl, env: {} }));
});
test('CPU host unchanged: /dev/dri ENOENT takes the absent path', async () => {
    const fsApi = cpuFs();
    const calls = [];
    fsApi.readdirSync = (dir) => { assert.equal(dir, '/dev/dri'); throw Object.assign(new Error('absent'), { code: 'ENOENT' }); };
    const snap = await readSnapshot({ dataDir: '/tmp', fsApi, execFileImpl: absentQuery(calls), env: {},
        vulkan: vulkanContractOf(VULKAN_KEYS), statfs: async () => ({ bsize: 1, blocks: 20, bavail: 10 }) });
    assert.equal(snap.gpu.state, 'absent');
    assert.equal(Object.hasOwn(snap.gpu, 'backend'), false);
    assert.equal(calls.length, 1);
});
