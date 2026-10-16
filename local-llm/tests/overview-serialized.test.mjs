// V5: the overview the agent publishes is the same bytes as before the engine moved, key order included. The goldens are compared
// as serialized JSON, not only as deep-equal values: the hardware limits rebuild the snapshot's keys in the order they always had,
// and a deep-equal comparison would not notice a key that moved.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { createStateStore, loadRunnerLocks, validateModel } from 'pworker/local-llm';

import { agentRunnerLockFile } from '../src/agentHost.mjs';
import { createAgentController, createRunnerInstaller, loadSeedCatalog } from '../src/engineOptions.mjs';
import { readSnapshot } from '../src/hardwareLimits.mjs';
import { collectCpuOverview } from './overview-cpu-scenarios.mjs';
import { collectOverviews } from './overview-scenarios.mjs';

const read = (name) => JSON.parse(fs.readFileSync(new URL(name, import.meta.url), 'utf8'));
const gpuGolden = read('./overview-golden-a84617d.json');
const cpuGolden = read('./overview-cpu-golden-b2b387f.json');

// The first place two serialized values differ, so a failure says where.
const serialized = (value) => JSON.stringify(value);
function firstByteDifference(actual, expected) {
    const a = serialized(actual);
    const b = serialized(expected);
    if (a === b) return null;
    let at = 0;
    while (at < a.length && a[at] === b[at]) at += 1;
    return `byte ${at}: ...${a.slice(Math.max(0, at - 60), at + 60)}... is not ...${b.slice(Math.max(0, at - 60), at + 60)}...`;
}

const makeInstaller = (arch) => ({ imageLockFile, dataDir }) => createRunnerInstaller({
    lock: loadRunnerLocks({ image: imageLockFile, agent: agentRunnerLockFile(arch) }), cacheRoot: path.join(dataDir, 'runners'), runRoot: path.join(dataDir, 'opt'),
});

for (const arch of ['arm64', 'x64']) {
    test(`the dedicated and unified overviews serialize to the golden bytes (${arch} agent lock)`, async () => {
        const actual = await collectOverviews({ createController: createAgentController, createStateStore, validateModel, makeInstaller: makeInstaller(arch) });
        for (const name of Object.keys(gpuGolden.scenarios)) assert.equal(firstByteDifference(actual[name], gpuGolden.scenarios[name]), null, name);
    });
}

for (const arch of ['x64', 'arm64']) for (const vulkanKeys of [false, true]) {
    const name = `${arch}, base contract${vulkanKeys ? ' plus Vulkan keys' : ''}`;
    test(`the CPU overview serializes to the golden bytes: ${name}`, async () => {
        const actual = await collectCpuOverview({ createController: createAgentController, readSnapshot, loadSeedCatalog, arch, vulkanKeys });
        assert.equal(firstByteDifference(actual, cpuGolden.scenarios[name]), null, name);
    });
}

test('the comparison sees a changed gateway model name and one key that moved', async () => {
    const [name] = Object.keys(gpuGolden.scenarios);
    const actual = (await collectOverviews({ createController: createAgentController, createStateStore, validateModel, makeInstaller: makeInstaller('x64') }))[name];
    assert.equal(firstByteDifference(actual, gpuGolden.scenarios[name]), null, 'the control: the unchanged overview matches');
    assert.match(String(firstByteDifference({ ...actual, gatewayModel: 'soul_gateway/local-llms/local-llm/other' }, gpuGolden.scenarios[name])), /^byte \d+:/);
    const { profile, ...rest } = actual;
    assert.deepEqual({ ...rest, profile }, actual, 'deep-equal does not see the moved key');
    assert.match(String(firstByteDifference({ ...rest, profile }, gpuGolden.scenarios[name])), /^byte \d+:/);
});
