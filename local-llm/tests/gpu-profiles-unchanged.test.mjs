// The dedicated and unified profiles stay what they were (Phase 3): the runners and models their overview
// shows are exactly those of a84617d, the head before llama.cpp's CPU build and the agent's runner lock,
// even with the agent's real locks loaded. Phase 3 lists no change for either profile, so nothing is allowed
// to differ. The golden file was made from that commit (see overview-scenarios.mjs).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { agentRunnerLockFile, loadRunnerLocks } from '../src/controller/runnerLock.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { SCENARIOS, collectOverviews } from './overview-scenarios.mjs';

const golden = JSON.parse(fs.readFileSync(new URL('./overview-golden-a84617d.json', import.meta.url), 'utf8'));

// The first place two JSON values differ, as a path, so a failure says where.
function firstDifference(actual, expected, where = '') {
    if (Object.is(actual, expected)) return null;
    const both = (value) => value !== null && typeof value === 'object';
    if (!both(actual) || !both(expected) || Array.isArray(actual) !== Array.isArray(expected)) {
        return `${where || '(root)'}: ${JSON.stringify(actual)?.slice(0, 120)} is not ${JSON.stringify(expected)?.slice(0, 120)}`;
    }
    for (const key of new Set([...Object.keys(actual), ...Object.keys(expected)])) {
        if (!Object.hasOwn(actual, key)) return `${where}.${key}: missing, expected ${JSON.stringify(expected[key])?.slice(0, 120)}`;
        if (!Object.hasOwn(expected, key)) return `${where}.${key}: ${JSON.stringify(actual[key])?.slice(0, 120)} was not there before`;
        const found = firstDifference(actual[key], expected[key], `${where}.${key}`);
        if (found) return found;
    }
    return null;
}

for (const arch of ['arm64', 'x64']) {
    test(`the dedicated and unified overviews are what they were at a84617d, with the ${arch} agent lock loaded`, async () => {
        let loaded = null;
        const actual = await collectOverviews({
            createController, createStateStore, validateModel,
            makeInstaller: ({ imageLockFile, dataDir }) => {
                const lock = loadRunnerLocks({ image: imageLockFile, agent: agentRunnerLockFile(arch) });
                loaded = lock;
                return createRunnerInstaller({ lock, cacheRoot: path.join(dataDir, 'runners'), runRoot: path.join(dataDir, 'opt') });
            },
        });
        // The real agent lock is in play: both the CPU build and, on arm64, Ollama are in it, for the installer to offer.
        assert.equal(loaded.origin['llama.cpp-cpu'], 'agent');
        assert.equal(loaded.origin.ollama, arch === 'arm64' ? 'agent' : undefined);
        assert.deepEqual(Object.keys(actual), Object.keys(SCENARIOS));
        for (const name of Object.keys(golden.scenarios)) {
            assert.equal(firstDifference(actual[name], golden.scenarios[name], name), null, name);
        }
        // What it means, in words: six runners, in this order, and none of them the CPU build, anywhere in the overview.
        for (const [name, overview] of Object.entries(actual)) {
            assert.deepEqual(overview.runners.map((runner) => runner.id), ['llama.cpp', 'ik_llama.cpp', 'ollama', 'vllm', 'tabbyapi', 'lmstudio'], name);
            for (const model of overview.models) {
                assert.equal(Object.hasOwn(model.runners, 'llama.cpp-cpu'), false, `${name} ${model.id}`);
                for (const weights of Object.values(model.weights)) assert.equal(weights.runners.includes('llama.cpp-cpu'), false, `${name} ${model.id}`);
            }
        }
    });
}
