// The dedicated and unified profiles stay what they were (Phase 3): the runners and models their overview
// shows are exactly those of a84617d, the head before llama.cpp's CPU build and the agent's runner lock,
// even with the agent's real locks loaded; and on a dedicated host, adding an Ollama tag and running it (or
// being refused) are what they were before tags were pinned, with no registry read. Phase 3 lists no change
// for either profile, so nothing is allowed to differ. The golden file was made from that commit (see
// overview-scenarios.mjs).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { createRunnerInstaller } from '../src/controller/runnerInstaller.mjs';
import { agentRunnerLockFile, loadRunnerLocks } from '../src/controller/runnerLock.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { OLLAMA_RUN_SCENARIOS, SCENARIOS, collectOllamaRuns, collectOverviews } from './overview-scenarios.mjs';

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

const agentLocks = (arch) => ({ imageLockFile, dataDir }) => createRunnerInstaller({
    lock: loadRunnerLocks({ image: imageLockFile, agent: agentRunnerLockFile(arch) }), cacheRoot: path.join(dataDir, 'runners'), runRoot: path.join(dataDir, 'opt'),
});

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

for (const arch of ['arm64', 'x64']) {
    test(`on a dedicated host, adding and running an Ollama tag is what it was at a84617d, with no registry read, with the ${arch} agent lock loaded`, async () => {
        const asked = [];
        const actual = await collectOllamaRuns({
            createController, createStateStore, validateModel, makeInstaller: agentLocks(arch),
            resolveOllama: async (tag) => { asked.push(tag); throw new Error('a dedicated host must not read the Ollama registry'); },
        });
        assert.deepEqual(asked, [], 'the registry was never asked');
        assert.deepEqual(Object.keys(actual), Object.keys(OLLAMA_RUN_SCENARIOS));
        for (const name of Object.keys(golden.ollamaRuns)) {
            assert.equal(firstDifference(actual[name], golden.ollamaRuns[name], name), null, name);
        }
        // In words: the tag is stored as typed, with no digest and no size; Run launches the image's Ollama with the driver path
        // and asks for no GPU option; the image that lacks Ollama refuses it as it did.
        const inImage = actual['dedicated, amd64 image with Ollama in it'];
        assert.deepEqual(inImage.added.ollama, { type: 'ollama', tag: 'gpt-oss:20b' });
        assert.deepEqual([inImage.phase, inImage.profile, inImage.process.command], ['ready', 'dedicated', '/opt/ollama/bin/ollama']);
        assert.equal(inImage.process.env.LD_LIBRARY_PATH, '/usr/local/nvidia/lib64');
        assert.equal(Object.hasOwn(inImage.process.env, 'CUDA_VISIBLE_DEVICES'), false);
        assert.deepEqual(inImage.calls.find(([url]) => url.endsWith('/api/generate'))[2].options, { num_ctx: 4096 });
        assert.equal(actual['dedicated, arm64 image without Ollama'].refused.code, 'runner_unavailable');
    });
}
