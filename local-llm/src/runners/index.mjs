import { codedError } from './params.mjs';
import { llamaCppRunner } from './llamaCpp.mjs';
import { lmStudioRunner } from './lmStudio.mjs';
import { ollamaRunner } from './ollama.mjs';
import { vllmRunner } from './vllm.mjs';

export const RUNNERS = Object.freeze({
    'llama.cpp': llamaCppRunner,
    ollama: ollamaRunner,
    vllm: vllmRunner,
    lmstudio: lmStudioRunner
});

export function getRunner(id) {
    if (typeof id !== 'string' || !Object.hasOwn(RUNNERS, id)) {
        throw codedError('unknown_runner', `Unknown runner: ${String(id)}`, { runner: id });
    }
    return RUNNERS[id];
}

export function runnerSummaries() {
    return Object.values(RUNNERS).map(({ id, displayName, weightFormat, pinnedVersion, supported, paramSchema }) => ({
        id, displayName, weightFormat, pinnedVersion, supported, paramSchema
    }));
}
