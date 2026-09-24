import { codedError } from './params.mjs';
import { ikLlamaCppRunner } from './ikLlamaCpp.mjs';
import { llamaCppRunner } from './llamaCpp.mjs';
import { ollamaRunner } from './ollama.mjs';
import { tabbyApiRunner } from './tabbyApi.mjs';
import { vllmRunner } from './vllm.mjs';

// Every runner is an adapter (llamaServer.mjs shows the full shape): identity,
// the weight format it reads, its loopback port and per-start key, its
// parameter schema and form metadata, detection, the start-up pipeline, the
// chat model name, its admission policy and its log-report parser. The
// controller has no per-runner branches; adding a runner adds an adapter here.
// LM Studio is deliberately absent (runners plan, R5 = b; DS000).
export const RUNNERS = Object.freeze({
    'llama.cpp': llamaCppRunner,
    'ik_llama.cpp': ikLlamaCppRunner,
    ollama: ollamaRunner,
    vllm: vllmRunner,
    tabbyapi: tabbyApiRunner
});

export function getRunner(id) {
    if (typeof id !== 'string' || !Object.hasOwn(RUNNERS, id)) {
        throw codedError('unknown_runner', `Unknown runner: ${String(id)}`, { runner: id });
    }
    return RUNNERS[id];
}

/** The loopback port of every supported runner; two runners never share one. */
export function defaultPorts(runners = RUNNERS) {
    const ports = {};
    const owners = new Map();
    for (const runner of Object.values(runners)) {
        if (!runner.supported) continue;
        if (!Number.isInteger(runner.port)) throw new Error(`Runner ${runner.id} has no port`);
        if (owners.has(runner.port)) {
            throw new Error(`Runners ${owners.get(runner.port)} and ${runner.id} both use port ${runner.port}`);
        }
        owners.set(runner.port, runner.id);
        ports[runner.id] = runner.port;
    }
    return Object.freeze(ports);
}

/** What the overview and the dashboard learn about each runner. */
export function runnerSummary(runner) {
    const { id, displayName, weightFormat, pinnedVersion, supported, paramSchema } = runner;
    return {
        id, displayName, weightFormat, pinnedVersion, supported, paramSchema,
        basicParams: runner.basicParams || [],
        moeParams: runner.moeParams || [],
    };
}

export function runnerSummaries(runners = RUNNERS) {
    return Object.values(runners).map(runnerSummary);
}
