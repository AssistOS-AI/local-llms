import { codedError } from './params.mjs';
import { ikLlamaCppRunner } from './ikLlamaCpp.mjs';
import { llamaCppRunner } from './llamaCpp.mjs';
import { lmStudioRunner } from './lmStudio.mjs';
import { ollamaRunner } from './ollama.mjs';
import { tabbyApiRunner } from './tabbyApi.mjs';
import { vllmRunner } from './vllm.mjs';

// Every runner is an adapter (llamaServer.mjs shows the full shape): identity,
// the weight format it reads, its loopback port and per-start key, its
// parameter schema and form metadata, detection, the start-up pipeline, the
// chat model name, its admission policy and its log-report parser. The
// controller has no per-runner branches; adding a runner adds an adapter here.
// LM Studio is for internal use only, behind an operator switch (runners plan
// I9, Phase R7; DS000).
export const RUNNERS = Object.freeze({
    'llama.cpp': llamaCppRunner,
    'ik_llama.cpp': ikLlamaCppRunner,
    ollama: ollamaRunner,
    vllm: vllmRunner,
    tabbyapi: tabbyApiRunner,
    lmstudio: lmStudioRunner
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

// Why a runner without parameters for a decided profile cannot run there (DS005).
const PROFILE_REFUSALS = Object.freeze({
    dedicated: (name) => `${name} has no parameters for the dedicated profile in this release.`,
    unified: (name) => `${name} is not available on a GPU that shares system memory in this release.`,
    cpu: (name) => `${name} needs an NVIDIA GPU in this release; on this machine models run on the CPU with the runners listed in the Runners tab.`,
});

/**
 * What the overview and the dashboard learn about each runner, with its
 * parameter schema for the hardware profile (null where it has no policy for
 * that profile, DS005). `profileUnsupportedReason` is added only when a decided
 * profile has no schema for the runner; an undecided profile (null) adds none,
 * so every runner keeps its place until the profile is known.
 */
export function runnerSummary(runner, profile = 'dedicated') {
    const { id, displayName, weightFormat, pinnedVersion, supported } = runner;
    const paramSchema = typeof runner.paramSchemaFor === 'function'
        ? runner.paramSchemaFor(profile)
        : (profile === 'dedicated' ? runner.paramSchema : null);
    return {
        id, displayName, weightFormat, pinnedVersion, supported, paramSchema,
        basicParams: runner.basicParams || [],
        moeParams: runner.moeParams || [],
        ...(profile && paramSchema === null && Object.hasOwn(PROFILE_REFUSALS, profile)
            ? { profileUnsupportedReason: PROFILE_REFUSALS[profile](displayName) } : {}),
    };
}

export function runnerSummaries(runners = RUNNERS, profile = 'dedicated') {
    return Object.values(runners).map((runner) => runnerSummary(runner, profile));
}
