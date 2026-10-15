// How this agent composes the pworker engine (DS002, DS003): its host (where the
// image keeps things), its Ploinky wording, its hardware limits and the name
// Soul Gateway gives its model. `createAgentController` is the one place that
// builds the controller; the container's main process, the tools and the tests
// all come through it.
//
// A file system or `execFile` that a caller gives is the machine the engine
// reads: the snapshot reader, the memory readers, the core counts and the
// limits all see it, so a test that stands in for the machine does so in one
// place. Production gives none and the engine uses the real ones.
//
// The runner table and the readers below are the engine's, bound to this
// agent's host, wording and limits, with the names they had when the engine
// was part of this repository.

import path from 'node:path';

import {
    createLocalLlmController, createRunnerInstaller as engineCreateRunnerInstaller, createRunners, defineLimits,
    loadSeedCatalog as engineLoadSeedCatalog, readSnapshot as engineReadSnapshot,
} from 'pworker/local-llm';
import { readImageContract as engineReadImageContract } from 'pworker/local-llm/internal/controller/deployments.mjs';
import { cpuFallbackExplanation as engineCpuFallbackExplanation } from 'pworker/local-llm/internal/controller/computeDiagnostics.mjs';
import {
    readGpu as engineReadGpu, readMemory as engineReadMemory, readMemoryPressure as engineReadMemoryPressure,
} from 'pworker/local-llm/internal/controller/hardware.mjs';
import { readVulkanGpu as engineReadVulkanGpu, vulkanContractOf as engineVulkanContractOf } from 'pworker/local-llm/internal/controller/vulkan.mjs';
import { defaultPorts as engineDefaultPorts } from 'pworker/local-llm/internal/runners/index.mjs';
import { createLlamaServerRunner as engineCreateLlamaServerRunner, hostOptions } from 'pworker/local-llm/internal/runners/llamaServer.mjs';

import { agentHost, agentSeedCatalogFile } from './agentHost.mjs';
import { AGENT_LIMITS, createAgentLimits } from './hardwareLimits.mjs';
import { PLOINKY_MESSAGES } from './ploinkyMessages.mjs';

/** The name Soul Gateway gives the agent's one model: <repo>/<agent>/<model id>. */
export const GATEWAY_MODEL_NAME = 'soul_gateway/local-llms/local-llm/default';

// The core counts of the runners read the machine through the limits' own context, so a given file system reaches them here.
function limitsOver(limits, fsApi) {
    if (!fsApi) return limits;
    return defineLimits({ ...limits, capCores: (count, context) => limits.capCores(count, { ...context, fsApi }) });
}

/**
 * The controller of this agent. `env` is used by reference (the operator's
 * switches are read when they matter). `fsApi` and `execFileImpl` stand in for
 * the machine; `readBudget` and `qualificationDataProvider` are trusted test
 * seams of the limits; `limits` replaces the agent's limits whole. Every other
 * option is a seam of the engine's controller and wins over what is composed.
 */
export function createAgentController({ env = process.env, fsApi, execFileImpl, readBudget, qualificationDataProvider, limits, ...seams } = {}) {
    const host = agentHost(env);
    const agentLimits = limits ?? createAgentLimits({ fsApi, readBudget, qualificationDataProvider });
    const machine = {
        ...(fsApi ? { fsApi } : {}),
        ...(execFileImpl ? { execFileImpl } : {}),
    };
    const composed = {};
    if (Object.keys(machine).length) {
        if (seams.snapshot === undefined && seams.readHardwareSnapshot === undefined) {
            composed.readHardwareSnapshot = (options) => engineReadSnapshot({ ...options, ...machine });
        }
        if (fsApi) {
            if (seams.readMemory === undefined) composed.readMemory = () => engineReadMemory({ fsApi, procDir: host.platform.procDir });
            if (seams.readPressure === undefined) composed.readPressure = () => engineReadMemoryPressure({ fsApi, procDir: host.platform.procDir });
            if (seams.imageContract === undefined) composed.imageContract = engineReadImageContract(host.imageContractFile, { fsApi });
            if (seams.runners === undefined) composed.runners = createRunners({ host, messages: PLOINKY_MESSAGES, limits: limitsOver(agentLimits, fsApi) });
        }
    }
    return createLocalLlmController({
        host, messages: PLOINKY_MESSAGES, limits: agentLimits, gatewayModelName: GATEWAY_MODEL_NAME, env, ...composed, ...seams,
    });
}

// --- The engine's tables and readers, bound to this agent --------------------------------------------------------

const TABLE_HOST = agentHost({});

/** The runner table of this agent's host, wording and limits: what the module-level table was. */
export const RUNNERS = createRunners({ host: TABLE_HOST, messages: PLOINKY_MESSAGES, limits: AGENT_LIMITS });

export function getRunner(id) {
    if (typeof id !== 'string' || !Object.hasOwn(RUNNERS, id)) throw Object.assign(new Error(`Unknown runner: ${String(id)}`), { code: 'unknown_runner', details: { runner: id } });
    return RUNNERS[id];
}

export const llamaCppRunner = RUNNERS['llama.cpp'];
export const ikLlamaCppRunner = RUNNERS['ik_llama.cpp'];
export const llamaCppCpuRunner = RUNNERS['llama.cpp-cpu'];
export const llamaCppVulkanRunner = RUNNERS['llama.cpp-vulkan'];
export const ollamaRunner = RUNNERS.ollama;
export const vllmRunner = RUNNERS.vllm;
export const tabbyApiRunner = RUNNERS.tabbyapi;
export const lmStudioRunner = RUNNERS.lmstudio;

/** The GPU as nvidia-smi reports it, worded for Ploinky, over this agent's nvidia-smi and GPU grant. */
export function readGpu(options = {}) {
    const host = agentHost(process.env);
    return engineReadGpu({ nvidiaSmi: host.nvidiaSmi, gpuAccess: host.gpuAccess, messages: PLOINKY_MESSAGES, ...options });
}

/** Why the profile is the CPU, worded for Ploinky. */
export function cpuFallbackExplanation(decision, recovery = null) {
    return engineCpuFallbackExplanation(decision, recovery, PLOINKY_MESSAGES);
}

/** The Vulkan section of the image contract, over this agent's Vulkan layout. */
export function vulkanContractOf(contract) {
    return engineVulkanContractOf(contract, TABLE_HOST.vulkan);
}

/** The image contract file of this agent's image (or `file`), as key=value pairs, or null. */
export function readImageContract(file = TABLE_HOST.imageContractFile, options = {}) {
    return engineReadImageContract(file, options);
}

/** The seed catalog: this agent's own `catalog/models.json` unless a file is given. */
export function loadSeedCatalog(file = agentSeedCatalogFile()) {
    return engineLoadSeedCatalog(file);
}

/** The Vulkan GPU, worded for Ploinky. */
export function readVulkanGpu(options = {}) {
    return engineReadVulkanGpu({ messages: PLOINKY_MESSAGES, ...options });
}

/** The on-demand runner installer, over the container's cache, run root, uv and python unless the caller names others. */
export function createRunnerInstaller(options = {}) {
    return engineCreateRunnerInstaller({
        cacheRoot: path.join(TABLE_HOST.dataDir, 'runners'), runRoot: TABLE_HOST.runRoot, uv: TABLE_HOST.uv, python: TABLE_HOST.python,
        platform: TABLE_HOST.platform, ...options,
    });
}

/** The loopback port of every supported runner of this agent's table (or of `runners`). */
export function defaultPorts(runners = RUNNERS) {
    return engineDefaultPorts(runners);
}

/** A llama-server runner over this agent's driver directory, Vulkan layout and limits unless the caller names others. */
export function createLlamaServerRunner(options) {
    return engineCreateLlamaServerRunner({ ...hostOptions(TABLE_HOST, AGENT_LIMITS), ...options });
}
