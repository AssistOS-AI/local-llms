// The container's hardware limits for the pworker engine (DS003, DS005): what a
// finite cgroup memory limit, a CPU quota and an MPS GPU share leave this
// agent. The engine knows no limit; it calls the hooks below at fixed points
// (pworker `lib/local-llm/limits.mjs` documents each one) and uses their
// answers as they are. Every key order, warning and refusal text here is the
// one the agent had before the engine moved, so the overview and the
// admission verdicts are unchanged byte for byte.
//
// The memory limit this process has seen is state of one limits value
// (`createAgentLimits`): a later failure to read `memory.max` never turns an
// established limit into unlimited, for the snapshot reader and the memory
// guard alike, because both read and update the same map, keyed by the file
// system reader. The module-level readers at the end share one map for the
// process, as the single process-wide map of the agent always did.

import fs from 'node:fs';

import { defineLimits } from 'pworker/local-llm';
import * as engineAdmission from 'pworker/local-llm/internal/controller/admission.mjs';
import * as engineHardware from 'pworker/local-llm/internal/controller/hardware.mjs';
import * as engineProfiles from 'pworker/local-llm/internal/controller/profiles.mjs';

import { agentHost } from './agentHost.mjs';
import {
    BUDGET_UNREADABLE, MPS_VARIABLES, attachCpuQuota, attachGpuBudget, attachMemoryBudget, budgetGuard, cpuQuotaOf,
    effectiveGpu, effectiveMemory, gpuBudgetOf, knownByte, memoryBudgetOf, mpsRunnerEnvironment, observeMemoryBudget,
    overviewBudget, parseCpuQuota, parseMpsBudget,
} from './controller/ploinkyBudget.mjs';
import { createVllmMpsQualificationResolver, resolveVllmMpsQualification, vllmMpsTuple } from './controller/vllmMpsQualification.mjs';
import { PLOINKY_MESSAGES } from './ploinkyMessages.mjs';

export const APU_BUDGET_REASON = 'APU GPU use is not available while this agent has a finite or unknown container memory limit; such a limit is not verified to cap shared GPU memory.';

// --- The container's cgroup readings, as they were read before the engine moved ---------------------------------

function readText(fsApi, file) {
    try {
        return fsApi.readFileSync(file, 'utf8');
    } catch {
        return null;
    }
}

/**
 * The container's memory limit and use from cgroup v2 (`memory.max`,
 * `memory.current`), or null when there is no limit (`max`) or it cannot be
 * read. The CPU profile caps its pool with it (DS005): /proc/meminfo shows the
 * host's memory, not the container's.
 */
function projectCgroupMemory(maxText, currentText) {
    const bytes = (text) => {
        const value = typeof text === 'string' && /^\d+$/.test(text.trim()) ? Number(text.trim()) : null;
        return Number.isSafeInteger(value) ? value : null;
    };
    const maxBytes = bytes(maxText);
    return maxBytes === null ? null : { maxBytes, currentBytes: bytes(currentText) };
}

export function readCgroupMemory({ fsApi = fs } = {}) {
    return projectCgroupMemory(readText(fsApi, '/sys/fs/cgroup/memory.max'), readText(fsApi, '/sys/fs/cgroup/memory.current'));
}

// A cgroup file's text; undefined when it does not exist (ENOENT), null when
// it exists but cannot be read (EACCES, EIO, ...). The legacy readers above
// keep collapsing both to null.
function readObservedText(fsApi, file) {
    try {
        return fsApi.readFileSync(file, 'utf8');
    } catch (error) {
        return error?.code === 'ENOENT' ? undefined : null;
    }
}

// The finite memory limit this process has seen, per filesystem reader (the
// real one in production): a later failure to read memory.max never turns an
// established limit into unlimited.
const establishedMemoryLimits = new WeakMap();

/**
 * The internal memory-budget observation (ploinkyBudget.mjs): unlike
 * readCgroupMemory, it keeps a literal `max` (no limit) apart from a file that
 * cannot be read, and a finite limit apart from an unreadable `memory.current`.
 * Used by admission and the memory guard; never published in the overview.
 * `limits` is the map of the limits value that reads (the process-wide one by default).
 */
function captureWith(limits, { fsApi = fs, established = limits.get(fsApi) ?? null } = {}) {
    const maxText = readObservedText(fsApi, '/sys/fs/cgroup/memory.max');
    const currentText = readObservedText(fsApi, '/sys/fs/cgroup/memory.current');
    const observation = observeMemoryBudget({ maxText, currentText, established });
    if (Number.isSafeInteger(observation.finiteMemoryBytes)) limits.set(fsApi, observation.finiteMemoryBytes);
    return { raw: projectCgroupMemory(maxText, currentText), observation };
}

export function captureCgroupMemory(options = {}) {
    return captureWith(establishedMemoryLimits, options);
}

export function readCgroupMemoryObservation(options = {}) {
    return captureCgroupMemory(options).observation;
}

/** The raw cgroup v2 CPU quota in CPUs (1.5 for `150000 100000`), or null without one. */
export function readCpuQuota({ fsApi = fs } = {}) {
    return parseCpuQuota(readText(fsApi, '/sys/fs/cgroup/cpu.max'));
}

// --- The limits the engine calls ---------------------------------------------------------------------------

const GPU_SHARE_UNSUPPORTED_ON_VULKAN = 'GPU shares need NVIDIA MPS; Vulkan GPUs cannot use this share.';
const GPU_SHARE_NEEDS_DEDICATED = 'Ploinky GPU sharing requires a verified dedicated-memory GPU. Unified or unknown memory-model GPUs cannot use this MPS share; clear it through Ploinky.';
const GPU_SHARE_TELEMETRY = 'The physical GPU capacity or current free memory cannot be read under this Ploinky share. Retry after repairing GPU telemetry.';
const APU_UNWATCHABLE = 'stopped: APU shared GPU memory cannot be watched under a finite or unknown container memory limit';
const APU_CONTAINER_LIMIT = 'APU shared GPU memory cannot be used under a container memory limit.';

/**
 * The limits of one engine instance. `fsApi` is the file system the container's
 * limits are read from (the real one in production). `readBudget` replaces the
 * memory guard's sample and `qualificationDataProvider` the reviewed
 * qualification data; both are trusted test seams, and production uses neither.
 * `established` is the map of the finite memory limits this instance has seen.
 */
export function createAgentLimits({ fsApi = fs, readBudget = null, qualificationDataProvider = undefined, established = new WeakMap() } = {}) {
    const capture = (options) => captureWith(established, options);
    const resolveQualification = typeof qualificationDataProvider === 'function'
        ? createVllmMpsQualificationResolver(qualificationDataProvider) : resolveVllmMpsQualification;
    return defineLimits({
        // The cgroup CPU quota caps the physical and performance core counts.
        capCores(count, { fsApi: reader = fsApi } = {}) {
            const quota = readText(reader, '/sys/fs/cgroup/cpu.max');
            const match = quota && /^(\d+)\s+(\d+)\s*$/.exec(quota.trim());
            if (match && Number(match[2]) > 0) return Math.min(count, Math.max(1, Math.ceil(Number(match[1]) / Number(match[2]))));
            return count;
        },
        // Read before memory, so a failing memory read still leaves an established limit recorded.
        observeSnapshot({ fsApi: reader = fsApi } = {}) {
            return capture({ fsApi: reader });
        },
        decorateSnapshot({ snapshot, fsApi: reader = fsApi, env, observation }) {
            const cgroup = observation ?? capture({ fsApi: reader });
            if (snapshot.gpu?.memoryModel === 'shared' && (cgroup.observation.memoryReadState === 'unknown'
                || Number.isSafeInteger(cgroup.observation.finiteMemoryBytes))) {
                snapshot.gpu = { available: false, state: 'unusable', backend: 'vulkan', reason: APU_BUDGET_REASON };
            }
            snapshot.cgroupMemory = cgroup.raw;
            // Internal budget facts (ploinkyBudget.mjs), not published with the snapshot.
            attachCpuQuota(snapshot, readCpuQuota({ fsApi: reader }));
            attachGpuBudget(snapshot, parseMpsBudget(env));
            return attachMemoryBudget(snapshot, cgroup.observation);
        },
        // The CPU profile sizes against the container's memory, not the host's.
        memoryPool(memory, snapshot, { amount }) {
            const cgroupMemory = snapshot?.cgroupMemory;
            const limit = cgroupMemory?.maxBytes;
            // A known limit of zero is a limit, not an absent one.
            if (!(Number.isSafeInteger(limit) && limit >= 0)) return null;
            const totalBytes = Math.min(memory.totalBytes, limit);
            let availableBytes = Math.min(memory.availableBytes, totalBytes);
            if (Number.isFinite(cgroupMemory.currentBytes)) availableBytes = Math.min(availableBytes, Math.max(0, limit - cgroupMemory.currentBytes));
            return {
                totalBytes, availableBytes, capped: totalBytes < memory.totalBytes || availableBytes < memory.availableBytes,
                warning: `A container memory limit of ${amount(cgroupMemory.maxBytes)} applies.`,
            };
        },
        refuseProfile({ stage, profile, snapshot, refuse }) {
            if (stage === 'admission') {
                const budget = memoryBudgetOf(snapshot);
                if (profile === 'apu' && budget.memoryReadState === 'known' && Number.isSafeInteger(budget.finiteMemoryBytes)) {
                    return refuse('incompatible', 'C3', APU_BUDGET_REASON);
                }
                return null;
            }
            if (stage === 'policy' && Number.isSafeInteger(snapshot?.cgroupMemory?.maxBytes)) {
                return refuse('incompatible', 'C3', APU_CONTAINER_LIMIT);
            }
            return null;
        },
        admit({ stage, snapshot, env, result, gpu, selected }) {
            const budget = gpuBudgetOf(snapshot, env);
            if (stage === 'before-policy') {
                // A finite container memory limit whose current use cannot be read refuses every profile, the CPU one
                // included, before any policy runs: temporary, never a permanent incompatibility.
                const unreadableBudget = budgetGuard(memoryBudgetOf(snapshot), result, snapshot?.memory);
                if (unreadableBudget) return unreadableBudget;
                if (budget.state === 'unknown') return result('incompatible', budget.reason, {}, [], budget.reasonCode);
                return null;
            }
            if (stage === 'vulkan') {
                return budget.state === 'known' ? result('incompatible', GPU_SHARE_UNSUPPORTED_ON_VULKAN, {}, [], 'gpu_share_unsupported') : null;
            }
            if (stage === 'gpu' && budget.state === 'known') {
                if (selected !== 'dedicated' || gpu.memoryModel !== 'dedicated') {
                    return { refusal: result('incompatible', GPU_SHARE_NEEDS_DEDICATED, {}, [], 'gpu_share_unsupported') };
                }
                if (![gpu.totalBytes, gpu.freeBytes].every((bytes) => Number.isFinite(bytes) && bytes >= 0 && bytes <= Number.MAX_SAFE_INTEGER) || gpu.totalBytes <= 0) {
                    return { refusal: result('insufficient-now', GPU_SHARE_TELEMETRY, {}, [], 'gpu_budget_unreadable') };
                }
                return { share: { gpu: effectiveGpu(gpu, budget) } };
            }
            return null;
        },
        effective({ snapshot }) {
            return { memory: effectiveMemory(snapshot?.memory || {}, memoryBudgetOf(snapshot)) };
        },
        refuseResourceRequest(context) {
            const { policy, gpu, refuse, gib } = context;
            if (policy === 'ollama') {
                const { estimate, marginBytes } = context;
                if (gpu.budgetBytes != null && estimate.gpuBytes !== null && estimate.gpuBytes > gpu.freeBytes - marginBytes) {
                    return refuse('insufficient-now', `Pinned GPU layers need about ${gib(estimate.gpuBytes)}; only ${gib(gpu.freeBytes)} of the Ploinky GPU budget is free now.`);
                }
                return null;
            }
            if (policy === 'vllm') {
                const { params, gpuMemoryUtilization, denominator } = context;
                if (gpu.budgetBytes != null && params.gpuMemoryUtilization != null && gpuMemoryUtilization * denominator > gpu.totalBytes) {
                    return refuse('incompatible', `gpuMemoryUtilization ${gpuMemoryUtilization} allocates ${gib(gpuMemoryUtilization * denominator)} against physical GPU memory, exceeding the ${gib(gpu.totalBytes)} Ploinky GPU share. Lower gpuMemoryUtilization or clear the share.`);
                }
                return null;
            }
            return null;
        },
        // The container budget Ploinky set (exact fractional CPUs, the memory limit), only when there is one.
        overview({ snapshot, env }) {
            return overviewBudget({ observation: memoryBudgetOf(snapshot), cpuQuota: cpuQuotaOf(snapshot), gpuShare: gpuBudgetOf(snapshot, env).gpuShare });
        },
        runnerEnvironment({ env, profile, adapterEnvironment }) {
            for (const key of MPS_VARIABLES) delete adapterEnvironment[key];
            return { ...adapterEnvironment, ...mpsRunnerEnvironment(env, profile) };
        },
        // Under a finite container memory limit the guard also samples its headroom; `watched` says there is one.
        guardSample() {
            const observation = readBudget ? readBudget() : capture({ fsApi }).observation;
            return { ...observation, watched: Number.isSafeInteger(observation?.finiteMemoryBytes) };
        },
        guardBreach(sample, { check, memory, available }) {
            if (check === 'watchable') {
                return sample?.memoryReadState !== 'known' || sample.finiteMemoryBytes !== null ? { message: APU_UNWATCHABLE } : null;
            }
            if (check === 'headroom') {
                if (sample?.memoryReadState !== 'known'
                    || (knownByte(sample.finiteMemoryBytes)
                        && (!knownByte(memory?.totalBytes) || !knownByte(memory?.availableBytes) || !knownByte(sample.headroomBytes)))) {
                    return { unreadable: `stopped: ${BUDGET_UNREADABLE}: a required host or container memory reading could not be read twice in a row, `
                        + 'so the runner cannot be watched' };
                }
                if (!Number.isSafeInteger(sample.finiteMemoryBytes) || !Number.isFinite(available)) return { available };
                return { available: Math.min(available, sample.finiteMemoryBytes, sample.headroomBytes) };
            }
            return null;
        },
        // A runner that needs MPS qualification is admitted under a GPU share only for a reviewed tuple.
        vllmQualification: (context) => resolveQualification(vllmMpsTuple({ gpu: context.gpu, runnerLockEntry: context.runnerLockEntry })),
    });
}

// --- Engine functions bound to this agent's limits, with the names and signatures they had ----------------------

/** The limits of this process: one value, one established-limit map, like the single process-wide map of old. */
export const AGENT_LIMITS = createAgentLimits({ established: establishedMemoryLimits });

// The nvidia-smi the container's host names (LOCAL_LLM_NVIDIA_SMI overrides it for tests).
const hostValues = () => agentHost(process.env);

export function readSnapshot(options = {}) {
    const host = hostValues();
    return engineHardware.readSnapshot({
        nvidiaSmi: host.nvidiaSmi, gpuAccess: host.gpuAccess, messages: PLOINKY_MESSAGES, limits: AGENT_LIMITS, ...options,
    });
}

export function physicalCoreCount(options = {}) {
    return engineHardware.physicalCoreCount({ limits: AGENT_LIMITS, ...options });
}

export function performanceCoreCount(options = {}) {
    return engineHardware.performanceCoreCount({ limits: AGENT_LIMITS, ...options });
}

/** The CPU pool within a container memory limit (`{ maxBytes, currentBytes }`, or null), without the limits' warning. */
export function cpuPool(memory = {}, cgroupMemory = null) {
    const { warning, ...pool } = engineProfiles.cpuPool(memory, { cgroupMemory }, AGENT_LIMITS);
    void warning;
    return pool;
}

// A caller that holds a share decided elsewhere passes it as `gpuBudget`. The limits find it on the snapshot, as the
// snapshot reader attaches it, so it goes on a view of the caller's snapshot, which is left as it was.
function withGpuBudget(snapshot, gpuBudget) {
    if (gpuBudget === undefined || snapshot === undefined || snapshot === null) return snapshot;
    return attachGpuBudget(Object.create(snapshot), gpuBudget);
}

export function admit({ gpuBudget, snapshot, limits = AGENT_LIMITS, env, ...input }) {
    return engineAdmission.admit({ ...input, snapshot: withGpuBudget(snapshot, gpuBudget), limits, env: env ?? {}, messages: PLOINKY_MESSAGES });
}

export function admitOllama(input) {
    return engineAdmission.admitOllama({ limits: AGENT_LIMITS, ...input });
}

export function admitVllm(input) {
    return engineAdmission.admitVllm({ limits: AGENT_LIMITS, ...input });
}

export function admitCpuLlamaServer({ cgroupMemory = null, snapshot, ...input }, result) {
    return engineProfiles.admitCpuLlamaServer({ snapshot: snapshot ?? { cgroupMemory }, limits: AGENT_LIMITS, messages: PLOINKY_MESSAGES, ...input }, result);
}

export function admitVenusLlamaServer({ cgroupMemory = null, snapshot, ...input }, result) {
    return engineProfiles.admitVenusLlamaServer({ snapshot: snapshot ?? { cgroupMemory }, limits: AGENT_LIMITS, ...input }, result);
}

export function admitApuLlamaServer({ cgroupMemory = null, snapshot, ...input }, result) {
    return engineProfiles.admitApuLlamaServer({ snapshot: snapshot ?? { cgroupMemory }, limits: AGENT_LIMITS, ...input }, result);
}

export function admitCpuOllama({ cgroupMemory = null, snapshot, ...input }, result) {
    return engineProfiles.admitCpuOllama({ snapshot: snapshot ?? { cgroupMemory }, limits: AGENT_LIMITS, messages: PLOINKY_MESSAGES, ...input }, result);
}
