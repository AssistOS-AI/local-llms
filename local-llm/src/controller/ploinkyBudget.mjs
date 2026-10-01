// Container budgets from Ploinky (DS003, DS005): what a finite cgroup memory
// limit and CPU quota leave this agent, kept apart from the raw hardware
// telemetry. Nothing here changes an unlimited machine: without a finite
// limit every view below is the original object.
//
// Readings are strict. A byte count is known when it parses to a safe integer
// of zero or more, so a finite zero is known (no headroom), while a missing,
// malformed, negative or non-finite reading is unknown and is never taken for
// zero or for unlimited. A cgroup file that says exactly `max` is unlimited,
// which is not the same as a file that cannot be read. An absent memory.max
// (no cgroup v2 memory file, as on hosts without a container limit) is no
// limit; a memory.max that exists but cannot be read, or holds anything but
// `max` or a byte count, is unknown.

const UNREADABLE_REASON = 'budget_unreadable';

/** A known byte count: a safe integer >= 0. Zero is known. */
export function knownByte(value) {
    return Number.isSafeInteger(value) && value >= 0;
}

/** Strictly parse a cgroup byte file: a decimal integer, `max` (unlimited), or unknown. */
export function parseCgroupBytes(text) {
    if (typeof text !== 'string') return { state: 'unknown', bytes: null };
    const value = text.trim();
    if (value === 'max') return { state: 'unlimited', bytes: null };
    if (!/^\d+$/.test(value)) return { state: 'unknown', bytes: null };
    const bytes = Number(value);
    return knownByte(bytes) ? { state: 'known', bytes } : { state: 'unknown', bytes: null };
}

/**
 * The raw CPU quota of cgroup v2 `cpu.max` as a fraction of CPUs (quota /
 * period), with no rounding and no minimum of one: `150000 100000` is 1.5.
 * `max` means no quota (null); anything unreadable is null too. Integer
 * thread selection (physicalCoreCount) stays separate.
 */
export function parseCpuQuota(text) {
    const match = typeof text === 'string' ? /^(max|\d+)\s+(\d+)\s*$/.exec(text.trim()) : null;
    if (!match || match[1] === 'max') return null;
    const quota = Number(match[1]);
    const period = Number(match[2]);
    if (!(quota > 0) || !(period > 0)) return null;
    return quota / period;
}

/**
 * The internal memory-budget observation: {memoryReadState, finiteMemoryBytes,
 * headroomBytes, reasonCode}. `maxText` is the file's text, `undefined` when
 * memory.max does not exist, or `null` when it exists but cannot be read.
 * `established` is a finite limit seen before: a later failure to read the
 * limit never turns it into unlimited.
 */
export function observeMemoryBudget({ maxText, currentText, established = null }) {
    if (maxText === undefined) {
        // No memory.max at all: no container limit (legacy), unless a finite
        // one was already established.
        return knownByte(established) ? unknownBudget(established) : noBudget();
    }
    const max = parseCgroupBytes(maxText);
    if (max.state === 'unlimited') return noBudget();
    // A limit beyond any real memory (all digits, past the safe range) is none,
    // as the legacy projection reads it.
    if (max.state === 'unknown' && typeof maxText === 'string' && /^\d+$/.test(maxText.trim())) return noBudget();
    if (max.state !== 'known') {
        // memory.max exists but cannot be read or is malformed: the limit is
        // unknown, never unlimited (an established one is kept).
        return unknownBudget(knownByte(established) ? established : null);
    }
    const finite = max.bytes;
    const current = parseCgroupBytes(currentText);
    if (current.state !== 'known') return unknownBudget(finite);
    return Object.freeze({
        memoryReadState: 'known',
        finiteMemoryBytes: finite,
        headroomBytes: Math.max(0, finite - current.bytes),
        reasonCode: null,
    });
}

function noBudget() {
    return Object.freeze({ memoryReadState: 'known', finiteMemoryBytes: null, headroomBytes: null, reasonCode: null });
}

function unknownBudget(finite) {
    return Object.freeze({ memoryReadState: 'unknown', finiteMemoryBytes: finite, headroomBytes: null, reasonCode: UNREADABLE_REASON });
}

// The snapshot carries the observation as a non-enumerable field, so it is
// used by admission but never published in the overview's hardware object.
const OBSERVATION = 'memoryBudget';

export function attachMemoryBudget(snapshot, observation) {
    if (snapshot && typeof snapshot === 'object' && observation) {
        Object.defineProperty(snapshot, OBSERVATION, { value: observation, enumerable: false, configurable: true, writable: false });
    }
    return snapshot;
}

const CPU_QUOTA = 'cpuQuota';

/** Attach the raw CPU quota (CPUs, or null) to a snapshot, non-enumerable like the memory observation. */
export function attachCpuQuota(snapshot, quota) {
    if (snapshot && typeof snapshot === 'object' && Number.isFinite(quota) && quota > 0) {
        Object.defineProperty(snapshot, CPU_QUOTA, { value: quota, enumerable: false, configurable: true, writable: false });
    }
    return snapshot;
}

export function cpuQuotaOf(snapshot) {
    const quota = snapshot?.[CPU_QUOTA];
    return Number.isFinite(quota) && quota > 0 ? quota : null;
}

/**
 * The observation for a snapshot: the attached one, else derived from the
 * legacy `cgroupMemory` projection ({maxBytes, currentBytes} or null).
 */
export function memoryBudgetOf(snapshot) {
    const attached = snapshot?.[OBSERVATION];
    if (attached) return attached;
    const legacy = snapshot?.cgroupMemory;
    if (!legacy || !knownByte(legacy.maxBytes)) return noBudget();
    if (!knownByte(legacy.currentBytes)) return unknownBudget(legacy.maxBytes);
    return Object.freeze({
        memoryReadState: 'known',
        finiteMemoryBytes: legacy.maxBytes,
        headroomBytes: Math.max(0, legacy.maxBytes - legacy.currentBytes),
        reasonCode: null,
    });
}

/**
 * The RAM a dedicated or unified policy sizes against, capped once by a known
 * finite budget: total = min(raw total, limit), available = min(raw available,
 * effective total, headroom). The raw figures are kept beside it
 * (`physicalTotalBytes` for utilization denominators). Without a finite budget
 * the original object is returned unchanged.
 */
export function effectiveMemory(memory, observation) {
    if (!observation || observation.memoryReadState !== 'known' || !knownByte(observation.finiteMemoryBytes)) return memory;
    const raw = memory || {};
    const rawTotal = knownByte(raw.totalBytes) ? raw.totalBytes : null;
    const rawAvailable = knownByte(raw.availableBytes) ? raw.availableBytes : null;
    const totalBytes = rawTotal === null ? observation.finiteMemoryBytes : Math.min(rawTotal, observation.finiteMemoryBytes);
    const candidates = [totalBytes, observation.headroomBytes];
    if (rawAvailable !== null) candidates.push(rawAvailable);
    return {
        ...raw,
        totalBytes,
        availableBytes: Math.min(...candidates),
        physicalTotalBytes: raw.totalBytes ?? null,
        physicalAvailableBytes: raw.availableBytes ?? null,
        budgetBytes: observation.finiteMemoryBytes,
    };
}

export const BUDGET_UNREADABLE_MESSAGE = 'This agent has a container memory limit, but its current memory use cannot be read, '
    + 'so no model is started on unverified headroom. Retry in a moment; if it persists, check the container\'s cgroup '
    + '(memory.current) and restart local-llm.';

export const BUDGET_LIMIT_UNREADABLE_MESSAGE = 'This agent\'s container memory limit cannot be read, so no model is started on an '
    + 'unknown budget. Retry in a moment; if it persists, check the container\'s cgroup (memory.max) and restart local-llm.';

/**
 * The refusal before any runner dispatch, including the CPU profile: a known
 * finite limit whose current use cannot be read. Temporary (insufficient-now),
 * never a permanent incompatibility. Null when admission may proceed.
 */
export function budgetGuard(observation, result) {
    if (observation?.memoryReadState !== 'unknown') return null;
    const message = knownByte(observation.finiteMemoryBytes) ? BUDGET_UNREADABLE_MESSAGE : BUDGET_LIMIT_UNREADABLE_MESSAGE;
    return result('insufficient-now', message, {}, [], UNREADABLE_REASON);
}

/**
 * The optional overview budget: exact fractional CPUs and the finite memory
 * limit, only when one is set; null otherwise (unlimited overviews unchanged).
 */
export function overviewBudget({ observation, cpuQuota = null, gpuShare = null }) {
    const memoryBytes = knownByte(observation?.finiteMemoryBytes) ? observation.finiteMemoryBytes : null;
    const cpus = Number.isFinite(cpuQuota) && cpuQuota > 0 ? cpuQuota : null;
    if (memoryBytes === null && cpus === null && !gpuShare) return null;
    return Object.freeze({ cpus, memoryBytes, source: 'ploinky', ...(gpuShare ? { gpuShare: Object.freeze({ ...gpuShare, assurance: 'best-effort' }) } : {}) });
}

export const MPS_VARIABLES = Object.freeze(['CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'CUDA_MPS_PINNED_DEVICE_MEM_LIMIT']);
export const MPS_BUDGET_INVALID_MESSAGE = 'The Ploinky GPU budget is incomplete or invalid. It requires the managed MPS pipe, an SM percentage of 1–100, and device 0 memory in whole MiB. Repair the GPU share through Ploinky and restart local-llm.';

export function parseMpsBudget(env = {}) {
    const present = MPS_VARIABLES.filter((key) => env[key] !== undefined);
    if (!present.length) return Object.freeze({ state: 'none', gpuShare: null, environment: null });
    const sm = env.CUDA_MPS_ACTIVE_THREAD_PERCENTAGE;
    const memory = typeof env.CUDA_MPS_PINNED_DEVICE_MEM_LIMIT === 'string' ? /^0=([0-9]+)M$/.exec(env.CUDA_MPS_PINNED_DEVICE_MEM_LIMIT) : null;
    const smPercent = typeof sm === 'string' && /^[0-9]+$/.test(sm) ? Number(sm) : NaN;
    const vramMiB = memory ? Number(memory[1]) : NaN;
    const vramBytes = vramMiB * 1024 * 1024;
    if (present.length !== 3 || env.CUDA_MPS_PIPE_DIRECTORY !== '/run/ploinky-mps-pipe'
        || !Number.isInteger(smPercent) || smPercent < 1 || smPercent > 100 || !Number.isSafeInteger(vramMiB) || vramMiB < 512 || !knownByte(vramBytes)) {
        return Object.freeze({ state: 'unknown', gpuShare: null, environment: null, reason: MPS_BUDGET_INVALID_MESSAGE, reasonCode: 'gpu_budget_invalid' });
    }
    return Object.freeze({
        state: 'known', gpuShare: Object.freeze({ smPercent, vramBytes }),
        environment: Object.freeze({ CUDA_MPS_PIPE_DIRECTORY: '/run/ploinky-mps-pipe', CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: String(smPercent), CUDA_MPS_PINNED_DEVICE_MEM_LIMIT: `0=${vramMiB}M` }),
    });
}

export function attachGpuBudget(snapshot, budget) {
    if (snapshot && typeof snapshot === 'object') Object.defineProperty(snapshot, 'gpuBudget', { value: budget, enumerable: false, configurable: true });
    return snapshot;
}

export function gpuBudgetOf(snapshot, env = {}) {
    return snapshot?.gpuBudget || parseMpsBudget(env);
}

export function effectiveGpu(gpu, budget) {
    if (budget?.state !== 'known') return gpu;
    const totalBytes = Math.min(gpu.totalBytes, budget.gpuShare.vramBytes);
    return { ...gpu, totalBytes, freeBytes: Math.min(gpu.freeBytes, totalBytes), deviceTotalBytes: gpu.totalBytes, budgetBytes: budget.gpuShare.vramBytes };
}

export function mpsRunnerEnvironment(env, profile) {
    if (profile === 'cpu') return {};
    const budget = parseMpsBudget(env);
    if (budget.state === 'unknown') throw Object.assign(new Error(budget.reason), { code: budget.reasonCode });
    return budget.environment || {};
}

export { UNREADABLE_REASON as BUDGET_UNREADABLE };
