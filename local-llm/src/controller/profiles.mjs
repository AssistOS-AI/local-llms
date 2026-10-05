// Hardware profiles (DS005). `dedicated`: the GPU has its own memory and the
// controller sizes models against it (DS003). `unified`: the GPU shares the
// machine's memory (NVIDIA GB10 in DGX Spark); there is one pool, read from
// /proc/meminfo, and the rules below apply. `cpu`: no NVIDIA GPU is usable, so
// models run on the CPU against the same kind of pool, capped by a container
// memory limit when there is one; decideProfile chooses it automatically.
//
// The unified constants come from the Phase 0 measurements on DGX Spark
// (plans/local-llm-multiarch-implementation-log.md) and stay provisional: the
// owner deferred the benchmark phases (2026-09-29), so none of them is a
// calibrated envelope. No per-model benchmark is a prerequisite for a Run;
// admission estimates from the model's data and checks the host now.

import {
    DEFAULT_KV_BYTES_PER_TOKEN,
    DEFAULT_LAYERS,
    VLLM_MAX_UTILIZATION,
    VLLM_OVERHEAD_BYTES,
    VLLM_RUNNER_RAM_BYTES,
    computeBufferBytes,
} from './admission.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

export const PROFILES = Object.freeze(['dedicated', 'unified', 'cpu']);

export const UNIFIED = Object.freeze({
    // Admission refuses a run needing more than MemTotal less this as incompatible:
    // the idle host held 8-13 GiB that was not available.
    hostReserveBytes: 16 * GIB,
    // The run must leave this much MemAvailable (insufficient-now otherwise), and
    // the memory guard stops the runner below it.
    floorBytes: 8 * GIB,
    // The CUDA context and the runner process beyond the buffers llama.cpp logs
    // (gpt-oss-20b at 128k x 4: logged buffers 15.7 GiB, measured 16.6-16.9 GiB).
    runtimeBytes: 1.5 * GIB,
    // llama-server's prompt cache in host RAM (`--cache-ram`, MiB): a hard bound
    // it allocates in one burst when a new task starts. 8192 is llama.cpp's
    // default and the measured baseline; lower sizes wait for their measurements.
    cacheRamMiB: 8192,
    // llama-server's -b and -ub on unified memory (fixed, like every other memory flag).
    batchSize: 2048,
    // The memory guard samples MemAvailable this often, loading or ready.
    guardSampleMs: 250,
    // PSI memory `full avg10` stops a runner only together with MemAvailable
    // below twice the floor: page-cache reclaim alone reached 14-28 with 45 GiB
    // or more available.
    psiStopAvg10: 50,
});

// The CPU profile's constants. Provisional, like the unified ones: no benchmark
// calibrates them. They are smaller than the unified values because the first
// CPU host is a 5.76 GiB Podman machine, where a 16 GiB reserve and an 8 GiB
// floor would refuse everything; the reserve and floor below scale with the
// pool and converge on the unified values on large hosts. The memory guard
// samples at UNIFIED.guardSampleMs and stops on UNIFIED.psiStopAvg10.
export const CPU = Object.freeze({
    // The runner process beyond the buffers llama.cpp logs.
    runtimeBytes: 512 * MIB,
    // llama-server's prompt cache in host RAM (`--cache-ram`, MiB): a hard bound.
    cacheRamMiB: 256,
    // llama-server's -b and -ub on the CPU (fixed, like every other memory flag).
    batchSize: 512,
});

// A GPU that stays unreadable this long no longer holds the CPU profile back.
export const UNREADABLE_COMMIT_MS = 60_000;

// The load modes llama-server may use on unified memory. They fill the page
// cache very differently (DS005), so an envelope rectangle names the one it
// was measured with and admits only that mode.
export const UNIFIED_LOAD_MODES = Object.freeze(['dio', 'none']);

/** The profile a snapshot calls for; a snapshot without a usable GPU keeps the dedicated rules. */
export function profileOf(snapshot) {
    const gpu = snapshot?.gpu;
    return gpu?.available && gpu.memoryModel === 'unified' ? 'unified' : 'dedicated';
}

function gib(bytes) {
    return `${(bytes / GIB).toFixed(1)} GiB`;
}

function rectanglesFor(model, runnerId) {
    return (model?.unified?.envelope || []).filter((entry) => entry.runner === runnerId);
}

/**
 * The measured rectangle of a model's unified envelope that holds these
 * parameters, or null. The load mode must be the one it was measured with:
 * no mode stands in for another.
 */
export function envelopeFor(model, runnerId, params) {
    return rectanglesFor(model, runnerId).find((entry) => entry.loadMode === params.loadMode
        && params.ctxSize <= entry.maxCtx && params.parallel <= entry.maxParallel && (!params.mtp || entry.mtp)) || null;
}

function slots(parallel) {
    return `${parallel} slot${parallel > 1 ? 's' : ''}`;
}

function describeEnvelope(model, runnerId) {
    return rectanglesFor(model, runnerId).map((entry) => `context ${entry.maxCtx} x ${slots(entry.maxParallel)}`
        + `${entry.mtp ? ' (MTP allowed)' : ''} with load mode ${entry.loadMode}`).join('; ');
}

const METADATA_HINT = 'add them to the model entry (Settings, or the local_llm_model_update tool) for a closer estimate';

// The pool checks every unified policy ends with: the host reserve decides
// whether it can ever fit, the floor whether it fits now, then the disk.
function poolVerdict({ needBytes, memory, disk, remainingDownloadBytes, result, estimate, warnings, tooLarge }) {
    if (needBytes > memory.totalBytes - UNIFIED.hostReserveBytes) {
        return result('incompatible', `Needs about ${gib(needBytes)} of the ${gib(memory.totalBytes)} shared memory pool, `
            + `which must keep ${gib(UNIFIED.hostReserveBytes)} for the host. ${tooLarge}`, estimate, warnings);
    }
    if (needBytes > memory.availableBytes - UNIFIED.floorBytes) {
        return result('insufficient-now', `Needs about ${gib(needBytes)} of shared memory and ${gib(UNIFIED.floorBytes)} kept free; `
            + `${gib(memory.availableBytes)} is available now. Other processes on this machine hold the rest.`, estimate, warnings);
    }
    if (Number.isFinite(disk?.freeBytes) && remainingDownloadBytes * 1.05 > disk.freeBytes) {
        return result('insufficient-now', `The weights need ${gib(remainingDownloadBytes * 1.05)} of free disk (download or copy, with a 5 % reserve); `
            + `${gib(disk.freeBytes)} is free.`, estimate, warnings);
    }
    return result('ok', null, estimate, warnings);
}

function unreadable(memory) {
    return !Number.isFinite(memory.totalBytes) || !Number.isFinite(memory.availableBytes);
}

const MEMORY_UNREADABLE = 'System memory cannot be read (/proc/meminfo), so nothing can be sized on unified memory.';

// The sizing fields of an entry added at run time that feed an estimate. Only
// the KV and MTP terms can be lowered this way: the weights are bound to the
// pinned size and sha256, and the compute, runtime and cache terms are fixed.
function userSizing(model, fields) {
    if (model.seed) return [];
    return fields.filter((field) => {
        const [group, key] = field.split('.');
        return model[group]?.[key] !== undefined && model[group]?.[key] !== null;
    });
}

// llama-server's need grows with the KV and MTP terms; vLLM's need is its share
// plus its runner RAM, so there these fields only decide whether the model is
// judged to fit the share. The warning names where the values came from
// (`sizingSource`, DS002): typed by hand (or `manual`) is the original text;
// a GGUF header read when the model was added is read again from the verified
// file after the download, and once it was (`sizingVerified`, set on the
// copy the controller admits with) the values are the file's own, unless the
// file's architecture could not be used (`sizingUnchecked`); a snapshot's
// config.json was checked against its digest at lookup.
function userSizingWarning(fields, runner = 'llama-server', model = {}) {
    const vllm = runner === 'vllm';
    const tail = vllm
        ? `an understated value only loosens the check that the weights and KV cache fit vLLM's share, so vLLM may fail `
            + 'to start; its memory need (the share plus its runner RAM) does not depend on it.'
        : 'an understated value makes the estimate too small, and then only the memory guard stands behind the run.';
    if (model.sizingSource === 'gguf-header' && typeof model.sizingUnchecked === 'string' && !vllm) {
        return `Sized with ${fields.join(', ')} from the GGUF header read when the model was added; the downloaded file's header could not `
            + `be used to check them (${model.sizingUnchecked}), so they stand unchecked: ${tail}`;
    }
    if (model.sizingSource === 'gguf-header' && model.sizingVerified === true && !vllm) {
        return `Sized with ${fields.join(', ')} from the GGUF header of the downloaded file, which was checked against its sha256; `
            + 'the header read when the model was added was read again from it.';
    }
    if (model.sizingSource === 'gguf-header') {
        const later = vllm
            ? 'vLLM loads the snapshot, not that file, so nothing reads it again: '
            : 'the header is read again from the verified file after the download, and the last admissions use the values it gives; until then ';
        return `Sized with ${fields.join(', ')} from the GGUF header read when the model was added; ${later}${tail}`;
    }
    if (model.sizingSource === 'config.json') {
        return `Sized with ${fields.join(', ')} from the config.json of the model's pinned snapshot, checked against its digest when the model `
            + `was looked up. Nothing checks these against the weights: ${tail}`;
    }
    return `Sized with ${fields.join(', ')} from the model entry added at run time. Nothing checks these against the weights: ${tail}`;
}

// Honest about what a default can miss (the defaults are DS003's).
function defaultedWarning(defaulted) {
    const notes = [];
    if (defaulted.includes('memory.kvBytesPerToken')) {
        notes.push(`memory.kvBytesPerToken (the ${DEFAULT_KV_BYTES_PER_TOKEN / 1024} KiB per token default can understate the KV cache `
            + 'several times for large dense models)');
    }
    if (defaulted.includes('memory.layers')) notes.push(`memory.layers (${DEFAULT_LAYERS} assumed for the MTP draft)`);
    if (defaulted.includes('contextLength')) notes.push('contextLength (the context is not capped at the model\'s training context)');
    return `Estimated without ${notes.join(', ')}; ${METADATA_HINT}.`;
}

/**
 * What llama-server needs on unified memory when no trusted envelope covers
 * the parameters: an estimate from the model's data and the fixed unified
 * flags, never a measurement. All weights are resident (-ngl 999); the KV
 * cache is f16 for the whole ctxSize (--kv-unified makes it the pool, not a
 * per-slot size); the compute buffers use the flash-attention formula at the
 * fixed micro-batch; MTP adds a draft context: one more layer of KV cache and
 * one more set of compute buffers; then the runtime constant and the prompt
 * cache's explicit bound. Inputs the entry lacks fall back to the dedicated
 * defaults and are named, so the admin can add them.
 */
export function estimateUnifiedLlamaServer({ model, source, params, memory: host = {} }) {
    const memory = model.memory || {};
    const defaulted = [];
    const kvPerToken = memory.kvBytesPerToken ?? (defaulted.push('memory.kvBytesPerToken'), DEFAULT_KV_BYTES_PER_TOKEN);
    const kvBytes = Math.round(kvPerToken * params.ctxSize + (memory.fixedKvBytes || 0));
    const computeBytes = Math.round(computeBufferBytes({ ubatchSize: UNIFIED.batchSize, flashAttn: 'on' }));
    let mtpBytes = 0;
    if (params.mtp) {
        const layers = memory.layers ?? (defaulted.push('memory.layers'), DEFAULT_LAYERS);
        mtpBytes = Math.round(kvBytes / layers + computeBytes);
    }
    if (!Number.isInteger(model.contextLength)) defaulted.push('contextLength');
    const cacheRamBytes = UNIFIED.cacheRamMiB * MIB;
    const weightsBytes = source.size || 0;
    const needBytes = weightsBytes + kvBytes + computeBytes + mtpBytes + UNIFIED.runtimeBytes + cacheRamBytes;
    const from = model.seed ? 'the catalog entry' : 'the model entry added at run time';
    return {
        weightsBytes,
        kvBytes,
        computeBytes,
        mtpBytes,
        runtimeBytes: UNIFIED.runtimeBytes,
        cacheRamBytes,
        unifiedBytes: needBytes,
        poolBytes: host.totalBytes ?? null,
        floorBytes: UNIFIED.floorBytes,
        hostReserveBytes: UNIFIED.hostReserveBytes,
        envelope: null,
        measured: false,
        defaulted,
        userSizing: userSizing(model, ['memory.kvBytesPerToken', 'memory.fixedKvBytes', ...(params.mtp ? ['memory.layers'] : [])]),
        basis: `estimate, not measured on this machine: pinned weight size, f16 KV cache for ctxSize at ${kvPerToken} bytes per token `
            + `(${defaulted.includes('memory.kvBytesPerToken') ? 'a default: the entry has no memory.kvBytesPerToken' : `from ${from}`}), `
            + `compute buffers${params.mtp ? ', an MTP draft context' : ''}, runtime and the prompt cache bound`
            + (defaulted.length ? `; defaults used for ${defaulted.join(', ')}` : ''),
    };
}

/**
 * The unified-memory policy of llama-server (DS005). A trusted envelope
 * rectangle (the shipped or operator catalog) that holds the parameters
 * gives the need from its logged buffers, transient, the runtime constant and
 * the prompt cache's bound. Every other run, bundled or added at run time, is
 * sized by estimateUnifiedLlamaServer: no benchmark is a prerequisite (owner,
 * 2026-09-29). Either way the host reserve, the floor and the disk are checked
 * against a fresh snapshot, and the memory guard is the backstop, not the
 * control: one load allocates faster than any sampling can follow.
 */
export function admitUnifiedLlamaServer({ runnerId, model, source, params, memory, disk, remainingDownloadBytes = 0 },
    result) {
    const rectangle = envelopeFor(model, runnerId, params);
    const warnings = [];
    let estimate;
    if (rectangle) {
        const cacheRamBytes = UNIFIED.cacheRamMiB * MIB;
        estimate = {
            weightsBytes: source.size || 0,
            bufferBytes: rectangle.bufferBytes,
            transientBytes: rectangle.transientBytes,
            runtimeBytes: UNIFIED.runtimeBytes,
            cacheRamBytes,
            unifiedBytes: rectangle.bufferBytes + rectangle.transientBytes + UNIFIED.runtimeBytes + cacheRamBytes,
            poolBytes: memory.totalBytes ?? null,
            floorBytes: UNIFIED.floorBytes,
            envelope: { loadMode: rectangle.loadMode, maxCtx: rectangle.maxCtx, maxParallel: rectangle.maxParallel, mtp: rectangle.mtp },
            measured: false,
            basis: 'catalog envelope data (llama.cpp buffers logged at the rectangle\'s corner, its transient margin, runtime and the prompt '
                + 'cache bound); not validated calibration for this host',
        };
        warnings.push('Sized from the catalog\'s envelope figures, which are not validated calibration for this host; the memory guard '
            + `stops the runner at once if available memory falls below ${gib(UNIFIED.floorBytes)}.`);
    } else {
        if (!(source.size > 0)) {
            return result('incompatible', `${model.displayName}'s weight size is unknown, so it cannot be sized on unified memory; `
                + 'add the model again so its files are pinned.', {});
        }
        estimate = estimateUnifiedLlamaServer({ model, source, params, memory });
        const measured = describeEnvelope(model, runnerId);
        if (measured) {
            warnings.push(`Outside the catalog's envelope (${measured}): context ${params.ctxSize} x ${slots(params.parallel)}`
                + `${params.mtp ? ' with MTP' : ''} with load mode ${params.loadMode} is sized by estimate.`);
        }
        warnings.push(`Not measured on unified memory: about ${gib(estimate.unifiedBytes)} is an estimate. The memory guard stops `
            + `the runner at once if available memory falls below ${gib(UNIFIED.floorBytes)}.`);
        if (estimate.defaulted.length) warnings.push(defaultedWarning(estimate.defaulted));
        if (estimate.userSizing.length) warnings.push(userSizingWarning(estimate.userSizing, 'llama-server', model));
    }
    if (unreadable(memory)) return result('incompatible', MEMORY_UNREADABLE, estimate, warnings);
    return poolVerdict({
        needBytes: estimate.unifiedBytes, memory, disk, remainingDownloadBytes, result, estimate, warnings,
        tooLarge: 'Reduce the context or the parallel slots, or pick a smaller model.',
    });
}

/**
 * vLLM on unified memory: experimental, and only when the operator opted in
 * (the vLLM runner's switch, DS004/DS005). Nothing here is measured. vLLM's
 * --gpu-memory-utilization is a share of the device's total, which on an
 * integrated GPU is the whole pool (MemTotal), so its budget is that share of
 * MemTotal; the API server and engine hold RAM beyond it (the dedicated
 * measurement, unvalidated here). An empty share is computed from this
 * snapshot: at most VLLM_MAX_UTILIZATION, within the host reserve, and
 * leaving the floor free. The run must still fit the reserve and the floor,
 * its weights and KV cache must fit the budget, and the memory guard watches it.
 */
export function admitUnifiedVllm({ model, source, params, memory, disk, remainingDownloadBytes = 0 }, result) {
    const weightsBytes = source.size || 0;
    const defaulted = [];
    const kvPerToken = model.memory?.kvBytesPerToken ?? (defaulted.push('memory.kvBytesPerToken'), DEFAULT_KV_BYTES_PER_TOKEN);
    const kvBytes = Math.round(kvPerToken * params.maxModelLen * (params.kvCacheDtype === 'fp8' ? 0.5 : 1));
    // What the share must hold: weights, KV cache and vLLM's activations, CUDA context and allocator slack
    // (VLLM_OVERHEAD_BYTES, as on a dedicated GPU). CUDA's total is the pool here, so no usable-share discount.
    const modelBytes = weightsBytes + kvBytes + VLLM_OVERHEAD_BYTES;
    const sizing = userSizing(model, ['memory.kvBytesPerToken']);
    const warnings = ['Experimental: vLLM has not been measured on unified memory. Its share and its RAM beyond it are estimates, '
        + `and the memory guard stops it at once if available memory falls below ${gib(UNIFIED.floorBytes)}.`];
    if (defaulted.length) warnings.push(defaultedWarning(defaulted));
    if (sizing.length) warnings.push(userSizingWarning(sizing, 'vllm', model));
    const base = {
        weightsBytes, kvBytes, overheadBytes: VLLM_OVERHEAD_BYTES, runnerRamBytes: VLLM_RUNNER_RAM_BYTES, floorBytes: UNIFIED.floorBytes,
        hostReserveBytes: UNIFIED.hostReserveBytes, poolBytes: memory.totalBytes ?? null, envelope: null, measured: false,
        experimental: true, defaulted, userSizing: sizing,
        basis: 'experimental estimate, not measured: vLLM\'s share of the shared pool plus its runner RAM (measured on a dedicated GPU); '
            + `the share must hold the snapshot, a KV cache for maxModelLen at ${kvPerToken} bytes per token `
            + `(${defaulted.includes('memory.kvBytesPerToken') ? 'a default: the entry has no memory.kvBytesPerToken' : 'from the model entry'}) `
            + 'and vLLM\'s overhead'
            + (defaulted.length ? `; defaults used for ${defaulted.join(', ')}` : ''),
    };
    if (unreadable(memory)) return result('incompatible', MEMORY_UNREADABLE, base, warnings);
    if (!(weightsBytes > 0)) {
        return result('incompatible', `${model.displayName}'s weight size is unknown, so it cannot be sized on unified memory.`, base, warnings);
    }
    // vLLM reads its share against the device's total, which here is the
    // physical pool: a container budget bounds the bytes (memory.totalBytes and
    // availableBytes are the capped view) but never changes that denominator.
    const denominator = Number.isFinite(memory.physicalTotalBytes) && memory.physicalTotalBytes > 0
        ? memory.physicalTotalBytes : memory.totalBytes;
    const shareOf = (bytes) => Math.floor((bytes / denominator) * 100) / 100;
    const maxShare = Math.min(VLLM_MAX_UTILIZATION, shareOf(memory.totalBytes - UNIFIED.hostReserveBytes - VLLM_RUNNER_RAM_BYTES));
    const adminSet = params.gpuMemoryUtilization !== null && params.gpuMemoryUtilization !== undefined;
    const share = adminSet
        ? params.gpuMemoryUtilization
        : Math.min(maxShare, shareOf(memory.availableBytes - UNIFIED.floorBytes - VLLM_RUNNER_RAM_BYTES));
    const budgetBytes = Math.round(Math.max(0, share) * denominator);
    const estimate = { ...base, gpuMemoryUtilization: share, budgetBytes, unifiedBytes: budgetBytes + VLLM_RUNNER_RAM_BYTES };
    // Whether it can ever fit: in the admin's share, or in the largest share the reserve allows.
    const ceilingBytes = adminSet ? budgetBytes : Math.max(0, maxShare) * denominator;
    if (modelBytes > ceilingBytes) {
        return result('incompatible', `Its weights, a KV cache for ${params.maxModelLen} tokens and vLLM's overhead need about ${gib(modelBytes)}; `
            + (adminSet
                ? `gpuMemoryUtilization ${share} gives vLLM about ${gib(budgetBytes)}. Raise it, or leave it empty.`
                : `vLLM can have at most about ${gib(ceilingBytes)} of this pool. Reduce maxModelLen or pick a smaller model.`),
        estimate, warnings);
    }
    if (!adminSet && share < 0.1) {
        return result('insufficient-now', `vLLM needs at least a 0.1 share of the pool with ${gib(UNIFIED.floorBytes)} kept free; `
            + `${gib(memory.availableBytes)} is available now.`, estimate, warnings);
    }
    if (modelBytes > budgetBytes) {
        return result('insufficient-now', `Its weights, KV cache and vLLM's overhead need about ${gib(modelBytes)}; the ${share} share that is free now `
            + `gives about ${gib(budgetBytes)}.`, estimate, warnings);
    }
    return poolVerdict({
        needBytes: estimate.unifiedBytes, memory, disk, remainingDownloadBytes, result, estimate, warnings,
        tooLarge: 'Lower gpuMemoryUtilization or leave it empty.',
    });
}

// ---------------------------------------------------------------- the cpu profile

/**
 * The reason a usable GPU is outside the compute capabilities the image's CUDA
 * runners were built for, or null. A GPU that does not report its capability
 * counts as outside when the image lists any; decideProfile treats that case as
 * unreadable rather than as a verdict.
 */
export function capabilityMismatch(gpu, capabilities = []) {
    const listed = (Array.isArray(capabilities) ? capabilities : []).filter(Boolean);
    if (!listed.length || !gpu?.available) return null;
    const capability = gpu.device?.computeCapability;
    if (capability && listed.includes(capability)) return null;
    return `This image's runners are built for GPUs of compute capability ${listed.join(', ')}; `
        + `${gpu.name} ${capability ? `is ${capability}` : 'did not report its compute capability'}.`;
}

/**
 * Which profile a hardware snapshot calls for (DS005). Pure: the caller keeps
 * the window state and the clock.
 *
 * Positive evidence decides at once: a usable GPU the image supports (its own
 * profile), a usable GPU whose reported capability the image was not built for
 * (`cpu`, cause `mismatch`), no nvidia-smi at all (`absent`), or a GPU whose
 * memory model is unusable. Everything else (`unreadable`, a snapshot without a
 * state, no GPU, no snapshot, a usable GPU that does not report its capability
 * on an image that lists them) may be transient, so it waits: `profile` is null
 * until `commitAfterMs` have passed since the first such snapshot, then `cpu`
 * (cause `unreadable-timeout`). A snapshot that decides resets the window.
 *
 * @returns {{ profile: 'dedicated'|'unified'|'cpu'|null, cause: string, reason: string|null, unreadable: { firstAtMs: number }|null }}
 */
export function decideProfile({ snapshot, capabilities = [], unreadable = null, nowMs = Date.now(), commitAfterMs = UNREADABLE_COMMIT_MS } = {}) {
    const gpu = snapshot?.gpu;
    let waitingReason = 'the hardware snapshot could not be read';
    if (gpu?.available) {
        const mismatch = capabilityMismatch(gpu, capabilities);
        if (!mismatch) return { profile: profileOf(snapshot), cause: 'gpu', reason: null, unreadable: null };
        if (gpu.device?.computeCapability) return { profile: 'cpu', cause: 'mismatch', reason: mismatch, unreadable: null };
        waitingReason = mismatch;
    } else if (gpu?.state === 'absent' || gpu?.state === 'unusable') {
        return { profile: 'cpu', cause: gpu.state, reason: gpu.reason || 'No NVIDIA GPU is usable by this agent.', unreadable: null };
    } else if (gpu?.reason) {
        waitingReason = gpu.reason;
    }
    const firstAtMs = Number.isFinite(unreadable?.firstAtMs) ? unreadable.firstAtMs : nowMs;
    const window = { firstAtMs };
    if (nowMs - firstAtMs >= commitAfterMs) return { profile: 'cpu', cause: 'unreadable-timeout', reason: waitingReason, unreadable: window };
    return { profile: null, cause: 'unreadable', reason: waitingReason, unreadable: window };
}

/** The CPU reserve: what admission leaves for the host, at least 1.5 GiB and at most 16 GiB. */
export function cpuHostReserveBytes(totalBytes) {
    return Math.ceil(Math.min(16 * GIB, Math.max(1.5 * GIB, 0.25 * totalBytes)));
}

/** The CPU floor: the memory the guard keeps available, at least 512 MiB and at most 8 GiB. */
export function cpuFloorBytes(totalBytes) {
    return Math.ceil(Math.min(8 * GIB, Math.max(512 * MIB, 0.10 * totalBytes)));
}

/**
 * The memory the CPU profile sizes against: MemTotal and MemAvailable, and
 * within a container memory limit (cgroup `memory.max`) at most that limit,
 * with what is available now at most the limit less `memory.current`.
 * /proc/meminfo shows the host's memory, not the container's. Memory that
 * cannot be read gives null figures, which admission refuses.
 */
export function cpuPool(memory = {}, cgroupMemory = null) {
    if (!Number.isFinite(memory?.totalBytes) || !Number.isFinite(memory?.availableBytes)) {
        return { totalBytes: null, availableBytes: null, capped: false };
    }
    let totalBytes = memory.totalBytes;
    let availableBytes = memory.availableBytes;
    const limit = cgroupMemory?.maxBytes;
    // A known limit of zero is a limit, not an absent one.
    if (Number.isSafeInteger(limit) && limit >= 0) {
        totalBytes = Math.min(totalBytes, limit);
        availableBytes = Math.min(availableBytes, totalBytes);
        if (Number.isFinite(cgroupMemory.currentBytes)) availableBytes = Math.min(availableBytes, Math.max(0, limit - cgroupMemory.currentBytes));
    }
    return { totalBytes, availableBytes, capped: totalBytes < memory.totalBytes || availableBytes < memory.availableBytes };
}

// A byte count for a message: MiB below 1 GiB, otherwise GiB with two decimals, because the CPU hosts are small.
function amount(bytes) {
    return bytes >= GIB ? `${(bytes / GIB).toFixed(2)} GiB` : `${Math.round(bytes / MIB)} MiB`;
}

const CPU_MEMORY_UNREADABLE = 'System memory cannot be read (/proc/meminfo), so nothing can be sized on the CPU.';

/**
 * What llama-server needs on the CPU: an estimate from the model's data and the
 * fixed CPU flags, never a measurement. All weights are resident (every shard;
 * `--n-gpu-layers 0`); the KV cache is f16 for the whole ctxSize (`--kv-unified`
 * makes it the pool above one slot); the compute buffers use the flash-attention
 * formula at the fixed micro-batch; then the runtime constant and the prompt
 * cache's explicit bound. Inputs the entry lacks fall back to the dedicated
 * defaults and are named, so the admin can add them.
 */
export function estimateCpuLlamaServer({ model, source, params, pool = {} }) {
    const memory = model.memory || {};
    const defaulted = [];
    const kvPerToken = memory.kvBytesPerToken ?? (defaulted.push('memory.kvBytesPerToken'), DEFAULT_KV_BYTES_PER_TOKEN);
    const kvBytes = Math.round(kvPerToken * params.ctxSize + (memory.fixedKvBytes || 0));
    const computeBytes = Math.round(computeBufferBytes({ ubatchSize: CPU.batchSize, flashAttn: 'on' }));
    if (!Number.isInteger(model.contextLength)) defaulted.push('contextLength');
    const cacheRamBytes = CPU.cacheRamMiB * MIB;
    const weightsBytes = source.size || 0;
    const ramBytes = weightsBytes + kvBytes + computeBytes + CPU.runtimeBytes + cacheRamBytes;
    const poolBytes = Number.isFinite(pool.totalBytes) ? pool.totalBytes : null;
    const from = model.seed ? 'the catalog entry' : 'the model entry added at run time';
    return {
        weightsBytes,
        kvBytes,
        computeBytes,
        runtimeBytes: CPU.runtimeBytes,
        cacheRamBytes,
        ramBytes,
        poolBytes,
        floorBytes: poolBytes === null ? null : cpuFloorBytes(poolBytes),
        hostReserveBytes: poolBytes === null ? null : cpuHostReserveBytes(poolBytes),
        measured: false,
        defaulted,
        userSizing: userSizing(model, ['memory.kvBytesPerToken', 'memory.fixedKvBytes']),
        basis: `estimate, not measured on this machine: pinned weight size, f16 KV cache for ctxSize at ${kvPerToken} bytes per token `
            + `(${defaulted.includes('memory.kvBytesPerToken') ? 'a default: the entry has no memory.kvBytesPerToken' : `from ${from}`}), `
            + 'compute buffers, runtime and the prompt cache bound'
            + (defaulted.length ? `; defaults used for ${defaulted.join(', ')}` : ''),
    };
}

// The lead of the "Runs on the CPU" warning, by the cause of the decision (DS005).
const CPU_LEADS = Object.freeze({
    absent: 'no NVIDIA GPU is attached',
    unusable: 'the NVIDIA GPU cannot be used',
    mismatch: 'this image\'s CUDA runners were not built for this GPU',
    'unreadable-timeout': 'the NVIDIA GPU could not be read for 60 s',
});

// What every CPU policy ends with: the warnings, then the verdict. The need must
// fit the pool less the host reserve (else `incompatible`), what is available now
// less the floor (else `insufficient-now`), and the disk; the memory guard stops the
// runner below the floor and is the backstop, not the control. `decision` says why
// this is the CPU (cause, reason); `gpuRecovered` is set by the controller when the
// GPU that was unreadable would now be chosen. `shrink` is the runner's own advice.
function cpuVerdict({ estimate, pool, model, cgroupMemory, disk, remainingDownloadBytes, gpu, decision, result, shrink }) {
    const warnings = [];
    const lead = CPU_LEADS[decision?.cause];
    const why = String(decision?.reason || '').replace(/\.+\s*$/, '').slice(0, 300);
    warnings.push(`Runs on the CPU: ${[lead, why && (lead ? `(${why})` : why)].filter(Boolean).join(' ') || 'no NVIDIA GPU is used'}. `
        + 'Generation is much slower than on a GPU.');
    if (Number.isFinite(estimate.floorBytes)) {
        warnings.push(`Not measured on this machine: about ${amount(estimate.ramBytes)} is an estimate. The memory guard stops the runner at once `
            + `if available memory falls below ${amount(estimate.floorBytes)}.`);
    }
    if (estimate.defaulted.length) warnings.push(defaultedWarning(estimate.defaulted));
    if (estimate.userSizing.length) warnings.push(userSizingWarning(estimate.userSizing, 'llama-server', model));
    if (pool.capped) warnings.push(`A container memory limit of ${amount(cgroupMemory.maxBytes)} applies.`);
    if (decision?.cause === 'unreadable-timeout' && decision.gpuRecovered && gpu?.available) {
        warnings.push(`The NVIDIA GPU is readable now (${gpu.name}); restart local-llm to use it.`);
    }
    if (!Number.isFinite(pool.totalBytes)) return result('incompatible', CPU_MEMORY_UNREADABLE, estimate, warnings);
    if (estimate.ramBytes > pool.totalBytes - estimate.hostReserveBytes) {
        return result('incompatible', `Needs about ${amount(estimate.ramBytes)} of the ${amount(pool.totalBytes)} of memory on this machine, `
            + `which must keep ${amount(estimate.hostReserveBytes)} for the host. ${shrink}`,
        estimate, warnings);
    }
    if (estimate.ramBytes > pool.availableBytes - estimate.floorBytes) {
        return result('insufficient-now', `Needs about ${amount(estimate.ramBytes)} of memory and ${amount(estimate.floorBytes)} kept free; `
            + `${amount(pool.availableBytes)} is available now. Other processes on this machine hold the rest.`, estimate, warnings);
    }
    if (Number.isFinite(disk?.freeBytes) && remainingDownloadBytes * 1.05 > disk.freeBytes) {
        return result('insufficient-now', `The weights need ${amount(remainingDownloadBytes * 1.05)} of free disk (download or copy, with a 5 % reserve); `
            + `${amount(disk.freeBytes)} is free.`, estimate, warnings);
    }
    return result('ok', null, estimate, warnings);
}

/**
 * The CPU policy of llama-server (DS005). There is no envelope and no benchmark:
 * estimateCpuLlamaServer sizes every run, bundled or added at run time.
 */
export function admitCpuLlamaServer({ model, source, params, memory = {}, cgroupMemory = null, disk, remainingDownloadBytes = 0, gpu, decision },
    result) {
    if (!(source.size > 0)) {
        return result('incompatible', `${model.displayName}'s weight size is unknown, so it cannot be sized on the CPU; `
            + 'add the model again so its files are pinned.', {});
    }
    const pool = cpuPool(memory, cgroupMemory);
    const estimate = estimateCpuLlamaServer({ model, source, params, pool });
    return cpuVerdict({ estimate, pool, model, cgroupMemory, disk, remainingDownloadBytes, gpu, decision, result,
        shrink: 'Reduce the context or the parallel slots, or pick a smaller model.' });
}

// Ollama's own need on the CPU beyond the weights and the KV cache: the server and its runner process (provisional, like every CPU constant).
export const OLLAMA_CPU = Object.freeze({ runtimeBytes: 768 * MIB });

/**
 * What Ollama needs on the CPU (DS005), from the tag's pinned size and never
 * from a measurement: all weights resident, an f16 KV cache for the whole
 * `numCtx` (Ollama runs one slot here), the compute buffers at llama.cpp's
 * flash-attention formula and Ollama's default batch of 512, and the server
 * and runner process. `kvBytesPerToken` falls back to the dedicated default and
 * is named when it does.
 */
export function estimateCpuOllama({ model, source, params, pool = {} }) {
    const memory = model.memory || {};
    const defaulted = [];
    const kvPerToken = memory.kvBytesPerToken ?? (defaulted.push('memory.kvBytesPerToken'), DEFAULT_KV_BYTES_PER_TOKEN);
    const kvBytes = Math.round(kvPerToken * params.numCtx + (memory.fixedKvBytes || 0));
    const computeBytes = Math.round(computeBufferBytes({ ubatchSize: 512, flashAttn: 'on' }));
    const weightsBytes = source.size || 0;
    const ramBytes = weightsBytes + kvBytes + computeBytes + OLLAMA_CPU.runtimeBytes;
    const poolBytes = Number.isFinite(pool.totalBytes) ? pool.totalBytes : null;
    const from = model.seed ? 'the catalog entry' : 'the model entry added at run time';
    return {
        weightsBytes,
        kvBytes,
        computeBytes,
        runtimeBytes: OLLAMA_CPU.runtimeBytes,
        ramBytes,
        poolBytes,
        floorBytes: poolBytes === null ? null : cpuFloorBytes(poolBytes),
        hostReserveBytes: poolBytes === null ? null : cpuHostReserveBytes(poolBytes),
        measured: false,
        defaulted,
        userSizing: userSizing(model, ['memory.kvBytesPerToken', 'memory.fixedKvBytes']),
        basis: `estimate, not measured on this machine: the tag's pinned size, f16 KV cache for numCtx at ${kvPerToken} bytes per token `
            + `(${defaulted.includes('memory.kvBytesPerToken') ? 'a default: the entry has no memory.kvBytesPerToken' : `from ${from}`}), `
            + 'compute buffers and the Ollama server and runner'
            + (defaulted.length ? `; defaults used for ${defaulted.join(', ')}` : ''),
    };
}

/**
 * The CPU policy of Ollama (DS005): sized from the tag's pinned size, so a tag
 * whose size is not pinned (an entry stored before tags were pinned) is refused
 * until an update pins it. Ollama pulls its own weights, so the pinned size is
 * also the disk the pull needs.
 */
export function admitCpuOllama({ model, source, params, memory = {}, cgroupMemory = null, disk, remainingDownloadBytes = 0, gpu, decision }, result) {
    if (!(source.size > 0)) {
        return result('incompatible', 'The tag\'s size is not pinned; update the model entry so it is pinned.', {});
    }
    const pool = cpuPool(memory, cgroupMemory);
    const estimate = estimateCpuOllama({ model, source, params, pool });
    return cpuVerdict({ estimate, pool, model, cgroupMemory, disk, remainingDownloadBytes, gpu, decision, result,
        shrink: 'Reduce the context or pick a smaller model.' });
}
