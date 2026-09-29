// Hardware profiles (DS005). `dedicated`: the GPU has its own memory and the
// controller sizes models against it (DS003). `unified`: the GPU shares the
// machine's memory (NVIDIA GB10 in DGX Spark); there is one pool, read from
// /proc/meminfo, and the rules below apply.
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

export const PROFILES = Object.freeze(['dedicated', 'unified']);

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
// judged to fit the share.
function userSizingWarning(fields, runner = 'llama-server') {
    const prefix = `Sized with ${fields.join(', ')} from the model entry added at run time. Nothing checks these against the weights: `;
    return runner === 'vllm'
        ? `${prefix}an understated value only loosens the check that the weights and KV cache fit vLLM's share, so vLLM may fail `
            + 'to start; its memory need (the share plus its runner RAM) does not depend on it.'
        : `${prefix}an understated value makes the estimate too small, and then only the memory guard stands behind the run.`;
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
        if (estimate.userSizing.length) warnings.push(userSizingWarning(estimate.userSizing));
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
    if (sizing.length) warnings.push(userSizingWarning(sizing, 'vllm'));
    const base = {
        weightsBytes, kvBytes, overheadBytes: VLLM_OVERHEAD_BYTES, runnerRamBytes: VLLM_RUNNER_RAM_BYTES, floorBytes: UNIFIED.floorBytes,
        hostReserveBytes: UNIFIED.hostReserveBytes, poolBytes: memory.totalBytes ?? null, envelope: null, measured: false,
        experimental: true, defaulted, userSizing: sizing,
        basis: 'experimental estimate, not measured: vLLM\'s share of the shared pool plus its runner RAM (measured on a dedicated GPU); '
            + `the share must hold the snapshot, a KV cache for maxModelLen at ${kvPerToken} bytes per token and vLLM's overhead`,
    };
    if (unreadable(memory)) return result('incompatible', MEMORY_UNREADABLE, base, warnings);
    if (!(weightsBytes > 0)) {
        return result('incompatible', `${model.displayName}'s weight size is unknown, so it cannot be sized on unified memory.`, base, warnings);
    }
    const shareOf = (bytes) => Math.floor((bytes / memory.totalBytes) * 100) / 100;
    const maxShare = Math.min(VLLM_MAX_UTILIZATION, shareOf(memory.totalBytes - UNIFIED.hostReserveBytes - VLLM_RUNNER_RAM_BYTES));
    const adminSet = params.gpuMemoryUtilization !== null && params.gpuMemoryUtilization !== undefined;
    const share = adminSet
        ? params.gpuMemoryUtilization
        : Math.min(maxShare, shareOf(memory.availableBytes - UNIFIED.floorBytes - VLLM_RUNNER_RAM_BYTES));
    const budgetBytes = Math.round(Math.max(0, share) * memory.totalBytes);
    const estimate = { ...base, gpuMemoryUtilization: share, budgetBytes, unifiedBytes: budgetBytes + VLLM_RUNNER_RAM_BYTES };
    // Whether it can ever fit: in the admin's share, or in the largest share the reserve allows.
    const ceilingBytes = adminSet ? budgetBytes : Math.max(0, maxShare) * memory.totalBytes;
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
