// Hardware profiles (DS005). `dedicated`: the GPU has its own memory and the
// controller sizes models against it (DS003). `unified`: the GPU shares the
// machine's memory (NVIDIA GB10 in DGX Spark); there is one pool, read from
// /proc/meminfo, and the rules below apply.
//
// The unified constants come from the Phase 0 measurements on DGX Spark
// (plans/local-llm-multiarch-implementation-log.md) and stay provisional until
// the safety follow-up's stop-latency and headroom measurements are in.

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
    // The memory guard samples MemAvailable this often, loading or ready.
    guardSampleMs: 250,
    // PSI memory `full avg10` stops a runner only together with MemAvailable
    // below twice the floor: page-cache reclaim alone reached 14-28 with 45 GiB
    // or more available.
    psiStopAvg10: 50,
});

/** The profile a snapshot calls for; a snapshot without a usable GPU keeps the dedicated rules. */
export function profileOf(snapshot) {
    const gpu = snapshot?.gpu;
    return gpu?.available && gpu.memoryModel === 'unified' ? 'unified' : 'dedicated';
}

function gib(bytes) {
    return `${(bytes / GIB).toFixed(1)} GiB`;
}

/** The measured rectangle of a model's unified envelope that holds these parameters, or null. */
export function envelopeFor(model, runnerId, params) {
    const rectangles = (model?.unified?.envelope || []).filter((entry) => entry.runner === runnerId);
    return rectangles.find((entry) => params.ctxSize <= entry.maxCtx && params.parallel <= entry.maxParallel
        && (!params.mtp || entry.mtp)) || null;
}

function describeEnvelope(model, runnerId) {
    const rectangles = (model?.unified?.envelope || []).filter((entry) => entry.runner === runnerId);
    return rectangles.map((entry) => `context ${entry.maxCtx} x ${entry.maxParallel} slot${entry.maxParallel > 1 ? 's' : ''}`
        + `${entry.mtp ? ' (MTP allowed)' : ''}`).join('; ');
}

/**
 * The unified-memory policy of llama-server (DS005). A run is admitted only
 * inside the model's measured envelope; its need is the buffers llama.cpp
 * logged at that rectangle's corner, the measured transient beyond them, the
 * runtime constant and the prompt cache's bound. The guard is a backstop, not
 * the control: one load allocates faster than any sampling can follow.
 */
export function admitUnifiedLlamaServer({ runnerId, displayName, model, source, params, memory, disk, remainingDownloadBytes = 0 },
    result) {
    const rectangle = envelopeFor(model, runnerId, params);
    if (!rectangle) {
        const measured = describeEnvelope(model, runnerId);
        return result('incompatible', measured
            ? `${model.displayName} was measured on unified memory with ${displayName} only up to ${measured}; `
                + `context ${params.ctxSize} x ${params.parallel} slot${params.parallel > 1 ? 's' : ''}${params.mtp ? ' with MTP' : ''} is outside it.`
            : `${model.displayName} has not been measured on unified memory with ${displayName}, so it cannot run here yet.`, {});
    }
    const cacheRamBytes = UNIFIED.cacheRamMiB * MIB;
    const needBytes = rectangle.bufferBytes + rectangle.transientBytes + UNIFIED.runtimeBytes + cacheRamBytes;
    const estimate = {
        weightsBytes: source.size || 0,
        bufferBytes: rectangle.bufferBytes,
        transientBytes: rectangle.transientBytes,
        runtimeBytes: UNIFIED.runtimeBytes,
        cacheRamBytes,
        unifiedBytes: needBytes,
        poolBytes: memory.totalBytes ?? null,
        floorBytes: UNIFIED.floorBytes,
        envelope: { maxCtx: rectangle.maxCtx, maxParallel: rectangle.maxParallel, mtp: rectangle.mtp },
        basis: 'measured envelope (llama.cpp buffers at the measured corner, measured transient, prompt cache bound)',
    };
    if (!Number.isFinite(memory.totalBytes) || !Number.isFinite(memory.availableBytes)) {
        return result('incompatible', 'System memory cannot be read (/proc/meminfo), so nothing can be sized on unified memory.', estimate);
    }
    if (needBytes > memory.totalBytes - UNIFIED.hostReserveBytes) {
        return result('incompatible', `Needs about ${gib(needBytes)} of the ${gib(memory.totalBytes)} shared memory pool, `
            + `which must keep ${gib(UNIFIED.hostReserveBytes)} for the host. Reduce the context or the parallel slots.`, estimate);
    }
    if (needBytes > memory.availableBytes - UNIFIED.floorBytes) {
        return result('insufficient-now', `Needs about ${gib(needBytes)} of shared memory and ${gib(UNIFIED.floorBytes)} kept free; `
            + `${gib(memory.availableBytes)} is available now. Other processes on this machine hold the rest.`, estimate);
    }
    if (Number.isFinite(disk?.freeBytes) && remainingDownloadBytes * 1.05 > disk.freeBytes) {
        return result('insufficient-now', `The weights need ${gib(remainingDownloadBytes * 1.05)} of free disk (download or copy, with a 5 % reserve); `
            + `${gib(disk.freeBytes)} is free.`, estimate);
    }
    return result('ok', null, estimate);
}
