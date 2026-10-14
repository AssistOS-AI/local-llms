// The GPU-aware half of the unified memory guard (DS005). On a qualified GB10
// under a finite Ploinky RAM budget the memory cgroup does not charge `device`
// (cuMemAlloc) memory, which only `nvidia-smi` shows. This module samples the
// runner's GPU memory on a timer and decides, once per guard tick, whether the
// charged memory plus the GPU memory exceeds the budget. It never decides on
// a reading that is stale, unmapped, unreadable or unqualified: it records why
// and leaves the other checks of the guard in charge.

import { parsePageCacheBytes, unifiedTupleQualified } from './hardware.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const EXEC_BOUND_MS = 1500;
const EXEC_HISTORY = 16;

function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * Whether a unified guard samples the GPU, from the GPU facts its Run was
 * admitted on (`{ gpu, shareKnown }`, kept in memory by recheckAdmission; none
 * gives `unqualified`) and whether a finite budget was latched at guard start.
 * `unavailable` means the sampler runs and no reading is valid yet.
 */
export function gpuGuardSignal(facts, budgeted) {
    if (!facts || !unifiedTupleQualified(facts.gpu)) return 'unqualified';
    return !budgeted || facts.shareKnown ? 'not-applicable' : 'unavailable';
}

/** The GB10 fields a guard publishes (DS005), in the order `status` shows them. */
export function gpuGuardStats(signal) {
    return { gpuBudgetSignal: signal, gpuHeldBytes: null, gpuChargedBytes: null, gpuSampleAt: null, gpuSampleAgeMs: null,
        gpuSampleExecMsMedian: null, gpuSamples: 0 };
}

/**
 * Starts the sampler (the first sample now) and returns `{ evaluate, unavailable, cancel }`.
 * `stats` is the guard's own stats object: the sampler writes only the `gpu*`
 * fields of it, and never after `cancel`.
 *
 * Single flight: the next sample is scheduled only after the previous exec has
 * settled. Each exec gets a signal that ends with the guard (`cancel`) or after
 * `min(1500, intervalMs)`, so a hung `nvidia-smi` is stopped, not waited for.
 */
export function startGpuBudgetGuard({ runnerPid, intervalMs, stats, readGpuProcesses, readProcGroup, readLeafStat, monotonicNow, wallNow }) {
    const stop = new AbortController();
    const execBoundMs = Math.min(EXEC_BOUND_MS, intervalMs);
    const execTimes = [];
    let cancelled = false;
    let timer = null;
    let execTimer = null;
    let latest = null;
    let sequence = 0;
    // The first over-budget reading of a streak, kept until a valid reading is under budget (R6-3).
    let pending = null;

    const schedule = () => {
        if (cancelled) return;
        timer = setTimeout(() => { void sample(); }, intervalMs);
        timer.unref?.();
    };

    // The summed GPU memory of the rows in the runner's process group, or null when no row is.
    const heldBy = (rows) => {
        let held = null;
        for (const row of rows) {
            let group = null;
            try { group = readProcGroup(row.pid); } catch {}
            if (group === runnerPid) held = (held ?? 0) + row.usedBytes;
        }
        return held;
    };

    async function sample() {
        timer = null;
        if (cancelled) return;
        const bound = new AbortController();
        execTimer = setTimeout(() => bound.abort(), execBoundMs);
        execTimer.unref?.();
        const startedMs = monotonicNow();
        let result;
        try {
            result = await readGpuProcesses({ signal: AbortSignal.any([stop.signal, bound.signal]) });
        } catch (error) {
            result = { ok: false, reason: String(error?.message || error) };
        } finally {
            clearTimeout(execTimer);
            execTimer = null;
        }
        // A cancelled guard drops a late result: no write, no next sample.
        if (cancelled) return;
        let held = null;
        let state = 'error';
        try {
            if (result?.ok === true && Array.isArray(result.rows)) {
                held = heldBy(result.rows);
                state = held === null ? 'unmapped' : 'observed';
            }
        } catch {}
        const at = monotonicNow();
        const execMs = Number.isFinite(result?.execMs) ? result.execMs : at - startedMs;
        execTimes.push(execMs);
        if (execTimes.length > EXEC_HISTORY) execTimes.shift();
        sequence += 1;
        latest = { state, gheldBytes: held, at, sequence };
        stats.gpuSamples += 1;
        stats.gpuSampleExecMsMedian = median(execTimes);
        if (state === 'observed') {
            stats.gpuHeldBytes = held;
            // The sample gpuHeldBytes came from, so a pair of the two is consistent.
            stats.gpuSampleAt = wallNow();
        }
        schedule();
    }
    void sample();

    const unavailable = () => {
        stats.gpuBudgetSignal = 'unavailable';
        return null;
    };

    /**
     * One decision per guard tick, from the tick's own budget observation (its
     * `currentBytes` is the same read as its limit). Returns `{ chargedBytes,
     * gpuBytes, budgetBytes }` to stop the runner, else null.
     *
     * charged = memory.current - (active_file + inactive_file). Over budget
     * stops at once when more than 1 GiB over; otherwise a newer sample must
     * confirm it (R6-3). A tick without a valid reading keeps the pending
     * confirmation; a valid reading under budget clears it.
     */
    function evaluate(observation) {
        const sampled = latest;
        stats.gpuSampleAgeMs = sampled ? Math.max(0, monotonicNow() - sampled.at) : null;
        if (!sampled || sampled.state !== 'observed' || stats.gpuSampleAgeMs > 2 * intervalMs) return unavailable();
        const budgetBytes = observation?.finiteMemoryBytes;
        const currentBytes = observation?.currentBytes;
        if (!Number.isSafeInteger(budgetBytes) || !Number.isSafeInteger(currentBytes) || currentBytes < 0) return unavailable();
        let pageCache = null;
        try { pageCache = parsePageCacheBytes(readLeafStat()); } catch {}
        if (pageCache === null) return unavailable();
        const chargedBytes = Math.max(0, currentBytes - pageCache);
        stats.gpuBudgetSignal = 'observed';
        stats.gpuChargedBytes = chargedBytes;
        const over = chargedBytes + sampled.gheldBytes - budgetBytes;
        if (over <= 0) {
            pending = null;
            return null;
        }
        if (over > GIB || (pending !== null && sampled.sequence > pending.sequence)) {
            return { chargedBytes, gpuBytes: sampled.gheldBytes, budgetBytes };
        }
        pending ??= { sequence: sampled.sequence };
        return null;
    }

    function cancel() {
        cancelled = true;
        clearTimeout(timer);
        clearTimeout(execTimer);
        timer = null;
        execTimer = null;
        stop.abort();
    }

    return { evaluate, unavailable, cancel };
}
