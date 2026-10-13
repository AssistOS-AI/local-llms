import crypto from 'node:crypto';

export function createObservationClock({ clock = () => performance.now(), epoch = crypto.randomUUID() } = {}) {
    const origin = clock(); let sequence = 0;
    return { epoch, now: () => Math.max(0, clock() - origin), next: () => ++sequence };
}

export function createStartupObservations(events) {
    let current = null;
    let handle = null;
    let accepting = false;
    const snapshot = () => current ? structuredClone(current) : null;
    function reset(deploymentId) {
        handle = null; accepting = true;
        current = { controllerEpoch: events.epoch, deploymentId, runnerIdentity: null, loadingObserved: false,
            latestHealthObservation: null, firstAcceptedHealthSequence: null, observationErrors: 0,
            pageCachePreparation: { policy: 'existing-best-effort-per-file-drop', phase: 'pending', filesConsidered: 0,
                attempts: 0, reportedSuccesses: 0, reportedFailures: 0, skippedNonRegular: 0, observationErrors: 0 } };
    }
    function attach(deploymentId, process) {
        if (!accepting || current?.deploymentId !== deploymentId) return null;
        handle = process;
        current.runnerIdentity = { runnerStartId: crypto.randomUUID(), pid: Number.isSafeInteger(process.pid) ? process.pid : null };
        return structuredClone(current.runnerIdentity);
    }
    function observe(deploymentId, process, event) {
        if (!accepting || current?.deploymentId !== deploymentId || handle !== process || !process?.running) return false;
        if (!Number.isInteger(event?.status) || event.status < 100 || event.status > 599 || typeof event.accepted !== 'boolean') { current.observationErrors += 1; return false; }
        const observation = { status: event.status, accepted: event.accepted, sequence: events.next(), monotonicMs: events.now() };
        current.latestHealthObservation = observation;
        if (event.status === 503) current.loadingObserved = true;
        if (event.accepted && current.firstAcceptedHealthSequence === null) current.firstAcceptedHealthSequence = observation.sequence;
        return true;
    }
    function cache(deploymentId, field) {
        if (!accepting || current?.deploymentId !== deploymentId) return;
        const counts = current.pageCachePreparation;
        if (field === 'complete') counts.phase = 'complete';
        else if (Object.hasOwn(counts, field) && typeof counts[field] === 'number') counts[field] = Math.min(Number.MAX_SAFE_INTEGER, counts[field] + 1);
    }
    function error(deploymentId) {
        if (accepting && current?.deploymentId === deploymentId) current.observationErrors += 1;
    }
    function stop(deployment, process) {
        const summary = snapshot();
        const receipt = { controllerEpoch: events.epoch, sequence: events.next(), deploymentId: deployment?.id ?? null,
            runnerIdentity: summary && summary.deploymentId === deployment?.id ? summary.runnerIdentity : null,
            phaseAtStop: deployment?.phase || 'idle', runnerRunning: !!process?.running && handle === process,
            latestHealthObservation: summary?.latestHealthObservation ?? null,
            firstAcceptedHealthSequence: summary?.firstAcceptedHealthSequence ?? null, monotonicMs: events.now(), startupObservation: summary };
        accepting = false;
        return receipt;
    }
    return Object.freeze({ reset, attach, observe, cache, error, snapshot, stop });
}
