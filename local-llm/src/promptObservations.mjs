import crypto from 'node:crypto';
import { LocalLlmError } from './errors.mjs';

export const OBSERVATION_ID_RE = /^[A-Za-z0-9_-]{8,128}$/;
const ACTIVE_MS = 300000;
const TERMINAL_MS = 600000;

export function createPromptObservations({ events, current }) {
    const records = new Map();
    const fail = (code) => { throw new LocalLlmError(code, 'The prompt observation is unavailable or no longer valid.'); };
    const publicRecord = ({ lease, started, ended, ...record }) => structuredClone({ ...record, elapsedMs: (ended ?? events.now()) - started });
    function terminal(entry, outcome, errorCode) {
        entry.phase = 'terminal'; entry.outcome = outcome; entry.errorCode = errorCode;
        entry.ended = events.now(); entry.terminalSequence = events.next();
    }
    function prune() {
        const now = events.now();
        for (const [id, entry] of records) {
            if (entry.phase === 'active' && now - entry.started >= ACTIVE_MS) terminal(entry, 'incomplete', 'observation_expired');
            if (entry.phase === 'terminal' && now - entry.ended >= TERMINAL_MS) records.delete(id);
        }
    }
    function begin({ observationId } = {}) {
        prune();
        if (typeof observationId !== 'string' || !OBSERVATION_ID_RE.test(observationId)) fail('invalid_observation');
        if (records.has(observationId)) fail('observation_duplicate');
        if (records.size >= 64 || [...records.values()].filter((entry) => entry.phase === 'active').length >= 16) fail('observation_capacity');
        const state = current();
        if (!state?.target || !state.runnerIdentity?.runnerStartId) fail('not_ready');
        const started = events.now(); const lease = crypto.randomBytes(32).toString('hex');
        const entry = { id: observationId, controllerEpoch: events.epoch, deploymentId: state.deploymentId,
            modelId: state.target.modelId, runnerId: state.target.runnerId, runnerIdentity: structuredClone(state.runnerIdentity),
            phase: 'active', beginSequence: events.next(), firstTokenSequence: null, firstTokenPresent: false,
            stopSequence: null, stopNotificationSequence: null, cancellation: null,
            terminalSequence: null, monotonicMs: started, outcome: null, errorCode: null, lease, started, ended: null };
        records.set(observationId, entry);
        return { target: state.target, lease, observation: publicRecord(entry) };
    }
    function entryFor({ observationId, lease } = {}) {
        prune(); const entry = records.get(observationId);
        if (!entry || entry.phase !== 'active' || typeof lease !== 'string' || lease !== entry.lease) fail('observation_stale');
        return entry;
    }
    function matches(entry) {
        let state; try { state = current(); } catch { return false; }
        return state?.deploymentId === entry.deploymentId && state.runnerIdentity?.runnerStartId === entry.runnerIdentity.runnerStartId;
    }
    function record(args) {
        const entry = entryFor(args);
        if (!matches(entry) || entry.stopSequence !== null || args.event !== 'first-token') fail('observation_stale');
        if (!entry.firstTokenPresent) { entry.firstTokenPresent = true; entry.firstTokenSequence = events.next(); }
        return { recorded: true, observation: publicRecord(entry) };
    }
    function observeStop(args) {
        const entry = entryFor(args);
        if (entry.stopSequence === null && !matches(entry)) fail('observation_stale');
        if (entry.stopSequence !== null && entry.stopNotificationSequence === null) entry.stopNotificationSequence = events.next();
        return { recorded: true, observation: publicRecord(entry) };
    }
    function finish(args) {
        const entry = entryFor(args);
        if (!['success', 'failed'].includes(args.outcome)) fail('invalid_observation');
        if (!matches(entry) && entry.stopSequence === null) fail('observation_stale');
        const errorCode = typeof args.errorCode === 'string' && /^[a-z][a-z0-9_]{0,63}$/.test(args.errorCode) ? args.errorCode : 'prompt_failed';
        const proof = args.cancellation;
        if (proof !== undefined || errorCode === 'prompt_cancelled_by_stop') {
            const positive = (value) => Number.isSafeInteger(value) && value > 0;
            if (args.outcome !== 'failed' || errorCode !== 'prompt_cancelled_by_stop' || !proof || typeof proof !== 'object' || Array.isArray(proof)
                || Object.keys(proof).length !== 3 || !['mechanism', 'stopSequence', 'notificationSequence'].every((key) => Object.hasOwn(proof, key))
                || proof.mechanism !== 'controller-stop-abort-v1' || !positive(proof.stopSequence) || !positive(proof.notificationSequence)
                || proof.stopSequence !== entry.stopSequence || proof.notificationSequence !== entry.stopNotificationSequence
                || proof.stopSequence >= proof.notificationSequence) fail('invalid_observation');
        }
        entry.cancellation = proof ? structuredClone(proof) : null;
        terminal(entry, args.outcome === 'success' ? 'success' : entry.stopSequence !== null ? 'interrupted-after-stop' : 'failed', args.outcome === 'success' ? null : errorCode);
        return { recorded: true, observation: publicRecord(entry) };
    }
    function stop(receipt) {
        prune();
        for (const entry of records.values()) if (entry.phase === 'active' && entry.deploymentId === receipt.deploymentId
            && entry.runnerIdentity.runnerStartId === receipt.runnerIdentity?.runnerStartId) entry.stopSequence ??= receipt.sequence;
    }
    return Object.freeze({ begin, record, observeStop, finish, stop, snapshot: () => { prune(); return [...records.values()].map(publicRecord); } });
}
