// Durable controller state under /data/state: the active deployment, the
// parameters last chosen per (model, runner), idempotency records for Run
// requests, and the user model registry. Every write is atomic (temp file,
// fsync, rename) so a crash or a drain never leaves a torn file.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

// Registry entries are kept as written; the catalog hides any that are not
// valid for its schema (catalog v3 migrates nothing).
export const STATE_VERSION = 1;
const MAX_REQUEST_RECORDS = 200;

// Accepted Run requests by request id. Every id the request pattern allows is
// an ordinary key, `__proto__`, `constructor` and `toString` included: a record
// counts only as the map's own property (never an inherited member), and it is
// defined as a data property (never assigned, so no setter such as
// `__proto__` runs and the map's prototype never changes). The map itself is an
// ordinary object, so it compares and serializes like the other state fields.
function defineRecord(map, requestId, record) {
    Object.defineProperty(map, requestId, { value: record, enumerable: true, writable: true, configurable: true });
}

/** A request map holding the own entries of `entries` (as loaded from the state file). */
export function requestMap(entries = {}) {
    const map = {};
    for (const [requestId, record] of Object.entries(entries)) defineRecord(map, requestId, record);
    return map;
}

/** The accepted record of `requestId`, or null: only an own entry counts. */
export function acceptedRequest(requests, requestId) {
    return typeof requestId === 'string' && requests && Object.hasOwn(requests, requestId) ? requests[requestId] : null;
}

/** Record an accepted request, whatever its id. */
export function recordRequest(requests, requestId, record) {
    defineRecord(requests, requestId, record);
}

export function emptyState() {
    return {
        version: STATE_VERSION,
        deployment: null,
        params: {},
        requests: requestMap(),
        registry: [],
        // Blob digests each Ollama tag's pulls have touched, so partial
        // downloads are counted and deleted per tag.
        ollamaPulls: {},
        // On-demand runner installs by runner id: phase, progress, error and
        // the licence acceptance (who and when).
        runnerInstalls: {},
    };
}

function writeAtomically(fsApi, target, text) {
    const directory = path.dirname(target);
    fsApi.mkdirSync(directory, { recursive: true, mode: 0o700 });
    const temporary = path.join(directory, `.${path.basename(target)}.${crypto.randomUUID()}.tmp`);
    const descriptor = fsApi.openSync(temporary, 'wx', 0o600);
    try {
        fsApi.writeFileSync(descriptor, text);
        fsApi.fsyncSync(descriptor);
    } finally {
        fsApi.closeSync(descriptor);
    }
    fsApi.renameSync(temporary, target);
}

function normalizeState(value) {
    const state = emptyState();
    if (value.deployment && typeof value.deployment === 'object') state.deployment = value.deployment;
    if (value.params && typeof value.params === 'object' && !Array.isArray(value.params)) state.params = value.params;
    if (value.requests && typeof value.requests === 'object' && !Array.isArray(value.requests)) state.requests = requestMap(value.requests);
    if (Array.isArray(value.registry)) state.registry = value.registry;
    if (value.ollamaPulls && typeof value.ollamaPulls === 'object' && !Array.isArray(value.ollamaPulls)) {
        state.ollamaPulls = value.ollamaPulls;
    }
    if (value.runnerInstalls && typeof value.runnerInstalls === 'object' && !Array.isArray(value.runnerInstalls)) {
        state.runnerInstalls = value.runnerInstalls;
    }
    return state;
}

export function createStateStore({ dataDir, fsApi = fs } = {}) {
    const file = path.join(dataDir, 'state', 'controller.json');

    function load() {
        let value;
        try {
            value = JSON.parse(fsApi.readFileSync(file, 'utf8'));
        } catch (error) {
            if (error?.code === 'ENOENT') return emptyState();
            // A corrupt file is kept for inspection and replaced by a clean state.
            try { fsApi.renameSync(file, `${file}.corrupt-${Date.now()}`); } catch {}
            return emptyState();
        }
        if (!value || typeof value !== 'object' || value.version !== STATE_VERSION) {
            // Another version's file is kept aside rather than overwritten by the next save.
            try { fsApi.renameSync(file, `${file}.unsupported-${Date.now()}`); } catch {}
            return emptyState();
        }
        return normalizeState(value);
    }

    function save(state) {
        // Own entries only; Object.fromEntries defines each as a data property, so a `__proto__` id stays a key.
        const requests = Object.entries(state.requests || {})
            .sort(([, left], [, right]) => String(right.at || '').localeCompare(String(left.at || '')))
            .slice(0, MAX_REQUEST_RECORDS);
        writeAtomically(fsApi, file, `${JSON.stringify({ ...state, requests: Object.fromEntries(requests) }, null, 2)}\n`);
    }

    return Object.freeze({ file, load, save });
}

/**
 * Bring a persisted state back after a container restart. Runner processes
 * and download streams did not survive, so nothing is running now: an
 * interrupted download becomes paused (resumable, but only on the next
 * explicit Run) and a starting or running model becomes idle. Weights stay.
 */
export function reconcileAfterRestart(state, now = new Date().toISOString()) {
    for (const install of Object.values(state.runnerInstalls || {})) {
        if (['downloading', 'installing'].includes(install?.phase)) {
            install.phase = 'paused';
            install.pausedReason = 'The agent restarted during the install; press Install to resume.';
            install.updatedAt = now;
        }
    }
    const deployment = state.deployment;
    if (!deployment) return state;
    if (['downloading', 'copying', 'verifying', 'pulling'].includes(deployment.phase)) {
        deployment.phase = 'paused';
        deployment.pausedReason = 'The agent restarted during the download; press Run to resume.';
    } else if (['starting', 'loading', 'ready', 'stopping'].includes(deployment.phase)) {
        deployment.phase = 'idle';
        deployment.runner = null;
    }
    deployment.updatedAt = now;
    return state;
}
