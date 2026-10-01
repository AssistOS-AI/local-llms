// The controller: one active deployment at a time (6 GB of VRAM), every
// mutation serialized through one command queue, downloads and runner
// processes owned here and never by a tool process.
//
//   idle -> downloading | copying | verifying -> starting -> ready -> stopping -> idle
//   (weights the controller fetches, such as a GGUF file)  plus paused and error
//   idle -> starting -> downloading -> loading -> ready -> stopping -> idle
//   (weights the runner fetches itself, such as an Ollama tag)
//
// Nothing here knows a runner by name. Each runner is an adapter
// (src/runners/index.mjs) that brings its port, key, start-up pipeline, chat
// model name and admission policy; each kind of weights has a store
// (weightStores.mjs) keyed by the source's type.

import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import { RUNNERS, defaultPorts, offeredIn, runnerSummary, schemaOf } from '../runners/index.mjs';
import { admit } from './admission.mjs';
import { WEIGHT_FORMATS, loadSeedCatalog, mergeCatalog, unsupportedRegistryEntries, validateModel } from './catalog.mjs';
import { createCommandQueue } from './commandQueue.mjs';
import {
    DownloadError,
    downloadArtifact,
    inspectArtifact,
    removeArtifact,
    resolveHuggingFaceArtifact,
    verifyArtifact,
    verifySnapshotFile,
} from './downloader.mjs';
import { ggufSizing, readGgufHeaderFile } from './ggufHeader.mjs';
import {
    readCgroupMemoryObservation,
    readMemory as readHostMemory,
    readMemoryPressure,
    readSnapshot,
    stoppedQueriesSettled,
    unreapedQueries as hardwareUnreaped,
    untilStopped,
} from './hardware.mjs';
import { lookupHuggingFaceModel } from './modelLookup.mjs';
import { BUDGET_UNREADABLE, cpuQuotaOf, memoryBudgetOf, overviewBudget } from './ploinkyBudget.mjs';
import {
    UNIFIED,
    UNREADABLE_COMMIT_MS,
    capabilityMismatch,
    cpuFloorBytes,
    cpuHostReserveBytes,
    cpuPool,
    decideProfile,
} from './profiles.mjs';
import { createRunnerInstaller, runTool } from './runnerInstaller.mjs';
import { loadRunnerLocks } from './runnerLock.mjs';
import { createLogBuffer, parseRunnerReport, startRunnerProcess } from './runnerProcess.mjs';
import { acceptedRequest, createStateStore, reconcileAfterRestart, recordRequest } from './stateStore.mjs';
import { createWeightStores } from './weightStores.mjs';
import { walkShared } from './workspaceReuse.mjs';
import { DRAIN_QUEUE_WAIT_MS, DRAIN_RUNNER_GRACE_MS } from '../drainBudget.mjs';

const ACTIVE_PHASES = new Set(['downloading', 'copying', 'verifying', 'starting', 'loading', 'ready', 'stopping']);
const TRANSFER_PHASES = new Set(['downloading', 'copying', 'verifying']);
const REQUEST_ID_RE = /^[A-Za-z0-9_-]{8,128}$/;
const PROBE_TIMEOUT_MS = 10_000;
const IMAGE_CONTRACT = '/opt/local-llm/source.contract';
const PROFILE_UNDECIDED = 'No usable GPU has been seen yet, so the hardware profile is not decided; run again once the GPU can be read. '
    + 'If the GPU is still unreadable 60 s after the first failed read, the next Run uses the CPU.';
// Recorded in the state file once the `cpu` profile has been added to stored two-profile user entries (DS002).
const CPU_MIGRATION = '2026-09-30';
// Hugging Face lookups in flight at once (each holds a request, and possibly a 32 MiB read of a model's start); one more is `busy`.
const MAX_LOOKUPS = 2;

/**
 * What the image says about itself (`/opt/local-llm/source.contract`, one
 * key=value per line), or null when there is none (tests, development). The
 * arm64 image names the GPUs its CUDA runners were built for in
 * `gpu_compute_capabilities` (DS005).
 */
export function readImageContract(file = IMAGE_CONTRACT, { fsApi = fs } = {}) {
    let text;
    try {
        text = fsApi.readFileSync(file, 'utf8');
    } catch {
        return null;
    }
    const contract = {};
    for (const line of text.split('\n')) {
        const match = /^([a-z_][a-z0-9_]*)=(.*)$/.exec(line.trim());
        if (match) contract[match[1]] = match[2];
    }
    return Object.freeze(contract);
}

/**
 * Drop a verified download's pages from the page cache (GNU dd's
 * `iflag=nocache count=0` calls posix_fadvise DONTNEED on the whole file; no
 * privilege is needed). A full page cache after a large download caused memory
 * pressure and noisy runs on DGX Spark (DS005). Best effort: a failure is logged.
 */
export function dropPageCache(file, { spawnSyncImpl = spawnSync } = {}) {
    const result = spawnSyncImpl('dd', [`if=${file}`, 'iflag=nocache', 'count=0', 'status=none'], {
        timeout: 30_000, stdio: 'ignore', env: { PATH: '/usr/bin:/bin', LANG: 'C.UTF-8' },
    });
    return result.status === 0 && !result.error;
}

// One log line per file or shard: how it was obtained (C12, R7).
function describeProvenance(entry) {
    const how = { 'in-place': `adopted in place at ${entry.source}`, copy: `copied from ${entry.source}`,
        download: `downloaded from ${entry.source}` }[entry.method] || `${entry.method} from ${entry.source}`;
    return entry.current
        ? `${entry.file}: already verified in the store (${how}, ${entry.bytes} bytes)`
        : `${entry.file}: ${how} (${entry.bytes} bytes), verified`;
}

function paramsKey(modelId, runnerId) {
    return `${modelId}|${runnerId}`;
}

function gib(bytes) {
    return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

// A stop, cancel or drain that lands between two awaited steps must not start
// the next one (a runner launched here would only be killed again).
function throwIfAborted(signal) {
    if (signal?.aborted) throw new LocalLlmError('aborted', 'Stopped while starting.');
}

function sleep(ms, signal) {
    return new Promise((resolve) => {
        const timer = setTimeout(resolve, ms);
        signal?.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true });
    });
}

export function createController({
    dataDir = '/data',
    env = process.env,
    // Test and development seams; production uses the shipped catalog and Hugging Face.
    seedCatalog = loadSeedCatalog(env.LOCAL_LLM_CATALOG_FILE || undefined),
    hfBaseUrl = env.LOCAL_LLM_HF_BASE_URL || 'https://huggingface.co',
    stateStore = createStateStore({ dataDir }),
    runners = RUNNERS,
    // A snapshot's queries stop when `signal` aborts (a Stop, a Cancel or the drain).
    snapshot = ({ signal } = {}) => readSnapshot({ dataDir, signal }),
    // Stopped hardware queries that have not exited yet (logged by the drain),
    // and a promise that settles once every stopped query has exited or been recorded.
    unreapedQueries = hardwareUnreaped,
    settleStoppedQueries = stoppedQueriesSettled,
    download = downloadArtifact,
    inspect = inspectArtifact,
    remove = removeArtifact,
    resolveHf = resolveHuggingFaceArtifact,
    lookup = lookupHuggingFaceModel,
    startRunner = startRunnerProcess,
    fetchImpl = globalThis.fetch,
    ports = defaultPorts(runners),
    apiKeyFactory = () => crypto.randomBytes(32).toString('base64url'),
    readyTimeoutMs = 20 * 60_000,
    stopGraceMs = 10_000,
    pollMs = 1000,
    // On-demand runners (runners plan §5.2): the lock shipped in the image and
    // the agent's own lock of this platform (the image's entry wins on a clash;
    // DS004), a verified cache under /data/runners, runnable copies under /opt/runners.
    runnerLocks = loadRunnerLocks({ image: env.LOCAL_LLM_RUNNER_LOCK || undefined }),
    // Where the installer says what it could not clean up; the controller's log is made below, and takes over.
    installWarnings = { write: () => {} },
    installer = createRunnerInstaller({
        lock: runnerLocks,
        cacheRoot: path.join(dataDir, 'runners'),
        runRoot: env.LOCAL_LLM_RUN_ROOT || undefined,
        onWarning: (message) => installWarnings.write(message),
    }),
    detectRunner = (runner) => runner.detect({ spawnSync, installer }),
    // Test seams for Hugging Face snapshots (the store's defaults are the real downloader).
    downloadSnapshot = undefined,
    inspectSnapshot = undefined,
    resolveSnapshot = undefined,
    // Reads an Ollama tag's identity from its registry when a model is added (DS002).
    resolveOllama = undefined,
    // The last check before loading reads what the real downloader wrote; an
    // injected downloader brings its own check, or none.
    verify = download === downloadArtifact ? verifyArtifact : async () => ({ notes: [] }),
    verifySnapshot = downloadSnapshot === undefined ? verifySnapshotFile : async () => ({ notes: [] }),
    // The agent's own /dev/shm (a private tmpfs), where PyTorch runners keep sockets.
    shmDir = '/dev/shm',
    // Host-memory guard for deployments whose admission sets a RAM floor
    // (vLLM with CPU offload): MemAvailable is sampled this often while the
    // runner loads, and less often once it is ready.
    readMemory = readHostMemory,
    memoryGuardLoadMs = 2000,
    memoryGuardReadyMs = 10_000,
    // The unified profile's guard (DS005): every runner, sampled this often in every phase.
    readPressure = readMemoryPressure,
    unifiedGuardMs = UNIFIED.guardSampleMs,
    // A container memory limit's current headroom (ploinkyBudget.mjs), sampled by the guard.
    readBudget = () => readCgroupMemoryObservation(),
    // The hardware profile; null decides it from the first snapshot (DS005).
    profile: fixedProfile = null,
    // What the image contains; null (no source.contract) makes every runner available.
    imageContract = readImageContract(),
    // The CPU architecture this process runs on (tests may choose it).
    hostArch = process.arch,
    fileExists = (file) => fs.existsSync(file),
    dropCache = dropPageCache,
    // The workspace's shared model directory (C12, DS002): Ploinky mounts
    // <workspace>/.data/shared at /shared in every agent.
    sharedModelsRoot = env.LOCAL_LLM_SHARED_MODELS || '/shared/models',
    // The CUDA driver's JIT cache, in the container's own filesystem (DS004).
    cudaCachePath = path.join(env.LOCAL_LLM_RUN_ROOT || '/opt/runners', '.cuda-cache'),
    now = () => new Date(),
} = {}) {
    const log = createLogBuffer({ file: path.join(dataDir, 'logs', 'runner.log') });
    installWarnings.write = (message) => log.append('controller', `runner install: ${message}`);
    const queue = createCommandQueue();
    const state = reconcileAfterRestart(stateStore.load(), now().toISOString());
    // Where the two runner locks met (DS004): an id in both keeps the image's entry; an agent lock that could not be read is left out.
    for (const id of runnerLocks?.clashes ?? []) {
        log.append('controller', `runner lock: ${id} is in the image's lock and in the agent's lock; the image's entry is used`);
    }
    for (const { file, reason } of runnerLocks?.ignored ?? []) {
        log.append('controller', `runner lock: the agent's lock ${file} is ignored: ${String(reason).slice(0, 300)}`);
    }
    // An entry only the agent's lock has, for a runner whose own executable the image already holds: the image wins here too.
    for (const [id, origin] of Object.entries(installer.lock?.origin ?? {})) {
        const definition = Object.hasOwn(runners, id) ? runners[id] : null;
        if (origin === 'agent' && definition?.executable && fileExists(definition.executable)) {
            log.append('controller', `runner lock: ${id} is in the agent's lock, but the image holds ${definition.executable}; the image's binary is used`);
        }
    }
    let detected = {};
    let job = null;
    // The acquisition planning of the Run being admitted, which Stop, Cancel and drain can abort (F5).
    let planning = null;
    // Stop and Cancel invalidate every Run submitted before them, queued or
    // running (DS001): a Run keeps the generation it was submitted in, and each
    // Stop or Cancel starts a new one. `submittedRuns` holds the current
    // generation's Runs that have not finished their command yet.
    let stopGeneration = 0;
    const submittedRuns = new Set();
    // Every hardware snapshot stops when the agent drains (DS001).
    const hardwareStop = new AbortController();
    // One runner install at a time, beside the deployment job.
    let installJob = null;
    // Installable through the controller: in a runner lock and with an adapter
    // here. A lock entry for a runner this release cannot run is installed only
    // by the CI install check. An entry of the image's lock is always offered.
    // One that only the agent's lock has (DS004) is offered on the cpu profile
    // alone, to a runner with a CPU policy, so the dedicated and unified profiles
    // see exactly the runners they did; and not when the image already holds the
    // runner's own executable (the image wins on a clash).
    const fromAgentLock = (definition) => installer.lock?.origin?.[definition.id] === 'agent';
    const offeredByLock = (definition) => installer.installable(definition.id)
        && (!fromAgentLock(definition)
            || (profile === 'cpu' && schemaOf(definition, 'cpu') !== null && !(definition.executable && fileExists(definition.executable))));
    const installable = (id) => typeof id === 'string' && Object.hasOwn(runners, id) && offeredByLock(runners[id]);
    // Listed by a lock at all, whatever the profile: Uninstall still works to free the disk.
    const lockListed = (id) => typeof id === 'string' && Object.hasOwn(runners, id) && installer.installable(id);
    let runner = null;
    let draining = false;
    // Speed of the last completion served by the ready runner, as the chat
    // responder reported it. In memory only: it describes this runner start.
    let lastCompletion = null;
    // Log sequence number before the current runner started: its report is
    // read only from lines after it, never from a previous runner's output.
    let runnerLogStart = 0;
    // Set when the memory guard stopped a deployment's runner: its message is
    // the deployment's error, not "exited unexpectedly".
    let memoryStop = null;
    // The hardware profile, decided once from the first snapshot that decides it
    // and kept for the container's lifetime; a later snapshot that disagrees is
    // reported by admission, not acted on (DS005). `profileDecision` says why
    // (cause and reason) and `unreadable` is the window of a GPU that could not
    // be read: it is in memory only, and a restart starts it again.
    let profile = fixedProfile;
    let profileDecision = fixedProfile ? { profile: fixedProfile, cause: 'fixed', reason: null, gpuName: null, decidedAt: null } : null;
    let unreadable = null;
    // The reason of the last snapshot that left the profile undecided.
    let undecidedReason = null;
    // The compute capabilities the image's CUDA runners were built for (arm64: 12.1); the amd64 image lists none.
    const capabilities = String(imageContract?.gpu_compute_capabilities || '').split(',').map((cap) => cap.trim()).filter(Boolean);
    // The unified guard's view of the current runner (in memory only).
    let guardStats = null;
    // Saved parameter sets already reported as no longer valid.
    const resetParams = new Set();

    // State this controller cannot read is reported, never misread (catalog v3
    // migrates nothing): a registry entry of an earlier schema stays in the
    // state file, is left out of the catalog, and is named here and in the overview.
    for (const entry of unsupportedRegistryEntries(state.registry, seedCatalog)) {
        log.append('controller', `model entry ${entry.id ?? '(no id)'} is not supported by this catalog and is not offered: ${entry.reason}`);
    }

    // The one change within catalog v3 that arrives with the `cpu` profile (DS002):
    // a stored user entry that lists both GPU profiles (the old default, "every
    // profile") also gets `cpu`, once; an entry that lists one profile keeps it.
    // The flag is set even when the registry is empty, and the save below keeps it.
    if (!state.cpuProfileMigration) {
        for (const entry of state.registry) {
            const listed = entry?.profiles;
            if (Array.isArray(listed) && listed.length === 2 && listed.includes('dedicated') && listed.includes('unified')) {
                listed.push('cpu');
                log.append('controller', `model entry ${entry.id ?? '(no id)'} is now also offered on the cpu profile`);
            }
        }
        state.cpuProfileMigration = CPU_MIGRATION;
    }

    const stores = createWeightStores({
        dataDir,
        env,
        hfBaseUrl,
        inspect,
        download,
        remove,
        resolveHf,
        state: () => state,
        save: () => save(),
        activeArtifact: () => (job !== null ? state.deployment?.artifact ?? null : null),
        ...(downloadSnapshot ? { downloadSnapshot } : {}),
        ...(inspectSnapshot ? { inspectSnapshot } : {}),
        ...(resolveSnapshot ? { resolveSnapshot } : {}),
        ...(resolveOllama ? { resolveOllama } : {}),
        verify,
        verifySnapshot,
        sharedModelsRoot,
    });

    // A drain has a fixed time budget (drainBudget.mjs); a Stop gives the
    // runner the full grace period.
    function runnerGraceMs() {
        return draining ? Math.min(stopGraceMs, DRAIN_RUNNER_GRACE_MS) : stopGraceMs;
    }

    // Every runner lookup goes through the injected table, so tests and the
    // overview see the same definitions the pipelines launch.
    function getRunner(id) {
        if (typeof id !== 'string' || !Object.hasOwn(runners, id)) {
            throw new LocalLlmError('unknown_runner', `Unknown runner: ${String(id)}`, { runner: id });
        }
        return runners[id];
    }
    stateStore.save(state);

    function save() {
        if (state.deployment) state.deployment.updatedAt = now().toISOString();
        stateStore.save(state);
    }

    // The profile is fixed by the first snapshot that decides it (decideProfile,
    // DS005): a usable GPU, or positive evidence that none can be used (`cpu`). A
    // transient failure (a cold nvidia-smi, a GPU not attached yet) leaves it
    // undecided until 60 s after the first one, then it is `cpu`. Parameters are
    // normalized, admitted, launched and guarded under the one committed profile:
    // a Run is refused while it is undecided.
    function commitProfile(snap) {
        if (profile) return profile;
        const decision = decideProfile({ snapshot: snap, capabilities, unreadable, nowMs: now().getTime(), commitAfterMs: UNREADABLE_COMMIT_MS });
        unreadable = decision.unreadable;
        if (!decision.profile) {
            undecidedReason = decision.reason;
            return null;
        }
        profile = decision.profile;
        profileDecision = {
            profile, cause: decision.cause, reason: decision.reason, gpuName: snap?.gpu?.name ?? null, decidedAt: now().toISOString(),
        };
        const because = profile === 'cpu' ? `${decision.cause}: ${String(decision.reason).replace(/\s+/g, ' ').slice(0, 200)}` : snap.gpu.name;
        log.append('controller', `hardware profile: ${profile}${because ? ` (${because})` : ''}`);
        return profile;
    }

    // What the overview and status say about the decision (a copy).
    function publicDecision() {
        return profileDecision ? { ...profileDecision } : null;
    }

    // The decision the CPU admission reads: why this is the CPU and, when it
    // was decided because the GPU stayed unreadable, whether that GPU would be
    // chosen now (a restart would use it; DS005).
    function decisionForAdmission(snap) {
        if (profile !== 'cpu' || !profileDecision) return undefined;
        const gpuRecovered = profileDecision.cause === 'unreadable-timeout'
            && ['dedicated', 'unified'].includes(decideProfile({ snapshot: snap, capabilities, unreadable: null, nowMs: now().getTime() }).profile);
        return { ...profileDecision, gpuRecovered };
    }

    // The pool limits the dashboard shows (DS005): the constants of the unified
    // profile, and on the CPU the reserve and floor of this machine's pool.
    function limitsFor(snap) {
        // The container budget Ploinky set (exact fractional CPUs, the memory
        // limit), only when there is one: unlimited overviews are unchanged.
        const budget = overviewBudget({ observation: memoryBudgetOf(snap), cpuQuota: cpuQuotaOf(snap) });
        const withBudget = (limits) => (budget ? { ...(limits || {}), budget } : limits);
        if (profile === 'unified') return withBudget({ floorBytes: UNIFIED.floorBytes, hostReserveBytes: UNIFIED.hostReserveBytes });
        if (profile !== 'cpu') return withBudget(null);
        const pool = cpuPool(snap?.memory, snap?.cgroupMemory);
        if (!Number.isFinite(pool.totalBytes)) return withBudget(null);
        return withBudget({
            floorBytes: cpuFloorBytes(pool.totalBytes), hostReserveBytes: cpuHostReserveBytes(pool.totalBytes),
            poolBytes: pool.totalBytes, availableBytes: pool.availableBytes,
        });
    }

    // A fresh snapshot that stops with the drain and, for a pending Run or a
    // job, with it. A stop answers at once, whatever stage the snapshot is in
    // (a query, the free-disk read) and even if the snapshot ignores its signal;
    // its late answer or failure is handled and never used, so after a stop
    // nothing is admitted or launched from it.
    async function takeSnapshot(signal = null) {
        const stop = signal ? AbortSignal.any([hardwareStop.signal, signal]) : hardwareStop.signal;
        let snap;
        try {
            snap = await untilStopped(Promise.resolve().then(() => snapshot({ signal: stop })), stop);
        } catch (error) {
            if (!stop.aborted) throw error;
        }
        // The drain first: a Run that meets it in a snapshot refuses as shutting down (DS001).
        if (hardwareStop.signal.aborted) {
            throw new LocalLlmError('shutting_down', 'The agent is restarting; run the model again once it is back.');
        }
        if (signal?.aborted) throw new LocalLlmError('aborted', 'Stopped while starting.');
        return snap;
    }

    // A stop is not a missing GPU: a snapshot that ends because the agent drains
    // or a Run is stopped must neither advance the unreadable window nor decide
    // the profile (DS005).
    const isStop = (error) => error?.code === 'shutting_down' || error?.code === 'aborted';

    async function currentProfile() {
        if (!profile) {
            let snap = null;
            try {
                snap = await takeSnapshot();
            } catch (error) {
                if (isStop(error)) return profile;
            }
            commitProfile(snap);
        }
        return profile;
    }

    // A Run's profile, or its refusal. Nothing is recorded when it is refused,
    // so the same request can be sent again once the GPU can be read.
    async function requireProfile(signal = null) {
        if (profile) return profile;
        let snap = null;
        try {
            snap = await takeSnapshot(signal);
        } catch (error) {
            if (isStop(error)) throw error;
        }
        if (commitProfile(snap)) return profile;
        const undecided = snap?.gpu?.reason || undecidedReason;
        const gpuReason = undecided ? `${String(undecided).replace(/\.$/, '')}. ` : '';
        const admission = { status: 'incompatible', reason: `${gpuReason}${PROFILE_UNDECIDED}`, estimate: { isEstimate: true }, warnings: [] };
        throw new LocalLlmError('admission_incompatible', admission.reason, { admission });
    }

    // The models offered in this profile (catalog v3 `profiles`).
    function catalog() {
        const current = profile || 'dedicated';
        return mergeCatalog(seedCatalog, state.registry).filter((model) => model.profiles.includes(current));
    }

    // Whether this image could ever run on unified memory (DS005). Only positive
    // evidence of the amd64 image rules it out: a source.contract that does not
    // say `architecture=arm64` and either names ik_llama.cpp (built only into
    // the amd64 image) or runs on an x64 CPU. `architecture=arm64` always counts
    // as possible, even alongside an ik_llama_cpp key or on an x64 CPU, and so
    // does no contract (tests, development) or an empty, malformed or
    // architecture-less one without ik_llama_cpp on arm64.
    const amd64Only = Boolean(imageContract) && imageContract.architecture !== 'arm64'
        && (Object.hasOwn(imageContract, 'ik_llama_cpp') || hostArch === 'x64');
    const unifiedPossible = !amd64Only;

    // Whether this image can run a runner at all (DS005): its executable is in
    // the image, or a runner lock lists it. Without a source.contract
    // (tests, development) every runner counts as available.
    function availabilityOf(definition) {
        // A runner that is for some profiles only is not available on the others, whatever the image holds.
        const unoffered = unofferedReason(definition);
        if (unoffered) return { available: false, reason: unoffered };
        if (!imageContract) return { available: true, reason: null };
        // The runner's executable is in the image, or a lock offers it (for an agent-lock entry, where it has a policy).
        const present = definition.executable
            ? fileExists(definition.executable) || (fromAgentLock(definition) && offeredByLock(definition))
            : offeredByLock(definition);
        return present
            ? { available: true, reason: null }
            : { available: false, reason: `${definition.displayName} is not available on this platform: this image does not include it.` };
    }

    // Why a runner a lock lists is not offered for install here (DS004).
    function notInstallableReason(definition) {
        return definition.executable && fileExists(definition.executable)
            ? `${definition.displayName} is part of this image and needs no install.`
            : `${definition.displayName} is not available on this platform: this image does not include it.`;
    }

    // The GPUs the image's CUDA runners were built for (arm64: 12.1). The amd64
    // image names none, and nothing is checked there.
    function gpuMismatch(snap) {
        return capabilityMismatch(snap?.gpu, capabilities);
    }

    // Why a runner has no policy for the committed profile, or null (DS005): on
    // the cpu profile, it has no parameter schema for it, as with every runner
    // but llama.cpp. The dedicated and unified profiles have no such refusal:
    // they refuse through parameter validation, admission and the runner's
    // switch, as they always did.
    function profileRefusal(definition, selected = profile) {
        return unofferedReason(definition, selected)
            ?? (selected ? runnerSummary(definition, selected).profileUnsupportedReason ?? null : null);
    }

    // Why a runner that declares the profiles it serves (llama.cpp's CPU build) is not offered here, or
    // null. The overview leaves such a runner out, and Run and Install refuse it before anything downloads.
    function unofferedReason(definition, selected = profile) {
        if (offeredIn(definition, selected)) return null;
        return `${definition.displayName} runs only on the ${definition.profiles.join(' or ')} profile`
            + `${selected ? `; this machine uses the ${selected} profile` : ''}.`;
    }

    // Admission with the checks the controller owns (availability, the image's
    // GPUs, the profile) before the runner's own policy. `params` were
    // normalized for `selected`; without a committed profile nothing is
    // admitted, so a later snapshot cannot pick rules those params do not match.
    function admitHere({ definition, model, source, params, snap, remainingDownloadBytes = 0, selected = profile }) {
        const refuse = (reason) => ({ status: 'incompatible', reason, estimate: { isEstimate: true }, warnings: [] });
        const availability = availabilityOf(definition);
        if (!availability.available) return refuse(availability.reason);
        // On the CPU no GPU is used, so the GPUs the image was built for do not matter.
        const mismatch = selected === 'cpu' ? null : gpuMismatch(snap);
        if (mismatch) return refuse(mismatch);
        // A snapshot without a GPU is refused by admit() with the GPU's own reason.
        if (!selected && snap?.gpu?.available) return refuse(PROFILE_UNDECIDED);
        return admit({
            runner: definition, model, source, params, snapshot: snap, remainingDownloadBytes, profile: selected || undefined,
            decision: selected === 'cpu' ? decisionForAdmission(snap) : undefined,
        });
    }

    function findModel(modelId) {
        const model = catalog().find((entry) => entry.id === modelId);
        if (!model) throw new LocalLlmError('unknown_model', `No model '${modelId}' in the catalog.`);
        return model;
    }

    // Any model's entry, whatever profiles it is offered in (Delete weights
    // must be able to free what another profile downloaded).
    function findAnyModel(modelId) {
        const model = mergeCatalog(seedCatalog, state.registry).find((entry) => entry.id === modelId);
        if (!model) throw new LocalLlmError('unknown_model', `No model '${modelId}' in the catalog.`);
        return model;
    }

    function storeFor(source) {
        const store = source && Object.hasOwn(stores, source.type) ? stores[source.type] : null;
        if (!store) throw new LocalLlmError('invalid_model', `No weight store for source type '${source?.type}'.`);
        return store;
    }

    // The artifact identity: runners that read the same file share it.
    function artifactKey(source) {
        return source ? storeFor(source).key(source) : null;
    }

    // The weights a runner reads for a model: the source of its weight format.
    function sourceFor(model, definition) {
        return model.sources[definition.weightFormat];
    }

    // Detection may read the disk (an installed runner's record), so it is async.
    async function runnerInfo(id) {
        if (!detected[id]) detected[id] = await detectRunner(runners[id]);
        return detected[id];
    }

    // Whether this deployment allows the runner at all (LM Studio, internal
    // use only: an environment switch the operator sets with `ploinky var`;
    // DS001 says who else can). Runners without a switch are always enabled.
    // A switch may depend on the committed hardware profile (vLLM on unified
    // memory, DS005). While it is undecided, an image that can never run on
    // unified memory counts as dedicated; otherwise the switch sees null.
    function gateOf(definition) {
        const gate = typeof definition.enabled === 'function'
            ? definition.enabled(env, profile ?? (unifiedPossible ? null : 'dedicated'))
            : null;
        return gate && gate.enabled === false
            ? { enabled: false, reason: gate.reason || `${definition.displayName} is not enabled on this deployment.` }
            : { enabled: true, reason: null };
    }

    function assertEnabled(definition) {
        const gate = gateOf(definition);
        if (!gate.enabled) throw new LocalLlmError('runner_disabled', gate.reason, { runner: definition.id });
    }

    function disabledAdmission(gate) {
        return { status: 'incompatible', reason: gate.reason, estimate: { isEstimate: true }, warnings: [] };
    }

    // After a runner is installed or removed while the agent runs.
    function redetectRunners() {
        detected = {};
    }

    function setPhase(phase, extra = {}) {
        const deployment = state.deployment;
        if (!deployment) return;
        deployment.phase = phase;
        Object.assign(deployment, extra);
        log.append('controller', `phase ${phase}${extra.error ? `: ${extra.error}` : ''}`);
        save();
    }

    function inUse(key) {
        const deployment = state.deployment;
        if (!deployment || !key) return false;
        return artifactKey(deployment.artifact) === key
            && (ACTIVE_PHASES.has(deployment.phase) || job !== null);
    }

    async function weightsState(source) {
        return source ? storeFor(source).state(source) : null;
    }

    // How the weights would be obtained and the disk that needs (C12, R4):
    // Run, its preview and the overview all use it for the disk term of
    // admission. It hashes and copies nothing.
    async function acquisitionOf(source, index = null, signal = undefined) {
        if (!source) return { bytesNeeded: 0, files: [] };
        const store = storeFor(source);
        if (store.plan) return store.plan(source, { index, signal });
        const disk = await store.state(source);
        return { bytesNeeded: disk?.total ? Math.max(0, disk.total - (disk.bytes || 0)) : 0, files: [] };
    }

    // `saved` is what an earlier Run left for this model and runner; a Run that normalizes again after it saved its own
    // result passes what was saved before it, so its own result is not taken for an earlier Run's explicit choice.
    function effectiveParams(model, runnerId, override, saved = state.params[paramsKey(model.id, runnerId)]) {
        const runnerDef = getRunner(runnerId);
        const options = { model, profile: profile || 'dedicated' };
        if (override) return runnerDef.normalizeParams(override, options);
        if (saved) {
            // Saved values that no longer validate (another profile, an older
            // parameter set) give way to the defaults instead of blocking the model.
            try {
                return runnerDef.normalizeParams(saved, options);
            } catch (error) {
                const key = paramsKey(model.id, runnerId);
                if (!resetParams.has(key)) {
                    resetParams.add(key);
                    log.append('controller', `saved parameters for ${model.id} on ${runnerId} no longer apply (${error.message}); using the defaults`);
                }
            }
        }
        return runnerDef.normalizeParams({}, options);
    }

    function publicDeployment() {
        const deployment = state.deployment;
        if (!deployment) return null;
        const { apiKey: _hidden, ...rest } = deployment;
        return structuredClone(rest);
    }

    // ---------------------------------------------------------------- reads

    // Admission for the Run form's current values, before anything is saved
    // or downloaded. Parameter errors come back as data for the form.
    async function previewRun({ modelId, runnerId, params } = {}, snap) {
        const model = findModel(modelId);
        const definition = getRunner(runnerId);
        const source = sourceFor(model, definition);
        if (!definition.supported || !source || !availabilityOf(definition).available) {
            return { modelId, runnerId, params: null, context: null,
                admission: admitHere({ definition, model, source, params: {}, snap }) };
        }
        const gate = gateOf(definition);
        if (!gate.enabled) return { modelId, runnerId, params: null, context: null, admission: disabledAdmission(gate) };
        // A runner with no policy for the cpu profile has no parameters to preview: say why, not a ParamError.
        const refusal = profileRefusal(definition);
        if (refusal) {
            return { modelId, runnerId, params: null, context: null,
                admission: { status: 'incompatible', reason: refusal, estimate: { isEstimate: true }, warnings: [] } };
        }
        let normalized;
        try {
            normalized = effectiveParams(model, runnerId, params && typeof params === 'object' ? params : undefined);
        } catch (error) {
            return { modelId, runnerId, error: error.message, field: error.details?.field ?? null };
        }
        const acquisition = await acquisitionOf(source);
        return {
            modelId,
            runnerId,
            params: normalized,
            context: definition.describeContext ? definition.describeContext(normalized, { model, profile: profile || 'dedicated' }) : null,
            acquisition,
            admission: admitHere({ definition, model, source, params: normalized, snap, remainingDownloadBytes: acquisition.bytesNeeded }),
        };
    }

    async function overview({ preview = null } = {}) {
        // One snapshot decides the profile and is admitted against, so the
        // parameters shown are normalized for the rules that admit them.
        const snap = await takeSnapshot();
        commitProfile(snap);
        const runnerList = [];
        // Why each runner has no policy for the decided profile (null while undecided).
        const refusals = {};
        for (const definition of Object.values(runners)) {
            // A runner for other profiles is not shown: a GPU host sees exactly the runners it always did.
            if (!offeredIn(definition, profile)) continue;
            const gate = gateOf(definition);
            const availability = availabilityOf(definition);
            // A runner the image lacks is not probed: the probe would only say
            // "Executable not found at …", and the platform reason is the cause.
            const info = availability.available
                ? await runnerInfo(definition.id)
                : { installed: false, version: null, reason: availability.reason };
            const summary = runnerSummary(definition, profile);
            refusals[definition.id] = summary.profileUnsupportedReason ?? null;
            runnerList.push({
                ...summary,
                ...(availability.available ? {} : { supported: false, unsupportedReason: availability.reason }),
                ...info,
                enabled: gate.enabled,
                ...(gate.enabled ? {} : { disabledReason: gate.reason }),
                ...(installable(definition.id) ? { install: await installInfo(definition.id) } : {}),
            });
        }
        const models = [];
        const listed = catalog();
        // One walk of /shared for every file the listed models could copy from there.
        const sizes = listed.flatMap((model) => Object.values(model.sources)
            .flatMap((source) => (source && storeFor(source).sizes ? storeFor(source).sizes(source) : [])));
        const index = sharedModelsRoot && sizes.length ? await walkShared(fs, [sharedModelsRoot], sizes) : new Map();
        for (const model of listed) {
            // One entry per weight format: runners that read it share the download.
            const weights = {};
            for (const [format, source] of Object.entries(model.sources)) {
                weights[format] = {
                    label: WEIGHT_FORMATS[format]?.label || format,
                    size: source?.size ?? null,
                    download: await weightsState(source),
                    acquisition: await acquisitionOf(source, index),
                    // The runners that can read them here: not one this deployment's operator left off.
                    runners: Object.values(runners)
                        .filter((definition) => offeredIn(definition, profile) && definition.supported && availabilityOf(definition).available
                            && gateOf(definition).enabled && definition.weightFormat === format)
                        .map((definition) => definition.id),
                };
            }
            const perRunner = {};
            for (const definition of Object.values(runners)) {
                const source = sourceFor(model, definition);
                if (!source || !offeredIn(definition, profile)) continue;
                let params = null;
                let paramError = null;
                const usable = definition.supported && availabilityOf(definition).available;
                // A runner with no policy for the profile has no parameters to show, and says why, not a ParamError.
                const refusal = refusals[definition.id];
                if (usable && !refusal) {
                    try { params = effectiveParams(model, definition.id); } catch (error) { paramError = error.message; }
                }
                const disk = definition.supported ? weights[definition.weightFormat].download : null;
                const remaining = definition.supported ? weights[definition.weightFormat].acquisition.bytesNeeded : 0;
                const gate = gateOf(definition);
                perRunner[definition.id] = {
                    format: definition.weightFormat,
                    size: source?.size ?? null,
                    download: disk,
                    params,
                    context: params && definition.describeContext ? definition.describeContext(params, { model, profile: profile || 'dedicated' }) : null,
                    admission: !availabilityOf(definition).available
                        ? admitHere({ definition, model, source, params: {}, snap })
                        : !gate.enabled ? disabledAdmission(gate) : refusal
                        ? { status: 'incompatible', reason: refusal, estimate: { isEstimate: true }, warnings: [] } : paramError
                        ? { status: 'incompatible', reason: paramError, estimate: {}, warnings: [] }
                        : admitHere({ definition, model, source, params: params || {}, snap, remainingDownloadBytes: remaining }),
                };
            }
            models.push({
                id: model.id,
                displayName: model.displayName,
                description: model.description,
                license: model.license,
                architecture: model.architecture,
                totalParams: model.totalParams,
                activeParams: model.activeParams,
                contextLength: model.contextLength,
                seed: model.seed,
                sources: model.sources,
                validated: model.validated[profile] || {},
                weights,
                runners: perRunner,
            });
        }
        return {
            profile,
            // Why: the cause and reason of the decision (null while undecided).
            profileDecision: publicDecision(),
            limits: limitsFor(snap),
            hardware: snap,
            runners: runnerList,
            // Registry entries this catalog cannot read, or that a seed's id hides; remove them and add the model again.
            unsupportedModels: unsupportedRegistryEntries(state.registry, seedCatalog),
            models,
            deployment: publicDeployment(),
            gatewayModel: 'soul_gateway/local-llms/local-llm/default',
            ...(preview ? { preview: await previewRun(preview, snap) } : {}),
        };
    }

    async function status({ sinceSeq = 0 } = {}) {
        // One snapshot serves the profile decision and the GPU the status shows.
        let snap = null;
        let stopped = false;
        try {
            snap = await takeSnapshot();
        } catch (error) {
            stopped = isStop(error);
        }
        if (!stopped) commitProfile(snap);
        const deployment = publicDeployment();
        const lines = log.since(Number(sinceSeq) || 0);
        const gpu = snap?.gpu ?? null;
        const definition = deployment && Object.hasOwn(runners, deployment.runnerId) ? runners[deployment.runnerId] : null;
        const parseReport = definition?.parseReport || parseRunnerReport;
        return {
            phase: deployment?.phase || 'idle',
            deployment,
            logs: lines,
            nextSeq: log.seq,
            gpu,
            runnerReport: parseReport(log.all().filter((line) => line.seq > runnerLogStart), { profile: deployment?.profile ?? profile }),
            context: describeDeploymentContext(definition, deployment),
            lastCompletion,
            profile: profile || null,
            profileDecision: publicDecision(),
            memoryGuard: guardStats ? { ...guardStats } : null,
        };
    }

    // A deployment recorded by an earlier controller may hold parameters this
    // one does not accept: its context is then unknown, not an error.
    function describeDeploymentContext(definition, deployment) {
        if (!deployment || !definition?.describeContext) return null;
        try {
            return definition.describeContext(deployment.params, { profile: deployment.profile || profile || 'dedicated' });
        } catch {
            return null;
        }
    }

    function recordCompletion(stats = {}) {
        const deployment = state.deployment;
        if (!deployment || deployment.phase !== 'ready') return { recorded: false };
        const count = (value) => (Number.isFinite(value) && value >= 0 && value < 1e9 ? value : null);
        lastCompletion = Object.freeze({
            at: now().toISOString(),
            deploymentId: deployment.id,
            runnerId: deployment.runnerId,
            modelId: deployment.modelId,
            promptTokens: count(stats.promptTokens),
            completionTokens: count(stats.completionTokens),
            promptTokensPerSecond: count(stats.promptTokensPerSecond),
            generationTokensPerSecond: count(stats.generationTokensPerSecond),
            source: stats.source === 'runner timings' ? 'runner timings' : 'usage',
        });
        const speed = lastCompletion.generationTokensPerSecond;
        log.append('controller', `completion served: ${lastCompletion.completionTokens ?? '?'} tokens`
            + `${speed === null ? '' : ` at ${speed.toFixed(1)} tokens/s`}`);
        return { recorded: true };
    }

    function chatTarget() {
        const deployment = state.deployment;
        if (!deployment || deployment.phase !== 'ready' || !runner?.running) {
            throw new LocalLlmError('not_ready', 'No local model is ready. Run one from Settings > Agents > Local LLMs.');
        }
        const definition = getRunner(deployment.runnerId);
        return {
            runnerId: deployment.runnerId,
            modelId: deployment.modelId,
            baseUrl: `http://127.0.0.1:${runner.port}`,
            apiKey: runner.apiKey || null,
            model: definition.chatModel(deployment),
            // The request options depend on the profile (Ollama on the cpu profile hides the GPU layers).
            requestOptions: definition.requestOptions?.(deployment.params, { profile: deployment.profile ?? undefined }) ?? null,
            // The chat responder sizes its completion budget by it (DS001). Only the cpu profile
            // has a budget of its own, so the other profiles' chat target stays what it was.
            ...(deployment.profile === 'cpu' ? { profile: 'cpu' } : {}),
        };
    }

    // ------------------------------------------------------------ pipelines

    async function waitForHttp(url, { headers = {}, signal, process, accept = (response) => response.ok } = {}) {
        const deadline = Date.now() + readyTimeoutMs;
        while (Date.now() < deadline) {
            if (signal?.aborted) throw new LocalLlmError('aborted', 'Stopped while starting.');
            if (process && !process.running) throw new LocalLlmError('runner_exited', 'The runner exited while starting.');
            try {
                // Each probe has its own deadline, so a runner that accepts the
                // connection and never answers cannot outlast readyTimeoutMs.
                const probeSignal = signal ? AbortSignal.any([signal, AbortSignal.timeout(PROBE_TIMEOUT_MS)]) : AbortSignal.timeout(PROBE_TIMEOUT_MS);
                const response = await fetchImpl(url, { headers, signal: probeSignal });
                if (accept(response)) return response;
            } catch (error) {
                if (signal?.aborted) throw new LocalLlmError('aborted', 'Stopped while starting.');
            }
            await sleep(pollMs, signal);
        }
        throw new LocalLlmError('start_timeout', `The runner did not become ready within ${readyTimeoutMs / 1000} s.`);
    }

    function runnerEnv(launchEnv) {
        // Only what the runner needs: never the agent's secrets or tokens. The
        // CUDA driver's JIT cache stays in the container's own filesystem, never
        // in /data, which other workspace processes can write (DS004).
        return {
            PATH: env.PATH || '/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin',
            HOME: path.join(dataDir, 'home'),
            LANG: 'C.UTF-8',
            CUDA_CACHE_PATH: cudaCachePath,
            ...launchEnv,
        };
    }

    // A helper step of a runner's start-up or watchdog (for example LM
    // Studio's `lms import`): its own process group, ended by `signal`, with
    // only the environment the adapter gives it. `json` returns the step's
    // last output line, parsed.
    function stepRunner(signal) {
        // `quiet` steps (a watchdog's routine check) are logged only when they fail.
        return async ({ label, command, args = [], env: stepEnv = {}, cwd = '/', json = false, quiet = false }) => {
            const result = await runTool({ command, args, env: stepEnv, cwd, signal });
            if (result.aborted) throw new LocalLlmError('aborted', 'Stopped while starting.');
            if (result.code !== 0) {
                const tail = result.output.trim().split('\n').slice(-3).join(' | ').slice(0, 600);
                throw new LocalLlmError('runner_step_failed', `${label} failed (exit ${result.code ?? result.signal}): ${tail}`);
            }
            if (!quiet) log.append('controller', `${label}: done`);
            if (!json) return { output: result.output };
            try {
                return JSON.parse(result.output.trim().split('\n').at(-1));
            } catch {
                throw new LocalLlmError('runner_step_failed', `${label} returned no result.`);
            }
        };
    }

    function launchRunner(deployment, launch, apiKey, port, { definition = null, runnerDir = null } = {}) {
        fs.mkdirSync(path.join(dataDir, 'home'), { recursive: true });
        log.addSecret(apiKey);
        runnerLogStart = log.seq;
        lastCompletion = null;
        log.append('controller', `starting ${launch.command} ${launch.args.join(' ')}`);
        const process = startRunner({
            command: launch.command, args: launch.args, env: runnerEnv(launch.env), cwd: launch.cwd || '/', log,
            filter: launch.outputFilter || null,
        });
        runner = {
            pid: process.pid,
            port,
            apiKey,
            deploymentId: deployment.id,
            exited: process.exited,
            stop: (options) => process.stop(options),
            kill: () => (typeof process.kill === 'function' ? process.kill() : process.stop({ graceMs: 0 })),
            get running() { return process.running; },
        };
        const current = runner;
        const cancelGuard = startMemoryGuard(deployment, current);
        process.exited.then((result) => {
            cancelGuard();
            current.stopWatchdog?.();
            // The adapter's clean-up once its process group is gone (LM Studio
            // deletes its server log and reaps anything left in its copy).
            try {
                definition?.afterExit?.({ runnerDir, log: (line) => log.append('controller', line) });
            } catch (error) {
                log.append('controller', `clean-up after ${definition.id} failed: ${error.message}`);
            }
            if (runner !== current) return;
            runner = null;
            if (memoryStop?.deploymentId === deployment.id) return;
            const live = state.deployment;
            if (live?.id === deployment.id && ['starting', 'loading', 'ready', 'downloading'].includes(live.phase)
                && !draining && live.phase !== 'stopping') {
                const tail = log.since(0, 5).map((entry) => entry.line).join(' | ');
                setPhase('error', {
                    error: `The runner exited unexpectedly (code ${result.code ?? 'none'}, signal ${result.signal ?? 'none'}). ${tail}`,
                });
            }
        });
        return runner;
    }

    // The memory guard. Dedicated profile: the backstop behind a RAM-floor
    // admission (vLLM with CPU offload, whose estimate comes from two runs on
    // one machine): while this runner loads and runs, sample MemAvailable;
    // below the floor, stop it at once. Unified profile (DS005): every runner,
    // sampled every 250 ms in every phase; below the floor, or when
    // /proc/meminfo cannot be read, or under heavy memory pressure with
    // MemAvailable already low, the runner's process group is killed at once.
    // The cpu profile is guarded the same way, against the floor its admission
    // recorded (a share of the pool, not the unified 8 GiB). It is a backstop,
    // not protection: one load allocates faster than any sampling can follow,
    // so admission must cover the whole known allocation.
    function startMemoryGuard(deployment, current) {
        const unified = deployment.profile === 'unified';
        // The pool profiles: one pool of memory, every runner watched in every phase.
        const pooled = unified || deployment.profile === 'cpu';
        const floor = unified ? UNIFIED.floorBytes
            : deployment.profile === 'cpu' ? deployment.admission?.estimate?.floorBytes
                : deployment.admission?.estimate?.ramFloorBytes;
        if (!Number.isFinite(floor) || floor <= 0) return () => {};
        let timer = null;
        let cancelled = false;
        // Under a finite container memory limit the guard also samples its
        // headroom, every 250 ms; two consecutive unreadable samples stop the
        // runner's process group (budget_unreadable). Without one, unchanged.
        const sampleBudget = () => {
            try { return readBudget(); } catch { return null; }
        };
        const budgeted = Number.isSafeInteger(sampleBudget()?.finiteMemoryBytes);
        let unreadableBudgetSamples = 0;
        if (pooled) guardStats = { floorBytes: floor, minAvailableBytes: null, maxPressureAvg10: null, samples: 0 };
        const schedule = () => {
            if (cancelled) return;
            const ready = state.deployment?.id === deployment.id && state.deployment.phase === 'ready';
            timer = setTimeout(tick, pooled || budgeted ? unifiedGuardMs : (ready ? memoryGuardReadyMs : memoryGuardLoadMs));
            timer.unref?.();
        };
        const breach = (message, { kill = pooled } = {}) => {
            cancelled = true;
            stopForMemory(deployment, current, message, { kill }).catch(() => {});
        };
        // The available memory within the container budget, or undefined when
        // the budget's current use could not be read (counted, then stopped).
        const withinBudget = (available) => {
            if (!budgeted) return available;
            const observation = sampleBudget();
            if (observation?.memoryReadState !== 'known') {
                unreadableBudgetSamples += 1;
                return undefined;
            }
            unreadableBudgetSamples = 0;
            if (!Number.isSafeInteger(observation.finiteMemoryBytes) || !Number.isFinite(available)) return available;
            return Math.min(available, observation.finiteMemoryBytes, observation.headroomBytes);
        };
        const budgetUnreadable = () => {
            if (unreadableBudgetSamples < 2) {
                schedule();
                return;
            }
            breach(`stopped: ${BUDGET_UNREADABLE}: the container memory limit's current use could not be read twice in a row, `
                + 'so the runner cannot be watched', { kill: true });
        };
        const tick = () => {
            // Pool profiles: watched until the process exits, through a graceful Stop
            // or Replace too (the exit cancels the guard). Dedicated: while current.
            if (cancelled || (!pooled && runner !== current)) return;
            let available = null;
            try {
                available = readMemory().availableBytes;
            } catch {}
            // An unreadable /proc/meminfo stops a pooled runner at once, before the budget is sampled.
            if (pooled && !Number.isFinite(available)) {
                breach('stopped: host memory cannot be read (/proc/meminfo), so the runner cannot be watched');
                return;
            }
            available = withinBudget(available);
            if (available === undefined) {
                budgetUnreadable();
                return;
            }
            if (pooled) {
                const stats = guardStats;
                const pressure = readPressure();
                stats.samples += 1;
                stats.minAvailableBytes = stats.minAvailableBytes === null ? available : Math.min(stats.minAvailableBytes, available);
                if (Number.isFinite(pressure)) stats.maxPressureAvg10 = Math.max(stats.maxPressureAvg10 ?? 0, pressure);
                if (available < floor) {
                    breach(`stopped: host memory below the floor (${gib(available)} available, ${gib(floor)} required)`);
                    return;
                }
                if (Number.isFinite(pressure) && pressure >= UNIFIED.psiStopAvg10 && available < 2 * floor) {
                    breach(`stopped: memory pressure ${pressure.toFixed(1)} % with ${gib(available)} available`);
                    return;
                }
                schedule();
                return;
            }
            if (Number.isFinite(available) && available < floor) {
                breach(`stopped: host memory below the floor (${gib(available)} available, ${gib(floor)} required)`);
                return;
            }
            schedule();
        };
        schedule();
        return () => {
            cancelled = true;
            clearTimeout(timer);
        };
    }

    async function stopForMemory(deployment, current, message, { kill = false } = {}) {
        memoryStop = { deploymentId: deployment.id, message };
        log.append('controller', message);
        // The host is short of memory now. Unified: the whole process group is
        // killed at once (a runner starved of memory has nothing to save).
        // Dedicated: SIGKILL follows SIGTERM within the drain's grace, not a Stop's.
        try {
            if (kill) await current.kill();
            else await current.stop({ graceMs: DRAIN_RUNNER_GRACE_MS });
        } catch {}
        // Once the agent drains, the drain settles the state; the log keeps the message.
        if (draining || queue.closed) return;
        try {
            await queue.run(async () => {
                if (runner === current) runner = null;
                if (job?.deploymentId === deployment.id) await job.promise.catch(() => {});
                // A Stop or a replace that came in meanwhile has already settled it.
                const live = state.deployment;
                if (live?.id === deployment.id && live.phase !== 'idle') setPhase('error', { error: message, runner: null });
            });
        } catch (error) {
            log.append('controller', `memory guard: the stop was not recorded (${error.message})`);
        }
    }

    function updateProgress(progress) {
        const deployment = state.deployment;
        if (!deployment) return;
        deployment.download = {
            bytes: progress.bytes,
            total: progress.total,
            rate: Math.round(progress.rate || 0),
            etaSeconds: progress.etaSeconds ?? null,
            transferred: progress.transferred || 0,
        };
        if (progress.phase && TRANSFER_PHASES.has(deployment.phase)) {
            // Copying from /shared and hashing show as their own phases (C12).
            deployment.phase = progress.phase;
        } else if (progress.total && progress.bytes >= progress.total && deployment.phase === 'downloading') {
            deployment.phase = 'verifying';
        }
        save();
    }

    // Admission again, from a fresh snapshot, under the profile the Run was
    // admitted with. The job's `signal` stops the snapshot's queries at once.
    async function recheckAdmission(deployment, model, { remainingDownloadBytes = 0, signal = null } = {}) {
        const definition = getRunner(deployment.runnerId);
        const source = deployment.artifact;
        const snap = await takeSnapshot(signal);
        const result = admitHere({
            definition, model, source, params: deployment.params, snap, remainingDownloadBytes, selected: deployment.profile,
        });
        deployment.admission = result;
        if (result.status !== 'ok') {
            throw new LocalLlmError(`admission_${result.status.replace('-', '_')}`, result.reason, { admission: result });
        }
    }

    // What a runner's start-up pipeline may use. The adapter launches its
    // process through `launch` (an argv and a minimal environment), waits
    // with `waitForHttp`, and may report phases and progress for weights it
    // fetches itself; the controller owns the process and the state.
    function startContext(deployment, model, signal, weights, runnerDir = null) {
        const definition = getRunner(deployment.runnerId);
        const port = ports[deployment.runnerId];
        const apiKey = definition.apiKey ? apiKeyFactory() : null;
        return Object.freeze({
            runner: definition,
            model,
            // The profile the Run was normalized and admitted under.
            profile: deployment.profile,
            params: deployment.params,
            artifact: deployment.artifact,
            weights,
            // An on-demand runner's runnable copy (null for runners in the image).
            runnerDir,
            // The admission just re-checked before launch; vLLM sizes its GPU share from it.
            admission: deployment.admission ?? null,
            shmDir,
            port,
            apiKey,
            dataDir,
            signal,
            fetch: fetchImpl,
            store: storeFor(deployment.artifact),
            launch: (spec) => launchRunner(deployment, spec, apiKey, port, { definition, runnerDir }),
            exec: stepRunner(signal),
            waitForHttp: (url, options = {}) => waitForHttp(url, { ...options, signal }),
            setPhase: (phase) => setPhase(phase),
            progress: updateProgress,
            record: (fields) => {
                Object.assign(deployment, fields);
                save();
            },
            recheckAdmission: () => recheckAdmission(deployment, model, { signal }),
            throwIfAborted: () => throwIfAborted(signal),
        });
    }

    // A user entry that says its sizing came from the GGUF header (`sizingSource: 'gguf-header'`): the header of the file
    // just verified, and the entry with what that header gives in place of the stored values, or the entry itself when
    // nothing differs. The stored values came from a file read before the entry was pinned; a repository that moved
    // between the lookup and Add, or a header that lied, shows here. A difference is logged. Typed values, seeds,
    // snapshots and runner-fetched Ollama tags are not re-read. A file whose header is malformed, or cannot be read, is
    // not sized at all. A well-formed header whose architecture this reader cannot use checks nothing: the stored (or
    // defaulted) values stand, a warning says so, and the Run goes on.
    async function verifiedGgufSizing(model, source, file) {
        if (model.seed || model.sizingSource !== 'gguf-header' || source?.type !== 'huggingface') return model;
        let sizing;
        try {
            sizing = ggufSizing((await readGgufHeaderFile(file)).kv);
        } catch (error) {
            throw new LocalLlmError(error?.code === 'invalid_gguf' ? 'invalid_gguf' : 'sizing_unreadable',
                `The downloaded ${path.basename(file)} has no readable GGUF header (${error.message}), so the sizing stored for ${model.id} cannot be checked.`);
        }
        if (sizing.arch === null) {
            log.append('controller', `${model.id}: the GGUF header of the verified file cannot be used to check the sizing stored when the model `
                + `was added (${sizing.notes[0]}); the stored values stand`);
            return Object.freeze({ ...model, sizingUnchecked: sizing.notes[0] });
        }
        const stored = {
            contextLength: model.contextLength ?? null,
            layers: model.memory?.layers ?? null,
            kvBytesPerToken: model.memory?.kvBytesPerToken ?? null,
            architecture: model.architecture,
        };
        const verified = {
            contextLength: sizing.contextLength,
            layers: sizing.layers,
            kvBytesPerToken: sizing.kvBytesPerToken,
            architecture: sizing.architecture,
        };
        const changed = Object.keys(verified).filter((key) => verified[key] !== stored[key]);
        if (changed.length === 0) return Object.freeze({ ...model, sizingVerified: true });
        const say = (value) => (value === null ? 'none' : String(value));
        log.append('controller', `${model.id}: the GGUF header of the verified file differs from the sizing stored when the model was added `
            + `(${changed.map((key) => `${key} ${say(stored[key])} -> ${say(verified[key])}`).join(', ')}); the verified values are used`);
        return Object.freeze({
            ...model,
            // The admission warnings say the values are the verified file's own.
            sizingVerified: true,
            architecture: verified.architecture,
            contextLength: verified.contextLength ?? undefined,
            memory: Object.freeze({ ...model.memory, layers: verified.layers ?? undefined, kvBytesPerToken: verified.kvBytesPerToken ?? undefined }),
        });
    }

    // The verified file changed the model's training context, which decides what a default context is and what an
    // explicit one may be. The parameters are worked out again from what the Run asked for (and what was saved before
    // it), not from the result already stored: a default context is then lowered to the verified one, and an explicit
    // one above it is refused here, before the runner starts. The admissions and the launch read `deployment.params`.
    function renormalizeParams(deployment, model, { requested, saved }) {
        try {
            deployment.params = effectiveParams(model, deployment.runnerId, requested, saved);
        } catch (error) {
            if (error?.code !== 'invalid_params') throw error;
            throw new LocalLlmError('invalid_params', `The downloaded file's header gives a training context of ${model.contextLength ?? 'no'} tokens. ${error.message}`,
                error.details);
        }
        state.params[paramsKey(model.id, deployment.runnerId)] = deployment.params;
        save();
    }

    async function runPipeline(deployment, model, signal, requestedParams = {}) {
        const definition = getRunner(deployment.runnerId);
        const store = storeFor(deployment.artifact);
        let weights = null;
        if (store.fetchedBy === 'controller') {
            setPhase('downloading');
            const fetched = await store.fetch({ artifact: deployment.artifact, signal, onProgress: updateProgress });
            throwIfAborted(signal);
            deployment.download = {
                ...(deployment.download || {}),
                bytes: fetched.bytes,
                total: fetched.bytes,
                transferred: fetched.bytesTransferred,
            };
            weights = { path: fetched.path };
            // Where every file and shard came from (C12, R7).
            deployment.provenance = fetched.provenance || [];
            for (const entry of deployment.provenance) log.append('controller', describeProvenance(entry));
            for (const note of fetched.notes || []) log.append('controller', note);
            // Every file of the download (each shard of a split GGUF); a snapshot directory is skipped.
            for (const file of fetched.files || [fetched.path]) {
                try {
                    if (fs.statSync(file).isFile() && !dropCache(file)) {
                        log.append('controller', `could not drop ${path.basename(file)} from the page cache (dd iflag=nocache)`);
                    }
                } catch {}
            }
            // The sizing read from the GGUF header at lookup is unverified until the download is: read it again from the
            // verified file, so this admission, the last one and the runner's start use what the file says (DS002).
            const verified = await verifiedGgufSizing(model, deployment.artifact, fetched.path);
            if (verified.contextLength !== model.contextLength) renormalizeParams(deployment, verified, requestedParams);
            model = verified;
            // An early refusal before the runner is prepared; the check that counts is the last one, below.
            await recheckAdmission(deployment, model, { signal });
            throwIfAborted(signal);
        }
        setPhase('starting');
        let runnerDir = null;
        if (installable(definition.id)) {
            log.append('controller', `preparing ${definition.id} from its verified cache`);
            const built = await installer.ensureRunnable(definition.id, { signal });
            runnerDir = installer.pathsFor(installer.entryFor(definition.id)).runDir;
            if (built.rebuilt) log.append('controller', `rebuilt ${definition.id} in ${built.seconds.toFixed(1)} s (${built.bytes} bytes)`);
            throwIfAborted(signal);
        }
        if (store.recheck) {
            // The last check before loading: an unchanged stat, or the bytes hashed again (C12); hashing shows as verifying.
            const checked = await store.recheck({
                artifact: deployment.artifact,
                signal,
                onProgress: (progress) => {
                    if (progress.phase === 'verifying' && state.deployment?.phase === 'starting') setPhase('verifying');
                    updateProgress(progress);
                },
            });
            if (state.deployment?.phase === 'verifying') setPhase('starting');
            for (const note of checked.notes || []) log.append('controller', note);
            throwIfAborted(signal);
        }
        // The last admission, after the runner is prepared and the weights
        // checked for the last time (which can take minutes of hashing):
        // memory or disk taken meanwhile refuses the start instead of
        // launching into it. Weights a runner fetches itself still need disk.
        const remainingDownloadBytes = store.fetchedBy === 'controller'
            ? 0 : (await acquisitionOf(deployment.artifact, null, signal)).bytesNeeded;
        throwIfAborted(signal);
        await recheckAdmission(deployment, model, { remainingDownloadBytes, signal });
        throwIfAborted(signal);
        const details = await definition.start(startContext(deployment, model, signal, weights, runnerDir));
        throwIfAborted(signal);
        if (!runner || runner.deploymentId !== deployment.id) {
            throw new LocalLlmError('runner_exited', 'The runner is not running after its start-up.');
        }
        setPhase('ready', { runner: { pid: runner.pid, port: runner.port, startedAt: now().toISOString(), ...(details || {}) } });
        startWatchdog(deployment, runner, definition, runnerDir);
    }

    // An adapter's periodic check while its runner is ready (LM Studio loads
    // any model a request names, so it unloads anything but ours). A check's
    // note is logged; a failed check is logged and does not stop the runner.
    function startWatchdog(deployment, current, definition, runnerDir) {
        const watchdog = definition.watchdog;
        if (!watchdog || typeof watchdog.check !== 'function') return;
        const abort = new AbortController();
        let timer = null;
        const tick = async () => {
            if (abort.signal.aborted || runner !== current || state.deployment?.id !== deployment.id || state.deployment.phase !== 'ready') return;
            try {
                const note = await watchdog.check({
                    deployment: publicDeployment(), modelId: deployment.modelId, port: current.port, runnerDir,
                    exec: stepRunner(abort.signal), signal: abort.signal,
                });
                if (note && !abort.signal.aborted) log.append('controller', `${definition.id} watchdog: ${note}`);
            } catch (error) {
                if (!abort.signal.aborted) log.append('controller', `${definition.id} watchdog failed: ${error.message}`);
            }
            if (!abort.signal.aborted) {
                timer = setTimeout(tick, watchdog.intervalMs || 30_000);
                timer.unref?.();
            }
        };
        timer = setTimeout(tick, watchdog.intervalMs || 30_000);
        timer.unref?.();
        current.stopWatchdog = () => {
            abort.abort();
            clearTimeout(timer);
        };
    }

    function startJob(deployment, model, requestedParams) {
        const controller = new AbortController();
        const current = {
            deploymentId: deployment.id,
            abort: controller,
            cancelReason: null,
            promise: null,
        };
        current.promise = runPipeline(deployment, model, controller.signal, requestedParams)
            .catch(async (error) => {
                const live = state.deployment;
                if (!live || live.id !== deployment.id) return;
                const aborted = controller.signal.aborted
                    || (error instanceof DownloadError && error.code === 'ABORTED');
                if (aborted) {
                    const wasTransfer = TRANSFER_PHASES.has(live.phase);
                    if (runner && runner.deploymentId === deployment.id) {
                        const stopping = runner;
                        runner = null;
                        await stopping.stop({ graceMs: runnerGraceMs() });
                    }
                    const pausing = ['cancel', 'drain'].includes(current.cancelReason) && wasTransfer;
                    const phase = pausing ? 'paused' : 'idle';
                    const reason = current.cancelReason === 'drain'
                        ? 'The agent restarted during the download; press Run to resume.'
                        : 'Download cancelled; press Run to resume.';
                    setPhase(phase, { runner: null, pausedReason: pausing ? reason : null });
                    return;
                }
                if (error instanceof DownloadError && ['PAUSED_ENOSPC', 'NETWORK'].includes(error.code)) {
                    setPhase('paused', { error: error.message, pausedReason: error.message });
                    return;
                }
                if (runner && runner.deploymentId === deployment.id) {
                    const stopping = runner;
                    runner = null;
                    await stopping.stop({ graceMs: runnerGraceMs() });
                }
                const guarded = memoryStop?.deploymentId === deployment.id ? memoryStop.message : null;
                setPhase('error', { error: guarded || error.message, runner: null });
            })
            .finally(() => {
                if (job === current) job = null;
            });
        job = current;
    }

    async function stopEverything(reason) {
        if (job) {
            job.cancelReason = reason;
            job.abort.abort();
            await job.promise;
        }
        if (runner) {
            const stopping = runner;
            runner = null;
            if (state.deployment && ACTIVE_PHASES.has(state.deployment.phase)) setPhase('stopping');
            await stopping.stop({ graceMs: runnerGraceMs() });
        }
        if (state.deployment && ACTIVE_PHASES.has(state.deployment.phase)) {
            setPhase(reason === 'cancel' && TRANSFER_PHASES.has(state.deployment.phase) ? 'paused' : 'idle', { runner: null });
        }
    }

    // ------------------------------------------------------------- commands

    function run({ requestId, modelId, runnerId, params, replace = false } = {}) {
        // The Run's generation is taken when it is submitted, before it waits in the queue:
        // a Stop or Cancel invoked after this, even while the Run still waits, invalidates it.
        const generation = stopGeneration;
        const submission = { requestId };
        submittedRuns.add(submission);
        const stale = () => generation !== stopGeneration;
        const cancelled = () => new LocalLlmError('cancelled', 'The Run was stopped before it started.');
        const command = queue.run(async () => {
            if (typeof requestId !== 'string' || !REQUEST_ID_RE.test(requestId)) {
                throw new LocalLlmError('invalid_request', 'requestId must be 8-128 letters, digits, dash or underscore.');
            }
            // The browser client may retry a timed-out call: the same request is a no-op.
            if (acceptedRequest(state.requests, requestId)) {
                return { duplicate: true, deployment: publicDeployment() };
            }
            // Submitted before a Stop or Cancel: nothing is read or recorded.
            if (stale()) throw cancelled();
            // Until its job starts, a Run can be stopped: Stop, Cancel and drain abort `planning` before
            // they queue behind it (F5), which ends a hardware snapshot or the /shared walk in progress.
            // The Run then fails with `cancelled` and records nothing, so the same request can be sent again.
            const planAbort = new AbortController();
            planning = planAbort;
            try {
                return await admitRun(planAbort.signal, { requestId, modelId, runnerId, params, replace, stale });
            } catch (error) {
                // A drain met in a snapshot keeps its answer; any other stop of a pending Run is `cancelled`.
                if (error.code === 'shutting_down') throw error;
                if (planAbort.signal.aborted || stale()) throw cancelled();
                throw error;
            } finally {
                if (planning === planAbort) planning = null;
            }
        });
        const settled = () => {
            submittedRuns.delete(submission);
        };
        command.then(settled, settled);
        return command;
    }

    // A Stop or Cancel invalidates every Run submitted so far. It reports
    // whether that cancelled new work: a Run whose request id is valid and not
    // already accepted. A queued retry of an accepted request is answered as a
    // duplicate, never cancelled, so it is not new work.
    function invalidateSubmittedRuns() {
        const pending = [...submittedRuns].some(({ requestId }) => typeof requestId === 'string'
            && REQUEST_ID_RE.test(requestId) && !acceptedRequest(state.requests, requestId));
        stopGeneration += 1;
        submittedRuns.clear();
        planning?.abort();
        return pending;
    }

    // A Run from its profile to its job's start. `signal` is the pending Run's:
    // after every step that waits it is checked, and the last check comes
    // right before anything is recorded; from there to the job's start nothing waits.
    async function admitRun(signal, { requestId, modelId, runnerId, params, replace, stale }) {
        const checkpoint = () => {
            if (signal.aborted || stale()) throw new LocalLlmError('aborted', 'Stopped before the Run started.');
        };
        await requireProfile(signal);
        checkpoint();
        const model = findModel(modelId);
        const definition = getRunner(runnerId);
        const source = sourceFor(model, definition);
        if (!definition.supported) {
            const result = admit({ runner: definition, model, source, params: {}, snapshot: {} });
            throw new LocalLlmError('runner_unsupported', result.reason);
        }
        // Before anything is downloaded: a runner this image does not have (DS005).
        const availability = availabilityOf(definition);
        if (!availability.available) throw new LocalLlmError('runner_unavailable', availability.reason, { runner: runnerId });
        // A runner with no policy for the profile (every runner but llama.cpp on the CPU) is refused before any lookup or download.
        const refusal = profileRefusal(definition);
        if (refusal) throw new LocalLlmError('runner_unavailable', refusal, { runner: runnerId });
        assertEnabled(definition);
        if (!source) throw new LocalLlmError('no_source', `${model.displayName} has no ${definition.displayName} source.`);
        if (installJob?.runnerId === runnerId) {
            throw new LocalLlmError('busy', `${definition.displayName} is being installed; run it once the install finishes.`);
        }
        if (installable(runnerId) && !(await installer.describe(runnerId)).installed) {
            throw new LocalLlmError('runner_not_installed', `${definition.displayName} is not installed; install it first.`);
        }
        checkpoint();
        const store = storeFor(source);
        if (!store.isPinned(source)) {
            throw new LocalLlmError('unpinned', 'The model source is not pinned to a commit; update the model entry.');
        }
        // What this Run asked for and what was saved before it: kept so the parameters can be worked out again if the verified
        // file changes the model's training context.
        // null (not undefined) says there was nothing saved, so the default of effectiveParams does not pick up this Run's own result.
        const requestedParams = { requested: params, saved: state.params[paramsKey(model.id, runnerId)] ?? null };
        const normalized = effectiveParams(model, runnerId, params);
        const current = state.deployment;
        if (current && (ACTIVE_PHASES.has(current.phase) || job)) {
            if (!replace) {
                throw new LocalLlmError('busy', `${current.modelId} on ${current.runnerId} is ${current.phase}; `
                    + 'stop it first or run with replace.');
            }
            await stopEverything('replace');
            checkpoint();
        }
        const disk = await weightsState(source);
        checkpoint();
        // Planning walks /shared; a walk with nothing left to read (an empty or missing root) returns normally after an abort.
        const acquisition = await acquisitionOf(source, null, signal);
        checkpoint();
        const snap = await takeSnapshot(signal);
        // The last check: from here to the job's start nothing waits, so a Stop or Cancel cannot fall in between.
        checkpoint();
        const admission = admitHere({
            definition, model, source, params: normalized, snap, remainingDownloadBytes: acquisition.bytesNeeded,
        });
        if (admission.status !== 'ok') {
            throw new LocalLlmError(`admission_${admission.status.replace('-', '_')}`, admission.reason, { admission });
        }
        state.params[paramsKey(model.id, runnerId)] = normalized;
        const at = now().toISOString();
        // The job holds this immutable copy: later registry edits cannot redirect it.
        if (draining) {
            throw new LocalLlmError('shutting_down', 'The agent is restarting; run the model again once it is back.');
        }
        state.deployment = {
            id: crypto.randomUUID(),
            requestId,
            modelId: model.id,
            runnerId,
            // Launch, the last admission and the memory guard use this profile.
            profile,
            params: normalized,
            artifact: structuredClone(source),
            phase: store.fetchedBy === 'runner' ? 'starting' : 'downloading',
            admission,
            acquisition: acquisition.files.map(({ file, method, bytesNeeded }) => ({ file, method, bytesNeeded })),
            download: { bytes: disk?.bytes || 0, total: disk?.total ?? null, rate: 0, etaSeconds: null, transferred: 0 },
            error: null,
            pausedReason: null,
            runner: null,
            logSeqStart: log.seq,
            createdAt: at,
            updatedAt: at,
        };
        recordRequest(state.requests, requestId, { deploymentId: state.deployment.id, at });
        save();
        startJob(state.deployment, model, requestedParams);
        return { accepted: true, deployment: publicDeployment() };
    }

    function stop() {
        invalidateSubmittedRuns();
        return queue.run(async () => {
            await stopEverything('stop');
            // Stop also clears a failed or paused deployment (a paused partial stays on disk).
            if (state.deployment && ['error', 'paused'].includes(state.deployment.phase)) {
                setPhase('idle', { runner: null, error: null, pausedReason: null });
            }
            return { deployment: publicDeployment() };
        });
    }

    function cancelDownload() {
        // A Cancel that stops a Run submitted before it (queued, or planning) has done its work: that Run
        // fails with `cancelled` and no download starts, so there may be nothing left to cancel at this command's turn.
        const stoppedRuns = invalidateSubmittedRuns();
        return queue.run(async () => {
            const deployment = state.deployment;
            if (!deployment || !job || !(TRANSFER_PHASES.has(deployment.phase) || deployment.phase === 'starting')) {
                if (stoppedRuns) return { deployment: publicDeployment() };
                throw new LocalLlmError('not_downloading', 'No download is in progress.');
            }
            await stopEverything('cancel');
            return { deployment: publicDeployment() };
        });
    }

    // Weights are named by format, or by a runner, which stands for the
    // format it reads: every runner of that format loses them.
    function weightsFormat({ runnerId, format }) {
        if (format !== undefined) {
            if (typeof format !== 'string' || !Object.hasOwn(WEIGHT_FORMATS, format)) {
                throw new LocalLlmError('invalid_request', `Unknown weight format: ${String(format)}`);
            }
            return format;
        }
        return getRunner(runnerId).weightFormat;
    }

    function deleteWeights({ modelId, runnerId, format } = {}) {
        return queue.run(async () => {
            await currentProfile();
            const model = findAnyModel(modelId);
            const weightFormat = weightsFormat({ runnerId, format });
            const source = model.sources[weightFormat];
            if (!source) throw new LocalLlmError('no_source', `${model.displayName} has no ${weightFormat} weights.`);
            if (inUse(artifactKey(source))) {
                throw new LocalLlmError('in_use', 'These weights are in use; stop the model or cancel the download first.');
            }
            const freed = await storeFor(source).remove(source);
            const deployment = state.deployment;
            if (deployment && artifactKey(deployment.artifact) === artifactKey(source)
                && ['paused', 'error', 'idle'].includes(deployment.phase)) {
                deployment.phase = 'idle';
                deployment.download = { bytes: 0, total: deployment.download?.total ?? null, rate: 0, etaSeconds: null, transferred: 0 };
                save();
            }
            return { freedBytes: freed };
        });
    }

    // A pin in flight ends when the agent drains, like a lookup. The pins that read a registry (an Ollama
    // tag, on the committed cpu profile alone) take the signal; the others read Hugging Face as they always did.
    const pinning = new Set();

    async function pinSources(entry) {
        if (draining) throw new LocalLlmError('shutting_down', 'The agent is restarting; add the model again once it is back.');
        const stop = new AbortController();
        pinning.add(stop);
        try {
            const sources = { ...entry.sources };
            for (const [format, source] of Object.entries(sources)) {
                sources[format] = await storeFor(source).pin(source, { profile, signal: stop.signal });
            }
            return { ...entry, sources };
        } catch (error) {
            if (stop.signal.aborted) throw new LocalLlmError('shutting_down', 'The agent is restarting; add the model again once it is back.');
            throw error;
        } finally {
            pinning.delete(stop);
        }
    }

    // Against every seed and every stored entry, whatever the profile or schema:
    // a user entry never reuses a seed id or another entry's id.
    function assertNewModelId(id) {
        if (seedCatalog.some((model) => model.id === id) || state.registry.some((entry) => entry?.id === id)) {
            throw new LocalLlmError('duplicate_model', `A model with id '${id}' already exists.`);
        }
    }

    function userModelIndex(id) {
        const index = state.registry.findIndex((model) => model.id === id);
        if (index < 0) {
            throw new LocalLlmError(seedCatalog.some((model) => model.id === id) ? 'read_only' : 'unknown_model',
                'Only user models can be changed; seed entries are read-only.');
        }
        return index;
    }

    function assertModelNotInUse(model) {
        for (const source of Object.values(model.sources)) {
            if (inUse(artifactKey(source))) {
                throw new LocalLlmError('in_use', 'This model is in use; stop it or cancel its download first.');
            }
        }
    }

    // Pinning reads Hugging Face metadata only; no weights are downloaded. It
    // runs outside the command queue, so a slow metadata request never holds
    // up Stop, Cancel or Run, and every check is repeated inside the queue.
    async function addModel(entry) {
        const draft = validateModel(entry, { seed: false });
        assertNewModelId(draft.id);
        const pinned = validateModel(await pinSources(draft), { seed: false });
        return queue.run(async () => {
            assertNewModelId(pinned.id);
            state.registry.push(JSON.parse(JSON.stringify({ ...pinned, seed: undefined })));
            save();
            return { model: pinned };
        });
    }

    // A lookup reads Hugging Face metadata and, for one GGUF file, the first megabytes of it (the header); it writes
    // nothing. Like addModel it runs outside the command queue, so a slow request never holds up Stop, Cancel or Run,
    // and every lookup in flight ends when the agent drains.
    const lookups = new Set();

    async function lookupModel(args) {
        if (draining) throw new LocalLlmError('shutting_down', 'The agent is restarting; look the model up again once it is back.');
        if (lookups.size >= MAX_LOOKUPS) throw new LocalLlmError('busy', `${MAX_LOOKUPS} lookups are already running; try again when one has finished.`);
        const stop = new AbortController();
        lookups.add(stop);
        try {
            return await lookup(args, { token: env.HF_TOKEN || '', fetchImpl, baseUrl: hfBaseUrl, signal: stop.signal });
        } finally {
            lookups.delete(stop);
        }
    }

    // An update keeps a source's pinned identity while it is otherwise
    // unchanged; it never silently re-resolves a branch.
    function carryPins(candidate, existing) {
        const sources = { ...candidate.sources };
        for (const [format, source] of Object.entries(sources)) {
            sources[format] = storeFor(source).carryPin(source, existing.sources[format]);
        }
        return { ...candidate, sources };
    }

    async function updateModel(entry) {
        const candidate = validateModel(entry);
        const existing = validateModel(state.registry[userModelIndex(candidate.id)]);
        const pinned = validateModel(await pinSources(carryPins(candidate, existing)), { seed: false });
        return queue.run(async () => {
            const index = userModelIndex(pinned.id);
            const current = validateModel(state.registry[index]);
            assertModelNotInUse(current);
            // A changed or removed source must not leave its downloaded weights
            // behind without a model to delete them from.
            for (const [format, source] of Object.entries(current.sources)) {
                const next = pinned.sources[format];
                if (next && artifactKey(next) === artifactKey(source)) continue;
                const disk = await weightsState(source);
                if (disk && disk.state !== 'absent' && disk.state !== 'unpinned') {
                    throw new LocalLlmError('weights_present',
                        `Delete the downloaded ${format} weights first; this change would leave them without a model.`);
                }
            }
            state.registry[index] = JSON.parse(JSON.stringify({ ...pinned, seed: undefined }));
            save();
            return { model: pinned };
        });
    }

    function removeModel({ modelId } = {}) {
        return queue.run(async () => {
            const index = state.registry.findIndex((model) => model.id === modelId);
            if (index < 0) {
                throw new LocalLlmError(seedCatalog.some((model) => model.id === modelId) ? 'read_only' : 'unknown_model',
                    'Only user models can be removed; seed entries are read-only.');
            }
            let model;
            try {
                model = validateModel(state.registry[index]);
            } catch {
                // An entry this catalog cannot read (an earlier schema) is not
                // offered and cannot be in use; remove it as it is.
                state.registry.splice(index, 1);
                save();
                return { removed: modelId, note: 'The entry was not supported by this catalog; downloaded weights, if any, stay on disk.' };
            }
            for (const [format, source] of Object.entries(model.sources)) {
                if (inUse(artifactKey(source))) {
                    throw new LocalLlmError('in_use', 'This model is in use; stop it or cancel its download first.');
                }
                const disk = await weightsState(source);
                if (disk && disk.state !== 'absent' && disk.state !== 'unpinned') {
                    throw new LocalLlmError('weights_present', `Delete the downloaded ${format} weights first.`);
                }
            }
            state.registry.splice(index, 1);
            save();
            return { removed: modelId };
        });
    }

    // ---------------------------------------------------------- installs

    async function installInfo(id) {
        const info = await installer.describe(id);
        const record = state.runnerInstalls?.[id] || null;
        return { ...info, state: record ? structuredClone(record) : null, installing: installJob?.runnerId === id };
    }

    function setInstall(id, fields) {
        state.runnerInstalls ||= {};
        state.runnerInstalls[id] = { ...(state.runnerInstalls[id] || {}), ...fields, updatedAt: now().toISOString() };
        save();
    }

    function startInstallJob(entry) {
        const controller = new AbortController();
        const current = { runnerId: entry.id, abort: controller, cancelReason: null, promise: null };
        current.promise = (async () => {
            setInstall(entry.id, { phase: 'downloading', error: null, pausedReason: null });
            const licence = state.runnerInstalls[entry.id]?.licence || null;
            await installer.fetchAll(entry, {
                signal: controller.signal,
                licence,
                onProgress: (progress) => setInstall(entry.id, {
                    download: { bytes: progress.bytes, total: progress.total, rate: Math.round(progress.rate || 0), transferred: progress.transferred },
                }),
            });
            setInstall(entry.id, { phase: 'installing' });
            const built = await installer.ensureRunnable(entry.id, { signal: controller.signal });
            // The old version's cache goes only after the new one is in place.
            await installer.pruneOtherVersions(entry.id);
            setInstall(entry.id, { phase: 'installed', version: entry.version, rebuild: { seconds: built.seconds, bytes: built.bytes }, installedAt: now().toISOString() });
            log.append('controller', `installed ${entry.id} ${entry.version}; runnable copy built in ${built.seconds.toFixed(1)} s`);
        })().catch((error) => {
            const aborted = controller.signal.aborted || (error instanceof DownloadError && error.code === 'ABORTED');
            if (aborted || (error instanceof DownloadError && ['PAUSED_ENOSPC', 'NETWORK'].includes(error.code))) {
                const reason = current.cancelReason === 'drain'
                    ? 'The agent restarted during the install; press Install to resume.'
                    : (aborted ? 'Install stopped; press Install to resume.' : error.message);
                setInstall(entry.id, { phase: 'paused', pausedReason: reason, error: aborted ? null : error.message });
                return;
            }
            setInstall(entry.id, { phase: 'error', error: error.message });
        }).finally(() => {
            if (installJob === current) installJob = null;
            redetectRunners();
        });
        installJob = current;
    }

    /**
     * Install one runner from the image's lock. Only the lock's files are ever
     * fetched; a licence that needs acceptance must be accepted, and the
     * acceptance is recorded with who (from the verified caller) and when.
     */
    function installRunner({ runnerId, acceptLicence = false, acceptedBy = null, ...rest } = {}) {
        return queue.run(async () => {
            if (Object.keys(rest).length) {
                throw new LocalLlmError('invalid_request', `Unexpected install fields: ${Object.keys(rest).join(', ')}`);
            }
            if (!lockListed(runnerId)) {
                throw new LocalLlmError('not_installable', `No installable runner '${String(runnerId)}' in this image's runner lock.`);
            }
            // A switch may depend on the profile (vLLM on unified memory), and a runner with no
            // policy for it cannot be installed: decide the profile first if a snapshot can.
            await currentProfile();
            const refusal = profileRefusal(getRunner(runnerId));
            if (refusal) throw new LocalLlmError('runner_unavailable', refusal, { runner: runnerId });
            // An entry only the agent's lock has is offered on the cpu profile alone.
            if (!installable(runnerId)) {
                throw new LocalLlmError('runner_unavailable', notInstallableReason(getRunner(runnerId)), { runner: runnerId });
            }
            assertEnabled(getRunner(runnerId));
            if (draining) throw new LocalLlmError('shutting_down', 'The agent is restarting; install again once it is back.');
            const entry = installer.entryFor(runnerId);
            if (installJob) {
                if (installJob.runnerId === runnerId) return { accepted: true, install: await installInfo(runnerId) };
                throw new LocalLlmError('busy', `${installJob.runnerId} is being installed; wait for it to finish.`);
            }
            const record = state.runnerInstalls?.[runnerId];
            const accepted = record?.licence && record.version === entry.version ? record.licence : null;
            if (entry.licence.requiresAcceptance && !accepted && acceptLicence !== true) {
                throw new LocalLlmError('licence_required', `Installing ${runnerId} needs its ${entry.licence.name} terms accepted.`,
                    { licence: entry.licence });
            }
            if ((await installer.describe(runnerId)).installed && (await installer.describe(runnerId)).runnable) {
                return { accepted: false, installed: true, install: await installInfo(runnerId) };
            }
            setInstall(runnerId, {
                version: entry.version,
                phase: 'downloading',
                download: { bytes: 0, total: entry.totalBytes, rate: 0, transferred: 0 },
                error: null,
                pausedReason: null,
                licence: accepted || (entry.licence.requiresAcceptance
                    ? { name: entry.licence.name, acceptedBy: typeof acceptedBy === 'string' && acceptedBy ? acceptedBy : 'unknown admin', acceptedAt: now().toISOString() }
                    : null),
            });
            startInstallJob(entry);
            return { accepted: true, install: await installInfo(runnerId) };
        });
    }

    function uninstallRunner({ runnerId } = {}) {
        return queue.run(async () => {
            if (!lockListed(runnerId)) {
                throw new LocalLlmError('not_installable', `No installable runner '${String(runnerId)}' in this image's runner lock.`);
            }
            const deployment = state.deployment;
            if (deployment?.runnerId === runnerId && (ACTIVE_PHASES.has(deployment.phase) || job !== null)) {
                throw new LocalLlmError('in_use', `${runnerId} is running; stop the model first.`);
            }
            if (installJob?.runnerId === runnerId) {
                installJob.cancelReason = 'uninstall';
                installJob.abort.abort();
                await installJob.promise;
            }
            const result = await installer.uninstall(runnerId);
            if (state.runnerInstalls?.[runnerId]) {
                delete state.runnerInstalls[runnerId];
                save();
            }
            redetectRunners();
            return result;
        });
    }

    /**
     * Drain for SIGTERM: stop admitting commands, checkpoint any download
     * (the partial and its identity stay), stop the runner, persist.
     */
    async function drain() {
        draining = true;
        planning?.abort();
        for (const stop of lookups) stop.abort();
        for (const stop of pinning) stop.abort();
        // The active job, then every hardware query (the job's, a command's or
        // the overview's), stop at once: a snapshot's nvidia-smi queries can
        // take 10 s each. A killed query is waited for during the command wait
        // below (HARDWARE_QUERY_REAP_MS), so the drain's worst case is unchanged.
        const active = job;
        if (active && !active.abort.signal.aborted) {
            active.cancelReason = 'drain';
            active.abort.abort();
        }
        hardwareStop.abort();
        // Let a command that is already running finish first (a Run refuses to
        // start a job once draining is set); bounded so it cannot block the drain.
        const waited = new AbortController();
        await Promise.race([queue.close(), sleep(DRAIN_QUEUE_WAIT_MS, waited.signal)]);
        waited.abort();
        if (active) await active.promise;
        if (job) {
            job.cancelReason ??= 'drain';
            job.abort.abort();
            await job.promise;
        }
        if (installJob) {
            installJob.cancelReason = 'drain';
            installJob.abort.abort();
            await installJob.promise;
        }
        if (runner) {
            const stopping = runner;
            runner = null;
            await stopping.stop({ graceMs: runnerGraceMs() });
        }
        // A stopped query that has not exited (a hung driver) is left to the host, never forgotten.
        // Every query was stopped when the drain began, so this waits at most HARDWARE_QUERY_REAP_MS from then.
        await settleStoppedQueries();
        for (const pid of unreapedQueries()) {
            log.append('controller', `nvidia-smi (pid ${pid}) was killed but has not exited; the drain goes on without it`);
        }
        if (state.deployment) {
            if (TRANSFER_PHASES.has(state.deployment.phase)) {
                state.deployment.phase = 'paused';
                state.deployment.pausedReason = 'The agent restarted during the download; press Run to resume.';
            } else if (ACTIVE_PHASES.has(state.deployment.phase)) {
                state.deployment.phase = 'idle';
                state.deployment.runner = null;
            }
        }
        save();
    }

    return Object.freeze({
        overview,
        status,
        chatTarget,
        recordCompletion,
        run,
        stop,
        cancelDownload,
        deleteWeights,
        addModel,
        lookupModel,
        updateModel,
        removeModel,
        installRunner,
        uninstallRunner,
        redetectRunners,
        drain,
        get state() { return state; },
        artifactPathsFor: (source) => stores.huggingface.paths(source),
    });
}
