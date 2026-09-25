// The controller: one active deployment at a time (6 GB of VRAM), every
// mutation serialized through one command queue, downloads and runner
// processes owned here and never by a tool process.
//
//   idle -> downloading -> verifying -> starting -> ready -> stopping -> idle
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
import { RUNNERS, defaultPorts, runnerSummary } from '../runners/index.mjs';
import { admit } from './admission.mjs';
import { WEIGHT_FORMATS, loadSeedCatalog, mergeCatalog, validateModel } from './catalog.mjs';
import { createCommandQueue } from './commandQueue.mjs';
import {
    DownloadError,
    downloadArtifact,
    inspectArtifact,
    removeArtifact,
    resolveHuggingFaceArtifact,
} from './downloader.mjs';
import { readMemory as readHostMemory, readSnapshot } from './hardware.mjs';
import { createRunnerInstaller } from './runnerInstaller.mjs';
import { loadRunnerLock } from './runnerLock.mjs';
import { createLogBuffer, parseRunnerReport, startRunnerProcess } from './runnerProcess.mjs';
import { createStateStore, reconcileAfterRestart } from './stateStore.mjs';
import { createWeightStores } from './weightStores.mjs';
import { DRAIN_QUEUE_WAIT_MS, DRAIN_RUNNER_GRACE_MS } from '../drainBudget.mjs';

const ACTIVE_PHASES = new Set(['downloading', 'verifying', 'starting', 'loading', 'ready', 'stopping']);
const TRANSFER_PHASES = new Set(['downloading', 'verifying']);
const REQUEST_ID_RE = /^[A-Za-z0-9_-]{8,128}$/;
const PROBE_TIMEOUT_MS = 10_000;

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
    snapshot = () => readSnapshot({ dataDir }),
    download = downloadArtifact,
    inspect = inspectArtifact,
    remove = removeArtifact,
    resolveHf = resolveHuggingFaceArtifact,
    startRunner = startRunnerProcess,
    fetchImpl = globalThis.fetch,
    ports = defaultPorts(runners),
    apiKeyFactory = () => crypto.randomBytes(32).toString('base64url'),
    readyTimeoutMs = 20 * 60_000,
    stopGraceMs = 10_000,
    pollMs = 1000,
    // On-demand runners (runners plan §5.2): the lock shipped in the image,
    // a verified cache under /data/runners, runnable copies under /opt/runners.
    installer = createRunnerInstaller({
        lock: loadRunnerLock(env.LOCAL_LLM_RUNNER_LOCK || undefined),
        cacheRoot: path.join(dataDir, 'runners'),
        runRoot: env.LOCAL_LLM_RUN_ROOT || undefined,
    }),
    detectRunner = (runner) => runner.detect({ spawnSync, installer }),
    // Test seams for Hugging Face snapshots (the store's defaults are the real downloader).
    downloadSnapshot = undefined,
    inspectSnapshot = undefined,
    resolveSnapshot = undefined,
    // The agent's own /dev/shm (a private tmpfs), where PyTorch runners keep sockets.
    shmDir = '/dev/shm',
    // Host-memory guard for deployments whose admission sets a RAM floor
    // (vLLM with CPU offload): MemAvailable is sampled this often while the
    // runner loads, and less often once it is ready.
    readMemory = readHostMemory,
    memoryGuardLoadMs = 2000,
    memoryGuardReadyMs = 10_000,
    now = () => new Date(),
} = {}) {
    const log = createLogBuffer({ file: path.join(dataDir, 'logs', 'runner.log') });
    const queue = createCommandQueue();
    const state = reconcileAfterRestart(stateStore.load(), now().toISOString());
    let detected = {};
    let job = null;
    // One runner install at a time, beside the deployment job.
    let installJob = null;
    // Installable through the controller: in the image's runner lock and with
    // an adapter here. A lock entry for a runner this release cannot run is
    // installed only by the CI install check.
    const installable = (id) => typeof id === 'string' && Object.hasOwn(runners, id) && installer.installable(id);
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

    function catalog() {
        return mergeCatalog(seedCatalog, state.registry);
    }

    function findModel(modelId) {
        const model = catalog().find((entry) => entry.id === modelId);
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

    function effectiveParams(model, runnerId, override) {
        const runnerDef = getRunner(runnerId);
        const saved = override ?? state.params[paramsKey(model.id, runnerId)] ?? {};
        return runnerDef.normalizeParams(saved, { model });
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
        if (!definition.supported || !source) {
            return { modelId, runnerId, params: null, context: null,
                admission: admit({ runner: definition, model, source, params: {}, snapshot: snap }) };
        }
        let normalized;
        try {
            normalized = effectiveParams(model, runnerId, params && typeof params === 'object' ? params : undefined);
        } catch (error) {
            return { modelId, runnerId, error: error.message, field: error.details?.field ?? null };
        }
        const disk = await weightsState(source);
        const remaining = disk?.total ? Math.max(0, disk.total - (disk.bytes || 0)) : 0;
        return {
            modelId,
            runnerId,
            params: normalized,
            context: definition.describeContext ? definition.describeContext(normalized) : null,
            admission: admit({ runner: definition, model, source, params: normalized, snapshot: snap, remainingDownloadBytes: remaining }),
        };
    }

    async function overview({ preview = null } = {}) {
        const snap = await snapshot();
        const runnerList = [];
        for (const definition of Object.values(runners)) {
            runnerList.push({
                ...runnerSummary(definition),
                ...(await runnerInfo(definition.id)),
                ...(installable(definition.id) ? { install: await installInfo(definition.id) } : {}),
            });
        }
        const models = [];
        for (const model of catalog()) {
            // One entry per weight format: runners that read it share the download.
            const weights = {};
            for (const [format, source] of Object.entries(model.sources)) {
                weights[format] = {
                    label: WEIGHT_FORMATS[format]?.label || format,
                    size: source?.size ?? null,
                    download: await weightsState(source),
                    runners: Object.values(runners)
                        .filter((definition) => definition.supported && definition.weightFormat === format)
                        .map((definition) => definition.id),
                };
            }
            const perRunner = {};
            for (const definition of Object.values(runners)) {
                const source = sourceFor(model, definition);
                if (!source) continue;
                let params = null;
                let paramError = null;
                if (definition.supported) {
                    try { params = effectiveParams(model, definition.id); } catch (error) { paramError = error.message; }
                }
                const disk = definition.supported ? weights[definition.weightFormat].download : null;
                const remaining = disk && disk.total ? Math.max(0, disk.total - (disk.bytes || 0)) : 0;
                perRunner[definition.id] = {
                    format: definition.weightFormat,
                    size: source?.size ?? null,
                    download: disk,
                    params,
                    context: params && definition.describeContext ? definition.describeContext(params) : null,
                    admission: paramError
                        ? { status: 'incompatible', reason: paramError, estimate: {}, warnings: [] }
                        : admit({
                            runner: definition, model, source, params: params || {}, snapshot: snap,
                            remainingDownloadBytes: remaining,
                        }),
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
                validated: model.validated,
                weights,
                runners: perRunner,
            });
        }
        return {
            hardware: snap,
            runners: runnerList,
            models,
            deployment: publicDeployment(),
            gatewayModel: 'soul_gateway/local-llms/local-llm/default',
            ...(preview ? { preview: await previewRun(preview, snap) } : {}),
        };
    }

    async function status({ sinceSeq = 0 } = {}) {
        const deployment = publicDeployment();
        const lines = log.since(Number(sinceSeq) || 0);
        let gpu = null;
        try {
            const snap = await snapshot();
            gpu = snap.gpu;
        } catch {}
        const definition = deployment && Object.hasOwn(runners, deployment.runnerId) ? runners[deployment.runnerId] : null;
        const parseReport = definition?.parseReport || parseRunnerReport;
        return {
            phase: deployment?.phase || 'idle',
            deployment,
            logs: lines,
            nextSeq: log.seq,
            gpu,
            runnerReport: parseReport(log.all().filter((line) => line.seq > runnerLogStart)),
            context: deployment && definition?.describeContext ? definition.describeContext(deployment.params) : null,
            lastCompletion,
        };
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
            requestOptions: definition.requestOptions?.(deployment.params) ?? null,
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
        // Only what the runner needs: never the agent's secrets or tokens.
        return {
            PATH: env.PATH || '/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin',
            HOME: path.join(dataDir, 'home'),
            LANG: 'C.UTF-8',
            ...launchEnv,
        };
    }

    function launchRunner(deployment, launch, apiKey, port) {
        fs.mkdirSync(path.join(dataDir, 'home'), { recursive: true });
        log.addSecret(apiKey);
        runnerLogStart = log.seq;
        lastCompletion = null;
        log.append('controller', `starting ${launch.command} ${launch.args.join(' ')}`);
        const process = startRunner({ command: launch.command, args: launch.args, env: runnerEnv(launch.env), cwd: launch.cwd || '/', log });
        runner = {
            pid: process.pid,
            port,
            apiKey,
            deploymentId: deployment.id,
            exited: process.exited,
            stop: (options) => process.stop(options),
            get running() { return process.running; },
        };
        const current = runner;
        const cancelGuard = startMemoryGuard(deployment, current);
        process.exited.then((result) => {
            cancelGuard();
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

    // The backstop behind a RAM-floor admission (vLLM with CPU offload, whose
    // estimate comes from two runs on one machine): while this runner loads and
    // runs, sample MemAvailable; below the floor, stop it at once.
    function startMemoryGuard(deployment, current) {
        const floor = deployment.admission?.estimate?.ramFloorBytes;
        if (!Number.isFinite(floor) || floor <= 0) return () => {};
        let timer = null;
        let cancelled = false;
        const schedule = () => {
            if (cancelled) return;
            const ready = state.deployment?.id === deployment.id && state.deployment.phase === 'ready';
            timer = setTimeout(tick, ready ? memoryGuardReadyMs : memoryGuardLoadMs);
            timer.unref?.();
        };
        const tick = () => {
            if (cancelled || runner !== current) return;
            let available = null;
            try {
                available = readMemory().availableBytes;
            } catch {}
            if (Number.isFinite(available) && available < floor) {
                cancelled = true;
                stopForMemory(deployment, current,
                    `stopped: host memory below the floor (${gib(available)} available, ${gib(floor)} required)`).catch(() => {});
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

    async function stopForMemory(deployment, current, message) {
        memoryStop = { deploymentId: deployment.id, message };
        log.append('controller', message);
        // The host is short of memory now: SIGKILL follows SIGTERM within the
        // drain's grace, not a Stop's.
        try {
            await current.stop({ graceMs: DRAIN_RUNNER_GRACE_MS });
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
        if (progress.total && progress.bytes >= progress.total && deployment.phase === 'downloading') {
            deployment.phase = 'verifying';
        }
        save();
    }

    async function recheckAdmission(deployment, model) {
        const definition = getRunner(deployment.runnerId);
        const source = deployment.artifact;
        const result = admit({ runner: definition, model, source, params: deployment.params, snapshot: await snapshot() });
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
            launch: (spec) => launchRunner(deployment, spec, apiKey, port),
            waitForHttp: (url, options = {}) => waitForHttp(url, { ...options, signal }),
            setPhase: (phase) => setPhase(phase),
            progress: updateProgress,
            record: (fields) => {
                Object.assign(deployment, fields);
                save();
            },
            recheckAdmission: () => recheckAdmission(deployment, model),
            throwIfAborted: () => throwIfAborted(signal),
        });
    }

    async function runPipeline(deployment, model, signal) {
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
            await recheckAdmission(deployment, model);
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
        const details = await definition.start(startContext(deployment, model, signal, weights, runnerDir));
        throwIfAborted(signal);
        if (!runner || runner.deploymentId !== deployment.id) {
            throw new LocalLlmError('runner_exited', 'The runner is not running after its start-up.');
        }
        setPhase('ready', { runner: { pid: runner.pid, port: runner.port, startedAt: now().toISOString(), ...(details || {}) } });
    }

    function startJob(deployment, model) {
        const controller = new AbortController();
        const current = {
            deploymentId: deployment.id,
            abort: controller,
            cancelReason: null,
            promise: null,
        };
        current.promise = runPipeline(deployment, model, controller.signal)
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
        return queue.run(async () => {
            if (typeof requestId !== 'string' || !REQUEST_ID_RE.test(requestId)) {
                throw new LocalLlmError('invalid_request', 'requestId must be 8-128 letters, digits, dash or underscore.');
            }
            // The browser client may retry a timed-out call: the same request is a no-op.
            if (state.requests[requestId]) {
                return { duplicate: true, deployment: publicDeployment() };
            }
            const model = findModel(modelId);
            const definition = getRunner(runnerId);
            const source = sourceFor(model, definition);
            if (!definition.supported) {
                const result = admit({ runner: definition, model, source, params: {}, snapshot: {} });
                throw new LocalLlmError('runner_unsupported', result.reason);
            }
            if (!source) throw new LocalLlmError('no_source', `${model.displayName} has no ${definition.displayName} source.`);
            if (installJob?.runnerId === runnerId) {
                throw new LocalLlmError('busy', `${definition.displayName} is being installed; run it once the install finishes.`);
            }
            if (installable(runnerId) && !(await installer.describe(runnerId)).installed) {
                throw new LocalLlmError('runner_not_installed', `${definition.displayName} is not installed; install it first.`);
            }
            const store = storeFor(source);
            if (!store.isPinned(source)) {
                throw new LocalLlmError('unpinned', 'The model source is not pinned to a commit; update the model entry.');
            }
            const normalized = effectiveParams(model, runnerId, params);
            const current = state.deployment;
            if (current && (ACTIVE_PHASES.has(current.phase) || job)) {
                if (!replace) {
                    throw new LocalLlmError('busy', `${current.modelId} on ${current.runnerId} is ${current.phase}; `
                        + 'stop it first or run with replace.');
                }
                await stopEverything('replace');
            }
            const disk = await weightsState(source);
            const remaining = disk?.total ? Math.max(0, disk.total - (disk.bytes || 0)) : 0;
            const admission = admit({
                runner: definition, model, source, params: normalized, snapshot: await snapshot(),
                remainingDownloadBytes: remaining,
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
                params: normalized,
                artifact: structuredClone(source),
                phase: store.fetchedBy === 'runner' ? 'starting' : 'downloading',
                admission,
                download: { bytes: disk?.bytes || 0, total: disk?.total ?? null, rate: 0, etaSeconds: null, transferred: 0 },
                error: null,
                pausedReason: null,
                runner: null,
                logSeqStart: log.seq,
                createdAt: at,
                updatedAt: at,
            };
            state.requests[requestId] = { deploymentId: state.deployment.id, at };
            save();
            startJob(state.deployment, model);
            return { accepted: true, deployment: publicDeployment() };
        });
    }

    function stop() {
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
        return queue.run(async () => {
            const deployment = state.deployment;
            if (!deployment || !job || !(TRANSFER_PHASES.has(deployment.phase) || deployment.phase === 'starting')) {
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
            const model = findModel(modelId);
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

    async function pinSources(entry) {
        const sources = { ...entry.sources };
        for (const [format, source] of Object.entries(sources)) {
            sources[format] = await storeFor(source).pin(source);
        }
        return { ...entry, sources };
    }

    function assertNewModelId(id) {
        if (catalog().some((model) => model.id === id)) {
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
            const model = validateModel(state.registry[index]);
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
            if (!installable(runnerId)) {
                throw new LocalLlmError('not_installable', `No installable runner '${String(runnerId)}' in this image's runner lock.`);
            }
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
            if (!installable(runnerId)) {
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
        // Let a command that is already running finish first (a Run refuses to
        // start a job once draining is set); bounded so it cannot block the drain.
        const waited = new AbortController();
        await Promise.race([queue.close(), sleep(DRAIN_QUEUE_WAIT_MS, waited.signal)]);
        waited.abort();
        if (job) {
            job.cancelReason = 'drain';
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
