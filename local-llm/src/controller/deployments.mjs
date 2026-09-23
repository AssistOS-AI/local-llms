// The controller: one active deployment at a time (6 GB of VRAM), every
// mutation serialized through one command queue, downloads and runner
// processes owned here and never by a tool process.
//
//   idle -> downloading -> verifying -> starting -> ready -> stopping -> idle
//   (llama.cpp)            plus paused and error
//   idle -> starting -> downloading -> loading -> ready -> stopping -> idle
//   (Ollama: the server runs first and pulls through /api/pull)

import { spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import { RUNNERS } from '../runners/index.mjs';
import { admit } from './admission.mjs';
import { loadSeedCatalog, mergeCatalog, validateModel } from './catalog.mjs';
import { createCommandQueue } from './commandQueue.mjs';
import {
    DownloadError,
    artifactPaths,
    downloadArtifact,
    inspectArtifact,
    removeArtifact,
    resolveHuggingFaceArtifact,
} from './downloader.mjs';
import { readSnapshot } from './hardware.mjs';
import { deleteOllamaModel, deleteOllamaPartials, partialPullBytes, readOllamaManifest } from './ollamaStore.mjs';
import { createLogBuffer, parseRunnerReport, startRunnerProcess } from './runnerProcess.mjs';
import { createStateStore, reconcileAfterRestart } from './stateStore.mjs';

const ACTIVE_PHASES = new Set(['downloading', 'verifying', 'starting', 'loading', 'ready', 'stopping']);
const TRANSFER_PHASES = new Set(['downloading', 'verifying']);
const REQUEST_ID_RE = /^[A-Za-z0-9_-]{8,128}$/;
const OLLAMA_DIGEST_RE = /^sha256:[0-9a-f]{64}$/;
const DEFAULT_PORTS = Object.freeze({ 'llama.cpp': 18080, ollama: 18434 });
const DRAIN_QUEUE_WAIT_MS = 5000;
const PROBE_TIMEOUT_MS = 10_000;

function paramsKey(modelId, runnerId) {
    return `${modelId}|${runnerId}`;
}

function artifactKey(runnerId, artifact) {
    if (!artifact) return null;
    return artifact.type === 'ollama'
        ? `ollama:${artifact.tag}`
        : `gguf:${artifact.repo}@${artifact.commit}/${artifact.file}`;
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
    ports = DEFAULT_PORTS,
    apiKeyFactory = () => crypto.randomBytes(32).toString('base64url'),
    readyTimeoutMs = 20 * 60_000,
    stopGraceMs = 10_000,
    pollMs = 1000,
    detectRunner = (runner) => runner.detect({ spawnSync }),
    now = () => new Date(),
} = {}) {
    const ggufRoot = path.join(dataDir, 'models', 'gguf');
    const ollamaModels = path.join(dataDir, 'models', 'ollama');
    const log = createLogBuffer({ file: path.join(dataDir, 'logs', 'runner.log') });
    const queue = createCommandQueue();
    const state = reconcileAfterRestart(stateStore.load(), now().toISOString());
    const detected = {};
    let job = null;
    let runner = null;
    let draining = false;
    // Speed of the last completion served by the ready runner, as the chat
    // responder reported it. In memory only: it describes this runner start.
    let lastCompletion = null;
    // Log sequence number before the current runner started: its report is
    // read only from lines after it, never from a previous runner's output.
    let runnerLogStart = 0;

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

    function runnerInfo(id) {
        if (!detected[id]) detected[id] = detectRunner(runners[id]);
        return detected[id];
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
        return artifactKey(deployment.runnerId, deployment.artifact) === key
            && (ACTIVE_PHASES.has(deployment.phase) || job !== null);
    }

    async function downloadState(model, runnerId) {
        const source = model.sources[runnerId];
        if (!source) return null;
        if (source.type === 'huggingface') {
            if (!source.commit) return { state: 'unpinned', bytes: 0, total: null };
            const inspected = await inspect({ root: ggufRoot, artifact: source });
            return { ...inspected, total: source.size };
        }
        const manifest = readOllamaManifest(ollamaModels, source.tag);
        if (manifest?.complete) return { state: 'complete', bytes: manifest.size, total: manifest.size };
        const partial = partialPullBytes(ollamaModels, state.ollamaPulls?.[source.tag] || []);
        return { state: partial ? 'partial' : 'absent', bytes: partial, total: source.size ?? null };
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
        const source = model.sources[runnerId] || (runnerId === 'vllm' ? model.sources['llama.cpp'] : undefined);
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
        const disk = await downloadState(model, runnerId);
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
        const runnerList = Object.values(runners).map((definition) => ({
            id: definition.id,
            displayName: definition.displayName,
            weightFormat: definition.weightFormat,
            pinnedVersion: definition.pinnedVersion,
            supported: definition.supported,
            paramSchema: definition.paramSchema,
            ...runnerInfo(definition.id),
        }));
        const models = [];
        for (const model of catalog()) {
            const perRunner = {};
            for (const definition of Object.values(runners)) {
                const source = model.sources[definition.id]
                    || (definition.id === 'vllm' ? model.sources['llama.cpp'] : undefined);
                if (!source && !['vllm', 'lmstudio'].includes(definition.id)) continue;
                let params = null;
                let paramError = null;
                if (definition.supported) {
                    try { params = effectiveParams(model, definition.id); } catch (error) { paramError = error.message; }
                }
                const disk = source && definition.supported ? await downloadState(model, definition.id) : null;
                const remaining = disk && disk.total ? Math.max(0, disk.total - (disk.bytes || 0)) : 0;
                perRunner[definition.id] = {
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
        const definition = deployment ? runners[deployment.runnerId] : null;
        return {
            phase: deployment?.phase || 'idle',
            deployment,
            logs: lines,
            nextSeq: log.seq,
            gpu,
            runnerReport: parseRunnerReport(log.all().filter((line) => line.seq > runnerLogStart)),
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
        return {
            runnerId: deployment.runnerId,
            modelId: deployment.modelId,
            baseUrl: `http://127.0.0.1:${runner.port}`,
            apiKey: runner.apiKey || null,
            model: deployment.runnerId === 'ollama' ? deployment.artifact.tag : deployment.modelId,
            requestOptions: deployment.runnerId === 'ollama'
                ? getRunner('ollama').requestOptions?.(deployment.params) ?? null
                : null,
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

    function launchRunner(deployment, launch, apiKey) {
        fs.mkdirSync(path.join(dataDir, 'home'), { recursive: true });
        log.addSecret(apiKey);
        runnerLogStart = log.seq;
        lastCompletion = null;
        log.append('controller', `starting ${launch.command} ${launch.args.join(' ')}`);
        const process = startRunner({ command: launch.command, args: launch.args, env: runnerEnv(launch.env), log });
        runner = {
            pid: process.pid,
            port: ports[deployment.runnerId],
            apiKey,
            deploymentId: deployment.id,
            exited: process.exited,
            stop: (options) => process.stop(options),
            get running() { return process.running; },
        };
        const current = runner;
        process.exited.then((result) => {
            if (runner !== current) return;
            runner = null;
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

    async function runLlamaCpp(deployment, model, signal) {
        setPhase('downloading');
        const result = await download({
            artifact: deployment.artifact,
            root: ggufRoot,
            token: env.HF_TOKEN || '',
            baseUrl: hfBaseUrl,
            onProgress: updateProgress,
            signal,
        });
        deployment.download = {
            ...(deployment.download || {}),
            bytes: deployment.artifact.size,
            total: deployment.artifact.size,
            transferred: result.bytesTransferred,
        };
        await recheckAdmission(deployment, model);
        setPhase('starting');
        const definition = getRunner('llama.cpp');
        const apiKey = apiKeyFactory();
        const launch = definition.buildLaunch({
            artifactPath: result.path, params: deployment.params, port: ports['llama.cpp'], apiKey, model,
        });
        const process = launchRunner(deployment, launch, apiKey);
        const base = `http://127.0.0.1:${process.port}`;
        // /health answers 503 while the model loads and 200 once it serves.
        await waitForHttp(`${base}/health`, { signal, process });
        await waitForHttp(`${base}/v1/models`, { headers: { authorization: `Bearer ${apiKey}` }, signal, process });
        setPhase('ready', { runner: { pid: process.pid, port: process.port, startedAt: now().toISOString() } });
    }

    // Remember which blobs a tag's pull touched, so its partial files can be
    // counted and deleted per tag rather than across every Ollama model.
    function recordPullDigest(tag, digest) {
        state.ollamaPulls ||= {};
        const digests = state.ollamaPulls[tag] ||= [];
        if (digests.includes(digest)) return;
        digests.push(digest);
        save();
    }

    async function streamOllamaPull(base, tag, signal) {
        const response = await fetchImpl(`${base}/api/pull`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: tag, stream: true }),
            signal,
        });
        if (!response.ok) throw new LocalLlmError('pull_failed', `Ollama pull failed with HTTP ${response.status}.`);
        const layers = new Map();
        const started = Date.now();
        let buffer = '';
        const decoder = new TextDecoder();
        for await (const chunk of response.body) {
            buffer += decoder.decode(chunk, { stream: true });
            let newline;
            while ((newline = buffer.indexOf('\n')) >= 0) {
                const line = buffer.slice(0, newline).trim();
                buffer = buffer.slice(newline + 1);
                if (!line) continue;
                const event = JSON.parse(line);
                if (event.error) throw new LocalLlmError('pull_failed', `Ollama pull failed: ${event.error}`);
                if (typeof event.digest === 'string' && OLLAMA_DIGEST_RE.test(event.digest)) recordPullDigest(tag, event.digest);
                if (event.digest && event.total) layers.set(event.digest, { total: event.total, completed: event.completed || 0 });
                const total = [...layers.values()].reduce((sum, layer) => sum + layer.total, 0);
                const bytes = [...layers.values()].reduce((sum, layer) => sum + layer.completed, 0);
                const elapsed = Math.max(1, (Date.now() - started) / 1000);
                updateProgress({ bytes, total, rate: bytes / elapsed, etaSeconds: null, transferred: bytes });
                if (event.status === 'success') return;
            }
        }
        throw new LocalLlmError('pull_failed', 'The Ollama pull stream ended before success.');
    }

    async function runOllama(deployment, model, signal) {
        setPhase('starting');
        const definition = getRunner('ollama');
        const launch = definition.buildLaunch({ params: deployment.params, port: ports.ollama, dataDir, model });
        const process = launchRunner(deployment, launch, null);
        const base = `http://127.0.0.1:${process.port}`;
        await waitForHttp(`${base}/api/version`, { signal, process });
        const tag = deployment.artifact.tag;
        const cached = readOllamaManifest(ollamaModels, tag);
        if (!cached?.complete) {
            setPhase('downloading');
            await streamOllamaPull(base, tag, signal);
        } else {
            deployment.download = { bytes: cached.size, total: cached.size, rate: 0, etaSeconds: null, transferred: 0 };
        }
        const pulled = readOllamaManifest(ollamaModels, tag);
        if (!pulled?.complete) throw new LocalLlmError('pull_failed', `Ollama reports ${tag} but its files are incomplete.`);
        if (state.ollamaPulls?.[tag]) {
            delete state.ollamaPulls[tag];
            save();
        }
        if (deployment.artifact.manifestDigest && pulled.manifestDigest !== deployment.artifact.manifestDigest) {
            throw new LocalLlmError('identity_changed', `The Ollama tag ${tag} now resolves to ${pulled.manifestDigest}, `
                + `not the pinned ${deployment.artifact.manifestDigest}; update the model entry to accept it.`);
        }
        deployment.resolved = { manifestDigest: pulled.manifestDigest, blobs: pulled.blobs.map((blob) => blob.digest) };
        await recheckAdmission(deployment, model);
        setPhase('loading');
        const options = definition.requestOptions(deployment.params);
        const load = await fetchImpl(`${base}/api/generate`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify({ model: tag, prompt: '', stream: false, ...options }),
            signal,
        });
        if (!load.ok) throw new LocalLlmError('load_failed', `Ollama could not load ${tag} (HTTP ${load.status}).`);
        await load.json().catch(() => null);
        const ps = await (await fetchImpl(`${base}/api/ps`, { signal })).json();
        const loaded = (ps.models || []).find((entry) => entry.name === tag || entry.model === tag);
        if (!loaded) throw new LocalLlmError('load_failed', `Ollama did not keep ${tag} loaded.`);
        setPhase('ready', {
            runner: {
                pid: process.pid,
                port: process.port,
                startedAt: now().toISOString(),
                ollama: { sizeBytes: loaded.size, sizeVramBytes: loaded.size_vram, contextLength: loaded.context_length },
            },
        });
    }

    function startJob(deployment, model) {
        const controller = new AbortController();
        const pipeline = deployment.runnerId === 'ollama' ? runOllama : runLlamaCpp;
        const current = {
            deploymentId: deployment.id,
            abort: controller,
            cancelReason: null,
            promise: null,
        };
        current.promise = pipeline(deployment, model, controller.signal)
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
                        await stopping.stop({ graceMs: stopGraceMs });
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
                    await stopping.stop({ graceMs: stopGraceMs });
                }
                setPhase('error', { error: error.message, runner: null });
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
            await stopping.stop({ graceMs: stopGraceMs });
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
            if (!definition.supported) {
                const result = admit({ runner: definition, model, source: model.sources['llama.cpp'], params: {}, snapshot: {} });
                throw new LocalLlmError('runner_unsupported', result.reason);
            }
            const source = model.sources[runnerId];
            if (!source) throw new LocalLlmError('no_source', `${model.displayName} has no ${definition.displayName} source.`);
            if (source.type === 'huggingface' && !source.commit) {
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
            const disk = await downloadState(model, runnerId);
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
                phase: runnerId === 'ollama' ? 'starting' : 'downloading',
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

    function deleteWeights({ modelId, runnerId } = {}) {
        return queue.run(async () => {
            const model = findModel(modelId);
            const source = model.sources[runnerId];
            if (!source) throw new LocalLlmError('no_source', `${model.displayName} has no ${runnerId} source.`);
            if (inUse(artifactKey(runnerId, source))) {
                throw new LocalLlmError('in_use', 'These weights are in use; stop the model or cancel the download first.');
            }
            let freed;
            if (source.type === 'huggingface') {
                if (!source.commit) return { freedBytes: 0 };
                freed = await remove({ root: ggufRoot, artifact: source });
            } else {
                const pulls = state.ollamaPulls || {};
                const claimedByOthers = Object.entries(pulls)
                    .filter(([tag]) => tag !== source.tag)
                    .flatMap(([, digests]) => digests);
                freed = deleteOllamaModel(ollamaModels, source.tag)
                    + deleteOllamaPartials(ollamaModels, { digests: pulls[source.tag] || [], claimedByOthers });
                if (pulls[source.tag]) {
                    delete pulls[source.tag];
                    save();
                }
            }
            const deployment = state.deployment;
            if (deployment && artifactKey(deployment.runnerId, deployment.artifact) === artifactKey(runnerId, source)
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
        for (const [runnerId, source] of Object.entries(sources)) {
            if (source?.type !== 'huggingface' || source.commit) continue;
            const resolved = await resolveHf({
                repo: source.repo,
                file: source.file,
                revision: source.revision || 'main',
                token: env.HF_TOKEN || '',
                baseUrl: hfBaseUrl,
            });
            sources[runnerId] = { ...source, commit: resolved.commit, size: resolved.size, sha256: resolved.sha256 };
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
        for (const [runnerId, source] of Object.entries(model.sources)) {
            if (inUse(artifactKey(runnerId, source))) {
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
        const pinned = validateModel(await pinSources(entry), { seed: false });
        return queue.run(async () => {
            assertNewModelId(pinned.id);
            state.registry.push(JSON.parse(JSON.stringify({ ...pinned, seed: undefined })));
            save();
            return { model: pinned };
        });
    }

    // An update keeps a source's pinned commit while its repository, file and
    // revision are unchanged; it never silently re-resolves a branch.
    function carryPins(candidate, existing) {
        const sources = { ...candidate.sources };
        for (const [runnerId, source] of Object.entries(sources)) {
            const previous = existing.sources[runnerId];
            if (source.type !== 'huggingface' || source.commit || previous?.type !== 'huggingface' || !previous.commit) continue;
            if (source.repo === previous.repo && source.file === previous.file && source.revision === previous.revision) {
                sources[runnerId] = { ...source, commit: previous.commit, size: previous.size, sha256: previous.sha256 };
            }
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
            for (const [runnerId, source] of Object.entries(current.sources)) {
                const next = pinned.sources[runnerId];
                if (next && artifactKey(runnerId, next) === artifactKey(runnerId, source)) continue;
                const disk = await downloadState(current, runnerId);
                if (disk && disk.state !== 'absent' && disk.state !== 'unpinned') {
                    throw new LocalLlmError('weights_present',
                        `Delete the downloaded ${runnerId} weights first; this change would leave them without a model.`);
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
            for (const [runnerId, source] of Object.entries(model.sources)) {
                if (inUse(artifactKey(runnerId, source))) {
                    throw new LocalLlmError('in_use', 'This model is in use; stop it or cancel its download first.');
                }
                const disk = await downloadState(model, runnerId);
                if (disk && disk.state !== 'absent' && disk.state !== 'unpinned') {
                    throw new LocalLlmError('weights_present', `Delete the downloaded ${runnerId} weights first.`);
                }
            }
            state.registry.splice(index, 1);
            save();
            return { removed: modelId };
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
        if (runner) {
            const stopping = runner;
            runner = null;
            await stopping.stop({ graceMs: stopGraceMs });
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
        drain,
        get state() { return state; },
        artifactPathsFor: (source) => artifactPaths({ root: ggufRoot, artifact: source }),
    });
}
