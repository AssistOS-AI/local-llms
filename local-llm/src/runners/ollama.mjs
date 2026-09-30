import { spawnSync as realSpawnSync } from 'node:child_process';
import path from 'node:path';

import { admissionResult, admitOllama } from '../controller/admission.mjs';
import { admitCpuOllama } from '../controller/profiles.mjs';
import { parseRunnerReport } from '../controller/runnerProcess.mjs';
import { LocalLlmError } from '../errors.mjs';
import {
    assertAbsolutePath,
    assertPort,
    deepFreeze,
    probeEnv,
    probeVersion,
    recommendedFor,
    validateParams
} from './params.mjs';

const ID = 'ollama';
const EXECUTABLE = '/opt/ollama/bin/ollama';
const NVIDIA_LIB_DIR = '/usr/local/nvidia/lib64';
const OLLAMA_DIGEST_RE = /^sha256:[0-9a-f]{64}$/;

const paramSchema = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        numCtx: {
            type: 'integer', minimum: 512, maximum: 131072, default: 4096,
            title: 'Context size',
            description: 'Context window in tokens (num_ctx).'
        },
        numGpu: {
            type: ['integer', 'null'], minimum: 0, maximum: 999, default: null,
            title: 'GPU layers',
            description: 'Number of whole layers offloaded to the GPU; empty lets Ollama decide the split.'
        },
        numThread: {
            type: ['integer', 'null'], minimum: 1, maximum: 256, default: null,
            title: 'CPU threads',
            description: 'Number of CPU threads; empty uses the Ollama default.'
        },
        flashAttention: {
            type: ['boolean', 'null'], default: null,
            title: 'Flash attention',
            description: 'Enable or disable flash attention; empty uses the Ollama default.'
        },
        kvCacheType: {
            type: 'string', enum: ['f16', 'q8_0', 'q4_0'], default: 'f16',
            title: 'KV cache type',
            description: 'Quantization type of the KV cache.'
        },
        keepAlive: {
            type: 'string', pattern: '^(-1|0|[1-9][0-9]{0,4}[smh])$', maxLength: 6, default: '30m',
            title: 'Keep alive',
            description: 'How long the model stays loaded: -1 forever, 0 unload immediately, or e.g. 30m.'
        }
    }
});

// The cpu profile (DS005): no GPU is used, so a layer count, flash attention and a KV cache type are
// not offered. The KV cache is f16 (OLLAMA_KV_CACHE_TYPE) and Ollama's own default decides flash attention.
const cpuParamSchema = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        numCtx: {
            type: 'integer', minimum: 512, maximum: 131072, default: 4096,
            title: 'Context size',
            description: 'Context window in tokens (num_ctx). The KV cache lives in this machine\'s memory, so a larger context needs more of it.'
        },
        numThread: paramSchema.properties.numThread,
        keepAlive: paramSchema.properties.keepAlive
    }
});

// The dedicated schema, and the cpu one. Other profiles have no policy: on unified memory the
// runner is refused by admission, as it always was.
function paramSchemaFor(profile = 'dedicated') {
    if (profile === 'cpu') return cpuParamSchema;
    return profile === 'dedicated' ? paramSchema : null;
}

// Only the cpu profile reads parameters of its own; every other profile reads what it always read.
function normalizeParams(params = {}, { model, profile = 'dedicated' } = {}) {
    if (profile === 'cpu') return validateParams(cpuParamSchema, params, { defaults: recommendedFor(model, ID, 'cpu') });
    return validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
}

function describeContext(params = {}, { model, profile = 'dedicated' } = {}) {
    const { numCtx } = normalizeParams(params, { model, profile });
    return { totalContext: numCtx, perRequestContext: numCtx, parallel: 1, kvUnified: false };
}

// Ollama parses a string keep_alive with Go's time.ParseDuration, which rejects a bare '-1';
// a JSON number is accepted and a negative one means "keep loaded forever".
function keepAliveValue(keepAlive) {
    return keepAlive === '-1' ? -1 : keepAlive;
}

export function requestOptions(params = {}, { model, profile } = {}) {
    const values = normalizeParams(params, { model, profile });
    const options = { num_ctx: values.numCtx };
    // On the cpu profile no layer goes to a GPU, whatever the environment shows (DS005).
    if (profile === 'cpu') {
        options.num_gpu = 0;
    } else if (values.numGpu != null) {
        options.num_gpu = values.numGpu;
    }
    if (values.numThread != null) {
        options.num_thread = values.numThread;
    }
    return { options, keep_alive: keepAliveValue(values.keepAlive) };
}

function parseVersion(output) {
    const match = /version is v?(\d+\.\d+\.\d+)/.exec(output);
    return match ? match[1] : null;
}

// The image's own binary when it has one (the amd64 image); otherwise the agent's lock may offer an
// install (the arm64 image has no Ollama, DS004), and detection asks the installer, which reads the
// cache and its install record. The probe is synchronous and the installer's answer is not, so only
// a caller that passes an installer can get a promise back.
function detect({ spawnSync = realSpawnSync, installer } = {}) {
    const inImage = probeVersion({
        spawnSync,
        executable: EXECUTABLE,
        args: ['--version'],
        env: probeEnv({ LD_LIBRARY_PATH: NVIDIA_LIB_DIR }),
        parse: parseVersion,
        pinnedVersion: ollamaRunner.pinnedVersion
    });
    if (inImage.installed || !installer?.installable(ID)) return inImage;
    return (async () => {
        const info = await installer.describe(ID);
        return info.installed
            ? { installed: true, version: info.version, reason: null }
            : { installed: false, version: null, reason: 'Not installed. An admin can Install it under Runners.' };
    })();
}

// `runnerDir` is the runnable copy of an install from the agent's lock (DS004); the image's binary otherwise.
// On the cpu profile no driver library is on the path and no device is visible, so the CUDA driver is not
// loaded even where a GPU is attached but unusable (DS003).
function buildLaunch({ params, port, dataDir, model, profile, runnerDir } = {}) {
    const values = normalizeParams(params, { model, profile });
    const root = assertAbsolutePath(dataDir, 'dataDir');
    const cpu = profile === 'cpu';
    const command = runnerDir ? path.join(assertAbsolutePath(runnerDir, 'runnerDir'), 'bin', 'ollama') : EXECUTABLE;
    const env = {
        HOME: `${root}/home`,
        OLLAMA_MODELS: `${root}/models/ollama`,
        OLLAMA_HOST: `127.0.0.1:${assertPort(port)}`,
        OLLAMA_NUM_PARALLEL: '1',
        OLLAMA_MAX_LOADED_MODELS: '1',
        OLLAMA_KV_CACHE_TYPE: values.kvCacheType ?? 'f16',
        // The OpenAI-compatible /v1 endpoint takes no `options`; this makes
        // num_ctx the server default so a chat request does not reload the
        // model with a different context.
        OLLAMA_CONTEXT_LENGTH: String(values.numCtx),
        OLLAMA_KEEP_ALIVE: values.keepAlive,
        OLLAMA_NO_CLOUD: '1',
        ...(cpu ? { CUDA_VISIBLE_DEVICES: '' } : { LD_LIBRARY_PATH: NVIDIA_LIB_DIR })
    };
    if (values.flashAttention != null) {
        env.OLLAMA_FLASH_ATTENTION = values.flashAttention ? '1' : '0';
    }
    return { command, args: ['serve'], env };
}

// Progress of `/api/pull`, one JSON object per line. Every blob digest the
// pull reports is recorded with the tag, so its partial files can be counted
// and deleted per tag.
async function streamPull(ctx, base, tag) {
    const response = await ctx.fetch(`${base}/api/pull`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: tag, stream: true }),
        signal: ctx.signal,
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
            if (typeof event.digest === 'string' && OLLAMA_DIGEST_RE.test(event.digest)) ctx.store.recordPull(tag, event.digest);
            if (event.digest && event.total) layers.set(event.digest, { total: event.total, completed: event.completed || 0 });
            const total = [...layers.values()].reduce((sum, layer) => sum + layer.total, 0);
            const bytes = [...layers.values()].reduce((sum, layer) => sum + layer.completed, 0);
            const elapsed = Math.max(1, (Date.now() - started) / 1000);
            ctx.progress({ bytes, total, rate: bytes / elapsed, etaSeconds: null, transferred: bytes });
            if (event.status === 'success') return;
        }
    }
    throw new LocalLlmError('pull_failed', 'The Ollama pull stream ended before success.');
}

// The server runs first and fetches the weights itself: start `ollama serve`,
// pull the tag unless it is complete, check the pinned manifest digest,
// re-check admission, then load the model and confirm Ollama keeps it loaded.
async function start(ctx) {
    const process = ctx.launch(ctx.runner.buildLaunch({
        params: ctx.params, port: ctx.port, dataDir: ctx.dataDir, model: ctx.model, profile: ctx.profile, runnerDir: ctx.runnerDir,
    }));
    const base = `http://127.0.0.1:${ctx.port}`;
    await ctx.waitForHttp(`${base}/api/version`, { process });
    const tag = ctx.artifact.tag;
    const cached = ctx.store.readManifest(tag);
    if (!cached?.complete) {
        ctx.setPhase('downloading');
        await streamPull(ctx, base, tag);
    } else {
        ctx.record({ download: { bytes: cached.size, total: cached.size, rate: 0, etaSeconds: null, transferred: 0 } });
    }
    const pulled = ctx.store.readManifest(tag);
    if (!pulled?.complete) throw new LocalLlmError('pull_failed', `Ollama reports ${tag} but its files are incomplete.`);
    ctx.store.clearPulls(tag);
    if (ctx.artifact.manifestDigest && pulled.manifestDigest !== ctx.artifact.manifestDigest) {
        throw new LocalLlmError('identity_changed', `The Ollama tag ${tag} now resolves to ${pulled.manifestDigest}, `
            + `not the pinned ${ctx.artifact.manifestDigest}; update the model entry to accept it.`);
    }
    ctx.record({ resolved: { manifestDigest: pulled.manifestDigest, blobs: pulled.blobs.map((blob) => blob.digest) } });
    ctx.throwIfAborted();
    await ctx.recheckAdmission();
    ctx.throwIfAborted();
    ctx.setPhase('loading');
    const load = await ctx.fetch(`${base}/api/generate`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: tag, prompt: '', stream: false, ...ctx.runner.requestOptions(ctx.params, { profile: ctx.profile }) }),
        signal: ctx.signal,
    });
    if (!load.ok) throw new LocalLlmError('load_failed', `Ollama could not load ${tag} (HTTP ${load.status}).`);
    await load.json().catch(() => null);
    const ps = await (await ctx.fetch(`${base}/api/ps`, { signal: ctx.signal })).json();
    const loaded = (ps.models || []).find((entry) => entry.name === tag || entry.model === tag);
    if (!loaded) throw new LocalLlmError('load_failed', `Ollama did not keep ${tag} loaded.`);
    return { ollama: { sizeBytes: loaded.size, sizeVramBytes: loaded.size_vram, contextLength: loaded.context_length } };
}

export const ollamaRunner = Object.freeze({
    id: ID,
    displayName: 'Ollama',
    weightFormat: 'ollama',
    pinnedVersion: '0.34.4',
    supported: true,
    executable: EXECUTABLE,
    port: 18434,
    // Ollama has no API key; it listens only on loopback inside the
    // container, and the Router's agent-port relay is closed (DS001).
    apiKey: false,
    paramSchema,
    paramSchemaFor,
    basicParams: Object.freeze(['numCtx']),
    moeParams: Object.freeze([]),
    normalizeParams,
    describeContext,
    detect,
    buildLaunch,
    start,
    chatModel: (deployment) => deployment.artifact.tag,
    requestOptions,
    admit: admitOllama,
    admitCpu: (input) => admitCpuOllama(input, admissionResult),
    parseReport: parseRunnerReport
});
