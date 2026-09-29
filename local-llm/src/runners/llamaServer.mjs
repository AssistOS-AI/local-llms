// The llama-server family: llama.cpp and ik_llama.cpp serve the same GGUF
// files with the same OpenAI-compatible server and the same parameters, but
// their command lines differ. A dialect names the differences; everything
// else, the readiness probes included, is shared.

import { spawnSync as realSpawnSync } from 'node:child_process';

import { admitLlamaServer, admissionResult } from '../controller/admission.mjs';
import { defaultThreads, performanceCoreCount, physicalCoreCount } from '../controller/hardware.mjs';
import { UNIFIED, admitUnifiedLlamaServer } from '../controller/profiles.mjs';
import { parseRunnerReport } from '../controller/runnerProcess.mjs';
import {
    ParamError,
    assertAbsolutePath,
    assertApiKey,
    assertPort,
    codedError,
    deepFreeze,
    probeEnv,
    probeVersion,
    recommendedFor,
    validateParams
} from './params.mjs';

const NVIDIA_LIB_DIR = '/usr/local/nvidia/lib64';
const KV_CACHE_TYPES = ['f16', 'q8_0', 'q4_0'];
// llama.cpp b10105 added --load-mode; b10875 removed --no-mmap, --mmap, --mlock and --direct-io.
export const LOAD_MODES = Object.freeze(['auto', 'none', 'mmap', 'mlock', 'mmap+mlock', 'dio']);

const LOAD_MODE_PARAM = Object.freeze({
    type: 'string', enum: LOAD_MODES, default: 'auto',
    title: 'Load mode',
    description: 'How the weights are read: auto (the runner decides), none (read without mmap), mmap, mlock, mmap+mlock, or dio (read with direct I/O, bypassing the page cache).'
});

const CHAT_TEMPLATE_KWARGS_PARAM = Object.freeze({
    type: 'object',
    additionalProperties: false,
    title: 'Chat template arguments',
    description: 'Extra arguments passed to the Jinja chat template.',
    properties: {
        reasoning_effort: {
            type: 'string', enum: ['low', 'medium', 'high'],
            title: 'Reasoning effort',
            description: 'Reasoning effort hint for models whose template supports it.'
        },
        preserve_thinking: {
            type: 'boolean',
            title: 'Preserve thinking',
            description: 'Keep earlier reasoning in the prompt, for templates that support it (Qwen3.6 with MTP).'
        }
    }
});

export const LLAMA_SERVER_PARAM_SCHEMA = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        ctxSize: {
            type: 'integer', minimum: 512, maximum: 131072, default: 16384,
            title: 'Context size',
            description: 'Total context window in tokens, shared by all parallel slots.'
        },
        nGpuLayers: {
            type: 'integer', minimum: 0, maximum: 999, default: 99,
            title: 'GPU layers',
            description: 'Number of model layers offloaded to the GPU (99 or more means all).'
        },
        nCpuMoe: {
            type: 'integer', minimum: 0, maximum: 256, default: 0,
            title: 'CPU MoE layers',
            description: 'Number of layers whose MoE expert weights stay in CPU RAM (0 keeps all on GPU).'
        },
        flashAttn: {
            type: 'string', enum: ['auto', 'on', 'off'], default: 'auto',
            title: 'Flash attention',
            description: 'Flash attention mode.'
        },
        cacheTypeK: {
            type: 'string', enum: KV_CACHE_TYPES, default: 'f16',
            title: 'K cache type',
            description: 'Quantization type of the KV cache keys.'
        },
        cacheTypeV: {
            type: 'string', enum: KV_CACHE_TYPES, default: 'f16',
            title: 'V cache type',
            description: 'Quantization type of the KV cache values.'
        },
        threads: {
            type: ['integer', 'null'], minimum: 1, maximum: 256, default: null,
            title: 'CPU threads',
            description: 'Number of CPU threads for generation; empty uses the physical CPU cores minus 2.'
        },
        parallel: {
            type: 'integer', minimum: 1, maximum: 16, default: 1,
            title: 'Parallel slots',
            description: 'Number of concurrent request slots; above 1 the slots share one unified KV cache.'
        },
        batchSize: {
            type: 'integer', minimum: 32, maximum: 8192, default: 2048,
            title: 'Batch size',
            description: 'Logical maximum batch size for prompt processing.'
        },
        ubatchSize: {
            type: 'integer', minimum: 32, maximum: 4096, default: 512,
            title: 'Micro-batch size',
            description: 'Physical maximum batch size; must not exceed the batch size.'
        },
        loadMode: LOAD_MODE_PARAM,
        chatTemplateKwargs: CHAT_TEMPLATE_KWARGS_PARAM
    }
});

// The unified profile (DS005): every flag that changes memory is fixed at the
// values the model's envelope was measured with (-ngl 999, -fa on, f16 K and V,
// -b and -ub 2048, --cache-ram); what an admin may choose is inside it.
export const LLAMA_SERVER_UNIFIED_PARAM_SCHEMA = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        ctxSize: {
            type: 'integer', minimum: 512, maximum: 262144, default: 32768,
            title: 'Context size',
            description: 'Total context window in tokens, shared by all parallel slots (at most the model\'s training context).'
        },
        parallel: {
            type: 'integer', minimum: 1, maximum: 16, default: 1,
            title: 'Parallel slots',
            description: 'Number of concurrent request slots; above 1 the slots share one unified KV cache.'
        },
        loadMode: {
            type: 'string', enum: ['dio', 'none'], default: 'dio',
            title: 'Load mode',
            description: 'dio reads the weights with direct I/O, so loading does not fill the page cache; none reads them through it.'
        },
        mtp: {
            type: 'boolean', default: false,
            title: 'MTP speculative decoding',
            description: 'Draft with the model\'s own multi-token-prediction head (--spec-type draft-mtp --spec-draft-n-max 3); only for models that have one. On by default where the catalog recommends it (Qwen3.6 MTP).'
        },
        threads: {
            type: ['integer', 'null'], minimum: 1, maximum: 256, default: null,
            title: 'CPU threads',
            description: 'Number of CPU threads; empty uses the high-performance cores (physical cores minus 2 where all cores are alike).'
        },
        chatTemplateKwargs: CHAT_TEMPLATE_KWARGS_PARAM
    }
});

const UNIFIED_BATCH = UNIFIED.batchSize;

function assertModelId(model) {
    if (typeof model?.id !== 'string' || model.id.length === 0) {
        throw codedError('invalid_launch', 'model.id must be a non-empty string', { field: 'model.id' });
    }
    return model.id;
}

/**
 * A llama-server runner.
 *
 * dialect.quietArgs      flags that turn off the web UI and set the log level
 * dialect.unifiedKv      whether parallel slots share one KV cache (--kv-unified)
 * dialect.loadModes      the load modes the server accepts
 * dialect.loadArgs(mode) the flags for a load mode
 * dialect.jinja(model)   whether to pass --jinja
 * dialect.parseVersion   the version from `--version` output, or null
 * dialect.unified        whether the runner has a unified-memory policy (DS005)
 *
 * cpuCores() counts the physical cores for the dedicated default --threads,
 * perfCores() the high-performance cores for the unified one; each is read
 * once, on the first launch that needs it.
 */
export function createLlamaServerRunner({
    id, displayName, executable, pinnedVersion, port, dialect,
    cpuCores = physicalCoreCount, perfCores = performanceCoreCount,
}) {
    const loadModes = dialect.loadModes || LOAD_MODES;
    const paramSchema = deepFreeze({
        ...LLAMA_SERVER_PARAM_SCHEMA,
        properties: {
            ...LLAMA_SERVER_PARAM_SCHEMA.properties,
            loadMode: { ...LOAD_MODE_PARAM, enum: loadModes.filter((mode) => LOAD_MODES.includes(mode)) },
        },
    });
    const schemas = Object.freeze({ dedicated: paramSchema, unified: dialect.unified ? LLAMA_SERVER_UNIFIED_PARAM_SCHEMA : null });
    let autoThreads = null;
    let unifiedThreads = null;
    // The servers default to few threads; physical cores minus 2 is faster
    // here and leaves room for the agent and the host (runners plan, I2). On
    // unified memory with cores of different capacity, the high-performance
    // cores. An admin-set value wins; the stored parameters keep "empty".
    const threadsFor = (values, profile) => values.threads ?? (profile === 'unified'
        ? (unifiedThreads ??= perfCores() ?? defaultThreads(cpuCores()))
        : (autoThreads ??= defaultThreads(cpuCores())));

    function paramSchemaFor(profile = 'dedicated') {
        return schemas[profile] ?? null;
    }

    function normalizeParams(params = {}, { model, profile = 'dedicated' } = {}) {
        const schema = paramSchemaFor(profile);
        if (!schema) throw new ParamError('(runner)', `${displayName} has no parameters for the ${profile} profile`);
        const values = validateParams(schema, params, { defaults: recommendedFor(model, id, profile) });
        if (profile === 'dedicated' && values.ubatchSize > values.batchSize) {
            throw new ParamError('ubatchSize', `must be <= batchSize (${values.batchSize})`);
        }
        // An explicit value wins over a default, but cannot make an unsupported
        // configuration valid: MTP needs weights with a prediction head.
        if (values.mtp && model?.mtp !== true) {
            throw new ParamError('mtp', `${model?.displayName || 'This model'} has no multi-token-prediction head (its entry does not declare mtp), `
                + 'so MTP cannot be turned on; set mtp to false, or declare mtp in the model entry if its weights have one');
        }
        if (Number.isInteger(model?.contextLength) && values.ctxSize > model.contextLength * values.parallel) {
            throw new ParamError('ctxSize', `must be <= ${model.contextLength * values.parallel} (the model's training context of `
                + `${model.contextLength} tokens per slot)`);
        }
        return values;
    }

    function describeContext(params = {}, { model, profile = 'dedicated' } = {}) {
        const { ctxSize, parallel } = normalizeParams(params, { model, profile });
        // A unified KV cache lets every slot use the whole context; otherwise
        // each slot gets an equal share of it.
        const perRequestContext = dialect.unifiedKv ? ctxSize : Math.floor(ctxSize / parallel);
        return { totalContext: ctxSize, perRequestContext, parallel, kvUnified: dialect.unifiedKv && parallel > 1 };
    }

    function detect({ spawnSync = realSpawnSync } = {}) {
        return probeVersion({
            spawnSync,
            executable,
            args: ['--version'],
            env: probeEnv({ LD_LIBRARY_PATH: NVIDIA_LIB_DIR }),
            parse: dialect.parseVersion,
            pinnedVersion
        });
    }

    function tuningArgs(values, profile) {
        if (profile === 'unified') {
            const args = ['--ctx-size', values.ctxSize, '--n-gpu-layers', 999, '--flash-attn', 'on',
                '--cache-type-k', 'f16', '--cache-type-v', 'f16', '--threads', threadsFor(values, profile), '-np', values.parallel];
            if (values.parallel > 1) args.push('--kv-unified');
            args.push('--batch-size', UNIFIED_BATCH, '--ubatch-size', UNIFIED_BATCH, '--cache-ram', UNIFIED.cacheRamMiB);
            if (values.mtp) args.push('--spec-type', 'draft-mtp', '--spec-draft-n-max', 3);
            return args;
        }
        const args = ['--ctx-size', values.ctxSize, '--n-gpu-layers', values.nGpuLayers];
        if (values.nCpuMoe > 0) {
            args.push('--n-cpu-moe', values.nCpuMoe);
        }
        args.push('--flash-attn', values.flashAttn);
        args.push('--cache-type-k', values.cacheTypeK, '--cache-type-v', values.cacheTypeV);
        args.push('--threads', threadsFor(values, profile));
        args.push('-np', values.parallel);
        if (dialect.unifiedKv && values.parallel > 1) {
            args.push('--kv-unified');
        }
        args.push('--batch-size', values.batchSize, '--ubatch-size', values.ubatchSize);
        return args;
    }

    function extraArgs(values, model) {
        const args = [...dialect.loadArgs(values.loadMode)];
        if (values.chatTemplateKwargs) {
            args.push('--chat-template-kwargs', JSON.stringify(values.chatTemplateKwargs));
        }
        if (dialect.jinja(model)) {
            args.push('--jinja');
        }
        return args;
    }

    function buildLaunch({ artifactPath, params, port: launchPort, apiKey, model, profile = 'dedicated' } = {}) {
        const values = normalizeParams(params, { model, profile });
        const args = [
            '-m', assertAbsolutePath(artifactPath, 'artifactPath'),
            '--host', '127.0.0.1',
            '--port', assertPort(launchPort),
            '--api-key', assertApiKey(apiKey),
            ...dialect.quietArgs,
            '--alias', assertModelId(model),
            ...tuningArgs(values, profile),
            ...extraArgs(values, model)
        ];
        return { command: executable, args: args.map(String), env: { LD_LIBRARY_PATH: NVIDIA_LIB_DIR } };
    }

    // /health answers once the model is loaded (llama.cpp returns 503 while it
    // loads; ik_llama.cpp accepts the connection and answers only then, so each
    // probe's own timeout keeps polling); /v1/models then answers with the key.
    async function start(ctx) {
        const process = ctx.launch(ctx.runner.buildLaunch({
            artifactPath: ctx.weights.path, params: ctx.params, port: ctx.port, apiKey: ctx.apiKey, model: ctx.model,
            profile: ctx.profile,
        }));
        const base = `http://127.0.0.1:${ctx.port}`;
        await ctx.waitForHttp(`${base}/health`, { process });
        await ctx.waitForHttp(`${base}/v1/models`, { headers: { authorization: `Bearer ${ctx.apiKey}` }, process });
        return {};
    }

    return Object.freeze({
        id,
        displayName,
        weightFormat: 'gguf',
        pinnedVersion,
        supported: true,
        executable,
        port,
        apiKey: true,
        paramSchema,
        paramSchemaFor,
        // Shown outside "Advanced" in the Run form; nCpuMoe only for MoE models.
        basicParams: Object.freeze(['ctxSize', 'nCpuMoe']),
        moeParams: Object.freeze(['nCpuMoe']),
        normalizeParams,
        describeContext,
        detect,
        buildLaunch,
        start,
        // --alias makes the model id the name the server answers to.
        chatModel: (deployment) => deployment.modelId,
        admit: admitLlamaServer,
        ...(dialect.unified ? {
            admitUnified: (input) => admitUnifiedLlamaServer({ ...input, runnerId: id, displayName }, admissionResult),
        } : {}),
        parseReport: parseRunnerReport
    });
}
