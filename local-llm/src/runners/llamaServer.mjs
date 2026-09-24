// The llama-server family: llama.cpp and ik_llama.cpp serve the same GGUF
// files with the same OpenAI-compatible server and the same parameters, but
// their command lines differ. A dialect names the differences; everything
// else, the readiness probes included, is shared.

import { spawnSync as realSpawnSync } from 'node:child_process';

import { admitLlamaServer } from '../controller/admission.mjs';
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
            description: 'Number of CPU threads for generation; empty uses the runner default.'
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
        mlock: {
            type: 'boolean', default: false,
            title: 'Lock model in RAM',
            description: 'Lock model weights in RAM so they cannot be swapped out.'
        },
        noMmap: {
            type: 'boolean', default: false,
            title: 'Disable mmap',
            description: 'Read the model into memory instead of memory-mapping the file.'
        },
        chatTemplateKwargs: {
            type: 'object',
            additionalProperties: false,
            title: 'Chat template arguments',
            description: 'Extra arguments passed to the Jinja chat template.',
            properties: {
                reasoning_effort: {
                    type: 'string', enum: ['low', 'medium', 'high'],
                    title: 'Reasoning effort',
                    description: 'Reasoning effort hint for models whose template supports it.'
                }
            }
        }
    }
});

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
 * dialect.loadArgs(v)    flags for mlock/noMmap
 * dialect.jinja(model)   whether to pass --jinja
 * dialect.parseVersion   the version from `--version` output, or null
 */
export function createLlamaServerRunner({ id, displayName, executable, pinnedVersion, port, dialect }) {
    const paramSchema = LLAMA_SERVER_PARAM_SCHEMA;

    function normalizeParams(params = {}, { model } = {}) {
        const values = validateParams(paramSchema, params, { defaults: recommendedFor(model, id) });
        if (values.ubatchSize > values.batchSize) {
            throw new ParamError('ubatchSize', `must be <= batchSize (${values.batchSize})`);
        }
        return values;
    }

    function describeContext(params = {}, { model } = {}) {
        const { ctxSize, parallel } = normalizeParams(params, { model });
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

    function tuningArgs(values) {
        const args = ['--ctx-size', values.ctxSize, '--n-gpu-layers', values.nGpuLayers];
        if (values.nCpuMoe > 0) {
            args.push('--n-cpu-moe', values.nCpuMoe);
        }
        args.push('--flash-attn', values.flashAttn);
        args.push('--cache-type-k', values.cacheTypeK, '--cache-type-v', values.cacheTypeV);
        if (values.threads != null) {
            args.push('--threads', values.threads);
        }
        args.push('-np', values.parallel);
        if (dialect.unifiedKv && values.parallel > 1) {
            args.push('--kv-unified');
        }
        args.push('--batch-size', values.batchSize, '--ubatch-size', values.ubatchSize);
        return args;
    }

    function extraArgs(values, model) {
        const args = [...dialect.loadArgs(values)];
        if (values.chatTemplateKwargs) {
            args.push('--chat-template-kwargs', JSON.stringify(values.chatTemplateKwargs));
        }
        if (dialect.jinja(model)) {
            args.push('--jinja');
        }
        return args;
    }

    function buildLaunch({ artifactPath, params, port: launchPort, apiKey, model } = {}) {
        const values = normalizeParams(params, { model });
        const args = [
            '-m', assertAbsolutePath(artifactPath, 'artifactPath'),
            '--host', '127.0.0.1',
            '--port', assertPort(launchPort),
            '--api-key', assertApiKey(apiKey),
            ...dialect.quietArgs,
            '--alias', assertModelId(model),
            ...tuningArgs(values),
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
        parseReport: parseRunnerReport
    });
}
