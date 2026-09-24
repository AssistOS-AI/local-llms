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

const ID = 'llama.cpp';
const EXECUTABLE = '/opt/llama.cpp/llama-server';
const NVIDIA_LIB_DIR = '/usr/local/nvidia/lib64';
const KV_CACHE_TYPES = ['f16', 'q8_0', 'q4_0'];

const paramSchema = deepFreeze({
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

function normalizeParams(params = {}, { model } = {}) {
    const values = validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
    if (values.ubatchSize > values.batchSize) {
        throw new ParamError('ubatchSize', `must be <= batchSize (${values.batchSize})`);
    }
    return values;
}

function describeContext(params = {}, { model } = {}) {
    const { ctxSize, parallel } = normalizeParams(params, { model });
    // With --kv-unified every slot can use the whole context, so it is not divided by parallel.
    return { totalContext: ctxSize, perRequestContext: ctxSize, parallel, kvUnified: parallel > 1 };
}

function parseVersion(output) {
    const match = /\bbuild (\d+)\b/.exec(output);
    return match ? `b${match[1]}` : null;
}

function detect({ spawnSync = realSpawnSync } = {}) {
    return probeVersion({
        spawnSync,
        executable: EXECUTABLE,
        args: ['--version'],
        env: probeEnv({ LD_LIBRARY_PATH: NVIDIA_LIB_DIR }),
        parse: parseVersion,
        pinnedVersion: llamaCppRunner.pinnedVersion
    });
}

// b11125 replaced --mlock/--no-mmap with a single --load-mode option.
function loadMode({ mlock, noMmap }) {
    if (noMmap && mlock) {
        return 'mlock';
    }
    if (noMmap) {
        return 'none';
    }
    return mlock ? 'mmap+mlock' : null;
}

function assertModelId(model) {
    if (typeof model?.id !== 'string' || model.id.length === 0) {
        throw codedError('invalid_launch', 'model.id must be a non-empty string', { field: 'model.id' });
    }
    return model.id;
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
    if (values.parallel > 1) {
        args.push('--kv-unified');
    }
    args.push('--batch-size', values.batchSize, '--ubatch-size', values.ubatchSize);
    return args;
}

function extraArgs(values, model) {
    const args = [];
    const mode = loadMode(values);
    if (mode) {
        args.push('--load-mode', mode);
    }
    if (values.chatTemplateKwargs) {
        args.push('--chat-template-kwargs', JSON.stringify(values.chatTemplateKwargs));
    }
    if (model?.requiresJinja) {
        args.push('--jinja');
    }
    return args;
}

function buildLaunch({ artifactPath, params, port, apiKey, model } = {}) {
    const values = normalizeParams(params, { model });
    const args = [
        '-m', assertAbsolutePath(artifactPath, 'artifactPath'),
        '--host', '127.0.0.1',
        '--port', assertPort(port),
        '--api-key', assertApiKey(apiKey),
        '--no-webui',
        '-lv', 4,
        '--alias', assertModelId(model),
        ...tuningArgs(values),
        ...extraArgs(values, model)
    ];
    return { command: EXECUTABLE, args: args.map(String), env: { LD_LIBRARY_PATH: NVIDIA_LIB_DIR } };
}

// llama-server answers /health with 503 while the model loads and 200 once it
// serves; /v1/models then proves that the per-start key is accepted.
async function start(ctx) {
    const process = ctx.launch(ctx.runner.buildLaunch({
        artifactPath: ctx.weights.path, params: ctx.params, port: ctx.port, apiKey: ctx.apiKey, model: ctx.model,
    }));
    const base = `http://127.0.0.1:${ctx.port}`;
    await ctx.waitForHttp(`${base}/health`, { process });
    await ctx.waitForHttp(`${base}/v1/models`, { headers: { authorization: `Bearer ${ctx.apiKey}` }, process });
    return {};
}

export const llamaCppRunner = Object.freeze({
    id: ID,
    displayName: 'llama.cpp',
    weightFormat: 'gguf',
    pinnedVersion: 'b11125',
    supported: true,
    executable: EXECUTABLE,
    port: 18080,
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
    // --alias makes the model id the name llama-server answers to.
    chatModel: (deployment) => deployment.modelId,
    admit: admitLlamaServer,
    parseReport: parseRunnerReport
});
