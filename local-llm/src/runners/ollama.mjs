import { spawnSync as realSpawnSync } from 'node:child_process';
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

function normalizeParams(params = {}, { model } = {}) {
    return validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
}

function describeContext(params = {}, { model } = {}) {
    const { numCtx } = normalizeParams(params, { model });
    return { totalContext: numCtx, perRequestContext: numCtx, parallel: 1, kvUnified: false };
}

// Ollama parses a string keep_alive with Go's time.ParseDuration, which rejects a bare '-1';
// a JSON number is accepted and a negative one means "keep loaded forever".
function keepAliveValue(keepAlive) {
    return keepAlive === '-1' ? -1 : keepAlive;
}

export function requestOptions(params = {}, { model } = {}) {
    const values = normalizeParams(params, { model });
    const options = { num_ctx: values.numCtx };
    if (values.numGpu != null) {
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

function detect({ spawnSync = realSpawnSync } = {}) {
    return probeVersion({
        spawnSync,
        executable: EXECUTABLE,
        args: ['--version'],
        env: probeEnv({ LD_LIBRARY_PATH: NVIDIA_LIB_DIR }),
        parse: parseVersion,
        pinnedVersion: ollamaRunner.pinnedVersion
    });
}

function buildLaunch({ params, port, dataDir, model } = {}) {
    const values = normalizeParams(params, { model });
    const root = assertAbsolutePath(dataDir, 'dataDir');
    const env = {
        HOME: `${root}/home`,
        OLLAMA_MODELS: `${root}/models/ollama`,
        OLLAMA_HOST: `127.0.0.1:${assertPort(port)}`,
        OLLAMA_NUM_PARALLEL: '1',
        OLLAMA_MAX_LOADED_MODELS: '1',
        OLLAMA_KV_CACHE_TYPE: values.kvCacheType,
        // The OpenAI-compatible /v1 endpoint takes no `options`; this makes
        // num_ctx the server default so a chat request does not reload the
        // model with a different context.
        OLLAMA_CONTEXT_LENGTH: String(values.numCtx),
        OLLAMA_KEEP_ALIVE: values.keepAlive,
        OLLAMA_NO_CLOUD: '1',
        LD_LIBRARY_PATH: NVIDIA_LIB_DIR
    };
    if (values.flashAttention != null) {
        env.OLLAMA_FLASH_ATTENTION = values.flashAttention ? '1' : '0';
    }
    return { command: EXECUTABLE, args: ['serve'], env };
}

export const ollamaRunner = Object.freeze({
    id: ID,
    displayName: 'Ollama',
    weightFormat: 'ollama',
    pinnedVersion: '0.34.3',
    supported: true,
    executable: EXECUTABLE,
    paramSchema,
    normalizeParams,
    describeContext,
    detect,
    buildLaunch,
    requestOptions
});
