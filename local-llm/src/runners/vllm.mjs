import { codedError, deepFreeze, recommendedFor, validateParams } from './params.mjs';

const ID = 'vllm';
const UNSUPPORTED_REASON = 'Not supported in this release: vLLM can be installed now, and running models on it arrives in a later release.';

const paramSchema = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        gpuMemoryUtilization: {
            type: 'number', minimum: 0.1, maximum: 0.99, default: 0.9,
            title: 'GPU memory utilization',
            description: 'Fraction of GPU memory vLLM may use for weights and KV cache.'
        },
        maxModelLen: {
            type: 'integer', minimum: 512, maximum: 131072,
            title: 'Max model length',
            description: 'Maximum context length per sequence; empty uses the model config.'
        },
        dtype: {
            type: 'string', enum: ['auto', 'float16', 'bfloat16'], default: 'auto',
            title: 'Data type',
            description: 'Data type for model weights and activations.'
        },
        quantization: {
            type: ['string', 'null'], pattern: '^[a-z0-9_]{1,32}$', maxLength: 32, default: null,
            title: 'Quantization',
            description: 'Quantization method name (e.g. awq, gptq); empty autodetects.'
        },
        kvCacheDtype: {
            type: 'string', enum: ['auto', 'fp8'], default: 'auto',
            title: 'KV cache type',
            description: 'Data type of the KV cache.'
        },
        cpuOffloadGb: {
            type: 'number', minimum: 0, maximum: 512, default: 0,
            title: 'CPU offload (GB)',
            description: 'GiB of model weights to offload to CPU RAM.'
        },
        cpuOffloadParams: {
            type: ['string', 'null'], pattern: '^[A-Za-z0-9_.*,-]{1,256}$', maxLength: 256, default: null,
            title: 'CPU offload parameters',
            description: 'Comma-separated parameter name patterns to offload to CPU.'
        },
        enforceEager: {
            type: 'boolean', default: false,
            title: 'Enforce eager mode',
            description: 'Disable CUDA graphs and always run in eager mode.'
        },
        maxNumSeqs: {
            type: 'integer', minimum: 1, maximum: 256, default: 1,
            title: 'Max sequences',
            description: 'Maximum number of sequences processed concurrently.'
        }
    }
});

function normalizeParams(params = {}, { model } = {}) {
    return validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
}

function describeContext(params = {}, { model } = {}) {
    const { maxModelLen, maxNumSeqs } = normalizeParams(params, { model });
    const context = maxModelLen ?? null;
    // vLLM's paged KV pool is shared by all sequences; each may grow up to maxModelLen.
    return { totalContext: context, perRequestContext: context, parallel: maxNumSeqs, kvUnified: true };
}

// vLLM is installed on demand (DS004), so detection asks the installer, which
// reads the cache and its install record; it never imports Python packages.
async function detect({ installer } = {}) {
    if (!installer?.installable(ID)) {
        return { installed: false, version: null, reason: 'This image\'s runner lock has no vLLM entry.' };
    }
    const info = await installer.describe(ID);
    if (!info.installed) return { installed: false, version: null, reason: 'Not installed. An admin can Install it under Runners.' };
    return { installed: true, version: info.version, reason: UNSUPPORTED_REASON };
}

function buildLaunch() {
    throw codedError('runner_unsupported', `Runner ${ID} is not supported: ${UNSUPPORTED_REASON}`, { runner: ID });
}

// Installable on demand since runners plan Phase R2; running it arrives in
// Phase R5. It reads Hugging Face snapshots, which the catalog does not offer
// yet, so no model lists it, and the controller refuses to run it with
// UNSUPPORTED_REASON.
const vllmRuntime = Object.freeze({
    id: ID,
    displayName: 'vLLM',
    weightFormat: 'hf',
    pinnedVersion: null,
    supported: false,
    unsupportedReason: UNSUPPORTED_REASON,
    executable: null,
    port: null,
    apiKey: true,
    paramSchema,
    basicParams: Object.freeze(['maxModelLen']),
    moeParams: Object.freeze([]),
    normalizeParams,
    describeContext,
    detect,
    buildLaunch
});

export { vllmRuntime, vllmRuntime as vllmRunner };
