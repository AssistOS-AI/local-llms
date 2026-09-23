import { codedError, deepFreeze, recommendedFor, validateParams } from './params.mjs';

const ID = 'lmstudio';
const UNSUPPORTED_REASON = 'Not supported or tested in this release; installable in a later release.';

const paramSchema = deepFreeze({
    type: 'object',
    additionalProperties: false,
    properties: {
        contextLength: {
            type: 'integer', minimum: 512, maximum: 131072, default: 4096,
            title: 'Context length',
            description: 'Context window in tokens.'
        },
        gpu: {
            type: ['string', 'number'], enum: ['off', 'max'], minimum: 0, maximum: 1, default: 'max',
            title: 'GPU offload',
            description: 'GPU offload: "off", "max", or a fraction between 0 and 1.'
        },
        parallel: {
            type: 'integer', minimum: 1, maximum: 16, default: 1,
            title: 'Parallel requests',
            description: 'Number of concurrent requests.'
        },
        ttl: {
            type: 'integer', minimum: 0, maximum: 86400, default: 3600,
            title: 'Idle TTL (seconds)',
            description: 'Seconds of inactivity before the model is unloaded.'
        }
    }
});

function normalizeParams(params = {}, { model } = {}) {
    return validateParams(paramSchema, params, { defaults: recommendedFor(model, ID) });
}

function describeContext(params = {}, { model } = {}) {
    const { contextLength, parallel } = normalizeParams(params, { model });
    return { totalContext: contextLength, perRequestContext: contextLength, parallel, kvUnified: false };
}

function detect() {
    return { installed: false, version: null, reason: UNSUPPORTED_REASON };
}

function buildLaunch() {
    throw codedError('runner_unsupported', `Runner ${ID} is not supported: ${UNSUPPORTED_REASON}`, { runner: ID });
}

export const lmStudioRunner = Object.freeze({
    id: ID,
    displayName: 'LM Studio',
    weightFormat: 'gguf',
    pinnedVersion: null,
    supported: false,
    executable: null,
    paramSchema,
    normalizeParams,
    describeContext,
    detect,
    buildLaunch
});
