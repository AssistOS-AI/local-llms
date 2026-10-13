import { LocalLlmError } from '../errors.mjs';
import { validateHuggingFaceSource, validateModel } from './catalog.mjs';

export const METADATA_PROFILES = Object.freeze(['dedicated', 'unified', 'cpu', 'venus']);
export const CATALOG_V3 = 'local-llm.catalog/v3';
export const CATALOG_V4 = 'local-llm.catalog/v4';

function invalid(field, message) {
    throw new LocalLlmError('invalid_model', `${field}: ${message}`, { field });
}

function object(value, keys, field) {
    if (!value || typeof value !== 'object' || Array.isArray(value)
        || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) invalid(field, 'must be an object');
    for (const key of Object.keys(value)) if (!keys.includes(key)) invalid(`${field}.${key}`, 'unsupported field');
}

function integer(value, min, max, field) {
    if (!Number.isSafeInteger(value) || value < min || value > max) invalid(field, `must be an integer from ${min} to ${max}`);
}

function text(value, max, field) {
    if (typeof value !== 'string' || !value.trim() || value.length > max || /[\x00-\x1f\x7f]/.test(value)) invalid(field, 'must be nonempty bounded plain text');
}

export function validateVenusParams(value, field, { complete = false } = {}) {
    object(value, ['ctxSize', 'parallel', 'loadMode'], field);
    if (complete || value.ctxSize !== undefined) integer(value.ctxSize, 512, 131072, `${field}.ctxSize`);
    if (complete || value.parallel !== undefined) integer(value.parallel, 1, 4, `${field}.parallel`);
    if ((complete || value.loadMode !== undefined) && !['auto', 'none'].includes(value.loadMode)) invalid(`${field}.loadMode`, 'must be auto or none');
    return structuredClone(value);
}

function sourceIdentity(source) {
    return JSON.stringify([source.type, source.repo, source.commit, source.file, source.size, source.sha256,
        source.shards?.map(({ file, size, sha256 }) => [file, size, sha256])]);
}

export function validateVenusMeasurements(value, field, source) {
    if (!Array.isArray(value) || value.length < 1 || value.length > 16) invalid(field, 'must contain 1-16 measurements');
    const ids = new Set();
    return value.map((record, index) => {
        const at = `${field}[${index}]`;
        object(record, ['schema', 'id', 'measuredAt', 'evidence', 'modelSource', 'configuration'], at);
        if (record.schema !== 'local-llm.venus-measurement/v1') invalid(`${at}.schema`, 'unsupported measurement schema');
        if (typeof record.id !== 'string' || !/^[a-z0-9][a-z0-9._-]{1,63}$/.test(record.id) || ids.has(record.id)) invalid(`${at}.id`, 'invalid or duplicate id');
        ids.add(record.id);
        if (typeof record.measuredAt !== 'string' || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{3})?Z$/.test(record.measuredAt)
            || !Number.isFinite(Date.parse(record.measuredAt))
            || new Date(record.measuredAt).toISOString() !== record.measuredAt.replace(/(?<!\.\d{3})Z$/, '.000Z')) invalid(`${at}.measuredAt`, 'must be a valid UTC ISO timestamp');
        object(record.evidence, ['runId', 'manifestSha256', 'reference'], `${at}.evidence`);
        if (typeof record.evidence.runId !== 'string' || !/^[A-Za-z0-9._-]{1,128}$/.test(record.evidence.runId)) invalid(`${at}.evidence.runId`, 'invalid run id');
        if (typeof record.evidence.manifestSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(record.evidence.manifestSha256)) invalid(`${at}.evidence.manifestSha256`, 'must be a SHA-256');
        text(record.evidence.reference, 400, `${at}.evidence.reference`);
        const pinned = validateHuggingFaceSource(record.modelSource, `${at}.modelSource`, { requirePin: true });
        integer(pinned.size, 1, Number.MAX_SAFE_INTEGER, `${at}.modelSource.size`);
        if (!source || sourceIdentity(pinned) !== sourceIdentity(source)) invalid(`${at}.modelSource`, 'must match the enclosing pinned GGUF artifact');
        const config = record.configuration;
        object(config, ['params', 'resolvedThreads', 'hostChip', 'hostRamBytes', 'vmCpus', 'vmRamBytes'], `${at}.configuration`);
        validateVenusParams(config.params, `${at}.configuration.params`, { complete: true });
        for (const key of ['resolvedThreads', 'vmCpus']) integer(config[key], 1, 4096, `${at}.configuration.${key}`);
        for (const key of ['hostRamBytes', 'vmRamBytes']) integer(config[key], 1, Number.MAX_SAFE_INTEGER, `${at}.configuration.${key}`);
        text(config.hostChip, 120, `${at}.configuration.hostChip`);
        return { ...structuredClone(record), modelSource: pinned };
    });
}

export function measurementStatus(model, runnerId, params) {
    const history = model?.seed === true ? model.validated?.venus?.[runnerId] || [] : [];
    const records = history.map((record) => {
        const reasons = Object.entries(record.configuration.params)
            .filter(([key, value]) => params && params[key] !== value).map(([key]) => `${key} differs from the historical measurement`);
        return { ...structuredClone(record), applicability: reasons.length ? 'mismatch' : 'unknown', reasons };
    });
    return { state: records.length ? 'historical' : 'unverified',
        applicability: records.length && records.every((record) => record.applicability === 'mismatch') ? 'mismatch' : 'unknown',
        reasons: records.flatMap((record) => record.reasons.map((reason) => ({ id: record.id, reason }))), records };
}

export function exportModelMetadata(models, { seed = false, targetSchema = CATALOG_V4, dropVenusMetadata = false } = {}) {
    if (![CATALOG_V3, CATALOG_V4].includes(targetSchema)) throw new LocalLlmError('catalog_schema_unsupported', 'Unsupported target catalog schema.');
    if (!Array.isArray(models)) throw new LocalLlmError('invalid_model', 'models must be an array');
    const dropped = [];
    const normalized = models.map((input) => {
        const entry = structuredClone(validateModel(input, { seed }));
        delete entry.seed;
        if (targetSchema === CATALOG_V3) {
            for (const namespace of ['recommended', 'validated']) {
                if (Object.hasOwn(entry[namespace], 'venus')) {
                    dropped.push({ modelId: entry.id, field: `${namespace}.venus` });
                    delete entry[namespace].venus;
                }
            }
        }
        return entry;
    });
    if (dropped.length && !dropVenusMetadata) throw new LocalLlmError('catalog_downgrade_lossy',
        `Export would drop ${dropped.map(({ modelId, field }) => `${modelId}.${field}`).join(', ')}`, { dropped });
    return { schema: targetSchema, models: normalized, dropped };
}
