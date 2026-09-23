import { isAbsolute } from 'node:path';

const API_KEY_PATTERN = /^[A-Za-z0-9_-]{32,128}$/;
const PROBE_TIMEOUT_MS = 10000;
const SNIPPET_LENGTH = 200;

export class ParamError extends Error {
    constructor(field, reason) {
        super(`Invalid parameter ${field}: ${reason}`);
        this.name = 'ParamError';
        this.code = 'invalid_params';
        this.details = { field, reason };
    }
}

export function codedError(code, message, details) {
    const error = new Error(message);
    error.code = code;
    if (details !== undefined) {
        error.details = details;
    }
    return error;
}

export function deepFreeze(value) {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
        Object.freeze(value);
        Object.values(value).forEach(deepFreeze);
    }
    return value;
}

export function isPlainObject(value) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
        return false;
    }
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
}

function typesOf(prop) {
    return Array.isArray(prop.type) ? prop.type : [prop.type];
}

// Picks the schema branch a value belongs to; constraints are then scoped to that branch
// (e.g. an enum applies only to the string side of ['string','number']).
function branchFor(types, value) {
    if (value === null) {
        return types.includes('null') ? 'null' : null;
    }
    if (typeof value === 'number') {
        if (types.includes('integer') && Number.isInteger(value)) {
            return 'integer';
        }
        return types.includes('number') && Number.isFinite(value) ? 'number' : null;
    }
    if (typeof value === 'boolean' || typeof value === 'string') {
        return types.includes(typeof value) ? typeof value : null;
    }
    return types.includes('object') && isPlainObject(value) ? 'object' : null;
}

function checkRange(prop, value, field) {
    if (Object.hasOwn(prop, 'minimum') && value < prop.minimum) {
        throw new ParamError(field, `must be >= ${prop.minimum}`);
    }
    if (Object.hasOwn(prop, 'maximum') && value > prop.maximum) {
        throw new ParamError(field, `must be <= ${prop.maximum}`);
    }
    return value;
}

function checkString(prop, value, field) {
    if (Object.hasOwn(prop, 'maxLength') && value.length > prop.maxLength) {
        throw new ParamError(field, `must be at most ${prop.maxLength} characters`);
    }
    if (Array.isArray(prop.enum) && !prop.enum.includes(value)) {
        throw new ParamError(field, `must be one of ${prop.enum.filter((v) => typeof v === 'string').join(', ')}`);
    }
    if (typeof prop.pattern === 'string' && !new RegExp(prop.pattern).test(value)) {
        throw new ParamError(field, `must match ${prop.pattern}`);
    }
    return value;
}

function checkValue(prop, value, field) {
    const types = typesOf(prop);
    const branch = branchFor(types, value);
    if (branch === null) {
        throw new ParamError(field, `must be of type ${types.join(' or ')}`);
    }
    if (branch === 'integer' || branch === 'number') {
        return checkRange(prop, value, field);
    }
    if (branch === 'string') {
        return checkString(prop, value, field);
    }
    if (branch === 'object') {
        return validateObject(prop, value, {}, `${field}.`);
    }
    return value;
}

function resolveValue(key, input, defaults, prop) {
    if (Object.hasOwn(input, key) && input[key] !== undefined) {
        return { present: true, value: input[key] };
    }
    if (Object.hasOwn(defaults, key) && defaults[key] !== undefined) {
        return { present: true, value: defaults[key] };
    }
    if (Object.hasOwn(prop, 'default')) {
        return { present: true, value: prop.default };
    }
    return { present: false };
}

function validateObject(schema, input, defaults, prefix) {
    if (!isPlainObject(input)) {
        throw new ParamError(prefix ? prefix.slice(0, -1) : '(root)', 'must be a plain object');
    }
    const properties = schema.properties || {};
    for (const key of Object.keys(input)) {
        if (!Object.hasOwn(properties, key)) {
            throw new ParamError(`${prefix}${key}`, 'unknown parameter');
        }
    }
    const result = {};
    for (const key of Object.keys(properties)) {
        const resolved = resolveValue(key, input, defaults, properties[key]);
        if (resolved.present) {
            result[key] = checkValue(properties[key], resolved.value, `${prefix}${key}`);
        }
    }
    return Object.freeze(result);
}

export function validateParams(schema, input, { defaults = {} } = {}) {
    if (!isPlainObject(defaults)) {
        throw new ParamError('(defaults)', 'must be a plain object');
    }
    return validateObject(schema, input, defaults, '');
}

export function recommendedFor(model, runnerId) {
    const recommended = model?.recommended;
    if (!isPlainObject(recommended) || !Object.hasOwn(recommended, runnerId)) {
        return {};
    }
    return recommended[runnerId] || {};
}

export function assertPort(port) {
    if (!Number.isInteger(port) || port < 1024 || port > 65535) {
        throw codedError('invalid_launch', 'port must be an integer between 1024 and 65535', { field: 'port' });
    }
    return port;
}

export function assertApiKey(apiKey) {
    if (typeof apiKey !== 'string' || !API_KEY_PATTERN.test(apiKey)) {
        throw codedError('invalid_launch', 'apiKey must match /^[A-Za-z0-9_-]{32,128}$/', { field: 'apiKey' });
    }
    return apiKey;
}

export function assertAbsolutePath(value, field) {
    if (typeof value !== 'string' || !isAbsolute(value) || value.includes('\0')) {
        throw codedError('invalid_launch', `${field} must be an absolute path`, { field });
    }
    return value;
}

function snippet(text) {
    return text.trim().slice(0, SNIPPET_LENGTH);
}

// Runs `<executable> <args>` without a shell and hands combined stdout+stderr to parse().
// Version probes get only what the executable needs to start, never the
// agent's tokens or secrets: the same rule as the runner environment.
export function probeEnv(extra = {}, source = process.env) {
    return {
        PATH: source.PATH || '/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin',
        HOME: source.HOME || '/tmp',
        LANG: 'C.UTF-8',
        ...extra,
    };
}

export function probeVersion({ spawnSync, executable, args, env, parse, pinnedVersion }) {
    let result;
    try {
        result = spawnSync(executable, args, { encoding: 'utf8', timeout: PROBE_TIMEOUT_MS, env });
    } catch (error) {
        result = { error };
    }
    if (result?.error) {
        const reason = result.error.code === 'ENOENT'
            ? `Executable not found at ${executable}`
            : `Version probe failed: ${result.error.code || result.error.message}`;
        return { installed: false, version: null, reason };
    }
    const output = `${String(result?.stdout ?? '')}\n${String(result?.stderr ?? '')}`;
    const version = parse(output);
    if (!version) {
        return { installed: false, version: null, reason: `Unrecognized version output: ${snippet(output)}` };
    }
    const reason = version === pinnedVersion ? null : `Installed ${version} differs from pinned ${pinnedVersion}`;
    return { installed: true, version, reason };
}
