// The runner lock shipped inside the image (/opt/local-llm/runners.lock.json):
// the only runners that can be installed on demand, and the only files they
// may download (runners plan §5.2). Admins choose a runner id, never a URL.
//
// {
//   "schema": "local-llm.runners-lock/v1",
//   "runners": {
//     "<id>": {
//       "version": "0.30.0",
//       "kind": "python" | "archive",
//       "licence": { "name", "url", "source"?, "notice"?, "requiresAcceptance", "proprietary"? },
//       "files": [{ "name", "url", "size", "sha256", "extract"?, "into"?, "strip"? }],
//       "check": { "distributions": { "<dist>": "<version>" }, "imports": [], "gpuImports": [],
//                  "optionalLibraries": { "<soname glob>": "<why this runner does not need it>" } }
//     }
//   }
// }
//
// A python runner's files are wheels (plus optional source archives with
// `extract`); `uv` installs exactly those wheels, offline, with their hashes.
// An archive runner's files are archives extracted into its runnable copy,
// dropping one leading directory unless the file says `strip: 0` (an archive
// whose entries sit at its root, like LM Studio's llmster tarball).
// A python runner may also pin data files it would otherwise download at run
// time (`into`: the directory of the runnable copy the file is copied to).
//
// A proprietary licence (LM Studio) forbids redistributing the software. Such
// an entry must require acceptance, and the CI install check validates it from
// the lock without downloading it, since a download would accept its terms on
// behalf of whoever runs CI.

import crypto from 'node:crypto';
import fs from 'node:fs';

import { LocalLlmError } from '../errors.mjs';

export const RUNNER_LOCK_SCHEMA = 'local-llm.runners-lock/v1';
export const DEFAULT_RUNNER_LOCK = '/opt/local-llm/runners.lock.json';

// Where pinned runner files may come from. Each file is also pinned by size
// and sha256, so the host list only narrows where a request can go.
export const ALLOWED_HOSTS = Object.freeze([
    'files.pythonhosted.org',
    'github.com',
    'codeload.github.com',
    'download.pytorch.org',
    // OpenAI's tiktoken vocabularies (gpt-oss's o200k_base on vLLM).
    'openaipublic.blob.core.windows.net',
    // LM Studio's headless daemon, llmster (proprietary).
    'llmster.lmstudio.ai',
]);

const ID_RE = /^[a-z0-9][a-z0-9._-]{0,31}$/;
const VERSION_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$/;
const FILE_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]{0,199}$/;
const EXTRACT_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const DIST_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/;
const MODULE_RE = /^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$/;
const SHA256_RE = /^[0-9a-f]{64}$/;
// A shared-library name, with '*' wildcards; it must start with a letter or digit.
const LIBRARY_PATTERN_RE = /^[A-Za-z0-9][A-Za-z0-9._+*-]{0,99}$/;
const TEXT_MAX = 2000;

function invalid(message) {
    return new LocalLlmError('invalid_runner_lock', message);
}

function plainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
        && [Object.prototype, null].includes(Object.getPrototypeOf(value));
}

function onlyKeys(value, allowed, field) {
    for (const key of Object.keys(value)) {
        if (!allowed.includes(key)) throw invalid(`${field} has an unsupported field '${key}'`);
    }
}

function text(value, field, { required = false, max = TEXT_MAX } = {}) {
    if (value === undefined || value === null) {
        if (required) throw invalid(`${field} is required`);
        return null;
    }
    if (typeof value !== 'string' || !value.trim() || value.length > max || /[\u0000-\u0008\u000b-\u001f]/.test(value)) {
        throw invalid(`${field} must be plain text of at most ${max} characters`);
    }
    return value;
}

export function validateLockUrl(value, field) {
    let url;
    try {
        url = new URL(value);
    } catch {
        throw invalid(`${field} is not a URL`);
    }
    if (url.protocol !== 'https:' || url.username || url.password || url.port) {
        throw invalid(`${field} must be a plain https URL`);
    }
    if (!ALLOWED_HOSTS.includes(url.hostname)) {
        throw invalid(`${field} is not on an allowed host (${ALLOWED_HOSTS.join(', ')})`);
    }
    return url.href;
}

function validateFile(value, field, kind) {
    if (!plainObject(value)) throw invalid(`${field} must be an object`);
    onlyKeys(value, ['name', 'url', 'size', 'sha256', 'extract', 'into', 'strip'], field);
    if (typeof value.name !== 'string' || !FILE_NAME_RE.test(value.name) || value.name.includes('..')) {
        throw invalid(`${field}.name must be a plain file name`);
    }
    if (!Number.isSafeInteger(value.size) || value.size <= 0) throw invalid(`${field}.size must be a positive byte count`);
    if (typeof value.sha256 !== 'string' || !SHA256_RE.test(value.sha256)) throw invalid(`${field}.sha256 must be 64 hex characters`);
    const extract = value.extract === undefined ? null : value.extract;
    if (extract !== null && (typeof extract !== 'string' || !EXTRACT_RE.test(extract))) {
        throw invalid(`${field}.extract must be a directory name`);
    }
    const into = value.into === undefined ? null : value.into;
    if (into !== null && (typeof into !== 'string' || !EXTRACT_RE.test(into))) {
        throw invalid(`${field}.into must be a directory name`);
    }
    const archive = /\.(tar\.gz|tgz)$/.test(value.name);
    const wheel = value.name.endsWith('.whl');
    if (value.strip !== undefined) {
        if (value.strip !== 0 && value.strip !== 1) throw invalid(`${field}.strip must be 0 or 1`);
        if (!archive) throw invalid(`${field}.strip is only for archives`);
    }
    if (into !== null && (wheel || archive || extract !== null)) {
        throw invalid(`${field}.into is only for data files, not wheels or archives`);
    }
    if (kind === 'archive' && !archive) throw invalid(`${field} must be a .tar.gz archive`);
    if (kind === 'python' && !wheel && !(archive && extract) && into === null) {
        throw invalid(`${field} must be a wheel, a .tar.gz archive with extract, or a data file with into`);
    }
    return Object.freeze({
        name: value.name,
        url: validateLockUrl(value.url, `${field}.url`),
        size: value.size,
        sha256: value.sha256,
        ...(extract ? { extract } : {}),
        ...(into ? { into } : {}),
        ...(value.strip !== undefined ? { strip: value.strip } : {}),
    });
}

function validateCheck(value, field) {
    if (value === undefined) {
        return Object.freeze({ distributions: Object.freeze({}), imports: Object.freeze([]), gpuImports: Object.freeze([]), optionalLibraries: Object.freeze({}) });
    }
    if (!plainObject(value)) throw invalid(`${field} must be an object`);
    onlyKeys(value, ['distributions', 'imports', 'gpuImports', 'optionalLibraries'], field);
    // Libraries the CI check may find missing because this runner never uses
    // the feature that loads them; each needs its reason.
    const optionalLibraries = {};
    if (value.optionalLibraries !== undefined) {
        if (!plainObject(value.optionalLibraries)) throw invalid(`${field}.optionalLibraries must map library names to reasons`);
        for (const [pattern, reason] of Object.entries(value.optionalLibraries)) {
            if (!LIBRARY_PATTERN_RE.test(pattern)) throw invalid(`${field}.optionalLibraries: '${pattern}' is not a library name pattern`);
            if (typeof reason !== 'string' || !reason.trim() || reason.length > 200) {
                throw invalid(`${field}.optionalLibraries.${pattern} needs a reason of at most 200 characters`);
            }
            optionalLibraries[pattern] = reason;
        }
    }
    const distributions = {};
    if (value.distributions !== undefined) {
        if (!plainObject(value.distributions)) throw invalid(`${field}.distributions must be an object`);
        for (const [name, version] of Object.entries(value.distributions)) {
            if (!DIST_RE.test(name) || typeof version !== 'string' || !VERSION_RE.test(version)) {
                throw invalid(`${field}.distributions.${name} must name a distribution and its version`);
            }
            distributions[name] = version;
        }
    }
    const modules = (list, name) => {
        if (list === undefined) return Object.freeze([]);
        if (!Array.isArray(list) || !list.every((entry) => typeof entry === 'string' && MODULE_RE.test(entry))) {
            throw invalid(`${field}.${name} must list Python module names`);
        }
        return Object.freeze([...list]);
    };
    return Object.freeze({
        distributions: Object.freeze(distributions),
        imports: modules(value.imports, 'imports'),
        gpuImports: modules(value.gpuImports, 'gpuImports'),
        optionalLibraries: Object.freeze(optionalLibraries),
    });
}

function validateRunner(id, value) {
    const field = `runners.${id}`;
    if (!ID_RE.test(id)) throw invalid(`${field}: runner ids are 1-32 lowercase letters, digits, dot, dash or underscore`);
    if (!plainObject(value)) throw invalid(`${field} must be an object`);
    onlyKeys(value, ['version', 'kind', 'licence', 'files', 'check'], field);
    if (typeof value.version !== 'string' || !VERSION_RE.test(value.version)) throw invalid(`${field}.version is invalid`);
    if (!['python', 'archive'].includes(value.kind)) throw invalid(`${field}.kind must be python or archive`);
    if (!plainObject(value.licence)) throw invalid(`${field}.licence must be an object`);
    onlyKeys(value.licence, ['name', 'url', 'source', 'notice', 'requiresAcceptance', 'proprietary'], `${field}.licence`);
    if (![undefined, true, false].includes(value.licence.proprietary)) throw invalid(`${field}.licence.proprietary must be true or false`);
    const licence = Object.freeze({
        name: text(value.licence.name, `${field}.licence.name`, { required: true, max: 80 }),
        url: text(value.licence.url, `${field}.licence.url`, { required: true, max: 400 }),
        source: text(value.licence.source, `${field}.licence.source`, { max: 400 }),
        notice: text(value.licence.notice, `${field}.licence.notice`),
        requiresAcceptance: value.licence.requiresAcceptance === true,
        proprietary: value.licence.proprietary === true,
    });
    if (licence.proprietary && !licence.requiresAcceptance) {
        throw invalid(`${field}: a proprietary licence must require acceptance`);
    }
    if (!Array.isArray(value.files) || value.files.length === 0 || value.files.length > 2000) {
        throw invalid(`${field}.files must list 1-2000 files`);
    }
    const files = value.files.map((file, index) => validateFile(file, `${field}.files[${index}]`, value.kind));
    const names = new Set();
    for (const file of files) {
        if (names.has(file.name)) throw invalid(`${field} lists ${file.name} twice`);
        names.add(file.name);
    }
    const entry = {
        id,
        version: value.version,
        kind: value.kind,
        licence,
        files: Object.freeze(files),
        check: validateCheck(value.check, `${field}.check`),
    };
    entry.totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    // Names this exact entry: a changed lock entry needs a fresh runnable copy.
    entry.digest = crypto.createHash('sha256')
        .update(JSON.stringify({ id, version: entry.version, kind: entry.kind, files }))
        .digest('hex');
    return Object.freeze(entry);
}

export function validateRunnerLock(document) {
    if (!plainObject(document)) throw invalid('The runner lock must be an object');
    onlyKeys(document, ['schema', 'runners'], 'lock');
    if (document.schema !== RUNNER_LOCK_SCHEMA) throw invalid(`Unsupported runner lock schema ${String(document.schema)}`);
    if (!plainObject(document.runners)) throw invalid('runners must be an object');
    const runners = {};
    for (const [id, value] of Object.entries(document.runners)) runners[id] = validateRunner(id, value);
    return Object.freeze({ schema: RUNNER_LOCK_SCHEMA, runners: Object.freeze(runners) });
}

/** The image's lock, or an empty lock when the image has none (an older image). */
export function loadRunnerLock(file = DEFAULT_RUNNER_LOCK, { fsApi = fs } = {}) {
    let text;
    try {
        text = fsApi.readFileSync(file, 'utf8');
    } catch (error) {
        if (error?.code === 'ENOENT') return Object.freeze({ schema: RUNNER_LOCK_SCHEMA, runners: Object.freeze({}) });
        throw error;
    }
    return validateRunnerLock(JSON.parse(text));
}
