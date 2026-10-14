// Model registries (DS002): where a model source's files come from. Four are
// built in (huggingface, dockerhub, ghcr, modelscope); administrators add more
// and set per-registry credentials through the admin tools, without a restart.
// The list lives in /data/state/registries.json, apart from controller.json,
// whose loader keeps only the keys it knows; credentials live encrypted in
// registry-credentials.json (registryCredentials.mjs).
//
// A lookup, pin or job takes a frozen snapshot of its registry (record and
// decrypted credential) once, at its start; later edits apply to later
// operations. Add and Update pin outside the command queue, so their queue
// block re-checks the snapshot (`recheck`) before anything is saved. Store
// paths, keys and Delete weights never consult this list, so a removed or
// disabled registry never hides a model or its weights.

import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import { createCredentialStore } from './registryCredentials.mjs';
import { writeAtomically } from './stateStore.mjs';

export const REGISTRY_ID_RE = /^[a-z0-9][a-z0-9-]{1,31}$/;
export const REGISTRY_KINDS = Object.freeze(['huggingface', 'oci', 'modelscope']);
export const MAX_ADMIN_REGISTRIES = 32;
export const REGISTRIES_FILE_VERSION = 1;
export const TOKEN_MAX = 4096;
export const TOKEN_USERNAME_MAX = 256;
// Printable characters, no colon: the OCI Basic credential is `tokenUsername:token` (RFC 7617).
export const TOKEN_USERNAME_RE = /^[^\u0000-\u001f\u007f:]{1,256}$/;
const MAX_AUTH_ORIGINS = 4;
const MAX_PREFIX = 256;
const MAX_URL = 512;
const LABEL_MAX = 60;
const SAVE_KEYS = Object.freeze(['id', 'kind', 'url', 'label', 'enabled', 'authOrigins']);
const DNS_LABEL_RE = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/;
const PREFIX_RE = /^(?:\/[A-Za-z0-9._~-]+)+$/;

// The weight formats each kind offers, and the source types each kind can serve.
const FORMATS = Object.freeze({
    huggingface: Object.freeze(['gguf', 'hf', 'exl3']),
    modelscope: Object.freeze(['gguf', 'hf', 'exl3']),
    oci: Object.freeze(['gguf', 'hf']),
});
const SERVES = Object.freeze({
    huggingface: Object.freeze(['huggingface', 'hf-snapshot']),
    modelscope: Object.freeze(['huggingface', 'hf-snapshot']),
    oci: Object.freeze(['oci', 'oci-snapshot']),
});
// Source types that name a registry (absent means huggingface). Ollama tags come from the Ollama runner's own client.
const REGISTRY_SOURCE_TYPES = new Set(['huggingface', 'hf-snapshot', 'oci', 'oci-snapshot']);

// Built-ins that administrators may enable, disable and give a credential, but not re-point or remove.
const SWITCHABLE_BUILT_INS = Object.freeze({
    dockerhub: Object.freeze({ kind: 'oci', url: 'https://registry-1.docker.io', label: 'Docker Hub', authOrigins: Object.freeze(['https://auth.docker.io']) }),
    ghcr: Object.freeze({ kind: 'oci', url: 'https://ghcr.io', label: 'GitHub Container Registry', authOrigins: Object.freeze(['https://ghcr.io']) }),
    modelscope: Object.freeze({ kind: 'modelscope', url: 'https://www.modelscope.cn', label: 'ModelScope', authOrigins: Object.freeze([]) }),
});
const BUILT_IN_IDS = Object.freeze(['huggingface', ...Object.keys(SWITCHABLE_BUILT_INS)]);

function invalid(message, field) {
    return new LocalLlmError('invalid_registry', message, { field });
}

function plainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
        && [Object.prototype, null].includes(Object.getPrototypeOf(value));
}

function dnsHostName(hostname) {
    if (!hostname || hostname.length > 253 || hostname.startsWith('[') || net.isIP(hostname)) return false;
    const labels = hostname.split('.');
    return labels.every((label) => DNS_LABEL_RE.test(label)) && !/^[0-9]+$/.test(labels.at(-1));
}

/**
 * A registry URL in its normal form (`https://host[:port][/prefix]`, no trailing slash), or `invalid_registry`.
 * The host must be a lowercase DNS name: an IP-literal host is refused after WHATWG normalization, so
 * `https://2130706433`, `https://0x7f.1` and `https://127.1` (all 127.0.0.1) are refused like `https://127.0.0.1`.
 * Input must already be in normal form (a trailing slash is allowed), so nothing an administrator typed is rewritten.
 */
export function normalizeRegistryUrl(value, { field = 'url', allowPrefix = false } = {}) {
    if (typeof value !== 'string' || !value || value.length > MAX_URL) throw invalid(`${field} must be an https URL of at most ${MAX_URL} characters`, field);
    let url;
    try {
        url = new URL(value);
    } catch {
        throw invalid(`${field} is not a URL`, field);
    }
    if (url.protocol !== 'https:') throw invalid(`${field} must use https`, field);
    if (url.username || url.password || value.includes('@')) throw invalid(`${field} must not carry a user name or password`, field);
    if (url.search || url.hash || value.includes('?') || value.includes('#')) throw invalid(`${field} must not carry a query or fragment`, field);
    if (url.hostname.startsWith('[') || net.isIP(url.hostname)) {
        throw invalid(`${field} names an IP address; a registry is named by its DNS host name`, field);
    }
    if (!dnsHostName(url.hostname)) throw invalid(`${field} must name a lowercase DNS host name`, field);
    const prefix = url.pathname === '/' ? '' : url.pathname.replace(/\/$/, '');
    if (prefix) {
        if (!allowPrefix) throw invalid(`${field} must be a bare origin (https://host[:port]) for this kind of registry`, field);
        if (prefix.length > MAX_PREFIX || !PREFIX_RE.test(prefix) || prefix.split('/').some((segment) => segment === '.' || segment === '..')) {
            throw invalid(`${field} has a path prefix that is not a plain path of at most ${MAX_PREFIX} characters`, field);
        }
    }
    const normal = `${url.origin}${prefix}`;
    if (value !== normal && value !== `${normal}/`) throw invalid(`${field} must be written in its normal form, ${normal}`, field);
    return normal;
}

function validateLabel(value, fallback) {
    if (value === undefined) return fallback;
    if (typeof value !== 'string' || value.length > LABEL_MAX || /[\u0000-\u001f\u007f]/.test(value)) {
        throw invalid(`label must be plain text of at most ${LABEL_MAX} characters`, 'label');
    }
    return value.trim() || fallback;
}

function validateAuthOrigins(value, kind, url) {
    if (kind !== 'oci') {
        if (value === undefined || (Array.isArray(value) && value.length === 0)) return Object.freeze([]);
        throw invalid('authOrigins applies to oci registries only', 'authOrigins');
    }
    if (value === undefined) return Object.freeze([new URL(url).origin]);
    if (!Array.isArray(value) || value.length === 0 || value.length > MAX_AUTH_ORIGINS) {
        throw invalid(`authOrigins must list 1-${MAX_AUTH_ORIGINS} https origins`, 'authOrigins');
    }
    const origins = value.map((origin, index) => normalizeRegistryUrl(origin, { field: `authOrigins[${index}]` }));
    return Object.freeze([...new Set(origins)]);
}

/** A complete administrator entry, as stored, or `invalid_registry`. Unknown keys are refused. */
export function validateRegistryEntry(value) {
    if (!plainObject(value)) throw invalid('registry must be an object', 'registry');
    for (const key of Object.keys(value)) {
        if (!SAVE_KEYS.includes(key)) throw invalid(`registry has an unsupported field '${key}'`, key);
    }
    if (typeof value.id !== 'string' || !REGISTRY_ID_RE.test(value.id)) {
        throw invalid('id must be 2-32 lowercase letters, digits or dashes, starting with a letter or digit', 'id');
    }
    if (BUILT_IN_IDS.includes(value.id)) throw invalid(`id '${value.id}' is reserved for a built-in registry`, 'id');
    if (!REGISTRY_KINDS.includes(value.kind)) throw invalid(`kind must be one of ${REGISTRY_KINDS.join(', ')}`, 'kind');
    const url = normalizeRegistryUrl(value.url, { allowPrefix: value.kind === 'huggingface' });
    if (value.enabled !== undefined && typeof value.enabled !== 'boolean') throw invalid('enabled must be true or false', 'enabled');
    return Object.freeze({
        id: value.id,
        kind: value.kind,
        url,
        label: validateLabel(value.label, value.id),
        enabled: value.enabled !== false,
        authOrigins: validateAuthOrigins(value.authOrigins, value.kind, url),
    });
}

const sameList = (left, right) => JSON.stringify([...left].sort()) === JSON.stringify([...right].sort());

function unknownRegistry(id) {
    return new LocalLlmError('registry_unknown', `No registry '${id}' is defined; add it under Registries, or change the model's source.`, { registry: id });
}

function registryChanged(id, why) {
    return new LocalLlmError('registry_changed', `Registry '${id}' ${why}; look the model up again and repeat the change.`, { registry: id });
}

export function createRegistryService({
    dataDir,
    // Passed explicitly by agentProcess.mjs, which takes it out of the environment; never read from the environment here.
    credentialsKey = null,
    hfBaseUrl = 'https://huggingface.co',
    fsApi = fs,
    now = () => new Date(),
} = {}) {
    const file = path.join(dataDir, 'state', 'registries.json');
    const credentials = createCredentialStore({ dataDir, credentialsKey, fsApi, now });
    const notes = [...credentials.notes];
    const huggingface = Object.freeze({ id: 'huggingface', kind: 'huggingface', url: hfBaseUrl, label: 'Hugging Face', enabled: true, authOrigins: Object.freeze([]) });
    let current = load();

    function defaults() {
        return { admin: [], builtins: Object.fromEntries(Object.keys(SWITCHABLE_BUILT_INS).map((id) => [id, true])) };
    }

    function load() {
        let value;
        try {
            value = JSON.parse(fsApi.readFileSync(file, 'utf8'));
        } catch (error) {
            if (error?.code === 'ENOENT') return defaults();
            try { fsApi.renameSync(file, `${file}.corrupt-${Date.now()}`); } catch {}
            notes.push('registries.json could not be parsed and was moved aside; the built-in registries are used');
            return defaults();
        }
        if (!plainObject(value) || value.version !== REGISTRIES_FILE_VERSION) {
            try { fsApi.renameSync(file, `${file}.unsupported-${Date.now()}`); } catch {}
            notes.push('registries.json has an unsupported format and was moved aside; the built-in registries are used');
            return defaults();
        }
        const loaded = defaults();
        for (const id of Object.keys(SWITCHABLE_BUILT_INS)) loaded.builtins[id] = value.builtins?.[id]?.enabled !== false;
        for (const entry of Array.isArray(value.registries) ? value.registries : []) {
            try {
                const record = validateRegistryEntry(entry);
                if (loaded.admin.some((other) => other.id === record.id)) throw invalid(`id '${record.id}' is listed twice`, 'id');
                if (loaded.admin.length >= MAX_ADMIN_REGISTRIES) throw invalid(`more than ${MAX_ADMIN_REGISTRIES} registries`, 'registries');
                loaded.admin.push(record);
            } catch (error) {
                notes.push(`a registries.json entry is not used: ${error.message}`);
            }
        }
        return loaded;
    }

    function persist(next) {
        const builtins = Object.fromEntries(Object.entries(next.builtins).map(([id, enabled]) => [id, { enabled }]));
        writeAtomically(fsApi, file, `${JSON.stringify({ version: REGISTRIES_FILE_VERSION, registries: next.admin, builtins }, null, 2)}\n`);
        current = next;
    }

    function records() {
        return [
            { ...huggingface, builtIn: true },
            ...Object.entries(SWITCHABLE_BUILT_INS).map(([id, record]) => ({ id, ...record, enabled: current.builtins[id], builtIn: true })),
            ...current.admin.map((record) => ({ ...record, builtIn: false })),
        ];
    }

    const find = (id) => records().find((record) => record.id === id) ?? null;

    /**
     * The frozen snapshot of registry `id` for one lookup, pin or job: its record, and its credential decrypted now
     * (null when none or unreadable), held as a non-enumerable property so it is never serialized with the record.
     */
    function snapshot(id) {
        const record = find(id);
        if (!record) throw unknownRegistry(id);
        if (!record.enabled) {
            throw new LocalLlmError('registry_disabled', `Registry '${id}' is disabled; enable it under Registries first.`, { registry: id });
        }
        const taken = { id: record.id, kind: record.kind, url: record.url, label: record.label, authOrigins: record.authOrigins };
        Object.defineProperty(taken, 'credential', { value: record.id === 'huggingface' ? null : credentials.open(record), enumerable: false });
        return Object.freeze(taken);
    }

    function registryOf(source, purpose) {
        const taken = snapshot(source.registry ?? 'huggingface');
        if (!SERVES[taken.kind].includes(source.type)) {
            const message = `Registry '${taken.id}' is a ${taken.kind} registry and cannot serve a ${source.type} source`;
            throw purpose === 'run' ? registryChanged(taken.id, `is now a ${taken.kind} registry and cannot serve this ${source.type} source`)
                : new LocalLlmError('invalid_model', `${message}.`, { registry: taken.id });
        }
        return taken;
    }

    /**
     * Snapshots of the registries `sources` name (by format), taken before a pin. Gone gives `registry_unknown`,
     * disabled `registry_disabled`; a source type the registry's kind cannot serve gives `invalid_model`.
     */
    function snapshotSources(sources) {
        const taken = [];
        for (const [format, source] of Object.entries(sources ?? {})) {
            if (REGISTRY_SOURCE_TYPES.has(source?.type)) taken.push(Object.freeze({ format, registry: registryOf(source, 'pin') }));
        }
        return Object.freeze(taken);
    }

    /**
     * N5: inside the Add or Update queue block, after the pin, every registry the pin used must still exist, be
     * enabled, and have the kind and URL of its snapshot. Nothing is decrypted.
     */
    function recheck(taken) {
        for (const { registry } of taken) {
            const record = find(registry.id);
            if (!record) throw unknownRegistry(registry.id);
            if (!record.enabled) throw new LocalLlmError('registry_disabled', `Registry '${registry.id}' was disabled while the model was being pinned.`, { registry: registry.id });
            if (record.kind !== registry.kind || record.url !== registry.url) throw registryChanged(registry.id, 'was changed while the model was being pinned');
        }
    }

    /** Before a Run is admitted: the source's registry exists, is enabled and still serves its type. Nothing is decrypted. */
    function assertRunnable(source) {
        if (!REGISTRY_SOURCE_TYPES.has(source?.type)) return;
        const id = source.registry ?? 'huggingface';
        const record = find(id);
        if (!record) throw unknownRegistry(id);
        if (!record.enabled) throw new LocalLlmError('registry_disabled', `Registry '${id}' is disabled; enable it under Registries to run this model.`, { registry: id });
        if (!SERVES[record.kind].includes(source.type)) throw registryChanged(id, `is now a ${record.kind} registry and cannot serve this ${source.type} source`);
    }

    /**
     * The five admin operations, bound to one controller: `queue` serializes every edit with the controller's other
     * commands, `references()` gives the stored entries and the deployment of the running job (raw, so entries this
     * catalog cannot read still hold their registry), `log(message)` writes the controller log.
     */
    function operations({ queue, log = () => {}, references = () => ({ entries: [], active: null }), hfToken = false }) {
        for (const note of notes) log(`registries: ${note}`);
        // One line per registry per start for a credential this key cannot read.
        for (const record of records()) {
            if (record.id !== 'huggingface' && credentials.state(record) === 'unreadable') {
                log(`registry ${record.id}: its stored credential cannot be read with this agent's key; re-enter it`);
            }
        }

        function usage() {
            const used = new Map();
            const add = (registry, modelId) => {
                if (typeof registry !== 'string') return;
                if (!used.has(registry)) used.set(registry, new Set());
                used.get(registry).add(String(modelId ?? '(no id)'));
            };
            const { entries = [], active = null } = references() ?? {};
            for (const entry of Array.isArray(entries) ? entries : []) {
                for (const source of Object.values(plainObject(entry?.sources) ? entry.sources : {})) add(source?.registry, entry?.id);
            }
            if (active) add(active.artifact?.registry, active.modelId);
            return used;
        }
        const usedBy = (id, used = usage()) => [...(used.get(id) ?? [])].sort();

        function publicView(record, used = usage()) {
            const credential = record.id === 'huggingface' ? (hfToken ? 'set' : 'none') : credentials.state(record);
            return {
                id: record.id,
                kind: record.kind,
                label: record.label,
                url: record.url,
                formats: [...FORMATS[record.kind]],
                enabled: record.enabled,
                builtIn: record.builtIn,
                editable: !record.builtIn,
                authOrigins: [...record.authOrigins],
                credential,
                usedBy: usedBy(record.id, used),
            };
        }

        // An existing registry an operation may change, or the error that says why not.
        function editable(id, { field = 'id' } = {}) {
            if (id === 'huggingface') throw new LocalLlmError('read_only', 'The huggingface registry is built in and uses the operator\'s HF_TOKEN; it cannot be changed.', { registry: id });
            if (typeof id !== 'string' || !REGISTRY_ID_RE.test(id)) throw invalid('id must name a registry', field);
            const record = find(id);
            if (!record) throw invalid(`No registry '${id}' is defined`, field);
            return record;
        }

        function inUse(id, what) {
            const models = usedBy(id);
            if (models.length) {
                throw new LocalLlmError('registry_in_use', `Registry '${id}' is used by ${models.join(', ')}; ${what}`, { registry: id, models });
            }
        }

        function save(input) {
            if (!plainObject(input)) throw invalid('registry must be an object', 'registry');
            for (const key of Object.keys(input)) {
                if (!SAVE_KEYS.includes(key)) throw invalid(`registry has an unsupported field '${key}'`, key);
            }
            const { id } = input;
            if (id === 'huggingface') editable(id);
            if (Object.hasOwn(SWITCHABLE_BUILT_INS, id)) {
                const extra = Object.keys(input).filter((key) => !['id', 'enabled'].includes(key));
                if (extra.length) throw invalid(`id '${id}' is reserved for a built-in registry, which can only be enabled or disabled and given a credential`, extra[0]);
                if (input.enabled !== undefined && typeof input.enabled !== 'boolean') throw invalid('enabled must be true or false', 'enabled');
                if (input.enabled !== undefined && input.enabled !== current.builtins[id]) {
                    persist({ ...current, builtins: { ...current.builtins, [id]: input.enabled } });
                }
                return { registry: publicView(find(id)) };
            }
            const existing = current.admin.find((record) => record.id === id) ?? null;
            if (!existing) {
                const record = validateRegistryEntry(input);
                if (current.admin.length >= MAX_ADMIN_REGISTRIES) throw invalid(`At most ${MAX_ADMIN_REGISTRIES} registries can be added`, 'registry');
                // A credential left under this id (a manual edit, a downgrade) never carries over to a new registry.
                credentials.remove(id);
                persist({ ...current, admin: [...current.admin, record] });
                return { registry: publicView(find(id)) };
            }
            const kind = input.kind ?? existing.kind;
            const url = input.url === undefined ? existing.url : normalizeRegistryUrl(input.url, { allowPrefix: kind === 'huggingface' });
            if (kind !== existing.kind) {
                inUse(id, 'its kind cannot change while a model uses it.');
                throw invalid('kind cannot change; remove the registry and add it again', 'kind');
            }
            if (url !== existing.url) inUse(id, 'its URL cannot change while a model uses it.');
            // Origins that only followed the old URL follow the new one.
            const followed = existing.kind === 'oci' && sameList(existing.authOrigins, [new URL(existing.url).origin]);
            const authOrigins = input.authOrigins ?? (url !== existing.url && followed ? undefined : existing.authOrigins);
            const record = validateRegistryEntry({ ...existing, ...input, url, authOrigins });
            // A credential is bound to where it may be sent: a new URL or new auth origins delete it first.
            if (record.url !== existing.url || !sameList(record.authOrigins, existing.authOrigins)) credentials.remove(id);
            persist({ ...current, admin: current.admin.map((entry) => (entry.id === id ? record : entry)) });
            return { registry: publicView(find(id)) };
        }

        function remove({ id } = {}) {
            const record = editable(id);
            if (record.builtIn) throw new LocalLlmError('read_only', `The built-in registry ${id} cannot be removed; disable it instead.`, { registry: id });
            inUse(id, 'remove or change those models first.');
            // The credential goes first: a crash between the two writes never leaves it for a later registry of this id.
            credentials.remove(id);
            persist({ ...current, admin: current.admin.filter((entry) => entry.id !== id) });
            return { removed: id };
        }

        // Token values are checked by length and character class only; no message or detail repeats them.
        function setCredential({ id, token, tokenUsername, ...rest } = {}) {
            const unexpected = Object.keys(rest);
            if (unexpected.length) throw invalid(`The credential has an unsupported field '${unexpected[0]}'`, unexpected[0]);
            const record = editable(id);
            if (!credentials.available) throw new LocalLlmError('credentials_unavailable', credentials.reason);
            if (typeof token !== 'string' || token.length < 1 || token.length > TOKEN_MAX || /[\u0000-\u001f\u007f]/.test(token)) {
                throw invalid(`token must be 1-${TOKEN_MAX} characters with no control characters`, 'token');
            }
            if (tokenUsername !== undefined && (typeof tokenUsername !== 'string' || !TOKEN_USERNAME_RE.test(tokenUsername))) {
                throw invalid(`tokenUsername must be 1-${TOKEN_USERNAME_MAX} printable characters with no colon`, 'tokenUsername');
            }
            credentials.seal(record, { token, username: tokenUsername });
            return { id, credential: 'set' };
        }

        function clearCredential({ id } = {}) {
            editable(id);
            credentials.remove(id);
            return { id, credential: 'none' };
        }

        return Object.freeze({
            listRegistries: async () => {
                const used = usage();
                return {
                    registries: records().map((record) => publicView(record, used)),
                    credentialStorage: { available: credentials.available, reason: credentials.reason },
                };
            },
            saveRegistry: ({ registry } = {}) => queue.run(async () => save(registry)),
            removeRegistry: (args) => queue.run(async () => remove(args)),
            setRegistryCredential: (args) => queue.run(async () => setCredential(args)),
            clearRegistryCredential: (args) => queue.run(async () => clearCredential(args)),
        });
    }

    return Object.freeze({ file, credentialsFile: credentials.file, snapshot, snapshotSources, recheck, assertRunnable, operations });
}
