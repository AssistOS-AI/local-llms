// Pure helpers for the Local LLMs settings modal. The presenter module
// (local-llm-settings.js) exports only its class, because Explorer's settings
// loader registers the first exported function it finds; everything testable
// without a browser lives here.

export const POLL_INTERVAL_MS = 1500;
export const LOG_LINES_KEPT = 200;
const SETTLED_PHASES = new Set(['idle', 'error', 'paused']);
// Labels until the agent's overview names its runners (rememberRunners).
const runnerLabels = new Map([['llama.cpp', 'llama.cpp'], ['ollama', 'Ollama'], ['vllm', 'vLLM']]);

export function escapeHtml(value = '') {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[character]);
}

/**
 * Text for confirm-action-modal. WebSkel writes modal props into a
 * data-message="..." attribute through innerHTML, so quotes and angle
 * brackets from a model name would break out of it.
 */
export function confirmMessage(text) {
    return String(text ?? '').replace(/"/g, "'").replace(/[<>]/g, '');
}

/** The JSON a local-llm tool wrote, from an MCP result; throws its error. */
export function parseToolResult(value) {
    if (value?.isError) {
        const text = value?.content?.find?.((item) => item.type === 'text')?.text || 'The tool failed.';
        throw new Error(text.replace(/^MCP error -?\d+:\s*/, ''));
    }
    if (value?.json && typeof value.json === 'object') return value.json;
    const text = value?.content?.find?.((item) => item.type === 'text')?.text;
    if (text === undefined) return value;
    const parsed = JSON.parse(text);
    if (parsed && parsed.ok === false) throw new Error(parsed.message || parsed.error || 'The tool failed.');
    return parsed;
}

export function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return '—';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1000 && unit < units.length - 1) {
        value /= 1000;
        unit += 1;
    }
    return `${value >= 100 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

export function formatMiB(bytes) {
    return Number.isFinite(bytes) ? `${Math.round(bytes / (1024 * 1024)).toLocaleString('en-US')} MiB` : '—';
}

export function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '—';
    const total = Math.round(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    if (hours) return `${hours} h ${minutes} min`;
    if (minutes) return `${minutes} min ${rest} s`;
    return `${rest} s`;
}

/** Keep each runner's display name from the overview, so labels follow the agent. */
export function rememberRunners(runners = []) {
    for (const runner of Array.isArray(runners) ? runners : []) {
        if (typeof runner?.id === 'string' && typeof runner.displayName === 'string') runnerLabels.set(runner.id, runner.displayName);
    }
}

export function runnerLabel(runnerId) {
    return runnerLabels.get(runnerId) || String(runnerId || '');
}

/** Delay before the next status poll after `failures` consecutive failures. */
export function pollBackoff(failures) {
    return Math.min(POLL_INTERVAL_MS * 2 ** Math.max(0, failures), 30_000);
}

/** Poll only while something is happening; settled phases wait for a user action. */
export function shouldPoll(phase) {
    return Boolean(phase) && !SETTLED_PHASES.has(phase);
}

export function newRequestId(cryptoApi = globalThis.crypto) {
    const id = typeof cryptoApi?.randomUUID === 'function'
        ? cryptoApi.randomUUID()
        : Array.from(cryptoApi.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, '0')).join('');
    return `ui-${id}`;
}

export function downloadLabel(download, size) {
    if (!download) return size ? `${formatBytes(size)} · not applicable` : 'not applicable';
    if (download.state === 'complete') return `${formatBytes(download.total ?? size)} · downloaded`;
    if (download.state === 'partial') return `${formatBytes(download.bytes)} of ${formatBytes(download.total ?? size)} · paused`;
    return `${formatBytes(download.total ?? size)} · not downloaded`;
}

export function admissionLabel(admission) {
    if (!admission) return '';
    if (admission.status === 'ok') return 'Fits this machine';
    if (admission.status === 'insufficient-now') return `Not now: ${admission.reason}`;
    return admission.reason || 'Not supported';
}

export function estimateLabel(admission) {
    const estimate = admission?.estimate || {};
    const parts = [];
    if (Number.isFinite(estimate.gpuBytes)) parts.push(`GPU about ${formatMiB(estimate.gpuBytes)}`);
    if (Number.isFinite(estimate.ramBytes)) parts.push(`RAM about ${formatMiB(estimate.ramBytes)}`);
    if (!parts.length) return '';
    return `Estimate: ${parts.join(', ')}${estimate.basis ? ` (${estimate.basis})` : ''}`;
}

export function contextLabel(context) {
    if (!context) return '';
    const total = Number(context.totalContext).toLocaleString('en-US');
    const perRequest = Number(context.perRequestContext).toLocaleString('en-US');
    return context.parallel > 1
        ? `${total} tokens total, ${perRequest} per request across ${context.parallel} slots${context.kvUnified ? ' (unified KV cache)' : ''}`
        : `${total} tokens per request`;
}

/** Options for the runner picker: every runner, with its state and reason. */
export function runnerOptions(overview, model) {
    const runners = Array.isArray(overview?.runners) ? overview.runners : [];
    return runners
        .filter((runner) => model?.runners?.[runner.id])
        .map((runner) => {
            const entry = model.runners[runner.id];
            let state;
            if (!runner.supported) state = 'not supported';
            else if (runner.enabled === false) state = 'not enabled on this deployment';
            else if (!runner.installed) state = 'not installed';
            else if (entry.admission?.status === 'ok') state = entry.download?.state === 'complete' ? 'ready to run' : 'will download';
            else state = entry.admission?.status === 'insufficient-now' ? 'not now' : 'does not fit';
            const version = runner.version ? ` ${runner.version}` : '';
            return { value: runner.id, label: `${runnerLabel(runner.id)}${version} · ${state}` };
        });
}

function schemaKind(property) {
    const types = Array.isArray(property.type) ? property.type : [property.type];
    const nullable = types.includes('null');
    const numeric = types.includes('integer') || types.includes('number');
    // A named value or a number (such as "off", "max" or a 0-1 fraction):
    // one text field that accepts either, not a select that hides the numbers.
    if (Array.isArray(property.enum) && numeric) return { kind: 'enumOrNumber', nullable, integer: !types.includes('number') };
    if (Array.isArray(property.enum)) return { kind: 'enum', nullable };
    if (types.includes('boolean')) return { kind: 'boolean', nullable };
    if (types.includes('integer') || types.includes('number')) return { kind: 'number', nullable, integer: types.includes('integer') };
    if (types.includes('string')) return { kind: 'string', nullable };
    return { kind: 'fixed', nullable };
}

/** Form fields generated from a runner's JSON Schema, pre-filled with values. */
export function fieldsFromSchema(schema, values = {}) {
    const properties = schema?.properties || {};
    return Object.entries(properties).map(([name, property]) => {
        const { kind, nullable, integer } = schemaKind(property);
        const value = Object.hasOwn(values, name) ? values[name] : property.default;
        const field = {
            name,
            kind,
            nullable,
            integer: Boolean(integer),
            title: property.title || name,
            description: property.description || '',
            value,
            min: property.minimum,
            max: property.maximum,
        };
        if (kind === 'enumOrNumber') {
            field.options = property.enum.filter((option) => typeof option === 'string').map((option) => ({ value: option, label: option }));
        } else if (kind === 'enum') {
            field.options = [
                ...(nullable ? [{ value: '', label: 'Runner default' }] : []),
                ...property.enum.filter((option) => option !== null).map((option) => ({ value: String(option), label: String(option) })),
            ];
        } else if (kind === 'boolean') {
            field.options = [
                ...(nullable ? [{ value: '', label: 'Runner default' }] : []),
                { value: 'true', label: 'On' },
                { value: 'false', label: 'Off' },
            ];
        }
        return field;
    });
}

/** Parameters from raw form strings; the server validates them again. */
export function paramsFromForm(fields, raw = {}) {
    const params = {};
    const errors = [];
    for (const field of fields) {
        if (field.kind === 'fixed') {
            if (field.value !== undefined) params[field.name] = field.value;
            continue;
        }
        const text = String(raw[field.name] ?? '').trim();
        if (text === '') {
            if (field.nullable) params[field.name] = null;
            else if (field.value !== undefined) params[field.name] = field.value;
            continue;
        }
        if (field.kind === 'enumOrNumber' && field.options.some((option) => option.value === text)) {
            params[field.name] = text;
            continue;
        }
        if (field.kind === 'enumOrNumber' && !Number.isFinite(Number(text))) {
            errors.push(`${field.title} must be ${field.options.map((option) => option.value).join(', ')} or a number.`);
            continue;
        }
        if (field.kind === 'number' || field.kind === 'enumOrNumber') {
            const number = Number(text);
            if (!Number.isFinite(number) || (field.integer && !Number.isInteger(number))) {
                errors.push(`${field.title} must be ${field.integer ? 'a whole number' : 'a number'}.`);
                continue;
            }
            if ((Number.isFinite(field.min) && number < field.min) || (Number.isFinite(field.max) && number > field.max)) {
                errors.push(`${field.title} must be between ${field.min} and ${field.max}.`);
                continue;
            }
            params[field.name] = number;
        } else if (field.kind === 'boolean') {
            params[field.name] = text === 'true';
        } else {
            params[field.name] = text;
        }
    }
    return { params, errors };
}

// An optional number from the form: absent when empty; the integer when it is
// one; otherwise the text as typed, so the controller's validation names the
// field instead of the form dropping it.
function optionalNumber(value) {
    const text = String(value ?? '').trim();
    if (!text) return undefined;
    const number = Number(text);
    return Number.isInteger(number) ? number : text;
}

// Where the form's sizing values came from (DS002); the same three the agent accepts in `sizingSource`.
const SIZING_SOURCES = Object.freeze(['gguf-header', 'config.json', 'manual']);
// The weight format the lookup tool is asked for, by the form's Source kind.
const LOOKUP_FORMATS = Object.freeze({ huggingface: 'gguf', hf: 'hf', exl3: 'exl3' });
const SIZING_FIELDS = Object.freeze(['contextLength', 'layers', 'kvBytesPerToken']);

/** The lookup tool's `format` for a Source kind, or null for a kind with no lookup (an Ollama tag). */
export function lookupFormatFor(kind) {
    return Object.hasOwn(LOOKUP_FORMATS, kind) ? LOOKUP_FORMATS[kind] : null;
}

/**
 * A user registry entry from the Add model form. The Source kind is one of:
 * `huggingface` (one GGUF file, read by llama.cpp and every other GGUF runner),
 * `hf` (a safetensors snapshot, read by vLLM), `exl3` (an EXL3 snapshot, read
 * by TabbyAPI) or `ollama` (one library tag). The optional sizing data is what
 * the memory estimate uses (contextLength, memory.layers,
 * memory.kvBytesPerToken); `sizingSource` says where it came from when a
 * lookup filled it (gguf-header, config.json), or manual when it was changed
 * by hand afterwards. `mtp` says whether the weights have a
 * multi-token-prediction head.
 */
export function modelEntryFromForm(raw = {}) {
    const id = String(raw.id || '').trim().toLowerCase();
    const displayName = String(raw.displayName || '').trim();
    const kind = ['ollama', 'hf', 'exl3'].includes(raw.sourceKind) ? raw.sourceKind : 'huggingface';
    const entry = { id, architecture: raw.architecture === 'moe' ? 'moe' : 'dense', sources: {} };
    if (displayName) entry.displayName = displayName;
    const license = String(raw.license || '').trim();
    if (license) entry.license = license;
    if (kind === 'ollama') {
        entry.sources.ollama = { type: 'ollama', tag: String(raw.tag || '').trim() };
    } else if (kind === 'hf' || kind === 'exl3') {
        // A snapshot is every top-level model file of the repository at one commit; the agent pins them when the entry is added.
        entry.sources[kind] = {
            type: 'hf-snapshot',
            repo: String(raw.repo || '').trim(),
            revision: String(raw.revision || '').trim() || 'main',
        };
    } else {
        const source = {
            type: 'huggingface',
            repo: String(raw.repo || '').trim(),
            file: String(raw.file || '').trim(),
            revision: String(raw.revision || '').trim() || 'main',
        };
        const quantization = String(raw.quantization || '').trim();
        if (quantization) source.quantization = quantization;
        entry.sources.gguf = source;
    }
    const contextLength = optionalNumber(raw.contextLength);
    if (contextLength !== undefined) entry.contextLength = contextLength;
    // The weights' multi-token-prediction head, which MTP needs (a checkbox).
    if (raw.mtp === true || raw.mtp === 'true' || raw.mtp === 'on') entry.mtp = true;
    const memory = {};
    const layers = optionalNumber(raw.layers);
    if (layers !== undefined) memory.layers = layers;
    const kvBytesPerToken = optionalNumber(raw.kvBytesPerToken);
    if (kvBytesPerToken !== undefined) memory.kvBytesPerToken = kvBytesPerToken;
    if (Object.keys(memory).length) entry.memory = memory;
    // Only a source the entry has: a GGUF header sizes a GGUF entry, a config.json a snapshot.
    const fits = { 'gguf-header': kind === 'huggingface', 'config.json': kind === 'hf' || kind === 'exl3', manual: true };
    if (SIZING_SOURCES.includes(raw.sizingSource) && fits[raw.sizingSource]) entry.sizingSource = raw.sizingSource;
    return entry;
}

/**
 * The `sizingSource` to record for the form as it stands. `lookup` is what the
 * last lookup filled in: `{ source, repo, revision, file, architecture,
 * values: { contextLength, layers, kvBytesPerToken } }`, or null. Values that
 * are still the looked-up ones, for the same repository, revision and file,
 * keep the lookup's source. Values changed by hand afterwards, or for another
 * source, are `manual`. With no lookup the entry keeps today's shape (no
 * `sizingSource`), whatever was typed.
 */
export function sizingSourceFor(raw = {}, lookup = null) {
    if (!lookup || !SIZING_SOURCES.includes(lookup.source)) return undefined;
    const text = (value) => String(value ?? '').trim();
    const typed = SIZING_FIELDS.some((name) => text(raw[name]) !== '');
    if (!typed) return undefined;
    const sameSource = text(raw.repo) === lookup.repo && (text(raw.revision) || 'main') === lookup.revision
        && (lookup.file === undefined || text(raw.file) === lookup.file)
        && (raw.architecture === 'moe' ? 'moe' : 'dense') === lookup.architecture;
    const unchanged = SIZING_FIELDS.every((name) => text(raw[name]) === text(lookup.values?.[name]));
    return sameSource && unchanged ? lookup.source : 'manual';
}

/** A model id (2-64 lowercase letters, digits, dot, dash, underscore) suggested from a repository and a quantization. */
export function suggestModelId(repo, quantization = '') {
    const name = String(repo || '').split('/').pop() || 'model';
    const id = `${name}${quantization ? `-${quantization}` : ''}`.toLowerCase()
        .replace(/[^a-z0-9._-]+/g, '-')
        .replace(/-{2,}/g, '-')
        .replace(/^[^a-z0-9]+/, '')
        .slice(0, 64)
        .replace(/[^a-z0-9]+$/, '');
    return id.length >= 2 ? id : 'model';
}

/** Merge newly received log lines, keeping the most recent ones. */
export function mergeLogs(existing = [], incoming = [], keep = LOG_LINES_KEPT) {
    const seen = new Set(existing.map((line) => line.seq));
    const merged = [...existing, ...incoming.filter((line) => !seen.has(line.seq))];
    return merged.slice(-keep);
}

export function progressPercent(download) {
    if (!download?.total) return null;
    return Math.max(0, Math.min(100, Math.floor((download.bytes / download.total) * 100)));
}
