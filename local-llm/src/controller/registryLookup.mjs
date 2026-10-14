// The model lookup across registries (DS002): the input's one normalizer, and the dispatch that sends a lookup to the
// code for its registry's kind. `registry` absent means huggingface.co. A Hugging Face-compatible registry (a mirror,
// with an optional path prefix) is looked up by the Hugging Face flow at its own base URL, with its own credential
// and a canary request first; a ModelScope registry by modelScope.mjs. The credential of the huggingface.co lookup is
// the operator's `HF_TOKEN` and no other registry ever receives it. An OCI registry is refused clearly until its
// lookup exists.

import { LocalLlmError } from '../errors.mjs';
import { splitGgufName } from './catalog.mjs';
import { DownloadError, assertFile, assertRepo, assertRevision } from './downloader.mjs';
import { lookupHuggingFaceModel } from './modelLookup.mjs';
import { lookupModelScope } from './modelScope.mjs';
import { REGISTRY_ID_RE } from './registries.mjs';

export { REGISTRY_ID_RE };

/**
 * The lookup tool's `repo` pattern: the widest form any kind of registry accepts (up to four segments, for an OCI
 * repository). The code enforces the narrower pattern of the registry's kind (`owner/name` for Hugging Face).
 */
export const LOOKUP_REPO_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}){1,3}$/;

const FORMATS = Object.freeze(['gguf', 'hf', 'exl3']);
const INPUT_KEYS = Object.freeze(['registry', 'provider', 'repo', 'revision', 'format', 'file']);
const HUGGINGFACE = Object.freeze({ id: 'huggingface', kind: 'huggingface', label: 'Hugging Face' });

function failure(code, message, details = undefined) {
    return new LocalLlmError(code, message, details);
}

/** The lookup's input, checked before any request; a bad value is `invalid_request`. */
export function normalizeLookupInput(args) {
    if (args === null || typeof args !== 'object' || Array.isArray(args)) throw failure('invalid_request', 'The lookup needs a repo.');
    for (const key of Object.keys(args)) {
        if (!INPUT_KEYS.includes(key)) throw failure('invalid_request', `The lookup has no '${key}' field.`);
    }
    if (args.registry !== undefined && (typeof args.registry !== 'string' || !REGISTRY_ID_RE.test(args.registry))) {
        throw failure('invalid_request', 'registry must be the id of a registry.');
    }
    const format = args.format ?? 'gguf';
    if (!FORMATS.includes(format)) throw failure('invalid_request', `format must be one of ${FORMATS.join(', ')}.`);
    if (args.provider !== undefined && (args.provider !== 'huggingface' || (args.registry ?? 'huggingface') !== 'huggingface')) {
        throw failure('invalid_request', 'provider must be huggingface, and applies to the huggingface registry only.');
    }
    const revision = args.revision ?? 'main';
    const file = args.file;
    try {
        assertRepo(args.repo);
        assertRevision(revision);
        if (file !== undefined) {
            if (format !== 'gguf') throw failure('invalid_request', 'file applies to a GGUF lookup only.');
            if (typeof file !== 'string' || file.length > 512) throw failure('invalid_request', 'file must be a path of at most 512 characters.');
            assertFile(file);
            const split = splitGgufName(file);
            if (split && split.index !== 1) {
                throw failure('invalid_request', `file names shard ${split.index} of ${split.count}; name the first shard of a split GGUF.`);
            }
        }
    } catch (error) {
        throw error instanceof DownloadError ? failure('invalid_request', error.message) : error;
    }
    return { repo: args.repo, revision, format, ...(file === undefined ? {} : { file }) };
}

/**
 * Look a model up in the registry `args.registry` names (huggingface.co when absent). `options` are the seams of
 * `lookupHuggingFaceModel` (`token` and `baseUrl` are huggingface.co's own), plus `registries`, the registry service
 * the snapshot of a named registry comes from. The answer carries `registry: { id, kind, label }`.
 */
export async function lookupInRegistry(args, { registries = null, ...options } = {}) {
    normalizeLookupInput(args);
    const { registry: id = 'huggingface', ...rest } = args;
    if (id === 'huggingface') {
        return { ...await lookupHuggingFaceModel(rest, options), registry: HUGGINGFACE };
    }
    if (!registries) throw failure('registry_unknown', `No registry '${id}' is defined.`, { registry: id });
    // Resolved once, here: an edit made while this lookup runs applies to the next one.
    const taken = registries.snapshot(id);
    if (taken.kind !== 'huggingface' && taken.kind !== 'modelscope') {
        throw failure('unsupported_artifact', `Registry '${id}' is a ${taken.kind} registry, which this version cannot look up yet.`, { registry: id });
    }
    const named = { ...options, token: taken.credential?.token ?? '', baseUrl: taken.url, subject: taken.label };
    const result = taken.kind === 'modelscope'
        ? await lookupModelScope(rest, named)
        : await lookupHuggingFaceModel(rest, { ...named, canary: true });
    return { ...result, registry: { id: taken.id, kind: taken.kind, label: taken.label } };
}
