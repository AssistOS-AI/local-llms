// The model lookup across registries (DS002): the input's one normalizer, and the dispatch that sends a lookup to the
// code for its registry's kind. `registry` absent means huggingface.co. A Hugging Face-compatible registry (a mirror,
// with an optional path prefix) is looked up by the Hugging Face flow at its own base URL, with its own credential
// and a canary request first; a ModelScope registry by modelScope.mjs. The credential of the huggingface.co lookup is
// the operator's `HF_TOKEN` and no other registry ever receives it. An OCI registry is looked up by ociRegistry.mjs.
// A named registry holds a credential, so its lookup runs only through the guarded transport.

import { LocalLlmError } from '../errors.mjs';
import { splitGgufName } from './catalog.mjs';
import { DownloadError, assertFile, assertRepo, assertRevision } from './downloader.mjs';
import { lookupHuggingFaceModel } from './modelLookup.mjs';
import { lookupModelScope } from './modelScope.mjs';
import { lookupOci } from './ociRegistry.mjs';
import { OCI_TAG_RE, isOciGgufFile, isOciRepository } from './ociSource.mjs';
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
// The built-in registries' kinds are fixed, so their input is checked before anything is read; an administrator's
// registry is checked by its kind once its snapshot is taken.
const BUILT_IN_KINDS = Object.freeze({ huggingface: 'huggingface', modelscope: 'modelscope', dockerhub: 'oci', ghcr: 'oci' });

function failure(code, message, details = undefined) {
    return new LocalLlmError(code, message, details);
}

/**
 * The lookup's input, checked before any request; a bad value is `invalid_request`. `kind` is the registry's kind:
 * `huggingface` and `modelscope` take `owner/name` and a revision (`main` when absent), `oci` a repository of 2-4
 * lowercase segments and a tag (`latest` when absent) and no exl3; null (a kind not known yet) checks only the
 * fields every kind shares and returns null.
 */
export function normalizeLookupInput(args, { kind = 'huggingface' } = {}) {
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
    if (kind === null) {
        if (typeof args.repo !== 'string' || args.repo.length > 255 || !LOOKUP_REPO_RE.test(args.repo)) throw failure('invalid_request', 'repo is not a repository name.');
        return null;
    }
    if (kind === 'oci') return normalizeOciInput(args, format);
    // The default branch: ModelScope's is master, Hugging Face's main.
    const revision = args.revision ?? (kind === 'modelscope' ? 'master' : 'main');
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

function normalizeOciInput(args, format) {
    if (format === 'exl3') throw failure('invalid_request', 'An OCI registry offers the gguf and hf formats.');
    if (!isOciRepository(args.repo)) throw failure('invalid_request', 'repo must be an OCI repository of 2-4 lowercase segments, such as ai/smollm2.');
    const revision = args.revision ?? 'latest';
    if (typeof revision !== 'string' || !OCI_TAG_RE.test(revision)) throw failure('invalid_request', 'revision must be an OCI tag.');
    const { file } = args;
    if (file !== undefined) {
        if (format !== 'gguf') throw failure('invalid_request', 'file applies to a GGUF lookup only.');
        if (!isOciGgufFile(file)) throw failure('invalid_request', 'file must be a .gguf file name.');
        const split = splitGgufName(file);
        if (split && split.index !== 1) throw failure('invalid_request', `file names shard ${split.index} of ${split.count}; name the first shard of a split GGUF.`);
    }
    return { repo: args.repo, revision, format, ...(file === undefined ? {} : { file }) };
}

/**
 * Look a model up in the registry `args.registry` names (huggingface.co when absent). `options` are the seams of
 * `lookupHuggingFaceModel` (`token` and `baseUrl` are huggingface.co's own), plus `registries`, the registry service
 * the snapshot of a named registry comes from. A named registry needs `fetchImpl` to be the guarded transport (its
 * `allowHttp` flag marks it): anything else is `internal_error`, before any request. The answer carries
 * `registry: { id, kind, label }`.
 */
export async function lookupInRegistry(args, { registries = null, ...options } = {}) {
    const asked = args !== null && typeof args === 'object' && typeof args.registry === 'string' ? args.registry : 'huggingface';
    normalizeLookupInput(args, { kind: Object.hasOwn(BUILT_IN_KINDS, asked) ? BUILT_IN_KINDS[asked] : null });
    const { registry: id = 'huggingface', ...rest } = args;
    if (id === 'huggingface') {
        return { ...await lookupHuggingFaceModel(rest, options), registry: HUGGINGFACE };
    }
    if (typeof options.fetchImpl !== 'function' || typeof options.fetchImpl.allowHttp !== 'boolean') {
        throw failure('internal_error', `Registry '${id}' cannot be looked up without the network guard.`, { registry: id });
    }
    if (!registries) throw failure('registry_unknown', `No registry '${id}' is defined.`, { registry: id });
    // Resolved once, here: an edit made while this lookup runs applies to the next one.
    const taken = registries.snapshot(id);
    normalizeLookupInput(args, { kind: taken.kind });
    const named = { ...options, token: taken.credential?.token ?? '', baseUrl: taken.url, subject: taken.label };
    let result;
    if (taken.kind === 'oci') result = await lookupOci(rest, { fetchImpl: options.fetchImpl, signal: options.signal, registry: taken });
    else if (taken.kind === 'modelscope') result = await lookupModelScope(rest, named);
    else result = await lookupHuggingFaceModel(rest, { ...named, canary: true });
    return { ...result, registry: { id: taken.id, kind: taken.kind, label: taken.label } };
}
