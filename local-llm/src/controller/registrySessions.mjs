// The session a pin or a download uses for one registry (DS002): the base URL (with a mirror's path prefix), the
// credential, and the fetch that carries it. A session is resolved once, when a pin or a job starts, from the
// registry's frozen snapshot, so a later edit never changes a request already under way. The credential goes to
// the registry's own origin only: the fetch follows redirects by hand and drops `Authorization` from any hop that
// leaves it (a file host's CDN). huggingface.co keeps the operator's `HF_TOKEN`; no other registry ever receives
// it, and a mirror receives only its own stored credential.

import { LocalLlmError } from '../errors.mjs';
import { followRedirects } from './netGuard.mjs';

/**
 * `forSource(source, { registry })` is the session for a Hugging Face-compatible source: the built-in huggingface.co
 * when it names no registry, else the registry's snapshot (`registry`, when the caller took one before its pin;
 * otherwise taken now from `registries`). Another kind of registry gives `unsupported_artifact`.
 */
export function createSessions({ env, hfBaseUrl, registryTransport = null, registries = null }) {
    const session = ({ baseUrl, token, registry = undefined, subject = undefined }) => {
        const via = (sameOriginOnly) => (registryTransport
            ? { fetchImpl: followRedirects(registryTransport, { homeOrigin: baseUrl, sameOriginOnly, ...(subject ? { subject } : {}) }) }
            : {});
        return Object.freeze({
            baseUrl,
            token,
            registry,
            // Metadata is the registry's own answer: it is read from its origin only. Files may be redirected to a CDN.
            metadataFetch: () => via(true),
            fileFetch: () => via(false),
        });
    };

    function forSource(source, { registry = null } = {}) {
        if (!source.registry) return session({ baseUrl: hfBaseUrl, token: env.HF_TOKEN || '' });
        // A named registry holds a credential, which is only ever sent through the guarded transport.
        if (!registryTransport) throw new LocalLlmError('internal_error', `Registry '${source.registry}' cannot be read without the network guard.`);
        let taken = registry?.id === source.registry ? registry : null;
        if (!taken) {
            if (!registries) {
                throw new LocalLlmError('registry_unknown', `No registry '${source.registry}' is defined.`, { registry: source.registry });
            }
            taken = registries.snapshotSources({ source })[0].registry;
        }
        if (taken.kind !== 'huggingface') {
            throw new LocalLlmError('unsupported_artifact', `Registry '${taken.id}' is a ${taken.kind} registry, which this version cannot read yet.`, { registry: taken.id });
        }
        return session({ baseUrl: taken.url, token: taken.credential?.token ?? '', registry: taken.id, subject: taken.label });
    }

    return Object.freeze({ forSource });
}
