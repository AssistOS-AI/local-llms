// Where each kind of weights lives and how it is pinned, inspected, fetched
// and deleted. A source's `type` selects its store: `huggingface` is one GGUF
// file that the controller downloads before the runner starts; `ollama` is a
// library tag that the Ollama runner pulls itself once it is running.
//
// Every store has: `type`; `fetchedBy` ('controller' or 'runner');
// `key(source)`, the artifact identity used for "in use" checks, so runners
// that read the same file share it; `isPinned(source)`; `state(source)`;
// `remove(source)`; `pin(source)` and `carryPin(next, previous)` for Add and
// Update model; and `fetch(...)` when the controller fetches.

import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import { artifactPaths } from './downloader.mjs';
import { deleteOllamaModel, deleteOllamaPartials, partialPullBytes, readOllamaManifest } from './ollamaStore.mjs';

export function createWeightStores({
    dataDir,
    env,
    hfBaseUrl,
    inspect,
    download,
    remove,
    resolveHf,
    state,
    save,
    // The artifact of the job that is running now, or null.
    activeArtifact,
}) {
    const ggufRoot = path.join(dataDir, 'models', 'gguf');
    const ollamaModels = path.join(dataDir, 'models', 'ollama');

    const huggingface = Object.freeze({
        type: 'huggingface',
        fetchedBy: 'controller',
        key: (source) => `gguf:${source.repo}@${source.commit}/${source.file}`,
        isPinned: (source) => Boolean(source.commit),
        async state(source) {
            if (!source.commit) return { state: 'unpinned', bytes: 0, total: null };
            const inspected = await inspect({ root: ggufRoot, artifact: source });
            return { ...inspected, total: source.size };
        },
        async fetch({ artifact, signal, onProgress }) {
            const result = await download({
                artifact,
                root: ggufRoot,
                token: env.HF_TOKEN || '',
                baseUrl: hfBaseUrl,
                onProgress,
                signal,
            });
            return { path: result.path, bytes: artifact.size, bytesTransferred: result.bytesTransferred };
        },
        async remove(source) {
            return source.commit ? remove({ root: ggufRoot, artifact: source }) : 0;
        },
        // Pinning reads Hugging Face metadata only; no weights are downloaded.
        async pin(source) {
            if (source.commit) return source;
            const resolved = await resolveHf({
                repo: source.repo,
                file: source.file,
                revision: source.revision || 'main',
                token: env.HF_TOKEN || '',
                baseUrl: hfBaseUrl,
            });
            return { ...source, commit: resolved.commit, size: resolved.size, sha256: resolved.sha256 };
        },
        // An update keeps the pinned commit while repository, file and
        // revision are unchanged; it never silently re-resolves a branch.
        carryPin(next, previous) {
            if (next.commit || previous?.type !== 'huggingface' || !previous.commit) return next;
            if (next.repo !== previous.repo || next.file !== previous.file || next.revision !== previous.revision) return next;
            return { ...next, commit: previous.commit, size: previous.size, sha256: previous.sha256 };
        },
        paths: (source) => artifactPaths({ root: ggufRoot, artifact: source }),
    });

    function manifestOrNull(tag) {
        try {
            return readOllamaManifest(ollamaModels, tag);
        } catch (error) {
            if (error?.code === 'invalid_manifest') return null;
            throw error;
        }
    }

    function pulls() {
        const current = state();
        current.ollamaPulls ||= {};
        return current.ollamaPulls;
    }

    const ollama = Object.freeze({
        type: 'ollama',
        fetchedBy: 'runner',
        key: (source) => `ollama:${source.tag}`,
        isPinned: () => true,
        async state(source) {
            const manifest = manifestOrNull(source.tag);
            if (manifest?.complete) return { state: 'complete', bytes: manifest.size, total: manifest.size };
            const partial = partialPullBytes(ollamaModels, state().ollamaPulls?.[source.tag] || []);
            return { state: partial ? 'partial' : 'absent', bytes: partial, total: source.size ?? null };
        },
        // Ollama tags can share blobs: never remove a partial another tag's
        // pull claims, and refuse while another tag's running pull has
        // reported a blob this deletion would touch.
        async remove(source) {
            const recorded = pulls();
            const claimedByOthers = Object.entries(recorded)
                .filter(([tag]) => tag !== source.tag)
                .flatMap(([, digests]) => digests);
            const running = activeArtifact();
            const activeTag = running?.type === 'ollama' ? running.tag : null;
            if (activeTag && activeTag !== source.tag) {
                const touched = new Set([
                    ...(recorded[source.tag] || []),
                    ...(manifestOrNull(source.tag)?.blobs.map((blob) => blob.digest) || []),
                ]);
                if ((recorded[activeTag] || []).some((digest) => touched.has(digest))) {
                    throw new LocalLlmError('in_use', `The running ${activeTag} download uses some of these files; `
                        + 'wait for it to finish or cancel it first.');
                }
            }
            const freed = deleteOllamaModel(ollamaModels, source.tag)
                + deleteOllamaPartials(ollamaModels, {
                    digests: recorded[source.tag] || [],
                    claimedByOthers,
                    keepOrphans: activeTag !== null,
                });
            if (recorded[source.tag]) {
                delete recorded[source.tag];
                save();
            }
            return freed;
        },
        pin: async (source) => source,
        carryPin: (next) => next,
        // Used by the Ollama runner while it pulls.
        readManifest: (tag) => readOllamaManifest(ollamaModels, tag),
        recordPull(tag, digest) {
            const digests = pulls()[tag] ||= [];
            if (digests.includes(digest)) return;
            digests.push(digest);
            save();
        },
        clearPulls(tag) {
            const recorded = state().ollamaPulls;
            if (!recorded?.[tag]) return;
            delete recorded[tag];
            save();
        },
    });

    return Object.freeze({ huggingface, ollama });
}
