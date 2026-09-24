// Where each kind of weights lives and how it is pinned, inspected, fetched
// and deleted. A source's `type` selects its store: `huggingface` is one GGUF
// file that the controller downloads before the runner starts; `ollama` is a
// library tag that the Ollama runner pulls itself once it is running;
// `hf-snapshot` is a directory of files (safetensors for vLLM, EXL3 for
// TabbyAPI) that the controller downloads file by file.
//
// Every store has: `type`; `fetchedBy` ('controller' or 'runner');
// `key(source)`, the artifact identity used for "in use" checks, so runners
// that read the same file share it; `isPinned(source)`; `state(source)`;
// `remove(source)`; `pin(source)` and `carryPin(next, previous)` for Add and
// Update model; and `fetch(...)` when the controller fetches.

import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import {
    artifactPaths,
    downloadSnapshotFile,
    inspectSnapshotFile,
    resolveHuggingFaceSnapshot,
    snapshotPaths,
} from './downloader.mjs';
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
    downloadSnapshot = downloadSnapshotFile,
    inspectSnapshot = inspectSnapshotFile,
    resolveSnapshot = resolveHuggingFaceSnapshot,
}) {
    const ggufRoot = path.join(dataDir, 'models', 'gguf');
    const ollamaModels = path.join(dataDir, 'models', 'ollama');
    const hfRoot = path.join(dataDir, 'models', 'hf');

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

    const snapshotFile = (source, file) => ({
        repo: source.repo, commit: source.commit, file: file.path, size: file.size,
        ...(file.sha256 !== undefined ? { sha256: file.sha256 } : { gitOid: file.gitOid }),
    });

    async function treeBytes(dir) {
        let total = 0;
        let entries = [];
        try { entries = await fs.promises.readdir(dir, { withFileTypes: true }); } catch { return 0; }
        for (const entry of entries) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) total += await treeBytes(full);
            else if (entry.isFile()) total += (await fs.promises.lstat(full)).size;
        }
        return total;
    }

    const hfSnapshot = Object.freeze({
        type: 'hf-snapshot',
        fetchedBy: 'controller',
        key: (source) => `hf:${source.repo}@${source.commit}`,
        isPinned: (source) => Boolean(source.commit && source.files?.length),
        // Complete only when every file has verified; a partial snapshot is never ready.
        async state(source) {
            if (!source.commit || !source.files) return { state: 'unpinned', bytes: 0, total: null };
            let bytes = 0;
            let complete = true;
            let partial = false;
            for (const file of source.files) {
                const inspected = await inspectSnapshot({ root: hfRoot, artifact: snapshotFile(source, file) });
                bytes += inspected.bytes;
                if (inspected.state !== 'complete') complete = false;
                if (inspected.state !== 'absent') partial = true;
            }
            return { state: complete ? 'complete' : (partial ? 'partial' : 'absent'), bytes, total: source.size };
        },
        async fetch({ artifact: source, signal, onProgress = () => {} }) {
            let done = 0;
            let transferred = 0;
            for (const file of source.files) {
                const result = await downloadSnapshot({
                    root: hfRoot,
                    artifact: snapshotFile(source, file),
                    token: env.HF_TOKEN || '',
                    baseUrl: hfBaseUrl,
                    signal,
                    onProgress: (progress) => onProgress({
                        ...progress,
                        bytes: done + progress.bytes,
                        total: source.size,
                        transferred: transferred + (progress.transferred ?? 0),
                    }),
                });
                done += file.size;
                transferred += result.bytesTransferred;
                onProgress({ bytes: done, total: source.size, transferred, rate: 0, etaSeconds: null });
            }
            const { dir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
            return { path: dir, bytes: source.size, bytesTransferred: transferred };
        },
        // Delete weights removes the whole snapshot and its bookkeeping.
        async remove(source) {
            if (!source.commit) return 0;
            const { dir, stateDir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
            const freed = await treeBytes(dir) + await treeBytes(stateDir);
            await fs.promises.rm(dir, { recursive: true, force: true });
            await fs.promises.rm(stateDir, { recursive: true, force: true });
            return freed;
        },
        // Pinning reads Hugging Face metadata only; no weights are downloaded.
        async pin(source) {
            if (source.commit && source.files) return source;
            const resolved = await resolveSnapshot({
                repo: source.repo,
                revision: source.revision || 'main',
                token: env.HF_TOKEN || '',
                baseUrl: hfBaseUrl,
            });
            const files = resolved.files.map((file) => ({ ...file }));
            return { ...source, commit: resolved.commit, files, size: files.reduce((sum, file) => sum + file.size, 0) };
        },
        // An update keeps the pin while repository and revision are unchanged.
        carryPin(next, previous) {
            if (next.commit || previous?.type !== 'hf-snapshot' || !previous.commit) return next;
            if (next.repo !== previous.repo || next.revision !== previous.revision) return next;
            return { ...next, commit: previous.commit, files: previous.files, size: previous.size };
        },
    });

    return Object.freeze({ huggingface, ollama, 'hf-snapshot': hfSnapshot });
}
