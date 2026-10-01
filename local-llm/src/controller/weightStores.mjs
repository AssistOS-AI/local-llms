// Where each kind of weights lives and how it is pinned, inspected, fetched
// and deleted. A source's `type` selects its store: `huggingface` is one GGUF
// file, or every shard of a split GGUF, that the controller downloads before
// the runner starts; `ollama` is a
// library tag that the Ollama runner pulls itself once it is running;
// `hf-snapshot` is a directory of files (safetensors for vLLM, EXL3 for
// TabbyAPI) that the controller downloads file by file.
//
// Every store has: `type`; `fetchedBy` ('controller' or 'runner');
// `key(source)`, the artifact identity used for "in use" checks, so runners
// that read the same file share it; `isPinned(source)`; `state(source)`;
// `remove(source)`; `pin(source)` and `carryPin(next, previous)` for Add and
// Update model; and, when the controller fetches, `fetch(...)`, `plan(...)`
// (how each file would be obtained and the disk it needs, C12) and
// `recheck(...)` (the last check before a runner loads the files).

import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import {
    artifactAcquisition,
    artifactPaths,
    downloadSnapshotFile,
    inspectSnapshotFile,
    resolveHuggingFaceSnapshot,
    snapshotFileAcquisition,
    snapshotPaths,
    verifyArtifact,
    verifySnapshotFile,
} from './downloader.mjs';
import { freedBytes, orderCandidates, regularFilesIn, walkShared } from './workspaceReuse.mjs';
import { splitGgufFiles, splitGgufName } from './catalog.mjs';
import {
    deleteOllamaModel,
    deleteOllamaPartials,
    fetchOllamaRegistryManifest,
    partialPullBytes,
    readOllamaManifest,
} from './ollamaStore.mjs';

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
    // Reads a tag's identity from the Ollama registry when a model is added (DS002).
    resolveOllama = fetchOllamaRegistryManifest,
    acquire = artifactAcquisition,
    acquireSnapshot = snapshotFileAcquisition,
    verify = verifyArtifact,
    verifySnapshot = verifySnapshotFile,
    // The workspace's shared model directory (C12): a pinned file found there
    // is used through a private, verified copy instead of a download.
    sharedModelsRoot = null,
    fsApi = fs,
}) {
    const adoptFrom = sharedModelsRoot ? [sharedModelsRoot] : [];
    // One walk of /shared per artifact (or per overview), for every size it needs.
    const sharedIndex = async (sizes, signal) => (adoptFrom.length ? walkShared(fsApi, adoptFrom, sizes, signal) : new Map());
    // The walk runs only when a file is first looked for there.
    const lazyCandidates = (files, signal) => {
        let index = null;
        return (file) => (adoptFrom.length ? async () => {
            index ??= await sharedIndex(files.map((entry) => entry.size), signal);
            return orderCandidates(index.get(file.size), file.file);
        } : null);
    };
    const sumPlans = (files) => ({ bytesNeeded: files.reduce((sum, entry) => sum + entry.bytesNeeded, 0), files });
    const ggufRoot = path.join(dataDir, 'models', 'gguf');
    const ollamaModels = path.join(dataDir, 'models', 'ollama');
    const hfRoot = path.join(dataDir, 'models', 'hf');

    // One file, or every shard of a split GGUF (DS002), each as the downloader's artifact.
    const filesOf = (source) => (source.shards
        ? source.shards.map((shard) => ({ ...source, shards: undefined, file: shard.file, size: shard.size, sha256: shard.sha256 }))
        : [source]);

    const huggingface = Object.freeze({
        type: 'huggingface',
        fetchedBy: 'controller',
        // A split GGUF is one artifact: its commit, first shard and shard count.
        key: (source) => `gguf:${source.repo}@${source.commit}/${source.file}${source.shards ? `#${source.shards.length}` : ''}`,
        isPinned: (source) => Boolean(source.commit) && (!splitGgufName(source.file) || Array.isArray(source.shards)),
        async state(source) {
            if (!huggingface.isPinned(source)) return { state: 'unpinned', bytes: 0, total: null };
            if (!source.shards) {
                const inspected = await inspect({ root: ggufRoot, artifact: source });
                return { ...inspected, total: source.size };
            }
            // The set is complete only when every shard is; bytes are counted over the set.
            const shards = [];
            for (const file of filesOf(source)) shards.push(await inspect({ root: ggufRoot, artifact: file }));
            const bytes = shards.reduce((sum, shard) => sum + (shard.bytes || 0), 0);
            const all = (value) => shards.every((shard) => shard.state === value);
            const stateOf = all('complete') ? 'complete' : all('absent') ? 'absent' : 'partial';
            return { state: stateOf, bytes, total: source.size, ...(source.shards ? { shards: shards.length } : {}) };
        },
        sizes: (source) => filesOf(source).map((file) => file.size),
        async plan(source, { index = null, signal } = {}) {
            if (!huggingface.isPinned(source)) return sumPlans([]);
            const files = filesOf(source);
            const found = index ?? await sharedIndex(files.map((file) => file.size), signal);
            const plans = [];
            for (const file of files) {
                const inspected = await inspect({ root: ggufRoot, artifact: file });
                plans.push(await acquire({ root: ggufRoot, artifact: file, inspected, candidates: orderCandidates(found.get(file.size), file.file) }));
            }
            return sumPlans(plans);
        },
        async recheck({ artifact, signal, onProgress }) {
            const notes = [];
            for (const file of filesOf(artifact)) {
                notes.push(...(await verify({ root: ggufRoot, artifact: file, baseUrl: hfBaseUrl, signal, onProgress })).notes);
            }
            return { notes };
        },
        async fetch({ artifact, signal, onProgress }) {
            // Shard by shard into one directory; progress counts the whole set.
            let done = 0;
            let transferred = 0;
            let first = null;
            const files = [];
            const provenance = [];
            const notes = [];
            const candidatesFor = lazyCandidates(filesOf(artifact), signal);
            for (const file of filesOf(artifact)) {
                const offset = done;
                const result = await download({
                    artifact: file,
                    root: ggufRoot,
                    token: env.HF_TOKEN || '',
                    baseUrl: hfBaseUrl,
                    candidates: candidatesFor(file),
                    onProgress: (progress) => onProgress?.({
                        ...progress,
                        bytes: offset + (progress.bytes || 0),
                        total: artifact.size,
                        transferred: transferred + (progress.transferred || 0),
                    }),
                    signal,
                });
                first ??= result.path;
                files.push(result.path);
                if (result.provenance) provenance.push(result.current ? { ...result.provenance, current: true } : result.provenance);
                notes.push(...(result.notes || []));
                done += file.size;
                transferred += result.bytesTransferred || 0;
            }
            return { path: first, files, bytes: artifact.size, bytesTransferred: transferred, provenance, notes };
        },
        async remove(source) {
            if (!source.commit) return 0;
            let freed = 0;
            for (const file of filesOf(source)) freed += await remove({ root: ggufRoot, artifact: file });
            return freed;
        },
        // Pinning reads Hugging Face metadata only; no weights are downloaded.
        async pin(source) {
            if (huggingface.isPinned(source)) return source;
            const resolve = (file, revision) => resolveHf({
                repo: source.repo,
                file,
                revision,
                token: env.HF_TOKEN || '',
                baseUrl: hfBaseUrl,
            });
            const resolved = await resolve(source.file, source.revision || 'main');
            const files = splitGgufFiles(source.file);
            if (!files) return { ...source, commit: resolved.commit, size: resolved.size, sha256: resolved.sha256 };
            // Every shard at the same commit.
            const shards = [{ file: source.file, size: resolved.size, sha256: resolved.sha256 }];
            for (const file of files.slice(1)) {
                const next = await resolve(file, resolved.commit);
                shards.push({ file, size: next.size, sha256: next.sha256 });
            }
            return { ...source, commit: resolved.commit, size: shards.reduce((sum, shard) => sum + shard.size, 0), shards };
        },
        // An update keeps the pinned commit while repository, file and
        // revision are unchanged; it never silently re-resolves a branch.
        carryPin(next, previous) {
            if (next.commit || previous?.type !== 'huggingface' || !previous.commit) return next;
            if (next.repo !== previous.repo || next.file !== previous.file || next.revision !== previous.revision) return next;
            return previous.shards
                ? { ...next, commit: previous.commit, size: previous.size, shards: previous.shards }
                : { ...next, commit: previous.commit, size: previous.size, sha256: previous.sha256 };
        },
        paths: (source) => artifactPaths({ root: ggufRoot, artifact: filesOf(source)[0] }),
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
        // On the committed cpu profile a tag is pinned when it is added: its manifest digest and the size of
        // its layers, read from the registry (manifest only, nothing is pulled). What the entry already
        // carries, typed or stored, must agree with the registry: a tag that moved since it was pinned, or a
        // size that was typed wrong (the CPU admission trusts it), is refused and never pinned silently. An
        // update that names a bare tag pins it afresh, which is how a moved tag is accepted. On every other
        // profile, and while the profile is undecided, nothing is read and the source is kept as it is: Add and
        // Run there are what they were before tags were pinned, and need no network.
        async pin(source, { profile = null, signal } = {}) {
            if (profile !== 'cpu') return source;
            const current = await resolveOllama(source.tag, { signal });
            if (source.manifestDigest && source.manifestDigest !== current.manifestDigest) {
                throw new LocalLlmError('identity_changed', `The Ollama tag ${source.tag} now resolves to ${current.manifestDigest}, `
                    + `not the pinned ${source.manifestDigest}; update the model entry without its manifestDigest and size to accept it.`);
            }
            if (source.size && source.size !== current.size) {
                throw new LocalLlmError('identity_changed', `The Ollama tag ${source.tag} is now ${current.size} bytes, not the pinned ${source.size}; `
                    + 'update the model entry without its manifestDigest and size to accept it.');
            }
            return { ...source, manifestDigest: current.manifestDigest, size: current.size };
        },
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
        sizes: (source) => (source.files || []).map((file) => file.size),
        async plan(source, { index = null, signal } = {}) {
            if (!hfSnapshot.isPinned(source)) return sumPlans([]);
            const found = index ?? await sharedIndex(source.files.map((file) => file.size), signal);
            const plans = [];
            for (const file of source.files) {
                const artifact = snapshotFile(source, file);
                const inspected = await inspectSnapshot({ root: hfRoot, artifact });
                plans.push(await acquireSnapshot({ root: hfRoot, artifact, inspected, candidates: orderCandidates(found.get(file.size), file.path) }));
            }
            return sumPlans(plans);
        },
        async recheck({ artifact: source, signal, onProgress }) {
            const notes = [];
            for (const file of source.files) {
                notes.push(...(await verifySnapshot({ root: hfRoot, artifact: snapshotFile(source, file), baseUrl: hfBaseUrl, signal, onProgress })).notes);
            }
            return { notes };
        },
        async fetch({ artifact: source, signal, onProgress = () => {} }) {
            let done = 0;
            let transferred = 0;
            const provenance = [];
            const notes = [];
            const artifacts = source.files.map((file) => snapshotFile(source, file));
            const candidatesFor = lazyCandidates(artifacts, signal);
            for (const artifact of artifacts) {
                const result = await downloadSnapshot({
                    root: hfRoot,
                    artifact,
                    token: env.HF_TOKEN || '',
                    baseUrl: hfBaseUrl,
                    candidates: candidatesFor(artifact),
                    signal,
                    onProgress: (progress) => onProgress({
                        ...progress,
                        bytes: done + progress.bytes,
                        total: source.size,
                        transferred: transferred + (progress.transferred ?? 0),
                    }),
                });
                done += artifact.size;
                transferred += result.bytesTransferred;
                if (result.provenance) provenance.push(result.current ? { ...result.provenance, current: true } : result.provenance);
                notes.push(...(result.notes || []));
                onProgress({ bytes: done, total: source.size, transferred, rate: 0, etaSeconds: null });
            }
            const { dir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
            return { path: dir, bytes: source.size, bytesTransferred: transferred, provenance, notes };
        },
        // Delete weights removes the whole snapshot and its bookkeeping; the
        // freed bytes count each inode whose last link goes with it (R8).
        async remove(source) {
            if (!source.commit) return 0;
            const { dir, stateDir } = snapshotPaths({ root: hfRoot, repo: source.repo, commit: source.commit });
            const freed = freedBytes([...await regularFilesIn(fsApi, dir), ...await regularFilesIn(fsApi, stateDir)]);
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
