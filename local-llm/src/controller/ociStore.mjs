// The weight stores of OCI sources (DS002): `oci`, one GGUF file or every shard of a split GGUF, and `oci-snapshot`,
// CNCF per-file safetensors. Both are fetched by the controller, blob by blob, from
// `<registry>/v2/<repository>/blobs/sha256:<hex>` through the registry's session (ociRegistry.mjs), which carries the
// bearer to the registry origin only; a blob's CDN gets no credential. A layer's digest is the sha256 of the raw file,
// so every file is verified like any model file.
//
// Files live under `_oci/<id>`, in a directory whose last segment is the repository's last segment joined to the
// manifest digest with `@` (`ai/smollm2@sha256-<hex>`). No repository segment can hold `@`, so no OCI model's
// directory can lie inside another's (N6). Paths and keys come from the source alone: only `pin` and `fetch` consult
// the registry, through a session made once at their start.

import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import { downloadPinned, inspectPinned, planPinned, removePinned, verifyPinned } from './downloader.mjs';
import { pinOciGguf, pinOciSnapshot } from './ociRegistry.mjs';
import { OCI_DIGEST_RE, isOciGgufFile, isOciRepository } from './ociSource.mjs';
import { REGISTRY_ID_RE } from './registries.mjs';
import { freedBytes, orderCandidates, regularFilesIn } from './workspaceReuse.mjs';
import { isSnapshotFile } from './catalog.mjs';

export const OCI_IDENTITY_KEYS = Object.freeze(['registry', 'repository', 'manifestDigest', 'file', 'size', 'sha256']);

function invalidSource(message) {
    return new LocalLlmError('invalid_model', message);
}

function assertInside(root, candidate) {
    const relative = path.relative(root, candidate);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw invalidSource('An OCI path escapes the weights root');
}

/** The path segments of one OCI model under a weights root: `_oci/<id>/<repository…>@sha256-<hex>`. */
export function ociModelSegments({ registry, repository, manifestDigest }) {
    if (typeof registry !== 'string' || !REGISTRY_ID_RE.test(registry)) throw invalidSource('Invalid OCI registry id');
    if (!isOciRepository(repository)) throw invalidSource('Invalid OCI repository');
    if (typeof manifestDigest !== 'string' || !OCI_DIGEST_RE.test(manifestDigest)) throw invalidSource('Invalid OCI manifest digest');
    const segments = repository.split('/');
    return ['_oci', registry, ...segments.slice(0, -1), `${segments.at(-1)}@sha256-${manifestDigest.slice('sha256:'.length)}`];
}

/** Where one file of an `oci` source lives; its partial and records sit beside it, as for a GGUF file. */
export function ociGgufPaths({ root, source, file }) {
    if (!isOciGgufFile(file)) throw invalidSource('Invalid OCI GGUF file name');
    const base = path.resolve(root);
    const dir = path.join(base, ...ociModelSegments(source));
    const target = path.join(dir, ...file.split('/'));
    assertInside(base, dir);
    assertInside(dir, target);
    const partial = `${target}.partial`;
    return { root: base, dir, file: target, partial, identity: `${partial}.json`, meta: `${target}.json` };
}

/** The directories of an `oci-snapshot` source: `dir` holds only verified files, `stateDir` the partials and records. */
export function ociSnapshotPaths({ root, source }) {
    const base = path.resolve(root);
    const segments = ociModelSegments(source);
    const dir = path.join(base, ...segments);
    const stateDir = path.join(base, '.state', ...segments);
    assertInside(base, dir);
    assertInside(base, stateDir);
    return { dir, stateDir };
}

function ociSnapshotFilePaths({ root, source, file }) {
    if (!isSnapshotFile(file)) throw invalidSource('Invalid OCI snapshot file name');
    const { dir, stateDir } = ociSnapshotPaths({ root, source });
    const target = path.join(dir, file);
    const partial = path.join(stateDir, `${file}.partial`);
    assertInside(dir, target);
    assertInside(stateDir, partial);
    return { root: path.resolve(root), dir, file: target, partial, identity: `${partial}.json`, meta: path.join(stateDir, `${file}.json`) };
}

export const ociGgufKey = (source) => `gguf:_oci/${source.registry}/${source.repository}@${source.manifestDigest}/${source.file}${source.shards ? `#${source.shards.length}` : ''}`;
export const ociSnapshotKey = (source) => `hf:_oci/${source.registry}/${source.repository}@${source.manifestDigest}`;

/**
 * The two OCI stores. `sessions.forOci(source, { registry })` gives a source's session; `lazyCandidates`,
 * `sharedIndex` and `sumPlans` are the shared-models helpers of weightStores.mjs; the rest are the downloader's
 * pinned-file operations (seams for tests).
 */
export function createOciStores({
    ggufRoot,
    hfRoot,
    sessions,
    lazyCandidates,
    sharedIndex,
    sumPlans,
    fsApi = fs,
    download = downloadPinned,
    verify = verifyPinned,
    inspect = inspectPinned,
    plan = planPinned,
    remove = removePinned,
}) {
    const identityOf = (source, file) => ({
        registry: source.registry, repository: source.repository, manifestDigest: source.manifestDigest, file: file.file, size: file.size, sha256: file.sha256,
    });
    // One file per shard of a split GGUF, else the file itself, each with what its record holds and where it lives.
    const ggufFiles = (source) => (source.shards ?? [{ file: source.file, size: source.size, sha256: source.sha256 }])
        .map((file) => ({ file: file.file, size: file.size, artifact: identityOf(source, file), paths: ociGgufPaths({ root: ggufRoot, source, file: file.file }) }));
    const snapshotFiles = (source) => source.files
        .map((file) => ({ file: file.path, size: file.size, artifact: identityOf(source, { file: file.path, size: file.size, sha256: file.sha256 }),
            paths: ociSnapshotFilePaths({ root: hfRoot, source, file: file.path }) }));

    // State, plan and recheck of a list of files; a set is complete only when every file is.
    async function stateOf(files, total) {
        let bytes = 0;
        const states = [];
        for (const entry of files) {
            const found = await inspect({ paths: entry.paths, artifact: entry.artifact, identityKeys: OCI_IDENTITY_KEYS, fsApi });
            bytes += found.bytes || 0;
            states.push(found.state);
        }
        const all = (value) => states.every((state) => state === value);
        return { state: all('complete') ? 'complete' : all('absent') ? 'absent' : 'partial', bytes, total };
    }
    async function planOf(files, index, signal) {
        const found = index ?? await sharedIndex(files.map((entry) => entry.size), signal);
        const plans = [];
        for (const entry of files) {
            const { paths, artifact } = entry;
            const inspected = await inspect({ paths, artifact, identityKeys: OCI_IDENTITY_KEYS, fsApi });
            plans.push(await plan({ paths, artifact, identityKeys: OCI_IDENTITY_KEYS, candidates: orderCandidates(found.get(entry.size), entry.file), inspected, fsApi }));
        }
        return sumPlans(plans);
    }
    async function recheckOf(files, signal, onProgress) {
        const notes = [];
        for (const { paths, artifact } of files) notes.push(...(await verify({ paths, artifact, identityKeys: OCI_IDENTITY_KEYS, signal, onProgress, fsApi })).notes);
        return { notes };
    }
    // Every file through the job's one session, at its registry blob URL; progress counts the whole set.
    async function fetchAll(source, files, { signal, onProgress = () => {} }) {
        const session = sessions.forOci(source);
        const candidatesFor = lazyCandidates(files, signal);
        let done = 0;
        let transferred = 0;
        const provenance = [];
        const notes = [];
        const paths = [];
        for (const entry of files) {
            const offset = done;
            const result = await download({
                paths: entry.paths,
                artifact: entry.artifact,
                identityKeys: OCI_IDENTITY_KEYS,
                url: session.blobUrl(`sha256:${entry.artifact.sha256}`),
                fetchImpl: session.fetch,
                candidates: candidatesFor(entry),
                signal,
                fsApi,
                onProgress: (progress) => onProgress({ ...progress, bytes: offset + (progress.bytes || 0), total: source.size, transferred: transferred + (progress.transferred || 0) }),
            });
            paths.push(result.path);
            if (result.provenance) provenance.push(result.current ? { ...result.provenance, current: true } : result.provenance);
            notes.push(...(result.notes || []));
            done += entry.size;
            transferred += result.bytesTransferred || 0;
        }
        return { paths, bytes: source.size, bytesTransferred: transferred, provenance, notes };
    }

    const oci = Object.freeze({
        type: 'oci',
        fetchedBy: 'controller',
        key: ociGgufKey,
        isPinned: (source) => Boolean(source.manifestDigest && source.layout && source.file && (source.sha256 || Array.isArray(source.shards))),
        async state(source) {
            if (!oci.isPinned(source)) return { state: 'unpinned', bytes: 0, total: null };
            const found = await stateOf(ggufFiles(source), source.size);
            return source.shards ? { ...found, shards: source.shards.length } : found;
        },
        sizes: (source) => (oci.isPinned(source) ? ggufFiles(source).map((entry) => entry.size) : []),
        plan: async (source, { index = null, signal } = {}) => (oci.isPinned(source) ? planOf(ggufFiles(source), index, signal) : sumPlans([])),
        recheck: ({ artifact, signal, onProgress }) => recheckOf(ggufFiles(artifact), signal, onProgress),
        async fetch({ artifact, signal, onProgress }) {
            const result = await fetchAll(artifact, ggufFiles(artifact), { signal, onProgress });
            return { path: result.paths[0], files: result.paths, bytes: result.bytes, bytesTransferred: result.bytesTransferred, provenance: result.provenance, notes: result.notes };
        },
        async remove(source) {
            if (!oci.isPinned(source)) return 0;
            let freed = 0;
            for (const entry of ggufFiles(source)) freed += await remove({ paths: entry.paths, fsApi });
            return freed;
        },
        async pin(source, { signal, registry } = {}) {
            if (oci.isPinned(source)) return source;
            return pinOciGguf(sessions.forOci(source, { registry }), source, { signal });
        },
        // An update keeps the pin while registry, repository, tag and file are unchanged (and names no other digest).
        carryPin(next, previous) {
            if (oci.isPinned(next) || previous?.type !== 'oci' || !oci.isPinned(previous)) return next;
            if (next.registry !== previous.registry || next.repository !== previous.repository || next.tag !== previous.tag || next.file !== previous.file
                || (next.manifestDigest !== undefined && next.manifestDigest !== previous.manifestDigest)) return next;
            const { manifestDigest, layout, size, sha256, shards } = previous;
            return { ...next, manifestDigest, layout, size, ...(shards ? { shards } : { sha256 }) };
        },
        paths: (source) => ociGgufPaths({ root: ggufRoot, source, file: source.file }),
    });

    const ociSnapshot = Object.freeze({
        type: 'oci-snapshot',
        fetchedBy: 'controller',
        key: ociSnapshotKey,
        isPinned: (source) => Boolean(source.manifestDigest && source.files?.length),
        state: async (source) => (ociSnapshot.isPinned(source) ? stateOf(snapshotFiles(source), source.size) : { state: 'unpinned', bytes: 0, total: null }),
        sizes: (source) => (source.files || []).map((file) => file.size),
        plan: async (source, { index = null, signal } = {}) => (ociSnapshot.isPinned(source) ? planOf(snapshotFiles(source), index, signal) : sumPlans([])),
        recheck: ({ artifact, signal, onProgress }) => recheckOf(snapshotFiles(artifact), signal, onProgress),
        async fetch({ artifact, signal, onProgress }) {
            const result = await fetchAll(artifact, snapshotFiles(artifact), { signal, onProgress });
            const { dir } = ociSnapshotPaths({ root: hfRoot, source: artifact });
            return { path: dir, bytes: result.bytes, bytesTransferred: result.bytesTransferred, provenance: result.provenance, notes: result.notes };
        },
        // The whole snapshot and its records; the freed bytes count each inode whose last link goes with it.
        async remove(source) {
            if (!source.manifestDigest) return 0;
            const { dir, stateDir } = ociSnapshotPaths({ root: hfRoot, source });
            const freed = freedBytes([...await regularFilesIn(fsApi, dir), ...await regularFilesIn(fsApi, stateDir)]);
            await fsApi.promises.rm(dir, { recursive: true, force: true });
            await fsApi.promises.rm(stateDir, { recursive: true, force: true });
            return freed;
        },
        async pin(source, { signal, registry } = {}) {
            if (ociSnapshot.isPinned(source)) return source;
            return pinOciSnapshot(sessions.forOci(source, { registry }), source, { signal });
        },
        carryPin(next, previous) {
            if (ociSnapshot.isPinned(next) || previous?.type !== 'oci-snapshot' || !ociSnapshot.isPinned(previous)) return next;
            if (next.registry !== previous.registry || next.repository !== previous.repository || next.tag !== previous.tag
                || (next.manifestDigest !== undefined && next.manifestDigest !== previous.manifestDigest)) return next;
            return { ...next, manifestDigest: previous.manifestDigest, files: previous.files, size: previous.size };
        },
    });

    return Object.freeze({ oci, 'oci-snapshot': ociSnapshot });
}
