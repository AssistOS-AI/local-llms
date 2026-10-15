// The Hugging Face lookup of the Add model form (DS002): the files a repository
// offers, the sizing read from a GGUF header or a config.json, the token that
// goes to the Hugging Face origin only, and the header read again from the
// verified file. Every request goes to a local fake Hugging Face; nothing here
// reaches the network.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { HF_FILE_SEGMENT_RE, HF_REPO_RE, HF_REVISION_RE } from 'pworker/local-llm/internal/controller/catalog.mjs';
import { lookupHuggingFaceModel, normalizeLookupInput } from 'pworker/local-llm/internal/controller/modelLookup.mjs';
import { createStateStore } from 'pworker/local-llm/internal/controller/stateStore.mjs';

import { createAgentController as createController } from '../src/engineOptions.mjs';

import { controllerHandlers } from '../src/controlHandlers.mjs';
import { modelEntryFromForm } from '../IDE-plugins/local-llm-settings/local-llm-settings-model.js';
import { TOOL_OPERATIONS, handleTool } from '../tools/local_llm_tool.mjs';
import { GGUF_TYPE as T, ggufBytes, modelPairs } from './gguf-fixture.mjs';

const MIB = 1024 * 1024;
const COMMIT = 'c'.repeat(40);
const OTHER_COMMIT = 'd'.repeat(40);
const REPO = 'acme/models';
const TOKEN = 'hf_secret_token_value';
const sha = (char) => char.repeat(64);
const codeOf = (code) => (error) => error?.code === code;

// ------------------------------------------------------------------ fake Hugging Face

// ------------------------------------------------------------------ fake Hugging Face

const lfs = (file, size, digest) => ({ type: 'file', path: file, size, oid: 'f'.repeat(40), lfs: { oid: digest, size, pointerSize: 134 } });

const directory = (file) => ({ type: 'directory', path: file, oid: '2'.repeat(40), size: 0 });

// A server that logs every request (with the Authorization and Range it got) and answers with `handler`.
async function listen(t, handler) {
    const requests = [];
    const server = http.createServer((req, res) => {
        requests.push({ method: req.method, url: req.url, authorization: req.headers.authorization, range: req.headers.range, encoding: req.headers['accept-encoding'] });
        try {
            handler(req, res, new URL(req.url, 'http://placeholder.invalid'));
        } catch (error) {
            res.writeHead(500);
            res.end(String(error));
        }
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => { server.closeAllConnections(); server.close(); });
    return { origin: `http://127.0.0.1:${server.address().port}`, requests };
}

function sendJson(res, status, body, headers = {}) {
    const text = JSON.stringify(body);
    res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text), ...headers });
    res.end(text);
}

// Serves the bytes of a file honouring a range request, like the real file host.
function sendFile(req, res, body) {
    const range = /^bytes=(\d+)-(\d+)?$/.exec(req.headers.range ?? '');
    if (!range) {
        res.writeHead(200, { 'Content-Length': body.length });
        res.end(body);
        return;
    }
    const start = Number(range[1]);
    const end = Math.min(body.length - 1, range[2] === undefined ? body.length - 1 : Number(range[2]));
    res.writeHead(206, { 'Content-Range': `bytes ${start}-${end}/${body.length}`, 'Content-Length': end - start + 1 });
    res.end(body.subarray(start, end + 1));
}

// A body sent as it is, either whole with a Content-Length or in pieces with none; `state.sent` counts what left before the client went away.
function sendRaw(res, body, { chunked = false, state = {} } = {}) {
    const bytes = Buffer.from(body);
    res.writeHead(200, { 'Content-Type': 'application/json', ...(chunked ? {} : { 'Content-Length': bytes.length }) });
    state.sent = 0;
    res.on('close', () => { state.closed = true; });
    let offset = 0;
    const pump = () => {
        while (offset < bytes.length && !res.destroyed) {
            const piece = bytes.subarray(offset, offset + 64 * 1024);
            const more = res.write(piece);
            offset += piece.length;
            state.sent = offset;
            if (!more) { res.once('drain', pump); return; }
        }
        if (!res.destroyed) res.end();
    };
    pump();
}

// A very large file that is never held in memory: the first bytes are real, the rest zeros, written with backpressure.
// `state` records how much was sent and whether the client went away first.
function streamVirtual(req, res, { size, head }, state) {
    const range = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range ?? '');
    const end = range ? Math.min(size - 1, Number(range[2])) : size - 1;
    res.writeHead(range ? 206 : 200, {
        'Content-Length': end + 1,
        ...(range ? { 'Content-Range': `bytes 0-${end}/${size}` } : {}),
    });
    const zeros = Buffer.alloc(64 * 1024);
    let sent = 0;
    state.sent = 0;
    state.complete = false;
    res.on('close', () => { state.closed = true; });
    const pump = () => {
        while (sent < end + 1 && !res.destroyed) {
            const piece = sent === 0 ? Buffer.concat([head, zeros.subarray(0, Math.max(0, zeros.length - head.length))]) : zeros.subarray(0, Math.min(zeros.length, end + 1 - sent));
            const more = res.write(piece);
            sent += piece.length;
            state.sent = sent;
            if (!more) { res.once('drain', pump); return; }
        }
        if (!res.destroyed) { state.complete = true; res.end(); }
    };
    pump();
}

/**
 * A fake repository. `tree[dir]` is a list of pages of entries (`''` is the root); `files[path]` is `{ body }`,
 * `{ redirect: (origin) => url }`, `{ status }` or `{ virtual: { size, head } }`; `info` is the model's JSON.
 */
function hfHandler(repo, origin) {
    const id = repo.id ?? REPO;
    return (req, res, url) => {
        const pathname = decodeURIComponent(url.pathname);
        if (pathname.startsWith('/cache/') && repo.cacheBody) { sendFile(req, res, repo.cacheBody); return; }
        if (pathname === `/api/models/${id}`) {
            if (repo.infoRedirect) { res.writeHead(302, { Location: repo.infoRedirect(origin) }); res.end(); return; }
            if (repo.infoStatus) { res.writeHead(repo.infoStatus); res.end(); return; }
            if (repo.infoRaw) { sendRaw(res, repo.infoRaw, repo.raw); return; }
            sendJson(res, 200, repo.info ?? { id, gated: false, cardData: { license: 'apache-2.0' } });
            return;
        }
        const revision = /^\/api\/models\/[^/]+\/[^/]+\/revision\/(.+)$/.exec(pathname);
        if (revision) {
            if (repo.revisionRedirect) { res.writeHead(302, { Location: repo.revisionRedirect(origin) }); res.end(); return; }
            if (repo.revisionStatus) { res.writeHead(repo.revisionStatus); res.end(); return; }
            if (repo.revisionRaw) { sendRaw(res, repo.revisionRaw, repo.raw); return; }
            // A branch that has moved on since a lookup answers with its newest commit; a commit answers as itself.
            sendJson(res, 200, { sha: revision[1] === 'main' && repo.mainCommit ? repo.mainCommit : COMMIT });
            return;
        }
        // A repository has one listing per commit: `commits` names them, and without it there is the one at COMMIT.
        const tree = new RegExp(`^/api/models/${id}/tree/([0-9a-f]{40})(?:/(.+))?$`).exec(pathname);
        if (tree) {
            const at = repo.commits?.[tree[1]] ?? (tree[1] === COMMIT ? repo : null);
            const dir = tree[2] ?? '';
            if (!at) { res.writeHead(404); res.end(); return; }
            const moved = repo.treeRedirect?.(origin, dir);
            if (moved) { res.writeHead(302, { Location: moved }); res.end(); return; }
            if (repo.treeStatus) { res.writeHead(repo.treeStatus); res.end(); return; }
            if (repo.treeRaw) { sendRaw(res, repo.treeRaw, repo.raw); return; }
            if (at.treeBody !== undefined) { sendJson(res, 200, at.treeBody); return; }
            const pages = at.tree?.[dir];
            if (!pages) { res.writeHead(404); res.end(); return; }
            const index = Number(url.searchParams.get('cursor') ?? 0);
            const next = index + 1 < pages.length || repo.endlessPages === true
                ? (repo.link ? repo.link(dir, index, origin) : `${origin}/api/models/${id}/tree/${tree[1]}${dir ? `/${dir}` : ''}?cursor=${index + 1}`)
                : null;
            sendJson(res, 200, pages[Math.min(index, pages.length - 1)], next ? { Link: `<${next}>; rel="next"` } : {});
            return;
        }
        const resolve = new RegExp(`^/${id}/resolve/([0-9a-f]{40})/(.+)$`).exec(pathname);
        if (resolve) {
            const at = repo.commits?.[resolve[1]] ?? (resolve[1] === COMMIT ? repo : null);
            const spec = at?.files?.[resolve[2]];
            if (!spec) { res.writeHead(404); res.end(); return; }
            if (spec.redirect) { res.writeHead(302, { Location: spec.redirect(origin) }); res.end(); return; }
            if (spec.status) { res.writeHead(spec.status); res.end(); return; }
            if (spec.virtual) { streamVirtual(req, res, spec.virtual, spec.state); return; }
            if (spec.ignoreRange) { res.writeHead(200, { 'Content-Length': spec.body.length }); res.end(spec.body); return; }
            sendFile(req, res, spec.body);
            return;
        }
        res.writeHead(404);
        res.end();
    };
}

async function fakeHf(t, repo) {
    const holder = { origin: null };
    const server = await listen(t, (req, res, url) => hfHandler(repo, holder.origin)(req, res, url));
    holder.origin = server.origin;
    return { ...server, repo };
}

// The header of a Qwen2.5-0.5B-like model: 24 layers, 14 heads, 2 KV heads, 896 wide, 32768 context.
const QWEN_HEADER = ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896, context: 32768 }) }).bytes;

const pad = (header, size = 4096) => Buffer.concat([header, Buffer.alloc(Math.max(0, size - header.length), 0xee)]);

// ------------------------------------------------------------------ the tests

test('lookup refuses bad repository, revision and file input before any request', async () => {
    let requests = 0;
    const fetchImpl = async () => { requests += 1; throw new Error('no request may be made for bad input'); };
    const bad = [
        { repo: `${'a'.repeat(96)}/${'b'.repeat(97)}` },
        { repo: 'Qwen/Qwen☃' }, { repo: 'noslash' }, { repo: '../etc/passwd' }, { repo: 'a/b/c' }, { repo: '' }, { repo: 7 }, {}, { repo: 'a..b/c' },
        { repo: REPO, revision: '../main' }, { repo: REPO, revision: 'a'.repeat(129) }, { repo: REPO, revision: 'has space' }, { repo: REPO, revision: 7 },
        { repo: REPO, file: '../x.gguf' }, { repo: REPO, file: 'a/b/c/d/e.gguf' }, { repo: REPO, file: 'model.bin' }, { repo: REPO, file: `${'a'.repeat(513)}.gguf` },
        { repo: REPO, file: 'with space.gguf' }, { repo: REPO, file: '/abs.gguf' }, { repo: REPO, file: 7 },
        { repo: REPO, file: 'm-00002-of-00003.gguf' },
        { repo: REPO, format: 'pt' }, { repo: REPO, format: 'hf', file: 'm.gguf' }, { repo: REPO, provider: 'modelscope' }, { repo: REPO, provider: null },
        { repo: REPO, token: 'hf_x' }, { repo: REPO, url: 'https://evil.example/x' },
    ];
    for (const args of [...bad, null, undefined, 'owner/name', ['owner/name']]) {
        await assert.rejects(() => lookupHuggingFaceModel(args, { fetchImpl, baseUrl: 'http://hf.invalid' }), codeOf('invalid_request'), JSON.stringify(args));
    }
    assert.equal(requests, 0, 'no request was made');
    // The valid forms pass the same gate, and so do the limits themselves.
    assert.deepEqual(normalizeLookupInput({ repo: REPO }), { repo: REPO, revision: 'main', format: 'gguf' });
    assert.deepEqual(normalizeLookupInput({ repo: `${'a'.repeat(96)}/${'b'.repeat(96)}`, revision: 'r'.repeat(128), format: 'exl3', provider: 'huggingface' }).format, 'exl3');
    assert.deepEqual(normalizeLookupInput({ repo: REPO, file: 'a/b/c/d.gguf' }).file, 'a/b/c/d.gguf');
    assert.deepEqual(normalizeLookupInput({ repo: REPO, file: 'm-00001-of-00003.gguf' }).file, 'm-00001-of-00003.gguf');
    // The tool's schema uses the same expressions, so the form, the tool and the controller agree.
    const config = JSON.parse(fs.readFileSync(new URL('../mcp-config.json', import.meta.url), 'utf8'));
    const tool = config.tools.find((entry) => entry.name === 'local_llm_model_lookup');
    const properties = tool.inputSchema.properties;
    assert.deepEqual(Object.keys(properties).sort(), ['file', 'format', 'provider', 'repo', 'revision']);
    assert.equal(properties.repo.pattern, HF_REPO_RE.source);
    assert.equal(properties.repo.maxLength, 193);
    assert.equal(properties.revision.pattern, HF_REVISION_RE.source);
    assert.equal(properties.revision.maxLength, 128);
    assert.deepEqual(properties.format.enum, ['gguf', 'hf', 'exl3']);
    // No `default` in the schema: Ploinky AgentServer refuses that keyword on a string and the agent never becomes ready.
    // The default lives in the code instead, which the normalisation above shows (no format gives gguf).
    assert.equal(Object.hasOwn(properties.format, 'default'), false);
    assert.deepEqual(properties.provider.enum, ['huggingface']);
    assert.equal(properties.file.maxLength, 512);
    assert.ok(new RegExp(properties.file.pattern).test('dir/M-Q4_K_M.gguf') && !new RegExp(properties.file.pattern).test('../x.gguf'));
    assert.deepEqual(tool.inputSchema.required, ['repo']);
    assert.equal(tool.inputSchema.additionalProperties, false);
    assert.equal(tool.timeoutMs, 60000);
    assert.ok(HF_FILE_SEGMENT_RE.test('M-Q4_K_M.gguf'));
});

const KEY = 'k'.repeat(43);
const ARM64_CONTRACT = Object.freeze({ architecture: 'arm64', llama_cpp: 'b11159', gpu_compute_capabilities: '12.1' });
const cpuSnapshot = () => ({
    gpu: { available: false, state: 'absent', reason: 'No GPU is attached to this agent: GPU not applied to this Box yet' },
    memory: { totalBytes: 6036128 * 1024, availableBytes: 3122576 * 1024 },
    cgroupMemory: null, disk: { freeBytes: 100_000 * MIB, totalBytes: 400_000 * MIB }, cpus: 4, cores: 4,
});

function deferred() {
    let resolve;
    const promise = new Promise((ok) => { resolve = ok; });
    return { promise, resolve };
}

// A controller on the cpu profile, its hardware and runner injected, its Hugging Face a fake one. `weights` is what a
// download leaves on disk for the pinned GGUF; `download` replaces the whole download.
const dedicatedSnapshot = () => ({
    gpu: { available: true, name: 'NVIDIA GeForce RTX 3060', memoryModel: 'dedicated', totalBytes: 12 * 1024 * MIB, freeBytes: 12 * 1024 * MIB, usedBytes: 0, processes: [] },
    memory: { totalBytes: 32 * 1024 * MIB, availableBytes: 28 * 1024 * MIB }, disk: { freeBytes: 400_000 * MIB, totalBytes: 800_000 * MIB },
});

function harness(t, { hf, weights = null, download = null, registry = null, seed = [], dedicated = false } = {}) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-lookup-'));
    if (registry) {
        fs.mkdirSync(path.join(dir, 'state'), { recursive: true });
        fs.writeFileSync(path.join(dir, 'state', 'controller.json'), JSON.stringify({ version: 1, deployment: null, params: {}, requests: {}, registry }));
    }
    const file = path.join(dir, 'weights.gguf');
    if (weights) fs.writeFileSync(file, weights);
    const started = [];
    const downloads = [];
    const controller = createController({
        dataDir: dir,
        env: { PATH: '/usr/bin', HF_TOKEN: TOKEN },
        seedCatalog: seed,
        hfBaseUrl: hf?.origin ?? 'http://hf.invalid',
        stateStore: createStateStore({ dataDir: dir }),
        snapshot: async () => (dedicated ? dedicatedSnapshot() : cpuSnapshot()),
        download: download ?? (async ({ artifact }) => { downloads.push(artifact.file); return { status: 'complete', path: file, bytesTransferred: 0 }; }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        startRunner({ command, args, env, log }) {
            const exit = deferred();
            const handle = {
                pid: 7000 + started.length, command, args, env, log, exited: exit.promise, running: true,
                async stop() { handle.running = false; exit.resolve({ code: 0, signal: 'SIGTERM', error: null }); return exit.promise; },
                async kill() { return handle.stop(); },
            };
            started.push(handle);
            return handle;
        },
        // The lookup and the pin talk to the fake Hugging Face; the runner's readiness probe gets a canned answer.
        fetchImpl: (url, init) => (hf && String(url).startsWith(hf.origin) ? fetch(url, init) : Promise.resolve({ ok: true, status: 200, json: async () => ({}) })),
        apiKeyFactory: () => KEY,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion, reason: null }),
        pollMs: 2,
        stopGraceMs: 50,
        // A GPU host: no image contract (every runner is present) and a dedicated GPU, so the profile is `dedicated`.
        imageContract: dedicated ? null : ARM64_CONTRACT,
        fileExists: (candidate) => dedicated || candidate === '/opt/llama.cpp/llama-server',
        sharedModelsRoot: null,
        readMemory: () => ({ totalBytes: 6036128 * 1024, availableBytes: 3122576 * 1024 }),
        readPressure: () => 0,
        unifiedGuardMs: 5,
        dropCache: () => true,
    });
    // The drain saves state and writes the log, so the directory goes only after it.
    t.after(async () => {
        await controller.drain();
        fs.rmSync(dir, { recursive: true, force: true });
    });
    const logLines = (text) => {
        let log = '';
        try { log = fs.readFileSync(path.join(dir, 'logs', 'runner.log'), 'utf8'); } catch { /* no log yet */ }
        return log.split('\n').filter((line) => line.includes(text));
    };
    return { controller, started, downloads, logLines, dir, file, stored: () => JSON.parse(fs.readFileSync(path.join(dir, 'state', 'controller.json'), 'utf8')) };
}

test('Add pins the commit a lookup read, so a branch that moved in between does not change what was sized', async (t) => {
    const older = { tree: { '': [[lfs('small-Q4_K_M.gguf', 400 * MIB, sha('a')), lfs('model.safetensors', 900 * MIB, sha('a'))]] }, files: { 'small-Q4_K_M.gguf': { body: pad(QWEN_HEADER) } } };
    const newer = { tree: { '': [[lfs('small-Q4_K_M.gguf', 500 * MIB, sha('b')), lfs('model.safetensors', 1000 * MIB, sha('b'))]] }, files: { 'small-Q4_K_M.gguf': { body: pad(QWEN_HEADER) } } };
    const hf = await fakeHf(t, { commits: { [COMMIT]: older, [OTHER_COMMIT]: newer }, mainCommit: COMMIT });
    const h = harness(t, { hf });
    const gguf = await h.controller.lookupModel({ repo: REPO, file: 'small-Q4_K_M.gguf' });
    const snapshot = await h.controller.lookupModel({ repo: REPO, format: 'hf' });
    assert.deepEqual([gguf.commit, snapshot.commit], [COMMIT, COMMIT]);
    // The branch moves on before Add is pressed.
    hf.repo.mainCommit = OTHER_COMMIT;
    const form = (kind, extra = {}) => ({
        id: `pinned-${kind}`, sourceKind: kind, repo: REPO, revision: '', file: 'small-Q4_K_M.gguf', contextLength: '32768', layers: '24', kvBytesPerToken: '12288', ...extra,
    });
    const pinned = await h.controller.addModel(modelEntryFromForm(form('huggingface', { sizingSource: 'gguf-header', commit: gguf.commit })));
    assert.deepEqual([pinned.model.sources.gguf.commit, pinned.model.sources.gguf.size, pinned.model.sources.gguf.sha256], [COMMIT, 400 * MIB, sha('a')]);
    const snap = await h.controller.addModel(modelEntryFromForm(form('hf', { sizingSource: 'config.json', commit: snapshot.commit })));
    assert.deepEqual([snap.model.sources.hf.commit, snap.model.sources.hf.size], [COMMIT, 900 * MIB]);
    assert.equal(h.stored().registry.find((entry) => entry.id === 'pinned-hf').sources.hf.revision, COMMIT);
    // The same form without the commit is what it was: main, which is now somewhere else.
    const drifted = await h.controller.addModel(modelEntryFromForm(form('huggingface', { id: 'drifted-gguf' })));
    assert.deepEqual([drifted.model.sources.gguf.commit, drifted.model.sources.gguf.size, drifted.model.sources.gguf.sha256], [OTHER_COMMIT, 500 * MIB, sha('b')]);
});

test('the lookup tool reaches the controller operation, admin only, and the control socket serves it', async () => {
    const seen = [];
    const call = async (op, args) => { seen.push([op, args]); return { ok: true }; };
    const admin = { user: { roles: ['admin'] } };
    await handleTool('local_llm_model_lookup', { repo: REPO, revision: 'main', format: 'gguf', file: 'm-Q4_K_M.gguf' }, { authInfo: admin, call });
    await handleTool('local_llm_model_lookup', { repo: REPO }, { authInfo: admin, call });
    assert.deepEqual(seen, [
        ['lookupModel', { repo: REPO, revision: 'main', format: 'gguf', file: 'm-Q4_K_M.gguf' }],
        ['lookupModel', { repo: REPO }],
    ]);
    await assert.rejects(() => handleTool('local_llm_model_lookup', { repo: REPO }, { authInfo: { user: { roles: ['user'] } }, call }), codeOf('admin_required'));
    assert.equal(TOOL_OPERATIONS.local_llm_model_lookup.op, 'lookupModel');
    const handlers = controllerHandlers({ lookupModel: async (args) => ({ looked: args }) });
    assert.deepEqual(await handlers.lookupModel({ repo: REPO }), { looked: { repo: REPO } });
});

