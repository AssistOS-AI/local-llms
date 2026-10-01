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

import { controllerHandlers } from '../src/controlHandlers.mjs';
import { admissionResult } from '../src/controller/admission.mjs';
import { HF_FILE_SEGMENT_RE, HF_REPO_RE, HF_REVISION_RE, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { DownloadError, fetchJson } from '../src/controller/downloader.mjs';
import { LOOKUP_METADATA_MAX_BYTES, lookupHuggingFaceModel, normalizeLookupInput, quantizationOf, sizingFromConfig } from '../src/controller/modelLookup.mjs';
import { admitCpuLlamaServer, admitUnifiedLlamaServer, admitUnifiedVllm } from '../src/controller/profiles.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { DRAIN_QUEUE_WAIT_MS } from '../src/drainBudget.mjs';
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

const lfs = (file, size, digest) => ({ type: 'file', path: file, size, oid: 'f'.repeat(40), lfs: { oid: digest, size, pointerSize: 134 } });
const plain = (file, size = 100) => ({ type: 'file', path: file, size, oid: '1'.repeat(40) });
const directory = (file) => ({ type: 'directory', path: file, oid: '2'.repeat(40), size: 0 });
const gitBlobOid = (bytes) => crypto.createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const sha256Of = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

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

const ask = (hf, args, options = {}) => lookupHuggingFaceModel(args, { baseUrl: hf.origin, ...options });
const queries = (hf) => hf.requests.map((request) => request.url.replace(/\?.*$/, ''));

// The header of a Qwen2.5-0.5B-like model: 24 layers, 14 heads, 2 KV heads, 896 wide, 32768 context.
const QWEN_HEADER = ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896, context: 32768 }) }).bytes;
// gpt-oss-20b as llama.cpp writes it: the architecture is "gpt-oss", with a hyphen, and every key carries that prefix.
const GPT_OSS_HEADER = ggufBytes({
    kv: modelPairs({
        arch: 'gpt-oss', layers: 24, heads: 64, kvHeads: 8, embedding: 2880, context: 131072, experts: 32, name: 'gpt-oss-20b',
        extra: [['gpt-oss.attention.key_length', T.u32, 64], ['gpt-oss.attention.value_length', T.u32, 64], ['gpt-oss.attention.sliding_window', T.u32, 128]],
    }),
}).bytes;
const headerOf = (options = {}) => pad(ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896, ...options }) }).bytes, 65536);
const QWEN_SIZING = { contextLength: 32768, architecture: 'dense', memory: { layers: 24, kvBytesPerToken: 12288 }, source: 'gguf-header' };
const pad = (header, size = 4096) => Buffer.concat([header, Buffer.alloc(Math.max(0, size - header.length), 0xee)]);

// ------------------------------------------------------------------ the tests

test('lookup lists GGUF files with sizes and sha256, grouping split shards under the first', async (t) => {
    const hf = await fakeHf(t, {
        tree: {
            '': [[
                plain('.gitattributes'), plain('README.md'),
                lfs('Model-Q4_K_M.gguf', 400 * MIB, sha('a')),
                lfs('Model-Q8_0.gguf', 700 * MIB, sha('b')),
                lfs('Model-IQ3_XS.gguf', 300 * MIB, sha('9')),
                lfs('Model-BF16.gguf', 1000 * MIB, sha('8')),
                plain('Model-small.gguf', 1000),
                lfs('big-Q6_K-00001-of-00002.gguf', 600 * MIB, sha('c')),
                lfs('big-Q6_K-00002-of-00002.gguf', 500 * MIB, sha('d')),
                lfs('partial-Q5_K_M-00001-of-00003.gguf', 100 * MIB, sha('e')),
                lfs('partial-Q5_K_M-00003-of-00003.gguf', 100 * MIB, sha('f')),
                lfs('evil<img src=x onerror=alert(1)>.gguf', 100 * MIB, sha('1')),
                lfs('odd name.gguf', 100 * MIB, sha('2')),
                directory('sub'), directory('bad dir name'),
            ]],
            sub: [[
                lfs('sub/deep-Q4_0-00001-of-00002.gguf', 50 * MIB, sha('3')),
                lfs('sub/deep-Q4_0-00002-of-00002.gguf', 60 * MIB, sha('4')),
                lfs('sub/single-f16.gguf', 70 * MIB, sha('5')),
                lfs('elsewhere/other.gguf', 70 * MIB, sha('6')),
                directory('sub/too-deep'),
            ]],
        },
    });
    const result = await ask(hf, { repo: REPO });
    assert.deepEqual([result.provider, result.repo, result.revision, result.commit, result.format], ['huggingface', REPO, 'main', COMMIT, 'gguf']);
    assert.deepEqual([result.gated, result.license, result.truncated, result.sizing], [false, 'apache-2.0', false, null]);
    assert.deepEqual(result.files.map((row) => row.file), [
        'Model-BF16.gguf', 'Model-IQ3_XS.gguf', 'Model-Q4_K_M.gguf', 'Model-Q8_0.gguf', 'big-Q6_K-00001-of-00002.gguf', 'sub/deep-Q4_0-00001-of-00002.gguf', 'sub/single-f16.gguf',
    ]);
    const row = (file) => result.files.find((entry) => entry.file === file);
    assert.deepEqual(row('Model-Q4_K_M.gguf'), { file: 'Model-Q4_K_M.gguf', size: 400 * MIB, sha256: sha('a'), gitOid: null, quantization: 'Q4_K_M', shards: null });
    assert.deepEqual(row('big-Q6_K-00001-of-00002.gguf'), {
        file: 'big-Q6_K-00001-of-00002.gguf', size: 1100 * MIB, sha256: null, gitOid: null, quantization: 'Q6_K',
        shards: [
            { file: 'big-Q6_K-00001-of-00002.gguf', size: 600 * MIB, sha256: sha('c') },
            { file: 'big-Q6_K-00002-of-00002.gguf', size: 500 * MIB, sha256: sha('d') },
        ],
    });
    assert.deepEqual(row('sub/deep-Q4_0-00001-of-00002.gguf').shards.map((shard) => shard.file), ['sub/deep-Q4_0-00001-of-00002.gguf', 'sub/deep-Q4_0-00002-of-00002.gguf']);
    assert.equal(row('sub/deep-Q4_0-00001-of-00002.gguf').size, 110 * MIB);
    assert.deepEqual([row('Model-IQ3_XS.gguf').quantization, row('Model-BF16.gguf').quantization, row('sub/single-f16.gguf').quantization], ['IQ3_XS', 'BF16', 'F16']);
    // Not listed: a file kept in git (no sha256 to pin), an incomplete split set, a later shard, a name Add would refuse, and
    // a file another directory's listing claimed.
    for (const file of ['Model-small.gguf', 'partial-Q5_K_M-00001-of-00003.gguf', 'big-Q6_K-00002-of-00002.gguf', 'odd name.gguf', 'elsewhere/other.gguf']) {
        assert.equal(row(file), undefined, file);
    }
    assert.equal(result.files.some((entry) => entry.file.includes('<')), false);
    // Every listed file is one Add accepts once it is given its revision.
    for (const entry of result.files) {
        const source = { type: 'huggingface', repo: REPO, file: entry.file, revision: 'main' };
        assert.doesNotThrow(() => validateModel({ id: 'listed', sources: { gguf: source } }), entry.file);
    }
    // The requests: the model, the revision, the root, and one level of subdirectory named like a file segment; no token without one.
    assert.deepEqual(queries(hf), [
        `/api/models/${REPO}`, `/api/models/${REPO}/revision/main`, `/api/models/${REPO}/tree/${COMMIT}`, `/api/models/${REPO}/tree/${COMMIT}/sub`,
    ]);
    assert.deepEqual(hf.requests.map((request) => request.authorization), [undefined, undefined, undefined, undefined]);
});

test('lookup sends the token to the Hugging Face origin on every request and lists every page of a directory', async (t) => {
    const page = (start, count) => Array.from({ length: count }, (_, index) => lfs(`m${start + index}-Q4_K_M.gguf`, (start + index + 1) * MIB, sha('a')));
    const hf = await fakeHf(t, { tree: { '': [page(0, 3), page(3, 3), page(6, 2)] } });
    const result = await ask(hf, { repo: REPO, revision: COMMIT }, { token: TOKEN });
    assert.equal(result.files.length, 8);
    assert.equal(result.truncated, false);
    // A revision that is already a commit needs no revision request; three tree pages are read.
    assert.deepEqual(queries(hf).filter((url) => url.includes('/revision/')), []);
    assert.equal(hf.requests.filter((request) => request.url.includes('/tree/')).length, 3);
    assert.ok(hf.requests.length >= 4);
    for (const request of hf.requests) assert.equal(request.authorization, `Bearer ${TOKEN}`, request.url);
});

test('lookup reads at most 32 directories, 500 files and 50 pages, and says when it stopped early', async (t) => {
    // 33 directories, one level: the 33rd is never listed.
    const names = Array.from({ length: 33 }, (_, index) => `dir${String(index).padStart(2, '0')}`);
    const crowded = await fakeHf(t, {
        tree: {
            '': [names.map(directory)],
            ...Object.fromEntries(names.map((name) => [name, [[lfs(`${name}/m-Q4_K_M.gguf`, MIB, sha('a'))]]])),
        },
    });
    const dirs = await ask(crowded, { repo: REPO });
    assert.equal(dirs.files.length, 32);
    assert.equal(dirs.truncated, true);
    assert.equal(crowded.requests.filter((request) => /\/tree\/[^/]+\/dir/.test(request.url)).length, 32);
    assert.equal(dirs.files.some((row) => row.file.startsWith('dir32/')), false);
    // 32 directories are within the bound.
    const exact = await fakeHf(t, { tree: { '': [names.slice(0, 32).map(directory)], ...Object.fromEntries(names.slice(0, 32).map((name) => [name, [[]]])) } });
    assert.equal((await ask(exact, { repo: REPO })).truncated, false);
    // 600 files in one answer: 500 are kept.
    const many = await fakeHf(t, { tree: { '': [Array.from({ length: 600 }, (_, index) => lfs(`m${String(index).padStart(3, '0')}-Q4_K_M.gguf`, MIB, sha('a')))] } });
    const files = await ask(many, { repo: REPO });
    assert.equal(files.files.length, 500);
    assert.equal(files.truncated, true);
    assert.equal(files.files[0].file, 'm000-Q4_K_M.gguf');
    // 500 files are within the bound.
    const fits = await fakeHf(t, { tree: { '': [Array.from({ length: 500 }, (_, index) => lfs(`m${String(index).padStart(3, '0')}-Q4_K_M.gguf`, MIB, sha('a')))] } });
    const all = await ask(fits, { repo: REPO });
    assert.deepEqual([all.files.length, all.truncated], [500, false]);
    // The page budget is shared by the root and the directories: 50 pages in all.
    const pages = await fakeHf(t, { tree: { '': [[directory('sub')]], sub: [[lfs('sub/m-Q4_K_M.gguf', MIB, sha('a'))]] }, endlessPages: true });
    const shared = await ask(pages, { repo: REPO });
    assert.equal(pages.requests.filter((request) => request.url.includes('/tree/')).length, 50);
    assert.equal(shared.truncated, true);
});

test('lookup reports gated repositories and licences, and 401 or 403 as access_denied naming HF_TOKEN', async (t) => {
    const tree = { '': [[lfs('a-Q4_K_M.gguf', MIB, sha('a'))]] };
    const gatedOf = async (info) => {
        const hf = await fakeHf(t, { info, tree });
        const { gated, license } = await ask(hf, { repo: REPO });
        return [gated, license];
    };
    assert.deepEqual(await gatedOf({ gated: 'manual', cardData: { license: 'apache-2.0' } }), ['manual', 'apache-2.0']);
    assert.deepEqual(await gatedOf({ gated: 'auto', cardData: { license: 'llama3.1' } }), ['auto', 'llama3.1']);
    assert.deepEqual(await gatedOf({ gated: false, tags: ['gguf', 'license:mit'] }), [false, 'mit']);
    assert.deepEqual(await gatedOf({ gated: false, cardData: { license: ['other', 'mit'] } }), [false, 'other']);
    assert.deepEqual(await gatedOf({ cardData: {} }), [null, null], 'a response that says nothing is null, not false');
    assert.deepEqual(await gatedOf({ gated: true, cardData: { license: 'x'.repeat(81) } }), [null, null]);
    assert.deepEqual(await gatedOf({ gated: 'manual', cardData: { license: 'evil\u0007\nlicense' }, tags: ['license:ok'] }), ['manual', 'ok']);
    // A refusal at any step names the token and the way to set it; never the token itself.
    const refusals = [
        { infoStatus: 401 }, { infoStatus: 403 }, { treeStatus: 401, tree }, { treeStatus: 403, tree },
        { tree, files: { 'a-Q4_K_M.gguf': { status: 401 } }, file: 'a-Q4_K_M.gguf' },
        { tree, files: { 'a-Q4_K_M.gguf': { status: 403 } }, file: 'a-Q4_K_M.gguf' },
    ];
    for (const { file, ...repo } of refusals) {
        const hf = await fakeHf(t, repo);
        await assert.rejects(() => ask(hf, { repo: REPO, ...(file ? { file } : {}) }, { token: TOKEN }), (error) => {
            assert.equal(error.code, 'access_denied', JSON.stringify(repo));
            assert.match(error.message, /HF_TOKEN/);
            assert.match(error.message, /ploinky var HF_TOKEN <token>/);
            assert.doesNotMatch(error.message, new RegExp(TOKEN));
            assert.equal(JSON.stringify(error.details ?? {}).includes(TOKEN), false);
            return true;
        });
    }
    // A missing repository, revision or file is not_found; any other status or a broken response is lookup_failed.
    for (const [repo, code] of [
        [{ infoStatus: 404 }, 'not_found'], [{ revisionStatus: 404, tree }, 'not_found'], [{ treeStatus: 404, tree }, 'not_found'],
        [{ infoStatus: 500 }, 'lookup_failed'], [{ treeStatus: 503, tree }, 'lookup_failed'], [{ treeBody: { not: 'a list' } }, 'lookup_failed'],
        [{ info: ['not', 'an', 'object'], tree }, 'lookup_failed'],
    ]) {
        const hf = await fakeHf(t, repo);
        await assert.rejects(() => ask(hf, { repo: REPO }), codeOf(code), JSON.stringify(repo));
    }
    const hf = await fakeHf(t, { tree, files: {} });
    await assert.rejects(() => ask(hf, { repo: REPO, file: 'a-Q4_K_M.gguf' }), codeOf('not_found'), 'a listed file whose bytes are missing');
    await assert.rejects(() => ask(hf, { repo: REPO, file: 'not-there.gguf' }), codeOf('not_found'));
    // A server that is not there at all (nothing listens on port 1).
    await assert.rejects(() => lookupHuggingFaceModel({ repo: REPO }, { baseUrl: 'http://127.0.0.1:1' }), codeOf('lookup_failed'));
});

test('lookup follows only same-origin tree pages and never sends the token to another origin', async (t) => {
    const tree = [[lfs('one-Q4_K_M.gguf', MIB, sha('a'))], [lfs('two-Q4_K_M.gguf', MIB, sha('b'))]];
    // A next link on another origin is not followed, and that server sees nothing at all.
    const elsewhere = await listen(t, (req, res) => sendJson(res, 200, [lfs('planted-Q4_K_M.gguf', MIB, sha('c'))]));
    const linked = await fakeHf(t, { tree: { '': tree }, link: (dir, index) => `${elsewhere.origin}/api/models/${REPO}/tree/${COMMIT}?cursor=${index + 1}` });
    const cut = await ask(linked, { repo: REPO }, { token: TOKEN });
    assert.deepEqual(cut.files.map((row) => row.file), ['one-Q4_K_M.gguf'], 'only the first page');
    assert.equal(cut.truncated, true, 'a listing cut short says so');
    assert.deepEqual(elsewhere.requests, [], 'the other origin was never contacted');
    // The same pages behind a same-origin link are read, with the token.
    const same = await fakeHf(t, { tree: { '': tree } });
    const whole = await ask(same, { repo: REPO }, { token: TOKEN });
    assert.deepEqual(whole.files.map((row) => row.file), ['one-Q4_K_M.gguf', 'two-Q4_K_M.gguf']);
    assert.equal(whole.truncated, false);
    // A server that always has a next page is read 50 pages, then the listing is cut.
    const endless = await fakeHf(t, { tree: { '': [[lfs('x-Q4_K_M.gguf', MIB, sha('a'))]] }, endlessPages: true });
    const capped = await ask(endless, { repo: REPO });
    assert.equal(endless.requests.filter((request) => request.url.includes('/tree/')).length, 50);
    assert.equal(capped.truncated, true);

    // A range GET redirected to another origin (another port): the file host there never receives the token; ours does.
    const cdn = await listen(t, (req, res) => sendFile(req, res, pad(QWEN_HEADER)));
    const redirected = await fakeHf(t, {
        tree: { '': [[lfs('m-Q4_K_M.gguf', 4096, sha('a'))]] },
        files: { 'm-Q4_K_M.gguf': { redirect: () => `${cdn.origin}/cdn/m.gguf?signature=abc` } },
    });
    const result = await ask(redirected, { repo: REPO, file: 'm-Q4_K_M.gguf' }, { token: TOKEN });
    assert.deepEqual({ ...result.sizing, readBytes: undefined, notes: undefined }, { ...QWEN_SIZING, readBytes: undefined, notes: undefined });
    assert.equal(cdn.requests.length, 1);
    assert.equal(cdn.requests[0].authorization, undefined, 'the CDN never receives Authorization');
    assert.equal(cdn.requests[0].range, `bytes=0-${32 * MIB - 1}`, 'the range survives the redirect');
    assert.ok(redirected.requests.every((request) => request.authorization === `Bearer ${TOKEN}`), 'every request to the Hugging Face origin carries the token');
    // A redirect to the same origin keeps the token on its second hop.
    const home = await fakeHf(t, {
        tree: { '': [[lfs('m-Q4_K_M.gguf', 4096, sha('a'))]] },
        files: { 'm-Q4_K_M.gguf': { redirect: (origin) => `${origin}/cache/m.gguf` } },
        cacheBody: pad(QWEN_HEADER),
    });
    assert.equal((await ask(home, { repo: REPO, file: 'm-Q4_K_M.gguf' }, { token: TOKEN })).sizing.memory.kvBytesPerToken, 12288);
    const second = home.requests.filter((request) => request.url.startsWith('/cache/'));
    assert.deepEqual(second.map((request) => request.authorization), [`Bearer ${TOKEN}`], 'the second hop is the same origin, so it carries the token');
    // The metadata calls are guarded too: the model's info redirected to another origin is not followed at all (see the next test).
    const infoHost = await listen(t, (req, res) => sendJson(res, 200, { gated: 'auto', cardData: { license: 'mit' } }));
    const moved = await fakeHf(t, { tree: { '': tree }, infoRedirect: () => `${infoHost.origin}/info` });
    await assert.rejects(() => ask(moved, { repo: REPO }, { token: TOKEN }), codeOf('lookup_failed'));
    assert.deepEqual(infoHost.requests, [], 'the metadata host behind a redirect is never contacted');
    // Redirect loops, a redirect with no place to go, and a redirect away from http(s) end the lookup.
    const loop = await fakeHf(t, { tree: { '': [[lfs('m-Q4_K_M.gguf', 4096, sha('a'))]] }, files: { 'm-Q4_K_M.gguf': { redirect: (origin) => `${origin}/${REPO}/resolve/${COMMIT}/m-Q4_K_M.gguf` } } });
    await assert.rejects(() => ask(loop, { repo: REPO, file: 'm-Q4_K_M.gguf' }), codeOf('lookup_failed'));
    assert.equal(loop.requests.filter((request) => request.url.includes('/resolve/')).length, 6, 'the original request and five redirects, then it gives up');
    // A redirect away from http(s) is refused before any request is made for it, and so is a step down from https to http.
    const seen = [];
    const recording = (url, init) => { seen.push(String(url)); return fetch(url, init); };
    for (const target of ['file:///etc/passwd', 'ftp://127.0.0.1/model.gguf']) {
        const hostile = await fakeHf(t, { tree: { '': [[lfs('m-Q4_K_M.gguf', 4096, sha('a'))]] }, files: { 'm-Q4_K_M.gguf': { redirect: () => target } } });
        await assert.rejects(() => ask(hostile, { repo: REPO, file: 'm-Q4_K_M.gguf' }, { fetchImpl: recording }), codeOf('lookup_failed'), target);
    }
    assert.deepEqual(seen.filter((url) => !url.startsWith('http://127.0.0.1')), [], 'no request was made to a non-http address');
    const downgraded = [];
    const stepDown = async (url) => { downgraded.push(String(url)); return { status: 302, headers: new Headers({ location: 'http://hf.example/api/models/acme/models' }), body: null }; };
    await assert.rejects(() => lookupHuggingFaceModel({ repo: REPO }, { baseUrl: 'https://hf.example', fetchImpl: stepDown, token: TOKEN }), codeOf('lookup_failed'));
    assert.deepEqual(downgraded, [`https://hf.example/api/models/${REPO}`], 'the http address was never requested');
});

test('lookup reads one file\'s header with a bounded range request and aborts once parsed', async (t) => {
    const state = {};
    const hf = await fakeHf(t, {
        tree: { '': [[lfs('m-Q4_K_M.gguf', 512 * MIB, sha('a')), lfs('split-Q4_K_M-00001-of-00002.gguf', 300 * MIB, sha('b')), lfs('split-Q4_K_M-00002-of-00002.gguf', 300 * MIB, sha('c'))]] },
        files: {
            'm-Q4_K_M.gguf': { virtual: { size: 512 * MIB, head: QWEN_HEADER }, state },
            'split-Q4_K_M-00001-of-00002.gguf': { body: pad(QWEN_HEADER) },
        },
    });
    const result = await ask(hf, { repo: REPO, file: 'm-Q4_K_M.gguf' }, { token: TOKEN });
    const { readBytes, notes, ...sizing } = result.sizing;
    assert.deepEqual(sizing, QWEN_SIZING);
    assert.deepEqual(notes, []);
    assert.ok(readBytes > 0 && readBytes <= 64 * 1024, `the header took ${readBytes} bytes`);
    const get = hf.requests.find((request) => request.url.endsWith('/m-Q4_K_M.gguf'));
    assert.equal(get.range, `bytes=0-${32 * MIB - 1}`, 'one ranged request for the first 32 MiB');
    // Raw bytes: a compressed body would make the byte range mean something else.
    assert.match(get.encoding, /identity/);
    assert.doesNotMatch(get.encoding, /gzip|deflate|br/);
    assert.equal(get.authorization, `Bearer ${TOKEN}`);
    // The server is cut off long before the 32 MiB it was asked to be ready to send, and the file's 512 MiB.
    const until = Date.now() + 3000;
    while (state.closed !== true && Date.now() < until) await new Promise((resolve) => setTimeout(resolve, 10));
    assert.equal(state.closed, true, 'the connection was closed by the client');
    assert.equal(state.complete, false, 'the response was not sent to its end');
    assert.ok(state.sent < 16 * MIB, `the server sent ${state.sent} bytes`);
    // A split set is read from its first shard, which holds the whole key-value section.
    const split = await ask(hf, { repo: REPO, file: 'split-Q4_K_M-00001-of-00002.gguf' });
    assert.deepEqual({ ...split.sizing, readBytes: undefined, notes: undefined }, { ...QWEN_SIZING, readBytes: undefined, notes: undefined });
    assert.equal(hf.requests.filter((request) => /00002-of-00002/.test(request.url)).length, 0, 'no other shard is read');
    // A server that ignores the range and sends the whole file is cut off at the same place.
    const ignoring = await fakeHf(t, { tree: { '': [[lfs('m-Q4_K_M.gguf', 4096, sha('a'))]] }, files: { 'm-Q4_K_M.gguf': { body: pad(QWEN_HEADER), ignoreRange: true } } });
    assert.equal((await ask(ignoring, { repo: REPO, file: 'm-Q4_K_M.gguf' })).sizing.memory.kvBytesPerToken, 12288);
    // Bytes that are not a GGUF header, or end inside it, are invalid_gguf, never another error.
    const bad = await fakeHf(t, {
        tree: { '': [[lfs('text-Q4_K_M.gguf', 4096, sha('a')), lfs('cut-Q4_K_M.gguf', 4096, sha('b')), lfs('empty-Q4_K_M.gguf', 4096, sha('d'))]] },
        files: {
            'text-Q4_K_M.gguf': { body: Buffer.from('<html>not a model</html>'.repeat(100)) },
            'cut-Q4_K_M.gguf': { body: QWEN_HEADER.subarray(0, QWEN_HEADER.length - 5) },
            'empty-Q4_K_M.gguf': { status: 416 },
        },
    });
    for (const file of ['text-Q4_K_M.gguf', 'cut-Q4_K_M.gguf', 'empty-Q4_K_M.gguf']) {
        await assert.rejects(() => ask(bad, { repo: REPO, file }), codeOf('invalid_gguf'), file);
    }
    // A file that is not among the listed ones, or names a later shard, is refused.
    await assert.rejects(() => ask(hf, { repo: REPO, file: 'nope.gguf' }), codeOf('not_found'));
    await assert.rejects(() => ask(hf, { repo: REPO, file: 'split-Q4_K_M-00002-of-00002.gguf' }), codeOf('invalid_request'));
});

test('lookup of an hf or exl3 snapshot derives sizing from its pinned config.json', async (t) => {
    const config = Buffer.from(JSON.stringify({ architectures: ['Qwen2ForCausalLM'], num_hidden_layers: 24, hidden_size: 896, num_attention_heads: 14, num_key_value_heads: 2, max_position_embeddings: 32768 }));
    const tree = (configEntry, extra = []) => ({
        '': [[
            configEntry, lfs('model.safetensors', 900 * MIB, sha('a')), lfs('model-00002.safetensors', 10 * MIB, sha('9')), plain('tokenizer.json', 5000), plain('generation_config.json', 120),
            lfs('pytorch_model.bin', 900 * MIB, sha('b')), plain('modeling.py', 400), plain('notes.md', 10), directory('original'), ...extra,
        ]],
    });
    // A config.json kept in git is pinned by its blob oid.
    const inGit = await fakeHf(t, { tree: tree({ ...plain('config.json', config.length), oid: gitBlobOid(config) }), files: { 'config.json': { body: config } } });
    for (const format of ['hf', 'exl3']) {
        const result = await ask(inGit, { repo: REPO, format }, { token: TOKEN });
        assert.equal(result.format, format);
        assert.deepEqual(result.files.map((row) => row.file), ['config.json', 'generation_config.json', 'model-00002.safetensors', 'model.safetensors', 'tokenizer.json'],
            'the top-level model files a snapshot takes, no .bin, .py or subdirectory');
        assert.deepEqual(result.files.find((row) => row.file === 'config.json'), { file: 'config.json', size: config.length, sha256: null, gitOid: gitBlobOid(config), quantization: null, shards: null });
        assert.equal(result.files.find((row) => row.file === 'model.safetensors').sha256, sha('a'));
        assert.deepEqual({ ...result.sizing, readBytes: undefined, notes: undefined },
            { contextLength: 32768, architecture: 'dense', memory: { layers: 24, kvBytesPerToken: 12288 }, source: 'config.json', readBytes: undefined, notes: undefined });
        assert.equal(result.sizing.readBytes, config.length);
    }
    assert.equal(inGit.requests.filter((request) => request.url.includes('/original')).length, 0, 'no subdirectory is read for a snapshot');
    // A config.json kept in LFS is pinned by its sha256.
    const inLfs = await fakeHf(t, { tree: tree(lfs('config.json', config.length, sha256Of(config))), files: { 'config.json': { body: config } } });
    assert.equal((await ask(inLfs, { repo: REPO, format: 'hf' })).sizing.memory.kvBytesPerToken, 12288);
    // Bytes that differ from the pinned digest, or are longer than the pinned size, are not used.
    const other = Buffer.from(JSON.stringify({ num_hidden_layers: 1 }).padEnd(config.length, ' '));
    for (const served of [other, Buffer.concat([config, Buffer.from(' ')]), config.subarray(0, config.length - 1)]) {
        const swapped = await fakeHf(t, { tree: tree({ ...plain('config.json', config.length), oid: gitBlobOid(config) }), files: { 'config.json': { body: served } } });
        await assert.rejects(() => ask(swapped, { repo: REPO, format: 'hf' }), codeOf('lookup_failed'));
    }
    const swappedLfs = await fakeHf(t, { tree: tree(lfs('config.json', other.length, sha256Of(config))), files: { 'config.json': { body: other } } });
    await assert.rejects(() => ask(swappedLfs, { repo: REPO, format: 'exl3' }), (error) => error.code === 'lookup_failed' && /digest/.test(error.message));
    // Not JSON (with a matching digest), or no config.json at all, or one beyond the read bound.
    const text = Buffer.from('this is not json');
    const notJson = await fakeHf(t, { tree: tree({ ...plain('config.json', text.length), oid: gitBlobOid(text) }), files: { 'config.json': { body: text } } });
    await assert.rejects(() => ask(notJson, { repo: REPO, format: 'hf' }), codeOf('lookup_failed'));
    const none = await fakeHf(t, { tree: { '': [[lfs('model.safetensors', MIB, sha('a'))]] } });
    assert.equal((await ask(none, { repo: REPO, format: 'hf' })).sizing, null);
    const huge = await fakeHf(t, { tree: tree(lfs('config.json', 2 * MIB, sha('7'))), files: { 'config.json': { body: Buffer.alloc(2 * MIB) } } });
    assert.equal((await ask(huge, { repo: REPO, format: 'hf' })).sizing, null);
    assert.equal(huge.requests.filter((request) => request.url.endsWith('/config.json')).length, 0, 'a config.json beyond the bound is not fetched');
    // What the config says: experts, a nested text_config, latent attention, an explicit head size.
    const sized = (value) => sizingFromConfig(value);
    assert.equal(sized({ num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, max_position_embeddings: 4096, num_local_experts: 8 }).architecture, 'moe');
    assert.equal(sized({ num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, max_position_embeddings: 4096, num_experts: 0 }).architecture, 'dense');
    assert.deepEqual(sized({ model_type: 'vision', text_config: { num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, num_key_value_heads: 2, max_position_embeddings: 4096 } }).memory,
        { layers: 4, kvBytesPerToken: 4 * 2 * 64 * 4 });
    assert.equal(sized({ num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, head_dim: 128, num_key_value_heads: 2 }).memory.kvBytesPerToken, 4 * 2 * 128 * 4);
    for (const marker of ['kv_lora_rank', 'q_lora_rank', 'mamba_d_state', 'ssm_cfg', 'linear_num_key_heads']) {
        const latent = sized({ num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, [marker]: 16 });
        assert.equal(latent.memory.kvBytesPerToken, null, marker);
        assert.ok(latent.notes.some((note) => note.includes(marker)), marker);
    }
    assert.equal(sized({ num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, layer_types: ['full_attention', 'linear_attention'] }).memory.kvBytesPerToken, null);
    assert.equal(sized({ num_hidden_layers: 4, hidden_size: 512, num_attention_heads: 8, layer_types: ['full_attention', 'sliding_attention'] }).memory.kvBytesPerToken, 4 * 8 * 64 * 4);
    assert.deepEqual(sized({}).memory, { layers: null, kvBytesPerToken: null });
    assert.equal(sized({ num_hidden_layers: 1025 }).memory.layers, null);
});

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

test('the quantization is read from the file name', () => {
    const cases = {
        'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf': 'Q4_K_M', 'qwen2.5-0.5b-instruct-q8_0.gguf': 'Q8_0', 'Model-IQ3_XS.gguf': 'IQ3_XS', 'Model-UD-Q3_K_XL.gguf': 'UD-Q3_K_XL',
        'Model-BF16.gguf': 'BF16', 'model-f16.gguf': 'F16', 'gpt-oss-20b-MXFP4.gguf': 'MXFP4', 'sub/dir/Big-Q6_K-00001-of-00009.gguf': 'Q6_K',
        'Qwen3-4B-Instruct-2507-Q4_K_M.gguf': 'Q4_K_M', 'Llama-3.2-1B-Instruct-Q5_K_S.gguf': 'Q5_K_S',
        'model.gguf': null, 'Qwen2.5-0.5B.gguf': null,
    };
    for (const [file, quantization] of Object.entries(cases)) assert.equal(quantizationOf(file), quantization, file);
});

test('the user-sizing warning names where the values came from, and keeps its text for values typed by hand', () => {
    const GIB = 1024 * MIB;
    const typedLlama = 'Sized with memory.kvBytesPerToken from the model entry added at run time. Nothing checks these against the weights: '
        + 'an understated value makes the estimate too small, and then only the memory guard stands behind the run.';
    const typedVllm = 'Sized with memory.kvBytesPerToken from the model entry added at run time. Nothing checks these against the weights: '
        + 'an understated value only loosens the check that the weights and KV cache fit vLLM\'s share, so vLLM may fail to start; '
        + 'its memory need (the share plus its runner RAM) does not depend on it.';
    const entry = (extra) => ({ id: 'u', displayName: 'u', seed: false, architecture: 'dense', contextLength: 32768, memory: { kvBytesPerToken: 12288 }, ...extra });
    const source = { type: 'huggingface', size: 400 * MIB };
    const disk = { freeBytes: 100_000 * MIB };
    const warningsOf = {
        cpu: (model) => admitCpuLlamaServer({
            model, source, params: { ctxSize: 4096, parallel: 1 }, memory: { totalBytes: 6036128 * 1024, availableBytes: 3122576 * 1024 }, disk,
            gpu: { available: false }, decision: { cause: 'absent', reason: 'No GPU' },
        }, admissionResult).warnings,
        unified: (model) => admitUnifiedLlamaServer({
            runnerId: 'llama.cpp', model, source, params: { ctxSize: 4096, parallel: 1, loadMode: 'mmap', mtp: false },
            memory: { totalBytes: 128 * GIB, availableBytes: 100 * GIB }, disk,
        }, admissionResult).warnings,
        vllm: (model) => admitUnifiedVllm({
            model, source, params: { maxModelLen: 4096, kvCacheDtype: 'auto', gpuMemoryUtilization: null },
            memory: { totalBytes: 128 * GIB, availableBytes: 100 * GIB }, disk,
        }, admissionResult).warnings,
    };
    const sizedWith = (runner, extra) => warningsOf[runner](entry(extra)).find((warning) => warning.startsWith('Sized with'));
    // Typed by hand, or marked manual: the text that was always there, word for word.
    for (const extra of [{}, { sizingSource: 'manual' }]) {
        assert.equal(sizedWith('cpu', extra), typedLlama, JSON.stringify(extra));
        assert.equal(sizedWith('unified', extra), typedLlama, JSON.stringify(extra));
        assert.equal(sizedWith('vllm', extra), typedVllm, JSON.stringify(extra));
    }
    // A GGUF header read at lookup: says it will be read again, and what happens until then.
    for (const runner of ['cpu', 'unified']) {
        assert.equal(sizedWith(runner, { sizingSource: 'gguf-header' }),
            'Sized with memory.kvBytesPerToken from the GGUF header read when the model was added; the header is read again from the verified file '
            + 'after the download, and the last admissions use the values it gives; until then an understated value makes the estimate too small, '
            + 'and then only the memory guard stands behind the run.', runner);
        // The copy the controller admits with after the download says the values are the file's own.
        assert.equal(sizedWith(runner, { sizingSource: 'gguf-header', sizingVerified: true }),
            'Sized with memory.kvBytesPerToken from the GGUF header of the downloaded file, which was checked against its sha256; '
            + 'the header read when the model was added was read again from it.', runner);
        assert.match(sizedWith(runner, { sizingSource: 'config.json' }),
            /^Sized with memory\.kvBytesPerToken from the config\.json of the model's pinned snapshot, checked against its digest when the model was looked up\. Nothing checks these against the weights: an understated value makes the estimate too small/, runner);
    }
    // vLLM loads a snapshot, not the GGUF file, so nothing reads that header again for its run.
    assert.match(sizedWith('vllm', { sizingSource: 'gguf-header' }), /^Sized with memory\.kvBytesPerToken from the GGUF header read when the model was added; vLLM loads the snapshot, not that file, so nothing reads it again: an understated value only loosens/);
    assert.match(sizedWith('vllm', { sizingSource: 'gguf-header', sizingVerified: true }), /nothing reads it again/);
    assert.match(sizedWith('vllm', { sizingSource: 'config.json' }), /^Sized with memory\.kvBytesPerToken from the config\.json of the model's pinned snapshot/);
    // A catalog entry carries no user-sizing warning whatever it says, and an entry with no user sizing has none.
    for (const runner of ['cpu', 'unified', 'vllm']) {
        assert.equal(sizedWith(runner, { seed: true, sizingSource: 'gguf-header' }), undefined, runner);
        assert.equal(sizedWith(runner, { memory: {}, sizingSource: 'gguf-header' }), undefined, runner);
    }
});

test('a hyphenated architecture is sized, and a header whose architecture cannot be used gives no sizing instead of an error', async (t) => {
    const named = (arch, extra = []) => ggufBytes({ kv: [['general.architecture', T.string, arch], [`${arch}.block_count`, T.u32, 24], ...extra] }).bytes;
    const hf = await fakeHf(t, {
        tree: { '': [[
            lfs('gpt-oss-20b-Q4_K_M.gguf', 4096, sha('a')), lfs('command-r-Q4_K_M.gguf', 4096, sha('b')), lfs('noarch-Q4_K_M.gguf', 4096, sha('c')),
            lfs('badname-Q4_K_M.gguf', 4096, sha('d')), lfs('dotted-Q4_K_M.gguf', 4096, sha('e')), lfs('text-Q4_K_M.gguf', 4096, sha('f')),
        ]] },
        files: {
            'gpt-oss-20b-Q4_K_M.gguf': { body: pad(GPT_OSS_HEADER) },
            'command-r-Q4_K_M.gguf': { body: pad(ggufBytes({ kv: modelPairs({ arch: 'command-r', layers: 40, heads: 64, kvHeads: 64, embedding: 8192, context: 131072 }) }).bytes) },
            'noarch-Q4_K_M.gguf': { body: ggufBytes({ kv: [['general.name', T.string, 'x']] }).bytes },
            'badname-Q4_K_M.gguf': { body: named('Bad Arch!') },
            'dotted-Q4_K_M.gguf': { body: named('a.b') },
            'text-Q4_K_M.gguf': { body: Buffer.from('<html>not a model</html>'.repeat(100)) },
        },
    });
    const sizingOf = async (file) => (await ask(hf, { repo: REPO, file })).sizing;
    const oss = await sizingOf('gpt-oss-20b-Q4_K_M.gguf');
    assert.deepEqual({ ...oss, readBytes: undefined }, {
        contextLength: 131072, architecture: 'moe', memory: { layers: 24, kvBytesPerToken: 24 * 8 * 128 * 2 }, source: 'gguf-header', readBytes: undefined, notes: [],
    });
    const commandR = await sizingOf('command-r-Q4_K_M.gguf');
    assert.deepEqual([commandR.contextLength, commandR.memory], [131072, { layers: 40, kvBytesPerToken: 40 * 64 * 256 * 2 }]);
    // Well-formed headers the reader cannot size: every value null, dense, one note; the pick goes on.
    for (const [file, note] of [
        ['noarch-Q4_K_M.gguf', /^general\.architecture is missing$/],
        ['badname-Q4_K_M.gguf', /^general\.architecture is not a name this agent can read/],
        ['dotted-Q4_K_M.gguf', /^general\.architecture is not a name this agent can read/],
    ]) {
        const sizing = await sizingOf(file);
        assert.deepEqual([sizing.contextLength, sizing.architecture, sizing.memory, sizing.source], [null, 'dense', { layers: null, kvBytesPerToken: null }, 'gguf-header'], file);
        assert.equal(sizing.notes.length, 1, file);
        assert.match(sizing.notes[0], note, file);
        assert.ok(sizing.readBytes > 0, file);
    }
    // A header that is malformed as GGUF is still invalid_gguf.
    await assert.rejects(() => ask(hf, { repo: REPO, file: 'text-Q4_K_M.gguf' }), codeOf('invalid_gguf'));
});

test('a metadata answer is taken from the Hugging Face origin only: a redirect to another origin is refused, never listed', async (t) => {
    // Another origin that would answer every metadata request with a plausible listing of its own.
    const planted = await listen(t, (req, res, url) => sendJson(res, 200, /\/revision\//.test(url.pathname) ? { sha: COMMIT }
        : url.pathname.includes('/tree/') ? [lfs('planted-Q4_K_M.gguf', MIB, sha('c'))] : { gated: false, cardData: { license: 'mit' } }));
    const real = { '': [[lfs('real-Q4_K_M.gguf', MIB, sha('a')), directory('sub')]], sub: [[lfs('sub/deep-Q4_K_M.gguf', MIB, sha('b'))]] };
    const cases = {
        'the model info': { tree: real, infoRedirect: () => `${planted.origin}/info` },
        'the revision': { tree: real, revisionRedirect: () => `${planted.origin}/revision` },
        'the first tree page': { tree: real, treeRedirect: () => `${planted.origin}/api/models/${REPO}/tree/${COMMIT}` },
        'a subdirectory page': { tree: real, treeRedirect: (origin, dir) => (dir === 'sub' ? `${planted.origin}/api/models/${REPO}/tree/${COMMIT}/sub` : null) },
    };
    for (const [what, repo] of Object.entries(cases)) {
        const hf = await fakeHf(t, repo);
        await assert.rejects(() => ask(hf, { repo: REPO }, { token: TOKEN }), (error) => {
            assert.equal(error.code, 'lookup_failed', what);
            assert.match(error.message, /redirected a metadata request to another origin/, what);
            assert.doesNotMatch(error.message, /planted|127\.0\.0\.1/, 'the message does not name the other origin');
            return true;
        }, what);
    }
    assert.deepEqual(planted.requests, [], 'the other origin was never contacted, with or without the token');
    // A redirect that stays on the origin is followed, and the token goes with it.
    let redirected = false;
    const same = await fakeHf(t, { tree: real, treeRedirect: (origin, dir) => {
        if (dir !== '' || redirected) return null;
        redirected = true;
        return `${origin}/api/models/${REPO}/tree/${COMMIT}?cursor=0`;
    } });
    const result = await ask(same, { repo: REPO }, { token: TOKEN });
    assert.deepEqual(result.files.map((row) => row.file), ['real-Q4_K_M.gguf', 'sub/deep-Q4_K_M.gguf']);
    assert.ok(same.requests.every((request) => request.authorization === `Bearer ${TOKEN}`));
    assert.equal(same.requests.filter((request) => request.url.includes('/tree/') && !request.url.includes('/sub')).length, 2, 'the original request and its redirect');
});

test('a metadata response larger than the cap is lookup_failed and read no further, while downloads keep reading whole bodies', async (t) => {
    assert.equal(LOOKUP_METADATA_MAX_BYTES, 8 * MIB);
    const tree = { '': [[lfs('a-Q4_K_M.gguf', MIB, sha('a'))]] };
    const padded = (shape, bytes) => JSON.stringify(shape('x'.repeat(bytes)));
    const at = (what, bytes) => ({
        info: { tree, infoRaw: padded((pad) => ({ gated: false, pad }), bytes) },
        revision: { tree, revisionRaw: padded((pad) => ({ sha: COMMIT, pad }), bytes) },
        tree: { treeRaw: padded((pad) => [{ type: 'file', path: 'a-Q4_K_M.gguf', pad }], bytes) },
    })[what];
    // More than the default cap, sent with a Content-Length and in pieces without one.
    for (const what of ['info', 'revision', 'tree']) {
        for (const chunked of [false, true]) {
            const state = {};
            const hf = await fakeHf(t, { ...at(what, 2 * LOOKUP_METADATA_MAX_BYTES), raw: { chunked, state } });
            await assert.rejects(() => ask(hf, { repo: REPO }), (error) => error.code === 'lookup_failed' && /larger than 8388608 bytes/.test(error.message), `${what} ${chunked ? 'chunked' : 'with a length'}`);
            // Either way the client stops reading and closes the connection: the server is left with most of the body unsent.
            const until = Date.now() + 3000;
            while (state.closed !== true && Date.now() < until) await new Promise((resolve) => setTimeout(resolve, 10));
            assert.equal(state.closed, true, `${what}: the connection was closed by the client`);
            assert.ok(state.sent < 2 * LOOKUP_METADATA_MAX_BYTES, `${what}: the server sent ${state.sent} of ${2 * LOOKUP_METADATA_MAX_BYTES} bytes`);
        }
    }
    // A declared length beyond the cap is refused on the headers alone: the body is never waited for (this server sends a few bytes and then stalls).
    const stalling = await listen(t, (req, res) => { res.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': 2 * LOOKUP_METADATA_MAX_BYTES }); res.write('{"pad":"'); });
    const began = Date.now();
    await assert.rejects(() => lookupHuggingFaceModel({ repo: REPO }, { baseUrl: stalling.origin, timeoutMs: 5000 }), (error) => error.code === 'lookup_failed' && /larger than 8388608 bytes/.test(error.message));
    assert.ok(Date.now() - began < 2000, `refused after ${Date.now() - began} ms, not at the 5 s timeout`);
    // The cap is an option; a body of exactly the cap is read, one byte more is not.
    const exact = (what, cap) => at(what, cap - Buffer.byteLength(padded((pad) => (what === 'tree' ? [{ type: 'file', path: 'a-Q4_K_M.gguf', pad }] : what === 'info' ? { gated: false, pad } : { sha: COMMIT, pad }), 0)));
    for (const what of ['info', 'revision', 'tree']) {
        const ok = await fakeHf(t, exact(what, 4096));
        const result = await ask(ok, { repo: REPO }, { metadataMaxBytes: 4096 });
        assert.equal(result.commit, COMMIT, what);
        const over = await fakeHf(t, at(what, 4096));
        await assert.rejects(() => ask(over, { repo: REPO }, { metadataMaxBytes: 4096 }), codeOf('lookup_failed'), what);
    }
    // fetchJson without a cap is what downloads use and still reads any size; with one it refuses too.
    const big = await listen(t, (req, res) => sendJson(res, 200, { pad: 'x'.repeat(9 * MIB) }));
    assert.equal((await fetchJson(`${big.origin}/x`, { fetchImpl: fetch, timeoutMs: 20_000 })).json.pad.length, 9 * MIB);
    await assert.rejects(() => fetchJson(`${big.origin}/x`, { fetchImpl: fetch, maxBytes: MIB }), (error) => error instanceof DownloadError && error.details.reason === 'too-large');
    const chunkedBig = await listen(t, (req, res) => sendRaw(res, JSON.stringify({ pad: 'x'.repeat(9 * MIB) }), { chunked: true }));
    await assert.rejects(() => fetchJson(`${chunkedBig.origin}/x`, { fetchImpl: fetch, maxBytes: MIB }), (error) => error.details?.reason === 'too-large');
    assert.equal((await fetchJson(`${chunkedBig.origin}/x`, { fetchImpl: fetch, timeoutMs: 20_000 })).json.pad.length, 9 * MIB);
});

test('without a token a 401 or 403 says the repository may not exist or needs access; with one it says access was refused', async (t) => {
    const tree = { '': [[lfs('a-Q4_K_M.gguf', MIB, sha('a'))]] };
    const cases = [
        ['the model', { infoStatus: 401 }, {}],
        ['the revision', { revisionStatus: 403, tree }, {}],
        ['the listing', { treeStatus: 401, tree }, {}],
        ['the file', { tree, files: { 'a-Q4_K_M.gguf': { status: 401 } } }, { file: 'a-Q4_K_M.gguf' }],
    ];
    for (const [what, repo, extra] of cases) {
        const hf = await fakeHf(t, repo);
        await assert.rejects(() => ask(hf, { repo: REPO, ...extra }), (error) => {
            assert.equal(error.code, 'access_denied', what);
            assert.match(error.message, /^Hugging Face did not find .*, or it needs access \(HTTP 40[13]\)\. For a gated or private repository, set a token with `ploinky var HF_TOKEN <token>` and restart local-llm\.$/, what);
            assert.doesNotMatch(error.message, /refused/, what);
            return true;
        });
        await assert.rejects(() => ask(hf, { repo: REPO, ...extra }, { token: TOKEN }), (error) => {
            assert.equal(error.code, 'access_denied', what);
            assert.match(error.message, /^Hugging Face refused access to .* \(HTTP 40[13]\)\. For a gated or private repository, set a token with `ploinky var HF_TOKEN <token>`/, what);
            assert.doesNotMatch(error.message, /did not find/, what);
            assert.doesNotMatch(error.message, new RegExp(TOKEN));
            return true;
        });
    }
    // A 404 is still plainly not found, token or not.
    const missing = await fakeHf(t, { infoStatus: 404 });
    for (const token of ['', TOKEN]) await assert.rejects(() => ask(missing, { repo: REPO }, { token }), (error) => error.code === 'not_found' && !/access/.test(error.message));
});

test('bidirectional control characters are dropped from a licence before it is shown or stored', async (t) => {
    const licenceOf = async (info) => (await ask(await fakeHf(t, { info, tree: { '': [[]] } }), { repo: REPO })).license;
    assert.equal(await licenceOf({ cardData: { license: 'mit\u202Egpl\u202C' } }), 'mitgpl');
    assert.equal(await licenceOf({ cardData: { license: '\u2066apache-2.0\u2069' } }), 'apache-2.0');
    assert.equal(await licenceOf({ cardData: { license: '\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069' }, tags: ['license:mit'] }), 'mit', 'nothing is left of the first, so the tag is used');
    assert.equal(await licenceOf({ cardData: { license: 'apache-2.0' } }), 'apache-2.0');
});

// ------------------------------------------------------------------ the controller

const GIB = 1024 * MIB;
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

async function until(predicate, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail('condition not reached');
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

// A repository with one LFS GGUF file of 400 MiB whose header is the Qwen one; Add pins it through this listing.
const modelRepo = () => ({ tree: { '': [[lfs('small-Q4_K_M.gguf', 400 * MIB, sha('a'))]] }, files: { 'small-Q4_K_M.gguf': { body: pad(QWEN_HEADER) } } });
const entryFor = (id, extra = {}) => ({
    id,
    sources: { gguf: { type: 'huggingface', repo: REPO, file: 'small-Q4_K_M.gguf', revision: 'main' } },
    ...extra,
});
const runOn = (h, modelId, requestId) => h.controller.run({ modelId, runnerId: 'llama.cpp', requestId });
const readyOrFailed = (h) => until(() => ['ready', 'error'].includes(h.controller.state.deployment?.phase));

test('the header is read again from the verified file and the last admission uses it', async (t) => {
    const hf = await fakeHf(t, modelRepo());
    // The file on disk says 12288 KV bytes per token; the entry was added with 1000 (what an older or different file said).
    const h = harness(t, { hf, weights: pad(QWEN_HEADER, 65536) });
    await h.controller.addModel(entryFor('understated', { sizingSource: 'gguf-header', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 1000 } }));
    // Before the download the offer says its sizing is from a header that is still to be checked.
    const offered = (await h.controller.overview()).models.find((model) => model.id === 'understated').runners['llama.cpp'].admission;
    assert.ok(offered.warnings.some((warning) => /^Sized with memory\.kvBytesPerToken from the GGUF header read when the model was added; the header is read again from the verified file/.test(warning)),
        offered.warnings.join(' | '));
    await runOn(h, 'understated', 'request-0001');
    await until(() => h.controller.state.deployment?.phase === 'ready');
    const { deployment } = h.controller.state;
    const ctx = deployment.params.ctxSize;
    assert.ok(ctx > 0);
    // The deployment's admission, which the last check wrote just before the runner started, is sized from the file.
    assert.equal(deployment.admission.estimate.kvBytes, 12288 * ctx, 'the estimate uses the verified 12288, not the stored 1000');
    assert.equal(deployment.admission.status, 'ok');
    assert.deepEqual(deployment.admission.estimate.defaulted, []);
    assert.ok(deployment.admission.warnings.some((warning) => /^Sized with memory\.kvBytesPerToken from the GGUF header of the downloaded file, which was checked against its sha256/.test(warning)),
        'the warning says the values are the verified file\'s own');
    assert.equal(h.started.length, 1);
    // The difference is named, with both values, once.
    const lines = h.logLines('differs from the sizing stored');
    assert.equal(lines.length, 1);
    assert.match(lines[0], /understated: the GGUF header of the verified file differs from the sizing stored when the model was added \(kvBytesPerToken 1000 -> 12288\); the verified values are used/);
    // The entry stored in the registry is not rewritten: the next Run reads the file again.
    assert.equal(h.stored().registry[0].memory.kvBytesPerToken, 1000);
    assert.equal(h.stored().registry[0].sizingSource, 'gguf-header');

    // A file whose header makes the model not fit refuses the Run with the verified numbers, and no runner starts.
    const big = ggufBytes({ kv: modelPairs({ layers: 24, heads: 14, kvHeads: 2, embedding: 896, context: 32768, extra: [['qwen2.attention.key_length', T.u32, 65536], ['qwen2.attention.value_length', T.u32, 65536]] }) }).bytes;
    const refused = harness(t, { hf, weights: pad(big, 65536) });
    await refused.controller.addModel(entryFor('too-big', { sizingSource: 'gguf-header', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 1000 } }));
    await runOn(refused, 'too-big', 'request-0002');
    await readyOrFailed(refused);
    assert.equal(refused.controller.state.deployment.phase, 'error');
    assert.match(refused.controller.state.deployment.error, /Needs about .* of the .* of memory/);
    assert.equal(refused.started.length, 0, 'the runner is never started');
    assert.equal(refused.logLines('differs from the sizing stored').length, 1);
});

test('a header identical to the stored sizing is not reported, and typed or seed sizing is not read again', async (t) => {
    const hf = await fakeHf(t, modelRepo());
    const same = harness(t, { hf, weights: pad(QWEN_HEADER, 65536) });
    await same.controller.addModel(entryFor('matching', { sizingSource: 'gguf-header', architecture: 'dense', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 } }));
    await runOn(same, 'matching', 'request-0003');
    await until(() => same.controller.state.deployment?.phase === 'ready');
    assert.deepEqual(same.logLines('differs from the sizing stored'), []);
    // Sizing typed by hand (no sizingSource, or manual) is the admin's: the file is not read for it, so a
    // file that is not even a GGUF does not matter.
    for (const extra of [{}, { sizingSource: 'manual' }]) {
        const typed = harness(t, { hf, weights: Buffer.from('not a gguf file at all') });
        const id = `typed-${Object.keys(extra).length}`;
        await typed.controller.addModel(entryFor(id, { ...extra, contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 1000 } }));
        await runOn(typed, id, 'request-0004');
        await until(() => typed.controller.state.deployment?.phase === 'ready');
        assert.equal(typed.controller.state.deployment.admission.estimate.kvBytes, 1000 * typed.controller.state.deployment.params.ctxSize, id);
        assert.deepEqual(typed.logLines('differs from the sizing stored'), [], id);
    }
});

test('a downloaded file whose header cannot be read is not sized, and fails the Run with invalid_gguf', async (t) => {
    const hf = await fakeHf(t, modelRepo());
    for (const weights of [Buffer.from('<html>this is not a GGUF</html>'), QWEN_HEADER.subarray(0, 100), Buffer.alloc(0)]) {
        const h = harness(t, { hf, weights });
        await h.controller.addModel(entryFor('unreadable', { sizingSource: 'gguf-header', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 } }));
        await runOn(h, 'unreadable', 'request-0005');
        await readyOrFailed(h);
        assert.equal(h.controller.state.deployment.phase, 'error');
        assert.match(h.controller.state.deployment.error, /weights\.gguf has no readable GGUF header .* cannot be checked/);
        assert.equal(h.started.length, 0);
    }
});

test('a looked-up entry is added with its sizing source, and added twice is duplicate_model', async (t) => {
    const hf = await fakeHf(t, modelRepo());
    const h = harness(t, { hf });
    const tree = () => fs.readdirSync(h.dir, { recursive: true }).sort();
    const before = tree();
    const looked = await h.controller.lookupModel({ repo: REPO, file: 'small-Q4_K_M.gguf' });
    assert.deepEqual(looked.files.map((row) => row.file), ['small-Q4_K_M.gguf']);
    assert.deepEqual(tree(), before, 'a lookup writes nothing under /data');
    assert.deepEqual(h.stored().registry, [], 'and records no model');
    assert.ok(hf.requests.every((request) => request.authorization === `Bearer ${TOKEN}`), 'the controller sends its HF_TOKEN to the lookup');
    const entry = entryFor('looked-up', {
        license: looked.license, sizingSource: 'gguf-header', architecture: looked.sizing.architecture,
        contextLength: looked.sizing.contextLength, memory: looked.sizing.memory,
    });
    const added = await h.controller.addModel(entry);
    assert.equal(added.model.sizingSource, 'gguf-header');
    assert.equal(added.model.sources.gguf.commit, COMMIT);
    assert.equal(added.model.sources.gguf.size, 400 * MIB);
    assert.equal(h.stored().registry[0].sizingSource, 'gguf-header');
    await assert.rejects(() => h.controller.addModel(entry), codeOf('duplicate_model'));
    assert.equal(h.stored().registry.length, 1);
    // The field is checked like the rest of an entry.
    for (const [extra, what] of [
        [{ sizingSource: 'guess' }, 'an unknown value'],
        [{ sizingSource: 'config.json' }, 'config.json on a GGUF-only entry'],
        [{ sizingSource: 7 }, 'a number'],
    ]) {
        await assert.rejects(() => h.controller.addModel(entryFor('bad-source', extra)), (error) => error.code === 'invalid_model' && error.details?.field === 'sizingSource', what);
    }
    await assert.rejects(() => h.controller.addModel({ id: 'tag-only', sources: { ollama: { type: 'ollama', tag: 'qwen2.5:0.5b' } }, sizingSource: 'gguf-header' }), codeOf('invalid_model'));
    assert.equal(validateModel({ id: 'snap', sources: { hf: { type: 'hf-snapshot', repo: REPO } }, sizingSource: 'config.json' }).sizingSource, 'config.json');
    assert.equal(validateModel(entryFor('none')).sizingSource, undefined);
    assert.equal('sizingSource' in validateModel(entryFor('none')), false, 'an entry without one keeps today\'s shape');
    // A repository that does not exist is not_found through the controller too (nothing was recorded).
    const missing = harness(t, { hf: await fakeHf(t, { infoStatus: 404 }) });
    await assert.rejects(() => missing.controller.lookupModel({ repo: 'acme/absent' }), codeOf('not_found'));
    await assert.rejects(() => missing.controller.lookupModel({ repo: '../bad' }), codeOf('invalid_request'));
});

test('a lookup during a download does not hold up Stop or Run, and a drain ends the lookup', async (t) => {
    // This controller's Hugging Face accepts every request and never answers, so a lookup on it stays pending.
    const hung = await listen(t, () => {});
    const gate = deferred();
    const seed = validateModel({
        id: 'seeded', profiles: ['cpu'], contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 },
        sources: { gguf: { type: 'huggingface', repo: REPO, file: 'small-Q4_K_M.gguf', revision: 'main', commit: COMMIT, size: 400 * MIB, sha256: sha('a') } },
    }, { seed: true });
    const h = harness(t, {
        hf: hung,
        seed: [seed],
        download: async ({ signal }) => {
            await new Promise((resolve, reject) => {
                signal?.addEventListener('abort', () => reject(new DownloadError('ABORTED', 'stopped', { retryable: true })), { once: true });
                gate.promise.then(resolve);
            });
        },
    });
    await runOn(h, 'seeded', 'request-0006');
    await until(() => h.controller.state.deployment?.phase === 'downloading');
    const lookup = h.controller.lookupModel({ repo: REPO });
    const outcome = lookup.then(() => 'resolved', (error) => error.code);
    await until(() => hung.requests.length === 1);
    // Stop answers at once, well inside the time a drain gives a running command, with the lookup still pending.
    const began = Date.now();
    await h.controller.stop();
    assert.ok(Date.now() - began < DRAIN_QUEUE_WAIT_MS, `Stop took ${Date.now() - began} ms`);
    assert.ok(['idle', 'paused'].includes(h.controller.state.deployment.phase), h.controller.state.deployment.phase);
    assert.equal(await Promise.race([outcome, new Promise((resolve) => setTimeout(() => resolve('still pending'), 50))]), 'still pending');
    // A Run and a Cancel are not held up either.
    const runBegan = Date.now();
    await runOn(h, 'seeded', 'request-0007');
    assert.ok(Date.now() - runBegan < DRAIN_QUEUE_WAIT_MS, `Run took ${Date.now() - runBegan} ms`);
    await until(() => h.controller.state.deployment?.phase === 'downloading');
    const cancelBegan = Date.now();
    await h.controller.cancelDownload();
    assert.ok(Date.now() - cancelBegan < DRAIN_QUEUE_WAIT_MS, `Cancel took ${Date.now() - cancelBegan} ms`);
    assert.equal(await Promise.race([outcome, new Promise((resolve) => setTimeout(() => resolve('still pending'), 20))]), 'still pending');
    // The drain ends the lookup with shutting_down, and a lookup after the drain is refused.
    const drain = h.controller.drain();
    assert.equal(await outcome, 'shutting_down');
    await drain;
    await assert.rejects(() => h.controller.lookupModel({ repo: REPO }), codeOf('shutting_down'));
    gate.resolve();
});

test('the verified re-read sizes a hyphenated architecture, and one it cannot use keeps the stored sizing with a warning instead of failing the Run', async (t) => {
    const hf = await fakeHf(t, modelRepo());
    // gpt-oss: the file says 49,152 KV bytes per token under an architecture named with a hyphen; the entry said 1,000.
    const oss = harness(t, { hf, weights: pad(GPT_OSS_HEADER, 65536) });
    await oss.controller.addModel(entryFor('oss-understated', { sizingSource: 'gguf-header', architecture: 'moe', contextLength: 131072, memory: { layers: 24, kvBytesPerToken: 1000 } }));
    await runOn(oss, 'oss-understated', 'request-oss-0001');
    await until(() => oss.controller.state.deployment?.phase === 'ready');
    const { deployment } = oss.controller.state;
    assert.equal(deployment.admission.estimate.kvBytes, 49152 * deployment.params.ctxSize, 'the hyphenated architecture was read and its KV size used');
    assert.equal(oss.logLines('differs from the sizing stored').length, 1);
    assert.match(oss.logLines('differs from the sizing stored')[0], /\(kvBytesPerToken 1000 -> 49152\)/);
    assert.ok(deployment.admission.warnings.some((warning) => /GGUF header of the downloaded file, which was checked against its sha256/.test(warning)));
    // An architecture this reader cannot use (none, a name with spaces, a dotted one): the Run goes on with what the entry says.
    const unusable = {
        'no architecture': ggufBytes({ kv: [['general.name', T.string, 'x'], ['qwen2.block_count', T.u32, 24]] }).bytes,
        'a name with a space': ggufBytes({ kv: [['general.architecture', T.string, 'Bad Arch!'], ['Bad Arch!.block_count', T.u32, 24]] }).bytes,
        'a dotted name': ggufBytes({ kv: [['general.architecture', T.string, 'a.b'], ['a.b.block_count', T.u32, 24]] }).bytes,
    };
    for (const [what, header] of Object.entries(unusable)) {
        const h = harness(t, { hf, weights: pad(header, 65536) });
        await h.controller.addModel(entryFor('keeps-stored', { sizingSource: 'gguf-header', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 1000 } }));
        await runOn(h, 'keeps-stored', 'request-keep-0001');
        await readyOrFailed(h);
        const run = h.controller.state.deployment;
        assert.equal(run.phase, 'ready', `${what}: the Run is not failed (${run.error})`);
        assert.equal(h.started.length, 1, what);
        assert.equal(run.admission.estimate.kvBytes, 1000 * run.params.ctxSize, `${what}: the stored KV size stands`);
        assert.equal(h.logLines('differs from the sizing stored').length, 0, what);
        const lines = h.logLines('cannot be used to check the sizing stored');
        assert.equal(lines.length, 1, what);
        assert.match(lines[0], /keeps-stored: the GGUF header of the verified file cannot be used to check the sizing stored when the model was added \(general\.architecture is (?:missing|not a name this agent can read.*)\); the stored values stand/, what);
        assert.ok(run.admission.warnings.some((warning) => /^Sized with memory\.kvBytesPerToken from the GGUF header read when the model was added; the downloaded file's header could not be used to check them \(general\.architecture is .*\), so they stand unchecked: an understated value/.test(warning)),
            `${what}: ${run.admission.warnings.join(' | ')}`);
        assert.equal(h.stored().registry[0].memory.kvBytesPerToken, 1000);
    }
    // An entry with no stored sizing is left with the estimate's defaults, which it names, and runs.
    const bare = harness(t, { hf, weights: pad(unusable['no architecture'], 65536) });
    await bare.controller.addModel(entryFor('keeps-defaults', { sizingSource: 'gguf-header' }));
    await runOn(bare, 'keeps-defaults', 'request-bare-0001');
    await readyOrFailed(bare);
    assert.equal(bare.controller.state.deployment.phase, 'ready');
    assert.deepEqual(bare.controller.state.deployment.admission.estimate.defaulted, ['memory.kvBytesPerToken', 'contextLength']);
    assert.equal(bare.logLines('cannot be used to check the sizing stored').length, 1);
    // A header that is malformed as GGUF still fails the Run with invalid_gguf, as before.
    const broken = harness(t, { hf, weights: Buffer.from('<html>not a GGUF file</html>') });
    await broken.controller.addModel(entryFor('malformed', { sizingSource: 'gguf-header', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 1000 } }));
    await runOn(broken, 'malformed', 'request-bad-0001');
    await readyOrFailed(broken);
    assert.equal(broken.controller.state.deployment.phase, 'error');
    assert.match(broken.controller.state.deployment.error, /has no readable GGUF header/);
});

test('when the verified file changes the training context, the parameters are worked out again from what the Run asked for', async (t) => {
    const hf = await fakeHf(t, modelRepo());
    const longEntry = (id, extra = {}) => entryFor(id, { sizingSource: 'gguf-header', contextLength: 32768, memory: { layers: 24, kvBytesPerToken: 12288 }, ...extra });
    const ctxArg = (process) => process.args[process.args.indexOf('--ctx-size') + 1];
    for (const dedicated of [false, true]) {
        const profile = dedicated ? 'dedicated' : 'cpu';
        // A default context: admitted against the stored 32768, then lowered to the 2048 the file says. Nothing else about the Run changes.
        const short = harness(t, { hf, weights: headerOf({ context: 2048 }), dedicated });
        await short.controller.addModel(longEntry('trained-short'));
        const accepted = await runOn(short, 'trained-short', `request-default-${profile}`);
        assert.ok(accepted.deployment.params.ctxSize > 2048, `${profile}: admitted with ${accepted.deployment.params.ctxSize}`);
        await until(() => short.controller.state.deployment?.phase === 'ready');
        const { deployment } = short.controller.state;
        assert.equal(deployment.profile, profile);
        assert.equal(deployment.params.ctxSize, 2048, profile);
        assert.deepEqual({ ...deployment.params, ctxSize: 0 }, { ...accepted.deployment.params, ctxSize: 0 }, `${profile}: only the context moved`);
        // The estimate, the recorded parameters, the saved ones and the launch all carry the one value.
        assert.equal(deployment.admission.estimate.kvBytes, 12288 * 2048, profile);
        assert.equal(ctxArg(short.started[0]), '2048', profile);
        assert.equal(short.stored().params['trained-short|llama.cpp'].ctxSize, 2048, profile);
        assert.match(short.logLines('differs from the sizing stored')[0], /contextLength 32768 -> 2048/);
        // An explicit context above what the file says is refused with the parameter error, after the download and before any runner starts.
        const explicit = harness(t, { hf, weights: headerOf({ context: 2048 }), dedicated });
        await explicit.controller.addModel(longEntry('trained-short'));
        const asked = await explicit.controller.run({ modelId: 'trained-short', runnerId: 'llama.cpp', requestId: `request-explicit-${profile}`, params: { ctxSize: 4096 } });
        assert.equal(asked.accepted, true, 'admitted against the stored training context');
        await readyOrFailed(explicit);
        const failed = explicit.controller.state.deployment;
        assert.equal(failed.phase, 'error', profile);
        assert.match(failed.error, /The downloaded file's header gives a training context of 2048 tokens\. Invalid parameter ctxSize: must be <= 2048/, profile);
        assert.equal(explicit.started.length, 0, `${profile}: the runner never started`);
    }
    // The explicit value the request named, at the verified context, is fine; above it is refused. Both on cpu.
    const explicitCtx = (ctxSize) => ({ modelId: 'trained-short', runnerId: 'llama.cpp', requestId: `request-ctx-${ctxSize}`, params: { ctxSize } });
    const atBound = harness(t, { hf, weights: headerOf({ context: 2048 }) });
    await atBound.controller.addModel(longEntry('trained-short'));
    await atBound.controller.run(explicitCtx(2048));
    await until(() => atBound.controller.state.deployment?.phase === 'ready');
    assert.equal(ctxArg(atBound.started[0]), '2048');
    const above = harness(t, { hf, weights: headerOf({ context: 2048 }) });
    await above.controller.addModel(longEntry('trained-short'));
    await above.controller.run(explicitCtx(2049));
    await readyOrFailed(above);
    assert.equal(above.controller.state.deployment.phase, 'error');
    assert.equal(above.started.length, 0);
    // A context the file raises: the default follows it up (the stored 2048 had capped it).
    const raised = harness(t, { hf, weights: headerOf({ context: 32768 }) });
    await raised.controller.addModel(longEntry('trained-short', { contextLength: 2048 }));
    const first = await runOn(raised, 'trained-short', 'request-raised-0001');
    assert.equal(first.deployment.params.ctxSize, 2048);
    await until(() => raised.controller.state.deployment?.phase === 'ready');
    assert.equal(raised.controller.state.deployment.params.ctxSize, 4096, 'the cpu default, no longer capped');
    assert.equal(ctxArg(raised.started[0]), '4096');
    // A context the file leaves as it is: the parameters are the ones admitted, untouched.
    const same = harness(t, { hf, weights: headerOf({ context: 32768 }) });
    await same.controller.addModel(longEntry('trained-short'));
    const untouched = await runOn(same, 'trained-short', 'request-same-0001');
    await until(() => same.controller.state.deployment?.phase === 'ready');
    assert.deepEqual(same.controller.state.deployment.params, untouched.deployment.params);
    // Parameters an earlier Run saved are not taken for this Run's own choice: a Run with no parameters that picks up a saved
    // explicit 4096 is admitted with it, and then the file's 2048 replaces the saved value with the default, not an error.
    const saved = harness(t, { hf, weights: headerOf({ context: 32768 }) });
    await saved.controller.addModel(longEntry('trained-short'));
    await saved.controller.run(explicitCtx(4096));
    await until(() => saved.controller.state.deployment?.phase === 'ready');
    await saved.controller.stop();
    fs.writeFileSync(saved.file, headerOf({ context: 2048 }));
    const later = await saved.controller.run({ modelId: 'trained-short', runnerId: 'llama.cpp', requestId: 'request-later-0001' });
    assert.equal(later.deployment.params.ctxSize, 4096, 'the saved value, admitted');
    await until(() => saved.controller.state.deployment?.requestId === 'request-later-0001' && ['ready', 'error'].includes(saved.controller.state.deployment.phase));
    assert.equal(saved.controller.state.deployment.phase, 'ready');
    assert.equal(saved.controller.state.deployment.params.ctxSize, 2048);
    // Saved parameters that are still valid under the verified context carry over untouched: they are an earlier Run's choice, and
    // working the parameters out again does not turn them back into defaults.
    const kept = harness(t, { hf, weights: headerOf({ context: 32768 }) });
    await kept.controller.addModel(longEntry('trained-short'));
    await kept.controller.run({ modelId: 'trained-short', runnerId: 'llama.cpp', requestId: 'request-kept-0001', params: { ctxSize: 1024, threads: 2 } });
    await until(() => kept.controller.state.deployment?.phase === 'ready');
    await kept.controller.stop();
    fs.writeFileSync(kept.file, headerOf({ context: 2048 }));
    const again = await kept.controller.run({ modelId: 'trained-short', runnerId: 'llama.cpp', requestId: 'request-kept-0002' });
    assert.deepEqual([again.deployment.params.ctxSize, again.deployment.params.threads], [1024, 2]);
    await until(() => kept.controller.state.deployment?.requestId === 'request-kept-0002' && ['ready', 'error'].includes(kept.controller.state.deployment.phase));
    assert.equal(kept.controller.state.deployment.phase, 'ready');
    assert.deepEqual([kept.controller.state.deployment.params.ctxSize, kept.controller.state.deployment.params.threads], [1024, 2]);
    assert.equal(ctxArg(kept.started.at(-1)), '1024');
});

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

test('at most two lookups run at once; the third is busy until one of them finishes', async (t) => {
    const held = [];
    const hung = await listen(t, (req, res) => { held.push(res); });
    const h = harness(t, { hf: hung });
    const finished = [];
    const start = () => h.controller.lookupModel({ repo: REPO }).then((value) => finished.push(['ok', value]), (error) => finished.push([error.code]));
    const first = start();
    const second = start();
    await until(() => held.length === 2);
    // The third is refused at once, says why, and sends nothing.
    await assert.rejects(() => h.controller.lookupModel({ repo: REPO }), (error) => error.code === 'busy' && /2 lookups are already running/.test(error.message));
    assert.equal(held.length, 2, 'the refused lookup sent no request');
    assert.deepEqual(finished, []);
    // One ends (the repository is not there), and its place is free again.
    held[0].writeHead(404);
    held[0].end();
    await until(() => finished.length === 1);
    assert.deepEqual(finished, [['not_found']]);
    const third = start();
    await until(() => held.length === 3);
    await assert.rejects(() => h.controller.lookupModel({ repo: REPO }), codeOf('busy'));
    // The drain ends the two that are left.
    const drain = h.controller.drain();
    await Promise.all([first, second, third]);
    await drain;
    assert.deepEqual(finished.map(([code]) => code).sort(), ['not_found', 'shutting_down', 'shutting_down']);
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
