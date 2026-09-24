// downloadFile: one pinned file from a fixed URL (a runner's lock entry),
// with the same resume, size, disk and sha256 rules as model weights.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { DownloadError, downloadFile, inspectFile } from '../src/controller/downloader.mjs';

const PAYLOAD = crypto.randomBytes(2 * 1024 * 1024 + 777);
const SHA256 = crypto.createHash('sha256').update(PAYLOAD).digest('hex');
const noSleep = async () => {};

function tempDir(t) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-file-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

// A server for one file; `cutAfter` drops the first connection after that many bytes.
async function serve(t, { body = PAYLOAD, cutAfter = null } = {}) {
    const requests = [];
    const server = http.createServer((req, res) => {
        requests.push({ url: req.url, range: req.headers.range ?? null });
        const range = /^bytes=(\d+)-$/.exec(req.headers.range ?? '');
        const start = range ? Number(range[1]) : 0;
        const headers = { 'Content-Length': body.length - start };
        if (range) headers['Content-Range'] = `bytes ${start}-${body.length - 1}/${body.length}`;
        res.writeHead(range ? 206 : 200, headers);
        if (cutAfter !== null && requests.length === 1) {
            res.write(body.subarray(start, start + cutAfter), () => res.destroy());
            return;
        }
        res.end(body.subarray(start));
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => server.close());
    return { url: `http://127.0.0.1:${server.address().port}/files/runner.whl`, requests };
}

const bigDisk = async () => ({ bavail: 1e12, bsize: 1 });

test('a pinned file downloads to its target and verifies', async (t) => {
    const dir = tempDir(t);
    const { url, requests } = await serve(t);
    const target = path.join(dir, 'runner.whl');
    const result = await downloadFile({ url, size: PAYLOAD.length, sha256: SHA256, target, statfs: bigDisk, allowHttp: true });
    assert.equal(result.bytesTransferred, PAYLOAD.length);
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex'), SHA256);
    assert.equal(fs.existsSync(`${target}.partial`), false);
    assert.deepEqual(await inspectFile({ url, size: PAYLOAD.length, sha256: SHA256, target }), { state: 'complete', bytes: PAYLOAD.length });
    // A second call finds it complete and makes no request.
    const again = await downloadFile({ url, size: PAYLOAD.length, sha256: SHA256, target, statfs: bigDisk, allowHttp: true });
    assert.equal(again.bytesTransferred, 0);
    assert.equal(requests.length, 1);
});

test('a dropped connection resumes with Range from the partial', async (t) => {
    const dir = tempDir(t);
    const { url, requests } = await serve(t, { cutAfter: 1024 * 1024 });
    const target = path.join(dir, 'runner.whl');
    await downloadFile({ url, size: PAYLOAD.length, sha256: SHA256, target, statfs: bigDisk, allowHttp: true, sleep: noSleep });
    assert.equal(requests[0].range, null);
    assert.match(requests[1].range, /^bytes=\d+-$/);
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex'), SHA256);
});

test('bytes that do not match the pinned sha256 are refused and removed', async (t) => {
    const dir = tempDir(t);
    const other = crypto.randomBytes(PAYLOAD.length);
    const { url } = await serve(t, { body: other });
    const target = path.join(dir, 'runner.whl');
    await assert.rejects(
        () => downloadFile({ url, size: PAYLOAD.length, sha256: SHA256, target, statfs: bigDisk, allowHttp: true }),
        (error) => error instanceof DownloadError && error.code === 'SHA256_MISMATCH',
    );
    assert.equal(fs.existsSync(target), false);
    assert.equal(fs.existsSync(`${target}.partial`), false);
});

test('a partial recorded for another URL or hash is discarded, not resumed', async (t) => {
    const dir = tempDir(t);
    const { url, requests } = await serve(t);
    const target = path.join(dir, 'runner.whl');
    fs.writeFileSync(`${target}.partial`, PAYLOAD.subarray(0, 4096));
    fs.writeFileSync(`${target}.partial.json`, JSON.stringify({ url: `${url}?old`, size: PAYLOAD.length, sha256: SHA256 }));
    await downloadFile({ url, size: PAYLOAD.length, sha256: SHA256, target, statfs: bigDisk, allowHttp: true });
    assert.equal(requests[0].range, null);
});

test('only https URLs are fetched, and a full disk refuses before any request', async (t) => {
    const dir = tempDir(t);
    const { url, requests } = await serve(t);
    const target = path.join(dir, 'runner.whl');
    await assert.rejects(() => downloadFile({ url, size: PAYLOAD.length, sha256: SHA256, target, statfs: bigDisk }),
        (error) => error instanceof DownloadError && error.code === 'INVALID_SOURCE');
    await assert.rejects(
        () => downloadFile({ url, size: PAYLOAD.length, sha256: SHA256, target, statfs: async () => ({ bavail: 10, bsize: 1 }), allowHttp: true }),
        (error) => error instanceof DownloadError && error.code === 'INSUFFICIENT_SPACE',
    );
    assert.equal(requests.length, 0);
});
