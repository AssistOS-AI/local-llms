// C12 (DS002): a pinned model file already in the workspace's shared
// directory is adopted, by hard link or copy and then verified, instead of
// downloaded; only the pinned identity decides a match.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { artifactPaths, downloadArtifact, removeArtifact } from '../src/controller/downloader.mjs';

const SIZE = 256 * 1024 + 7;
const PAYLOAD = crypto.randomBytes(SIZE);
const SHA256 = crypto.createHash('sha256').update(PAYLOAD).digest('hex');
const ARTIFACT = Object.freeze({
    repo: 'org/model-GGUF', file: 'model-Q8_0.gguf', revision: 'main', commit: 'c'.repeat(40), size: SIZE, sha256: SHA256,
});

function setup(t) {
    const base = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-shared-'));
    t.after(() => fs.rmSync(base, { recursive: true, force: true }));
    const root = path.join(base, 'data', 'models', 'gguf');
    const shared = path.join(base, 'shared', 'models');
    fs.mkdirSync(shared, { recursive: true });
    const fetches = [];
    const fetchImpl = async (url) => {
        fetches.push(String(url));
        return new Response(PAYLOAD, { status: 200, headers: { 'content-length': String(SIZE) } });
    };
    const get = (options = {}) => downloadArtifact({
        artifact: ARTIFACT, root, adoptFrom: [shared], fetchImpl, baseUrl: 'http://hf.invalid', sleep: async () => {}, ...options,
    });
    return { base, root, shared, fetches, get, paths: artifactPaths({ root, artifact: ARTIFACT }) };
}

const meta = (paths) => JSON.parse(fs.readFileSync(paths.meta, 'utf8'));

test('a verified file in /shared/models is hard-linked and verified instead of downloaded; Delete frees only a last link', async (t) => {
    const s = setup(t);
    // Any name, any depth: only the size and sha256 decide.
    const source = path.join(s.shared, 'someone', 'else', 'renamed.bin');
    fs.mkdirSync(path.dirname(source), { recursive: true });
    fs.writeFileSync(source, PAYLOAD);
    const result = await s.get();
    assert.deepEqual(result.adopted, { source, method: 'link' });
    assert.equal(s.fetches.length, 0, 'nothing downloaded');
    assert.equal(fs.statSync(s.paths.file).ino, fs.statSync(source).ino);
    assert.deepEqual(meta(s.paths).adopted, { from: source, method: 'link' });
    const stat = fs.lstatSync(s.paths.file);
    assert.deepEqual(meta(s.paths).stat, { dev: stat.dev, ino: stat.ino, size: stat.size, mtimeMs: stat.mtimeMs, ctimeMs: stat.ctimeMs });
    // A second Run finds it complete and unchanged: no hashing, no download.
    assert.equal((await s.get()).reverified, undefined);
    // Delete removes only the agent's link; the shared copy stays and nothing is freed.
    // (Only the agent's small verification record counts as freed.)
    assert.ok(await removeArtifact({ root: s.root, artifact: ARTIFACT }) < 4096);
    assert.ok(fs.existsSync(source));
    assert.equal(fs.existsSync(s.paths.file), false);
});

test('across filesystems the file is copied, then verified', async (t) => {
    const s = setup(t);
    const source = path.join(s.shared, 'model.gguf');
    fs.writeFileSync(source, PAYLOAD);
    const fsApi = { ...fs, promises: { ...fs.promises, link: async () => { throw Object.assign(new Error('EXDEV'), { code: 'EXDEV' }); } } };
    const result = await s.get({ fsApi });
    assert.deepEqual(result.adopted, { source, method: 'copy' });
    assert.equal(s.fetches.length, 0);
    assert.notEqual(fs.statSync(s.paths.file).ino, fs.statSync(source).ino);
    // The agent's own copy frees its size on Delete.
    const freed = await removeArtifact({ root: s.root, artifact: ARTIFACT });
    assert.ok(freed >= SIZE && freed < SIZE + 4096, String(freed));
});

test('a file of the right size but the wrong bytes is discarded, and the pinned file is downloaded', async (t) => {
    const s = setup(t);
    const wrong = Buffer.from(PAYLOAD);
    wrong[1000] ^= 0xff;
    const source = path.join(s.shared, 'model-Q8_0.gguf');
    fs.writeFileSync(source, wrong);
    const result = await s.get();
    assert.equal(result.adopted, undefined);
    assert.equal(s.fetches.length, 1);
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(s.paths.file)).digest('hex'), SHA256);
    assert.notEqual(fs.statSync(s.paths.file).ino, fs.statSync(source).ino, 'the wrong file was not kept');
    assert.deepEqual(fs.readFileSync(source), wrong, 'the shared file is left as it was');
});

test('symbolic links in /shared/models are never followed, whatever they point at', async (t) => {
    const s = setup(t);
    const outside = path.join(s.base, 'outside.gguf');
    fs.writeFileSync(outside, PAYLOAD);
    fs.symlinkSync(outside, path.join(s.shared, 'model.gguf'));
    fs.mkdirSync(path.join(s.base, 'elsewhere'));
    fs.writeFileSync(path.join(s.base, 'elsewhere', 'model.gguf'), PAYLOAD);
    fs.symlinkSync(path.join(s.base, 'elsewhere'), path.join(s.shared, 'dir-link'));
    const result = await s.get();
    assert.equal(result.adopted, undefined);
    assert.equal(s.fetches.length, 1, 'downloaded instead');
});

test('a verified file whose inode changed since (ctime) is checked again before use; changed bytes are discarded', async (t) => {
    const s = setup(t);
    const source = path.join(s.shared, 'model.gguf');
    fs.writeFileSync(source, PAYLOAD);
    await s.get();
    // Another agent touches the shared path: the same bytes, a new ctime. Checked again, kept.
    const later = new Date(Date.now() + 5000);
    fs.utimesSync(source, later, later);
    const again = await s.get();
    assert.equal(again.reverified, true);
    assert.equal(s.fetches.length, 0);
    // Now it rewrites the bytes in place through the shared link: discarded, downloaded again.
    const fd = fs.openSync(source, 'r+');
    fs.writeSync(fd, Buffer.from([PAYLOAD[10] ^ 0xff]), 0, 1, 10);
    fs.closeSync(fd);
    const third = await s.get();
    assert.equal(third.reverified, undefined);
    assert.equal(s.fetches.length, 1);
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(s.paths.file)).digest('hex'), SHA256);
    assert.notEqual(fs.statSync(s.paths.file).ino, fs.statSync(source).ino);
});

test('a file verified before stats were recorded is checked once, then trusted by its stat', async (t) => {
    const s = setup(t);
    await s.get();
    assert.equal(s.fetches.length, 1);
    const record = meta(s.paths);
    delete record.stat;
    fs.writeFileSync(s.paths.meta, JSON.stringify(record));
    assert.equal((await s.get()).reverified, true);
    assert.equal((await s.get()).reverified, undefined);
    assert.equal(s.fetches.length, 1);
});
