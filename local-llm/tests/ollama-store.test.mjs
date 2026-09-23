import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { deleteOllamaModel, manifestPath, readOllamaManifest } from '../src/controller/ollamaStore.mjs';

function store(t) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-ollama-store-'));
    t.after(() => fs.rmSync(root, { recursive: true, force: true }));
    const modelsDir = path.join(root, 'models', 'ollama');
    fs.mkdirSync(path.join(modelsDir, 'blobs'), { recursive: true });
    return { root, modelsDir };
}

function writeManifest(modelsDir, tag, layers) {
    const target = manifestPath(modelsDir, tag);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, JSON.stringify({ schemaVersion: 2, layers }));
}

test('a manifest digest that is not sha256:<64 hex> never reaches the file system', (t) => {
    const { root, modelsDir } = store(t);
    // blobs/sha256-../../../outside.txt resolves beside the models directory.
    const outside = path.join(root, 'models', 'outside.txt');
    fs.writeFileSync(outside, 'keep me');
    const good = `sha256:${'a'.repeat(64)}`;
    fs.writeFileSync(path.join(modelsDir, 'blobs', `sha256-${'a'.repeat(64)}`), 'abcd');
    writeManifest(modelsDir, 'evil:latest', [
        { digest: good, size: 4, mediaType: 'application/vnd.ollama.image.model' },
        { digest: 'sha256:../../../outside.txt', size: 7, mediaType: 'application/vnd.ollama.image.model' },
    ]);
    assert.throws(() => readOllamaManifest(modelsDir, 'evil:latest'), /invalid layer digest/);
    assert.equal(deleteOllamaModel(modelsDir, 'evil:latest'), 0);
    assert.equal(fs.readFileSync(outside, 'utf8'), 'keep me');
});
