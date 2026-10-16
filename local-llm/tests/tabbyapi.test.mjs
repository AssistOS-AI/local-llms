// ExLlamaV3 + TabbyAPI (runners plan §5.6, Phase R6): installed on demand
// after an admin accepts TabbyAPI's AGPL notice (DS004), reads an EXL3
// snapshot, runs from its own source directory in the runnable copy on
// loopback, and requires its per-start key.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { loadSeedCatalog } from '../src/engineOptions.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

const SEED = loadSeedCatalog();
const QWEN8 = SEED.find((model) => model.id === 'qwen3-8b-exl3');

function snapshot({ freeMiB = 6000 } = {}) {
    return {
        gpu: { available: true, name: 'Test GPU', totalBytes: 6144 * MIB, usedBytes: (6144 - freeMiB) * MIB, freeBytes: freeMiB * MIB, processes: [] },
        memory: { totalBytes: 31 * GIB, availableBytes: 24 * GIB },
        disk: { freeBytes: 300 * GIB, totalBytes: 500 * GIB },
    };
}

test('the seed catalog pins Qwen3-8B EXL3 4.0 bpw as an EXL3 snapshot', () => {
    assert.ok(QWEN8);
    const source = QWEN8.sources.exl3;
    assert.equal(source.repo, 'turboderp/Qwen3-8B-exl3');
    assert.equal(source.revision, '4.0bpw');
    assert.equal(source.commit, '1fd66d10f8fbdf071a0ff35842a2d1bf0df94b45');
    assert.equal(source.files.length, 8);
    assert.equal(source.size, 5206949815);
    assert.equal(QWEN8.recommended.dedicated.tabbyapi.cacheMode, 'Q4');
});

