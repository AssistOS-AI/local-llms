// The runner framework: every runner is an adapter that brings its own
// port, key, start-up pipeline, chat model name, admission policy and form
// metadata. A new runner is an adapter plus data; the controller has no
// per-runner branches.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { WEIGHT_FORMATS } from 'pworker/local-llm/internal/controller/catalog.mjs';

import { RUNNERS } from '../src/engineOptions.mjs';

test('every runnerId enum in mcp-config.json lists exactly the registry, and the format enum the weight formats', () => {
    const config = JSON.parse(fs.readFileSync(new URL('../mcp-config.json', import.meta.url), 'utf8'));
    const enums = [];
    const visit = (node) => {
        if (!node || typeof node !== 'object') return;
        if (node.runnerId?.enum) enums.push(node.runnerId.enum);
        for (const value of Object.values(node)) visit(value);
    };
    visit(config);
    assert.equal(enums.length, 3);
    for (const values of enums) assert.deepEqual(values, Object.keys(RUNNERS));
    const remove = config.tools.find((tool) => tool.name === 'local_llm_weights_delete');
    assert.deepEqual(remove.inputSchema.properties.format.enum, Object.keys(WEIGHT_FORMATS));
});

