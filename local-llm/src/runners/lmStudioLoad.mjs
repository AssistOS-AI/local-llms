#!/usr/bin/env node
// The LM Studio adapter's helper (lmStudio.mjs): one short process per step,
// in its own process group, so a Stop or a drain ends it. It talks to llmster
// only through LM Studio's MIT SDK, pinned in the image.
//
//   wait   {daemon, sdk}                               until llmster answers
//   load   {daemon, sdk, path, identifier, config}     load one model
//
// `load` finds LM Studio's own key for the file we imported (never derived
// from our model id) and loads it with the given config under our identifier.
// It prints one JSON line. The SDK connects as a guest client, which LM Studio
// lets load a model but not start its server (observed 2026-09-25: "does not
// have the required permission: httpServer.start"), so the server, the list
// of loaded models and unloads go through the privileged `lms` CLI instead.

import { createRequire } from 'node:module';
import { setTimeout as sleep } from 'node:timers/promises';

const WAIT_MS = 120_000;
const INDEX_WAIT_MS = 30_000;

function fail(message) {
    process.stderr.write(`${message}\n`);
    process.exit(1);
}

const quiet = { info() {}, debug() {}, warn() {}, error() {} };

async function connect({ daemon, sdk }) {
    const { LMStudioClient } = createRequire(`${sdk}/`)('@lmstudio/sdk');
    const deadline = Date.now() + WAIT_MS;
    let last = null;
    while (Date.now() < deadline) {
        const client = new LMStudioClient({ baseUrl: daemon, logger: quiet });
        try {
            const version = await client.system.getLMStudioVersion();
            return { client, version };
        } catch (error) {
            last = error;
        }
        await sleep(500);
    }
    fail(`LM Studio did not answer within ${WAIT_MS / 1000} s: ${last?.message ?? 'no answer'}`);
}

async function load(options) {
    const { client } = await connect(options);
    const deadline = Date.now() + INDEX_WAIT_MS;
    let model = null;
    while (!model && Date.now() < deadline) {
        model = (await client.system.listDownloadedModels('llm')).find((entry) => entry.path === options.path) ?? null;
        if (!model) await sleep(250);
    }
    if (!model) fail(`LM Studio did not index ${options.path} within ${INDEX_WAIT_MS / 1000} s.`);
    await client.llm.load(model.modelKey, { identifier: options.identifier, config: options.config, verbose: false });
    return { modelKey: model.modelKey, identifier: options.identifier };
}

const [command, json] = process.argv.slice(2);
let options;
try {
    options = JSON.parse(json);
} catch {
    fail('usage: lmStudioLoad.mjs wait|load <json>');
}
try {
    const steps = { wait: async () => ({ version: (await connect(options)).version }), load: () => load(options) };
    if (!Object.hasOwn(steps, command)) fail(`unknown step ${command}`);
    const result = await steps[command]();
    process.stdout.write(`${JSON.stringify(result)}\n`, () => process.exit(0));
} catch (error) {
    fail(error?.message ?? String(error));
}
