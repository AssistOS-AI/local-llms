// The host the pworker engine gets for this container (src/agentHost.mjs): every default is the literal the controller,
// the runners and the hardware readers had before the engine moved (the base file and line are named at each), and every
// environment override keeps the semantics it had there: an empty value takes the default.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { agentHost, agentRunnerLockFile, agentSeedCatalogFile } from '../src/agentHost.mjs';

const CATALOG = new URL('../catalog/', import.meta.url).pathname;

test('the defaults are the paths the image has always had', () => {
    const host = agentHost({});
    // deployments.mjs:147 (data), :185 and :227 (run root), :225 (shared models), :74 (image contract), runnerLock.mjs:43 (image lock),
    // hardware.mjs:24 (nvidia-smi), deployments.mjs:151 (Hugging Face).
    assert.equal(host.dataDir, '/data');
    assert.equal(host.runRoot, '/opt/runners');
    assert.equal(host.sharedModelsRoot, '/shared/models');
    assert.equal(host.imageContractFile, '/opt/local-llm/source.contract');
    assert.equal(host.imageRunnerLockFile, '/opt/local-llm/runners.lock.json');
    assert.equal(host.nvidiaSmi, '/usr/local/nvidia/bin/nvidia-smi');
    assert.equal(host.hfBaseUrl, 'https://huggingface.co');
    // The seed catalog is the agent's own file.
    assert.equal(host.seedCatalogFile, path.join(CATALOG, 'models.json'));
    assert.equal(host.seedCatalogFile, agentSeedCatalogFile());
    assert.ok(fs.existsSync(host.seedCatalogFile));
    // llamaCpp.mjs:30 and :52, ikLlamaCpp.mjs:20, ollama.mjs:19, vulkan.mjs:10-11 and :19-21, lmStudio.mjs:34, runnerInstaller.mjs:29-31.
    assert.deepEqual({ ...host.executables }, {
        'llama.cpp': '/opt/llama.cpp/llama-server', 'ik_llama.cpp': '/opt/ik_llama.cpp/llama-server',
        'llama.cpp-vulkan': '/opt/llama.cpp-vulkan/llama-server', ollama: '/opt/ollama/bin/ollama',
    });
    assert.deepEqual({ ...host.vulkan }, { serverPath: '/opt/llama.cpp-vulkan/llama-server', vulkaninfo: '/usr/bin/vulkaninfo', icdDir: '/opt/local-llm/vulkan/icd.d' });
    assert.equal(host.lmStudioSdkDir, '/opt/local-llm/lmstudio-sdk');
    assert.equal(host.nvidiaLibDir, '/usr/local/nvidia/lib64');
    assert.equal(host.uv, '/usr/local/bin/uv');
    assert.equal(host.python, '/usr/bin/python3');
    assert.equal(host.shmDir, '/dev/shm');
    assert.equal(host.toolPath, '/usr/local/nvidia/bin:/usr/local/bin:/usr/bin:/bin');
    assert.equal(host.platform.name, 'linux');
    assert.equal(Object.isFrozen(host), true);
});

test('each environment override reaches its field, and an empty value takes the default', () => {
    const smi = '/opt/fixture/nvidia-smi';
    const overrides = {
        LOCAL_LLM_DATA_DIR: ['dataDir', '/tmp/llm/data'],
        LOCAL_LLM_RUN_ROOT: ['runRoot', '/tmp/llm/runners'],
        LOCAL_LLM_SHARED_MODELS: ['sharedModelsRoot', '/tmp/llm/shared'],
        LOCAL_LLM_RUNNER_LOCK: ['imageRunnerLockFile', '/tmp/llm/image.lock.json'],
        LOCAL_LLM_CATALOG_FILE: ['seedCatalogFile', '/tmp/llm/models.json'],
        LOCAL_LLM_NVIDIA_SMI: ['nvidiaSmi', smi],
        LOCAL_LLM_HF_BASE_URL: ['hfBaseUrl', 'http://127.0.0.1:8099'],
    };
    const defaults = agentHost({});
    for (const [variable, [field, value]] of Object.entries(overrides)) {
        const host = agentHost({ [variable]: value });
        assert.equal(host[field], value, variable);
        // Nothing else moves with it.
        for (const other of Object.values(overrides).map(([name]) => name).filter((name) => name !== field)) assert.equal(host[other], defaults[other], `${variable} leaves ${other}`);
        // `||`: an empty value is the default, as it was.
        assert.equal(agentHost({ [variable]: '' })[field], defaults[field], `${variable} empty`);
    }
    // Every override at once, and the run root decides the cache directories the controller derives from it.
    const all = agentHost(Object.fromEntries(Object.entries(overrides).map(([variable, [, value]]) => [variable, value])));
    assert.deepEqual(Object.values(overrides).map(([field]) => all[field]), Object.values(overrides).map(([, value]) => value));
});

test('a relative path override is resolved against the working directory, as a read of it would have been', () => {
    const relative = agentHost({ LOCAL_LLM_CATALOG_FILE: 'catalog/models.json', LOCAL_LLM_RUN_ROOT: 'runners/' });
    assert.equal(relative.seedCatalogFile, path.resolve('catalog/models.json'));
    assert.equal(relative.runRoot, path.resolve('runners'));
    // The same file, written another way, is one path.
    assert.equal(agentHost({ LOCAL_LLM_CATALOG_FILE: './catalog/../catalog/models.json' }).seedCatalogFile, path.resolve('catalog/models.json'));
    // An absolute one with a trailing slash is normalized the same way.
    assert.equal(agentHost({ LOCAL_LLM_DATA_DIR: '/tmp/llm/data/' }).dataDir, '/tmp/llm/data');
});

test('a bare nvidia-smi name is looked up in PATH like execFile did; a name that is nowhere stays a path that does not exist', () => {
    const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-smi-')));
    try {
        fs.writeFileSync(path.join(dir, 'smi-fixture'), '#!/bin/sh\n', { mode: 0o755 });
        fs.writeFileSync(path.join(dir, 'smi-not-executable'), '#!/bin/sh\n', { mode: 0o644 });
        const env = { PATH: `/nonexistent:${dir}` };
        assert.equal(agentHost({ ...env, LOCAL_LLM_NVIDIA_SMI: 'smi-fixture' }).nvidiaSmi, path.join(dir, 'smi-fixture'));
        // A name with a directory in it is a path, not looked up.
        assert.equal(agentHost({ ...env, LOCAL_LLM_NVIDIA_SMI: 'sub/smi-fixture' }).nvidiaSmi, path.resolve('sub/smi-fixture'));
        // Found nowhere, or not executable: no PATH entry answers, so the path is the working directory's, which does not exist.
        assert.equal(agentHost({ ...env, LOCAL_LLM_NVIDIA_SMI: 'smi-not-executable' }).nvidiaSmi, path.resolve('smi-not-executable'));
        assert.equal(agentHost({ ...env, LOCAL_LLM_NVIDIA_SMI: 'smi-absent' }).nvidiaSmi, path.resolve('smi-absent'));
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
});

test('the agent runner lock is chosen by architecture: x64 and arm64 have one, anything else none (null, never undefined)', () => {
    const host = agentHost({});
    // runnerLock.mjs:59 and :327-330 at base.
    assert.equal(host.agentRunnerLockFile('x64'), path.join(CATALOG, 'runners.lock.linux-amd64.json'));
    assert.equal(host.agentRunnerLockFile('arm64'), path.join(CATALOG, 'runners.lock.linux-arm64.json'));
    for (const arch of ['riscv64', 'ia32', 's390x', 'amd64', 'arm', '', 'toString', '__proto__', null]) {
        assert.equal(agentRunnerLockFile(arch), null, String(arch));
        assert.equal(host.agentRunnerLockFile(arch), null, String(arch));
    }
    for (const arch of ['x64', 'arm64']) assert.ok(fs.existsSync(host.agentRunnerLockFile(arch)), arch);
    // Without an argument it is this process's architecture, as `process.arch` was the default at base.
    assert.equal(agentRunnerLockFile(), agentRunnerLockFile(process.arch));
});

test('GPU access is the reader of the Ploinky environment: the grant\'s status and reason, attributed to ploinky-environment', () => {
    const { gpuAccess } = agentHost({});
    assert.equal(gpuAccess.source, 'ploinky-environment');
    assert.deepEqual(gpuAccess.report({ PLOINKY_GPU_STATUS: 'unavailable', PLOINKY_GPU_REASON: 'revoked' }), { unavailable: true, reason: 'revoked' });
    assert.deepEqual(gpuAccess.report({ PLOINKY_GPU_STATUS: 'granted' }), { unavailable: false, reason: undefined });
    assert.deepEqual(gpuAccess.report({}), { unavailable: false, reason: undefined });
});

test('a value the engine\'s host cannot hold is refused with the field it is in, not read half way', () => {
    for (const [variable, field] of [['LOCAL_LLM_RUN_ROOT', 'runRoot'], ['LOCAL_LLM_DATA_DIR', 'dataDir'], ['LOCAL_LLM_SHARED_MODELS', 'sharedModelsRoot'],
        ['LOCAL_LLM_RUNNER_LOCK', 'imageRunnerLockFile'], ['LOCAL_LLM_CATALOG_FILE', 'seedCatalogFile']]) {
        assert.throws(() => agentHost({ [variable]: '/a:b' }), (error) => error.code === 'invalid_host' && error.details?.field === field, variable);
    }
    assert.throws(() => agentHost({ LOCAL_LLM_HF_BASE_URL: 'ftp://example.org' }), (error) => error.code === 'invalid_host' && error.details?.field === 'hfBaseUrl');
    assert.throws(() => agentHost({ LOCAL_LLM_HF_BASE_URL: 'not a url' }), (error) => error.code === 'invalid_host');
});
