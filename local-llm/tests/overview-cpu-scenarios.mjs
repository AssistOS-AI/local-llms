// These scenarios run against the base controller before the Vulkan change.
// Hardware reads use readSnapshot; host facts, runner locks and process execution are injected.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { agentRunnerLockFile, loadRunnerLocks } from '../src/controller/runnerLock.mjs';

export const VULKAN_KEYS = Object.freeze({
    vulkan_contract: '1', vulkan_llama_cpp: 'b11159',
    vulkan_llama_cpp_server: '/opt/llama.cpp-vulkan/llama-server', vulkan_mesa: '25.3.6+libkrun.3',
    vulkan_icd_radv: '/opt/local-llm/vulkan/icd.d/radeon_icd.json',
    vulkan_icd_venus: '/opt/local-llm/vulkan/icd.d/virtio_icd.json', vulkaninfo: '/usr/bin/vulkaninfo',
});

export function cpuFs() {
    return {
        readFileSync(file) {
            if (file === '/proc/meminfo') return 'MemTotal: 6291456 kB\nMemAvailable: 5242880 kB\nSwapFree: 0 kB\nMemFree: 4194304 kB\nCached: 1048576 kB\n';
            if (file === '/proc/cpuinfo') return 'processor: 0\ncore id: 0\n';
            throw Object.assign(new Error('absent'), { code: 'ENOENT' });
        },
        readdirSync() { throw Object.assign(new Error('absent'), { code: 'ENOENT' }); },
    };
}

export async function collectCpuOverview({ createController, readSnapshot, loadSeedCatalog, arch, vulkanKeys = false }) {
    const dataDir = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), 'local-llm-cpu-golden-'));
    const calls = [];
    const imageContract = { architecture: arch === 'arm64' ? 'arm64' : 'amd64', llama_cpp: 'b11159', ...(vulkanKeys ? VULKAN_KEYS : {}) };
    const controller = createController({
        dataDir, env: { PATH: '/usr/bin' }, hostArch: arch, imageContract,
        seedCatalog: loadSeedCatalog().filter((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m'), sharedModelsRoot: null,
        // Both simulated architectures retain the ARM64 lock data captured in the golden.
        runnerLocks: loadRunnerLocks({
            image: path.join(dataDir, 'absent-image-runner-lock.json'),
            agent: agentRunnerLockFile('arm64'),
        }),
        detectRunner: (runner) => ({ installed: false, version: null, reason: 'fixture' }),
        fileExists: () => false,
        snapshot: async ({ signal }) => {
            const snap = await readSnapshot({ dataDir, signal, fsApi: cpuFs(), env: {},
                execFileImpl(command, args, options, callback) {
                    calls.push(command);
                    callback(Object.assign(new Error('absent'), { code: 'ENOENT' }), '', '');
                },
                statfs: async () => ({ bsize: 4096, blocks: 100000, bavail: 90000 }),
            });
            delete snap.at;
            snap.cpus = 4;
            snap.cores = 4;
            return snap;
        },
    });
    try {
        const overview = JSON.parse(JSON.stringify(await controller.overview()));
        delete overview.profileDecision.decidedAt;
        const log = fs.readFileSync(path.join(dataDir, 'logs/runner.log'), 'utf8').split('\n')
            .filter((line) => line.includes('hardware profile:')).map((line) => line.slice(line.indexOf('[controller]')));
        return { overview, log, calls };
    } finally {
        await controller.drain();
        fs.rmSync(dataDir, { recursive: true, force: true });
    }
}
