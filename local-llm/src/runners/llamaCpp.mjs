import { createLlamaServerRunner } from './llamaServer.mjs';

// b11125 replaced --mlock/--no-mmap with a single --load-mode option.
function loadArgs({ mlock, noMmap }) {
    if (noMmap && mlock) {
        return ['--load-mode', 'mlock'];
    }
    if (noMmap) {
        return ['--load-mode', 'none'];
    }
    return mlock ? ['--load-mode', 'mmap+mlock'] : [];
}

function parseVersion(output) {
    const match = /\bbuild (\d+)\b/.exec(output);
    return match ? `b${match[1]}` : null;
}

export const llamaCppRunner = createLlamaServerRunner({
    id: 'llama.cpp',
    displayName: 'llama.cpp',
    executable: '/opt/llama.cpp/llama-server',
    pinnedVersion: 'b11125',
    port: 18080,
    dialect: Object.freeze({
        // -lv 4 prints the device, offload and CUDA buffer lines the status reads.
        quietArgs: Object.freeze(['--no-webui', '-lv', '4']),
        unifiedKv: true,
        loadArgs,
        jinja: (model) => Boolean(model?.requiresJinja),
        parseVersion,
    }),
});
