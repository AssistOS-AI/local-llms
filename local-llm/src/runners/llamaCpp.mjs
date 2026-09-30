import { LOAD_MODES, createLlamaServerRunner } from './llamaServer.mjs';

// b10105 added --load-mode; b10875 removed --no-mmap, --mmap, --mlock and
// --direct-io. `auto` passes no flag: the server then avoids mmap on an
// integrated GPU by itself.
function loadArgs(loadMode) {
    return loadMode && loadMode !== 'auto' ? ['--load-mode', loadMode] : [];
}

function parseVersion(output) {
    const match = /\bbuild (\d+)\b/.exec(output);
    return match ? `b${match[1]}` : null;
}

export const llamaCppRunner = createLlamaServerRunner({
    id: 'llama.cpp',
    displayName: 'llama.cpp',
    executable: '/opt/llama.cpp/llama-server',
    pinnedVersion: 'b11159',
    port: 18080,
    dialect: Object.freeze({
        // -lv 4 prints the device, offload and CUDA buffer lines the status reads.
        quietArgs: Object.freeze(['--no-webui', '-lv', '4']),
        unifiedKv: true,
        loadModes: LOAD_MODES,
        loadArgs,
        unified: true,
        cpu: true,
        jinja: (model) => Boolean(model?.requiresJinja),
        parseVersion,
    }),
});
