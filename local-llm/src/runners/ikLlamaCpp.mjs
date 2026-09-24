import { createLlamaServerRunner } from './llamaServer.mjs';

// ik_llama.cpp: a llama.cpp fork with faster hybrid CPU/GPU MoE, built into
// the image from a pinned commit (runners plan R1, R6 = a). Its server keeps
// most of llama-server's command line; the differences are in the dialect
// below. It must never get -rtr with experts in RAM (its README warns that
// repacked tensors then always run on the CPU).
const PINNED_COMMIT = '20f7a72edd7049fe5a87eef2b5e9a50ae109ca4b';

// `version: <commit count> (<short commit>)`; the count depends on the clone depth.
function parseVersion(output) {
    const match = /^version: \d+ \(([0-9a-f]{7,40})\)/m.exec(output);
    return match ? match[1].slice(0, 7) : null;
}

export const ikLlamaCppRunner = createLlamaServerRunner({
    id: 'ik_llama.cpp',
    displayName: 'ik_llama.cpp',
    // Not on PATH: llama.cpp's server has the same name.
    executable: '/opt/ik_llama.cpp/llama-server',
    pinnedVersion: PINNED_COMMIT.slice(0, 7),
    port: 18081,
    dialect: Object.freeze({
        // ik has no --no-webui or -lv; its default log level already prints
        // the offload and buffer lines.
        quietArgs: Object.freeze(['--webui', 'none']),
        // No --kv-unified: the context is split across the parallel slots.
        unifiedKv: false,
        loadArgs: ({ mlock, noMmap }) => [...(mlock ? ['--mlock'] : []), ...(noMmap ? ['--no-mmap'] : [])],
        // ik turns Jinja chat templates off by default; llama.cpp has them on.
        jinja: () => true,
        parseVersion,
    }),
});
