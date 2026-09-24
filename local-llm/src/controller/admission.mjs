// Admission: can this (model, artifact, runner, parameters) run here, now?
//
// The result keeps two refusals apart: `incompatible` (this hardware can
// never run these parameters) and `insufficient-now` (another process holds
// the memory at the moment). Numbers are estimates, labelled as such, from
// the model's measured memory profile when the catalog has one (gpt-oss-20b
// was calibrated on the RTX 3060 in Phase 0) and from the file size otherwise.

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const CUDA_CONTEXT_BYTES = 200 * MIB;
const RUNNER_RAM_BYTES = 768 * MIB;
const GPU_MARGIN_BYTES = 256 * MIB;
const RAM_MARGIN_BYTES = 1 * GIB;
const KV_FACTOR = Object.freeze({ f16: 1, q8_0: 0.53125, q4_0: 0.28125 });
const DEFAULT_KV_BYTES_PER_TOKEN = 64 * 1024;
const DEFAULT_LAYERS = 48;
const UNKNOWN_MOE_EXPERT_FRACTION = 0.9;

function gib(bytes) {
    return `${(bytes / GIB).toFixed(1)} GiB`;
}

function result(status, reason, estimate, warnings = []) {
    return Object.freeze({ status, reason, estimate: Object.freeze({ ...estimate, isEstimate: true }), warnings: Object.freeze(warnings) });
}

// Custom runner policies build their answers with the same shape.
export { result as admissionResult };

// Weights split between GPU and CPU for llama.cpp. For a mixture-of-experts
// model, --n-cpu-moe keeps the expert tensors of that many layers in RAM
// while attention and shared weights stay on the GPU.
function llamaWeightSplit(model, size, params) {
    const memory = model.memory || {};
    const layers = memory.layers || DEFAULT_LAYERS;
    const layerFraction = Math.min(1, params.nGpuLayers / layers);
    const profiled = Number.isFinite(memory.nonExpertBytes) && Number.isFinite(memory.expertBytesPerLayer);
    let nonExpert;
    let expertPerLayer;
    if (profiled) {
        nonExpert = memory.nonExpertBytes;
        expertPerLayer = memory.expertBytesPerLayer;
    } else if (model.architecture === 'moe') {
        expertPerLayer = (size * UNKNOWN_MOE_EXPERT_FRACTION) / layers;
        nonExpert = size - expertPerLayer * layers;
    } else {
        nonExpert = size;
        expertPerLayer = 0;
    }
    const gpuLayers = Math.round(layerFraction * layers);
    const expertLayersOnGpu = Math.max(0, gpuLayers - Math.min(params.nCpuMoe || 0, layers));
    const gpuWeights = Math.min(size, nonExpert * layerFraction + expertPerLayer * expertLayersOnGpu);
    return { gpuWeights, cpuWeights: Math.max(0, size - gpuWeights), profiled };
}

function computeBufferBytes(params) {
    const ubatch = params.ubatchSize || 512;
    if (params.flashAttn === 'off') {
        // Measured: 2,146 MiB at 16k context and a 512 ubatch without flash attention.
        return 2146 * MIB * (params.ctxSize / 16384) * (ubatch / 512);
    }
    // Measured: 175 MiB at a 256 ubatch and 215 MiB at 512 with flash attention.
    return (135 + 0.16 * ubatch) * MIB;
}

export function estimateLlamaCpp({ model, source, params }) {
    const size = source.size || 0;
    const { gpuWeights, cpuWeights, profiled } = llamaWeightSplit(model, size, params);
    const kvFactor = (KV_FACTOR[params.cacheTypeK] + KV_FACTOR[params.cacheTypeV]) / 2 || 1;
    const kvPerToken = model.memory?.kvBytesPerToken || DEFAULT_KV_BYTES_PER_TOKEN;
    const kvBytes = kvPerToken * params.ctxSize * kvFactor + (model.memory?.fixedKvBytes || 0);
    const computeBytes = computeBufferBytes(params);
    const gpuBytes = params.nGpuLayers > 0 ? gpuWeights + kvBytes + computeBytes + CUDA_CONTEXT_BYTES : 0;
    return {
        weightsBytes: size,
        gpuWeightsBytes: Math.round(gpuWeights),
        cpuWeightsBytes: Math.round(cpuWeights),
        kvBytes: Math.round(kvBytes),
        computeBytes: Math.round(computeBytes),
        gpuBytes: Math.round(gpuBytes),
        ramBytes: Math.round(cpuWeights + RUNNER_RAM_BYTES),
        basis: profiled ? 'measured memory profile' : 'file size heuristic (no memory profile for this model)',
    };
}

export function estimateOllama({ model, source, params }) {
    const size = source.size || 0;
    const layers = model.memory?.layers || DEFAULT_LAYERS;
    const kvBytes = (model.memory?.kvBytesPerToken || DEFAULT_KV_BYTES_PER_TOKEN) * params.numCtx
        * (KV_FACTOR[params.kvCacheType] || 1) + (model.memory?.fixedKvBytes || 0);
    // Ollama places whole layers itself unless numGpu pins the count.
    const gpuWeights = params.numGpu === null || params.numGpu === undefined
        ? null
        : size * Math.min(1, params.numGpu / layers);
    return {
        weightsBytes: size,
        gpuWeightsBytes: gpuWeights === null ? null : Math.round(gpuWeights),
        kvBytes: Math.round(kvBytes),
        gpuBytes: gpuWeights === null ? null : Math.round(gpuWeights + kvBytes + CUDA_CONTEXT_BYTES),
        totalBytes: Math.round(size + kvBytes + CUDA_CONTEXT_BYTES),
        basis: size ? 'download size' : 'unknown size until the tag is pulled',
    };
}

function ramWarning(ramBytes, memory) {
    if (!memory?.availableBytes || !ramBytes) return [];
    if (ramBytes <= memory.availableBytes * 0.5) return [];
    return [`Uses about ${gib(ramBytes)} of system RAM; ${gib(memory.availableBytes)} is available now. `
        + 'Close other applications or reduce the context if the desktop becomes slow.'];
}

function diskShortage(remainingBytes, disk) {
    return Number.isFinite(disk?.freeBytes) && remainingBytes * 1.05 > disk.freeBytes;
}

function otherGpuUsers(gpu) {
    const users = (gpu.processes || []).map((process) => `${process.name} (${gib(process.usedBytes)})`);
    return users.length ? `; other GPU users: ${users.join(', ')}` : '';
}

// The llama-server policy: weights split between GPU and RAM by nCpuMoe,
// KV cache and compute buffers on the GPU. Used by llama.cpp and every
// runner that serves the same GGUF files with the same memory layout.
export function admitLlamaServer({ model, source, params, gpu, memory, disk, remainingDownloadBytes = 0 }) {
    const estimate = estimateLlamaCpp({ model, source, params });
    const warnings = ramWarning(estimate.ramBytes, memory);
    if (estimate.gpuBytes > gpu.totalBytes * 0.97) {
        return result('incompatible', `Needs about ${gib(estimate.gpuBytes)} of GPU memory; the GPU has `
            + `${gib(gpu.totalBytes)}. Keep more expert layers in RAM (nCpuMoe) or reduce ctxSize.`, estimate, warnings);
    }
    if (memory.totalBytes && estimate.ramBytes > memory.totalBytes) {
        return result('incompatible', `Needs about ${gib(estimate.ramBytes)} of RAM; this machine has `
            + `${gib(memory.totalBytes)}.`, estimate, warnings);
    }
    if (estimate.gpuBytes > gpu.freeBytes - GPU_MARGIN_BYTES) {
        return result('insufficient-now', `Needs about ${gib(estimate.gpuBytes)} of GPU memory; `
            + `${gib(gpu.freeBytes)} is free now${otherGpuUsers(gpu)}.`, estimate, warnings);
    }
    if (memory.availableBytes && estimate.ramBytes > memory.availableBytes - RAM_MARGIN_BYTES) {
        return result('insufficient-now', `Needs about ${gib(estimate.ramBytes)} of RAM; `
            + `${gib(memory.availableBytes)} is available now.`, estimate, warnings);
    }
    if (diskShortage(remainingDownloadBytes, disk)) {
        return result('insufficient-now', `The download needs ${gib(remainingDownloadBytes * 1.05)} of free disk; `
            + `${gib(disk.freeBytes)} is free.`, estimate, warnings);
    }
    return result('ok', null, estimate, warnings);
}

// The Ollama policy: Ollama places whole layers itself unless numGpu pins them.
export function admitOllama({ model, source, params, gpu, memory, disk, remainingDownloadBytes = 0 }) {
    const estimate = estimateOllama({ model, source, params });
    const gpuShare = estimate.gpuBytes ?? Math.min(estimate.totalBytes, Math.max(0, gpu.freeBytes - GPU_MARGIN_BYTES));
    const ramBytes = Math.max(0, estimate.totalBytes - gpuShare) + RUNNER_RAM_BYTES;
    const withRam = { ...estimate, ramBytes };
    const warnings = ramWarning(ramBytes, memory);
    if (estimate.gpuBytes !== null && estimate.gpuBytes > gpu.totalBytes * 0.97) {
        return result('incompatible', `numGpu puts about ${gib(estimate.gpuBytes)} on a GPU with `
            + `${gib(gpu.totalBytes)}; lower numGpu or leave it unset.`, withRam, warnings);
    }
    if (memory.totalBytes && estimate.totalBytes > gpu.totalBytes + memory.totalBytes) {
        return result('incompatible', `Needs about ${gib(estimate.totalBytes)} of GPU memory and RAM together.`,
            withRam, warnings);
    }
    if (gpu.freeBytes < GIB) {
        return result('insufficient-now', `Only ${gib(gpu.freeBytes)} of GPU memory is free${otherGpuUsers(gpu)}.`,
            withRam, warnings);
    }
    if (memory.availableBytes && ramBytes > memory.availableBytes - RAM_MARGIN_BYTES) {
        return result('insufficient-now', `Needs about ${gib(ramBytes)} of RAM; `
            + `${gib(memory.availableBytes)} is available now.`, withRam, warnings);
    }
    if (diskShortage(remainingDownloadBytes, disk)) {
        return result('insufficient-now', `The download needs ${gib(remainingDownloadBytes * 1.05)} of free disk; `
            + `${gib(disk.freeBytes)} is free.`, withRam, warnings);
    }
    return result('ok', null, withRam, warnings);
}

// vLLM on this GPU (measured with Qwen3-4B-AWQ on the RTX 3060 Laptop):
// CUDA reports about 94 % of nvidia-smi's total as usable, and besides the
// weights and the KV cache vLLM holds activations, the CUDA context and
// allocator slack, budgeted at 768 MiB.
const VLLM_USABLE_SHARE = 0.94;
const VLLM_OVERHEAD_BYTES = 768 * MIB;
const VLLM_MAX_UTILIZATION = 0.9;
const VLLM_RUNNER_RAM_BYTES = 3 * GIB;

// The vLLM policy: all weights on the GPU unless the admin explicitly offloads
// some to RAM (cpuOffloadGb, much slower), a KV cache sized for maxModelLen,
// and a GPU memory share (--gpu-memory-utilization) taken from what is free.
export function admitVllm({ model, source, params, gpu, memory, disk, remainingDownloadBytes = 0 }) {
    const size = source.size || 0;
    const offloadBytes = Math.round((params.cpuOffloadGb || 0) * GIB);
    const gpuWeightsBytes = Math.max(0, size - offloadBytes);
    const kvFactor = params.kvCacheDtype === 'fp8' ? 0.5 : 1;
    const kvBytes = Math.round((model.memory?.kvBytesPerToken || DEFAULT_KV_BYTES_PER_TOKEN) * params.maxModelLen * kvFactor);
    const gpuBytes = gpuWeightsBytes + kvBytes + VLLM_OVERHEAD_BYTES;
    const usable = (share) => share * gpu.totalBytes * VLLM_USABLE_SHARE;
    const fromFree = Math.floor(Math.min(VLLM_MAX_UTILIZATION, (gpu.freeBytes - 2 * GPU_MARGIN_BYTES) / gpu.totalBytes) * 100) / 100;
    const gpuMemoryUtilization = params.gpuMemoryUtilization ?? fromFree;
    const ramBytes = offloadBytes + VLLM_RUNNER_RAM_BYTES;
    const estimate = {
        weightsBytes: size, gpuWeightsBytes, cpuWeightsBytes: offloadBytes, kvBytes, gpuBytes, ramBytes,
        gpuMemoryUtilization, basis: 'snapshot size, KV cache for maxModelLen and a measured vLLM overhead',
    };
    const warnings = ramWarning(ramBytes, memory);
    if (offloadBytes > 0) {
        warnings.push(`Offloading ${gib(offloadBytes)} of weights to system RAM makes generation much slower: `
            + 'every token reads them over PCIe.');
    }
    if (offloadBytes > size) {
        return result('incompatible', `cpuOffloadGb (${gib(offloadBytes)}) is larger than the model (${gib(size)}).`, estimate, warnings);
    }
    // Whether it can ever fit: at the admin's share when set, else at vLLM's usual maximum.
    const ceiling = params.gpuMemoryUtilization ?? VLLM_MAX_UTILIZATION;
    if (gpuBytes > usable(ceiling)) {
        if (params.gpuMemoryUtilization !== null && params.gpuMemoryUtilization !== undefined) {
            return result('incompatible', `Needs about ${gib(gpuBytes)} of GPU memory; gpuMemoryUtilization ${params.gpuMemoryUtilization} `
                + `gives vLLM about ${gib(usable(ceiling))}. Raise gpuMemoryUtilization or leave it empty.`, estimate, warnings);
        }
        const needed = Math.ceil((gpuBytes - usable(ceiling)) / GIB);
        return result('incompatible', `Needs about ${gib(gpuBytes)} of GPU memory for its weights and a KV cache for `
            + `${params.maxModelLen} tokens; vLLM can use about ${gib(usable(ceiling))} of this GPU. Reduce maxModelLen, `
            + `pick a smaller model, or offload weights to RAM with cpuOffloadGb (at least ${needed + (offloadBytes / GIB)} GiB; much slower).`,
        estimate, warnings);
    }
    if (memory.totalBytes && ramBytes > memory.totalBytes) {
        return result('incompatible', `Needs about ${gib(ramBytes)} of RAM; this machine has ${gib(memory.totalBytes)}.`, estimate, warnings);
    }
    // Busy now: the share does not fit in the free memory, or what is free
    // leaves vLLM less than its minimum share (a large GPU mostly in use).
    if (gpuMemoryUtilization < 0.1 || gpuMemoryUtilization * gpu.totalBytes > gpu.freeBytes - GPU_MARGIN_BYTES
        || gpuBytes > usable(gpuMemoryUtilization)) {
        return result('insufficient-now', `Needs about ${gib(gpuBytes)} of GPU memory; ${gib(gpu.freeBytes)} is free now`
            + `${otherGpuUsers(gpu)}.`, estimate, warnings);
    }
    if (memory.availableBytes && ramBytes > memory.availableBytes - RAM_MARGIN_BYTES) {
        return result('insufficient-now', `Needs about ${gib(ramBytes)} of RAM; ${gib(memory.availableBytes)} is available now.`,
            estimate, warnings);
    }
    if (diskShortage(remainingDownloadBytes, disk)) {
        return result('insufficient-now', `The download needs ${gib(remainingDownloadBytes * 1.05)} of free disk; `
            + `${gib(disk.freeBytes)} is free.`, estimate, warnings);
    }
    return result('ok', null, estimate, warnings);
}

/**
 * The checks every runner shares, then the runner's own policy
 * (`runner.admit`), which sizes the estimate for how that runner uses memory.
 *
 * @param {{ runner, model, source, params, snapshot, remainingDownloadBytes? }} input
 * @returns {{ status: 'ok'|'incompatible'|'insufficient-now', reason, estimate, warnings }}
 */
export function admit({ runner, model, source, params, snapshot, remainingDownloadBytes = 0 }) {
    if (!runner.supported || typeof runner.admit !== 'function') {
        return result('incompatible', runner.unsupportedReason || `${runner.displayName} is not supported in this release.`, {});
    }
    if (!source) {
        return result('incompatible', `${model.displayName} has no ${runner.displayName} source.`, {});
    }
    const gpu = snapshot?.gpu;
    if (!gpu?.available) {
        return result('incompatible', gpu?.reason || 'No GPU is available to this agent.', {});
    }
    return runner.admit({
        model, source, params, gpu, memory: snapshot.memory || {}, disk: snapshot.disk, remainingDownloadBytes,
    });
}
