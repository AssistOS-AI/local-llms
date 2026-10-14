// Unified memory under a Ploinky RAM budget (DS003, DS005): the host reserve and
// the floor belong to the physical pool, the budget is checked on its own and is
// advisory for GPU device memory, and vLLM's automatic share is sized to the model.
//
// Admission is driven through admit() with the snapshot's budget observation
// attached as the production reader attaches it; the controller is used where a
// test must also see that no runner started. Hardware, downloads and runners are
// injected: no /proc, GPU, network or container is used. The A-LL ids are the
// GB10 hardware-limits plan's acceptance vectors (r3, amended by r4).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { VLLM_OVERHEAD_BYTES, VLLM_RUNNER_RAM_BYTES, admit, admissionResult } from '../src/controller/admission.mjs';
import { loadSeedCatalog, validateModel } from '../src/controller/catalog.mjs';
import { createController } from '../src/controller/deployments.mjs';
import { attachMemoryBudget, effectiveMemory, observeMemoryBudget } from '../src/controller/ploinkyBudget.mjs';
import { UNIFIED, admitUnifiedVllm } from '../src/controller/profiles.mjs';
import { createStateStore } from '../src/controller/stateStore.mjs';
import { llamaCppRunner } from '../src/runners/llamaCpp.mjs';
import { vllmRunner } from '../src/runners/vllm.mjs';

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const KEY = 'k'.repeat(43);
// The DGX Spark's MemTotal, as the base goldens use it.
const PT = 128450985984;
const SEED = loadSeedCatalog();
const SMALL = SEED.find((model) => model.id === 'qwen2.5-0.5b-instruct-q4_k_m');
const GB10 = Object.freeze({
    available: true, name: 'NVIDIA GB10', driverVersion: '580.159.03', memoryModel: 'unified',
    totalBytes: null, usedBytes: null, freeBytes: null, processes: [],
    device: { pciDeviceId: '0x2E1210DE', computeCapability: '12.1', addressingMode: 'ATS' },
});

// A Qwen2.5-0.5B snapshot added at run time (the catalog has no vLLM entry for it): 999586347 weight bytes.
// `weights` and `kvBytesPerToken` let a vector size it differently.
function hfModel({ id = 'qwen2.5-0.5b-instruct-hf', weights = 999585688, kvBytesPerToken = 12288, seed = false } = {}) {
    return validateModel({
        id, displayName: 'Qwen2.5 0.5B Instruct (HF)', architecture: 'dense', contextLength: 32768, profiles: ['unified'],
        memory: { layers: 24, kvBytesPerToken },
        sources: { hf: { type: 'hf-snapshot', repo: 'Qwen/Qwen2.5-0.5B-Instruct', revision: 'main', commit: 'c'.repeat(40), files: [
            { path: 'config.json', size: 659, sha256: 'a'.repeat(64) }, { path: 'model.safetensors', size: weights, sha256: 'b'.repeat(64) },
        ] } },
    }, { seed });
}
const QWEN = hfModel();
const WEIGHTS = QWEN.sources.hf.files.reduce((sum, file) => sum + file.size, 0);

// The snapshot admission reads. A finite `max` attaches the cgroup observation exactly as the production reader does.
function snapshot({ total = PT, available = 112 * GIB, max = null, current = 0 } = {}) {
    const snap = { gpu: structuredClone(GB10), memory: { totalBytes: total, availableBytes: available }, disk: { freeBytes: 400 * GIB }, cpus: 20 };
    return max === null ? snap : attachMemoryBudget(snap, observeMemoryBudget({ maxText: `${max}\n`, currentText: `${current}\n` }));
}

function llama(options, params = {}) {
    return admit({
        runner: llamaCppRunner, model: SMALL, source: SMALL.sources.gguf, snapshot: snapshot(options), profile: 'unified',
        params: llamaCppRunner.normalizeParams(params, { model: SMALL, profile: 'unified' }),
    });
}

function vllm(options, params = {}, model = QWEN) {
    return admit({
        runner: vllmRunner, model, source: model.sources.hf, snapshot: snapshot(options), profile: 'unified',
        params: vllmRunner.normalizeParams(params, { model, profile: 'unified' }),
    });
}

// llama.cpp's need for the default parameters, read from an unbudgeted admission (the need does not depend on the host).
const NEED = llama({}).estimate.unifiedBytes;

test('LL.g3-no-double-count', () => {
    // 21 % of the pool as the budget: the base took the 16 GiB reserve out of the budget again and refused for good.
    const verdict = llama({ total: PT, available: 120 * GIB, max: 26974617600, current: GIB });
    assert.equal(verdict.status, 'ok', verdict.reason);
    assert.equal(verdict.reasonCode, undefined);
});

test('LL.budget-boundary', () => {
    // Equality needs memory.current at 0, so the headroom is the whole budget.
    const exact = llama({ max: NEED, current: 0 });
    assert.equal(exact.status, 'ok', exact.reason);
    const short = llama({ max: NEED - 1, current: 0 });
    assert.equal(short.status, 'incompatible');
    assert.equal(short.reasonCode, 'budget_exceeded');
    // The words name the estimate, the budget, the advisory nature and the smallest budget with its fixed part.
    assert.equal(short.reason, `Needs about ${(NEED / GIB).toFixed(1)} GiB of shared memory; this agent's Ploinky RAM budget is ${((NEED - 1) / GIB).toFixed(1)} GiB. `
        + 'On unified memory GPU device memory (cudaMalloc) is not charged to that budget, so local-llm keeps the run inside it by this estimate. '
        + `The smallest budget this run fits is about ${(NEED / GIB).toFixed(1)} GiB, of which 9.5 GiB is llama.cpp's fixed prompt cache and runtime.`);
    assert.equal(UNIFIED.cacheRamMiB * MIB + UNIFIED.runtimeBytes, 9.5 * GIB, 'the fixed part at the defaults');
    // The spec 95 assertion matches both texts.
    assert.match(short.reason, /this agent's Ploinky RAM budget is/);
});

test('LL.headroom-boundary', () => {
    const budget = 64 * GIB;
    const exact = llama({ max: budget, current: budget - NEED });
    assert.equal(exact.status, 'ok', exact.reason);
    const short = llama({ max: budget, current: budget - NEED + 1 });
    assert.equal(short.status, 'insufficient-now');
    assert.equal(short.reasonCode, 'budget_insufficient_now');
    assert.match(short.reason, /of this agent's 64\.0 GiB budget is free now/);
    // A budget that cannot hold the need is not a headroom question.
    assert.equal(llama({ max: NEED - 1, current: NEED - 1 }).reasonCode, 'budget_exceeded');
});

test('LL.physical-reserve-kept', () => {
    const noBudget = llama({ total: NEED + 16 * GIB - 1, available: 400 * GIB });
    assert.equal(noBudget.status, 'incompatible');
    assert.match(noBudget.reason, /must keep 16\.0 GiB/);
    assert.equal(noBudget.reasonCode, undefined);
    assert.equal(llama({ total: NEED + 16 * GIB, available: 400 * GIB }).status, 'ok');
    // A budget that holds the need does not lift the physical reserve.
    const budgeted = llama({ total: NEED + 16 * GIB - 1, available: 400 * GIB, max: 64 * GIB, current: 0 });
    assert.equal(budgeted.status, 'incompatible');
    assert.match(budgeted.reason, /must keep 16\.0 GiB/);
    assert.equal(budgeted.reasonCode, undefined);
});

test('LL.physical-floor-kept', () => {
    const verdict = llama({ available: NEED + 8 * GIB - 1, max: 100 * GIB, current: 0 });
    assert.equal(verdict.status, 'insufficient-now');
    assert.equal(verdict.reasonCode, undefined);
    assert.match(verdict.reason, /8\.0 GiB kept free/);
    assert.equal(llama({ available: NEED + 8 * GIB, max: 100 * GIB, current: 0 }).status, 'ok');
});

test('LL.floor-judged-apart-from-headroom', () => {
    // A budget with only the need left is enough: the 8 GiB floor is checked against the physical MemAvailable, not the headroom.
    assert.equal(llama({ available: 100 * GIB, max: 40 * GIB, current: 40 * GIB - NEED }).status, 'ok');
    assert.equal(llama({ available: 100 * GIB, max: 40 * GIB, current: 40 * GIB - NEED + 1 }).status, 'insufficient-now');
});

// The six base verdicts of the unlimited unified host, frozen at base 7fdbff5a2083 as literals (the plan's
// goldens.json, key A-LL-6). Nothing here is recomputed on the candidate. Inputs are the golden's own.
const GOLDEN_PARAMS = Object.freeze({ ctxSize: 32768, parallel: 1, loadMode: 'dio', mtp: false, threads: null });
const GOLDEN_NEED = 11579755688;
const GOLDEN_LLAMA_SOURCE = Object.freeze({ ...SMALL.sources.gguf, size: 491400032 });
const UNLIMITED_GOLDEN = Object.freeze([
    { name: 'v1 ok', total: 128450985984, available: 120259084288, status: 'ok', reason: null },
    { name: 'v2 incompatible by host reserve (Pt = need + 16GiB - 1)', total: 28759624871, available: 429496729600, status: 'incompatible',
        reason: 'Needs about 10.8 GiB of the 26.8 GiB shared memory pool, which must keep 16.0 GiB for the host. Reduce the context or the parallel slots, or pick a smaller model.' },
    { name: 'v3 insufficient-now by floor (Pa = need + 8GiB - 1)', total: 128450985984, available: 20169690279, status: 'insufficient-now',
        reason: 'Needs about 10.8 GiB of shared memory and 8.0 GiB kept free; 18.8 GiB is available now. Other processes on this machine hold the rest.' },
    { name: 'v4 boundary need = Pt - 16GiB exactly (Pa ample)', total: 28759624872, available: 429496729600, status: 'ok', reason: null },
    { name: 'v5 boundary need = Pa - 8GiB exactly (Pt ample)', total: 128450985984, available: 20169690280, status: 'ok', reason: null },
]);

// The frozen base estimate and warning of the llama.cpp vectors (goldens.json A-LL-6): only poolBytes follows the vector's total.
const GOLDEN_LLAMA_WARNINGS = Object.freeze([
    'Not measured on unified memory: about 10.8 GiB is an estimate. The memory guard stops the runner at once if available memory falls below 8.0 GiB.',
]);
const goldenLlamaEstimate = (poolBytes) => ({
    weightsBytes: 491400032, kvBytes: 402653184, computeBytes: 485155144, mtpBytes: 0, runtimeBytes: 1610612736, cacheRamBytes: 8589934592,
    unifiedBytes: 11579755688, poolBytes, floorBytes: 8589934592, hostReserveBytes: 17179869184, envelope: null, measured: false, defaulted: [],
    userSizing: [],
    basis: 'estimate, not measured on this machine: pinned weight size, f16 KV cache for ctxSize at 12288 bytes per token (from the catalog entry), compute buffers, runtime and the prompt cache bound',
    isEstimate: true,
});
const withoutKeys = (object, keys) => Object.fromEntries(Object.entries(object).filter(([key]) => !keys.includes(key)));

test('LL.unlimited-golden', () => {
    for (const vector of UNLIMITED_GOLDEN) {
        const verdict = admit({
            runner: llamaCppRunner, model: SMALL, source: GOLDEN_LLAMA_SOURCE, profile: 'unified', params: GOLDEN_PARAMS,
            snapshot: { gpu: structuredClone(GB10), memory: { totalBytes: vector.total, availableBytes: vector.available }, disk: { freeBytes: 429496729600 }, cpus: 20 },
        });
        assert.equal(verdict.status, vector.status, vector.name);
        assert.equal(verdict.reason, vector.reason, vector.name);
        assert.equal(verdict.estimate.unifiedBytes, GOLDEN_NEED, vector.name);
        assert.equal(verdict.reasonCode, undefined, vector.name);
        // The whole frozen estimate and the warnings, apart from the one additive field.
        assert.deepEqual(withoutKeys(verdict.estimate, ['physicalPoolBytes']), goldenLlamaEstimate(vector.total), vector.name);
        assert.deepEqual([...verdict.warnings], GOLDEN_LLAMA_WARNINGS, vector.name);
    }
    // v6: the unlimited vLLM run. Its status and reason are the base's; its share is the A-LL-21 change.
    const v6 = vllm({ total: 128450985984, available: 120259084288 });
    assert.equal(v6.status, 'ok');
    assert.equal(v6.reason, null);
    // Its frozen estimate, apart from the additive field and the four values the model-sized default changes on purpose.
    assert.deepEqual(withoutKeys(v6.estimate, ['physicalPoolBytes', 'sizedBytes', 'gpuMemoryUtilization', 'budgetBytes', 'unifiedBytes']), {
        weightsBytes: 999586347, kvBytes: 50331648, overheadBytes: 805306368, runnerRamBytes: 3221225472, floorBytes: 8589934592,
        hostReserveBytes: 17179869184, poolBytes: 128450985984, envelope: null, measured: false, experimental: true, defaulted: [],
        userSizing: ['memory.kvBytesPerToken'],
        basis: 'experimental estimate, not measured: vLLM\'s share of the shared pool plus its runner RAM (measured on a dedicated GPU); the share must hold the snapshot, a KV cache for maxModelLen at 12288 bytes per token (from the model entry) and vLLM\'s overhead',
        isEstimate: true,
    });
    // Its frozen warnings: the first is the base text; the second is the user-sizing warning, whose vLLM tail was reworded on purpose.
    assert.equal(v6.warnings.length, 2);
    assert.equal(v6.warnings[0], 'Experimental: vLLM has not been measured on unified memory. Its share and its RAM beyond it are estimates, and the memory guard stops it at once if available memory falls below 8.0 GiB.');
    assert.ok(v6.warnings[1].startsWith('Sized with memory.kvBytesPerToken from the model entry added at run time. Nothing checks these against the weights: an understated value '));
});

test('LL.vllm-auto-share-budget', () => {
    // 12 GiB of weights and 256 sequences of 4096 tokens: the model-sized share (0.20) is above what the budget leaves,
    // so the budget binds; the single-sequence weights and KV cache fit 0.15 of the pool.
    const model = hfModel({ id: 'budget-share-model', weights: 12 * GIB });
    const options = { total: 128 * GIB, available: 120 * GIB, max: 24 * GIB, current: GIB };
    const verdict = vllm(options, { maxNumSeqs: 256 }, model);
    assert.equal(verdict.status, 'ok', verdict.reason);
    // min(shareOf(min(Pt - R, B) - runner RAM) = 0.16, shareOf(min(Pa - F, Hb) - runner RAM) = 0.15)
    assert.equal(verdict.estimate.gpuMemoryUtilization, 0.15);
    assert.ok(verdict.estimate.sizedBytes > 0.14 * 128 * GIB);
    assert.ok(verdict.estimate.weightsBytes + verdict.estimate.kvBytes + VLLM_OVERHEAD_BYTES <= 0.15 * 128 * GIB);
    assert.ok(verdict.estimate.unifiedBytes <= 24 * GIB - GIB, 'the share and the runner RAM stay inside the headroom');
    assert.equal(verdict.estimate.physicalPoolBytes, 128 * GIB);
    // With one sequence the model-sized share is the smaller one, and the budget does not raise it.
    const single = vllm(options, {}, model);
    assert.equal(single.estimate.gpuMemoryUtilization, Math.max(0.1, Math.ceil((single.estimate.sizedBytes / (128 * GIB)) * 100) / 100));
    assert.ok(single.estimate.gpuMemoryUtilization < 0.15);
});

// A-LL-21 (r4 A1). The base golden: share 0.84 and 107898828227 bytes for this snapshot.
const BASE_GOLDEN_SHARE = 0.84;
const BASE_GOLDEN_BUDGET_BYTES = 107898828227;

test('LL.vllm-default-share-unbudgeted', () => {
    assert.equal(WEIGHTS, 999586347);
    const verdict = vllm({ total: PT, available: 112 * GIB });
    // sizedBytes = 999586347 + 50331648 + 805306368, which is 0.02 of the pool: the schema minimum 0.1 binds.
    assert.equal(verdict.status, 'ok', verdict.reason);
    assert.equal(verdict.estimate.gpuMemoryUtilization, 0.1);
    assert.equal(verdict.estimate.budgetBytes, 12845098598);
    assert.equal(verdict.estimate.sizedBytes, 1855224363);
    // Never more than the base default, which took what was free.
    assert.ok(verdict.estimate.gpuMemoryUtilization < BASE_GOLDEN_SHARE && verdict.estimate.budgetBytes < BASE_GOLDEN_BUDGET_BYTES);
    // The reported pool stays MemTotal, and the physical pool is reported beside it.
    assert.equal(verdict.estimate.poolBytes, PT);
    assert.equal(verdict.estimate.physicalPoolBytes, PT);
});

test('LL.vllm-default-share-budgeted', () => {
    const budget = Math.round(0.25 * PT);
    assert.equal(budget, 32112746496);
    const verdict = vllm({ total: PT, available: 112 * GIB, max: budget, current: GIB });
    assert.equal(verdict.status, 'ok', verdict.reason);
    // The budget does not raise the model-sized share (what the budget leaves would allow 0.21).
    assert.equal(verdict.estimate.gpuMemoryUtilization, 0.1);
    assert.ok(verdict.estimate.budgetBytes + VLLM_RUNNER_RAM_BYTES <= budget - GIB);
    // The pool the estimate reports is the capped one; the physical pool is beside it.
    assert.equal(verdict.estimate.poolBytes, budget);
    assert.equal(verdict.estimate.physicalPoolBytes, PT);
    // At base this was insufficient-now: maxShare was 0.09 after the reserve was taken out of the budget.
});

test('LL.vllm-default-share-sized-to-sequences', () => {
    // 256 sequences of 32768 tokens: the model-sized share binds below the pool's cap.
    const verdict = vllm({ total: PT, available: 112 * GIB }, { maxNumSeqs: 256, maxModelLen: 32768 });
    assert.equal(verdict.status, 'ok', verdict.reason);
    assert.equal(verdict.estimate.gpuMemoryUtilization, 0.82);
    assert.equal(verdict.estimate.sizedBytes, 104884107819);
    // The single-sequence check is what decides whether the model fits at all.
    assert.equal(verdict.estimate.weightsBytes + verdict.estimate.kvBytes + VLLM_OVERHEAD_BYTES, WEIGHTS + 12288 * 32768 + VLLM_OVERHEAD_BYTES);
});

test('LL.vllm-admin-share-over-budget', () => {
    const verdict = vllm({ total: PT, available: 112 * GIB, max: Math.round(0.25 * PT), current: GIB }, { gpuMemoryUtilization: 0.5 });
    assert.equal(verdict.status, 'incompatible');
    assert.equal(verdict.reasonCode, 'budget_exceeded');
    assert.match(verdict.reason, /this agent's Ploinky RAM budget is 29\.9 GiB/);
    assert.match(verdict.reason, /The smallest budget this run fits is about [\d.]+ GiB\.$/, 'vLLM has no fixed llama.cpp part');
    assert.doesNotMatch(verdict.reason, /llama\.cpp/);
    // The admin value is kept: the budget lowers nothing the admin set.
    assert.equal(verdict.estimate.gpuMemoryUtilization, 0.5);
    assert.equal(verdict.estimate.unifiedBytes, Math.round(0.5 * PT) + VLLM_RUNNER_RAM_BYTES);
});

test('LL.vllm-small-budget-cannot-hold-minimum-share', () => {
    // 12 GiB less the runner's 3 GiB is 0.07 of the pool: not even the smallest share, whatever the model.
    const verdict = vllm({ total: PT, available: 112 * GIB, max: 12 * GIB, current: 0 });
    assert.equal(verdict.status, 'incompatible');
    assert.equal(verdict.reasonCode, 'budget_exceeded');
    assert.match(verdict.reason, /at least a 0\.1 share of the pool \(12\.0 GiB\) plus 3\.0 GiB for its runner; this agent's Ploinky RAM budget is 12\.0 GiB\./);
    // r4 nit 4: a model too large for that share still reports the budget, not a fit message.
    const big = hfModel({ id: 'big-weights-model', weights: 14 * GIB });
    const large = vllm({ total: PT, available: 112 * GIB, max: 12 * GIB, current: 0 }, {}, big);
    assert.equal(large.status, 'incompatible');
    assert.equal(large.reasonCode, 'budget_exceeded');
    assert.doesNotMatch(large.reason, /can have at most/);
    // Without a budget a pool too small for 0.1 keeps the existing reserve message path.
    const tiny = vllm({ total: 19 * GIB, available: 19 * GIB });
    assert.equal(tiny.status, 'incompatible');
    assert.equal(tiny.reasonCode, undefined);
    assert.match(tiny.reason, /can have at most/);
});

test('LL.vllm-share-below-minimum-names-what-binds', () => {
    // The budget's headroom binds: 6 GiB left of 64 GiB, with the host free.
    const headroom = vllm({ total: PT, available: 112 * GIB, max: 64 * GIB, current: 58 * GIB });
    assert.equal(headroom.status, 'insufficient-now');
    assert.equal(headroom.reasonCode, 'budget_insufficient_now');
    assert.equal(headroom.reason, `vLLM needs at least a 0.1 share of the pool (${(0.1 * PT / GIB).toFixed(1)} GiB) plus 3.0 GiB for its runner; `
        + "6.0 GiB of this agent's 64.0 GiB budget is free now.");
    // The host's free memory binds: the physical MemAvailable is named, not the capped view (15 GiB of headroom here).
    const host = vllm({ total: PT, available: 20 * GIB, max: 64 * GIB, current: 49 * GIB });
    assert.equal(host.status, 'insufficient-now');
    assert.equal(host.reasonCode, undefined);
    assert.equal(host.reason, 'vLLM needs at least a 0.1 share of the pool with 8.0 GiB kept free; 20.0 GiB is available now.');
    // No budget: the same host text.
    const plain = vllm({ total: PT, available: 20 * GIB });
    assert.equal(plain.reason, host.reason);
    assert.equal(plain.reasonCode, undefined);
});

test('LL.vllm-model-too-large-for-budget-names-the-budget', () => {
    const model = hfModel({ id: 'twenty-gib-model', weights: 20 * GIB });
    // 20 GiB of weights against a 20 GiB budget: the budget, not the reserve, holds the largest share down.
    const budgeted = vllm({ total: PT, available: 112 * GIB, max: 20 * GIB, current: 0 }, {}, model);
    assert.equal(budgeted.status, 'incompatible');
    assert.equal(budgeted.reasonCode, 'budget_exceeded');
    assert.match(budgeted.reason, /this agent's Ploinky RAM budget is 20\.0 GiB, which leaves vLLM at most about [\d.]+ GiB of this pool/);
    // The reserve binds when the budget is larger: the text and the missing code are the base's.
    const reserve = hfModel({ id: 'twenty-one-gib-model', weights: 21 * GIB });
    const bound = vllm({ total: 40 * GIB, available: 40 * GIB, max: 30 * GIB, current: 0 }, {}, reserve);
    assert.equal(bound.status, 'incompatible');
    assert.equal(bound.reasonCode, undefined);
    assert.match(bound.reason, /vLLM can have at most about [\d.]+ GiB of this pool/);
    assert.doesNotMatch(bound.reason, /RAM budget/);
    // The 0.9 cap binds, not the budget: 0.92 from the budget, 0.98 from the reserve, 0.9 as the ceiling.
    const total = Math.round(1259.5 * GIB);
    const capped = hfModel({ id: 'one-thousand-gib-model', weights: 1140 * GIB });
    const cap = vllm({ total, available: total, max: Math.round(1168.8 * GIB), current: 0 }, {}, capped);
    assert.equal(cap.status, 'incompatible');
    assert.equal(cap.reasonCode, undefined);
    assert.match(cap.reason, /vLLM can have at most about 1133\.\d GiB of this pool/);
    assert.doesNotMatch(cap.reason, /RAM budget/);
});

test('LL.vllm-max-num-seqs-unusable-sizes-one-sequence', () => {
    // A direct caller may omit maxNumSeqs (the schema supplies 1); a value that is not a positive number sizes one sequence, never NaN.
    const params = { maxModelLen: 4096, kvCacheDtype: 'auto', gpuMemoryUtilization: null };
    const host = { totalBytes: PT, availableBytes: 112 * GIB };
    const one = admitUnifiedVllm({ model: QWEN, source: QWEN.sources.hf, params: { ...params, maxNumSeqs: 1 }, memory: host, disk: { freeBytes: 400 * GIB } }, admissionResult);
    for (const maxNumSeqs of [undefined, 0, -1, Number.NaN, Number.POSITIVE_INFINITY, '4']) {
        const verdict = admitUnifiedVllm({ model: QWEN, source: QWEN.sources.hf, params: { ...params, maxNumSeqs }, memory: host, disk: { freeBytes: 400 * GIB } },
            admissionResult);
        const label = String(maxNumSeqs);
        assert.equal(verdict.estimate.sizedBytes, 1855224363, label);
        assert.equal(verdict.estimate.sizedBytes, one.estimate.sizedBytes, label);
        assert.ok(Number.isFinite(verdict.estimate.gpuMemoryUtilization), label);
        assert.equal(verdict.estimate.gpuMemoryUtilization, one.estimate.gpuMemoryUtilization, label);
        assert.equal(verdict.status, 'ok', verdict.reason);
    }
});

test('the capped memory view keeps the physical figures and the budget beside it', () => {
    const observation = observeMemoryBudget({ maxText: `${32 * GIB}\n`, currentText: `${5 * GIB}\n` });
    const view = effectiveMemory({ totalBytes: PT, availableBytes: 100 * GIB }, observation);
    assert.equal(view.totalBytes, 32 * GIB);
    assert.equal(view.availableBytes, 27 * GIB);
    assert.equal(view.physicalTotalBytes, PT);
    assert.equal(view.physicalAvailableBytes, 100 * GIB);
    assert.equal(view.budgetBytes, 32 * GIB);
    assert.equal(view.budgetHeadroomBytes, 27 * GIB);
    // Without a budget the original object is returned, as before.
    const plain = { totalBytes: PT, availableBytes: 100 * GIB };
    assert.equal(effectiveMemory(plain, observeMemoryBudget({ maxText: 'max\n', currentText: '1\n' })), plain);
});

test('a unified llama.cpp estimate reports the budget-capped pool and the physical pool', () => {
    const budget = 40 * GIB;
    const verdict = llama({ max: budget, current: 0 });
    assert.equal(verdict.estimate.poolBytes, budget);
    assert.equal(verdict.estimate.physicalPoolBytes, PT);
    assert.equal(llama({}).estimate.poolBytes, PT);
    assert.equal(llama({}).estimate.physicalPoolBytes, PT);
});

// ---- the controller: refusals launch nothing, and the spec-97 invariants hold.

function controllerAt(t, { total = PT, available = 112 * GIB, max = null, current = 0, model = hfModel({ seed: true }) } = {}) {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-unified-budget-'));
    const dataDir = path.join(root, 'data');
    const weights = path.join(root, 'weights.gguf');
    fs.writeFileSync(weights, 'x');
    const started = [];
    const observation = () => observeMemoryBudget({ maxText: max === null ? 'max\n' : `${max}\n`, currentText: `${current}\n` });
    const controller = createController({
        dataDir,
        env: { PATH: '/usr/bin', LOCAL_LLM_VLLM_UNIFIED: 'experimental', LOCAL_LLM_RUN_ROOT: path.join(root, 'opt', 'runners') },
        seedCatalog: [...SEED, model],
        stateStore: createStateStore({ dataDir }),
        snapshot: async () => attachMemoryBudget({ gpu: structuredClone(GB10), memory: { totalBytes: total, availableBytes: available },
            disk: { freeBytes: 400 * GIB }, cpus: 20 }, observation()),
        download: async () => ({ status: 'complete', path: weights, bytesTransferred: 0 }),
        inspect: async () => ({ state: 'absent', bytes: 0 }),
        remove: async () => 0,
        inspectSnapshot: async () => ({ state: 'complete', bytes: 1 }),
        downloadSnapshot: async () => ({ status: 'complete', bytesTransferred: 0 }),
        verify: async () => ({ notes: [] }),
        verifySnapshot: async () => ({ notes: [] }),
        installer: {
            installable: () => true,
            describe: async () => ({ installed: true, runnable: true, version: 'test', totalBytes: 1, files: 1, cache: { state: 'complete' }, licence: {} }),
            ensureRunnable: async () => ({ rebuilt: false, seconds: 0, bytes: 1 }),
            entryFor: (id) => ({ id, version: 'test' }),
            pathsFor: () => ({ runDir: path.join(root, 'opt', 'runners', 'run') }),
        },
        startRunner({ command, args, env }) {
            let running = true;
            let resolveExit;
            const handle = {
                pid: 6000 + started.length, command, args, env,
                exited: new Promise((resolve) => { resolveExit = resolve; }),
                get running() { return running; },
                async stop() { running = false; resolveExit({ code: 0, signal: 'SIGTERM', error: null }); return handle.exited; },
                async kill() { running = false; resolveExit({ code: null, signal: 'SIGKILL', error: null }); return handle.exited; },
            };
            started.push(handle);
            return handle;
        },
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({}) }),
        apiKeyFactory: () => KEY,
        detectRunner: (runner) => ({ installed: runner.supported, version: runner.pinnedVersion ?? 'test', reason: null }),
        imageContract: null,
        fileExists: () => true,
        sharedModelsRoot: null,
        shmDir: path.join(root, 'shm'),
        readMemory: () => ({ totalBytes: total, availableBytes: available }),
        readPressure: () => 0,
        readBudget: observation,
        unifiedGuardMs: 60_000,
        dropCache: () => true,
        pollMs: 2,
        stopGraceMs: 20,
    });
    t.after(async () => { await controller.stop(); fs.rmSync(root, { recursive: true, force: true }); });
    return { controller, started, model };
}

async function until(predicate, label, timeoutMs = 3000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (await predicate()) return;
        await new Promise((resolve) => setTimeout(resolve, 3));
    }
    assert.fail(`timed out waiting for ${label}`);
}

test('LL.vllm-budget-refusals-launch-nothing', async (t) => {
    const small = controllerAt(t, { max: 12 * GIB });
    await assert.rejects(small.controller.run({ requestId: 'request-small-budget', modelId: small.model.id, runnerId: 'vllm' }), (error) => {
        assert.equal(error.code, 'admission_incompatible');
        assert.equal(error.details.admission.reasonCode, 'budget_exceeded');
        assert.match(error.message, /at least a 0\.1 share of the pool/);
        return true;
    });
    assert.equal(small.started.length, 0, 'no runner started');
    const over = controllerAt(t, { max: Math.round(0.25 * PT), current: GIB });
    await assert.rejects(over.controller.run({ requestId: 'request-over-budget', modelId: over.model.id, runnerId: 'vllm', params: { gpuMemoryUtilization: 0.5 } }), (error) => {
        assert.equal(error.code, 'admission_incompatible');
        assert.equal(error.details.admission.reasonCode, 'budget_exceeded');
        assert.match(error.message, /this agent's Ploinky RAM budget is/);
        return true;
    });
    assert.equal(over.started.length, 0, 'no runner started');
});

// Explorer spec 97 (dgx-spark-vllm): admin share 0.1, maxModelLen 4096, no budget.
test('LL.spec-97-invariants: the preview, the budget bytes and the ready deployment keep the admin share 0.1', async (t) => {
    const { controller, started, model } = controllerAt(t);
    const params = { gpuMemoryUtilization: 0.1, maxModelLen: 4096 };
    const { preview } = await controller.overview({ preview: { modelId: model.id, runnerId: 'vllm', params } });
    assert.equal(preview.admission.status, 'ok', preview.admission.reason);
    const { estimate } = preview.admission;
    // 1. experimental, share 0.1, and poolBytes is the memory's total
    assert.equal(estimate.experimental, true);
    assert.equal(estimate.gpuMemoryUtilization, 0.1);
    assert.equal(estimate.poolBytes, PT);
    // 2. the budget is that share of the physical total
    assert.equal(estimate.budgetBytes, Math.round(0.1 * PT));
    // 3. the ready deployment keeps both
    assert.equal((await controller.run({ requestId: 'request-spec-97', modelId: model.id, runnerId: 'vllm', params })).accepted, true);
    await until(() => controller.state.deployment?.phase === 'ready' || controller.state.deployment?.phase === 'error', 'settled');
    assert.equal(controller.state.deployment.phase, 'ready', controller.state.deployment.error);
    assert.equal(controller.state.deployment.params.gpuMemoryUtilization, 0.1);
    assert.equal(controller.state.deployment.admission.estimate.gpuMemoryUtilization, 0.1);
    assert.equal(started.length, 1);
    assert.equal(started[0].args[started[0].args.indexOf('--gpu-memory-utilization') + 1], '0.1');
});

test('LL.spec-97-invariants: an admin share is never lowered by the model-sized default', async (t) => {
    const { controller, model } = controllerAt(t);
    const { preview } = await controller.overview({ preview: { modelId: model.id, runnerId: 'vllm', params: { gpuMemoryUtilization: 0.3 } } });
    assert.equal(preview.admission.estimate.gpuMemoryUtilization, 0.3);
    assert.equal(preview.admission.estimate.budgetBytes, Math.round(0.3 * PT));
});
