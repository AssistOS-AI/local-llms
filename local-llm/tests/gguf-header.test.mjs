// The bounded GGUF header reader and the sizing read from it (DS002). The
// bytes are untrusted: every bound is exercised with a claim that only that
// bound refuses, and a truncation at any offset must be invalid_gguf.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { GGUF_LIMITS, createGgufHeaderReader, ggufSizing, readGgufHeaderFile } from '../src/controller/ggufHeader.mjs';
import { GGUF_TYPE as T, ggufBytes, modelPairs, u32, u64 } from './gguf-fixture.mjs';

const isInvalidGguf = (error) => error?.name === 'LocalLlmError' && error.code === 'invalid_gguf';

function feed(reader, bytes, step) {
    let state = 'more';
    for (let offset = 0; offset < bytes.length && state === 'more'; offset += step) {
        state = reader.push(bytes.subarray(offset, Math.min(bytes.length, offset + step)));
    }
    return state;
}

// Every allocation a block asks of Buffer, so a claim that becomes an allocation shows.
function watchAllocations(work) {
    const sizes = [];
    const originals = {};
    for (const name of ['alloc', 'allocUnsafe', 'allocUnsafeSlow']) {
        originals[name] = Buffer[name];
        Buffer[name] = (size, ...rest) => { sizes.push(size); return originals[name].call(Buffer, size, ...rest); };
    }
    const concat = Buffer.concat;
    Buffer.concat = (list, total) => { sizes.push(total ?? list.reduce((sum, part) => sum + part.length, 0)); return concat.call(Buffer, list, total); };
    try {
        work();
    } finally {
        Object.assign(Buffer, originals);
        Buffer.concat = concat;
    }
    return sizes;
}

const PAIRS = [
    ...modelPairs({ layers: 4, heads: 8, kvHeads: 2, embedding: 512, context: 4096, experts: 8 }),
    ['t.u8', T.u8, 200], ['t.i8', T.i8, -5], ['t.u16', T.u16, 65535], ['t.i16', T.i16, -300],
    ['t.u32', T.u32, 4_000_000_000], ['t.i32', T.i32, -70000], ['t.f32', T.f32, 0.5], ['t.bool', T.bool, true],
    ['t.u64', T.u64, 2 ** 40], ['t.i64', T.i64, -(2 ** 40)], ['t.f64', T.f64, 1.25],
    ['t.big', T.u64, 2n ** 63n],
    ['t.kvlist', T.array, { type: T.u32, items: [2, 2, 0, 4] }],
    ['t.flags', T.array, { type: T.bool, items: [true, false] }],
    ['t.tokens', T.array, { type: T.string, items: ['a', 'bb', 'ccc'] }],
    ['t.nested', T.array, { type: T.array, items: [{ type: T.u32, items: [1, 2] }, { type: T.string, items: ['x'] }] }],
    ['__proto__', T.u32, 7],
];

test('the header reader parses GGUF v3 key-values and stops before the tensor data', () => {
    const tail = Buffer.alloc(200_000, 0xee);
    const { bytes, headerLength } = ggufBytes({ tensorCount: 291, kv: PAIRS, tail });
    const chunkings = [1, 7, 4096, bytes.length];
    for (const step of chunkings) {
        const reader = createGgufHeaderReader();
        // One byte short of the last pair is not done; the last byte completes it.
        assert.equal(feed(reader, bytes.subarray(0, headerLength - 1), step), 'more', `step ${step}`);
        assert.equal(reader.push(bytes.subarray(headerLength - 1, headerLength)), 'done');
        assert.equal(reader.bytes, headerLength, 'only the header was consumed');
        // The tensor data after it is never read, whatever is pushed.
        assert.equal(reader.push(tail), 'done');
        assert.equal(reader.bytes, headerLength);
        const { version, tensorCount, kv } = reader.end();
        assert.equal(version, 3);
        assert.equal(tensorCount, 291);
        assert.equal(Object.getPrototypeOf(kv), null, 'a key named __proto__ is only a key');
        assert.equal(kv.__proto__, 7);
        assert.equal(({}).polluted, undefined);
        assert.equal(kv['general.architecture'], 'qwen2');
        assert.equal(kv['qwen2.block_count'], 4);
        assert.equal(kv['qwen2.expert_count'], 8);
        assert.deepEqual([kv['t.u8'], kv['t.i8'], kv['t.u16'], kv['t.i16'], kv['t.u32'], kv['t.i32']], [200, -5, 65535, -300, 4_000_000_000, -70000]);
        assert.deepEqual([kv['t.f32'], kv['t.bool'], kv['t.u64'], kv['t.i64'], kv['t.f64']], [0.5, true, 2 ** 40, -(2 ** 40), 1.25]);
        assert.equal(kv['t.big'], 2n ** 63n, 'a 64-bit value beyond 2^53 stays exact, as a BigInt');
        assert.deepEqual(kv['t.kvlist'], [2, 2, 0, 4]);
        assert.deepEqual(kv['t.flags'], [true, false]);
        assert.deepEqual(kv['t.tokens'], { skipped: true, length: 3 }, 'a string array is read past by length');
        assert.deepEqual(kv['t.nested'], { skipped: true, length: 2 }, 'a nested array is read past');
    }
    // Version 2 is accepted; 1 and 4 are not.
    assert.equal(createGgufHeaderReader().push(ggufBytes({ version: 2, kv: [] }).bytes), 'done');
    for (const version of [0, 1, 4, 0x03000000]) {
        assert.throws(() => createGgufHeaderReader().push(ggufBytes({ version }).bytes), isInvalidGguf, String(version));
    }
    assert.throws(() => createGgufHeaderReader().push(Buffer.from('GGUX\u0003\u0000\u0000\u0000')), isInvalidGguf, 'bad magic');
    // A key that appears twice, or a value of an unknown type, is not a GGUF header.
    assert.throws(() => createGgufHeaderReader().push(ggufBytes({ kv: [['a.b', T.u32, 1], ['a.b', T.u32, 2]] }).bytes), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader().push(ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(99)]), kvCount: 1 }).bytes), isInvalidGguf);
    // Once a reader failed it keeps failing; push rejects anything that is not bytes.
    const broken = createGgufHeaderReader();
    assert.throws(() => broken.push(Buffer.from('nope, not a GGUF file')), isInvalidGguf);
    assert.throws(() => broken.push(Buffer.alloc(1)), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader().push('GGUF'), isInvalidGguf);
});

test('a numeric array is kept up to 4,096 entries and read past beyond that, so a header cannot become a large retained list', () => {
    const items = (count) => Array.from({ length: count }, (_, index) => index % 251);
    const { bytes } = ggufBytes({ kv: [['a.kept', T.array, { type: T.u8, items: items(4096) }], ['a.long', T.array, { type: T.u8, items: items(4097) }], ['a.last', T.u32, 9]] });
    const reader = createGgufHeaderReader();
    assert.equal(reader.push(bytes), 'done');
    const { kv } = reader.end();
    assert.equal(kv['a.kept'].length, 4096);
    assert.deepEqual(kv['a.long'], { skipped: true, length: 4097 });
    assert.equal(kv['a.last'], 9, 'the pair after a skipped array is read from the right place');
});

test('the header reader refuses counts, lengths and nesting beyond its bounds without allocating for them', () => {
    const MIB = 1024 * 1024;
    // Each claim is beyond exactly one bound: the smaller ones lie under the 32 MiB total, so only their own bound refuses them.
    const claims = {
        'kv_count of 2^63-1': ggufBytes({ kvCount: 2n ** 63n - 1n }).bytes,
        'kv_count of 4097 (the bound is 4096)': ggufBytes({ kvCount: 4097 }).bytes,
        'tensor count beyond 2^53': ggufBytes({ tensorCount: 2n ** 63n - 1n }).bytes,
        'a string length of 2^40': ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.string), u64(2 ** 40)]), kvCount: 1 }).bytes,
        'a string length of 5 MiB (the bound is 4 MiB)': ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.string), u64(5 * MIB)]), kvCount: 1 }).bytes,
        'an array count of 10^9': ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.array), u32(T.u8), u64(1e9)]), kvCount: 1 }).bytes,
        'an array count of 5,000,000 (the bound is 4,000,000)': ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.array), u32(T.u8), u64(5_000_000)]), kvCount: 1 }).bytes,
        'a string array with a string of 2^40': ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.array), u32(T.string), u64(1), u64(2 ** 40)]), kvCount: 1 }).bytes,
        'a key of 300 bytes': ggufBytes({ rawPairs: Buffer.concat([u64(300), Buffer.alloc(300, 0x61), u32(T.u32), u32(1)]), kvCount: 1 }).bytes,
        'an empty key': ggufBytes({ rawPairs: Buffer.concat([u64(0), u32(T.u32), u32(1)]), kvCount: 1 }).bytes,
        'a key length of 2^40': ggufBytes({ rawPairs: u64(2 ** 40), kvCount: 1 }).bytes,
        'arrays nested 3 deep': ggufBytes({
            rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.array), u32(T.array), u64(1), u32(T.array), u64(1), u32(T.array), u64(1), u32(T.u8), u64(0)]),
            kvCount: 1,
        }).bytes,
        'an unknown array element type': ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.array), u32(77), u64(1)]), kvCount: 1 }).bytes,
    };
    const sizes = watchAllocations(() => {
        for (const [what, bytes] of Object.entries(claims)) {
            const reader = createGgufHeaderReader();
            assert.throws(() => reader.push(bytes), isInvalidGguf, what);
            assert.ok(reader.bytes < 200, `${what}: consumed ${reader.bytes} bytes`);
        }
    });
    assert.ok(Math.max(0, ...sizes) < 4096, `no allocation follows a claim (largest ${Math.max(0, ...sizes)} bytes)`);
    // Two levels of nesting are allowed and read past.
    const twoDeep = ggufBytes({ kv: [['k', T.array, { type: T.array, items: [{ type: T.u32, items: [1] }] }]] }).bytes;
    assert.equal(createGgufHeaderReader().push(twoDeep), 'done');
    // A claim within the bounds is taken only as its bytes arrive: 10 bytes of a 4 MiB string hold no 4 MiB.
    const claimed = ggufBytes({ rawPairs: Buffer.concat([u64(1), Buffer.from('k'), u32(T.string), u64(4 * MIB), Buffer.alloc(10, 0x61)]), kvCount: 1 }).bytes;
    const held = watchAllocations(() => assert.equal(createGgufHeaderReader().push(claimed), 'more'));
    assert.ok(Math.max(0, ...held) < 4096, `a legitimate 4 MiB claim allocates nothing up front (largest ${Math.max(0, ...held)})`);
    const reader = createGgufHeaderReader();
    reader.push(claimed);
    assert.throws(() => reader.end(), isInvalidGguf);
    // The total: whatever the parser consumes stays under 32 MiB. Nine arrays of 4,000,000 bytes are each within their
    // own bounds; the ninth takes the header past 32 MiB.
    const array = Buffer.concat([u64(1), Buffer.from('k'), u32(T.array), u32(T.u8), u64(4_000_000)]);
    const total = createGgufHeaderReader();
    total.push(ggufBytes({ kvCount: 9 }).bytes);
    const zeros = Buffer.alloc(MIB);
    let pushed = 0;
    assert.throws(() => {
        for (let index = 0; index < 9; index += 1) {
            total.push(Buffer.concat([u64(2), Buffer.from(`k${index}`), u32(T.array), u32(T.u8), u64(4_000_000)]));
            for (let sent = 0; sent < 4_000_000; sent += MIB) { total.push(zeros.subarray(0, Math.min(MIB, 4_000_000 - sent))); pushed += 1; }
        }
    }, isInvalidGguf);
    assert.ok(total.bytes <= GGUF_LIMITS.maxBytes, `consumed ${total.bytes}`);
    assert.ok(array.length > 0 && pushed >= 32, 'the reader was fed past 32 MiB before it refused');
    // The bounds are the reader's options: a smaller total refuses a small header.
    const small = ggufBytes({ kv: [['general.name', T.string, 'x'.repeat(300)]] }).bytes;
    assert.equal(createGgufHeaderReader().push(small), 'done');
    assert.throws(() => createGgufHeaderReader({ maxBytes: 128 }).push(small), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader({ maxStringBytes: 200 }).push(small), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader({ maxKv: 0 }).push(small), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader({ maxKeyBytes: 4 }).push(small), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader({ maxDepth: 1 }).push(twoDeep), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader({ maxArrayCount: 1 }).push(ggufBytes({ kv: [['k', T.array, { type: T.u8, items: [1, 2] }]] }).bytes), isInvalidGguf);
});

test('a header truncated at any offset is invalid_gguf', () => {
    const { bytes, headerLength } = ggufBytes({ tensorCount: 12, kv: PAIRS, tail: Buffer.alloc(64, 1) });
    for (let offset = 0; offset < headerLength; offset += 1) {
        // One chunk, then the same bytes one at a time: the verdict does not depend on how they arrive.
        for (const step of [offset || 1, 1]) {
            const reader = createGgufHeaderReader();
            assert.equal(feed(reader, bytes.subarray(0, offset), step), 'more', `offset ${offset} step ${step}`);
            assert.throws(() => reader.end(), isInvalidGguf, `offset ${offset} step ${step}`);
        }
    }
    const whole = createGgufHeaderReader();
    assert.equal(whole.push(bytes.subarray(0, headerLength)), 'done');
    assert.equal(whole.end().tensorCount, 12);
    // An empty input, and a result asked for too early, are the same verdict.
    assert.throws(() => createGgufHeaderReader().end(), isInvalidGguf);
    assert.throws(() => createGgufHeaderReader().result(), isInvalidGguf);
    // Truncation inside a claim that is only a string array or a numeric array skipped by length is truncation too.
    const skipped = ggufBytes({ kv: [['a', T.array, { type: T.string, items: ['x', 'yy'] }], ['b', T.array, { type: T.u64, items: [1, 2, 3] }]] }).bytes;
    for (let offset = 0; offset < skipped.length; offset += 1) {
        const reader = createGgufHeaderReader();
        assert.equal(reader.push(skipped.subarray(0, offset)), 'more', String(offset));
        assert.throws(() => reader.end(), isInvalidGguf, String(offset));
    }
});

test('readGgufHeaderFile reads a header from a file in chunks, stops before the tensor data, and refuses files without one', async (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'local-llm-gguf-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const file = path.join(dir, 'model.gguf');
    const { bytes, headerLength } = ggufBytes({ kv: PAIRS, tail: Buffer.alloc(3 * 1024 * 1024, 0xee) });
    fs.writeFileSync(file, bytes);
    let read = 0;
    const fsApi = { promises: { async open(target, mode) {
        const handle = await fs.promises.open(target, mode);
        return { read: async (...args) => { const out = await handle.read(...args); read += out.bytesRead; return out; }, close: () => handle.close() };
    } } };
    const { kv, tensorCount } = await readGgufHeaderFile(file, { fsApi, chunkBytes: 4096 });
    assert.equal(kv['qwen2.block_count'], 4);
    assert.equal(tensorCount, 0);
    assert.ok(read >= headerLength && read < headerLength + 2 * 4096, `read ${read} of a ${bytes.length}-byte file with a ${headerLength}-byte header`);
    fs.writeFileSync(path.join(dir, 'cut.gguf'), bytes.subarray(0, headerLength - 3));
    await assert.rejects(() => readGgufHeaderFile(path.join(dir, 'cut.gguf')), isInvalidGguf);
    fs.writeFileSync(path.join(dir, 'text.gguf'), 'this is not a model');
    await assert.rejects(() => readGgufHeaderFile(path.join(dir, 'text.gguf')), isInvalidGguf);
    fs.writeFileSync(path.join(dir, 'empty.gguf'), '');
    await assert.rejects(() => readGgufHeaderFile(path.join(dir, 'empty.gguf')), isInvalidGguf);
});

test('ggufSizing: layers, training context, f16 KV bytes per token and experts; head_count_kv defaults to head_count', () => {
    const read = (options) => {
        const { bytes } = ggufBytes({ kv: modelPairs(options) });
        const reader = createGgufHeaderReader();
        assert.equal(reader.push(bytes), 'done');
        return ggufSizing(reader.end().kv);
    };
    // Qwen2.5-0.5B: 24 layers, 14 heads, 2 KV heads, 896 wide: a 64-wide head, so 24 x 2 x (64 + 64) x 2 bytes.
    assert.deepEqual(read({}), { arch: 'qwen2', layers: 24, contextLength: 32768, kvBytesPerToken: 12288, architecture: 'dense', notes: [] });
    // No head_count_kv: every head has its own KV (no grouped-query attention).
    assert.equal(read({ kvHeads: null, layers: 2, heads: 4, embedding: 256 }).kvBytesPerToken, 2 * 4 * (64 + 64) * 2);
    // Experts make it a mixture of experts; zero experts do not.
    assert.equal(read({ experts: 32 }).architecture, 'moe');
    assert.equal(read({ experts: 0 }).architecture, 'dense');
    // The key and value sizes, when the header states them, replace the embedding over the head count.
    const explicit = ggufSizing({
        'general.architecture': 'gptoss', 'gptoss.block_count': 24, 'gptoss.context_length': 131072, 'gptoss.embedding_length': 2880,
        'gptoss.attention.head_count': 64, 'gptoss.attention.head_count_kv': 8, 'gptoss.attention.key_length': 64, 'gptoss.attention.value_length': 64,
    });
    assert.equal(explicit.kvBytesPerToken, 24 * 8 * 128 * 2);
    assert.equal(explicit.contextLength, 131072);
    // Each defaults separately: only the key size is stated, so the value size is the embedding over the head count.
    const onlyKey = ggufSizing({
        'general.architecture': 'x', 'x.block_count': 2, 'x.embedding_length': 512, 'x.attention.head_count': 8, 'x.attention.key_length': 32,
    });
    assert.equal(onlyKey.kvBytesPerToken, 2 * 8 * (32 + 64) * 2);
    // A per-layer list of KV heads is summed layer by layer; a layer with none adds nothing.
    const listed = ggufSizing({
        'general.architecture': 'x', 'x.block_count': 4, 'x.embedding_length': 512, 'x.attention.head_count': [8, 8, 8, 0],
        'x.attention.head_count_kv': [2, 0, 4, 0],
    });
    assert.equal(listed.kvBytesPerToken, (2 + 4) * (64 + 64) * 2);
    // The architecture name is the key prefix, and only a plain name is accepted.
    for (const arch of [undefined, '', 'Has Space', 'UPPER', 'a'.repeat(41), 'dot.ted', 7]) {
        assert.throws(() => ggufSizing({ 'general.architecture': arch }), isInvalidGguf, String(arch));
    }
    assert.throws(() => ggufSizing(null), isInvalidGguf);
    // Values outside their ranges are left out, each with a note.
    const outside = ggufSizing({
        'general.architecture': 'x', 'x.block_count': 1025, 'x.context_length': 511, 'x.embedding_length': 512, 'x.attention.head_count': 8,
    });
    assert.deepEqual([outside.layers, outside.contextLength, outside.kvBytesPerToken], [null, null, null]);
    assert.equal(outside.notes.length, 2);
    assert.equal(ggufSizing({ 'general.architecture': 'x', 'x.context_length': 2 ** 22 + 1, 'x.block_count': 0 }).contextLength, null);
    assert.equal(ggufSizing({ 'general.architecture': 'x', 'x.context_length': 512, 'x.block_count': 1 }).contextLength, 512);
});

test('ggufSizing: hybrid, recurrent and latent-attention architectures leave the KV size to the default', () => {
    const base = { 'x.block_count': 4, 'x.context_length': 4096, 'x.embedding_length': 512, 'x.attention.head_count': 8, 'x.attention.head_count_kv': 2 };
    assert.equal(ggufSizing({ 'general.architecture': 'x', ...base }).kvBytesPerToken, 4 * 2 * 128 * 2, 'control: the same model without the marker');
    // Any <arch>.ssm.* key (Mamba, hybrids), and the latent-attention rank (DeepSeek, MLA): null with a note naming the key.
    for (const marker of ['x.ssm.state_size', 'x.ssm.conv_kernel', 'x.attention.kv_lora_rank']) {
        const sizing = ggufSizing({ 'general.architecture': 'x', ...base, [marker]: 16 });
        assert.equal(sizing.kvBytesPerToken, null, marker);
        assert.ok(sizing.notes.some((note) => note.includes(marker)), marker);
        // The layers and the context are still read.
        assert.deepEqual([sizing.layers, sizing.contextLength], [4, 4096]);
    }
    // A marker of another architecture's prefix is not this architecture's.
    assert.equal(ggufSizing({ 'general.architecture': 'x', ...base, 'y.ssm.state_size': 16 }).kvBytesPerToken, 4 * 2 * 128 * 2);
    // A per-layer list whose length differs from block_count, or that is too long to keep, is not summed.
    for (const list of [[2, 2, 2], [2, 2, 2, 2, 2], { skipped: true, length: 100000 }, [2, 2, 'two', 2], [2, 2, -1, 2]]) {
        const sizing = ggufSizing({ 'general.architecture': 'x', ...base, 'x.attention.head_count_kv': list });
        assert.equal(sizing.kvBytesPerToken, null, JSON.stringify(list));
        assert.ok(sizing.notes.length > 0);
    }
    // No head count, or no head size to divide the embedding by: nothing to compute.
    const { 'x.attention.head_count': _heads, ...noHeads } = base;
    assert.equal(ggufSizing({ 'general.architecture': 'x', ...noHeads }).kvBytesPerToken, null);
    const { 'x.embedding_length': _embedding, ...noEmbedding } = base;
    assert.equal(ggufSizing({ 'general.architecture': 'x', ...noEmbedding }).kvBytesPerToken, null);
    // A result outside 1 to 2^30 is null: 1024 layers of 65536 KV heads of 2 x 1024 x 2 bytes is 2^38.
    const huge = ggufSizing({
        'general.architecture': 'x', 'x.block_count': 1024, 'x.attention.head_count': 65536, 'x.attention.head_count_kv': 65536,
        'x.attention.key_length': 1024, 'x.attention.value_length': 1024,
    });
    assert.equal(huge.kvBytesPerToken, null);
    assert.ok(huge.notes.some((note) => /outside 1 to/.test(note)));
    // The largest result that is accepted and the first that is not: one layer is kvHeads x (2 + 2) x 2 = 8 x kvHeads bytes.
    const edge = (kvHeads) => ggufSizing({
        'general.architecture': 'x', 'x.block_count': 1, 'x.attention.head_count': kvHeads, 'x.attention.head_count_kv': kvHeads,
        'x.attention.key_length': 2, 'x.attention.value_length': 2,
    }).kvBytesPerToken;
    assert.equal(edge(2 ** 27), 2 ** 30);
    assert.equal(edge(2 ** 27 + 1), null);
    assert.equal(edge(0), null, 'no KV heads at all gives 0, which is below 1');
});
