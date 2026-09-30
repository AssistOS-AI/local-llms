// Builds GGUF bytes for tests: a header (magic, version, counts, key-value
// pairs) followed by anything standing in for the tensor descriptions and data.
// Not a test file itself (the gate runs tests/*.test.mjs only).

export const GGUF_TYPE = Object.freeze({
    u8: 0, i8: 1, u16: 2, i16: 3, u32: 4, i32: 5, f32: 6, bool: 7, string: 8, array: 9, u64: 10, i64: 11, f64: 12,
});

export const u32 = (value) => {
    const bytes = Buffer.alloc(4);
    bytes.writeUInt32LE(value);
    return bytes;
};

export const u64 = (value) => {
    const bytes = Buffer.alloc(8);
    bytes.writeBigUInt64LE(BigInt(value));
    return bytes;
};

export const ggufString = (text) => {
    const body = Buffer.from(text, 'utf8');
    return Buffer.concat([u64(body.length), body]);
};

/** The encoded body of one value: `{ array: { type, items } }` for an array, otherwise `[type, value]`. */
export function ggufValue(type, value) {
    switch (type) {
        case GGUF_TYPE.u8: return Buffer.from([value]);
        case GGUF_TYPE.i8: return Buffer.from([value & 0xff]);
        case GGUF_TYPE.u16: { const b = Buffer.alloc(2); b.writeUInt16LE(value); return b; }
        case GGUF_TYPE.i16: { const b = Buffer.alloc(2); b.writeInt16LE(value); return b; }
        case GGUF_TYPE.u32: return u32(value);
        case GGUF_TYPE.i32: { const b = Buffer.alloc(4); b.writeInt32LE(value); return b; }
        case GGUF_TYPE.f32: { const b = Buffer.alloc(4); b.writeFloatLE(value); return b; }
        case GGUF_TYPE.bool: return Buffer.from([value ? 1 : 0]);
        case GGUF_TYPE.string: return ggufString(value);
        case GGUF_TYPE.u64: return u64(value);
        case GGUF_TYPE.i64: { const b = Buffer.alloc(8); b.writeBigInt64LE(BigInt(value)); return b; }
        case GGUF_TYPE.f64: { const b = Buffer.alloc(8); b.writeDoubleLE(value); return b; }
        case GGUF_TYPE.array: return Buffer.concat([u32(value.type), u64(value.items.length), ...value.items.map((item) => ggufValue(value.type, item))]);
        default: throw new Error(`no encoder for GGUF type ${type}`);
    }
}

/** A key-value pair: [key, type, value]. */
export const ggufPair = ([key, type, value]) => Buffer.concat([ggufString(key), u32(type), ggufValue(type, value)]);

/**
 * `kv` is a list of [key, type, value]. `rawPairs` is appended after them as-is (to write a pair the encoder would
 * not); `kvCount` overrides the count written in the header. Returns the whole file prefix and where the header ends.
 */
export function ggufBytes({ version = 3, tensorCount = 0, kv = [], rawPairs = Buffer.alloc(0), kvCount = kv.length, tail = Buffer.alloc(0) } = {}) {
    const header = Buffer.concat([Buffer.from('GGUF'), u32(version), u64(tensorCount), u64(kvCount), ...kv.map(ggufPair), rawPairs]);
    return { bytes: Buffer.concat([header, tail]), headerLength: header.length };
}

/** The pairs of a typical decoder-only model, each field overridable; `extra` pairs are appended. */
export function modelPairs({
    arch = 'qwen2', layers = 24, heads = 14, kvHeads = 2, embedding = 896, context = 32768, experts = null, name = 'Test Model', extra = [],
} = {}) {
    return [
        ['general.architecture', GGUF_TYPE.string, arch],
        ['general.name', GGUF_TYPE.string, name],
        [`${arch}.block_count`, GGUF_TYPE.u32, layers],
        [`${arch}.context_length`, GGUF_TYPE.u32, context],
        [`${arch}.embedding_length`, GGUF_TYPE.u32, embedding],
        [`${arch}.attention.head_count`, GGUF_TYPE.u32, heads],
        ...(kvHeads === null ? [] : [[`${arch}.attention.head_count_kv`, GGUF_TYPE.u32, kvHeads]]),
        ...(experts === null ? [] : [[`${arch}.expert_count`, GGUF_TYPE.u32, experts]]),
        ...extra,
    ];
}
