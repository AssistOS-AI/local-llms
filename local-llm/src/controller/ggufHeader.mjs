// A bounded, streaming reader for the start of a GGUF file: the magic, the
// version and the key-value section, which is where the architecture, the
// layer count, the training context and the attention shape live. It stops
// once the last key-value pair is parsed, before the tensor descriptions and
// the tensor data, so a caller reads a few kilobytes to a few megabytes of a
// model file that may be tens of gigabytes.
//
// The bytes are untrusted (a lookup reads them from a repository anyone may
// publish to, a controller download's header only after its sha256 matched).
// Every length and count in the header is checked against a bound before it
// is used, and nothing is allocated for a claim: a string or array is taken
// only as its bytes arrive, a count is never turned into an allocation, and a
// total of `maxBytes` (32 MiB) bounds everything the parser consumes. Any
// violation, a bad magic or version, or a header that ends early at `end()`
// is `LocalLlmError('invalid_gguf')`, never another error type.

import fs from 'node:fs';

import { LocalLlmError } from '../errors.mjs';

const MIB = 1024 * 1024;

/** The bounds of the header reader (DS002). Tests may pass smaller ones. */
export const GGUF_LIMITS = Object.freeze({
    maxBytes: 32 * MIB,
    maxKv: 4096,
    maxKeyBytes: 256,
    maxStringBytes: 4 * MIB,
    maxArrayCount: 4_000_000,
    maxDepth: 2,
    // Numbers kept across every array of the header; arrays past it are read past and appear as skipped.
    maxKeptNumbers: 65_536,
});

// A numeric array longer than this (a per-layer list never is: a model has at
// most 1024 layers) is read past and not stored, and so is any array that would
// take the header past `maxKeptNumbers` kept numbers in all, so a 32 MiB
// header cannot become tens of millions of retained numbers.
const MAX_KEPT_ARRAY = 4096;
// An architecture name is the prefix of its keys ("qwen2.block_count"), so it has no dot; llama.cpp's own
// names use letters, digits, underscores and hyphens ("gpt-oss", "command-r", "falcon-h1").
const ARCH_RE = /^[a-z0-9_-]{1,40}$/;

const TYPE_ARRAY = 9;
const TYPE_STRING = 8;
// GGUF value types 0-12 other than string (8) and array (9): byte size and decoder.
const SCALARS = Object.freeze({
    0: { size: 1, read: (b) => b.readUInt8(0) },
    1: { size: 1, read: (b) => b.readInt8(0) },
    2: { size: 2, read: (b) => b.readUInt16LE(0) },
    3: { size: 2, read: (b) => b.readInt16LE(0) },
    4: { size: 4, read: (b) => b.readUInt32LE(0) },
    5: { size: 4, read: (b) => b.readInt32LE(0) },
    6: { size: 4, read: (b) => b.readFloatLE(0) },
    7: { size: 1, read: (b) => b.readUInt8(0) !== 0 },
    10: { size: 8, read: (b) => safeNumber(b.readBigUInt64LE(0)) },
    11: { size: 8, read: (b) => safeNumber(b.readBigInt64LE(0)) },
    12: { size: 8, read: (b) => b.readDoubleLE(0) },
});

// A 64-bit integer as a number when it is exact, otherwise as the BigInt (which no sizing rule accepts).
function safeNumber(value) {
    return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(value) : value;
}

function invalid(message) {
    return new LocalLlmError('invalid_gguf', message);
}

/**
 * A reader fed with the file's leading bytes in chunks of any size.
 *
 * `push(chunk)` returns `'more'` while the key-value section is incomplete and
 * `'done'` once it is parsed (bytes after it are ignored). `end()` says the
 * bytes are over: it returns `{ version, tensorCount, kv }`, or throws
 * `invalid_gguf` when the header is incomplete. `result()` returns the same
 * once parsing is done. `kv` has no prototype, so a key named `__proto__` is
 * just a key. Scalars and strings are kept; a numeric array of up to 4,096
 * entries is kept as an array (up to 65,536 numbers in all); a longer numeric
 * array, one past that total, a string array and a nested array are read past
 * and appear as `{ skipped: true, length }`.
 */
export function createGgufHeaderReader(options = {}) {
    const limits = Object.freeze({ ...GGUF_LIMITS, ...options });
    const queue = [];
    let head = 0;
    let available = 0;
    let consumed = 0;
    let parsed = null;
    let failure = null;
    let keptNumbers = 0;

    function pull(count) {
        const first = queue[0];
        if (first.length - head >= count) {
            const out = first.subarray(head, head + count);
            head += count;
            if (head === first.length) {
                queue.shift();
                head = 0;
            }
            available -= count;
            return out;
        }
        const parts = [];
        let left = count;
        while (left > 0) {
            const chunk = queue[0];
            const step = Math.min(left, chunk.length - head);
            parts.push(chunk.subarray(head, head + step));
            head += step;
            left -= step;
            if (head === chunk.length) {
                queue.shift();
                head = 0;
            }
        }
        available -= count;
        return Buffer.concat(parts, count);
    }

    function drop(count) {
        let left = count;
        while (left > 0) {
            const chunk = queue[0];
            const step = Math.min(left, chunk.length - head);
            head += step;
            left -= step;
            if (head === chunk.length) {
                queue.shift();
                head = 0;
            }
        }
        available -= count;
    }

    function charge(count) {
        if (consumed + count > limits.maxBytes) throw invalid(`The GGUF header is longer than ${limits.maxBytes} bytes`);
    }

    // The next `count` bytes; waits for them. The total bound is checked against the claim first.
    function* take(count) {
        charge(count);
        while (available < count) yield;
        const bytes = count === 0 ? Buffer.alloc(0) : pull(count);
        consumed += count;
        return bytes;
    }

    // Reads past `count` bytes without keeping them.
    function* skip(count) {
        charge(count);
        let left = count;
        while (left > 0) {
            while (available === 0) yield;
            const step = Math.min(left, available);
            drop(step);
            left -= step;
        }
        consumed += count;
    }

    // A 64-bit length or count, refused above `max` before anything depends on it.
    function* length(what, max) {
        const value = (yield* take(8)).readBigUInt64LE(0);
        if (value > BigInt(max)) throw invalid(`The GGUF ${what} ${value} is beyond the bound of ${max}`);
        return Number(value);
    }

    function* valueType() {
        const type = (yield* take(4)).readUInt32LE(0);
        if (type !== TYPE_STRING && type !== TYPE_ARRAY && !Object.hasOwn(SCALARS, type)) throw invalid(`Unknown GGUF value type ${type}`);
        return type;
    }

    function* stringValue(keep) {
        const size = yield* length('string length', limits.maxStringBytes);
        if (!keep) {
            yield* skip(size);
            return null;
        }
        return (yield* take(size)).toString('utf8');
    }

    function* arrayValue(depth, keep) {
        if (depth > limits.maxDepth) throw invalid(`GGUF arrays are nested more than ${limits.maxDepth} deep`);
        const type = yield* valueType();
        const count = yield* length('array length', limits.maxArrayCount);
        if (type === TYPE_STRING) {
            for (let index = 0; index < count; index += 1) yield* stringValue(false);
            return { skipped: true, length: count };
        }
        if (type === TYPE_ARRAY) {
            for (let index = 0; index < count; index += 1) yield* arrayValue(depth + 1, false);
            return { skipped: true, length: count };
        }
        const { size, read } = SCALARS[type];
        if (!keep || count > MAX_KEPT_ARRAY || keptNumbers + count > limits.maxKeptNumbers) {
            yield* skip(size * count);
            return { skipped: true, length: count };
        }
        keptNumbers += count;
        const bytes = yield* take(size * count);
        const items = new Array(count);
        for (let index = 0; index < count; index += 1) items[index] = read(bytes.subarray(index * size, (index + 1) * size));
        return items;
    }

    function* value(type) {
        if (type === TYPE_STRING) return yield* stringValue(true);
        if (type === TYPE_ARRAY) return yield* arrayValue(1, true);
        return SCALARS[type].read(yield* take(SCALARS[type].size));
    }

    function* parse() {
        if ((yield* take(4)).toString('latin1') !== 'GGUF') throw invalid('Not a GGUF file (bad magic)');
        const version = (yield* take(4)).readUInt32LE(0);
        if (version !== 2 && version !== 3) throw invalid(`Unsupported GGUF version ${version}`);
        const tensorCount = yield* length('tensor count', Number.MAX_SAFE_INTEGER);
        const kvCount = yield* length('key-value count', limits.maxKv);
        const kv = Object.create(null);
        for (let index = 0; index < kvCount; index += 1) {
            const keyBytes = yield* length('key length', limits.maxKeyBytes);
            if (keyBytes === 0) throw invalid('A GGUF key is empty');
            const key = (yield* take(keyBytes)).toString('utf8');
            const type = yield* valueType();
            if (Object.hasOwn(kv, key)) throw invalid('A GGUF key appears twice');
            kv[key] = yield* value(type);
        }
        parsed = Object.freeze({ version, tensorCount, kv });
    }

    const steps = parse();

    function fail(error) {
        failure = error instanceof LocalLlmError && error.code === 'invalid_gguf' ? error : invalid('The GGUF header is malformed');
        queue.length = 0;
        available = 0;
        return failure;
    }

    return Object.freeze({
        push(chunk) {
            if (failure) throw failure;
            if (parsed) return 'done';
            if (!(chunk instanceof Uint8Array)) throw fail(new TypeError('not bytes'));
            if (chunk.length > 0) {
                queue.push(Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength));
                available += chunk.length;
            }
            try {
                if (steps.next().done) {
                    queue.length = 0;
                    available = 0;
                    return 'done';
                }
            } catch (error) {
                throw fail(error);
            }
            return 'more';
        },
        end() {
            if (failure) throw failure;
            if (!parsed) throw fail(invalid('The GGUF header ends before its key-value section does'));
            return parsed;
        },
        result() {
            if (failure) throw failure;
            if (!parsed) throw invalid('The GGUF header is not complete yet');
            return parsed;
        },
        /** The bytes of the header consumed so far. */
        get bytes() { return consumed; },
    });
}

/**
 * The key-value section of the GGUF file at `file`, read in chunks until it is
 * complete, never more than the reader's bound. A file that ends first, or
 * whose header breaks a bound, is `invalid_gguf`.
 */
export async function readGgufHeaderFile(file, { fsApi = fs, chunkBytes = 64 * 1024, ...options } = {}) {
    const reader = createGgufHeaderReader(options);
    const handle = await fsApi.promises.open(file, 'r');
    try {
        const buffer = Buffer.alloc(chunkBytes);
        // A read may return fewer bytes than asked for: the next one starts where this one ended.
        for (let position = 0; ;) {
            const { bytesRead } = await handle.read(buffer, 0, chunkBytes, position);
            if (bytesRead === 0) return reader.end();
            position += bytesRead;
            if (reader.push(Buffer.from(buffer.subarray(0, bytesRead))) === 'done') return reader.result();
        }
    } finally {
        await handle.close();
    }
}

function scalar(kv, key, min, max) {
    const value = Object.hasOwn(kv, key) ? kv[key] : undefined;
    return Number.isSafeInteger(value) && value >= min && value <= max ? value : null;
}

// A per-layer list (head counts) or one number for every layer, as numbers: an array of `layers` numbers, or null.
function perLayer(kv, key, layers, notes) {
    const value = Object.hasOwn(kv, key) ? kv[key] : undefined;
    if (Number.isSafeInteger(value)) return new Array(layers).fill(value);
    if (Array.isArray(value)) {
        if (value.length === layers && value.every((entry) => Number.isSafeInteger(entry) && entry >= 0)) return value;
        notes.push(`${key} does not list one number per layer`);
        return null;
    }
    if (value?.skipped) notes.push(`${key} is too long to be a per-layer list`);
    return null;
}

/**
 * What the memory estimate needs from a GGUF key-value section (DS002): the
 * layers, the training context, the f16 KV cache bytes per token, and whether
 * the model is a mixture of experts. `kvBytesPerToken` is null, with a note,
 * when the arithmetic does not apply (a hybrid, recurrent or latent-attention
 * architecture), a per-layer list does not match the layer count, or the
 * result is outside 1 to 2^30: the estimate then uses its default and says so.
 *
 * A header whose architecture name cannot be used as a key prefix (missing, or
 * anything but letters, digits, underscore and hyphen, up to 40 characters) is
 * a well-formed header this reader cannot size: every value is null, `arch` is
 * null and a note says why. That is never an error, so it cannot fail a pick or
 * a Run; `invalid_gguf` is for headers that are malformed as GGUF.
 */
export function ggufSizing(kv) {
    const read = kv && typeof kv === 'object' ? kv : {};
    const arch = Object.hasOwn(read, 'general.architecture') ? read['general.architecture'] : undefined;
    if (typeof arch !== 'string' || !ARCH_RE.test(arch)) {
        const why = typeof arch === 'string'
            ? 'general.architecture is not a name this agent can read: letters, digits, underscore and hyphen, at most 40 characters'
            : 'general.architecture is missing';
        return { arch: null, layers: null, contextLength: null, kvBytesPerToken: null, architecture: 'dense', notes: [why] };
    }
    const key = (name) => `${arch}.${name}`;
    const notes = [];
    const layers = scalar(read, key('block_count'), 1, 1024);
    if (layers === null) notes.push(`${key('block_count')} is missing or outside 1 to 1024`);
    const contextLength = scalar(read, key('context_length'), 512, 2 ** 22);
    if (contextLength === null) notes.push(`${key('context_length')} is missing or outside 512 to ${2 ** 22}`);
    const experts = Object.hasOwn(read, key('expert_count')) ? read[key('expert_count')] : undefined;
    const architecture = Number.isSafeInteger(experts) && experts > 0 ? 'moe' : 'dense';
    return { arch, layers, contextLength, kvBytesPerToken: kvBytesPerToken(read, arch, layers, notes), architecture, notes };
}

function kvBytesPerToken(kv, arch, layers, notes) {
    const key = (name) => `${arch}.${name}`;
    if (layers === null) return null;
    const latent = Object.keys(kv).find((name) => name.startsWith(`${arch}.ssm.`) || name === key('attention.kv_lora_rank'));
    if (latent) {
        notes.push(`${latent} marks a hybrid, recurrent or latent-attention model, so its KV cache is not sized from the head counts`);
        return null;
    }
    const heads = perLayer(kv, key('attention.head_count'), layers, notes);
    if (heads === null) {
        notes.push(`${key('attention.head_count')} is missing`);
        return null;
    }
    // The KV heads default to the head count (a model without grouped-query attention).
    const kvHeads = Object.hasOwn(kv, key('attention.head_count_kv')) ? perLayer(kv, key('attention.head_count_kv'), layers, notes) : heads;
    if (kvHeads === null) return null;
    const embedding = scalar(kv, key('embedding_length'), 1, 2 ** 31);
    const keyLength = scalar(kv, key('attention.key_length'), 1, 2 ** 20);
    const valueLength = scalar(kv, key('attention.value_length'), 1, 2 ** 20);
    let total = 0;
    for (let layer = 0; layer < layers; layer += 1) {
        if (kvHeads[layer] === 0) continue;
        // Key and value sizes each default to the embedding length over the head count.
        const fallback = embedding !== null && heads[layer] > 0 ? Math.floor(embedding / heads[layer]) : null;
        const keySize = keyLength ?? fallback;
        const valueSize = valueLength ?? fallback;
        if (!(keySize > 0) || !(valueSize > 0)) {
            notes.push(`${key('attention.key_length')} and ${key('embedding_length')} give no head size`);
            return null;
        }
        // f16: two bytes per element, for the keys and the values.
        total += kvHeads[layer] * (keySize + valueSize) * 2;
    }
    if (!Number.isSafeInteger(total) || total < 1 || total > 2 ** 30) {
        notes.push(`the KV cache size computed from the head counts (${total}) is outside 1 to ${2 ** 30}`);
        return null;
    }
    return total;
}
