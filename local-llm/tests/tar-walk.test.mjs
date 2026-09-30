// Whether a decompressed .tar is whole (DS004): its headers are walked from the start, member by member, and
// it is whole only where a block of zeros is found exactly where a header is due, followed by a second one.
// The tail of a tar is no test: a member whose data ends in zeros, cut at its end, looks like an archive's end.
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { tarProblem } from '../src/controller/runnerInstaller.mjs';

const BLOCK = 512;
const END = Buffer.alloc(2 * BLOCK);
const roundUp = (bytes) => Math.ceil(bytes / BLOCK) * BLOCK;

function tempDir(t, name) {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), `local-llm-${name}-`));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    return dir;
}

// A ustar header, checksum included. `size` is what the header says (a number, or the raw 12-byte text); `fields` overrides a byte range.
function header({ name, size = 0, type = '0', sizeText = null, base256 = false, magic = 'ustar\0', mutate = null }) {
    const block = Buffer.alloc(BLOCK);
    block.write(name, 0, 100, 'utf8');
    block.write('0000644\0', 100);
    block.write('0000000\0', 108);
    block.write('0000000\0', 116);
    if (base256) {
        block[124] = 0x80;
        block.writeBigUInt64BE(BigInt(size), 128);
    } else {
        block.write(sizeText ?? `${size.toString(8).padStart(11, '0')}\0`, 124, 12, 'latin1');
    }
    block.write('00000000000\0', 136);
    block.write('        ', 148);
    block.write(type, 156, 1, 'latin1');
    block.write(`${magic}00`, 257, 8, 'latin1');
    if (mutate) mutate(block);
    const sum = block.reduce((total, byte) => total + byte, 0);
    block.write(`${sum.toString(8).padStart(6, '0')}\0 `, 148, 8, 'latin1');
    return block;
}

// A member: its header and its data padded to whole blocks. `size` is what the header claims, which may differ from the data.
function member(name, data, options = {}) {
    const bytes = Buffer.isBuffer(data) ? data : Buffer.from(data);
    return Buffer.concat([header({ name, size: bytes.length, ...options }), bytes, Buffer.alloc(roundUp(bytes.length) - bytes.length)]);
}

const pax = (records, type = 'x') => {
    const text = Object.entries(records).map(([key, value]) => {
        // A record's length counts itself and is in bytes.
        const body = ` ${key}=${value}\n`;
        const bytes = Buffer.byteLength(body);
        let length = bytes + 1;
        while (String(length).length + bytes !== length) length = String(length).length + bytes;
        return `${length}${body}`;
    }).join('');
    return member('PaxHeader/x', text, { type });
};

const noise = (bytes) => crypto.randomBytes(bytes);
const zeros = (bytes) => Buffer.alloc(bytes);

function problemOf(t, tar, options) {
    const file = path.join(tempDir(t, 'tar-walk'), 'x.tar');
    fs.writeFileSync(file, tar);
    return tarProblem(file, options);
}

// ---------------------------------------------------------------- whole, and cut short

test('a whole tar is accepted whatever its last member ends in, with or without the padding of a record', async (t) => {
    const members = [member('a.dat', noise(3000)), member('empty.txt', ''), member('dir/', '', { type: '5' }), member('zeros.bin', zeros(2048)), member('last-zeros.bin', zeros(1024))];
    const whole = Buffer.concat([...members, END]);
    assert.equal(await problemOf(t, whole), null);
    assert.equal(await problemOf(t, Buffer.concat([whole, zeros(10240 - (whole.length % 10240))])), null, 'padded to a 10 KiB record');
    assert.equal(await problemOf(t, END), null, 'an empty archive');
    assert.equal(await problemOf(t, Buffer.concat([member('x', 'y'), END, noise(700)])), null, 'what follows the end is not read');
});

test('a tar cut where a member whose data ends in zeros ends is refused: the tail of a tar is no test', async (t) => {
    // The verifier's case: a data file, then 2,048 zero bytes, then what the cut throws away.
    const first = member('a.dat', noise(3000));
    const second = member('zeros.bin', zeros(2048));
    const tar = Buffer.concat([first, second, member('later.bin', noise(100)), member('lib/libggml-cpu.so', noise(5000)), END]);
    const cut = first.length + second.length;
    assert.equal(cut, first.length + second.length);
    assert.ok(tar.subarray(cut - 2 * BLOCK, cut).every((byte) => byte === 0), 'the premise: the last two blocks of the cut are zeros');
    assert.equal(cut % BLOCK, 0);
    assert.deepEqual(await problemOf(t, tar.subarray(0, cut)), { cut: 'no end-of-archive blocks' });
    // Every cut of it, at and around every block: only the whole tar (and what follows its end) is accepted.
    const accepted = [];
    for (let at = 0; at <= tar.length; at += 1) {
        if (at % BLOCK > 1 && at % BLOCK < BLOCK - 1) continue;
        if ((await problemOf(t, tar.subarray(0, at))) === null) accepted.push(at);
    }
    assert.deepEqual(accepted, [tar.length], 'only the whole tar');
    assert.deepEqual(await problemOf(t, tar.subarray(0, tar.length - BLOCK)), { cut: 'only one end-of-archive block' });
    assert.deepEqual(await problemOf(t, Buffer.concat([tar.subarray(0, tar.length - 2 * BLOCK), zeros(BLOCK), noise(BLOCK)])), { cut: 'a block of zeros is not followed by a second one' });
    assert.deepEqual(await problemOf(t, Buffer.alloc(0)), { cut: 'no end-of-archive blocks' });
    assert.deepEqual(await problemOf(t, zeros(BLOCK)), { cut: 'only one end-of-archive block' });
});

// ---------------------------------------------------------------- headers that are not headers

test('a header with a wrong checksum, no ustar magic, a size that is not a number or one that runs past the end is refused', async (t) => {
    const good = Buffer.concat([member('a.dat', noise(700)), member('b.dat', noise(700)), END]);
    assert.equal(await problemOf(t, good), null);
    // One byte of the second header's name changed: its checksum no longer adds up.
    const corrupted = Buffer.from(good);
    corrupted[member('a.dat', noise(700)).length + 3] ^= 0x01;
    assert.deepEqual(await problemOf(t, corrupted), { cut: `the header at byte ${member('a.dat', noise(700)).length}: its checksum is wrong` });
    assert.deepEqual(await problemOf(t, Buffer.concat([member('a.dat', 'x', { magic: '\0\0\0\0\0\0' }), END])), { cut: 'the header at byte 0: the ustar magic is missing' });
    assert.deepEqual(await problemOf(t, Buffer.concat([member('a.dat', 'x', { sizeText: 'zzzzzzzzzzz\0' }), END])), { cut: 'the header at byte 0: its size is not a number' });
    assert.deepEqual(await problemOf(t, Buffer.concat([member('a.dat', 'x', { sizeText: '89abcdefghi\0' }), END])), { cut: 'the header at byte 0: its size is not a number' });
    // Data where a header is due (a size that was too small) is no header.
    const small = Buffer.concat([header({ name: 'a.dat', size: 10 }), noise(2 * BLOCK), END]);
    assert.match((await problemOf(t, small)).cut, /^the header at byte 1024: /);
    // A size that runs past the end of the file, and a file that ends inside a header.
    assert.deepEqual(await problemOf(t, Buffer.concat([header({ name: 'a.dat', size: 100000 }), noise(BLOCK), END])), { cut: 'a member runs past the end of the file' });
    assert.deepEqual(await problemOf(t, Buffer.concat([header({ name: 'a.dat', size: 2 ** 40 }), END])), { cut: 'a member runs past the end of the file' });
    assert.deepEqual(await problemOf(t, good.subarray(0, member('a.dat', noise(700)).length + 100)), { cut: 'no end-of-archive blocks' });
    // A tar's data padding cut off the last member: the data is whole, the padding is not.
    assert.deepEqual(await problemOf(t, Buffer.concat([header({ name: 'a.dat', size: 700 }), noise(700)])), { cut: 'a member runs past the end of the file' });
});

// ---------------------------------------------------------------- pax, GNU long names, base 256, and members with no data

test('a pax size override, a global one, and a malformed pax header are honoured as the format says', async (t) => {
    // The member header says 0 bytes; the pax record says 5,000, and the data is 5,000 bytes.
    const overridden = Buffer.concat([pax({ size: 5000 }), member('big.bin', noise(5000), { size: 0 }), END]);
    assert.equal(await problemOf(t, overridden), null);
    // Without the override honoured, the data would be taken for headers: the same bytes with the record dropped are refused.
    assert.ok(await problemOf(t, Buffer.concat([member('big.bin', noise(5000), { size: 0 }), END])));
    // A pax size past the end of the file is a cut; the override is for the next member only.
    assert.deepEqual(await problemOf(t, Buffer.concat([pax({ size: 500000 }), member('big.bin', noise(5000), { size: 0 }), END])), { cut: 'a member runs past the end of the file' });
    assert.equal(await problemOf(t, Buffer.concat([pax({ size: 5000 }), member('big.bin', noise(5000), { size: 0 }), member('after.bin', noise(600)), END])), null);
    // A global record applies to every member from there on, until a local one overrides it.
    assert.equal(await problemOf(t, Buffer.concat([pax({ size: 700 }, 'g'), member('a.bin', noise(700), { size: 0 }), pax({ size: 1200 }), member('b.bin', noise(1200), { size: 0 }), END])), null);
    // Other records (path, mtime, non-ASCII) are read past; a record that is not well formed is a fault.
    assert.equal(await problemOf(t, Buffer.concat([pax({ path: 'über/ünï.txt', mtime: '1.5' }), member('x', noise(10)), END])), null);
    assert.deepEqual(await problemOf(t, Buffer.concat([member('PaxHeader/x', '99 size=5\n', { type: 'x' }), member('x', 'y'), END])), { cut: 'a pax header is malformed' });
    assert.deepEqual(await problemOf(t, Buffer.concat([member('PaxHeader/x', '12 size=abc\n', { type: 'x' }), member('x', 'y'), END])), { cut: 'a pax size is not a number' });
    // A pax header over the limit is refused without being read.
    const big = Buffer.concat([header({ name: 'PaxHeader/x', type: 'x', size: 1024 * 1024 + 1 }), zeros(roundUp(1024 * 1024 + 1)), END]);
    assert.deepEqual(await problemOf(t, big), { limit: 'a pax header is larger than 1048576 bytes' });
    assert.deepEqual(await problemOf(t, Buffer.concat([pax({ path: 'p'.repeat(200) }), member('x', 'y'), END]), { maxPax: 100 }), { limit: 'a pax header is larger than 100 bytes' });
});

test('GNU long name and link headers are skipped by their size, and a number may be GNU base 256', async (t) => {
    const name = `${'d'.repeat(120)}/${'f'.repeat(120)}.txt`;
    const longName = member('././@LongLink', `${name}\0`, { type: 'L' });
    const longLink = member('././@LongLink', `${name}\0`, { type: 'K' });
    assert.equal(await problemOf(t, Buffer.concat([longName, member(name.slice(0, 99), noise(900)), longLink, member('link', '', { type: '2' }), END])), null);
    // Cut in the long name's data, or with a size that runs past the end.
    assert.deepEqual(await problemOf(t, Buffer.concat([longName, member('x', 'y'), END]).subarray(0, BLOCK + 100)), { cut: 'a member runs past the end of the file' });
    assert.deepEqual(await problemOf(t, Buffer.concat([header({ name: '././@LongLink', type: 'L', size: 9_000_000 }), END])), { cut: 'a member runs past the end of the file' });
    // Base 256 (used for sizes over 8 GiB, and by some writers for any): 700 bytes skipped as 1,024.
    assert.equal(await problemOf(t, Buffer.concat([member('a.bin', noise(700), { base256: true }), member('b.bin', noise(10)), END])), null);
    assert.ok(await problemOf(t, Buffer.concat([header({ name: 'a.bin', size: 100, base256: true }), noise(700), END])), 'a base-256 size that is too small leaves data where a header is due');
    assert.deepEqual(await problemOf(t, Buffer.concat([header({ name: 'a.bin', size: 2 ** 50, base256: true }), END])), { cut: 'a member runs past the end of the file' });
    // Links, devices, directories and fifos have no data whatever their size field says, as GNU tar reads them.
    for (const type of ['1', '2', '3', '4', '5', '6']) {
        assert.equal(await problemOf(t, Buffer.concat([member('n', '', { type, size: 4096 }), member('after', noise(10)), END])), null, `type ${type}`);
    }
});

test('more members than the limit is refused as such, not as a cut', async (t) => {
    const four = Buffer.concat([member('a', 'x'), member('b', 'x'), member('c', 'x'), member('d', 'x'), END]);
    assert.equal(await problemOf(t, four, { maxMembers: 4 }), null);
    assert.deepEqual(await problemOf(t, four, { maxMembers: 3 }), { limit: 'it lists more than 3 members' });
});

// ---------------------------------------------------------------- what real tar writers make

const flavour = spawnSync('tar', ['--version'], { encoding: 'utf8' }).stdout?.includes('bsdtar') ? 'bsd' : 'gnu';
const FORMATS = flavour === 'bsd' ? ['ustar', 'pax', 'paxr', 'gnutar'] : ['ustar', 'gnu', 'posix', 'oldgnu'];
const hasPython = spawnSync('python3', ['--version'], { stdio: 'ignore' }).status === 0;
const LONG_PATH = `${'d'.repeat(60)}/${'f'.repeat(60)}.txt`;

// The same members in every writer: short and long names, a non-ASCII name, a member of noise and a last member of zeros.
function writeFiles(dir) {
    const files = { 'a.dat': noise(3000), [LONG_PATH]: noise(700), 'über.txt': Buffer.from('ü'), 'empty.txt': Buffer.alloc(0), 'zeros.bin': zeros(2048) };
    for (const [name, content] of Object.entries(files)) {
        fs.mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
        fs.writeFileSync(path.join(dir, name), content);
    }
    return Object.keys(files);
}

// Where the member called `name` ends, found by walking the headers by size (an oracle that knows nothing of tarProblem).
function endOf(tar, name) {
    for (let at = 0; at + BLOCK <= tar.length;) {
        const headerBlock = tar.subarray(at, at + BLOCK);
        const size = Number.parseInt(headerBlock.subarray(124, 135).toString().replace(/\0/g, '').trim() || '0', 8);
        const end = at + BLOCK + roundUp('12345 6'.includes(String.fromCharCode(headerBlock[156])) ? 0 : size);
        if (headerBlock.subarray(0, 100).toString().replace(/\0.*/s, '') === name) return end;
        at = end;
    }
    throw new Error(`no member ${name}`);
}

async function checkWriter(t, label, tar) {
    // Whole, every time.
    assert.equal(await problemOf(t, tar), null, `${label}: whole`);
    // Cut at every block: refused until the two blocks of zeros that end the tar are there, accepted from then on.
    const last = endOf(tar, 'zeros.bin');
    const refusedUntil = last + 2 * BLOCK;
    for (let at = 0; at <= tar.length; at += BLOCK) {
        assert.equal((await problemOf(t, tar.subarray(0, at))) === null, at >= refusedUntil, `${label}: cut at ${at} (the last member ends at ${last})`);
    }
}

test('the archives of bsdtar, GNU tar and Python tarfile in ustar, GNU and pax formats are whole, and cut short at every block they are refused', { timeout: 120_000 }, async (t) => {
    const dir = tempDir(t, 'tar-src');
    const names = writeFiles(dir);
    const out = tempDir(t, 'tar-out');
    let made = 0;
    for (const format of FORMATS) {
        const file = path.join(out, `${format}.tar`);
        const result = spawnSync('tar', [flavour === 'bsd' ? '--format' : `--format=${format}`, ...(flavour === 'bsd' ? [format] : []), '-cf', file, '-C', dir, ...names],
            { env: { ...process.env, COPYFILE_DISABLE: '1' }, stdio: 'pipe' });
        if (result.status !== 0) continue;
        await checkWriter(t, `${flavour} tar ${format}`, fs.readFileSync(file));
        made += 1;
    }
    assert.ok(made >= 3, `${made} tar formats written`);
    if (hasPython) {
        for (const format of ['USTAR_FORMAT', 'GNU_FORMAT', 'PAX_FORMAT']) {
            const file = path.join(out, `py-${format}.tar`);
            execFileSync('python3', ['-c', `
import sys, tarfile
with tarfile.open(sys.argv[1], 'w', format=getattr(tarfile, '${format}')) as tar:
    for name in sys.argv[3:]:
        tar.add(sys.argv[2] + '/' + name, arcname=name, recursive=False)
`, file, dir, ...names]);
            await checkWriter(t, `python tarfile ${format}`, fs.readFileSync(file));
        }
    }
});
