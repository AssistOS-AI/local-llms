// The reviewed vLLM-under-MPS calibration entry of the RTX 3060 Laptop tuple (stage-one run 752291a0, evidence digest 39bcd924...), through
// production's own resolver (resolveVllmMpsQualification, never a fake). The entry is a CANDIDATE pending stage-two model qualification:
// these tests pin that exactly this tuple resolves to the calibration's evidence digest, that every changed field does not, and that the
// committed table holds exactly that one reviewed entry and states its candidate status.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import { applyQualificationEntry } from '../tools/vllm_mps_calibration.mjs';
import { resolveVllmMpsQualification } from '../src/controller/vllmMpsQualification.mjs';

const TUPLE = Object.freeze({
    runnerLockDigest: 'd9065d01d086a952146366f0d6f2a3c54b0d179bc3a33c64cf3103d189363f62',
    driverVersion: '595.91.07',
    gpuPciDeviceId: '0x256010DE',
    computeCapability: '8.6',
    deviceTotalBytes: 6442450944,
});
const EVIDENCE_DIGEST = '39bcd924696fb6f3ec050c1ecbd59c6a0c7a11459badb0dbbcf1e857aadb746d';
const UNQUALIFIED = Object.freeze({
    qualified: false,
    code: 'vllm_mps_unqualified',
    fix: 'Use a qualified runner or clear this GPU share; this vLLM/driver/device combination needs MPS qualification.',
});
// The calibration tool's own rendering of the entry (proposed.source of the calibration artifact), as the module holds it.
const ENTRY_SOURCE = [
    '    Object.freeze({',
    '        runnerLockDigest: "d9065d01d086a952146366f0d6f2a3c54b0d179bc3a33c64cf3103d189363f62",',
    '        driverVersion: "595.91.07",',
    '        gpuPciDeviceId: "0x256010DE",',
    '        computeCapability: "8.6",',
    '        deviceTotalBytes: 6442450944,',
    "        denominator: 'physical-device',",
    `        evidenceDigest: '${EVIDENCE_DIGEST}',`,
    '    }),',
].join('\n');
const MODULE = new URL('../src/controller/vllmMpsQualification.mjs', import.meta.url);
const moduleSource = () => fs.readFileSync(MODULE, 'utf8');

test('L3Q.the-reviewed-tuple-resolves-to-the-calibration-evidence-digest-through-the-production-resolver', () => {
    assert.deepEqual(resolveVllmMpsQualification({ ...TUPLE }), { qualified: true, denominator: 'physical-device', evidenceDigest: EVIDENCE_DIGEST });
    // The same answer for a tuple object that carries the fields in another order and nothing else.
    const reordered = Object.fromEntries(Object.entries(TUPLE).reverse());
    assert.deepEqual(resolveVllmMpsQualification(reordered), { qualified: true, denominator: 'physical-device', evidenceDigest: EVIDENCE_DIGEST });
});

test('L3Q.every-single-changed-tuple-field-resolves-unqualified', () => {
    const changed = {
        runnerLockDigest: 'e9065d01d086a952146366f0d6f2a3c54b0d179bc3a33c64cf3103d189363f62',
        driverVersion: '595.91.08',
        gpuPciDeviceId: '0x256110DE',
        computeCapability: '8.7',
        deviceTotalBytes: TUPLE.deviceTotalBytes + 1,
    };
    assert.deepEqual(Object.keys(changed), Object.keys(TUPLE), 'all five tuple fields are changed, one at a time');
    for (const [field, value] of Object.entries(changed)) {
        assert.deepEqual(resolveVllmMpsQualification({ ...TUPLE, [field]: value }), UNQUALIFIED, `a changed ${field}`);
    }
    // A missing, malformed or wrongly typed field is unqualified too, never a partial match.
    for (const field of Object.keys(TUPLE)) {
        const { [field]: _removed, ...without } = TUPLE;
        assert.deepEqual(resolveVllmMpsQualification(without), UNQUALIFIED, `a missing ${field}`);
    }
    assert.deepEqual(resolveVllmMpsQualification({ ...TUPLE, deviceTotalBytes: String(TUPLE.deviceTotalBytes) }), UNQUALIFIED);
    assert.deepEqual(resolveVllmMpsQualification({ ...TUPLE, gpuPciDeviceId: '0x256010de' }), UNQUALIFIED);
    assert.deepEqual(resolveVllmMpsQualification(null), UNQUALIFIED);
});

test('L3Q.the-committed-table-holds-exactly-the-one-reviewed-entry-in-the-tools-rendering', () => {
    const source = moduleSource();
    const declaration = /const REVIEWED_QUALIFICATIONS = Object\.freeze\(\[\n([\s\S]*?)\n\]\);/.exec(source);
    assert.ok(declaration, 'the declaration keeps the layout the data-entry tool edits');
    assert.equal(declaration[1], ENTRY_SOURCE);
    assert.equal(source.split('Object.freeze({').length - 1, 1, 'one entry');
    // The calibration tool recognises the committed entry: its evidence and its tuple are both already in the table.
    const entry = { runnerLockDigest: TUPLE.runnerLockDigest, driverVersion: TUPLE.driverVersion, gpuPciDeviceId: TUPLE.gpuPciDeviceId, computeCapability: TUPLE.computeCapability,
        deviceTotalBytes: TUPLE.deviceTotalBytes, denominator: 'physical-device', evidenceDigest: EVIDENCE_DIGEST };
    assert.throws(() => applyQualificationEntry(source, { entry, source: ENTRY_SOURCE }), /already in REVIEWED_QUALIFICATIONS/);
    assert.throws(() => applyQualificationEntry(source, { entry: { ...entry, evidenceDigest: 'f'.repeat(64) }, source: ENTRY_SOURCE }), /already holds an entry for this tuple/);
});

test('L3Q.the-comment-above-the-table-states-a-candidate-pending-stage-two-and-claims-nothing-finished', () => {
    const source = moduleSource();
    const comment = source.slice(0, source.indexOf('const REVIEWED_QUALIFICATIONS')).split('\n').filter((line) => line.startsWith('//')).join('\n');
    for (const text of ['CANDIDATE', '752291a0', EVIDENCE_DIGEST, 'PENDING stage-two model qualification']) assert.ok(comment.includes(text), text);
    assert.doesNotMatch(comment, /None is qualified yet/);
    assert.doesNotMatch(comment, /\b(?:is|are|was|were) (?:now |fully )?qualified\b/i);
    assert.doesNotMatch(comment, /\breleased\b|\bqualification (?:is )?(?:complete|finished|done)\b/i);
});
