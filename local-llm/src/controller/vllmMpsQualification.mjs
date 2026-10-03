import { createHash } from 'node:crypto';

const TUPLE_FIELDS = Object.freeze(['runnerLockDigest', 'driverVersion', 'gpuPciDeviceId', 'computeCapability', 'deviceTotalBytes']);
const FIX = 'Use a qualified runner or clear this GPU share; this vLLM/driver/device combination needs MPS qualification.';

// Entries require approved calibration and a subsequent normal model readiness,
// text-response and cleanup check for that exact tuple.
// The one entry below is a CANDIDATE calibration entry: stage-one run 752291a0, calibration evidence digest
// 39bcd924696fb6f3ec050c1ecbd59c6a0c7a11459badb0dbbcf1e857aadb746d. It is PENDING stage-two model qualification (public
// admission, model readiness, a text response, resource observation and cleanup for this exact tuple), which has not run.
// It states no finished qualification and no release; any changed tuple field invalidates the calibration.
const REVIEWED_QUALIFICATIONS = Object.freeze([
    Object.freeze({
        runnerLockDigest: "d9065d01d086a952146366f0d6f2a3c54b0d179bc3a33c64cf3103d189363f62",
        driverVersion: "595.91.07",
        gpuPciDeviceId: "0x256010DE",
        computeCapability: "8.6",
        deviceTotalBytes: 6442450944,
        denominator: 'physical-device',
        evidenceDigest: '39bcd924696fb6f3ec050c1ecbd59c6a0c7a11459badb0dbbcf1e857aadb746d',
    }),
]);

function canonical(value) {
    if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
    if (value && typeof value === 'object') return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
    return JSON.stringify(value);
}

export function vllmRunnerLockDigest(entry) {
    if (entry?.id !== 'vllm' || typeof entry.version !== 'string' || !Array.isArray(entry.files) || !entry.files.length
        || !/^[0-9a-f]{64}$/.test(entry.digest || '')) return null;
    // The selected validated entry, including its checks, supplies the pins.
    // Computed summaries are not part of its identity. The installer separately
    // verifies that its runnable copy matches this selected entry's digest.
    const { id, version, kind, files, check } = entry;
    return createHash('sha256').update(canonical({ id, version, kind, files, check })).digest('hex');
}

function validTuple(value) {
    return value && /^[0-9a-f]{64}$/.test(value.runnerLockDigest || '')
        && typeof value.driverVersion === 'string' && /^[0-9]+(?:\.[0-9]+){1,3}$/.test(value.driverVersion)
        && typeof value.gpuPciDeviceId === 'string' && /^0x[0-9A-F]{8}$/.test(value.gpuPciDeviceId)
        && typeof value.computeCapability === 'string' && /^[0-9]+\.[0-9]+$/.test(value.computeCapability)
        && Number.isSafeInteger(value.deviceTotalBytes) && value.deviceTotalBytes > 0;
}

export function createVllmMpsQualificationResolver(dataProvider) {
    return (tuple) => {
        const unavailable = { qualified: false, code: 'vllm_mps_unqualified', fix: FIX };
        if (!validTuple(tuple)) return unavailable;
        let entries;
        try { entries = dataProvider(); } catch { return unavailable; }
        if (!Array.isArray(entries) || entries.length > 256) return unavailable;
        const match = entries.find((entry) => validTuple(entry)
            && Object.keys(entry).every((field) => [...TUPLE_FIELDS, 'denominator', 'evidenceDigest'].includes(field))
            && TUPLE_FIELDS.every((field) => entry[field] === tuple[field])
            && entry.denominator === 'physical-device' && /^[0-9a-f]{64}$/.test(entry.evidenceDigest || ''));
        return match ? { qualified: true, denominator: 'physical-device', evidenceDigest: match.evidenceDigest } : unavailable;
    };
}

const productionResolver = createVllmMpsQualificationResolver(() => REVIEWED_QUALIFICATIONS);

export function resolveVllmMpsQualification(tuple) {
    return productionResolver(tuple);
}
