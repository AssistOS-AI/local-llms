import { createHash } from 'node:crypto';

const TUPLE_FIELDS = Object.freeze(['runnerLockDigest', 'driverVersion', 'gpuPciDeviceId', 'computeCapability', 'deviceTotalBytes']);
const FIX = 'Use a qualified runner or clear this GPU share; this vLLM/driver/device combination needs MPS qualification.';

// Entries require approved calibration and a subsequent normal model readiness,
// text-response and cleanup check for that exact tuple. None is qualified yet.
const REVIEWED_QUALIFICATIONS = Object.freeze([]);

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
