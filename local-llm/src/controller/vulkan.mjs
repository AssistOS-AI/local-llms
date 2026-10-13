import { execFile } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import { runHardwareQuery } from './hardware.mjs';
import { computeDiagnostic } from './computeDiagnostics.mjs';

const MIB = 1024 * 1024;
export const VULKAN_PROFILES = Object.freeze(['vulkan', 'venus', 'apu']);
export const VULKAN_SERVER = '/opt/llama.cpp-vulkan/llama-server';
export const VULKAN_ALL_ICD_FILES = '/opt/local-llm/vulkan/icd.d/radeon_icd.json:/opt/local-llm/vulkan/icd.d/virtio_icd.json';
export const isVulkanProfile = (profile) => VULKAN_PROFILES.includes(profile);
export const catalogProfileOf = (profile) => profile === 'vulkan' ? 'dedicated' : ['venus', 'apu'].includes(profile) ? 'cpu' : profile;
export const APU_BUDGET_REASON = 'APU GPU use is not available while this agent has a finite or unknown container memory limit; such a limit is not verified to cap shared GPU memory.';
export const vulkanRefusal = (name) => `${name} is not available on a Vulkan GPU in this release; use llama.cpp (Vulkan build).`;

export function vulkanContractOf(contract) {
    const icd = (value) => typeof value === 'string' && path.posix.isAbsolute(value)
        && path.posix.normalize(value) === value && value.startsWith('/opt/local-llm/vulkan/icd.d/') && value.endsWith('.json') && !value.includes(':');
    if (contract?.vulkan_contract !== '1' || !/^b\d+$/.test(contract?.vulkan_llama_cpp || '')
        || contract?.vulkan_llama_cpp_server !== VULKAN_SERVER || contract?.vulkaninfo !== '/usr/bin/vulkaninfo'
        || !/^[0-9A-Za-z.+_-]{1,64}$/.test(contract?.vulkan_mesa || '')
        || !icd(contract?.vulkan_icd_radv) || !icd(contract?.vulkan_icd_venus)) return null;
    return Object.freeze({ ...Object.fromEntries(Object.entries(contract).filter(([key]) => key.startsWith('vulkan'))) });
}

export function hasRenderNode({ fsApi = fs, devDir = '/dev/dri' } = {}) {
    try {
        return fsApi.readdirSync(devDir).some((name) => /^renderD\d+$/.test(name)
            && fsApi.lstatSync(path.join(devDir, name)).isCharacterDevice());
    } catch { return false; }
}

export function vulkanProbeEnv(icdFiles) {
    if (typeof icdFiles !== 'string' || !icdFiles || !icdFiles.split(':').every((file) => path.posix.isAbsolute(file)
        && path.posix.normalize(file) === file && file.endsWith('.json'))) throw new Error('Vulkan ICD files must be absolute .json paths.');
    return { PATH: '/usr/bin:/bin', HOME: '/tmp', LANG: 'C.UTF-8', VK_DRIVER_FILES: icdFiles,
        VK_LOADER_LAYERS_DISABLE: '~all~', MESA_SHADER_CACHE_DISABLE: 'true' };
}

export function parseVulkanSummary(text) {
    const blocks = String(text).split(/^GPU(\d+):\s*$/m);
    const devices = [];
    for (let i = 1; i < blocks.length; i += 2) {
        const device = { index: Number(blocks[i]) };
        for (const field of ['deviceName', 'deviceType', 'vendorID', 'driverID']) {
            const matches = [...blocks[i + 1].matchAll(new RegExp(`^\\s*${field}\\s*=\\s*(\\S[^\\r\\n]*)$`, 'gm'))];
            if (matches.length !== 1) throw new Error(`GPU${device.index} has no unique ${field}.`);
            device[field] = matches[0][1].trim();
        }
        for (const field of ['deviceID', 'driverInfo', 'apiVersion']) {
            const matches = [...blocks[i + 1].matchAll(new RegExp(`^\\s*${field}\\s*=\\s*(\\S[^\\r\\n]*)$`, 'gm'))];
            if (matches.length === 1) device[field] = matches[0][1].trim();
        }
        devices.push(device);
    }
    if (!devices.length && !/^Vulkan Instance Version:\s*\d+(?:\.\d+)+\s*$/m.test(String(text))) throw new Error('Malformed Vulkan summary.');
    return devices;
}

export function parseListDevices(text) {
    const lines = String(text).trimEnd().split(/\r?\n/);
    if (lines.shift() !== 'Available devices:') throw new Error('Missing Available devices header.');
    if (lines.length === 1 && lines[0] === '  (none)') return [];
    if (!lines.length) throw new Error('Missing device rows.');
    return lines.map((line) => {
        const match = /^  (Vulkan\d+): (.*) \((\d+) MiB, (\d+) MiB free\)$/.exec(line);
        if (!match) throw new Error('Malformed Vulkan device row.');
        return { id: match[1], description: match[2], totalMiB: Number(match[3]), freeMiB: Number(match[4]) };
    });
}

export async function readVulkanGpu({ execFileImpl = execFile, contract, signal, apuEnabled = false } = {}) {
    let observedDriver = null;
    const unavailable = (state, reason, code) => ({ available: false, state, backend: 'vulkan', reason,
        diagnostic: computeDiagnostic(code, { backend: 'vulkan', driver: observedDriver, message: reason }) });
    async function query(command, args, icds) {
        const result = await runHardwareQuery(execFileImpl, command, args, { signal, env: vulkanProbeEnv(icds) });
        if (result.aborted) throw result.error;
        return result;
    }
    const summary = await query(contract.vulkaninfo, ['--summary'], `${contract.vulkan_icd_radv}:${contract.vulkan_icd_venus}`);
    if (!summary.ok) return unavailable(/ERROR_INCOMPATIBLE_DRIVER/.test(summary.stdout + summary.stderr) ? 'unusable' : 'unreadable',
        `vulkaninfo failed: ${(summary.stderr || summary.error?.message || '').trim().slice(0, 300)}`, 'vulkan_query_unreadable');
    let devices;
    try { devices = parseVulkanSummary(summary.stdout); }
    catch (error) { return unavailable('unreadable', error.message, 'vulkan_response_invalid'); }
    const accepted = devices.filter((device) => ['DRIVER_ID_MESA_RADV', 'DRIVER_ID_MESA_VENUS'].includes(device.driverID));
    if (accepted.length !== 1 || devices.some((device) => device.driverID === 'DRIVER_ID_MESA_LLVMPIPE'
        || device.deviceType === 'PHYSICAL_DEVICE_TYPE_CPU')) return unavailable('unusable', 'Vulkan needs exactly one RADV or Venus GPU and no software device.', 'vulkan_device_unsupported');
    const device = accepted[0];
    const driver = device.driverID === 'DRIVER_ID_MESA_RADV' ? 'radv' : 'venus';
    observedDriver = driver;
    const integrated = driver === 'radv' && device.deviceType === 'PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU';
    if (integrated && !apuEnabled) return unavailable('unusable', `An AMD integrated GPU (${device.deviceName}) is attached. APU use is off: the operator enables it with ploinky var LOCAL_LLM_AMD_APU experimental and a restart of local-llm.`, 'vulkan_device_unsupported');
    if (driver === 'radv' && !integrated && device.deviceType !== 'PHYSICAL_DEVICE_TYPE_DISCRETE_GPU') return unavailable('unusable', 'Only discrete or experimental integrated AMD Radeon GPUs are supported through RADV.', 'vulkan_device_unsupported');
    const icdFile = contract[driver === 'radv' ? 'vulkan_icd_radv' : 'vulkan_icd_venus'];
    const list = await query(contract.vulkan_llama_cpp_server, ['--list-devices'], icdFile);
    if (!list.ok) return unavailable('unreadable', `Vulkan device query failed: ${(list.stderr || list.error?.message || '').trim().slice(0, 300)}`, 'vulkan_query_unreadable');
    let rows;
    try { rows = parseListDevices(list.stdout); }
    catch (error) { return unavailable('unreadable', error.message, 'vulkan_response_invalid'); }
    if (rows.length !== 1 || rows[0].id !== 'Vulkan0' || rows[0].description !== device.deviceName) return unavailable('unusable', 'The Vulkan runner must see exactly Vulkan0 matching vulkaninfo.', 'vulkan_device_mismatch');
    const totalBytes = rows[0].totalMiB * MIB;
    const freeBytes = rows[0].freeMiB * MIB;
    if (![totalBytes, freeBytes].every((value) => Number.isSafeInteger(value) && value >= 0)) return unavailable('unreadable', 'Vulkan memory figures exceed safe byte counts.', 'vulkan_memory_invalid');
    if (totalBytes <= 0 || freeBytes > totalBytes) return unavailable('unusable', 'Vulkan reported no usable memory capacity.', 'vulkan_memory_invalid');
    return { available: true, backend: 'vulkan', driver, name: device.deviceName, deviceType: device.deviceType,
        vendorId: device.vendorID, memoryModel: integrated ? 'shared' : driver === 'radv' ? 'dedicated' : 'venus', totalBytes,
        ...(integrated ? Object.fromEntries(['deviceID', 'driverInfo', 'apiVersion'].filter((key) => device[key]).map((key) => [key, device[key]])) : {}),
        usedBytes: totalBytes - freeBytes, freeBytes, processes: [], icdFile };
}
