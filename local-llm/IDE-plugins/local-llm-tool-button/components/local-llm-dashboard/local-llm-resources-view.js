import { escapeHtml, formatMiB } from '../../../local-llm-settings/local-llm-settings-model.js';

import { hardwareCardsHtml } from './local-llm-dashboard-view.js?v=compact-capacities-2';

export function hardwareSummaryHtml(hardware = {}, gpuOverride = null, options = {}) {
    return hardwareCardsHtml(hardware, gpuOverride, { ...options, compact: true });
}

export function hardwareWarningsHtml({ profile, profileDecision, limits, hardware } = {}) {
    const messages = [...(limits?.warnings || [])];
    if (profile === 'apu') messages.unshift('AMD integrated GPU support is experimental and unqualified.');
    if (profile === 'vulkan') messages.unshift('AMD Vulkan support is unverified on hardware.');
    if (profile === 'venus') messages.unshift(`Apple GPU headroom is unknown; provisional cap ${formatMiB(limits?.deviceCapBytes)}.`);
    if (profile === 'cpu' && profileDecision?.cause && profileDecision.cause !== 'absent') {
        const explanation = profileDecision.explanation;
        const message = [explanation?.summary || profileDecision.reason || hardware?.gpu?.reason, explanation?.restartHint].filter(Boolean).join(' ');
        if (message) messages.push(message);
    }
    return [...new Set(messages)].map(message => `<p class="settings-status">${escapeHtml(message)}</p>`).join('');
}

