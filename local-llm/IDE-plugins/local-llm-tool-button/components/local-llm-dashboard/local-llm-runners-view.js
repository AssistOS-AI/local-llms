import { escapeHtml, formatBytes, progressPercent } from '../../../local-llm-settings/local-llm-settings-model.js';
import { iconHtml } from './local-llm-icons.js';

const RUNNER_PHASES = new Set(['starting', 'loading', 'ready', 'stopping']);
const alphabetical = new Intl.Collator('en', { sensitivity: 'base', numeric: true });

export function runnersIntroText(runners = []) {
    return runners.length ? 'Runners execute your local models. Availability depends on this workspace’s hardware.' : '';
}

function runnerState(runner) {
    const install = runner.install;
    const state = install?.state || {};
    // Platform restrictions take precedence over a failed executable probe.
    const lacking = (runner.supported === false ? runner.unsupportedReason : '') || runner.profileUnsupportedReason;
    if (lacking || runner.supported === false) return { key: 'unavailable', label: 'Unsupported', icon: 'blocked', reason: lacking || 'Not available on this platform.' };
    if (runner.enabled === false) return { key: 'disabled', label: 'Disabled', icon: 'blocked', reason: runner.disabledReason || 'This runner is not enabled on this deployment.' };
    if (install?.installing) return { key: 'busy', label: state.phase === 'downloading' ? 'Downloading' : 'Installing' };
    if (state.phase === 'error') return { key: 'error', label: 'Installation failed', icon: 'warning', reason: state.error || 'The runner installation failed.' };
    if (state.phase === 'paused') return { key: 'paused', label: 'Installation paused', icon: 'warning', reason: state.pausedReason || 'The runner installation is paused.' };
    if (runner.installed || install?.installed) return { key: 'installed', label: 'Installed', icon: 'check' };
    if (install) return { key: 'absent', label: 'Not installed', icon: 'download' };
    return { key: 'unavailable', label: 'Unavailable', icon: 'blocked', reason: runner.reason || 'Not available on this deployment.' };
}

function licenceHtml(licence = {}, accepted = null, compact = false) {
    const links = [
        licence.url ? `<a href="${escapeHtml(licence.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(licence.name || 'Licence')}</a>` : escapeHtml(licence.name || ''),
        !compact && licence.source ? `<a href="${escapeHtml(licence.source)}" target="_blank" rel="noopener noreferrer">source</a>` : '',
    ].filter(Boolean).join(' · ');
    const acceptance = accepted?.acceptedBy
        ? ` · accepted by ${escapeHtml(accepted.acceptedBy)} on ${escapeHtml(String(accepted.acceptedAt || '').slice(0, 10))}` : '';
    if (compact) return `<div class="local-llm-runner-fact" data-llm-runner-licence><dt>Licence</dt><dd>${links || 'Not reported'}</dd></div>`;
    return links ? `<div class="settings-card-meta">Licence: ${links}${acceptance}</div>` : '';
}

function progressHtml(install) {
    if (!install?.installing) return '';
    const state = install.state || {};
    const download = state.download;
    if (!download?.total || !['downloading', 'installing'].includes(state.phase)) return '';
    const percent = progressPercent(download) ?? 0;
    const rate = download.rate ? ` · ${formatBytes(download.rate)}/s` : '';
    return `<div class="local-llm-runner-progress"><progress class="local-llm-progress" max="100" value="${percent}" aria-label="Install progress">${percent}%</progress>
        <span class="settings-card-meta">${escapeHtml(`${formatBytes(download.bytes)} of ${formatBytes(download.total)}${rate}`)}</span></div>`;
}

function detailsHtml(runner, status, version) {
    const install = runner.install;
    const state = install?.state || {};
    const contents = [
        version ? `<span>Version ${escapeHtml(version)}</span>` : '<span>Version not reported.</span>',
        `<span>${install ? 'Installed on demand from pinned files.' : 'Provided by the workspace image when available.'}</span>`,
        status.reason ? `<span>${escapeHtml(status.reason)}</span>` : '',
        runner.enabled === false && runner.disabledReason && runner.disabledReason !== status.reason ? `<span>${escapeHtml(runner.disabledReason)}</span>` : '',
        install ? `<span>${escapeHtml(formatBytes(install.totalBytes))} to download</span>` : '',
        install ? licenceHtml(install.licence, state.licence) : '',
        install?.licence?.notice ? `<span>${escapeHtml(install.licence.notice)}</span>` : '',
        state.rebuild?.seconds ? `<span>Set up in ${Math.round(state.rebuild.seconds)} s</span>` : '',
    ].join('');
    return `<div class="local-llm-runner-details">${contents}</div>`;
}

/** Stable alphabetical cards; state and actions still come from the controller. */
export function runnersPanelHtml(runners = [], { deployment = null } = {}) {
    const busy = runners.some((runner) => runner.install?.installing);
    const sorted = [...runners].sort((a, b) => alphabetical.compare(a.displayName || a.id, b.displayName || b.id) || alphabetical.compare(a.id, b.id));
    const cards = sorted.map((runner) => {
        const name = runner.displayName || runner.id;
        const install = runner.install;
        const status = runnerState(runner);
        const version = runner.version || install?.version || '';
        const inUse = deployment?.runnerId === runner.id && RUNNER_PHASES.has(deployment.phase);
        // Encode every code point, not punctuation replacement, to avoid ID collisions.
        const helpId = `localLlmRunner-${Array.from(String(runner.id), char => char.codePointAt(0).toString(16)).join('-')}`;
        let action = '';
        if (install?.installed && !install.installing) {
            action = `<button type="button" class="gray-button" id="${helpId}-action" data-local-action="uninstallRunner ${escapeHtml(runner.id)}" aria-label="Uninstall ${escapeHtml(name)}">Uninstall</button>`;
        } else if (install && !install.installing && !busy && runner.enabled !== false && runner.supported !== false && !runner.profileUnsupportedReason) {
            action = `<button type="button" class="general-button" id="${helpId}-action" data-local-action="installRunner ${escapeHtml(runner.id)}" aria-label="Install ${escapeHtml(name)}">Install</button>`;
        }
        const stateIcon = status.key === 'busy'
            ? '<span class="local-llm-runner-spinner" aria-hidden="true"><span class="spinner-orbit"><span></span><span></span><span></span></span></span>'
            : iconHtml(status.icon);
        const downloadSize = Number.isFinite(install?.totalBytes) && install.totalBytes >= 0 ? formatBytes(install.totalBytes) : 'Not reported';
        const notice = install?.licence?.notice && (install.licence.proprietary || install.licence.requiresAcceptance)
            ? `<p class="settings-card-meta local-llm-runner-notice">${escapeHtml(install.licence.notice)}</p>` : '';
        return `<li class="local-llm-runner-item" data-runner-id="${escapeHtml(runner.id)}" data-runner-state="${status.key}">
            <header class="local-llm-runner-heading"><h3 class="settings-card-title">${escapeHtml(name)}</h3>
                <button type="button" class="gray-button local-llm-info" id="${helpId}-button" popovertarget="${helpId}" aria-label="Details for ${escapeHtml(name)}">${iconHtml('info')}</button></header>
            <div id="${helpId}" popover class="settings-card settings-card-static local-llm-help" data-llm-runner-help><strong>${escapeHtml(name)}</strong>${detailsHtml(runner, status, version)}</div>
            <div class="local-llm-runner-state">${stateIcon}<span>${status.label}</span><span class="local-llm-runner-state-actions">${!install && runner.installed ? '<span class="settings-card-meta">Included in image</span>' : ''}${inUse ? '<span class="status-badge success">In use</span>' : ''}${action}</span></div>
            <dl class="local-llm-runner-facts settings-card-meta">
                <div class="local-llm-runner-fact"><dt>Version</dt><dd>${escapeHtml(version || 'Not reported')}</dd></div>
                <div class="local-llm-runner-fact"><dt>Download</dt><dd>${escapeHtml(downloadSize)}</dd></div>
                ${licenceHtml(install?.licence, null, true)}
            </dl>
            <div class="local-llm-runner-notes">${status.reason ? `<p class="${status.key === 'error' ? 'settings-status error' : 'settings-card-meta'} local-llm-runner-reason">${escapeHtml(status.reason)}</p>` : ''}${progressHtml(install)}${notice}</div>
        </li>`;
    }).join('');
    return `<ul class="local-llm-runner-list local-llm-runner-grid" aria-label="Runners">${cards || '<li class="settings-empty-state">No runners available.</li>'}</ul>`;
}
