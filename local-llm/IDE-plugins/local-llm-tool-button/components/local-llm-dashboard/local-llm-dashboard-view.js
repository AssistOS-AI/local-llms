// Pure HTML builders for the Local LLMs dashboard. They take plain data and
// return markup, so the layout rules (no Run button in the table, one in the
// detail panel) can be tested without a browser.
import {
    admissionLabel,
    contextLabel,
    downloadLabel,
    escapeHtml,
    formatBytes,
    formatDuration,
    formatMiB,
    progressPercent,
    runnerLabel,
} from '../../../local-llm-settings/local-llm-settings-model.js';

export const ACTIVE_PHASES = new Set(['downloading', 'verifying', 'pulling', 'starting', 'loading', 'ready', 'stopping']);
const DOWNLOAD_PHASES = new Set(['downloading', 'verifying', 'pulling', 'paused']);

/** The runners that get a column in the Models table: those that can run here. */
export function tableRunners(runners = []) {
    return (Array.isArray(runners) ? runners : []).filter((runner) => runner?.supported);
}

function labelOf(runnerId, runners = []) {
    return runners.find((runner) => runner.id === runnerId)?.displayName || runnerLabel(runnerId);
}

/**
 * Split a runner's form fields into the basic ones and the "Advanced" rest.
 * The runner names its basic fields (the ones that decide whether a model
 * fits); those in `moeParams` show only for a mixture-of-experts model.
 */
export function splitRunFields(fields = [], runner = null, model = null) {
    const moeOnly = new Set(runner?.moeParams || []);
    const wanted = (runner?.basicParams || [])
        .filter((name) => !moeOnly.has(name) || model?.architecture === 'moe');
    const basic = [];
    const advanced = [];
    for (const field of fields) (wanted.includes(field.name) ? basic : advanced).push(field);
    basic.sort((left, right) => wanted.indexOf(left.name) - wanted.indexOf(right.name));
    return { basic, advanced };
}

function meter({ label, value, max, text }) {
    if (!Number.isFinite(value) || !Number.isFinite(max) || max <= 0) return '';
    const clamped = Math.max(0, Math.min(value, max));
    return `<meter class="local-llm-meter" min="0" max="${max}" value="${clamped}" low="${max * 0.7}" high="${max * 0.9}" optimum="0"
                   aria-label="${escapeHtml(label)}" title="${escapeHtml(text)}">${escapeHtml(text)}</meter>`;
}

function statCard(name, title, value, meta = '', extra = '') {
    return `
        <div class="settings-card settings-card-static local-llm-stat" data-llm-card="${escapeHtml(name)}">
            <div class="settings-card-meta local-llm-stat-title">${escapeHtml(title)}</div>
            <div class="settings-card-title local-llm-stat-value">${escapeHtml(value)}</div>
            ${extra}
            ${meta ? `<div class="settings-card-meta">${escapeHtml(meta)}</div>` : ''}
        </div>`;
}

/** The GPU, RAM and disk cards. `gpu` may come from a fresher status poll. */
export function hardwareCardsHtml(hardware = {}, gpuOverride = null) {
    const gpu = gpuOverride?.available ? { ...(hardware.gpu || {}), ...gpuOverride } : (hardware.gpu || {});
    const cards = [];
    if (gpu.available) {
        const users = Array.isArray(gpu.processes) && gpu.processes.length
            ? `in use by ${gpu.processes.map((process) => process.name).join(', ')}`
            : 'no process is using it';
        cards.push(statCard('gpu', 'GPU memory', `${formatMiB(gpu.usedBytes)} of ${formatMiB(gpu.totalBytes)}`,
            `${gpu.name || 'GPU'} · ${users}`,
            meter({ label: 'GPU memory used', value: gpu.usedBytes, max: gpu.totalBytes, text: `${formatMiB(gpu.usedBytes)} used` })));
    } else {
        cards.push(statCard('gpu', 'GPU', 'Not available', gpu.reason || 'No GPU is attached to this agent.'));
    }
    const memory = hardware.memory || {};
    cards.push(statCard('ram', 'RAM available', formatMiB(memory.availableBytes),
        Number.isFinite(memory.totalBytes) ? `of ${formatMiB(memory.totalBytes)}` : ''));
    const disk = hardware.disk || {};
    cards.push(statCard('disk', 'Disk free', formatBytes(disk.freeBytes),
        Number.isFinite(disk.totalBytes) ? `of ${formatBytes(disk.totalBytes)}, for model weights` : 'for model weights'));
    return cards.join('');
}

/** The deployment card: what runs, how it is doing, and its actions. */
export function statusCardHtml(status = {}) {
    const deployment = status.deployment || null;
    const phase = status.phase || deployment?.phase || 'idle';
    const actions = `
        <div class="settings-card-actions local-llm-stat-actions">
            <button type="button" class="general-button secondary" data-llm-cancel data-local-action="cancelDownload">Cancel download</button>
            <button type="button" class="gray-button" data-llm-stop data-local-action="stopDeployment">Stop</button>
        </div>`;
    if (!deployment || phase === 'idle') {
        const last = deployment ? `Last run: ${deployment.modelId} on ${runnerLabel(deployment.runnerId)}` : 'Select a model below, then Run';
        return `
            <div class="settings-card settings-card-static local-llm-stat local-llm-stat-status" data-llm-card="status">
                <div class="settings-card-meta local-llm-stat-title">Running model</div>
                <div class="settings-card-title local-llm-stat-value">None</div>
                <div class="settings-card-meta">${escapeHtml(last)}</div>
                ${actions}
            </div>`;
    }
    const badge = phase === 'ready' ? 'success' : phase === 'error' ? 'error' : '';
    const lines = [];
    const download = deployment.download || null;
    if (download && DOWNLOAD_PHASES.has(phase)) {
        const percent = progressPercent(download) ?? 0;
        const rate = download.rate ? `${formatBytes(download.rate)}/s` : '';
        const eta = Number.isFinite(download.etaSeconds) ? `about ${formatDuration(download.etaSeconds)} left` : '';
        lines.push(`<progress class="local-llm-progress" max="100" value="${percent}" aria-label="Download progress">${percent}%</progress>`);
        lines.push(`<div class="settings-card-meta">${escapeHtml([`${formatBytes(download.bytes)} of ${formatBytes(download.total)}`, rate, eta].filter(Boolean).join(' · '))}</div>`);
    }
    const report = status.runnerReport || {};
    const reportParts = [
        report.device ? `device ${report.device}` : '',
        report.offloaded ? `${report.offloaded.layers}/${report.offloaded.of} layers on the GPU` : '',
        Number.isFinite(report.totalMiB) ? `buffers ${report.totalMiB.toLocaleString('en-US')} MiB` : '',
    ].filter(Boolean);
    if (reportParts.length) lines.push(`<div class="settings-card-meta">${escapeHtml(reportParts.join(' · '))}</div>`);
    if (status.context) lines.push(`<div class="settings-card-meta">Context: ${escapeHtml(contextLabel(status.context))}</div>`);
    const last = status.lastCompletion;
    if (last && Number.isFinite(last.generationTokensPerSecond)) {
        lines.push(`<div class="settings-card-meta">Last completion: ${escapeHtml(String(last.completionTokens ?? '?'))} tokens at ${escapeHtml(last.generationTokensPerSecond.toFixed(1))} tokens/s (${escapeHtml(last.source || 'runner')})</div>`);
    }
    if (deployment.error) lines.push(`<div class="settings-status error">${escapeHtml(deployment.error)}</div>`);
    if (deployment.pausedReason && phase === 'paused') lines.push(`<div class="settings-card-meta">${escapeHtml(deployment.pausedReason)}</div>`);
    return `
        <div class="settings-card settings-card-static local-llm-stat local-llm-stat-status" data-llm-card="status">
            <div class="settings-card-meta local-llm-stat-title">Running model</div>
            <div class="local-llm-stat-line">
                <span class="settings-card-title local-llm-stat-value">${escapeHtml(deployment.modelId)} on ${escapeHtml(runnerLabel(deployment.runnerId))}</span>
                <span class="status-badge ${badge}">${escapeHtml(phase)}</span>
            </div>
            ${lines.join('')}
            ${actions}
        </div>`;
}

function runnerCell(entry) {
    if (!entry) return '<td class="settings-card-meta">—</td>';
    return `
        <td>
            <div>${escapeHtml(downloadLabel(entry.download, entry.size))}</div>
            <div class="settings-card-meta">${escapeHtml(admissionLabel(entry.admission))}</div>
        </td>`;
}

function modelChips(model, activeModelId) {
    return [
        model.architecture === 'moe' ? '<span class="settings-chip">MoE</span>' : '',
        model.seed ? '<span class="settings-chip">catalog</span>' : '<span class="settings-chip">added</span>',
        model.id === activeModelId ? '<span class="status-badge success">in use</span>' : '',
    ].join('');
}

function modelSizes(model) {
    return [model.totalParams, model.activeParams ? `${model.activeParams} active` : '']
        .filter(Boolean).join(' · ');
}

/**
 * The models table. It has no buttons: a row selects its model, and the
 * detail panel holds the model's actions and its one Run button.
 */
export function modelsTableHtml(models = [], { selectedId = '', activeModelId = '', runners = [] } = {}) {
    if (!models.length) return '<div class="settings-empty-state">No models in the catalog.</div>';
    const columns = tableRunners(runners);
    const rows = models.map((model) => {
        const selected = model.id === selectedId;
        return `
            <tr class="local-llm-model-row${selected ? ' active' : ''}" data-local-action="selectModel ${escapeHtml(model.id)}"
                data-model-id="${escapeHtml(model.id)}" tabindex="0" aria-selected="${selected ? 'true' : 'false'}">
                <td>
                    <div class="local-llm-model-name">
                        <span class="settings-card-title">${escapeHtml(model.displayName || model.id)}</span>
                        ${modelChips(model, activeModelId)}
                    </div>
                    <div class="settings-card-meta">${escapeHtml(modelSizes(model) || model.id)}</div>
                </td>
                <td>${escapeHtml(model.license || '—')}</td>
                ${columns.map((runner) => runnerCell(model.runners?.[runner.id])).join('')}
            </tr>`;
    }).join('');
    return `
        <div class="local-llm-table-scroll">
            <table class="local-llm-table" aria-label="Models">
                <thead>
                    <tr>
                        <th scope="col">Model</th>
                        <th scope="col">Licence</th>
                        ${columns.map((runner) => `<th scope="col">${escapeHtml(labelOf(runner.id, runners))}</th>`).join('')}
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

/** Whether any of this model's weights are on disk (downloaded or partial). */
export function hasWeightsOnDisk(model) {
    const entries = model?.weights ? Object.values(model.weights) : Object.values(model?.runners || {});
    return entries.some((entry) => ['complete', 'partial'].includes(entry?.download?.state));
}

/**
 * The selected model's heading, its weights once per format (runners that
 * read the same format share one download, so Delete weights sits there),
 * and each supported runner's fit. An added model also gets Remove.
 */
export function detailInfoHtml(model, { runners = [], activeModelId = '' } = {}) {
    if (!model) return '';
    const supported = new Set(tableRunners(runners).map((runner) => runner.id));
    const weights = Object.entries(model.weights || {}).map(([format, entry]) => {
        const deletable = entry.download && ['complete', 'partial'].includes(entry.download.state);
        const readers = (entry.runners || []).filter((runnerId) => supported.has(runnerId))
            .map((runnerId) => labelOf(runnerId, runners));
        return `
                <li class="local-llm-runner-item">
                    <div>
                        <div class="settings-card-title">${escapeHtml(entry.label || format)}</div>
                        <div class="settings-card-meta">${escapeHtml(downloadLabel(entry.download, entry.size))}${readers.length ? ` · used by ${escapeHtml(readers.join(', '))}` : ''}</div>
                    </div>
                    ${deletable ? `<button type="button" class="gray-button" data-local-action="deleteWeights ${escapeHtml(model.id)} ${escapeHtml(format)}">Delete weights</button>` : ''}
                </li>`;
    }).join('');
    const fits = Object.entries(model.runners || {})
        .filter(([runnerId]) => supported.has(runnerId))
        .map(([runnerId, entry]) => `
                <li class="local-llm-runner-item">
                    <div>
                        <div class="settings-card-title">${escapeHtml(labelOf(runnerId, runners))}</div>
                        <div class="settings-card-meta">${escapeHtml(admissionLabel(entry.admission))}</div>
                    </div>
                </li>`).join('');
    const facts = [modelSizes(model), model.license, model.id].filter(Boolean).join(' · ');
    // The server removes a model only once none of its weights are on disk.
    const weightsPresent = hasWeightsOnDisk(model);
    const remove = model.seed ? '' : `
            <div class="local-llm-remove">
                <button type="button" class="gray-button" data-local-action="removeModel ${escapeHtml(model.id)}"
                        ${weightsPresent ? 'disabled title="Delete its weights first"' : ''}>Remove</button>
                ${weightsPresent ? '<span class="settings-card-meta">Delete its weights first to remove it.</span>' : ''}
            </div>`;
    return `
        <div class="local-llm-detail-head">
            <div>
                <h3 class="settings-section-title local-llm-model-name">${escapeHtml(model.displayName || model.id)} ${modelChips(model, activeModelId)}</h3>
                <p class="settings-section-description">${escapeHtml(facts)}</p>
            </div>
            ${remove}
        </div>
        <ul class="local-llm-runner-list" aria-label="Weights">${weights}</ul>
        <ul class="local-llm-runner-list" aria-label="Runners">${fits || '<li class="settings-card-meta">No runner can run this model.</li>'}</ul>`;
}

/** GPU and RAM estimate meters for the run form. */
export function estimateHtml({ admission = null, context = null, error = '' } = {}, hardware = {}) {
    if (error) return `<div class="settings-status error">${escapeHtml(error)}</div>`;
    const estimate = admission?.estimate || {};
    const gpu = hardware.gpu || {};
    const memory = hardware.memory || {};
    const parts = [];
    if (Number.isFinite(estimate.gpuBytes) && gpu.available) {
        parts.push(`
            <div class="local-llm-estimate-row">
                <span class="settings-card-meta">GPU about ${escapeHtml(formatMiB(estimate.gpuBytes))} of ${escapeHtml(formatMiB(gpu.totalBytes))}</span>
                ${meter({ label: 'Estimated GPU memory', value: estimate.gpuBytes, max: gpu.totalBytes, text: `GPU about ${formatMiB(estimate.gpuBytes)}` })}
            </div>`);
    }
    if (Number.isFinite(estimate.ramBytes) && Number.isFinite(memory.availableBytes)) {
        parts.push(`
            <div class="local-llm-estimate-row">
                <span class="settings-card-meta">RAM about ${escapeHtml(formatMiB(estimate.ramBytes))} of ${escapeHtml(formatMiB(memory.availableBytes))} available</span>
                ${meter({ label: 'Estimated RAM', value: estimate.ramBytes, max: memory.availableBytes, text: `RAM about ${formatMiB(estimate.ramBytes)}` })}
            </div>`);
    }
    const warnings = Array.isArray(admission?.warnings) ? admission.warnings : [];
    const summary = [admissionLabel(admission), estimate.basis ? `estimate from ${estimate.basis}` : '', ...warnings].filter(Boolean).join(' · ');
    return `
        ${parts.join('')}
        ${context ? `<div class="settings-card-meta">Context: ${escapeHtml(contextLabel(context))}</div>` : ''}
        ${summary ? `<div class="settings-card-meta">${escapeHtml(summary)}</div>` : ''}`;
}
