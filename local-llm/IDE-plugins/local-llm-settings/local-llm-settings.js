import {
    POLL_INTERVAL_MS,
    admissionLabel,
    confirmMessage,
    contextLabel,
    downloadLabel,
    escapeHtml,
    estimateLabel,
    fieldsFromSchema,
    formatBytes,
    formatDuration,
    formatMiB,
    mergeLogs,
    modelEntryFromForm,
    newRequestId,
    paramsFromForm,
    parseToolResult,
    pollBackoff,
    progressPercent,
    runnerLabel,
    runnerOptions,
    shouldPoll,
} from './local-llm-settings-model.js';

const TABS = Object.freeze(['models', 'deployment']);
const PREVIEW_DELAY_MS = 400;
const ACTIVE_PHASES = new Set(['downloading', 'verifying', 'pulling', 'starting', 'loading', 'ready', 'stopping']);

async function callLocalLlm(tool, args = {}) {
    const client = window.webSkel?.appServices?.getClient?.('local-llm');
    if (!client?.callTool) throw new Error('The local-llm agent is unavailable. Enable it from the Marketplace first.');
    return parseToolResult(await client.callTool(tool, args));
}

// Explorer's settings loader registers the first exported function of this
// module as the presenter, so this class is the module's only export.
export class LocalLlmSettings {
    constructor(element, invalidate) {
        this.element = element;
        this.invalidate = invalidate;
        this.overview = null;
        this.status = null;
        this.logs = [];
        this.nextSeq = 0;
        this.activeTab = 'models';
        this.busy = false;
        this.runModelId = '';
        this.runFields = [];
        this.pollTimer = null;
        this.polling = false;
        this.pollAgain = false;
        this.pollFailures = 0;
        this.previewTimer = null;
        this.previewSerial = 0;
        this.closed = false;
        this.invalidate();
    }

    beforeRender() {}

    async afterRender() {
        this.statusLine = this.element.querySelector('#localLlmStatus');
        this.hardware = this.element.querySelector('#localLlmHardware');
        this.tabList = this.element.querySelector('#localLlmTabs');
        this.fullscreenButton = this.element.querySelector('[data-llm-fullscreen]');
        this.tabs = new Map(TABS.map((tab) => [tab, this.element.querySelector(`[data-llm-tab="${tab}"]`)]));
        this.panels = new Map(TABS.map((tab) => [tab, this.element.querySelector(`[data-llm-panel="${tab}"]`)]));
        this.modelsRegion = this.element.querySelector('[data-llm-models]');
        this.addForm = this.element.querySelector('[data-llm-form="add"]');
        this.addButton = this.element.querySelector('[data-llm-add]');
        this.sourceKind = this.element.querySelector('#localLlmSourceKind');
        this.runForm = this.element.querySelector('[data-llm-form="run"]');
        this.runnerSelect = this.element.querySelector('#localLlmRunner');
        this.promptForm = this.element.querySelector('[data-llm-form="prompt"]');
        this.promptResult = this.element.querySelector('[data-llm-prompt-result]');
        this.deploymentCard = this.element.querySelector('[data-llm-deployment]');
        this.logView = this.element.querySelector('[data-llm-log]');
        this.gatewayNote = this.element.querySelector('[data-llm-gateway]');

        this.addForm?.addEventListener('submit', (event) => this.submitAddModel(event));
        this.runForm?.addEventListener('submit', (event) => this.submitRun(event));
        this.promptForm?.addEventListener('submit', (event) => this.submitPrompt(event));
        this.sourceKind?.addEventListener('change', () => this.updateSourceFields());
        this.runnerSelect?.addEventListener('change', () => { void this.renderRunFields(); });
        this.runForm?.addEventListener('input', (event) => {
            if (event.target?.closest?.('[data-run-params]')) this.schedulePreview();
        });
        this.runForm?.addEventListener('change', (event) => {
            if (event.target?.closest?.('[data-run-params]')) this.schedulePreview();
        });
        this.tabList?.addEventListener('keydown', (event) => this.handleTabKeydown(event));

        await Promise.all([...this.element.querySelectorAll('custom-select')]
            .map((select) => select.presenterReadyPromise)
            .filter(Boolean));
        this.updateTabUI();
        this.updateSourceFields();
        await this.refresh();
    }

    afterUnload() {
        this.closed = true;
        this.stopPolling();
        clearTimeout(this.previewTimer);
    }

    // ------------------------------------------------------------ plumbing

    setStatus(message = '', type = '') {
        if (!this.statusLine) return;
        this.statusLine.textContent = message;
        for (const state of ['loading', 'success', 'error']) {
            this.statusLine.classList.toggle(state, type === state);
        }
    }

    setBusy(value) {
        this.busy = Boolean(value);
        this.modelsRegion?.setAttribute('aria-busy', this.busy ? 'true' : 'false');
        for (const control of this.element.querySelectorAll('button, input, textarea')) {
            if (!control.matches('[data-window-control]')) control.disabled = this.busy;
        }
        for (const select of this.element.querySelectorAll('custom-select')) {
            select.toggleAttribute('disabled', this.busy);
            select.webSkelPresenter?.applyDisabledState?.();
        }
        if (!this.busy) this.updateDeploymentButtons();
    }

    async setSelectOptions(select, options, selectedValue = '') {
        if (!select) return;
        await select.presenterReadyPromise;
        select.setAttribute('data-options', encodeURIComponent(JSON.stringify(options)));
        select.setAttribute('data-selected', selectedValue);
        select.webSkelPresenter?.setOptions?.(options, selectedValue);
        select.value = selectedValue;
    }

    async withBusy(progress, action) {
        if (this.busy) return undefined;
        this.setBusy(true);
        this.setStatus(progress, 'loading');
        try {
            return await action();
        } catch (error) {
            this.setStatus(error?.message || 'The request failed.', 'error');
            return undefined;
        } finally {
            this.setBusy(false);
        }
    }

    // ------------------------------------------------------------ overview

    async refresh() {
        const done = await this.withBusy('Loading models and hardware…', async () => {
            await this.loadOverview();
            this.setStatus('');
            return true;
        });
        if (done) this.startPollingIfActive();
    }

    async refreshOverview() {
        await this.refresh();
    }

    async loadOverview() {
        this.overview = await callLocalLlm('local_llm_overview');
        this.renderHardware();
        this.renderModels();
        this.status = { ...(this.status || {}), phase: this.overview.deployment?.phase || 'idle', deployment: this.overview.deployment };
        this.renderDeployment();
        if (this.gatewayNote && this.overview.gatewayModel) {
            this.gatewayNote.textContent = `Other agents use the running model through Soul Gateway as ${this.overview.gatewayModel}.`;
        }
    }

    renderHardware() {
        if (!this.hardware) return;
        const hardware = this.overview?.hardware || {};
        const gpu = hardware.gpu || {};
        const parts = [];
        if (gpu.available) {
            parts.push(`${gpu.name || 'GPU'}: ${formatMiB(gpu.usedBytes)} used of ${formatMiB(gpu.totalBytes)}`);
            if (Array.isArray(gpu.processes) && gpu.processes.length) {
                parts.push(`in use by ${gpu.processes.map((process) => process.name).join(', ')}`);
            }
        } else {
            parts.push(gpu.reason || 'No GPU is available to this agent.');
        }
        if (hardware.memory?.availableBytes) parts.push(`RAM ${formatMiB(hardware.memory.availableBytes)} available`);
        if (hardware.disk?.freeBytes) parts.push(`disk ${formatBytes(hardware.disk.freeBytes)} free`);
        this.hardware.innerHTML = `
            <div class="settings-card-title">Hardware</div>
            <div class="settings-card-meta">${parts.map(escapeHtml).join(' · ')}</div>
        `;
    }

    renderModels() {
        if (!this.modelsRegion) return;
        const models = Array.isArray(this.overview?.models) ? this.overview.models : [];
        if (!models.length) {
            this.modelsRegion.innerHTML = '<div class="settings-empty-state">No models in the catalog.</div>';
            return;
        }
        const activeModel = ACTIVE_PHASES.has(this.overview?.deployment?.phase) ? this.overview.deployment.modelId : '';
        const rows = models.map((model) => {
            const id = escapeHtml(model.id);
            const chips = [
                model.architecture === 'moe' ? '<span class="settings-chip">MoE</span>' : '',
                model.seed ? '<span class="settings-chip">catalog</span>' : '<span class="settings-chip">added</span>',
                model.id === activeModel ? '<span class="status-badge success">in use</span>' : '',
            ].join('');
            const sizes = [model.totalParams, model.activeParams ? `${model.activeParams} active` : '']
                .filter(Boolean).map(escapeHtml).join(' · ');
            const runnerCells = ['llama.cpp', 'ollama'].map((runnerId) => {
                const entry = model.runners?.[runnerId];
                if (!entry) return '<td class="settings-card-meta">—</td>';
                const deletable = entry.download && ['complete', 'partial'].includes(entry.download.state);
                return `
                    <td>
                        <div>${escapeHtml(downloadLabel(entry.download, entry.size))}</div>
                        <div class="settings-card-meta">${escapeHtml(admissionLabel(entry.admission))}</div>
                        ${deletable ? `<button type="button" class="gray-button local-llm-inline-button" data-local-action="deleteWeights ${id} ${escapeHtml(runnerId)}">Delete weights</button>` : ''}
                    </td>`;
            }).join('');
            return `
                <tr>
                    <td>
                        <div class="local-llm-model-name">
                            <span class="settings-card-title">${escapeHtml(model.displayName || model.id)}</span>
                            ${chips}
                        </div>
                        <div class="settings-card-meta">${sizes || escapeHtml(model.id)}</div>
                    </td>
                    <td>${escapeHtml(model.license || '—')}</td>
                    ${runnerCells}
                    <td>
                        <div class="local-llm-row-actions">
                            <button type="button" class="general-button" data-local-action="openRun ${id}">Run…</button>
                            ${model.seed ? '' : `<button type="button" class="gray-button" data-local-action="removeModel ${id}">Remove</button>`}
                        </div>
                    </td>
                </tr>`;
        }).join('');
        this.modelsRegion.innerHTML = `
            <div class="local-llm-table-scroll">
                <table class="local-llm-table">
                    <thead>
                        <tr>
                            <th scope="col">Model</th>
                            <th scope="col">Licence</th>
                            <th scope="col">llama.cpp</th>
                            <th scope="col">Ollama</th>
                            <th scope="col"><span class="local-llm-visually-hidden">Actions</span></th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    }

    findModel(modelId) {
        return (this.overview?.models || []).find((model) => model.id === modelId) || null;
    }

    // ------------------------------------------------------------ tabs

    updateTabUI() {
        for (const tab of TABS) {
            const active = tab === this.activeTab;
            const button = this.tabs.get(tab);
            button?.classList.toggle('active', active);
            button?.setAttribute('aria-selected', active ? 'true' : 'false');
            button?.setAttribute('tabindex', active ? '0' : '-1');
            const panel = this.panels.get(tab);
            if (panel) panel.hidden = !active;
        }
    }

    switchTab(_target, tab) {
        if (!TABS.includes(tab)) return;
        this.activeTab = tab;
        this.updateTabUI();
    }

    handleTabKeydown(event) {
        const current = event.target?.closest?.('[data-llm-tab]');
        if (!current || !this.tabList?.contains(current)) return;
        const index = TABS.indexOf(current.dataset.llmTab);
        let next = '';
        if (event.key === 'ArrowRight') next = TABS[(index + 1) % TABS.length];
        else if (event.key === 'ArrowLeft') next = TABS[(index - 1 + TABS.length) % TABS.length];
        else if (event.key === 'Home') next = TABS[0];
        else if (event.key === 'End') next = TABS[TABS.length - 1];
        if (!next) return;
        event.preventDefault();
        this.switchTab(null, next);
        this.tabs.get(next)?.focus();
    }

    // ------------------------------------------------------------ add model

    openAddModel() {
        if (this.busy || !this.addForm) return;
        this.runForm.hidden = true;
        this.addForm.hidden = false;
        this.addButton?.setAttribute('aria-expanded', 'true');
        this.addForm.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
        this.addForm.elements.id?.focus();
    }

    cancelAddModel() {
        if (!this.addForm) return;
        this.addForm.reset();
        this.addForm.hidden = true;
        this.addButton?.setAttribute('aria-expanded', 'false');
        this.updateSourceFields();
    }

    updateSourceFields() {
        const kind = this.sourceKind?.value === 'ollama' ? 'ollama' : 'huggingface';
        for (const field of this.element.querySelectorAll('[data-source-field]')) {
            const active = field.dataset.sourceField === kind;
            field.hidden = !active;
            const input = field.querySelector('input');
            if (input) input.required = active && ['repo', 'file', 'tag'].includes(input.name);
        }
    }

    async submitAddModel(event) {
        event.preventDefault();
        if (this.busy || !this.addForm?.reportValidity()) return;
        const raw = Object.fromEntries(new FormData(this.addForm));
        raw.sourceKind = this.sourceKind?.value || 'huggingface';
        raw.architecture = this.element.querySelector('#localLlmArchitecture')?.value || 'dense';
        const entry = modelEntryFromForm(raw);
        const added = await this.withBusy(`Adding ${entry.id} and pinning its source…`, async () => {
            await callLocalLlm('local_llm_model_add', { model: entry });
            await this.loadOverview();
            return true;
        });
        if (added) {
            this.cancelAddModel();
            this.setStatus(`${entry.id} was added. Nothing was downloaded.`, 'success');
        }
    }

    async removeModel(_target, modelId) {
        const model = this.findModel(modelId);
        if (!model || model.seed || this.busy) return;
        const confirmed = await assistOS.UI.showModal('confirm-action-modal', {
            message: confirmMessage(`Remove ${model.displayName || model.id} from the list? Downloaded weights are not deleted by this; delete them first if you want the disk space back.`),
        }, true);
        if (!confirmed) return;
        const removed = await this.withBusy(`Removing ${model.id}…`, async () => {
            await callLocalLlm('local_llm_model_remove', { modelId: model.id });
            await this.loadOverview();
            return true;
        });
        if (removed) this.setStatus(`${model.id} was removed.`, 'success');
    }

    async deleteWeights(_target, modelId, runnerId) {
        const model = this.findModel(modelId);
        const entry = model?.runners?.[runnerId];
        if (!entry || this.busy) return;
        const confirmed = await assistOS.UI.showModal('confirm-action-modal', {
            message: confirmMessage(`Delete the ${runnerLabel(runnerId)} weights of ${model.displayName || model.id} (${formatBytes(entry.download?.bytes ?? entry.size)})? The next Run downloads them again.`),
        }, true);
        if (!confirmed) return;
        const deleted = await this.withBusy('Deleting weights…', async () => {
            const result = await callLocalLlm('local_llm_weights_delete', { modelId, runnerId });
            await this.loadOverview();
            return result;
        });
        if (deleted) this.setStatus(`Deleted ${formatBytes(deleted.freedBytes)}.`, 'success');
    }

    // ------------------------------------------------------------ run form

    async openRun(_target, modelId) {
        const model = this.findModel(modelId);
        if (!model || this.busy || !this.runForm) return;
        this.runModelId = model.id;
        this.addForm.hidden = true;
        this.addButton?.setAttribute('aria-expanded', 'false');
        const title = this.runForm.querySelector('[data-run-title]');
        if (title) title.textContent = `Run ${model.displayName || model.id}`;
        const options = runnerOptions(this.overview, model);
        const preferred = options.find((option) => model.runners[option.value]?.admission?.status === 'ok')
            || options.find((option) => this.overview.runners.find((runner) => runner.id === option.value)?.supported)
            || options[0];
        await this.setSelectOptions(this.runnerSelect, options, preferred?.value || '');
        this.runForm.hidden = false;
        await this.renderRunFields();
        this.runForm.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
    }

    cancelRun() {
        if (!this.runForm) return;
        this.runForm.hidden = true;
        this.runModelId = '';
        clearTimeout(this.previewTimer);
    }

    async renderRunFields() {
        const model = this.findModel(this.runModelId);
        const runnerId = this.runnerSelect?.value || '';
        const runner = (this.overview?.runners || []).find((entry) => entry.id === runnerId);
        const entry = model?.runners?.[runnerId];
        const container = this.runForm?.querySelector('[data-run-params]');
        const note = this.runForm?.querySelector('[data-run-runner-note]');
        const submit = this.runForm?.querySelector('[data-run-submit]');
        if (!container || !model || !runner) return;
        const runnable = runner.supported && runner.installed;
        if (note) {
            const reason = !runner.supported || !runner.installed ? runner.reason : '';
            note.textContent = [
                runner.version ? `${runnerLabel(runner.id)} ${runner.version}` : runnerLabel(runner.id),
                reason,
                entry?.download ? downloadLabel(entry.download, entry.size) : '',
            ].filter(Boolean).join(' · ');
        }
        if (submit) submit.disabled = !runnable;
        this.runFields = runnable ? fieldsFromSchema(runner.paramSchema, entry?.params || {}) : [];
        container.innerHTML = this.runFields.map((field) => this.renderField(field)).join('');
        this.renderEstimate({ context: entry?.context, admission: entry?.admission });
        // Selects inserted here get their presenter asynchronously; read them only once ready.
        await Promise.all([...container.querySelectorAll('custom-select')]
            .map((select) => select.presenterReadyPromise)
            .filter(Boolean));
        if (runnable) this.schedulePreview();
    }

    renderField(field) {
        const id = `localLlmParam-${field.name}`;
        const help = field.description ? `<span class="settings-card-meta">${escapeHtml(field.description)}</span>` : '';
        if (field.kind === 'fixed') {
            return `
                <div class="local-llm-field">
                    <span class="form-label">${escapeHtml(field.title)}</span>
                    <code class="local-llm-fixed-value">${escapeHtml(JSON.stringify(field.value ?? null))}</code>
                    ${help}
                </div>`;
        }
        if (field.kind === 'enum' || field.kind === 'boolean') {
            const selected = field.value === null || field.value === undefined ? '' : String(field.value);
            return `
                <label class="local-llm-field">
                    <span class="form-label">${escapeHtml(field.title)}</span>
                    <custom-select id="${escapeHtml(id)}"
                                   data-presenter="custom-select"
                                   aria-label="${escapeHtml(field.title)}"
                                   data-name="${escapeHtml(field.name)}"
                                   data-param="${escapeHtml(field.name)}"
                                   data-selected="${escapeHtml(selected)}"
                                   data-options="${escapeHtml(encodeURIComponent(JSON.stringify(field.options)))}"></custom-select>
                    ${help}
                </label>`;
        }
        const value = field.value === null || field.value === undefined ? '' : String(field.value);
        const bounds = field.kind === 'number'
            ? `type="number" ${Number.isFinite(field.min) ? `min="${field.min}"` : ''} ${Number.isFinite(field.max) ? `max="${field.max}"` : ''} step="${field.integer ? 1 : 'any'}"`
            : 'type="text"';
        return `
            <label class="local-llm-field">
                <span class="form-label">${escapeHtml(field.title)}</span>
                <input class="form-input" id="${escapeHtml(id)}" name="${escapeHtml(field.name)}" data-param="${escapeHtml(field.name)}" ${bounds}
                       value="${escapeHtml(value)}" ${field.nullable ? 'placeholder="Runner default"' : 'required'}>
                ${help}
            </label>`;
    }

    readRunForm() {
        const raw = {};
        for (const control of this.runForm?.querySelectorAll('[data-param]') || []) {
            raw[control.dataset.param] = control.value ?? '';
        }
        return paramsFromForm(this.runFields, raw);
    }

    renderEstimate({ context = null, admission = null, error = '' } = {}) {
        const contextLine = this.runForm?.querySelector('[data-run-context]');
        const admissionLine = this.runForm?.querySelector('[data-run-admission]');
        if (contextLine) contextLine.textContent = context ? `Context: ${contextLabel(context)}` : '';
        if (!admissionLine) return;
        if (error) {
            admissionLine.textContent = error;
            return;
        }
        const warnings = Array.isArray(admission?.warnings) ? admission.warnings : [];
        admissionLine.textContent = [admissionLabel(admission), estimateLabel(admission), ...warnings].filter(Boolean).join(' · ');
    }

    schedulePreview() {
        clearTimeout(this.previewTimer);
        this.previewTimer = setTimeout(() => { void this.previewRun(); }, PREVIEW_DELAY_MS);
    }

    async previewRun() {
        const modelId = this.runModelId;
        const runnerId = this.runnerSelect?.value || '';
        if (!modelId || !runnerId || this.closed) return;
        const { params, errors } = this.readRunForm();
        if (errors.length) {
            this.renderEstimate({ error: errors.join(' ') });
            return;
        }
        const serial = ++this.previewSerial;
        try {
            const result = await callLocalLlm('local_llm_overview', { preview: { modelId, runnerId, params } });
            if (serial !== this.previewSerial || this.runModelId !== modelId) return;
            const preview = result.preview || {};
            this.renderEstimate(preview.error ? { error: preview.error } : preview);
        } catch (error) {
            if (serial === this.previewSerial) this.renderEstimate({ error: error?.message || 'The estimate is unavailable.' });
        }
    }

    async submitRun(event) {
        event.preventDefault();
        const model = this.findModel(this.runModelId);
        const runnerId = this.runnerSelect?.value || '';
        if (!model || !runnerId || this.busy || !this.runForm?.reportValidity()) return;
        const runner = (this.overview?.runners || []).find((entry) => entry.id === runnerId);
        if (!runner?.supported || !runner.installed) {
            this.setStatus(runner?.reason || `${runnerLabel(runnerId)} cannot run models in this release.`, 'error');
            return;
        }
        const { params, errors } = this.readRunForm();
        if (errors.length) {
            this.setStatus(errors.join(' '), 'error');
            return;
        }
        let replace = false;
        const current = this.status?.deployment || this.overview?.deployment;
        if (current && ACTIVE_PHASES.has(current.phase)) {
            replace = await assistOS.UI.showModal('confirm-action-modal', {
                message: confirmMessage(`${current.modelId} is ${current.phase} on ${runnerLabel(current.runnerId)}. Stop it and run ${model.displayName || model.id} instead?`),
            }, true);
            if (!replace) return;
        }
        const started = await this.withBusy(`Starting ${model.displayName || model.id} on ${runnerLabel(runnerId)}…`, async () => {
            // A fresh request id per press: a retried call is a no-op on the server.
            const result = await callLocalLlm('local_llm_run', {
                requestId: newRequestId(), modelId: model.id, runnerId, params, replace: Boolean(replace),
            });
            this.status = { phase: result.deployment?.phase, deployment: result.deployment };
            this.logs = [];
            this.nextSeq = 0;
            return result;
        });
        if (!started) return;
        this.cancelRun();
        this.setStatus(started.duplicate ? 'That run was already requested.' : 'Run accepted.', 'success');
        this.switchTab(null, 'deployment');
        this.renderDeployment();
        this.startPollingIfActive(true);
    }

    // ------------------------------------------------------------ deployment

    // Status polling is single-flight: one timer and at most one request at a
    // time, however many actions (Run, Send, Stop, Cancel) ask for a poll.
    schedulePoll(delay = POLL_INTERVAL_MS) {
        if (this.closed) return;
        clearTimeout(this.pollTimer);
        this.pollTimer = setTimeout(() => {
            this.pollTimer = null;
            void this.poll();
        }, delay);
    }

    startPollingIfActive(force = false) {
        const phase = this.status?.phase || this.overview?.deployment?.phase;
        if (this.closed || !(force || shouldPoll(phase))) return;
        if (this.polling) {
            this.pollAgain = true;
            return;
        }
        if (force) this.schedulePoll(0);
        else if (!this.pollTimer) this.schedulePoll(POLL_INTERVAL_MS);
    }

    stopPolling() {
        clearTimeout(this.pollTimer);
        this.pollTimer = null;
    }

    async poll() {
        if (this.closed || !this.element.isConnected) return;
        if (this.polling) {
            this.pollAgain = true;
            return;
        }
        this.stopPolling();
        this.polling = true;
        let phase = '';
        let failed = false;
        try {
            const status = await callLocalLlm('local_llm_status', { sinceSeq: this.nextSeq });
            this.status = status;
            this.logs = mergeLogs(this.logs, status.logs || []);
            this.nextSeq = status.nextSeq ?? this.nextSeq;
            phase = status.phase;
            this.pollFailures = 0;
            this.renderDeployment();
        } catch (error) {
            // A transient failure (a Router restart, a timeout) must not end
            // polling mid-download: retry with a backoff while the modal is open.
            failed = true;
            this.pollFailures += 1;
            this.setStatus(error?.message || 'Status is unavailable.', 'error');
        } finally {
            this.polling = false;
        }
        if (this.closed || !this.element.isConnected) return;
        if (this.pollAgain) {
            this.pollAgain = false;
            this.schedulePoll(0);
        } else if (failed) {
            this.schedulePoll(pollBackoff(this.pollFailures));
        } else if (shouldPoll(phase)) {
            this.schedulePoll(POLL_INTERVAL_MS);
        } else if (!this.busy) {
            // Download states and admission change when a job settles.
            await this.loadOverview().catch(() => {});
        }
    }

    updateDeploymentButtons() {
        const phase = this.status?.phase || 'idle';
        const stop = this.element.querySelector('[data-llm-stop]');
        const cancel = this.element.querySelector('[data-llm-cancel]');
        const send = this.element.querySelector('[data-llm-send]');
        // Stop also clears a failed or paused deployment.
        if (stop) stop.disabled = this.busy || !(ACTIVE_PHASES.has(phase) || ['error', 'paused'].includes(phase));
        if (cancel) cancel.disabled = this.busy || !['downloading', 'verifying', 'pulling'].includes(phase);
        if (send) send.disabled = this.busy || phase !== 'ready';
    }

    renderDeployment() {
        this.updateDeploymentButtons();
        if (!this.deploymentCard) return;
        const status = this.status || {};
        const deployment = status.deployment;
        const phase = status.phase || deployment?.phase || 'idle';
        if (!deployment || phase === 'idle') {
            const last = deployment ? ` The last run was ${escapeHtml(deployment.modelId)} on ${escapeHtml(runnerLabel(deployment.runnerId))}.` : '';
            this.deploymentCard.innerHTML = `<div class="settings-empty-state">No model is running. Choose Run on the Models tab.${last}</div>`;
            this.renderLogs();
            return;
        }
        const badge = phase === 'ready' ? 'success' : phase === 'error' ? 'error' : '';
        const download = deployment.download || null;
        const percent = progressPercent(download);
        const rows = [];
        if (download && ['downloading', 'verifying', 'pulling', 'paused'].includes(phase)) {
            const rate = download.rate ? `${formatBytes(download.rate)}/s` : '';
            const eta = Number.isFinite(download.etaSeconds) ? `about ${formatDuration(download.etaSeconds)} left` : '';
            rows.push(`
                <div class="local-llm-progress">
                    <progress max="100" value="${percent ?? 0}" aria-label="Download progress">${percent ?? 0}%</progress>
                    <span class="settings-card-meta">${escapeHtml([`${formatBytes(download.bytes)} of ${formatBytes(download.total)}`, rate, eta].filter(Boolean).join(' · '))}</span>
                </div>`);
        }
        const gpu = status.gpu;
        if (gpu?.available) rows.push(`<div class="settings-card-meta">GPU memory: ${escapeHtml(formatMiB(gpu.usedBytes))} used of ${escapeHtml(formatMiB(gpu.totalBytes))}</div>`);
        const report = status.runnerReport || {};
        const reportParts = [
            report.device ? `device ${report.device}` : '',
            report.offloaded ? `${report.offloaded.layers}/${report.offloaded.of} layers on the GPU` : '',
            Number.isFinite(report.totalMiB) ? `runner buffers ${report.totalMiB.toLocaleString('en-US')} MiB` : '',
        ].filter(Boolean);
        if (reportParts.length) rows.push(`<div class="settings-card-meta">Runner: ${escapeHtml(reportParts.join(' · '))}</div>`);
        if (status.context) rows.push(`<div class="settings-card-meta">Context: ${escapeHtml(contextLabel(status.context))}</div>`);
        const last = status.lastCompletion;
        if (last && Number.isFinite(last.generationTokensPerSecond)) {
            rows.push(`<div class="settings-card-meta">Last completion: ${escapeHtml(String(last.completionTokens ?? '?'))} tokens at ${escapeHtml(last.generationTokensPerSecond.toFixed(1))} tokens/s (${escapeHtml(last.source)})</div>`);
        }
        if (deployment.error) rows.push(`<div class="settings-status error">${escapeHtml(deployment.error)}</div>`);
        this.deploymentCard.innerHTML = `
            <div class="local-llm-deployment-head">
                <div>
                    <div class="settings-card-title">${escapeHtml(deployment.modelId)} on ${escapeHtml(runnerLabel(deployment.runnerId))}</div>
                    <div class="settings-card-meta">Estimates shown before Run are approximate; these values come from the runner.</div>
                </div>
                <span class="status-badge ${badge}">${escapeHtml(phase)}</span>
            </div>
            ${rows.join('')}
        `;
        this.renderLogs();
    }

    renderLogs() {
        if (!this.logView) return;
        const atBottom = this.logView.scrollTop + this.logView.clientHeight >= this.logView.scrollHeight - 8;
        this.logView.textContent = this.logs.map((entry) => `[${entry.stream}] ${entry.line}`).join('\n') || 'No runner output yet.';
        if (atBottom) this.logView.scrollTop = this.logView.scrollHeight;
    }

    async stopDeployment() {
        const stopped = await this.withBusy('Stopping…', async () => {
            await callLocalLlm('local_llm_stop');
            return true;
        });
        if (stopped) {
            this.setStatus('Stopped.', 'success');
            this.startPollingIfActive(true);
        }
    }

    async cancelDownload() {
        const cancelled = await this.withBusy('Cancelling the download…', async () => {
            await callLocalLlm('local_llm_download_cancel');
            return true;
        });
        if (cancelled) {
            this.setStatus('Download paused. The next Run resumes it.', 'success');
            this.startPollingIfActive(true);
        }
    }

    async submitPrompt(event) {
        event.preventDefault();
        if (this.busy || !this.promptForm?.reportValidity()) return;
        const prompt = String(this.promptForm.elements.prompt?.value || '');
        const maxTokens = Number(this.promptForm.elements.maxTokens?.value || 256);
        const result = await this.withBusy('Waiting for the model…', async () => callLocalLlm('local_llm_test_prompt', { prompt, maxTokens }));
        if (!result || !this.promptResult) return;
        const speed = Number.isFinite(result.generationTokensPerSecond)
            ? `${result.generationTokensPerSecond.toFixed(1)} tokens/s generation`
            : 'speed not reported';
        const promptSpeed = Number.isFinite(result.promptTokensPerSecond) ? ` · ${result.promptTokensPerSecond.toFixed(1)} tokens/s prompt` : '';
        this.promptResult.innerHTML = `
            <pre class="local-llm-answer">${escapeHtml(result.text || '(empty answer)')}</pre>
            <div class="settings-card-meta">${escapeHtml(`${speed}${promptSpeed} · ${result.completionTokens ?? '?'} tokens in ${result.elapsedMs} ms (${result.statsSource || 'no stats'})`)}</div>
        `;
        this.setStatus('');
        await this.poll();
    }

    // ------------------------------------------------------------ window

    toggleFullscreen() {
        const dialog = this.element.closest('dialog');
        if (!dialog) return;
        const isFullscreen = !dialog.classList.contains('is-fullscreen');
        dialog.classList.toggle('is-fullscreen', isFullscreen);
        this.fullscreenButton?.setAttribute('aria-pressed', isFullscreen ? 'true' : 'false');
    }

    closeModal() {
        this.closed = true;
        this.stopPolling();
        assistOS.UI.closeModal(this.element);
    }
}
