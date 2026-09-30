import {
    POLL_INTERVAL_MS,
    confirmMessage,
    downloadLabel,
    escapeHtml,
    fieldsFromSchema,
    formatBytes,
    lookupCommitFor,
    lookupFormatFor,
    mergeLogs,
    modelEntryFromForm,
    newRequestId,
    paramsFromForm,
    parseToolResult,
    pollBackoff,
    rememberRunners,
    runnerLabel,
    runnerOptions,
    shouldPoll,
    sizingSourceFor,
    stripBidi,
    suggestModelId,
} from '../../../local-llm-settings/local-llm-settings-model.js';
import {
    ACTIVE_PHASES,
    detailInfoHtml,
    estimateHtml,
    hasWeightsOnDisk,
    hardwareCardsHtml,
    installMessage,
    lookupResultsHtml,
    modelsTableHtml,
    runnersIntroText,
    runnersPanelHtml,
    splitRunFields,
    statusCardHtml,
} from './local-llm-dashboard-view.js';

const TABS = Object.freeze(['models', 'playground', 'logs', 'runners']);
const PREVIEW_DELAY_MS = 400;
// While the dashboard is open and visible, hardware and download states are
// re-read on this interval, so the GPU card never shows a stale value.
export const OVERVIEW_REFRESH_MS = 7000;
// While a runner installs, its progress is read more often.
const INSTALL_REFRESH_MS = 1500;

async function callLocalLlm(tool, args = {}) {
    const client = window.webSkel?.appServices?.getClient?.('local-llm');
    if (!client?.callTool) throw new Error('The local-llm agent is unavailable. Enable it from the Marketplace first.');
    return parseToolResult(await client.callTool(tool, args));
}

export class LocalLlmDashboard {
    constructor(element, invalidate) {
        this.element = element;
        this.invalidate = invalidate;
        this.overview = null;
        this.status = null;
        this.logs = [];
        this.nextSeq = 0;
        this.activeTab = 'models';
        this.busy = false;
        this.selectedModelId = '';
        this.detailMode = 'empty';
        this.runModelId = '';
        this.runFields = [];
        this.pollTimer = null;
        this.polling = false;
        this.pollAgain = false;
        this.pollFailures = 0;
        this.previewTimer = null;
        this.previewSerial = 0;
        this.overviewTimer = null;
        // The Add model form's lookup: its last result and request, what it filled in (to tell hand-edited values from
        // looked-up ones), and which request is current. A lookup never makes the dashboard busy, so Stop and Cancel stay usable.
        this.lookup = null;
        this.lookupSizing = null;
        this.lookupSerial = 0;
        this.lookupPending = false;
        this.rendered = {};
        this.closed = false;
        this.onVisibilityChange = () => {
            if (!globalThis.document?.hidden) this.scheduleOverviewRefresh(0);
        };
        this.invalidate();
    }

    beforeRender() {}

    async afterRender() {
        const find = (selector) => this.element.querySelector(selector);
        this.statusLine = find('#localLlmStatus');
        this.hardware = find('[data-llm-hardware]');
        this.deploymentCard = find('[data-llm-deployment]');
        this.tabList = find('#localLlmTabs');
        this.tabs = new Map(TABS.map((tab) => [tab, find(`[data-llm-tab="${tab}"]`)]));
        this.panels = new Map(TABS.map((tab) => [tab, find(`[data-llm-panel="${tab}"]`)]));
        this.modelsRegion = find('[data-llm-models]');
        this.detailEmpty = find('[data-llm-detail-empty]');
        this.detailInfo = find('[data-llm-detail-info]');
        this.addForm = find('[data-llm-form="add"]');
        this.addButton = find('[data-llm-add]');
        this.lookupRegion = find('[data-llm-lookup]');
        this.lookupButton = find('[data-llm-lookup-button]');
        this.sizingLabel = find('[data-llm-sizing-label]');
        this.sourceKind = find('#localLlmSourceKind');
        this.runForm = find('[data-llm-form="run"]');
        this.runnerSelect = find('#localLlmRunner');
        this.advanced = find('[data-run-advanced]');
        this.promptForm = find('[data-llm-form="prompt"]');
        this.promptResult = find('[data-llm-prompt-result]');
        this.promptNote = find('[data-llm-prompt-note]');
        this.logView = find('[data-llm-log]');
        this.runnersRegion = find('[data-llm-runners]');
        this.runnersIntro = find('[data-llm-runners-intro]');
        this.follow = find('[data-llm-follow]');
        this.gatewayNote = find('[data-llm-gateway]');

        this.addForm?.addEventListener('submit', (event) => this.submitAddModel(event));
        this.runForm?.addEventListener('submit', (event) => this.submitRun(event));
        this.promptForm?.addEventListener('submit', (event) => this.submitPrompt(event));
        this.sourceKind?.addEventListener('change', () => { this.resetLookup(); this.updateSourceFields(); });
        // A results list for another repository or revision would mislead.
        this.addForm?.addEventListener('input', (event) => {
            if (['repo', 'revision'].includes(event.target?.name)) this.resetLookup();
        });
        this.runnerSelect?.addEventListener('change', () => { void this.renderRunFields(); });
        for (const type of ['input', 'change']) {
            this.runForm?.addEventListener(type, (event) => {
                if (event.target?.closest?.('[data-param]')) this.schedulePreview();
            });
        }
        this.tabList?.addEventListener('keydown', (event) => this.handleTabKeydown(event));
        this.modelsRegion?.addEventListener('keydown', (event) => this.handleRowKeydown(event));
        this.follow?.addEventListener('change', () => this.renderLogs(true));
        globalThis.document?.addEventListener?.('visibilitychange', this.onVisibilityChange);

        await Promise.all([...this.element.querySelectorAll('custom-select')]
            .map((select) => select.presenterReadyPromise)
            .filter(Boolean));
        this.updateTabUI();
        this.updateSourceFields();
        this.updateDetailMode();
        await this.refresh();
        this.scheduleOverviewRefresh();
    }

    afterUnload() {
        this.closed = true;
        this.stopPolling();
        clearTimeout(this.previewTimer);
        clearTimeout(this.overviewTimer);
        this.overviewTimer = null;
        globalThis.document?.removeEventListener?.('visibilitychange', this.onVisibilityChange);
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
            if (!control.matches('[data-llm-tab]')) control.disabled = this.busy;
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

    // Replace a region's markup only when it changed, so a live refresh keeps
    // focus and scroll position when nothing moved.
    setHtml(key, element, html) {
        if (!element || this.rendered[key] === html) return false;
        this.rendered[key] = html;
        element.innerHTML = html;
        return true;
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
        rememberRunners(this.overview?.runners);
        const deployment = this.overview.deployment || null;
        // A status poll carries logs and runner details; keep them unless the
        // overview reports a different deployment.
        if (!this.status || this.status.deployment?.id !== deployment?.id || !shouldPoll(this.status.phase)) {
            this.status = { ...(this.status || {}), phase: deployment?.phase || 'idle', deployment };
        }
        this.renderHardware();
        this.renderModels();
        this.renderDetailInfo();
        this.renderRunners();
        this.renderDeployment();
        if (this.gatewayNote && this.overview.gatewayModel) {
            this.gatewayNote.textContent = 'Sends one prompt to the running model inside this agent and shows the runner\'s speed. '
                + `Other agents use the running model through Soul Gateway as ${this.overview.gatewayModel}.`;
        }
    }

    scheduleOverviewRefresh(delay = OVERVIEW_REFRESH_MS) {
        if (this.closed) return;
        clearTimeout(this.overviewTimer);
        this.overviewTimer = setTimeout(() => {
            this.overviewTimer = null;
            void this.liveRefresh();
        }, delay);
    }

    async liveRefresh() {
        if (this.closed || this.element.isConnected === false) return;
        if (!globalThis.document?.hidden && !this.busy) {
            await this.loadOverview().catch(() => {});
        }
        if (!globalThis.document?.hidden) {
            const installing = (this.overview?.runners || []).some((runner) => runner.install?.installing);
            this.scheduleOverviewRefresh(installing ? INSTALL_REFRESH_MS : OVERVIEW_REFRESH_MS);
        }
    }

    renderRunners() {
        const runners = this.overview?.runners || [];
        const intro = runnersIntroText(runners, { profile: this.overview?.profile });
        if (this.runnersIntro && this.rendered.runnersIntro !== intro) {
            this.rendered.runnersIntro = intro;
            this.runnersIntro.textContent = intro;
        }
        this.setHtml('runners', this.runnersRegion, runnersPanelHtml(runners));
    }

    async installRunner(_target, runnerId) {
        const runner = (this.overview?.runners || []).find((entry) => entry.id === runnerId);
        const install = runner?.install;
        if (!install || this.busy) return;
        const licence = install.licence || {};
        const confirmed = await assistOS.UI.showModal('confirm-action-modal', { message: confirmMessage(installMessage(runner)) }, true);
        if (!confirmed) return;
        const started = await this.withBusy(`Installing ${runner.displayName || runner.id}…`, async () => {
            const result = await callLocalLlm('local_llm_runner_install', { runnerId, acceptLicence: Boolean(licence.requiresAcceptance) });
            await this.loadOverview();
            return result;
        });
        if (started) {
            this.setStatus(`Installing ${runner.displayName || runner.id}; progress shows under Runners.`, 'success');
            this.scheduleOverviewRefresh(INSTALL_REFRESH_MS);
        }
    }

    async uninstallRunner(_target, runnerId) {
        const runner = (this.overview?.runners || []).find((entry) => entry.id === runnerId);
        if (!runner?.install || this.busy) return;
        const confirmed = await assistOS.UI.showModal('confirm-action-modal', {
            message: confirmMessage(`Uninstall ${runner.displayName || runner.id}? Its downloaded files are deleted; installing it again downloads them again.`),
        }, true);
        if (!confirmed) return;
        const removed = await this.withBusy(`Uninstalling ${runner.displayName || runner.id}…`, async () => {
            const result = await callLocalLlm('local_llm_runner_uninstall', { runnerId });
            await this.loadOverview();
            return result;
        });
        if (removed) this.setStatus(`Uninstalled ${runner.displayName || runner.id}; freed ${formatBytes(removed.freedBytes)}.`, 'success');
    }

    activeModelId() {
        const deployment = this.status?.deployment || this.overview?.deployment;
        const phase = this.status?.phase || deployment?.phase;
        return ACTIVE_PHASES.has(phase) ? deployment?.modelId || '' : '';
    }

    renderHardware() {
        const gpu = shouldPoll(this.status?.phase) ? this.status?.gpu : null;
        const { profile = null, profileDecision = null, limits = null } = this.overview || {};
        this.setHtml('hardware', this.hardware, hardwareCardsHtml(this.overview?.hardware || {}, gpu, { profile, decision: profileDecision, limits }));
    }

    renderModels() {
        const models = Array.isArray(this.overview?.models) ? this.overview.models : [];
        this.setHtml('models', this.modelsRegion, modelsTableHtml(models, {
            selectedId: this.selectedModelId,
            activeModelId: this.activeModelId(),
            runners: this.overview?.runners || [],
        }));
    }

    renderDetailInfo() {
        const model = this.findModel(this.selectedModelId);
        if (this.selectedModelId && !model) {
            this.selectedModelId = '';
            this.detailMode = this.detailMode === 'add' ? 'add' : 'empty';
            this.updateDetailMode();
        }
        this.setHtml('detail', this.detailInfo, detailInfoHtml(model, {
            runners: this.overview?.runners || [],
            activeModelId: this.activeModelId(),
        }));
    }

    findModel(modelId) {
        return (this.overview?.models || []).find((model) => model.id === modelId) || null;
    }

    // ------------------------------------------------------------ tabs

    updateTabUI() {
        for (const tab of TABS) {
            const active = tab === this.activeTab;
            const button = this.tabs?.get(tab);
            button?.classList.toggle('active', active);
            button?.setAttribute('aria-selected', active ? 'true' : 'false');
            button?.setAttribute('tabindex', active ? '0' : '-1');
            const panel = this.panels?.get(tab);
            if (panel) panel.hidden = !active;
        }
        if (this.activeTab === 'logs') this.renderLogs(true);
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

    // ------------------------------------------------------------ selection

    handleRowKeydown(event) {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const row = event.target?.closest?.('tr[data-model-id]');
        if (!row) return;
        event.preventDefault();
        void this.selectModel(row, row.dataset.modelId);
    }

    updateDetailMode() {
        if (this.detailEmpty) this.detailEmpty.hidden = this.detailMode !== 'empty';
        if (this.detailInfo) this.detailInfo.hidden = this.detailMode !== 'model';
        if (this.runForm) this.runForm.hidden = this.detailMode !== 'model';
        if (this.addForm) this.addForm.hidden = this.detailMode !== 'add';
        this.addButton?.setAttribute('aria-expanded', this.detailMode === 'add' ? 'true' : 'false');
    }

    async selectModel(_target, modelId) {
        const model = this.findModel(modelId);
        if (!model || this.busy) return;
        this.selectedModelId = model.id;
        this.detailMode = 'model';
        this.updateDetailMode();
        this.renderModels();
        this.renderDetailInfo();
        await this.openRun(model);
    }

    // ------------------------------------------------------------ add model

    openAddModel() {
        if (this.busy || !this.addForm) return;
        this.detailMode = 'add';
        this.updateDetailMode();
        this.addForm.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' });
        this.addForm.elements?.id?.focus();
    }

    cancelAddModel() {
        if (!this.addForm) return;
        this.addForm.reset();
        this.resetLookup({ forget: true });
        this.detailMode = this.selectedModelId ? 'model' : 'empty';
        this.updateDetailMode();
        this.updateSourceFields();
    }

    updateSourceFields() {
        const kind = ['ollama', 'hf', 'exl3'].includes(this.sourceKind?.value) ? this.sourceKind.value : 'huggingface';
        for (const field of this.element.querySelectorAll('[data-source-field]')) {
            // A field names every kind it belongs to, separated by spaces.
            const active = String(field.dataset.sourceField || '').split(/\s+/).includes(kind);
            field.hidden = !active;
            const input = field.querySelector('input');
            if (input) input.required = active && ['repo', 'file', 'tag'].includes(input.name);
        }
    }

    // ------------------------------------------------------------ lookup

    readAddForm() {
        return Object.fromEntries(new FormData(this.addForm));
    }

    setFormValue(name, value) {
        const control = this.addForm?.elements?.[name];
        if (control) control.value = value;
    }

    async setArchitecture(value) {
        const select = this.element.querySelector('#localLlmArchitecture');
        if (!select) return;
        const options = JSON.parse(decodeURIComponent(select.getAttribute('data-options') || '%5B%5D'));
        await this.setSelectOptions(select, options, value === 'moe' ? 'moe' : 'dense');
    }

    renderLookup(result, options = {}) {
        if (this.lookupRegion) this.lookupRegion.innerHTML = result ? lookupResultsHtml(result, options) : '';
    }

    setSizingLabel(text = '') {
        if (!this.sizingLabel) return;
        this.sizingLabel.textContent = text;
        this.sizingLabel.hidden = !text;
    }

    // Clears the lookup's list and label (another repository, revision or source kind). `forget` also forgets what it filled in.
    resetLookup({ forget = false } = {}) {
        this.lookupSerial += 1;
        this.lookupPending = false;
        this.lookup = null;
        if (forget) this.lookupSizing = null;
        this.renderLookup(null);
        this.setSizingLabel('');
        if (this.lookupButton) this.lookupButton.disabled = false;
    }

    // One lookup request; null when a newer request or a reset replaced it, or when it failed (the error is shown).
    async requestLookup(args) {
        const serial = ++this.lookupSerial;
        this.lookupPending = true;
        if (this.lookupButton) this.lookupButton.disabled = true;
        // A file's header is read with the list still showing; only a list request replaces it.
        if (args.file) this.setSizingLabel(`Reading the GGUF header of ${args.file}…`);
        else this.renderLookup({ message: 'Looking up the repository…' });
        try {
            const result = await callLocalLlm('local_llm_model_lookup', args);
            if (serial !== this.lookupSerial || this.closed) return null;
            this.lookup = { args, result };
            return result;
        } catch (error) {
            if (serial === this.lookupSerial && !this.closed) {
                const message = error?.message || 'The lookup failed.';
                if (args.file && this.lookup) {
                    // The pick failed, so the form keeps the file it had: the list shows that one selected, not the one that failed.
                    this.renderLookup(this.lookup.result, { selectedFile: String(this.addForm?.elements?.file?.value || '').trim() });
                    this.setSizingLabel('');
                    this.setStatus(message, 'error');
                } else {
                    this.renderLookup({ error: message });
                }
            }
            return null;
        } finally {
            if (serial === this.lookupSerial) {
                this.lookupPending = false;
                if (this.lookupButton) this.lookupButton.disabled = false;
            }
        }
    }

    async lookupModel() {
        if (this.lookupPending || !this.addForm) return;
        const format = lookupFormatFor(this.sourceKind?.value || 'huggingface');
        const repo = String(this.addForm.elements?.repo?.value || '').trim();
        const revision = String(this.addForm.elements?.revision?.value || '').trim();
        if (!format) return;
        if (!repo) {
            this.renderLookup({ error: 'Enter a repository such as owner/name first.' });
            return;
        }
        const result = await this.requestLookup({ repo, format, ...(revision ? { revision } : {}) });
        if (!result) return;
        const selected = String(this.addForm.elements?.file?.value || '').trim();
        this.renderLookup(result, { selectedFile: selected });
        // A snapshot is sized from its config.json; a GGUF file is sized once it is picked.
        if (format !== 'gguf') await this.fillFromLookup({ result });
    }

    async pickLookupFile(_target, indexText) {
        const row = this.lookup?.result?.files?.[Number(indexText)];
        if (!row || this.lookupPending) return;
        this.renderLookup(this.lookup.result, { selectedFile: row.file });
        const result = await this.requestLookup({ ...this.lookup.args, file: row.file });
        if (!result) return;
        this.renderLookup(result, { selectedFile: row.file });
        await this.fillFromLookup({ result, row });
    }

    // Fills the form from a lookup: the file, its quantization, sizing, and an id, name and licence when those are
    // empty. The looked-up values are remembered, so a value changed by hand afterwards is recorded as manual, and so is
    // the commit, which Add pins. A value the lookup could not read leaves the field alone, except that a value an earlier
    // lookup put there (another file's) is cleared rather than kept for this one.
    async fillFromLookup({ result, row = null }) {
        const sizing = result.sizing;
        const text = (value) => (Number.isFinite(value) ? String(value) : '');
        const current = (name) => String(this.addForm.elements?.[name]?.value || '');
        if (row) {
            this.setFormValue('file', row.file);
            this.setFormValue('quantization', row.quantization || '');
        }
        const name = String(result.repo || '').split('/').pop();
        if (!current('id').trim()) this.setFormValue('id', suggestModelId(result.repo, row?.quantization || ''));
        // The form's own limit on the name is 120 characters.
        if (!current('displayName').trim()) this.setFormValue('displayName', [name, row?.quantization].filter(Boolean).join(' ').slice(0, 120).trim());
        const license = typeof result.license === 'string' ? stripBidi(result.license).trim() : '';
        if (!current('license').trim() && license && license.length <= 80) this.setFormValue('license', license);
        const values = {
            contextLength: text(sizing?.contextLength),
            layers: text(sizing?.memory?.layers),
            kvBytesPerToken: text(sizing?.memory?.kvBytesPerToken),
        };
        const before = this.lookupSizing?.values || {};
        for (const [field, value] of Object.entries(values)) {
            if (value !== '' || current(field) === '' || current(field) === (before[field] ?? null)) this.setFormValue(field, value);
        }
        const read = Object.values(values).some((value) => value !== '');
        // A header the reader could not size says nothing about the architecture: the choice on the form stands.
        if (read) await this.setArchitecture(sizing.architecture);
        const architecture = read ? sizing.architecture : (this.element.querySelector('#localLlmArchitecture')?.value || 'dense');
        this.lookupSizing = {
            source: sizing?.source ?? null,
            repo: result.repo,
            revision: result.revision || 'main',
            ...(row ? { file: row.file } : {}),
            architecture: architecture === 'moe' ? 'moe' : 'dense',
            ...(typeof result.commit === 'string' ? { commit: result.commit } : {}),
            values,
        };
        const notes = Array.isArray(sizing?.notes) && sizing.notes.length ? ` ${sizing.notes.join('; ')}.` : '';
        const from = sizing?.source === 'config.json' ? 'config.json' : 'the GGUF header';
        if (!sizing) this.setSizingLabel(row ? '' : 'No config.json was found, so the memory estimate uses its defaults.');
        else if (!read) this.setSizingLabel(`Nothing could be read from ${from}, so the memory estimate uses its defaults.${notes}`);
        else {
            this.setSizingLabel(sizing.source === 'gguf-header'
                ? `Sizing read from the GGUF header; checked again after download.${notes}`
                : `Sizing read from config.json, checked against its pinned digest.${notes}`);
        }
    }

    async submitAddModel(event) {
        event.preventDefault();
        if (this.busy || !this.addForm?.reportValidity()) return;
        const raw = this.readAddForm();
        raw.sourceKind = this.sourceKind?.value || 'huggingface';
        raw.architecture = this.element.querySelector('#localLlmArchitecture')?.value || 'dense';
        raw.sizingSource = sizingSourceFor(raw, this.lookupSizing);
        // Add pins the commit whose files the lookup read, even if the branch has moved since.
        raw.commit = lookupCommitFor(raw, this.lookupSizing);
        const entry = modelEntryFromForm(raw);
        const added = await this.withBusy(`Adding ${entry.id} and pinning its source…`, async () => {
            await callLocalLlm('local_llm_model_add', { model: entry });
            await this.loadOverview();
            return true;
        });
        if (!added) return;
        this.addForm.reset();
        this.resetLookup({ forget: true });
        this.updateSourceFields();
        this.setStatus(`${entry.id} was added. Nothing was downloaded.`, 'success');
        await this.selectModel(null, entry.id);
    }

    async removeModel(_target, modelId) {
        const model = this.findModel(modelId);
        if (!model || model.seed || this.busy) return;
        if (hasWeightsOnDisk(model)) {
            this.setStatus(`Delete the downloaded weights of ${model.displayName || model.id} first; then Remove works.`, 'error');
            return;
        }
        const confirmed = await assistOS.UI.showModal('confirm-action-modal', {
            message: confirmMessage(`Remove ${model.displayName || model.id} from the list? You can add it again later.`),
        }, true);
        if (!confirmed) return;
        const removed = await this.withBusy(`Removing ${model.id}…`, async () => {
            await callLocalLlm('local_llm_model_remove', { modelId: model.id });
            this.selectedModelId = '';
            this.detailMode = 'empty';
            this.updateDetailMode();
            await this.loadOverview();
            return true;
        });
        if (removed) this.setStatus(`${model.id} was removed.`, 'success');
    }

    async deleteWeights(_target, modelId, format) {
        const model = this.findModel(modelId);
        const entry = model?.weights?.[format];
        if (!entry || this.busy) return;
        const readers = (entry.runners || []).map((runnerId) => runnerLabel(runnerId)).join(', ');
        const confirmed = await assistOS.UI.showModal('confirm-action-modal', {
            message: confirmMessage(`Delete the ${entry.label || format} of ${model.displayName || model.id} (${formatBytes(entry.download?.bytes ?? entry.size)})?`
                + `${readers ? ` It is used by ${readers}.` : ''} The next Run downloads it again.`),
        }, true);
        if (!confirmed) return;
        const deleted = await this.withBusy('Deleting weights…', async () => {
            const result = await callLocalLlm('local_llm_weights_delete', { modelId, format });
            await this.loadOverview();
            return result;
        });
        if (deleted) {
            this.setStatus(`Deleted ${formatBytes(deleted.freedBytes)}.`, 'success');
            await this.renderRunFields();
        }
    }

    // ------------------------------------------------------------ run form

    async openRun(model) {
        if (!model || !this.runForm) return;
        this.runModelId = model.id;
        const options = runnerOptions(this.overview, model);
        const preferred = options.find((option) => model.runners[option.value]?.admission?.status === 'ok')
            || options.find((option) => this.overview.runners.find((runner) => runner.id === option.value)?.supported)
            || options[0];
        await this.setSelectOptions(this.runnerSelect, options, preferred?.value || '');
        if (this.advanced) this.advanced.open = false;
        await this.renderRunFields();
    }

    async renderRunFields() {
        const model = this.findModel(this.runModelId);
        const runnerId = this.runnerSelect?.value || '';
        const runner = (this.overview?.runners || []).find((entry) => entry.id === runnerId);
        const entry = model?.runners?.[runnerId];
        const basic = this.runForm?.querySelector('[data-run-basic]');
        const advanced = this.runForm?.querySelector('[data-run-params-advanced]');
        const summary = this.runForm?.querySelector('[data-run-advanced-summary]');
        const note = this.runForm?.querySelector('[data-run-runner-note]');
        const submit = this.runForm?.querySelector('[data-run-submit]');
        if (!basic || !advanced || !model || !runner) return;
        // A runner this image lacks, or with no parameters for the hardware profile, cannot run here.
        const runnable = runner.supported && runner.installed && Boolean(runner.paramSchema);
        if (note) {
            const reason = !runner.supported ? (runner.unsupportedReason || runner.reason)
                : !runner.installed ? runner.reason
                    : !runner.paramSchema ? (runner.profileUnsupportedReason || 'not available on this hardware profile') : '';
            note.textContent = [
                runner.version ? `${runnerLabel(runner.id)} ${runner.version}` : runnerLabel(runner.id),
                reason,
                entry?.download ? downloadLabel(entry.download, entry.size) : '',
            ].filter(Boolean).join(' · ');
        }
        if (submit) submit.disabled = !runnable || this.busy;
        this.runFields = runnable ? fieldsFromSchema(runner.paramSchema, entry?.params || {}) : [];
        const split = splitRunFields(this.runFields, runner, model);
        basic.innerHTML = split.basic.map((field) => this.renderField(field)).join('');
        advanced.innerHTML = split.advanced.map((field) => this.renderField(field)).join('');
        if (this.advanced) this.advanced.hidden = split.advanced.length === 0;
        if (summary) summary.textContent = `Advanced settings (${split.advanced.length})`;
        this.renderEstimate({ context: entry?.context, admission: entry?.admission });
        // Selects inserted here get their presenter asynchronously; read them only once ready.
        await Promise.all([...this.runForm.querySelectorAll('[data-run-basic] custom-select, [data-run-params-advanced] custom-select')]
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
        if (field.kind === 'enumOrNumber') {
            // Free text with the named values offered, since a number is valid too.
            const listId = `${id}-choices`;
            const hint = `${field.options.map((option) => option.value).join(', ')} or a number from ${field.min} to ${field.max}`;
            return `
                <label class="local-llm-field">
                    <span class="form-label">${escapeHtml(field.title)}</span>
                    <input class="form-input" id="${escapeHtml(id)}" name="${escapeHtml(field.name)}" data-param="${escapeHtml(field.name)}" type="text"
                           list="${escapeHtml(listId)}" value="${escapeHtml(value)}" placeholder="${escapeHtml(hint)}" ${field.nullable ? '' : 'required'}>
                    <datalist id="${escapeHtml(listId)}">${field.options.map((option) => `<option value="${escapeHtml(option.value)}"></option>`).join('')}</datalist>
                    ${help}
                </label>`;
        }
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

    renderEstimate(estimate = {}) {
        const region = this.runForm?.querySelector('[data-run-estimate]');
        if (region) region.innerHTML = estimateHtml(estimate, this.overview?.hardware || {}, { limits: this.overview?.limits });
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
            this.setStatus(runner?.unsupportedReason || runner?.reason || `${runnerLabel(runnerId)} cannot run models in this release.`, 'error');
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
        this.setStatus(started.duplicate ? 'That run was already requested.' : 'Run accepted. Progress shows in the Running model card above.', 'success');
        this.renderDeployment();
        this.renderModels();
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
            const wasActive = this.activeModelId();
            this.status = status;
            this.logs = mergeLogs(this.logs, status.logs || []);
            this.nextSeq = status.nextSeq ?? this.nextSeq;
            phase = status.phase;
            this.pollFailures = 0;
            this.renderDeployment();
            this.renderHardware();
            if (wasActive !== this.activeModelId()) {
                this.renderModels();
                this.renderDetailInfo();
            }
        } catch (error) {
            // A transient failure (a Router restart, a timeout) must not end
            // polling mid-download: retry with a backoff while the dashboard is open.
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
        if (stop) {
            const stoppable = ACTIVE_PHASES.has(phase) || ['error', 'paused'].includes(phase);
            stop.disabled = this.busy || !stoppable;
            stop.hidden = !stoppable;
        }
        if (cancel) {
            const downloading = ['downloading', 'copying', 'verifying', 'pulling'].includes(phase);
            cancel.disabled = this.busy || !downloading;
            cancel.hidden = !downloading;
        }
        if (send) send.disabled = this.busy || phase !== 'ready';
        if (this.promptNote) {
            this.promptNote.textContent = phase === 'ready'
                ? ''
                : 'No model is ready. Select a model on the Models tab and press Run; Send works once it is ready.';
        }
    }

    renderDeployment() {
        this.setHtml('deployment', this.deploymentCard, statusCardHtml(this.status || {}));
        this.updateDeploymentButtons();
        this.renderLogs();
    }

    renderLogs(force = false) {
        if (!this.logView) return;
        const follow = this.follow ? this.follow.checked : true;
        const text = this.logs.map((entry) => `[${entry.stream}] ${entry.line}`).join('\n') || 'No runner output yet.';
        if (this.logView.textContent !== text) this.logView.textContent = text;
        if (follow || force) this.logView.scrollTop = this.logView.scrollHeight;
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
}
