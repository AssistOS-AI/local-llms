import { confirmMessage, escapeHtml, stripBidi } from '../../../local-llm-settings/local-llm-settings-model.js';

// The Registries tab and the Add form's Registry selector. The built-in Hugging Face registry has no actions at all: it
// is read-only and uses the operator's HF_TOKEN. The other built-ins can be enabled, disabled and given a credential;
// the registries an administrator added can also be edited and removed.
const READ_ONLY_ID = 'huggingface';
const KIND_LABELS = Object.freeze({ huggingface: 'Hugging Face-compatible', oci: 'OCI', modelscope: 'ModelScope' });
const CREDENTIAL_LABELS = Object.freeze({ set: 'credential set', none: 'no credential', unreadable: 'credential unreadable' });
const REGISTRY_FIELDS = Object.freeze(['id', 'kind', 'url', 'label', 'authOrigins']);

const text = (value) => stripBidi(String(value ?? ''));

/** The options of the Add form's Registry selector: the enabled registries that offer `format`, in the list's order. */
export function registryOptionsFor(registries = [], format = '') {
    return (Array.isArray(registries) ? registries : [])
        .filter((registry) => registry?.enabled === true && Array.isArray(registry.formats) && registry.formats.includes(format))
        .map((registry) => ({ value: String(registry.id), label: text(registry.label || registry.id) }));
}

/** The labels and placeholders of the Add form's repository fields for a registry kind. */
export function registryFieldLabels(kind = 'huggingface') {
    if (kind === 'oci') {
        return {
            repo: { label: 'Repository', placeholder: 'ai/smollm2' },
            revision: { label: 'Tag', placeholder: 'latest' },
            file: { label: 'GGUF file (optional)', placeholder: 'Derived when the artifact holds one' },
        };
    }
    return {
        repo: { label: 'Repository', placeholder: 'unsloth/Qwen3.6-35B-A3B-GGUF' },
        revision: { label: 'Revision', placeholder: kind === 'modelscope' ? 'master' : 'main' },
        file: { label: 'GGUF file', placeholder: 'Qwen3.6-35B-A3B-UD-Q3_K_M.gguf' },
    };
}

/** Writes the field labels and placeholders of the Add form (`[data-label-for]` spans, the inputs of `form`) for a registry kind. */
export function applyFieldLabels(element, form, kind = 'huggingface') {
    for (const [name, entry] of Object.entries(registryFieldLabels(kind))) {
        const label = element?.querySelector?.(`[data-label-for="${name}"]`);
        if (label) label.textContent = entry.label;
        const input = form?.elements?.[name];
        if (input) input.placeholder = entry.placeholder;
    }
}

function rowActions(registry, storage) {
    if (registry.id === READ_ONLY_ID) return '';
    const id = escapeHtml(registry.id);
    const unavailable = storage?.available === false;
    const button = (action, label, extra = '') => `<button type="button" class="gray-button" data-local-action="${action} ${id}"${extra}>${label}</button>`;
    return [
        registry.editable ? button('editRegistry', 'Edit') : '',
        button('toggleRegistry', registry.enabled ? 'Disable' : 'Enable'),
        button('openRegistryCredential', 'Set credential', ` data-llm-credential-open${unavailable ? ' disabled' : ''}`),
        registry.credential === 'set' || registry.credential === 'unreadable' ? button('clearRegistryCredential', 'Clear credential') : '',
        registry.builtIn ? '' : button('removeRegistry', 'Remove'),
    ].filter(Boolean).join('');
}

/**
 * The Registries list. It shows the state of each credential (set, none, unreadable) and never a value: the list the
 * agent returns carries none, and nothing here is built from what was typed into the credential form.
 */
export function registriesListHtml(list = {}) {
    const registries = Array.isArray(list.registries) ? list.registries : [];
    const storage = list.credentialStorage || null;
    const note = storage?.available === false
        ? `<div class="settings-card-meta" data-llm-credential-reason>${escapeHtml(`Credentials cannot be stored: ${text(storage.reason) || 'no reason was given'}.`)}</div>`
        : '';
    const items = registries.map((registry) => {
        const used = Array.isArray(registry.usedBy) && registry.usedBy.length ? `Used by ${registry.usedBy.map(text).join(', ')}` : 'Not used by any model';
        const origins = registry.kind === 'oci' && Array.isArray(registry.authOrigins) && registry.authOrigins.length
            ? ` · sign-in from ${registry.authOrigins.map(text).join(', ')}` : '';
        return `
            <li class="local-llm-runner-item" data-registry-id="${escapeHtml(registry.id)}">
                <div>
                    <div class="settings-card-title">${escapeHtml(text(registry.label || registry.id))}</div>
                    <div class="settings-card-meta">${escapeHtml(`${KIND_LABELS[registry.kind] || text(registry.kind)} · ${text(registry.url)}${origins}`)}</div>
                    <div class="settings-card-meta">${escapeHtml(`${registry.enabled ? 'Enabled' : 'Disabled'} · ${CREDENTIAL_LABELS[registry.credential] || 'no credential'} · ${used}`)}</div>
                </div>
                <div class="settings-card-actions">${rowActions(registry, storage)}</div>
            </li>`;
    }).join('');
    return `${note}<ul class="local-llm-runner-list" aria-label="Registries">${items || '<li class="settings-card-meta">No registries.</li>'}</ul>`;
}

/** The `registry` object `local_llm_registry_save` takes, from the add or edit form's fields. */
export function registryFromForm(values = {}, { editing = false } = {}) {
    const value = (name) => String(values[name] ?? '').trim();
    const registry = { id: value('id') };
    if (!editing && value('kind')) registry.kind = value('kind');
    if (value('url')) registry.url = value('url');
    if (value('label')) registry.label = value('label');
    const kind = editing ? '' : value('kind');
    const origins = value('authOrigins').split(/[\s,]+/).filter(Boolean);
    if (origins.length && (kind === 'oci' || editing)) registry.authOrigins = origins;
    return { registry };
}

/**
 * The Registries tab. It owns the registry list, the add or edit form and the credential form; `dashboard` is the
 * presenter, which supplies the element, the status line, the busy lock and `call` (a local-llm tool call).
 */
export class RegistriesTab {
    constructor(dashboard, call) {
        this.dashboard = dashboard;
        this.call = call;
        this.list = { registries: [], credentialStorage: { available: true, reason: '' } };
        this.loaded = false;
        this.serial = 0;
        this.editingId = '';
        this.credentialId = '';
    }

    find(selector) {
        return this.dashboard.element.querySelector(selector);
    }

    // The registry records, as the agent listed them (empty until the first list arrives).
    get items() {
        return this.list.registries;
    }

    kindOf(id) {
        return this.items.find((registry) => registry.id === id)?.kind || 'huggingface';
    }

    credentialsAvailable() {
        return this.list.credentialStorage?.available !== false;
    }

    // Reads the list. A failure leaves the previous one and is shown in the tab; the Add form then keeps what it had.
    async load() {
        const serial = ++this.serial;
        const region = this.find('[data-llm-registries]');
        try {
            const list = await this.call('local_llm_registry_list');
            if (serial !== this.serial || this.dashboard.closed) return false;
            this.list = { registries: Array.isArray(list?.registries) ? list.registries : [], credentialStorage: list?.credentialStorage || { available: true, reason: '' } };
            this.loaded = true;
            this.render();
            await this.dashboard.updateRegistrySelect();
            return true;
        } catch (error) {
            if (serial === this.serial && region && !this.dashboard.closed) {
                this.dashboard.rendered.registries = null;
                region.innerHTML = `<div class="settings-status error">${escapeHtml(error?.message || 'The registries could not be listed.')}</div>`;
            }
            return false;
        }
    }

    render() {
        this.dashboard.setHtml('registries', this.find('[data-llm-registries]'), registriesListHtml(this.list));
        this.applyCredentialState();
    }

    // The busy lock enables every button when it ends, so the storage rule is applied again after it.
    applyCredentialState() {
        const unavailable = !this.credentialsAvailable();
        const reason = unavailable ? text(this.list.credentialStorage.reason) : '';
        const targets = this.dashboard.element.querySelectorAll('[data-llm-credential-open], [data-llm-credential-submit]');
        for (const control of targets) {
            if (unavailable) {
                control.disabled = true;
                control.title = reason;
            } else {
                control.removeAttribute?.('title');
            }
        }
        const note = this.find('[data-llm-credential-note]');
        if (note) note.textContent = unavailable ? `Credentials cannot be stored: ${reason || 'no reason was given'}.` : '';
    }

    // ------------------------------------------------------------ add or edit a registry

    form() {
        return this.find('[data-llm-form="registry"]');
    }

    openForm(id = '') {
        const form = this.form();
        if (!form || this.dashboard.busy) return;
        const record = this.items.find((registry) => registry.id === id);
        this.editingId = record?.editable ? record.id : '';
        form.reset();
        for (const name of REGISTRY_FIELDS) {
            const control = form.elements?.[name];
            if (!control) continue;
            const value = this.editingId ? (name === 'authOrigins' ? (record.authOrigins || []).join(' ') : record[name]) : '';
            control.value = value ?? '';
            control.readOnly = Boolean(this.editingId) && (name === 'id' || name === 'kind');
        }
        const heading = this.find('[data-llm-registry-heading]');
        if (heading) heading.textContent = this.editingId ? `Edit ${text(record.label || record.id)}` : 'Add registry';
        form.hidden = false;
        form.elements?.[this.editingId ? 'url' : 'id']?.focus?.();
    }

    closeForm() {
        const form = this.form();
        this.editingId = '';
        if (form) {
            form.reset();
            form.hidden = true;
        }
    }

    async submitForm(event) {
        event.preventDefault();
        const form = this.form();
        if (this.dashboard.busy || !form?.reportValidity?.()) return;
        const editing = Boolean(this.editingId);
        const values = Object.fromEntries(REGISTRY_FIELDS.map((name) => [name, form.elements?.[name]?.value]));
        const { registry } = registryFromForm(values, { editing });
        const saved = await this.dashboard.withBusy(`Saving ${registry.id}…`, async () => this.call('local_llm_registry_save', { registry }));
        if (!saved) return;
        this.closeForm();
        await this.load();
        this.dashboard.setStatus(`${registry.id} was saved.`, 'success');
    }

    async toggle(id) {
        const registry = this.items.find((entry) => entry.id === id);
        if (!registry || registry.id === READ_ONLY_ID || this.dashboard.busy) return;
        const enabled = !registry.enabled;
        const saved = await this.dashboard.withBusy(`${enabled ? 'Enabling' : 'Disabling'} ${id}…`, async () => this.call('local_llm_registry_save', { registry: { id, enabled } }));
        if (!saved) return;
        await this.load();
        this.dashboard.setStatus(`${id} is ${enabled ? 'enabled' : 'disabled'}.`, 'success');
    }

    async remove(id) {
        const registry = this.items.find((entry) => entry.id === id);
        if (!registry || registry.builtIn || this.dashboard.busy) return;
        const used = Array.isArray(registry.usedBy) && registry.usedBy.length ? ` It is used by ${registry.usedBy.join(', ')}; remove those models first.` : '';
        const confirmed = await assistOS.UI.showModal('confirm-action-modal', {
            message: confirmMessage(`Remove ${registry.label || registry.id} and its credential?${used}`),
        }, true);
        if (!confirmed) return;
        const removed = await this.dashboard.withBusy(`Removing ${id}…`, async () => this.call('local_llm_registry_remove', { id }));
        if (!removed) return;
        await this.load();
        this.dashboard.setStatus(`${id} was removed.`, 'success');
    }

    // ------------------------------------------------------------ credentials

    credentialForm() {
        return this.find('[data-llm-form="registry-credential"]');
    }

    // Shows the credential form for one registry. Its password input is empty every time and never filled by the page.
    openCredential(id) {
        const form = this.credentialForm();
        const registry = this.items.find((entry) => entry.id === id);
        if (!form || !registry || registry.id === READ_ONLY_ID || this.dashboard.busy || !this.credentialsAvailable()) return;
        this.credentialId = id;
        form.reset();
        const heading = this.find('[data-llm-credential-heading]');
        if (heading) heading.textContent = `Credential for ${text(registry.label || id)}`;
        const username = this.find('[data-llm-credential-username]');
        if (username) username.hidden = registry.kind !== 'oci';
        form.hidden = false;
        this.applyCredentialState();
        form.elements?.token?.focus?.();
    }

    closeCredential() {
        const form = this.credentialForm();
        this.credentialId = '';
        if (form) {
            form.reset();
            form.hidden = true;
        }
    }

    // The typed token goes only into the `local_llm_registry_credential_set` call. It is read from the input and the
    // input is emptied before the call starts, so neither a failure nor a slow answer leaves it on the page; it is
    // not kept on this object, and no status message repeats it.
    async submitCredential(event) {
        event.preventDefault();
        const form = this.credentialForm();
        const id = this.credentialId;
        if (!form || !id || this.dashboard.busy || !this.credentialsAvailable() || !form.reportValidity?.()) return;
        const tokenInput = form.elements?.token;
        const usernameInput = form.elements?.tokenUsername;
        const tokenUsername = this.kindOf(id) === 'oci' ? String(usernameInput?.value || '').trim() : '';
        const args = { id, token: String(tokenInput?.value || ''), ...(tokenUsername ? { tokenUsername } : {}) };
        if (tokenInput) tokenInput.value = '';
        if (usernameInput) usernameInput.value = '';
        const saved = await this.dashboard.withBusy('Saving the credential…', async () => this.call('local_llm_registry_credential_set', args));
        if (!saved) return;
        this.closeCredential();
        await this.load();
        this.dashboard.setStatus(`The credential of ${id} was saved.`, 'success');
    }

    async clearCredential(id) {
        const registry = this.items.find((entry) => entry.id === id);
        if (!registry || registry.id === READ_ONLY_ID || this.dashboard.busy) return;
        const cleared = await this.dashboard.withBusy(`Clearing the credential of ${id}…`, async () => this.call('local_llm_registry_credential_clear', { id }));
        if (!cleared) return;
        if (this.credentialId === id) this.closeCredential();
        await this.load();
        this.dashboard.setStatus(`The credential of ${id} was cleared.`, 'success');
    }
}
