// The toolbar entry, placed like Soul Gateway's: it opens the Local LLMs
// dashboard in Explorer's expanded modal (component mode, fullscreen).
export class LocalLlmToolButton {
    constructor(element, invalidate) {
        this.element = element;
        this.invalidate = invalidate;
        this.hostContext = {};
        this.invalidate();
    }

    beforeRender() {}

    afterRender() {
        this.button = this.element.querySelector('#localLlmToolButton');
        this.iconImageEl = this.element.querySelector('.local-llm-tool-button-icon-image');
        this.labelEl = this.element.querySelector('.local-llm-tool-button-label');
        this.button?.addEventListener('click', this.openDashboard);
        this.syncButtonMetadata();
        if (this.button) this.button.hidden = false;
    }

    afterUnload() {
        this.button?.removeEventListener('click', this.openDashboard);
    }

    updateHostContext(context = {}) {
        this.hostContext = context;
        this.syncButtonMetadata();
    }

    syncButtonMetadata() {
        const text = (value) => (typeof value === 'string' && value.trim() ? value.trim() : '');
        const label = text(this.hostContext?.pluginLabel) || this.element.getAttribute('data-plugin-label') || 'Local LLMs';
        const tooltip = text(this.hostContext?.pluginTooltip) || this.element.getAttribute('data-plugin-tooltip') || label;
        const icon = text(this.hostContext?.pluginIcon) || this.element.getAttribute('data-plugin-icon') || '';
        if (this.labelEl) this.labelEl.textContent = label;
        if (this.iconImageEl && icon) this.iconImageEl.src = icon;
        if (this.button) {
            this.button.title = tooltip;
            this.button.setAttribute('aria-label', tooltip);
        }
    }

    openDashboard = (event) => {
        event?.preventDefault?.();
        event?.stopPropagation?.();
        const descriptor = this.hostContext?.pluginToolbarModal;
        if (!descriptor) return;
        void globalThis.assistOS.UI.openExpandedModal({
            ...descriptor,
            title: this.hostContext?.pluginLabel || 'Local LLMs',
        });
    };
}
