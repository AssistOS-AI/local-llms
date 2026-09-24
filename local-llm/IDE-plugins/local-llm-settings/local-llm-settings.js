// Settings → Agents → Local LLMs. The page itself is the local-llm-dashboard
// component, which the toolbar button opens in Explorer's full-screen panel;
// this entry opens the same panel and closes itself.
const DASHBOARD = 'local-llm-dashboard';
const DASHBOARD_PRESENTER = 'LocalLlmDashboard';
const DASHBOARD_TITLE = 'Local LLMs';

async function fetchText(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${url} answered ${response.status}.`);
    return response.text();
}

// Explorer registers the dashboard when the toolbar plugin loads. Settings can
// open first, so register it the way Explorer's settings loader registers a
// component: template, stylesheet and presenter from the plugin files.
async function ensureDashboardRegistered({ baseUrl, webSkel, loadRegistration }) {
    if (globalThis.customElements?.get?.(DASHBOARD)) return;
    const [template, css, presenterModule, registration] = await Promise.all([
        fetchText(`${baseUrl}.html`),
        fetchText(`${baseUrl}.css`),
        import(`${baseUrl}.js`),
        loadRegistration(),
    ]);
    await registration.registerRuntimeComponent(webSkel, {
        name: DASHBOARD,
        componentType: 'components',
        type: 'components',
        loadedTemplate: template,
        loadedCSSs: [css],
        presenterClassName: DASHBOARD_PRESENTER,
        presenterModule,
    });
}

// Explorer's settings loader registers the first exported function of this
// module as the presenter, so this class is the module's only export.
export class LocalLlmSettings {
    constructor(element, invalidate, {
        ui = () => globalThis.assistOS?.UI,
        webSkel = () => globalThis.assistOS?.webSkel,
        baseUrl = new URL(`../local-llm-tool-button/components/${DASHBOARD}/${DASHBOARD}`, import.meta.url).href,
        loadRegistration = () => import('/explorer/shared/ui/runtime-component-registration.js'),
    } = {}) {
        this.element = element;
        this.invalidate = invalidate;
        this.ui = ui;
        this.webSkel = webSkel;
        this.baseUrl = baseUrl;
        this.loadRegistration = loadRegistration;
        this.invalidate();
    }

    beforeRender() {}

    async afterRender() {
        this.statusLine = this.element.querySelector('[data-llm-launcher-status]');
        await this.openDashboard();
    }

    setStatus(message, type) {
        if (!this.statusLine) return;
        this.statusLine.textContent = message;
        for (const state of ['loading', 'error']) this.statusLine.classList.toggle(state, type === state);
    }

    async openDashboard() {
        const ui = this.ui();
        if (typeof ui?.openExpandedModal !== 'function') {
            this.setStatus('This Explorer cannot open the Local LLMs panel; use the Local LLMs button in the toolbar.', 'error');
            return false;
        }
        this.setStatus('Opening Local LLMs…', 'loading');
        try {
            await ensureDashboardRegistered({ baseUrl: this.baseUrl, webSkel: this.webSkel(), loadRegistration: this.loadRegistration });
        } catch (error) {
            this.setStatus(`Local LLMs could not be loaded: ${error?.message || error}`, 'error');
            return false;
        }
        void ui.openExpandedModal({ mode: 'component', component: DASHBOARD, title: DASHBOARD_TITLE });
        this.closeModal();
        return true;
    }

    closeModal() {
        this.ui()?.closeModal?.(this.element);
    }
}
