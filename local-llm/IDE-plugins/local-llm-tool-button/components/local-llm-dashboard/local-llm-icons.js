const shapes = {
    cpu: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/>',
    activity: '<path d="M2 12h4l3-8 6 16 3-8h4"/>',
    memory: '<rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10v4m5-4v4m5-4v4M7 18v3m5-3v3m5-3v3"/>',
    disk: '<path d="m5 4-3 11v4a1 1 0 0 0 1 1h18a1 1 0 0 0 1-1v-4L19 4Z M2 15h20m-5 2.5h.01m-4 0h.01"/>',
    box: '<path d="m12 3 9 5v8l-9 5-9-5V8Zm-9 5 9 5 9-5m-9 5v8M7.5 5.5l9 5"/>',
    download: '<path d="M12 3v10m-4-4 4 4 4-4M5 13H3v7h18v-7h-2m-3 4h.01"/>',
    plus: '<path d="M12 5v14M5 12h14"/>',
    refresh: '<path d="M20 7v5h-5M4 17v-5h5M6.5 6.5A7 7 0 0 1 20 12M4 12a7 7 0 0 0 13.5 5.5"/>',
    play: '<path d="m7 4 14 8-14 8Z"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10h.01"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    blocked: '<circle cx="12" cy="12" r="9"/><path d="m6 6 12 12"/>',
    warning: '<path d="m12 3 10 18H2Z M12 9v4m0 4h.01"/>',
};

export function iconHtml(name) {
    if (!Object.hasOwn(shapes, name)) throw new Error(`Unknown local LLM icon: ${name}`);
    return `<svg class="local-llm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${shapes[name]}</svg>`;
}
