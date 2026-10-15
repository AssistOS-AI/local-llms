import { escapeHtml } from '../../../local-llm-settings/local-llm-settings-model.js';

import { iconHtml } from './local-llm-icons.js';

export function helpHtml(id, label, text) {
    return `<button type="button" class="gray-button local-llm-info" popovertarget="${escapeHtml(id)}" aria-label="${escapeHtml(label)}">${iconHtml('info')}</button><span id="${escapeHtml(id)}" popover class="settings-card settings-card-static local-llm-help">${escapeHtml(text)}</span>`;
}

// Native popovers own light-dismiss and keyboard activation. Hover/focus add
// desktop help without requiring an extra click; touch retains native click.
export function initHelp(element) {
    let hovered = null;
    const show = event => {
        const button = event.target?.closest?.('[popovertarget]');
        if (!button || !element.contains(button)) return;
        const target = element.querySelector(`#${button.getAttribute('popovertarget')}`);
        if (!target || target.matches(':popover-open')) return;
        target.showPopover({ source: button });
        hovered = target;
    };
    const hide = event => {
        const button = event.target?.closest?.('[popovertarget]');
        if (!button || button.contains(event.relatedTarget) || !hovered) return;
        if (hovered.contains(event.relatedTarget)) return;
        hovered.hidePopover();
        hovered = null;
    };
    const escape = event => {
        if (event.key !== 'Escape') return;
        const open = element.querySelector('[popover]:popover-open');
        if (!open) return;
        event.preventDefault();
        event.stopPropagation();
        open.hidePopover();
        hovered = null;
    };
    const click = event => {
        const button = event.target?.closest?.('[popovertarget]');
        if (!button || !element.contains(button)) return;
        const target = element.querySelector(`#${button.getAttribute('popovertarget')}`);
        if (target?.matches(':popover-open')) event.preventDefault();
    };
    element.addEventListener('pointerover', show);
    element.addEventListener('focusin', show);
    element.addEventListener('pointerout', hide);
    element.addEventListener('focusout', hide);
    element.addEventListener('keydown', escape);
    element.addEventListener('click', click);
    return () => {
        hovered?.hidePopover();
        element.removeEventListener('pointerover', show);
        element.removeEventListener('focusin', show);
        element.removeEventListener('pointerout', hide);
        element.removeEventListener('focusout', hide);
        element.removeEventListener('keydown', escape);
        element.removeEventListener('click', click);
    };
}

export function initAddModelDialog({ dialog, trigger, form, isBusy, onClose }) {
    const cancel = event => {
        event.preventDefault();
        event.stopPropagation();
        if (!isBusy()) onClose();
    };
    const keydown = event => {
        if (event.key !== 'Escape') return;
        event.stopPropagation();
        if (isBusy()) event.preventDefault();
    };
    const backdrop = event => {
        if (event.target !== dialog || isBusy()) return;
        const box = dialog.getBoundingClientRect();
        if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) onClose();
    };
    const closed = () => trigger?.setAttribute('aria-expanded', 'false');
    const invalid = event => {
        const details = event.target.closest('details');
        if (details) details.open = true;
    };
    dialog.addEventListener('cancel', cancel);
    dialog.addEventListener('keydown', keydown);
    dialog.addEventListener('click', backdrop);
    dialog.addEventListener('close', closed);
    form.addEventListener('invalid', invalid, true);
    return () => {
        dialog.removeEventListener('cancel', cancel);
        dialog.removeEventListener('keydown', keydown);
        dialog.removeEventListener('click', backdrop);
        dialog.removeEventListener('close', closed);
        form.removeEventListener('invalid', invalid, true);
        if (dialog.open) dialog.close();
    };
}
