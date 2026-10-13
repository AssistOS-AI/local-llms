import { LocalLlmError } from './errors.mjs';

function readWithSignal(reader, signal, failed) {
    const stopped = () => signal?.reason instanceof LocalLlmError ? signal.reason : new LocalLlmError('prompt_timeout', 'The prompt deadline expired.');
    if (signal?.aborted) return Promise.reject(stopped());
    return new Promise((resolve, reject) => {
        const finish = (callback, value) => { signal?.removeEventListener('abort', aborted); callback(value); };
        const aborted = () => finish(reject, stopped());
        signal?.addEventListener('abort', aborted, { once: true });
        Promise.resolve().then(() => reader.read()).then((value) => finish(resolve, value), () => {
            const error = new LocalLlmError('prompt_transport_failed', 'The prompt body transport failed.');
            failed(error); finish(reject, error);
        });
    });
}

export async function readPromptStream(response, { signal, onFirstToken, abort = () => {}, onSettlement = () => {} } = {}) {
    const failed = (error) => onSettlement({ error });
    const fail = (code) => { const error = new LocalLlmError(code, 'The observed prompt stream did not complete safely.'); failed(error); throw error; };
    if (!response.body?.getReader) fail('prompt_stream_invalid');
    const reader = response.body.getReader(); const decoder = new TextDecoder('utf-8', { fatal: true });
    let buffer = ''; let bytes = 0; let done = false; let first = false;
    let text = ''; let textChars = 0; let reasoningChars = 0; let finishReason = null; let statistics = {};
    async function frame(value) {
        if (Buffer.byteLength(value) > 65536) fail('prompt_stream_overflow');
        const lines = value.split('\n'); const data = [];
        for (const line of lines) {
            if (line.startsWith(':') || !line) continue;
            if (!line.startsWith('data:')) fail('prompt_stream_invalid');
            data.push(line.slice(5).replace(/^ /, ''));
        }
        if (!data.length) return;
        if (done) fail('prompt_stream_invalid');
        const payload = data.join('\n');
        if (payload === '[DONE]') { done = true; return; }
        let event; try { event = JSON.parse(payload); } catch { fail('prompt_stream_invalid'); }
        if (!event || typeof event !== 'object' || Array.isArray(event) || event.error) fail('prompt_stream_invalid');
        if (event.choices !== undefined && !Array.isArray(event.choices)) fail('prompt_stream_invalid');
        const choice = event.choices?.[0]; const delta = choice?.delta;
        if (delta && (typeof delta !== 'object' || Array.isArray(delta))) fail('prompt_stream_invalid');
        for (const key of ['content', 'reasoning_content', 'reasoning']) if (delta?.[key] != null && typeof delta[key] !== 'string') fail('prompt_stream_invalid');
        const content = delta?.content || ''; const reasoning = delta?.reasoning_content || delta?.reasoning || '';
        if (!first && (content.length || reasoning.length)) {
            try { await onFirstToken(); }
            catch (error) {
                const classified = ['observation_control_failed', 'observation_incomplete', 'prompt_timeout'].includes(error?.code)
                    ? error : new LocalLlmError('observation_control_failed', 'The first-token observation failed.');
                failed(classified); throw classified;
            }
            first = true;
        }
        textChars += content.length; text += content.slice(0, Math.max(0, 16000 - text.length)); reasoningChars += reasoning.length;
        if (choice?.finish_reason != null) finishReason = choice.finish_reason;
        if (event.usage) statistics.usage = event.usage;
        if (event.timings) statistics.timings = event.timings;
    }
    try {
        while (true) {
            const chunk = await readWithSignal(reader, signal, failed);
            if (chunk.done) break;
            bytes += chunk.value.byteLength;
            if (bytes > 1048576) fail('prompt_stream_overflow');
            try { buffer += decoder.decode(chunk.value, { stream: true }); } catch { fail('prompt_stream_invalid'); }
            buffer = buffer.replace(/\r\n/g, '\n');
            let end;
            while ((end = buffer.indexOf('\n\n')) !== -1) { await frame(buffer.slice(0, end)); buffer = buffer.slice(end + 2); }
            if (Buffer.byteLength(buffer) > 65536) fail('prompt_stream_overflow');
        }
        try { buffer += decoder.decode(); } catch { fail('prompt_stream_invalid'); }
        if (!done || buffer.trim()) fail('prompt_stream_truncated');
        const body = { text, truncated: textChars > 16000, reasoningChars, finishReason, statistics };
        onSettlement({ body });
        return body;
    } catch (error) {
        abort();
        throw error;
    } finally {
        Promise.resolve(reader.cancel()).catch(() => {});
        try { reader.releaseLock(); } catch {}
    }
}
