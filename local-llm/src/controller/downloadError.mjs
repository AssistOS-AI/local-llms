// The error every weight and file transfer reports, with a stable `code`
// (DownloadError codes are listed in DS002), and the cancellation helpers the
// transfer, copy and hashing loops share.

export class DownloadError extends Error {
    constructor(code, message = code, { retryable = false, details } = {}) {
        super(message);
        this.name = 'DownloadError';
        this.code = code;
        this.retryable = retryable;
        if (details !== undefined) {
            this.details = details;
        }
    }
}

export function abortedError() {
    return new DownloadError('ABORTED', 'Download stopped; the partial file is kept for resume', { retryable: true });
}

export function throwIfAborted(signal) {
    if (signal?.aborted) {
        throw abortedError();
    }
}
