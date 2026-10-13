// Errors that cross the controller socket keep a stable code for the tools
// and the Settings UI; messages are for people.

export class LocalLlmError extends Error {
    constructor(code, message, details = undefined) {
        super(message);
        this.name = 'LocalLlmError';
        this.code = code;
        if (details !== undefined) this.details = details;
    }
}

export function serializeError(error) {
    return {
        code: typeof error?.code === 'string' ? error.code : 'internal_error',
        message: String(error?.message || error || 'Unknown error'),
        ...(error?.details !== undefined ? { details: error.details } : {}),
    };
}
