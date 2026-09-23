// One queue for every mutation (run, cancel, stop, delete, model add, update
// and remove). Each command runs alone, in arrival order, so two commands
// can never interleave their checks and writes. Long jobs (downloads, runner
// start) run outside the queue but register what they hold, and commands
// that would touch a held artifact are rejected.

export function createCommandQueue() {
    let tail = Promise.resolve();
    let closed = false;

    function run(command) {
        if (closed) {
            return Promise.reject(Object.assign(new Error('The controller is shutting down.'), { code: 'shutting_down' }));
        }
        const result = tail.then(() => command());
        tail = result.catch(() => {});
        return result;
    }

    function close() {
        closed = true;
        return tail;
    }

    return Object.freeze({ run, close, get closed() { return closed; } });
}
