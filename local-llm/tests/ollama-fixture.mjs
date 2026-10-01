// A fake Ollama registry and manifests for the tests of the pin at Add (DS002): a local HTTP
// server that answers the manifest endpoint, and manifests of known sizes and exact bytes.
import crypto from 'node:crypto';
import http from 'node:http';

export const sha256 = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');

/** A manifest as JSON bytes: a config and layers of the given sizes; `spacing` changes the bytes, and so the digest, not the content. */
export function manifestBytes({ config = 500, layers = [1000, 2000], spacing = 2 } = {}) {
    return Buffer.from(JSON.stringify({
        schemaVersion: 2,
        mediaType: 'application/vnd.docker.distribution.manifest.v2+json',
        config: { mediaType: 'application/vnd.docker.container.image.v1+json', digest: `sha256:${'c'.repeat(64)}`, size: config },
        layers: layers.map((size, index) => ({ mediaType: 'application/vnd.ollama.image.model', digest: `sha256:${String(index + 1).repeat(64)}`, size })),
    }, null, spacing));
}

/**
 * A registry on a free loopback port. `routes` maps a request path to `{ status, headers, body }`; any other
 * path is a 404. `requests` records what each request carried.
 */
export async function registry(t, routes) {
    const requests = [];
    const server = http.createServer((req, res) => {
        requests.push({ url: req.url, method: req.method, accept: req.headers.accept ?? null, authorization: req.headers.authorization ?? null });
        const route = typeof routes === 'function' ? routes(req) : routes[req.url];
        if (!route) {
            res.writeHead(404);
            res.end('{}');
            return;
        }
        if (route.hang) return;
        res.writeHead(route.status ?? 200, route.headers ?? {});
        res.end(route.body ?? '');
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    t.after(() => { server.closeAllConnections?.(); server.close(); });
    return { base: `http://127.0.0.1:${server.address().port}`, requests };
}
