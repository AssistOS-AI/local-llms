// Registry credentials at rest (DS002): each admin-entered token is sealed with
// AES-256-GCM under a key derived from LOCAL_LLM_CREDENTIALS_KEY, the generated
// secret Ploinky gives the agent, and bound to its registry's definition through
// the additional authenticated data. The file holds ciphertext only and is
// written atomically at 0600. The key reaches this module only as an explicit
// option (agentProcess.mjs takes it out of the environment); nothing here reads
// the environment, logs, or returns a token.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

import { LocalLlmError } from '../errors.mjs';
import { writeAtomically } from './stateStore.mjs';

export const CREDENTIALS_KEY_RE = /^[0-9a-f]{64}$/;
export const CREDENTIALS_FILE_VERSION = 1;
const ENC_INFO = 'local-llm/registry-credentials/aes-256-gcm/v1';
const KEY_ID_INFO = 'local-llm/registry-credentials/key-id/v1';
const AAD_PREFIX = 'local-llm/registry-credential/v1';
const IV_BYTES = 12;
const TAG_BYTES = 16;

export const CREDENTIALS_UNAVAILABLE = 'The agent did not receive LOCAL_LLM_CREDENTIALS_KEY; restart local-llm with a Ploinky that supplies generated secrets.';
const CREDENTIALS_MALFORMED = 'The agent received a LOCAL_LLM_CREDENTIALS_KEY that is not 64 lowercase hex characters, so registry credentials cannot be stored; restart local-llm with a Ploinky that supplies generated secrets.';

/**
 * The encryption key and key id derived from the credentials key (the 32 bytes its 64 hex characters encode), or
 * null when the key is absent or malformed. HKDF-SHA256 with an empty salt and one `info` per use.
 */
export function deriveCredentialKeys(credentialsKey) {
    if (typeof credentialsKey !== 'string' || !CREDENTIALS_KEY_RE.test(credentialsKey)) return null;
    const ikm = Buffer.from(credentialsKey, 'hex');
    try {
        return Object.freeze({
            encKey: Buffer.from(crypto.hkdfSync('sha256', ikm, Buffer.alloc(0), ENC_INFO, 32)),
            keyId: Buffer.from(crypto.hkdfSync('sha256', ikm, Buffer.alloc(0), KEY_ID_INFO, 8)).toString('hex'),
        });
    } finally {
        ikm.fill(0);
    }
}

/** The additional authenticated data that binds a credential to its registry's id, kind, URL and auth origins. */
export function credentialAad({ id, kind, url, authOrigins = [] }) {
    return Buffer.from([AAD_PREFIX, id, kind, url, [...authOrigins].sort().join(' ')].join('\n'), 'utf8');
}

function plainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function emptyFile() {
    return { keyId: null, entries: Object.create(null) };
}

export function createCredentialStore({ dataDir, credentialsKey = null, fsApi = fs, now = () => new Date() } = {}) {
    const file = path.join(dataDir, 'state', 'registry-credentials.json');
    const keys = deriveCredentialKeys(credentialsKey);
    const reason = keys ? null : (credentialsKey === null || credentialsKey === undefined || credentialsKey === ''
        ? CREDENTIALS_UNAVAILABLE : CREDENTIALS_MALFORMED);
    // What loading the file found that an administrator should know (logged once by the registry service).
    const notes = [];
    let data = load();

    function load() {
        let value;
        try {
            value = JSON.parse(fsApi.readFileSync(file, 'utf8'));
        } catch (error) {
            if (error?.code === 'ENOENT') return emptyFile();
            // An unreadable file is kept aside for inspection, never overwritten by the next write.
            try { fsApi.renameSync(file, `${file}.corrupt-${Date.now()}`); } catch {}
            notes.push('the registry credentials file could not be parsed and was moved aside; re-enter the registry credentials');
            return emptyFile();
        }
        if (!plainObject(value) || value.version !== CREDENTIALS_FILE_VERSION || !plainObject(value.entries)) {
            try { fsApi.renameSync(file, `${file}.unsupported-${Date.now()}`); } catch {}
            notes.push('the registry credentials file has an unsupported format and was moved aside; re-enter the registry credentials');
            return emptyFile();
        }
        const entries = Object.create(null);
        for (const [id, entry] of Object.entries(value.entries)) entries[id] = entry;
        return { keyId: typeof value.keyId === 'string' ? value.keyId : null, entries };
    }

    function write(next) {
        const entries = {};
        for (const id of Object.keys(next.entries).sort()) entries[id] = next.entries[id];
        writeAtomically(fsApi, file, `${JSON.stringify({ version: CREDENTIALS_FILE_VERSION, keyId: next.keyId, entries }, null, 2)}\n`);
        data = next;
    }

    const entryOf = (id) => (Object.hasOwn(data.entries, id) ? data.entries[id] : null);

    /** The stored credential of `record`, decrypted, or null when there is none or it cannot be read with this key. */
    function open(record) {
        const entry = entryOf(record.id);
        if (!entry || !keys || !plainObject(entry)) return null;
        // An entry sealed under another key (a rotated master key) is not even tried.
        if ((typeof entry.keyId === 'string' ? entry.keyId : data.keyId) !== keys.keyId) return null;
        if (![entry.iv, entry.ciphertext, entry.tag].every((part) => typeof part === 'string')) return null;
        const iv = Buffer.from(entry.iv, 'base64');
        const tag = Buffer.from(entry.tag, 'base64');
        if (iv.length !== IV_BYTES || tag.length !== TAG_BYTES) return null;
        let plaintext = null;
        try {
            const decipher = crypto.createDecipheriv('aes-256-gcm', keys.encKey, iv, { authTagLength: TAG_BYTES });
            decipher.setAAD(credentialAad(record));
            decipher.setAuthTag(tag);
            plaintext = Buffer.concat([decipher.update(Buffer.from(entry.ciphertext, 'base64')), decipher.final()]);
            const value = JSON.parse(plaintext.toString('utf8'));
            if (!plainObject(value) || typeof value.token !== 'string' || !value.token) return null;
            return Object.freeze({ token: value.token, ...(typeof value.username === 'string' ? { username: value.username } : {}) });
        } catch {
            // A failed tag check (another registry's ciphertext, a changed URL, a corrupt tag) reads as unreadable.
            return null;
        } finally {
            plaintext?.fill(0);
        }
    }

    /** 'none', 'set' or 'unreadable': 'set' only when the entry decrypts with this key for this exact registry. */
    function state(record) {
        if (!entryOf(record.id)) return 'none';
        return open(record) ? 'set' : 'unreadable';
    }

    /** Seal and store a credential for `record`, replacing any earlier one. */
    function seal(record, { token, username }) {
        if (!keys) throw new LocalLlmError('credentials_unavailable', reason);
        const iv = crypto.randomBytes(IV_BYTES);
        const cipher = crypto.createCipheriv('aes-256-gcm', keys.encKey, iv, { authTagLength: TAG_BYTES });
        cipher.setAAD(credentialAad(record));
        const plaintext = Buffer.from(JSON.stringify({ token, ...(username !== undefined ? { username } : {}) }), 'utf8');
        let ciphertext;
        try {
            ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final()]);
        } finally {
            plaintext.fill(0);
        }
        const entries = Object.assign(Object.create(null), data.entries);
        entries[record.id] = {
            keyId: keys.keyId,
            iv: iv.toString('base64'),
            ciphertext: ciphertext.toString('base64'),
            tag: cipher.getAuthTag().toString('base64'),
            updatedAt: now().toISOString(),
        };
        // Every entry this key can read carries its key id; the file's id names the key of the last write.
        write({ keyId: keys.keyId, entries });
    }

    /**
     * Delete the credential of `id`; true when there was one. The file is rewritten, never deleted, even when its
     * last entry goes: it then holds `entries: {}`. Works without the key.
     */
    function remove(id) {
        if (!entryOf(id)) return false;
        const entries = Object.assign(Object.create(null), data.entries);
        delete entries[id];
        write({ keyId: data.keyId, entries });
        return true;
    }

    return Object.freeze({
        file,
        available: keys !== null,
        reason,
        notes: Object.freeze(notes),
        state,
        open,
        seal,
        remove,
        storedIds: () => Object.keys(data.entries),
    });
}
