# Catalog v3 compatibility fixture

These files capture the catalog validator and raw state store at repository revision `c70ecd013793b919fd9a51ce8ae5001a2fd1dae9`. `provenance.json` records each original path, original SHA-256 and captured SHA-256. The catalog, state store and error module are whole, unmodified files. The profiles module contains only its two unchanged exported constants consumed by the catalog, copied from the recorded source lines. No current product module is imported by this fixture.

The fixture is used only by offline compatibility tests. It preserves the old reader's v4-envelope rejection and its handling of unsupported registry entries through an unrelated state save. Catalog files are always passed explicitly to the old loader; no historical seed or runner process is required. Refreshing this fixture would change the compatibility boundary, so ordinary source changes must leave it unchanged.
