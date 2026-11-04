# MixedImputer 0.2.2 — Distribution Rename

**Release date:** 2026-11-04  
**Distribution:** `mixedimputer` (no hyphen)  
**Import package:** `mixedimputer`  
**Version:** `0.2.2`

## What changed

This release moves the PyPI distribution name from `mixed-imputer` to
`mixedimputer`, matching the import package and the renamed GitHub repository.
There are **no runtime code or API changes** from `0.2.1`.

- `pyproject.toml` project name: `mixedimputer`
- wheel/sdist filenames: `mixedimputer-0.2.2-*`
- README install instructions and PyPI badges updated
- optional-extra error messages updated
- version bumped to `0.2.2`

The old `mixed-imputer` project remains available for existing pins and
reproducibility. A metadata-only deprecation release on the old name is the
planned migration path so `pip install mixed-imputer` can point users at
`mixedimputer` without installing duplicate `mixedimputer/` code.

## PyPI status

`https://pypi.org/project/mixedimputer/` was still 404 when this release was
prepared. PyPI's ultranormalization check may reject the first upload as too
similar to `mixed-imputer` until the name override from PyPI support is applied.
If the upload returns a `400 ... too similar to an existing project` error,
do not retry repeatedly; wait for the support override.

## Artifacts

```text
dist/mixedimputer-0.2.2-py3-none-any.whl
SHA-256 07ca1cb797250867dd076d1309f9fce5904eadf49325667408513bf08d2a82ea

dist/mixedimputer-0.2.2.tar.gz
SHA-256 8d11221290d03d32cd1705c04e5bc31bfe179b7213dae62711ac284e3b58cdfc
```

`twine check` passes for both.

## Upload (manual, after support override)

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<your PyPI API token>"

.\.venv\Scripts\python.exe -m twine upload --non-interactive --repository pypi `
  dist\mixedimputer-0.2.2-py3-none-any.whl `
  dist\mixedimputer-0.2.2.tar.gz
```

Do **not** upload the older `mixed-imputer`/`mixed_imputer` artifacts in
`dist/`.

## Post-upload verification

```powershell
.\.venv\Scripts\python.exe -m pip index versions mixedimputer
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --force-reinstall mixedimputer==0.2.2
.\.venv\Scripts\python.exe -c "import mixedimputer; print(mixedimputer.__version__)"
```

Expected: `0.2.2`.

## Migration plan

1. Publish `mixedimputer==0.2.2` once PyPI allows the name.
2. Keep existing `mixed-imputer` releases available.
3. Publish a final `mixed-imputer` metadata-only deprecation release depending
   on `mixedimputer==0.2.2` and containing no import files.
4. Keep future releases on `mixedimputer`.
