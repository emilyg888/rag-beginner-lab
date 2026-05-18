# Archive Manifest - 2026-05-18 Housekeeping

## Archived items

| Original path | Archived path | Reason | Validation performed | Risk |
|---|---|---|---|---|
| `archive/helper_utils.py` | `src_archives/2026-05-18_housekeeping/archive/helper_utils.py` | Legacy tutorial helper superseded by current `src/` scripts and not part of tracked source. | `rg` found no references outside `archive/`; `.gitignore` ignored `archive/`; current tracked runtime is under `src/`. | Low |
| `archive/helper_utils_old.py` | `src_archives/2026-05-18_housekeeping/archive/helper_utils_old.py` | Older duplicate helper with stale APIs and unused imports. | `rg` found no references outside `archive/`; `.gitignore` ignored `archive/`; current tracked runtime is under `src/`. | Low |
| `archive/index_old.py` | `src_archives/2026-05-18_housekeeping/archive/index_old.py` | Old fixed-document indexing script superseded by `src/index.py`, which accepts a PDF path. | `rg` found no references outside `archive/`; `.gitignore` ignored `archive/`; current tracked entry point is `src/index.py`. | Low |
| `archive/query_old.py` | `src_archives/2026-05-18_housekeeping/archive/query_old.py` | Old fixed-collection query script superseded by `src/query.py`, which derives the collection from the PDF path. | `rg` found no references outside `archive/`; `.gitignore` ignored `archive/`; current tracked entry point is `src/query.py`. | Low |
| `archive/python - <<'EOF'.py` | `src_archives/2026-05-18_housekeeping/archive/python - <<'EOF'.py` | Scratch file with shell-fragment filename, not referenced by runtime or docs. | `rg` found no references outside `archive/`; `.gitignore` ignored `archive/`; filename indicates accidental scratch output. | Low |

## Notes

- Items were moved, not deleted.
- The original `archive/` directory was ignored by git before housekeeping, so these files were local-only before being captured in `src_archives/`.
- Review these files before permanent removal if any lab material still depends on the legacy explanations.
