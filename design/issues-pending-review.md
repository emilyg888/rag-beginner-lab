# Issues Pending Review

## Summary

| ID | Severity | Area | Issue | Recommended action | Status |
|---|---|---|---|---|---|
| ISSUE-001 | High | Config/Security | A local `.env` file exists in the repository directory. It is ignored by git, but secrets should be reviewed and kept out of docs, logs, and commits. | Confirm no secret values are tracked or copied into generated artifacts; rotate credentials if exposure is suspected. | Pending review |
| ISSUE-002 | Medium | Tests | No formal automated test suite exists. Housekeeping SIT passed syntax and dependency smoke checks, but full E2E was not run because it requires OpenAI API calls. | Add a small smoke test or documented test script; document when to run API-backed E2E validation. | Pending review |
| ISSUE-003 | Medium | Runtime | `src/query.py` accepts a PDF path but uses a hard-coded user question. | Consider accepting the question as a CLI argument for repeatable lab runs. | Pending review |
| ISSUE-004 | Medium | Data | `.gitignore` whitelists `data/sample.pdf`, but the local lab command and docs use `data/microsoft-annual-report.pdf`, which is ignored and local-only. | Decide whether to rename/commit a safe sample PDF or update docs to use `data/sample.pdf`. | Pending review |
| ISSUE-005 | Low | Retrieval Metadata | Indexed metadata stores source and chunk index but not source page number. | Add page-aware chunk metadata if citations or auditability are required. | Pending review |
| ISSUE-006 | Low | Workspace Hygiene | Untracked slide assets were present before housekeeping: `slides/LLM_RAG_Interaction_Flow_Diagram.md` and `slides/WhyRAG.PNG`. | Decide whether these are intentional new assets and add them in a separate content commit if needed. | Pending review |

## SIT Results

| Command | Result | Notes |
|---|---|---|
| `.venv/bin/python -m py_compile src/index.py src/query.py src/visualize_embeddings.py` | Passed | Syntax compilation completed for all tracked runtime scripts. |
| `.venv/bin/python -c "import chromadb, pypdf, openai, langchain_text_splitters, dotenv, numpy, umap, matplotlib; print('dependency import smoke passed')"` | Passed | Dependency imports passed after installing `requirements.txt` into `.venv`. Matplotlib used a temporary cache because `~/.matplotlib` was not writable. |

## Archived Code Review

| Original path | Archived path | Reason | Review needed? |
|---|---|---|---|
| `archive/helper_utils.py` | `src_archives/2026-05-18_housekeeping/archive/helper_utils.py` | Ignored legacy helper, unreferenced outside `archive/`, superseded by current `src/` scripts. | No |
| `archive/helper_utils_old.py` | `src_archives/2026-05-18_housekeeping/archive/helper_utils_old.py` | Older duplicate helper with stale APIs and unused imports. | No |
| `archive/index_old.py` | `src_archives/2026-05-18_housekeeping/archive/index_old.py` | Old fixed-document indexing script superseded by `src/index.py`. | No |
| `archive/query_old.py` | `src_archives/2026-05-18_housekeeping/archive/query_old.py` | Old fixed-collection query script superseded by `src/query.py`. | No |
| `archive/python - <<'EOF'.py` | `src_archives/2026-05-18_housekeeping/archive/python - <<'EOF'.py` | Scratch file with accidental shell-fragment filename. | No |

## Detailed Issues

### ISSUE-001 - Local `.env` requires secret hygiene review

- Severity: High
- Area: Config/Security
- Evidence: `.env` exists in the repository directory and is ignored by `.gitignore`.
- Impact: Secret values can be accidentally exposed if copied into docs, logs, screenshots, or commits.
- Recommended action: Confirm no secret values are tracked or copied into generated artifacts; rotate credentials if exposure is suspected.
- Status: Pending review

### ISSUE-002 - No formal automated test suite

- Severity: Medium
- Area: Tests
- Evidence: No `tests/` directory or configured test runner was found. Syntax and dependency smoke checks passed in `.venv`, but full indexing/query validation requires OpenAI API calls.
- Impact: Changes can only be validated through smoke checks or API-backed manual runs unless a test harness exists.
- Recommended action: Add a small smoke test or documented test script; document when to run API-backed E2E validation.
- Status: Pending review

### ISSUE-003 - Query question is hard-coded

- Severity: Medium
- Area: Code
- Evidence: `src/query.py` sets `user_question = "What was the total revenue for the year?"`.
- Impact: Lab users cannot query arbitrary questions without editing source code.
- Recommended action: Accept the question as a CLI argument while keeping the current question as an optional default.
- Status: Pending review

### ISSUE-004 - Sample data naming mismatch

- Severity: Medium
- Area: Docs/Data
- Evidence: `.gitignore` allows `data/sample.pdf`, while docs and local state reference `data/microsoft-annual-report.pdf`.
- Impact: A fresh clone may not have the PDF required by the documented commands.
- Recommended action: Decide whether to rename/commit a safe sample PDF or update docs to use `data/sample.pdf`.
- Status: Pending review

### ISSUE-005 - Page numbers are not captured in retrieval metadata

- Severity: Low
- Area: Architecture
- Evidence: `src/index.py` stores `source` and `chunk_index`, but no page number.
- Impact: Answers can show source file provenance, but not precise page citations.
- Recommended action: Track page-to-chunk provenance during ingestion if citation quality matters.
- Status: Pending review

### ISSUE-006 - Pre-existing untracked slide assets

- Severity: Low
- Area: Docs
- Evidence: Initial `git status --short` showed `slides/LLM_RAG_Interaction_Flow_Diagram.md` and `slides/WhyRAG.PNG` as untracked.
- Impact: These assets may be intentional content updates, but they are outside this housekeeping pass until reviewed.
- Recommended action: Decide whether to add them in a separate content commit.
- Status: Pending review
