# Architecture

## 1. Purpose

This repository is a beginner-oriented Retrieval-Augmented Generation (RAG) lab. It teaches how a PDF document is converted into a persistent semantic index and then queried with grounded generation.

The audience is learners, data engineers, solution architects, and trainers who need a small enterprise-style RAG walkthrough rather than a production service.

## 2. Current System Shape

The project is a Python script-based lab, not a packaged application. Runtime code lives in `src/`, learning materials live in `labs/` and `slides/`, and generated local state is expected under `data/chroma/`.

There are two primary phases:

1. `src/index.py` ingests a PDF, chunks text, creates OpenAI embeddings, and persists chunks in Chroma.
2. `src/query.py` loads the matching Chroma collection, augments a hard-coded question with a hypothetical answer, retrieves context, and asks OpenAI for a context-grounded response.

`src/visualize_embeddings.py` is an optional teaching aid for embedding-space intuition.

## 3. Component Map

| Component | Path | Responsibility | Key dependencies |
|---|---|---|---|
| Indexing script | `src/index.py` | Read a PDF path, extract text, split into chunks, generate embeddings, and rebuild a Chroma collection. | `pypdf`, `langchain-text-splitters`, `chromadb`, `python-dotenv`, OpenAI embeddings |
| Query script | `src/query.py` | Load a Chroma collection, create a HyDE-style augmented query, retrieve chunks, and synthesize a grounded answer. | `chromadb`, `openai`, `python-dotenv`, OpenAI chat and embeddings |
| Embedding visualization | `src/visualize_embeddings.py` | Demonstrate high-dimensional embedding projection for lab intuition. | `numpy`, `umap-learn`, `matplotlib` |
| Lab instructions | `labs/` | Human-readable lab flow and optional exercises. | Markdown/PDF readers |
| Slide assets | `slides/` | Presentation material for the lab. | PowerPoint or compatible viewer |
| Local data | `data/` | Holds sample PDFs and generated Chroma state. | Local filesystem |
| Archived legacy code | `src_archives/2026-05-18_housekeeping/` | Preserves unused local archive files found during housekeeping. | None at runtime |

## 4. Runtime Flow

```text
PDF path -> src/index.py -> text extraction -> chunking -> OpenAI embeddings -> Chroma collection
User/lab question -> src/query.py -> HyDE augmentation -> Chroma retrieval -> OpenAI response -> grounded answer
```

## 5. Data Flow

`src/index.py` reads a PDF from the path passed on the command line. It extracts non-empty page text, performs two-pass recursive chunking, hashes each chunk into a stable ID, attaches metadata containing the source filename and chunk index, and writes the resulting collection to `data/chroma/`.

`src/query.py` derives the collection name from the PDF filename, loads the persisted Chroma collection, creates a hypothetical answer for the hard-coded question, combines the question and hypothetical answer for retrieval, then passes retrieved chunks to the chat model with a context-only instruction.

The Chroma store and sample PDFs are local state. The checked-in `.gitignore` excludes `data/chroma/`, `.env`, cache files, and most `data/*.pdf` files.

## 6. Configuration

Configuration is loaded from environment variables via `python-dotenv`. The scripts require OpenAI API credentials to be available in the environment or a local `.env` file.

Do not commit `.env` or secret values. The current `.gitignore` excludes `.env`, `.env.*`, and `.env/`.

Important local paths:

| Path | Meaning |
|---|---|
| `data/chroma/` | Generated persistent Chroma database |
| `data/*.pdf` | Local PDF inputs; ignored by default except `data/sample.pdf` |
| `requirements.txt` | Pinned Python dependency set |

## 7. Testing and SIT

There is no formal test suite in the repository. Housekeeping SIT used syntax compilation plus dependency import smoke checks in a local Python 3.11 virtual environment:

| Command | Result | Notes |
|---|---|---|
| `.venv/bin/python -m py_compile src/index.py src/query.py src/visualize_embeddings.py` | Passed | Validates Python syntax for the tracked runtime scripts. |
| `.venv/bin/python -c "import chromadb, pypdf, openai, langchain_text_splitters, dotenv, numpy, umap, matplotlib; print('dependency import smoke passed')"` | Passed | Dependency imports succeed after installing `requirements.txt` in `.venv`. Matplotlib warned that `~/.matplotlib` was not writable and used a temporary cache. |

Full end-to-end indexing and query SIT requires OpenAI API credentials and may call external model APIs.

## 8. Deployment / Execution

This lab is run locally from the repository root.

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Index a PDF:

```bash
python src/index.py data/microsoft-annual-report.pdf
```

Query the indexed collection:

```bash
python src/query.py data/microsoft-annual-report.pdf
```

The query script currently uses a hard-coded question: `What was the total revenue for the year?`

## 9. Governance / Operational Notes

- Answers are intended to be grounded in retrieved context only.
- Chunks include source filename and chunk index metadata, but page numbers are not currently captured.
- The vector store is local generated state and should not be committed.
- API keys must remain in local environment configuration only.
- The lab is a teaching implementation and does not include production concerns such as authentication, request tracing, automated evaluation, access control, or model cost controls.

## 10. Known Gaps

See `design/issues-pending-review.md`.
