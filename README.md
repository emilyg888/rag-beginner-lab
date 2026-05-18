# Retrieval-Augmented Generation (RAG) Beginner Lab

## Overview

This repository contains a concept-first beginner lab for Retrieval-Augmented Generation (RAG). It demonstrates how a PDF document becomes searchable semantic memory, how retrieval and generation are separated, and how answers can be grounded in retrieved context.

The implementation is intentionally small: local Python scripts, a persistent Chroma vector store, OpenAI embeddings, and an OpenAI chat model for grounded answer synthesis.

## Architecture Summary

The lab has two primary runtime phases:

1. `src/index.py` reads a PDF, extracts text, chunks it, creates embeddings, and persists the resulting collection in Chroma.
2. `src/query.py` loads the matching collection, retrieves relevant chunks, and asks the model to answer using only the retrieved context.

`src/visualize_embeddings.py` is optional and exists only to help learners understand embeddings in vector space.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/index.py` | Index a PDF into a persistent Chroma collection. |
| `src/query.py` | Retrieve context and generate a grounded answer. |
| `src/visualize_embeddings.py` | Optional embedding visualization demo. |
| `labs/` | Lab instructions and tutorial materials. |
| `slides/` | Presentation assets for the lab. |
| `data/` | Local PDFs and generated Chroma state. |
| `design/architecture.md` | Architecture and operational notes. |
| `design/issues-pending-review.md` | SIT results and issues for review. |
| `src_archives/` | Housekeeping archive for legacy code moved out of active paths. |

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` or export environment variables with the required OpenAI API credentials. Do not commit `.env` or secret values.

## Run

Index a PDF:

```bash
python src/index.py data/microsoft-annual-report.pdf
```

Query the indexed collection:

```bash
python src/query.py data/microsoft-annual-report.pdf
```

The query script currently uses the built-in lab question: `What was the total revenue for the year?`

## Test / SIT

Housekeeping SIT results:

| Command | Result | Notes |
|---|---|---|
| `.venv/bin/python -m py_compile src/index.py src/query.py src/visualize_embeddings.py` | Passed | Validates syntax for the runtime scripts. |
| Python dependency import smoke check | Passed | Dependency imports passed after installing `requirements.txt` in `.venv`. |

Full end-to-end indexing and query checks require OpenAI API credentials and may call external model APIs.

## Configuration

The scripts load environment variables with `python-dotenv`. The main required configuration is OpenAI API access. Generated vector-store state is written under `data/chroma/`.

## Documentation

- Architecture: `design/architecture.md`
- Pending review issues: `design/issues-pending-review.md`
- Lab structure map: `labs/Lab_Structure_Map.md`

## Current Status

Housekeeping on 2026-05-18 created architecture documentation, recorded pending review issues, and moved ignored legacy archive files into `src_archives/2026-05-18_housekeeping/`.
