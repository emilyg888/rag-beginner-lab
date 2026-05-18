"""
helper_utils.py
----------------
Utility functions used across the RAG lab.

IMPORTANT FOR STUDENTS:
This file contains *supporting helpers*, not core RAG logic.
The real RAG system lives in:
- index.py  (indexing phase)
- query.py  (query phase)
"""

import numpy as np
from pypdf import PdfReader


# ==========================================================
# Embedding visualisation helper
# ==========================================================
def project_embeddings(embeddings, umap_transform):
    """
    Projects high-dimensional embeddings into a lower-dimensional
    space (typically 2D) for visualisation.

    CONCEPT:
    - Embeddings live in very high dimensions (e.g. 1536)
    - Humans cannot interpret that space directly
    - UMAP is used ONLY for visual understanding, not retrieval

    This function is NOT part of core RAG.
    It is a diagnostic / teaching tool.
    """
    return umap_transform.transform(embeddings)


# ==========================================================
# Text formatting helper (presentation only)
# ==========================================================
def word_wrap(text, width=87):
    """
    Wraps long text into fixed-width lines for console output.

    CONCEPT:
    - This has nothing to do with AI or RAG
    - It exists purely to make outputs readable in a terminal
    """
    return "\n".join(
        text[i: i + width] for i in range(0, len(text), width)
    )


# ==========================================================
# Document ingestion helper (early-stage / tutorial use)
# ==========================================================
def extract_text_from_pdf(file_path):
    """
    Extracts raw text from a PDF file.

    CONCEPT:
    - RAG systems cannot work directly on PDFs
    - All documents must be converted into plain text first
    - In production, this step is usually handled by
      a dedicated ingestion service

    This function demonstrates the *idea* of document ingestion,
    not a production-grade solution.
    """
    pages_text = []

    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    return "\n".join(pages_text)


# ==========================================================
# Legacy / tutorial-only Chroma loader
# ==========================================================
def load_chroma(filename, collection_name, embedding_function):
    """
    Loads a document into Chroma in a single step.

    ⚠️ IMPORTANT FOR STUDENTS:
    This function represents an *early tutorial pattern*.
    It mixes ingestion, embedding, and storage in one place.

    In real systems:
    - Indexing is separated (index.py)
    - Querying is separated (query.py)
    - Persistence, metadata, and idempotency are explicit

    This function is useful for demos,
    but not how production RAG is structured.
    """

    # Step 1: Extract text from document
    text = extract_text_from_pdf(filename)

    # Step 2: Naively split text into chunks
    # (real systems use structured chunking strategies)
    paragraphs = text.split("\n\n")

    # Step 3: Generate embeddings
    # Each paragraph becomes a vector
    embeddings = [embedding_function(p) for p in paragraphs]

    # Step 4: Create a Chroma collection
    # NOTE: This uses an in-memory client and no persistence
    import chromadb
    collection = chromadb.Client().create_collection(collection_name)

    # Step 5: Store text + embeddings
    for i, (paragraph, embedding) in enumerate(zip(paragraphs, embeddings)):
        collection.add(
            ids=str(i),
            documents=paragraph,
            embeddings=embedding,
        )

    return collection
