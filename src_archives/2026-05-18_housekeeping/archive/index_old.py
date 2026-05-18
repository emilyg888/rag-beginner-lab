"""
index.py
--------
Builds a vector index from source documents.

STUDENT GOAL:
Understand how raw documents become searchable vectors.
"""

import os
from dotenv import load_dotenv
from pypdf import PdfReader

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

import chromadb


# --------------------------------------------------
# 1. Load environment variables (API keys, etc.)
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# 2. Load the PDF document
# --------------------------------------------------
# We extract raw text from each page of the PDF.
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [page.extract_text().strip() for page in reader.pages]

# Remove empty pages (very common in PDFs)
pdf_texts = [text for text in pdf_texts if text]

print(f"Loaded {len(pdf_texts)} non-empty pages")

# --------------------------------------------------
# 3. Split text into chunks
# --------------------------------------------------
# Why splitting?
# - Embedding models have token limits
# - Smaller chunks = better retrieval precision

# First pass: semantic-ish splitting (paragraphs, sentences)
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0,
)

character_chunks = character_splitter.split_text("\n\n".join(pdf_texts))

print(f"Final number of character chunks: {len(character_chunks)}")

# Second pass: enforce token-sized chunks
token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=0,
)

token_chunks = []
for chunk in character_chunks:
    token_chunks.extend(token_splitter.split_text(chunk))

print(f"Final number of tokenchunks: {len(token_chunks)}")

# --------------------------------------------------
# 4. Create embedding function
# --------------------------------------------------
# This tells Chroma how to convert text → vectors
embedding_function = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)

# --------------------------------------------------
# 5. Create Chroma vector store
# --------------------------------------------------
# Think of this like a "persistent semantic database"
from chromadb import PersistentClient
from chromadb.config import Settings

chroma_client = PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)


collection_name = "microsoft-annual-report"

existing_collections = [c.name for c in chroma_client.list_collections()]

if collection_name in existing_collections:
    print("ℹ️ Collection exists — reusing")
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
else:
    print("🆕 Creating new collection")
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )


# Each document must have a unique ID
import hashlib

def make_chunk_id(text: str, index: int) -> str:
    raw = f"{index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

ids = [
    make_chunk_id(chunk, i)
    for i, chunk in enumerate(token_chunks)
]



# --------------------------------------------------
# 6. Add only new chunks to Chroma for idempotency
# --------------------------------------------------
# --------------------------------------------
# Build metadata aligned with each chunk
# --------------------------------------------

metadatas = []

for i, chunk in enumerate(token_chunks):
    metadatas.append({
        "source": "microsoft-annual-report.pdf",
        "chunk_index": i,
        # Optional but recommended if you have it:
        # "page": page_number_lookup[i],
    })

print(len(ids), len(token_chunks), len(metadatas))

collection_name = "microsoft-annual-report"

if collection_name in [c.name for c in chroma_client.list_collections()]:
    print("♻️ Rebuilding existing collection")
    chroma_client.delete_collection(collection_name)

collection = chroma_client.create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

assert len(ids) == len(set(ids)), "Duplicate IDs detected before add()"

collection.add(
    ids=ids,
    documents=token_chunks,
    metadatas=metadatas,
)

print(f"Indexed {collection.count()} chunks")


# After adding documents, force persistence:

print("✅ Collection persisted to disk")

print(f"Indexed {collection.count()} chunks into Chroma")

print("✅ Indexing complete. You can now run query.py")
