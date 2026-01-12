"""
index.py
--------
Indexing phase of the RAG system.

Purpose:
Convert documents into a persistent semantic index.

Usage: 
> python src/index.py <path_to_pdf>
"""
import sys
from pathlib import Path
import hashlib
import os
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb import PersistentClient
from chromadb.config import Settings

# --------------------------------------------------
# Paths and environment
# --------------------------------------------------
# --------------------------------------------------
# Read document path from command-line
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
if len(sys.argv) < 2:
    raise RuntimeError(
        "Usage: python src/index.py <path_to_pdf>"
    )

DOC_PATH = Path(sys.argv[1])

if not DOC_PATH.exists():
    raise RuntimeError(f"Document not found: {DOC_PATH}")

# Load environment variables
load_dotenv()

# --------------------------------------------------
# 1. Load document
# --------------------------------------------------
reader = PdfReader(str(DOC_PATH))
pages = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
print(f"Loaded {len(pages)} non-empty pages")

# --------------------------------------------------
# 2. Chunking
# --------------------------------------------------
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0,
)

character_chunks = character_splitter.split_text("\n\n".join(pages))
print(f"Character chunks: {len(character_chunks)}")

token_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=0,
)

token_chunks = []
for chunk in character_chunks:
    token_chunks.extend(token_splitter.split_text(chunk))

print(f"Token chunks: {len(token_chunks)}")

# --------------------------------------------------
# 3. Embedding function
# --------------------------------------------------
embedding_function = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)

# --------------------------------------------------
# 4. Persistent vector store
# --------------------------------------------------
chroma_client = PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)

COLLECTION_NAME = DOC_PATH.stem.lower().replace(" ", "-")

if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
    print("♻️ Rebuilding existing collection")
    chroma_client.delete_collection(COLLECTION_NAME)

collection = chroma_client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

# --------------------------------------------------
# 5. Stable IDs + metadata
# --------------------------------------------------


def make_chunk_id(text: str, index: int) -> str:
    return hashlib.sha256(f"{index}:{text}".encode("utf-8")).hexdigest()


ids = [make_chunk_id(chunk, i) for i, chunk in enumerate(token_chunks)]
assert len(ids) == len(set(ids)), "Duplicate IDs detected"

metadatas = [
    {
        "source": DOC_PATH.name,
        "chunk_index": i,
    }
    for i in range(len(token_chunks))
]

# --------------------------------------------------
# 6. Add to vector store
# --------------------------------------------------
collection.add(
    ids=ids,
    documents=token_chunks,
    metadatas=metadatas,
)

print(f"Indexed {collection.count()} chunks")
print("✅ Indexing complete")
