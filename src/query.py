"""
query.py
--------
Query phase of the RAG system.

Purpose:
Retrieve relevant context and generate grounded answers.

Usage: 
> python src/query.py <path_to_pdf>
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from chromadb import PersistentClient
from chromadb.config import Settings

# --------------------------------------------------
# Paths and environment
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
if len(sys.argv) < 2:
    raise RuntimeError("Usage: python src/query.py <path_to_pdf>")

DOC_PATH = Path(sys.argv[1])

COLLECTION_NAME = DOC_PATH.stem.lower().replace(" ", "-")

load_dotenv()

openai_client = OpenAI()

embedding_function = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)

# --------------------------------------------------
# Load vector store
# --------------------------------------------------
chroma_client = PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)

# --------------------------------------------------
# User query
# --------------------------------------------------
user_question = "What was the total revenue for the year?"
print(f"\nQUESTION:\n{user_question}")

# --------------------------------------------------
# HyDE-style query augmentation
# --------------------------------------------------


def generate_hypothetical_answer(question: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


hypothetical_answer = generate_hypothetical_answer(user_question)
augmented_query = f"{user_question} {hypothetical_answer}"

# --------------------------------------------------
# Retrieval
# --------------------------------------------------
results = collection.query(
    query_texts=[augmented_query],
    n_results=8,
    include=["documents", "metadatas"],
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]

print("\nRETRIEVED CONTEXT:")
for doc, meta in zip(documents, metadatas):
    print(f"- (page unknown) {doc[:200]}...")

# --------------------------------------------------
# Answer synthesis
# --------------------------------------------------
context = "\n\n".join(documents)

prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "Not stated in the document."

QUESTION:
{user_question}

CONTEXT:
{context}
"""

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Use provided context only."},
        {"role": "user", "content": prompt},
    ],
)

print("\nFINAL ANSWER:")
print(response.choices[0].message.content)
