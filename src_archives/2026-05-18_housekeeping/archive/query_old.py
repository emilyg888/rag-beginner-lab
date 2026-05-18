"""
query.py
--------
Runs Retrieval-Augmented Generation (RAG).

STUDENT GOAL:
Understand how retrieval + LLM reasoning work together.
"""

import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from helper_utils import word_wrap
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# --------------------------------------------------
# 1. Setup environment and clients
# --------------------------------------------------
load_dotenv()

openai_client = OpenAI()

embedding_function = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)

# Connect to the existing Chroma collection
from chromadb import PersistentClient
from chromadb.config import Settings

chroma_client = PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)



print("Available collections:")
print(chroma_client.list_collections())


collection = chroma_client.get_collection(
    name="microsoft-annual-report",
    embedding_function=embedding_function
)

# --------------------------------------------------
# 2. Define the user question
# --------------------------------------------------
user_question = "What was the total revenue for the year?"

print("\nUSER QUESTION:")
print(user_question)

# --------------------------------------------------
# 3. (Optional) HyDE-style query augmentation
# --------------------------------------------------
# We generate a *hypothetical answer* to improve retrieval quality.


def generate_hypothetical_answer(question: str) -> str:
    """
    Generates an example answer that *might* appear in the document.
    This helps retrieval find better context.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a financial analyst writing a short example answer."
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


hypothetical_answer = generate_hypothetical_answer(user_question)

augmented_query = f"{user_question} {hypothetical_answer}"

print("\nAUGMENTED QUERY:")
print(word_wrap(augmented_query))

# --------------------------------------------------
# 4. Retrieve relevant chunks
# --------------------------------------------------
results = collection.query(
    query_texts=[augmented_query],
    n_results=8
)

retrieved_chunks = results["documents"][0]

print("\nTOP RETRIEVED CHUNKS:")
for i, chunk in enumerate(retrieved_chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(word_wrap(chunk[:600]))

# --------------------------------------------------
# 5. Answer synthesis (grounded generation)
# --------------------------------------------------
# IMPORTANT RULE:
# The model may ONLY use retrieved context.
context = "\n\n".join(retrieved_chunks)

answer_prompt = f"""
You are a financial analyst.

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
        {"role": "user", "content": answer_prompt},
    ],
)

final_answer = response.choices[0].message.content

print("\nFINAL ANSWER:")
print(word_wrap(final_answer))
