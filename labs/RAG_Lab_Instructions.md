
# RAG Beginner Lab – Instruction Sheet

## Objective
Understand how Retrieval-Augmented Generation (RAG) systems work at a conceptual level.

## What You Will Learn
- Why RAG is needed
- How data flows from documents to answers
- The role of indexing vs querying
- Industry-standard RAG design principles

## Lab Structure

### Part 1 – Indexing (index.py)
- Documents are ingested and converted to text
- Text is split into chunks
- Chunks are embedded into vectors
- Vectors and metadata are stored in a vector database

### Part 2 – Querying (query.py)
- A user asks a question
- The question is embedded
- Relevant chunks are retrieved
- An LLM generates an answer using retrieved context

## Key Concepts to Remember
- Retrieval always happens before generation
- Chunking improves accuracy
- Metadata enables trust and citations
- Indexing and querying are separate responsibilities

## Industry Perspective
This architecture mirrors real systems used in:
- Enterprise search
- Internal knowledge assistants
- Financial and regulatory AI systems

## Completion Criteria
You can explain:
- What RAG is
- Why vector databases are used
- Why retrieval quality matters
- How RAG reduces hallucinations
