
"""
visualize_embeddings.py
-----------------------
OPTIONAL LAB SCRIPT (INTUITION ONLY)

Purpose:
Visualise how embeddings behave in vector space.

IMPORTANT:
- This script is NOT part of a production RAG system.
- It exists only to build intuition and debug retrieval.
- Production RAG systems do NOT visualise embeddings.

Key ideas:
- Embeddings are spatial
- Similarity means geometric closeness
- Retrieval is distance-based reasoning over meaning
"""

import numpy as np
import umap
import matplotlib.pyplot as plt

def visualise_embeddings(
    dataset_embeddings,
    retrieved_embeddings,
    query_embedding,
    augmented_query_embedding=None,
    title="Embedding Space"
):
    reducer = umap.UMAP(random_state=42)

    all_embeddings = list(dataset_embeddings)
    all_embeddings.append(query_embedding)

    if augmented_query_embedding is not None:
        all_embeddings.append(augmented_query_embedding)

    all_embeddings = np.array(all_embeddings)
    projected = reducer.fit_transform(all_embeddings)

    dataset_proj = projected[: len(dataset_embeddings)]
    query_proj = projected[len(dataset_embeddings)]

    plt.figure(figsize=(8, 8))

    plt.scatter(
        dataset_proj[:, 0],
        dataset_proj[:, 1],
        s=10,
        color="lightgray",
        label="Document Chunks",
    )

    if retrieved_embeddings is not None:
        retrieved_proj = reducer.transform(retrieved_embeddings)
        plt.scatter(
            retrieved_proj[:, 0],
            retrieved_proj[:, 1],
            s=80,
            facecolors="none",
            edgecolors="green",
            label="Retrieved Chunks",
        )

    plt.scatter(
        query_proj[0],
        query_proj[1],
        s=150,
        marker="X",
        color="red",
        label="Original Query",
    )

    if augmented_query_embedding is not None:
        augmented_proj = projected[-1]
        plt.scatter(
            augmented_proj[0],
            augmented_proj[1],
            s=150,
            marker="X",
            color="orange",
            label="Augmented Query",
        )

    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    """
    Demo runner for embedding visualisation.

    This section exists ONLY for learning purposes.
    It generates fake embeddings so students can see a plot
    without wiring the full RAG pipeline.
    """

    # -----------------------------
    # Create fake embeddings
    # -----------------------------
    np.random.seed(42)

    dataset_embeddings = np.random.normal(size=(200, 1536))
    retrieved_embeddings = dataset_embeddings[:10]
    query_embedding = np.random.normal(size=(1536,))
    augmented_query_embedding = np.random.normal(size=(1536,))

    # -----------------------------
    # Visualise
    # -----------------------------
    visualise_embeddings(
        dataset_embeddings=dataset_embeddings,
        retrieved_embeddings=retrieved_embeddings,
        query_embedding=query_embedding,
        augmented_query_embedding=augmented_query_embedding,
        title="Embedding Space (Demo / Intuition Only)"
    )
