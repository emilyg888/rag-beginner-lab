
> 📦 **Optional Lab: Visualising Embeddings (Intuition Only)**
>
📘 Lab Note: Why This Visualisation Uses Fake Embeddings

Why are we using fake (synthetic) embeddings here?

This visualisation uses randomly generated embeddings on purpose. The goal of this lab is to help learners build intuition about how embeddings and retrieval work geometrically, not to demonstrate a production pipeline.

In real RAG systems, embeddings are high-dimensional vectors (often 1,000+ dimensions) that represent meaning. Visualising them directly is impossible, so we project them into 2D using tools like UMAP. By using synthetic embeddings, learners can safely see how documents form clusters, how queries land near relevant content, and why similarity means closeness — without introducing extra complexity, API calls, costs, or dependencies on the rest of the RAG system.

Important: Production RAG systems do not visualise embeddings. This step exists only for learning, debugging, and conceptual understanding.

> **Purpose**
> This optional lab helps learners *see* how embeddings and retrieval work geometrically.
>
> **What this shows**
> - Embeddings are points in a vector space
> - Documents cluster by meaning
> - Similarity means closeness
> - Retrieval selects nearest neighbours
> - Augmented queries often move closer to relevant clusters
>
> **Important**
> - This visualisation is NOT part of a production RAG system
> - Used only for intuition and debugging
> - Real systems never visualise embeddings
>
> **Takeaway**
> RAG works through distance-based reasoning over meaning,
> not keyword matching.