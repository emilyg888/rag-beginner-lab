Folder and File System Map
rag-beginner-lab/
├── README.md       
├── src/
│   ├── index.py
│   ├── query.py
│   └── visualize_embeddings.py
├── data/
│   ├── chroma/
│   └── sample.pdf
├── labs/
│   ├── RAG_Lab_Instructions.md
│   └── RAG_Optional_Lab_Embedding_Visualisation.md
├── slides/
│   ├── Intro_to_RAG_Lab_Tutorial.pptx
│   └── RAG_Embedding_Intuition_Slide.pptx
├── requirements.txt
└── .gitignore


RAG Workflow Overview:
┌─────────────────────────────────────────────┐
│                User Input                   │
│        (PDF path via terminal)              │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│                 index.py                    │
│        (Indexing / Offline Phase)            │
│                                             │
│  • Read PDF document                         │
│  • Extract raw text                          │
│  • Split text into chunks                    │
│  • Generate embeddings                       │
│  • Attach metadata                           │
│                                             │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              data/chroma/                   │
│        (Persistent Vector Store)             │
│                                             │
│  • Stored embeddings (vectors)               │
│  • Stored text chunks                        │
│  • Stored metadata (source, chunk id, etc.)  │
│                                             │
│  (This persists across runs)                 │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│                 query.py                    │
│        (Query / Online Phase)                │
│                                             │
│  • Accept user question                      │
│  • Embed the question                        │
│  • Retrieve relevant chunks from Chroma      │
│  • (Optional) Augment the query (HyDE)       │
│  • Generate grounded answer with LLM         │
│                                             │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              Final Answer                   │
│   (Grounded in retrieved document context)  │
└─────────────────────────────────────────────┘

Python Script Function Descriptions:

index.py
  ├─ PDF ingestion
  ├─ chunking
  ├─ embedding
  ├─ vector DB persistence
  └─ metadata creation

query.py
  ├─ load persisted vector DB
  ├─ retrieval
  ├─ query augmentation (HyDE)
  └─ grounded answer generation

visualize_embeddings.py
  ├─ OPTIONAL utilities
  ├─ visualization of embeddings
  ├─ dimensionality reduction
