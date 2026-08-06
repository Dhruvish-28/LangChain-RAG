[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://langchain--rag.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-Hybrid_RAG-green)

# Advanced Hybrid RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) application built using **LangChain**, **FAISS**, **BM25**, **Cross-Encoder Reranking**, **Google Gemini 2.5 Flash**, and **Streamlit**.

Unlike a basic RAG implementation, this project combines multiple retrieval techniques, history-aware query reformulation, hybrid search, reranking, and conversation memory to provide highly relevant, context-aware answers from multiple uploaded documents.

---

## 🚀 Live Demo
https://langchain--rag.streamlit.app/

# Demo

### FOR VIDEO DEMO CHECK RELEASES

### Application Interface
![Application interface](results/UI.png)

### File Upload
![File uploads](results/file_upload.png)

### Chat & Response Generation
![Working](results/working.png)

---

# Table of Contents

- Overview
- Features
- System Architecture
- Project Workflow
- Tech Stack
- Retrieval Pipeline
- Project Structure
- Evaluation
- Installation
- Usage
- Metadata Display
- Supported File Types
- Retrieval Techniques
- Prompt Engineering
- Conversation Memory
- Performance Improvements
- Future Enhancements
- License

---

# Overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models by retrieving relevant knowledge from user-provided documents before generating responses.

This project allows users to upload multiple documents and chat with them naturally while maintaining conversational context.

Instead of relying solely on semantic similarity, the system combines:

- Dense Retrieval (FAISS Index)
- Sparse Retrieval (BM25)
- Maximum Marginal Relevance (MMR)
- History-aware Query Reformulation
- Cross-Encoder Reranking

to maximize retrieval quality before passing context to Gemini 2.5 Flash.

---

# Features

## Multi-format Document Support

Supports:

- PDF
- DOCX
- TXT
- Markdown (.md)

Multiple documents can be uploaded simultaneously.

---

## Hybrid Retrieval

Instead of relying on a single retriever, the project combines:

- FAISS Vector Search
- BM25 Keyword Search
- Maximum Marginal Relevance (MMR)

Hybrid retrieval improves recall by combining semantic similarity with keyword matching.

---

## History-Aware Retrieval

The retriever understands follow-up questions.

Example:

User:

> What is Machine Learning?

Follow-up:

> What are its advantages?

Instead of retrieving documents using only:

```
What are its advantages?
```

the retriever first reformulates the question using previous conversation history.

---

## Cross-Encoder Reranking

Retrieved chunks are reranked using a Cross-Encoder model.

Benefits:

- Removes irrelevant chunks
- Improves final context quality
- Better answer accuracy

---

## Multi-Document Chat

Users can upload multiple documents.

The chatbot automatically retrieves information from all uploaded files without requiring document selection.

---

## Conversation Memory

Maintains previous conversations.

Only the latest conversation history is sent to the LLM to:

- reduce token usage
- preserve context
- improve follow-up responses

---

## Metadata Display

Every generated response displays:

- Source document(s)
- Response generation time
- Input tokens
- Output tokens
- Total tokens

---

## Streaming Responses

Responses are streamed progressively for a natural conversational experience.

---

## Error Handling

Gracefully handles:

- Unsupported file formats
- Empty document uploads
- Gemini Resource exhaustion
- Processing failures

---

## Reset Knowledge Base

Allows users to clear:

- uploaded documents
- vector database
- conversation history

with a single click.

---

# System Architecture

```
                User Uploads Documents
                         │
                         ▼
                 Document Loaders
                         │
                         ▼
             Recursive Character Splitter
                         │
                         ▼
             HuggingFace Embeddings
                         │
                         ▼
              	FAISS Vector Index
                         │
                         ▼
                 Hybrid Retrieval
         ┌──────────────┴──────────────┐
         │                             │
     FAISS (MMR)                   BM25
         │                             │
         └──────────────┬──────────────┘
                        ▼
              History-Aware Retriever
                        ▼
              Cross-Encoder Reranker
                        ▼
                 Top Relevant Chunks
                        ▼
                Prompt Construction
                        ▼
               Gemini 2.5 Flash LLM
                        ▼
               Streamlit Chat Interface
```

---

# Project Workflow

1. Upload documents.
2. Process documents.
3. Split documents into chunks.
4. Generate embeddings.
5. Build FAISS index.
6. Create BM25 retriever.
7. Combine retrievers.
8. Reformulate follow-up queries.
9. Retrieve relevant chunks.
10. Rerank retrieved chunks.
11. Construct prompt.
12. Generate answer using Gemini.
13. Display answer with metadata.

---

# Tech Stack

## Language

- Python

## Framework

- LangChain

## LLM

- Google Gemini 2.5 Flash

## Embedding Model

- sentence-transformers
- all-MiniLM-L6-v2

Embedding Dimension:

384

---

## Vector Search

FAISS

Index Type:

Flat Index (LangChain FAISS)

Similarity Metric:

Cosine Similarity
(L2 normalization performed internally)

---

## Sparse Retrieval

BM25

---

## Reranker

CrossEncoder

---

## UI

Streamlit

---

## Project Structure

```text
Advanced-Hybrid-RAG/
│
├── ingestion/                     # Document ingestion pipeline
│   ├── loaders.py                 # Multi-format document loaders
│   ├── splitter.py                # Text chunking
│   ├── embeddings.py              # Embedding generation
│   └── embedding_model.py         # HuggingFace embedding model
│
├── retrieval/                     # Retrieval pipeline
│   ├── retriever.py               # History-aware retrieval
│   ├── prompt_response.py         # Prompt construction & response generation
│   └── ensemble_retriever.py      # Hybrid (FAISS + BM25 + Reranker)
│
├── models/
│   ├── llm_model.py               # Gemini LLM configuration
│   └── transformer.py             # CrossEncoder reranker
│
├── evaluation/                    # Deterministic RAG evaluation harness
│   ├── run_evaluation.py          # Entry point (benchmark + report)
│   ├── config.py                  # Config dataclass & quick/full grids
│   ├── report.py                  # Generates SUMMARY_REPORT.md
│   ├── data_builder/              # Synthetic corpus + 33 ground-truth Q&As
│   ├── core/                      # Corpus, retrievers, benchmark loop
│   ├── metrics/                   # Recall@K, Precision@K, MRR, evidence
│   └── results/                   # master_summary.csv + SUMMARY_REPORT.md
│
├── UI.py                          # Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

# Evaluation

A deterministic, LLM-free evaluation harness benchmarks the retrieval pipeline on a fixed 33-question dataset built from a 6-document corpus. All metrics are scored lexically against verbatim ground-truth excerpts — no LLM calls, no API keys, fully reproducible offline.

```bash
python -m evaluation.run_evaluation --quick   # ~1 min, offline
```

| Config | Recall@3 | Recall@K | MRR | Evidence Coverage | Full Evidence@K |
| --- | --- | --- | --- | --- | --- |
| Hybrid + rerank (500×50) | **1.000** | **1.000** | **0.949** | **1.000** | **1.000** |
| Hybrid ensemble (500×50) | 1.000 | 1.000 | 0.904 | 1.000 | 1.000 |
| BM25 (500×50) | 0.970 | 0.970 | 0.859 | 0.970 | 0.970 |
| Vector (FAISS) (500×50) | 0.939 | 0.939 | 0.879 | 0.939 | 0.939 |
| Vector + rerank (500×50) | 0.939 | 0.939 | 0.909 | 0.939 | 0.939 |

**Key takeaways:**

- **Reranking lifts rank quality:** MRR improves from **0.904 → 0.949** (ensemble + rerank) vs. ensemble alone.
- **Hybrid beats single retrievers:** ensemble (`Recall@K` = 1.000) outperforms FAISS-only (0.939) and BM25-only (0.970).
- **Perfect evidence coverage:** the full pipeline surfaces all supporting gold excerpts for every question (Full Evidence@K = 1.000).

Full methodology and per-config results: [`evaluation/README.md`](evaluation/README.md) and `evaluation/results/SUMMARY_REPORT.md`.

---

# Installation

Clone repository

```bash
git clone <repository-url>
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GEMINI_API_KEY=YOUR_API_KEY
```

Run

```bash
streamlit run UI.py
```

---

# Performance Improvements

Compared to a basic RAG system, this project adds:

- Hybrid Retrieval
- History-Aware Retrieval
- MMR
- BM25
- Cross-Encoder Reranking
- Streaming Responses
- Conversation Memory
- Metadata Tracking
- Error Handling

These enhancements significantly improve retrieval quality, contextual understanding, and overall user experience.

---

# Future Enhancements

- OCR support for scanned PDFs
- Table extraction
- Image understanding with multimodal models
- Citation highlighting inside answers
- Persistent vector database
- User authentication
- Cloud deployment
- Docker support

---

# License

This project is intended for educational and portfolio purposes.