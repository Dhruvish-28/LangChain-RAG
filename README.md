# Advanced Hybrid RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) application built using **LangChain**, **FAISS**, **BM25**, **Cross-Encoder Reranking**, **Google Gemini 2.5 Flash**, and **Streamlit**.

Unlike a basic RAG implementation, this project combines multiple retrieval techniques, history-aware query reformulation, hybrid search, reranking, and conversation memory to provide highly relevant, context-aware answers from multiple uploaded documents.

---

# Demo

*(Add screenshots or GIFs here)*

---

# Table of Contents

- Overview
- Features
- System Architecture
- Project Workflow
- Tech Stack
- Retrieval Pipeline
- Project Structure
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

- Dense Retrieval (FAISS HNSW)
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

- FAISS HNSW Vector Search
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
- Gemini API quota exhaustion
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
              FAISS HNSW Vector Index
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
5. Build FAISS HNSW index.
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

HNSW (Hierarchical Navigable Small World)

Parameters:

```
M = 32
efConstruction = 200
efSearch = 100
```

Similarity Metric:

Cosine Similarity

(L2 normalization + Inner Product)

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

# Project Structure

```
Advanced-Hybrid-RAG/

│
├── ingestion/
│   ├── loaders.py
│   ├── splitter.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── retrieval/
│   ├── retriever.py
│   ├── reranker.py
│   ├── prompt_response.py
│   └── llm_model.py
│
├── UI.py
├── requirements.txt
├── README.md
└── .gitignore
```

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

# Supported File Types

- PDF
- DOCX
- TXT
- Markdown

---

# Retrieval Techniques

## Dense Retrieval

Semantic search using HuggingFace embeddings and FAISS HNSW.

---

## MMR

Improves diversity by avoiding redundant chunks.

---

## BM25

Improves lexical retrieval for exact keywords.

---

## History-Aware Retrieval

Uses previous conversation to rewrite ambiguous follow-up questions.

---

## Cross-Encoder Reranker

Scores retrieved chunks using a Cross-Encoder before sending them to the LLM.

---

# Prompt Engineering

The prompt instructs Gemini to:

- answer only from retrieved context
- avoid hallucinations
- respond concisely
- state when information is unavailable

---

# Conversation Memory

Stores previous messages in session state.

Only the latest conversation history is passed to the retriever and LLM to preserve context while minimizing token consumption.

---

# Metadata Display

Each answer includes:

- Source document
- Response time
- Input tokens
- Output tokens
- Total tokens

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