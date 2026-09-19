# 🧠 AI Knowledge Assistant: Advanced RAG Pipeline

> A robust, locally-backed AI chat application built to answer complex questions directly from your PDF documents using a two-stage Retrieval-Augmented Generation (RAG) architecture.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF69B4.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-LLM-8E75B2.svg)

---

## 📖 About

When working with large, domain-specific documents, standard AI chatbots often fail—they either hallucinate answers, lack access to private data, or lose context in massive files. 

This project was built to solve these exact problems. It is an **AI Knowledge Assistant** that doesn't just guess; it mathematically searches your personal PDFs, extracts the exact paragraphs containing the answer, rigorously re-evaluates their relevance, and forces the LLM to base its answer *only* on those facts.

By implementing an **Advanced RAG** approach (incorporating cross-encoder reranking), this application bridges the gap between raw document storage and intelligent, conversational data retrieval.

---

## 🎯 Assignment Requirements Mapping

This project was built to fulfill specific assignment criteria. Here is how every requirement translates into the final implementation:

| Core Requirement | Implementation Details |
| :--- | :--- |
| **PDF Knowledge Source** | `PyMuPDF` (`fitz`) handles fast, high-fidelity text extraction from multiple PDFs. |
| **Chunking Strategy** | Recursive text splitting (500 characters, 50-char overlap) preserves semantic context. |
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) runs locally (zero-cost, private). |
| **Vector Store** | `ChromaDB` persistent local storage handles fast Top-K cosine similarity searches. |
| **Advanced RAG (Reranking)** | Post-retrieval cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores and filters chunks. |
| **LLM Generation** | Google `gemini-3.7-flash` processes the reranked context to formulate answers. |
| **UI Interface** | `Streamlit` provides a clean, conversational chat interface with sidebar controls. |
| **Optional Features Done** | ✅ Source Citations ✅ Conversation History ✅ Multi-document support |

---

## ⚙️ How it works (The Architecture)

Basic RAG systems rely solely on vector similarity, which is fast but prone to false positives (retrieving text that uses similar words but has a different meaning). This project uses a **Two-Stage Retrieval Pipeline** to guarantee precision:

```text
1. USER ASKS QUESTION
          ↓
2. EMBEDDING GENERATION
   Query is converted to a 384-dimensional dense vector via MiniLM-L6-v2.
          ↓
3. STAGE 1: BROAD RETRIEVAL (ChromaDB)
   Fast mathematical search pulls the Top 10 most similar chunks from the vector database.
          ↓
4. STAGE 2: PRECISION RERANKING (Cross-Encoder)
   The Top 10 chunks are passed to ms-marco-MiniLM. Unlike standard embeddings, 
   the cross-encoder reads the question and chunk *simultaneously*, calculating a 
   highly accurate semantic relevance score.
          ↓
5. RELEVANCE GATING
   The system filters out low-scoring chunks. Only the absolute best 3 chunks survive.
   (If no chunks pass the threshold, the system safely responds: "Not Found in Documents").
          ↓
6. LLM GENERATION
   Gemini 2.0 receives the Top 3 chunks + Conversation History and drafts a grounded answer.
          ↓
7. FRONTEND
   Answer is streamed to the user alongside exact, expandable source citations.
```

---

## ✨ Features

* **Natural Language Chat:** Ask questions exactly as if you were talking to an expert who memorized your documents.
* **Post-Retrieval Reranking:** Employs a cross-encoder model to drastically improve relevance over standard vector databases.
* **Strict Hallucination Guardrails:** The system is explicitly prompted—and programmatically gated—to inform you if the answer isn't in the provided documents.
* **Exact Source Citations:** Every answer includes expandable citations showing the exact document filename, page number, and the raw text excerpt used.
* **Conversation Memory:** Retains recent chat history so you can ask contextual follow-up questions naturally (e.g., *"Can you summarize that last point?"*).
* **Local Privacy & Zero Embedding Costs:** Heavy lifting (chunking, embedding, reranking) runs entirely on your local CPU/GPU, not in the cloud.

---
## 📁 Project Structure

```text
rag-project/
├── app.py                 # The main Streamlit chat interface (Frontend)
├── ingest.py              # CLI script to process PDFs into the vector database
├── config.py              # Centralized settings (chunk size, models, thresholds)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
│
├── data/
│   └── documents/         # 📂 Place your raw PDF files in here
│
├── chroma_db/             # Auto-generated local vector database storage
│
└── src/                   # Core pipeline modules
    ├── pdf_loader.py      # PyMuPDF extraction logic & metadata tracking
    ├── chunker.py         # Splits text into overlapping semantic segments
    ├── embeddings.py      # Local MiniLM embedding generation singleton
    ├── vector_store.py    # ChromaDB database wrapper and operations
    ├── reranker.py        # Cross-encoder scoring and sorting logic
    ├── llm.py             # Google Gemini API integration and prompt engineering
    └── rag_pipeline.py    # Orchestrates the Retrieve -> Rerank -> Generate flow
```

---

## 🚀 Getting Started

### 1. Prerequisites
* Python 3.10 or higher
* A Google Gemini API Key

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/purplepot/adv-rag.git
cd adv-rag
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (you can copy `.env.example`) and add your Google API key:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```
*(Alternatively, you can input this directly in the app's sidebar during runtime).*

### 4. Build the Knowledge Base
Add your PDF documents to the `data/documents/` folder, then run the ingestion script to process them, generate embeddings, and build the local vector database:
```bash
python ingest.py
```

### 5. Launch the App
Start the Streamlit chat interface:
```bash
streamlit run app.py
```

