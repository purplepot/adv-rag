# 🧠 AI Knowledge Assistant

> An advanced Retrieval-Augmented Generation (RAG) chat application that accurately answers questions based on your local PDF documents, complete with precise source citations.

---

## About

Navigating large, complex PDF documents to find specific information can be time-consuming and prone to human error. Furthermore, standard AI chatbots often hallucinate when they lack specific context, making them unreliable for strict document-based querying.

This project solves this by building a highly accurate **Two-Stage RAG Pipeline**. Instead of relying on general AI knowledge, the system mathematically retrieves the most relevant paragraphs from your local PDFs, reranks them for extreme precision using a cross-encoder, and forces the LLM to formulate its answer strictly based on those facts—all presented in a clean, user-friendly Streamlit interface.

---

## Demo

**Chat — AI generates answers strictly from provided context:**

*(Add a screenshot here of the chat interface successfully answering a question)*

**Citations — Exact source tracking for verification:**

*(Add a screenshot here showing the expandable source citations with page numbers)*

---

## How it works

Unlike basic RAG systems that rely solely on fast vector similarity (which can be inaccurate and lead to hallucinations), this application uses an **Advanced Two-Stage Pipeline** to eliminate irrelevant context:

```text
User asks a question
        ↓
Query converted to vector (MiniLM-L6-v2, runs locally)
        ↓
ChromaDB performs Stage 1: Fast Initial Retrieval
(Top 10 mathematically similar chunks retrieved)
        ↓
Cross-Encoder performs Stage 2: Precision Reranking
(Scores the exact relationship between query and chunk)
        ↓
Strict Relevance Gate
(Top 3 absolutely best chunks selected)
        ↓
Gemini 2.0 Flash generates grounded answer + Source Citations
        ↓
Frontend displays chat answer & expandable source snippets
```

This ensures maximum accuracy across every stage of the pipeline:

| Component | Technology | Role |
| --- | --- | --- |
| **PDF Extraction** | PyMuPDF | Flawlessly extracts document text while maintaining page metadata. |
| **Embeddings** | all-MiniLM-L6-v2 | Creates 384-dimensional dense vectors locally, at zero API cost. |
| **Vector Store** | ChromaDB | Persistently stores document chunks for fast similarity search. |
| **Reranker** | ms-marco-MiniLM-L-6-v2 | Cross-encoder that filters out false-positives for pinpoint accuracy. |
| **Generation** | Google Gemini | Formulates final human-readable answers strictly from the context. |

---

## Features

- **Natural Language Chat** — Query complex PDFs exactly as if you were talking to a human expert.
- **Post-Retrieval Reranking** — Employs a cross-encoder model to drastically improve relevance over standard vector databases.
- **Strict Hallucination Guardrails** — The system will explicitly inform you if the answer isn't contained in the provided documents.
- **Exact Source Citations** — Every answer includes expandable citations showing the exact document name, page number, and text excerpt.
- **Conversation Memory** — Retains recent chat history so you can ask contextual follow-up questions naturally.
- **Local Privacy & Zero Embedding Costs** — Heavy lifting (chunking, embedding, reranking) runs entirely on your local machine.

---

## Project Structure

```text
rag-project/
├── app.py                 # The main Streamlit chat interface
├── ingest.py              # CLI script to process PDFs into the vector database
├── config.py              # Centralized settings (chunk size, models, thresholds)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
│
├── data/
│   └── documents/         # 📂 Place your PDF files in here
│
├── chroma_db/             # Auto-generated local vector database storage
│
└── src/                   # Core pipeline modules
    ├── pdf_loader.py      # PyMuPDF extraction logic
    ├── chunker.py         # Splits text into overlapping segments
    ├── embeddings.py      # Local MiniLM embedding generation
    ├── vector_store.py    # ChromaDB database operations
    ├── reranker.py        # Cross-encoder scoring and sorting
    ├── llm.py             # Google Gemini API integration
    └── rag_pipeline.py    # Orchestrates the Retrieve -> Rerank -> Generate flow
```

---

## Getting Started

### 1. Prerequisites
* Python 3.10+
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
*(Alternatively, you can input this directly in the app's sidebar).*

### 4. Build the Knowledge Base
Add your PDF documents to the `data/documents/` folder, then run the ingestion script to build the vector database:
```bash
python ingest.py
```

### 5. Launch the App
Start the Streamlit chat interface:
```bash
streamlit run app.py
```
