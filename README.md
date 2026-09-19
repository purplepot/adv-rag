# AI Knowledge Assistant

A highly accurate Retrieval-Augmented Generation (RAG) chat application that answers questions based on your local PDF documents. The system is designed to provide grounded answers with exact source citations, minimizing hallucinations through a robust two-stage retrieval pipeline.

## 🧠 How It Works (The Architecture)

The system operates in two main phases: **Document Ingestion** and **Query Processing**.

### 1. Document Ingestion (Backend Pipeline)
Before you can chat, the system processes your documents to build a knowledge base:
* **Extraction:** Uses `PyMuPDF` to accurately extract text from PDFs page-by-page.
* **Chunking:** Splits the text into small, overlapping chunks (500 characters) to preserve context without overwhelming the LLM.
* **Embedding:** Converts text into numerical vectors using a local embedding model (`all-MiniLM-L6-v2`).
* **Storage:** Saves the vectors and metadata (filename, page numbers) in a local `ChromaDB` vector database for fast searching.

### 2. Query Processing (Two-Stage Retrieval)
When a user asks a question, the system doesn't just do a simple search. It uses a **Two-Stage Pipeline** to ensure extreme accuracy:

1. **Stage 1: Initial Retrieval (Fast & Broad)**
   The user's question is converted into a vector, and ChromaDB performs a fast mathematical search to retrieve the top 10 most similar chunks. *Why not stop here?* Because vector similarity is fast but can sometimes return superficially similar text that doesn't actually answer the question.

2. **Stage 2: Cross-Encoder Reranking (Slow & Precise)**
   The top 10 chunks are passed to a local Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`). Unlike normal embeddings, a cross-encoder reads the user's question and the document chunk *at the exact same time*, calculating a highly accurate relevance score. 
   
   *How this helps:* It re-orders the chunks, dropping irrelevant ones and elevating the ones that truly answer the question. It acts as a strict filter, ensuring only the absolute best 3 chunks make it to the final stage.

3. **Stage 3: LLM Generation**
   The top 3 reranked chunks are sent to Google Gemini (`gemini-2.0-flash`) along with the conversation history. The LLM is strictly instructed to answer *only* using the provided chunks and to cite its sources (e.g., `[Source: document.pdf, Page 4]`).

## 📁 Project Structure

```text
rag-project/
├── app.py                 # The main Streamlit chat interface
├── ingest.py              # Run this to process PDFs into the vector database
├── config.py              # Centralized settings (chunk size, models, etc.)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (Google API Key)
│
├── data/
│   └── documents/         # 📂 Place your PDF files in here
│
├── chroma_db/             # Auto-generated local vector database storage
│
└── src/                   # Core pipeline modules
    ├── chunker.py         # Splits text into overlapping segments
    ├── embeddings.py      # Local MiniLM embedding generation
    ├── llm.py             # Google Gemini API integration
    ├── pdf_loader.py      # PyMuPDF extraction logic
    ├── rag_pipeline.py    # Orchestrates the Retrieve -> Rerank -> Generate flow
    ├── reranker.py        # Cross-encoder scoring and sorting
    └── vector_store.py    # ChromaDB database operations
```

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* A Google Gemini API Key

### Installation
1. Clone this repository and navigate into it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API Key:
   Create a `.env` file in the root directory and add:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```
   *(Alternatively, you can input this directly in the app's sidebar).*

### Usage

1. **Add Documents:** Place your PDF files inside the `data/documents/` folder.
2. **Ingest Data:** Run the ingestion script to build your local vector database.
   ```bash
   python ingest.py
   ```
3. **Start Chatting:** Launch the web interface.
   ```bash
   streamlit run app.py
   ```
