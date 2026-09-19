"""
Configuration for the RAG Knowledge Assistant.
All tunable parameters are centralized here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Gemini ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_TEMPERATURE = 0.2

# ── Embedding Model (runs locally) ────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── Reranker Model (runs locally) ─────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_SCORE_THRESHOLD = -3  # Cross-encoder scores range ~(-10 to +10); be permissive here, LLM handles final filtering

# ── Chunking ──────────────────────────────────────────────────
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────
TOP_K_RETRIEVAL = 10      # Candidates from vector search
TOP_N_RERANKED = 3        # Final chunks after reranking

# ── ChromaDB ──────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "knowledge_base"

# ── Documents ─────────────────────────────────────────────────
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")

# ── Conversation ──────────────────────────────────────────────
MAX_CONVERSATION_HISTORY = 5  # Number of past exchanges to include

# ── System Prompt for LLM ─────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI Knowledge Assistant. Your role is to answer 
questions based STRICTLY on the provided context from the knowledge base documents.

RULES:
1. Answer ONLY using information from the provided context chunks.
2. If the context does not contain enough information to answer the question, 
   respond with: "I couldn't find relevant information in the provided documents. 
   Please provide more context or ask about topics covered in the knowledge base."
3. Always cite your sources by mentioning the document name and page number.
4. Format citations as: [Source: filename, Page X]
5. Be concise but thorough in your answers.
6. If the user's question is a follow-up, use the conversation history for context 
   but still ground your answer in the provided documents.
7. Do NOT make up or infer information beyond what is explicitly stated in the context.
"""
