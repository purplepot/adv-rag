"""
Streamlit Chat UI for the RAG Knowledge Assistant.
Chat-only interface with source citations and conversation history.

Usage:
    streamlit run app.py
"""

import streamlit as st
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import config
from src.rag_pipeline import query as rag_query
from src.vector_store import get_collection_count, get_document_sources
from src.pdf_loader import load_all_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_embeddings
from src.vector_store import add_documents, clear_collection


# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🧠",
    layout="centered",
)


# ── Session State Initialization ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = config.GOOGLE_API_KEY


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Knowledge Assistant")
    st.divider()

    # API Key
    st.subheader("🔑 API Configuration")
    api_key_input = st.text_input(
        "Google API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your Google Gemini API key",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        config.GOOGLE_API_KEY = api_key_input

    st.divider()

    # Knowledge Base Stats
    st.subheader("📊 Knowledge Base")
    doc_count = get_collection_count()
    sources = get_document_sources()

    st.metric("Total Chunks", doc_count)
    st.metric("Documents", len(sources))

    if sources:
        with st.expander("📄 Loaded Documents"):
            for src in sources:
                st.write(f"• {src}")

    st.divider()

    # Ingestion
    st.subheader("📥 Document Ingestion")
    st.caption(f"PDF folder: `data/documents/`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Ingest", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                try:
                    pages = load_all_pdfs(config.DOCUMENTS_DIR)
                    if not pages:
                        st.error("No PDF files found!")
                    else:
                        chunks = chunk_documents(
                            pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP
                        )
                        texts = [c["text"] for c in chunks]
                        embeddings = get_embeddings(texts)
                        clear_collection()
                        add_documents(chunks, embeddings)
                        st.success(f"✅ Ingested {len(chunks)} chunks!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_collection()
            st.success("Collection cleared!")
            st.rerun()

    st.divider()

    # Clear Chat
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with Advanced RAG")
    st.caption("Embeddings: MiniLM-L6-v2 (local)")
    st.caption("Reranker: ms-marco-MiniLM (local)")
    st.caption(f"LLM: {config.GEMINI_MODEL}")


# ── Main Chat Area ────────────────────────────────────────────
st.title("🧠 AI Knowledge Assistant")
st.caption("Ask questions about the loaded knowledge base documents")

# Show warning if no documents
if doc_count == 0:
    st.warning(
        "⚠️ No documents ingested yet. Add PDF files to `data/documents/` "
        "and click **Ingest** in the sidebar."
    )

# Show warning if no API key
if not st.session_state.api_key:
    st.warning(
        "⚠️ No Google API key configured. Enter your key in the sidebar "
        "or set `GOOGLE_API_KEY` in your `.env` file."
    )

# Display welcome message if no history
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(
            "Hello! I'm your AI Knowledge Assistant. 👋\n\n"
            "I can answer questions based on the documents loaded in my knowledge base. "
            "Ask me anything about the content, and I'll provide answers with source citations.\n\n"
            "If your question is outside the scope of the loaded documents, "
            "I'll let you know."
        )

# Display conversation history
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for src in sources:
                        score_display = f" — Relevance: {src['score']:.3f}" if src.get('score') else ""
                        st.markdown(
                            f"**📄 {src['source']}** (Page {src['page']}){score_display}"
                        )
                        st.caption(src.get("excerpt", ""))
                        st.divider()


# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            # Build conversation history for context
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # Exclude current question
            ]

            # Execute RAG pipeline
            result = rag_query(
                question=prompt,
                conversation_history=history,
                api_key=st.session_state.api_key,
            )

            answer = result["answer"]
            sources = result["sources"]

        # Display answer
        st.markdown(answer)

        # Display sources
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for src in sources:
                    score_display = f" — Relevance: {src['score']:.3f}" if src.get('score') else ""
                    st.markdown(
                        f"**📄 {src['source']}** (Page {src['page']}){score_display}"
                    )
                    st.caption(src.get("excerpt", ""))
                    st.divider()

        # Show retrieval stats
        if result["num_candidates"] > 0:
            st.caption(
                f"📊 Retrieved {result['num_candidates']} candidates → "
                f"Reranked to {result['num_reranked']} chunks"
            )

    # Save to history (with sources for display)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
