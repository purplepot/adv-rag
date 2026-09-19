"""
PDF Ingestion Script.
Loads PDF documents, chunks them, generates embeddings, and stores in ChromaDB.

Usage:
    python ingest.py
"""

import sys
import time
import config
from src.pdf_loader import load_all_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_embeddings
from src.vector_store import add_documents, clear_collection, get_collection_count


def ingest(clear_existing: bool = True):
    """
    Ingest all PDF documents from the configured directory into ChromaDB.
    
    Args:
        clear_existing: If True, clear the existing collection before ingesting.
    """
    print("=" * 60)
    print("📚 RAG Knowledge Assistant — Document Ingestion")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Load PDFs
    print(f"\n📂 Loading PDFs from: {config.DOCUMENTS_DIR}")
    pages = load_all_pdfs(config.DOCUMENTS_DIR)
    if not pages:
        print("❌ No documents found. Please add PDF files to data/documents/")
        sys.exit(1)
    print(f"   Total pages extracted: {len(pages)}")

    # Step 2: Chunk documents
    print(f"\n✂️  Chunking (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
    chunks = chunk_documents(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"   Total chunks created: {len(chunks)}")

    # Step 3: Generate embeddings
    print(f"\n🧮 Generating embeddings ({config.EMBEDDING_MODEL})...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)
    print(f"   Embeddings generated: {len(embeddings)}")

    # Step 4: Store in ChromaDB
    print(f"\n💾 Storing in ChromaDB ({config.CHROMA_COLLECTION_NAME})...")
    if clear_existing:
        print("   Clearing existing collection...")
        clear_collection()

    num_added = add_documents(chunks, embeddings)
    print(f"   Documents stored: {num_added}")

    # Summary
    elapsed = time.time() - start_time
    total_count = get_collection_count()
    print("\n" + "=" * 60)
    print(f"✅ Ingestion complete!")
    print(f"   Total documents in collection: {total_count}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
