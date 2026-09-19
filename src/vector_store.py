"""
ChromaDB vector store operations.
Handles persistent storage, document upsert, and similarity search.
"""

import chromadb
import config


def _get_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def get_collection():
    """Get or create the knowledge base collection."""
    client = _get_client()
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_documents(
    chunks: list[dict],
    embeddings: list[list[float]]
) -> int:
    """
    Add document chunks with embeddings to the vector store.
    
    Args:
        chunks: List of chunk dicts with 'id', 'text', 'metadata'.
        embeddings: Corresponding embedding vectors.
        
    Returns:
        Number of documents added.
    """
    collection = get_collection()

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        # ChromaDB metadata values must be str, int, float, or bool
        meta = {}
        for key, value in chunk["metadata"].items():
            meta[key] = str(value) if not isinstance(value, (int, float, bool)) else value
        metadatas.append(meta)

    # Upsert in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    return len(ids)


def query(
    query_embedding: list[float],
    top_k: int = None,
) -> dict:
    """
    Query the vector store for similar documents.
    
    Args:
        query_embedding: Embedding vector for the query.
        top_k: Number of results to return.
        
    Returns:
        ChromaDB query results dict with 'ids', 'documents', 'metadatas', 'distances'.
    """
    if top_k is None:
        top_k = config.TOP_K_RETRIEVAL

    collection = get_collection()
    count = collection.count()

    if count == 0:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Ensure we don't request more results than available
    top_k = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def get_collection_count() -> int:
    """Get the number of documents in the collection."""
    collection = get_collection()
    return collection.count()


def clear_collection():
    """Delete and recreate the collection."""
    client = _get_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # Collection might not exist
    return get_collection()


def get_document_sources() -> list[str]:
    """Get a list of unique source documents in the collection."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    # Get all metadatas
    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if "source" in meta:
            sources.add(meta["source"])
    return sorted(sources)
