"""
Advanced RAG Pipeline Orchestrator.
Coordinates the full query pipeline: embed → retrieve → rerank → generate.
"""

from src.embeddings import get_single_embedding
from src.vector_store import query as vector_query, get_collection_count
from src.reranker import rerank
from src.llm import generate_answer, generate_no_context_response
import config


def query(
    question: str,
    conversation_history: list[dict] = None,
    api_key: str = None,
) -> dict:
    """
    Execute the full Advanced RAG pipeline.
    
    Pipeline:
        1. Embed the user query
        2. Retrieve top-K candidates from ChromaDB
        3. Rerank using cross-encoder
        4. Check relevance gate
        5. Generate answer with Gemini (or return "not found")
    
    Args:
        question: The user's question.
        conversation_history: Previous conversation messages.
        api_key: Optional Google API key override.
        
    Returns:
        Dict with:
            - 'answer': The generated answer string
            - 'sources': List of source citation dicts
            - 'num_candidates': Number of initial retrieval candidates
            - 'num_reranked': Number of chunks after reranking
    """
    # Check if knowledge base has any documents
    doc_count = get_collection_count()
    if doc_count == 0:
        return {
            "answer": (
                "⚠️ No documents have been ingested yet. "
                "Please add PDF files to the `data/documents/` folder "
                "and run the ingestion process."
            ),
            "sources": [],
            "num_candidates": 0,
            "num_reranked": 0,
        }

    # Step 1: Embed the query
    query_embedding = get_single_embedding(question)

    # Step 2: Retrieve top-K candidates from vector store
    retrieval_results = vector_query(query_embedding, top_k=config.TOP_K_RETRIEVAL)

    # Parse retrieval results into a list of document dicts
    candidates = []
    if retrieval_results["documents"] and retrieval_results["documents"][0]:
        for doc_text, metadata, distance in zip(
            retrieval_results["documents"][0],
            retrieval_results["metadatas"][0],
            retrieval_results["distances"][0],
        ):
            candidates.append({
                "text": doc_text,
                "metadata": metadata,
                "distance": distance,
            })

    num_candidates = len(candidates)

    # Step 3: Rerank using cross-encoder
    reranked = rerank(question, candidates, top_n=config.TOP_N_RERANKED)
    num_reranked = len(reranked)

    # Step 4: Relevance gate — if no documents pass the threshold
    if not reranked:
        return {
            "answer": generate_no_context_response(),
            "sources": [],
            "num_candidates": num_candidates,
            "num_reranked": 0,
        }

    # Step 5: Generate answer using Gemini
    answer = generate_answer(
        query=question,
        context_chunks=reranked,
        conversation_history=conversation_history,
        api_key=api_key,
    )

    # Extract source citations
    sources = []
    seen = set()
    for chunk in reranked:
        meta = chunk.get("metadata", {})
        source_key = f"{meta.get('source', 'Unknown')}|{meta.get('page', '?')}"
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "source": meta.get("source", "Unknown"),
                "page": meta.get("page", "?"),
                "excerpt": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "score": chunk.get("rerank_score", 0),
            })

    return {
        "answer": answer,
        "sources": sources,
        "num_candidates": num_candidates,
        "num_reranked": num_reranked,
    }
