"""
Local embedding generation using sentence-transformers (MiniLM-L6-v2).
Singleton pattern to avoid reloading the model on every call.
"""

from sentence_transformers import SentenceTransformer
import config

_model = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (singleton)."""
    global _model
    if _model is None:
        print(f"🔄 Loading embedding model: {config.EMBEDDING_MODEL}...")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded ({config.EMBEDDING_DIMENSION}D vectors)")
    return _model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        List of embedding vectors (each is a list of floats).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,
        batch_size=32,
    )
    return embeddings.tolist()


def get_single_embedding(text: str) -> list[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Text string to embed.
        
    Returns:
        Embedding vector as a list of floats.
    """
    return get_embeddings([text])[0]
