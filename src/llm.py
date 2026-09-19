"""
Google Gemini LLM integration for answer generation.
Uses the google-genai SDK with system instructions for grounded responses.
"""

from google import genai
from google.genai import types
import config


def _get_client() -> genai.Client:
    """Create a Gemini API client."""
    api_key = config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError(
            "Google API key not found. Set GOOGLE_API_KEY in your .env file "
            "or enter it in the Streamlit sidebar."
        )
    return genai.Client(api_key=api_key)


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "Unknown")
        page = chunk.get("metadata", {}).get("page", "?")
        score = chunk.get("rerank_score", None)
        score_str = f" (relevance: {score:.3f})" if score is not None else ""

        context_parts.append(
            f"--- Context Chunk {i}{score_str} ---\n"
            f"[Source: {source}, Page {page}]\n"
            f"{chunk['text']}\n"
        )
    return "\n".join(context_parts)


def _format_history(conversation_history: list[dict]) -> str:
    """Format conversation history for the LLM prompt."""
    if not conversation_history:
        return ""

    history_parts = ["--- Previous Conversation ---"]
    for msg in conversation_history[-config.MAX_CONVERSATION_HISTORY * 2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_parts.append(f"{role}: {msg['content']}")

    return "\n".join(history_parts) + "\n\n"


def generate_answer(
    query: str,
    context_chunks: list[dict],
    conversation_history: list[dict] = None,
    api_key: str = None,
) -> str:
    """
    Generate an answer using Gemini, grounded in the provided context.
    
    Args:
        query: The user's question.
        context_chunks: List of reranked chunk dicts with 'text' and 'metadata'.
        conversation_history: Previous conversation messages.
        api_key: Optional API key override (e.g., from Streamlit sidebar).
        
    Returns:
        Generated answer string with source citations.
    """
    # Allow runtime API key override
    if api_key:
        original_key = config.GOOGLE_API_KEY
        config.GOOGLE_API_KEY = api_key

    try:
        client = _get_client()

        # Build the prompt
        context_str = _format_context(context_chunks)
        history_str = _format_history(conversation_history or [])

        user_prompt = (
            f"{history_str}"
            f"--- Retrieved Knowledge Base Context ---\n"
            f"{context_str}\n\n"
            f"--- User Question ---\n"
            f"{query}\n\n"
            f"Please answer the question based ONLY on the provided context. "
            f"Include source citations in the format [Source: filename, Page X]."
        )

        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=config.SYSTEM_PROMPT,
                temperature=config.GEMINI_TEMPERATURE,
            ),
        )

        return response.text

    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

    finally:
        if api_key:
            config.GOOGLE_API_KEY = original_key


def generate_no_context_response() -> str:
    """Return a standard response when no relevant context is found."""
    return (
        "I couldn't find relevant information in the provided documents. "
        "Please provide more context or ask about topics covered in the "
        "knowledge base."
    )
