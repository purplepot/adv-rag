"""
Text chunking for RAG pipeline.
Splits document text into overlapping chunks with metadata preservation.
"""


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks using a recursive strategy.
    Tries to split on paragraph boundaries first, then sentences, then words.
    
    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        
    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try splitting by paragraphs first, then sentences, then by size
    separators = ["\n\n", "\n", ". ", " "]

    for separator in separators:
        if separator in text:
            parts = text.split(separator)
            chunks = []
            current_chunk = ""

            for part in parts:
                # Add separator back (except for first part)
                candidate = part if not current_chunk else current_chunk + separator + part

                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # Start new chunk with overlap from previous
                    if chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-chunk_overlap:]
                        current_chunk = overlap_text + separator + part
                    else:
                        current_chunk = part

                    # If a single part is too large, force-split it
                    if len(current_chunk) > chunk_size:
                        while len(current_chunk) > chunk_size:
                            chunks.append(current_chunk[:chunk_size].strip())
                            current_chunk = current_chunk[chunk_size - chunk_overlap:]

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if chunks:
                return chunks

    # Fallback: hard split by character count
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    return [c for c in chunks if c]


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:
    """
    Split a list of document pages into chunks with metadata.
    
    Args:
        documents: List of dicts with 'text' and 'metadata' keys.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of chunk dicts with 'text', 'metadata' (including chunk_index).
    """
    all_chunks = []
    global_index = 0

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        chunks = _split_text(text, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id": f"{metadata['source']}_p{metadata['page']}_c{i}",
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "global_index": global_index,
                }
            })
            global_index += 1

    return all_chunks
