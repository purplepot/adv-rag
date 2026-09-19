"""
PDF text extraction using PyMuPDF (fitz).
Extracts text page-by-page from PDF documents with metadata.
"""

import os
import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a single PDF file.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        List of dicts with keys: 'text', 'metadata'
        Each dict represents one page.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    filename = os.path.basename(file_path)
    doc = pymupdf.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if text:  # Skip empty pages
            pages.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1,  # 1-indexed
                    "total_pages": len(doc),
                }
            })

    doc.close()
    return pages


def load_all_pdfs(directory: str) -> list[dict]:
    """
    Load and extract text from all PDF files in a directory.
    
    Args:
        directory: Path to the directory containing PDF files.
        
    Returns:
        Combined list of page dicts from all PDFs.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Documents directory not found: {directory}")

    all_pages = []
    pdf_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"⚠️  No PDF files found in {directory}")
        return all_pages

    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        try:
            pages = load_pdf(file_path)
            all_pages.extend(pages)
            print(f"  ✅ Loaded: {pdf_file} ({len(pages)} pages)")
        except Exception as e:
            print(f"  ❌ Error loading {pdf_file}: {e}")

    return all_pages
