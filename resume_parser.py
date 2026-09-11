"""
Extract plain text from an uploaded resume file.
Supports .pdf, .docx, .txt. Used by app/streamlit_app.py right after upload,
before the text goes into the classifier / recommender.
"""

from pathlib import Path
from typing import Union

import docx  # python-docx
import pypdf  # pypdf


def _read_pdf(file) -> str:
    reader = pypdf.PdfReader(file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)


def _read_txt(file) -> str:
    raw = file.read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return raw


def parse_resume(file, filename: Union[str, None] = None) -> str:
    """
    file: a file-like object (e.g. Streamlit's UploadedFile, or an open() handle).
    filename: original filename, used to pick the parser by extension.
              If omitted, tries file.name (Streamlit UploadedFile has this).
    """
    name = filename or getattr(file, "name", "")
    ext = Path(name).suffix.lower()

    if ext == ".pdf":
        return _read_pdf(file)
    if ext == ".docx":
        return _read_docx(file)
    if ext == ".txt":
        return _read_txt(file)

    raise ValueError(f"Unsupported resume file type: '{ext or 'unknown'}'. Use PDF, DOCX, or TXT.")
