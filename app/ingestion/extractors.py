"""Extract plain text from uploaded files (PDF / DOCX / MD / TXT / HTML).

Heavy parsers (pypdf, python-docx, bs4) are imported lazily inside each branch
so ``app.main`` imports even when a parser isn't installed; a missing parser
surfaces as a clear ExtractionError when that file type is actually uploaded.
"""

from __future__ import annotations

import io
import logging
import re

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Raised when a file can't be parsed into text."""


# Map common extensions to a normalized type.
_EXT_TYPE = {
    "pdf": "pdf",
    "docx": "docx",
    "doc": "docx",
    "md": "markdown",
    "markdown": "markdown",
    "txt": "text",
    "text": "text",
    "html": "html",
    "htm": "html",
}

SUPPORTED_EXTENSIONS = sorted(set(_EXT_TYPE))


def detect_type(filename: str, content_type: str = "") -> str:
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if ext in _EXT_TYPE:
        return _EXT_TYPE[ext]
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return "pdf"
    if "word" in ct or "officedocument" in ct:
        return "docx"
    if "html" in ct:
        return "html"
    if "markdown" in ct:
        return "markdown"
    return "text"


def extract_text(raw: bytes, filename: str, content_type: str = "") -> str:
    """Return extracted plain text for a file's bytes. Raises ExtractionError."""
    kind = detect_type(filename, content_type)
    try:
        if kind == "pdf":
            return _from_pdf(raw)
        if kind == "docx":
            return _from_docx(raw)
        if kind == "html":
            return _from_html(_decode(raw))
        # markdown + text: keep as-is (markdown is readable text).
        return _decode(raw)
    except ExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"Failed to extract {kind} from {filename}: {exc}") from exc


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _from_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError("pypdf is not installed; cannot parse PDF") from exc
    reader = PdfReader(io.BytesIO(raw))
    parts = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(p.strip() for p in parts if p.strip())
    if not text.strip():
        raise ExtractionError("PDF contained no extractable text (scanned/image-only?)")
    return text


def _from_docx(raw: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:
        raise ExtractionError("python-docx is not installed; cannot parse DOCX") from exc
    document = docx.Document(io.BytesIO(raw))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def _from_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: crude tag strip if bs4 isn't available.
        return re.sub(r"<[^>]+>", " ", html)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()
