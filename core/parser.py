"""Read text from PDF / TXT / pasted input."""
import io
import re

from pypdf import PdfReader


def pdf_to_text(file) -> str:
    """Extract text from an uploaded PDF (file-like object)."""
    reader = PdfReader(io.BytesIO(file.read()))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return clean_text("\n".join(pages))


def uploaded_to_text(uploaded) -> str:
    """Streamlit UploadedFile -> text (supports .pdf and .txt)."""
    if uploaded is None:
        return ""
    if uploaded.name.lower().endswith(".pdf"):
        return pdf_to_text(uploaded)
    return clean_text(uploaded.read().decode("utf-8", errors="ignore"))


def clean_text(text: str) -> str:
    text = text.replace("•", "- ").replace("", "- ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
