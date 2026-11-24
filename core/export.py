import io

from docx import Document


def to_docx(text: str) -> bytes:
    doc = Document()
    for line in text.splitlines():
        s = line.rstrip()
        if s.startswith(("- ", "* ", "• ")):
            doc.add_paragraph(s[2:], style="List Bullet")
        elif s.isupper() and len(s) < 40 and s:
            doc.add_heading(s.title(), level=2)
        else:
            doc.add_paragraph(s)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
