"""Извлечение plain-text из загруженных файлов."""
import io
from pypdf import PdfReader
from docx import Document


def parse_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # повреждённая страница — пропускаем
            continue
    return "\n".join(pages)


def parse_docx(file_bytes: bytes) -> str:
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_file(filename: str, file_bytes: bytes) -> tuple[str, str]:
    """Возвращает (текст, source_type)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(file_bytes), "pdf"
    if name.endswith(".docx"):
        return parse_docx(file_bytes), "docx"
    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore"), "txt"
    raise ValueError(f"Неподдерживаемый формат файла: {filename}")