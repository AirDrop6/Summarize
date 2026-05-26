"""CRUD-операции для истории запросов."""
import hashlib
from dataclasses import dataclass
from app.db.connection import get_conn


@dataclass
class GenerationParams:
    min_length: int
    max_length: int
    num_beams: int = 4
    no_repeat_ngram_size: int = 3


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_or_create_model(name: str, revision: str = "main") -> int:
    """Идемпотентно создаёт запись модели и возвращает её id."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO models (name, revision) VALUES (%s, %s)
            ON CONFLICT (name, revision) DO UPDATE SET name = EXCLUDED.name
            RETURNING model_id
            """,
            (name, revision),
        )
        return cur.fetchone()[0]


def get_or_create_document(
    full_text: str, source_type: str, filename: str | None
) -> int:
    """Возвращает document_id; если документ с таким хэшем уже есть — возвращает существующий."""
    content_hash = _sha256(full_text)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT document_id FROM documents WHERE content_hash = %s",
            (content_hash,),
        )
        row = cur.fetchone()
        if row:
            return row[0]

        cur.execute(
            """
            INSERT INTO documents (content_hash, source_type, filename, full_text, char_count)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING document_id
            """,
            (content_hash, source_type, filename, full_text, len(full_text)),
        )
        return cur.fetchone()[0]


def save_summary(
    document_id: int,
    model_id: int,
    summary_text: str,
    used_extractive: bool,
    latency_ms: int,
    params: GenerationParams,
) -> int:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO summaries
                (document_id, model_id, summary_text, char_count, used_extractive, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING summary_id
            """,
            (document_id, model_id, summary_text, len(summary_text), used_extractive, latency_ms),
        )
        summary_id = cur.fetchone()[0]
        cur.execute(
            """
            INSERT INTO generation_params
                (summary_id, min_length, max_length, num_beams, no_repeat_ngram_size)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (summary_id, params.min_length, params.max_length,
             params.num_beams, params.no_repeat_ngram_size),
        )
        return summary_id