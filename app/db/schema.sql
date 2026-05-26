-- Схема БД для сервиса генерации аннотаций. Нормализация: 3НФ.
-- Создаётся при первом старте контейнера postgres (init-скрипт).

CREATE TABLE IF NOT EXISTS models (
    model_id     SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    revision     TEXT NOT NULL DEFAULT 'main',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (name, revision)
);

CREATE TABLE IF NOT EXISTS documents (
    document_id   BIGSERIAL PRIMARY KEY,
    content_hash  CHAR(64) NOT NULL UNIQUE,      -- sha256, защита от дублей
    source_type   TEXT NOT NULL CHECK (source_type IN ('txt', 'pdf', 'docx')),
    filename      TEXT,                           -- оригинальное имя файла, может быть NULL
    full_text     TEXT NOT NULL,
    char_count    INTEGER NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

CREATE TABLE IF NOT EXISTS summaries (
    summary_id    BIGSERIAL PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    model_id      INTEGER NOT NULL REFERENCES models(model_id),
    summary_text  TEXT NOT NULL,
    char_count    INTEGER NOT NULL,
    used_extractive BOOLEAN NOT NULL,             -- был ли применён LexRank
    latency_ms    INTEGER NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_document_id ON summaries(document_id);
CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at DESC);

CREATE TABLE IF NOT EXISTS generation_params (
    summary_id   BIGINT PRIMARY KEY REFERENCES summaries(summary_id) ON DELETE CASCADE,
    min_length   INTEGER NOT NULL,
    max_length   INTEGER NOT NULL,
    num_beams    INTEGER NOT NULL DEFAULT 4,
    no_repeat_ngram_size INTEGER NOT NULL DEFAULT 3
);