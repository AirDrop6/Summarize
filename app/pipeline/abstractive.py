"""Inference ruT5 / mBART для генерации аннотации."""
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import settings
from app.pipeline.extractive import lexrank_compress

_tokenizer = None
_model = None


def load_model():
    """Ленивая загрузка модели. Вызывается из streamlit_app через @cache_resource."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    model_id = settings.local_model_path or settings.model_name
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    _model.eval()
    return _tokenizer, _model


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def generate_summary(
    text: str,
    min_length: int,
    max_length: int,
    num_beams: int = 6,
    no_repeat_ngram_size: int = 4,
    length_penalty: float = 1.2,
) -> tuple[str, bool, int]:
    """
    Параметры min_length/max_length — в ТОКЕНАХ.
    Ориентировочно: 1 токен ≈ 3 символа кириллицы.

    Возвращает (аннотация, был_ли_использован_extractive, latency_ms).
    """
    tokenizer, model = load_model()

    used_extractive = False
    if _token_len(tokenizer, text) > settings.extractive_threshold_tokens:
        text = lexrank_compress(text, settings.extractive_target_sentences)
        used_extractive = True

    inputs = tokenizer(
        text,
        max_length=settings.max_input_tokens,
        truncation=True,
        return_tensors="pt",
    )

    # Генерируем чуть длиннее запрошенного, чтобы модель
    # не схлопнулась раньше времени и не оборвала фразу на полуслове.
    gen_min = max(min_length, 30)
    gen_max = max_length + 20

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            min_length=gen_min,
            max_length=gen_max,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
            repetition_penalty=1.2,
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    summary = trim_to_sentence_boundary(summary)
    return summary, used_extractive, latency_ms


def trim_to_sentence_boundary(text: str) -> str:
    """Обрезает текст до последнего полного предложения, если есть.

    Модель с early_stopping иногда обрывается посреди фразы. Лучше потерять
    последнее неполное предложение, чем выдать пользователю огрызок.
    """
    if not text:
        return text
    # Ищем последнее предложение, оканчивающееся на . ! ?
    matches = list(re.finditer(r"[.!?](?:\s|$)", text))
    if not matches:
        return text
    last_end = matches[-1].end()
    return text[:last_end].strip()