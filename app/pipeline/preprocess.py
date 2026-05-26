"""Очистка и сегментация текста."""
import re
import nltk

_NLTK_READY = False


def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
    for pkg, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


def clean_text(text: str) -> str:
    """Базовая очистка: переносы строк, дефисы переносов, множественные пробелы."""
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    _ensure_nltk()
    return nltk.sent_tokenize(text, language="russian")


def get_russian_stopwords() -> frozenset[str]:
    """Русские стоп-слова из NLTK. Кешируется на уровне процесса."""
    _ensure_nltk()
    from nltk.corpus import stopwords
    return frozenset(stopwords.words("russian"))