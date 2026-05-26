"""Подготовка датасета для fine-tune.

Источники:
- `ntmerl` — IlyaGusev/rus_scientific_papers с HF (научный домен, CyberLeninka)
- `gazeta` — IlyaGusev/gazeta (новостной домен, для проверки пайплайна)
- `custom` — собственный JSONL с полями text и summary
"""
import argparse
import re
from datasets import load_dataset, Dataset, DatasetDict


# Префикс "Длина аннотации NN | " в начале article_text у ntmerl
_LEAK_PREFIX_RE = re.compile(r"^Длина аннотации[:\s]*\d+\s*\|\s*", re.IGNORECASE)
# Символы-невидимки в начале строк (BOM, zero-width)
_INVISIBLE_RE = re.compile(r"[\ufeff\u200b\u200c\u200d]")


def _clean_ntmerl_text(text: str) -> str:
    """Убирает префикс с указанием длины аннотации (это data leak — модель
    может выучить, что длина целевой аннотации указана в начале входа)."""
    text = _INVISIBLE_RE.sub("", text)
    text = _LEAK_PREFIX_RE.sub("", text)
    return text.strip()


def load_ntmerl(
    max_samples: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    min_text_chars: int = 500,
    min_summary_chars: int = 50,
) -> DatasetDict:
    """Загружает ntmerl/rus_scientific_papers и делит на train/val/test.

    Фильтрация:
    - Удаляются строки с пустыми/None abstract_text.
    - Удаляются слишком короткие тексты и аннотации (мусор парсинга).
    - Префикс "Длина аннотации NN | " вычищается из текста.
    """
    raw = load_dataset("ntmerl/rus_scientific_papers", split="train")

    # Чистка
    def _map(example):
        return {
            "text": _clean_ntmerl_text(example["article_text"] or ""),
            "summary": (example["abstract_text"] or "").strip(),
        }

    cleaned = raw.map(_map, remove_columns=raw.column_names)

    # Фильтр по длинам и непустоте
    def _ok(example):
        return (
            len(example["text"]) >= min_text_chars
            and len(example["summary"]) >= min_summary_chars
        )

    filtered = cleaned.filter(_ok)
    print(f"После фильтрации осталось {len(filtered)} из {len(raw)} примеров")

    if max_samples:
        filtered = filtered.select(range(min(max_samples, len(filtered))))

    # Делим на train/val/test
    # сначала test, потом val из остатка
    split1 = filtered.train_test_split(test_size=test_ratio, seed=seed)
    test_ds = split1["test"]
    rest = split1["train"]
    val_size_adj = val_ratio / (1.0 - test_ratio)  # корректировка относительно остатка
    split2 = rest.train_test_split(test_size=val_size_adj, seed=seed)

    return DatasetDict({
        "train": split2["train"],
        "validation": split2["test"],
        "test": test_ds,
    })


def load_gazeta(max_samples: int | None = None) -> DatasetDict:
    ds = load_dataset("IlyaGusev/gazeta", revision="v2.0")
    if max_samples:
        ds["train"] = ds["train"].select(range(min(max_samples, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(range(min(1000, len(ds["validation"]))))
    return ds


def load_custom_jsonl(path: str) -> DatasetDict:
    """Ожидает JSONL с полями text и summary."""
    return load_dataset("json", data_files=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ntmerl", "gazeta", "custom"], default="ntmerl")
    parser.add_argument("--path", help="Путь к JSONL (для source=custom)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.source == "ntmerl":
        ds = load_ntmerl(args.max_samples)
    elif args.source == "gazeta":
        ds = load_gazeta(args.max_samples)
    else:
        ds = load_custom_jsonl(args.path)

    print(ds)
    if "train" in ds and len(ds["train"]) > 0:
        ex = ds["train"][0]
        print(f"\n=== Пример из train ===")
        print(f"text ({len(ex['text'])} симв.): {ex['text'][:300]}...")
        print(f"summary ({len(ex['summary'])} симв.): {ex['summary'][:300]}")