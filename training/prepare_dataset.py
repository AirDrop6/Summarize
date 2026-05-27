"""Подготовка датасета для fine-tune.

Источники:
- `ntmerl` — IlyaGusev/rus_scientific_papers с HF (научный домен, CyberLeninka)
- `gazeta` — IlyaGusev/gazeta (новостной домен, для проверки пайплайна)
- `custom` — собственный JSONL с полями text и summary
"""
import argparse
import os
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

def load_small_student_corpus(
    repo_url: str = "https://github.com/Astromis/Small-Student-Science-Corpus.git",
    clone_dir: str = "/tmp/small_student_corpus",
    min_text_chars: int = 500,
    min_summary_chars: int = 50,
) -> Dataset:
    """Загружает датасет Small-Student-Science-Corpus из GitHub.

    Структура репозитория: каждый текст — отдельный JSON-файл с полями
    "header", "abstract", "keys", "text".
    """
    import subprocess
    import json
    from pathlib import Path

    if not os.path.exists(clone_dir):
        print(f"Клонирую {repo_url}...")
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, clone_dir])

    # Поиск всех JSON-файлов в репозитории
    json_files = list(Path(clone_dir).rglob("*.json"))
    print(f"Найдено JSON-файлов: {len(json_files)}")

    rows = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        text = data.get("text", "") or ""
        summary = data.get("abstract", "") or ""

        if isinstance(text, list):
            text = " ".join(text)
        if isinstance(summary, list):
            summary = " ".join(summary)

        text = str(text).strip()
        summary = str(summary).strip()

        if len(text) >= min_text_chars and len(summary) >= min_summary_chars:
            rows.append({"text": text, "summary": summary})

    print(f"После фильтрации Small-Student: {len(rows)}")
    return Dataset.from_list(rows)


def load_combined(
    max_samples: int | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """ntmerl + Small-Student-Science-Corpus, объединённые в один train-сет.

    Test-сет берётся ТОЛЬКО из ntmerl, чтобы сравнение с экспериментами на
    чистом ntmerl было корректным.
    """
    import os
    # 1. ntmerl с уже подготовленными test/val
    ntmerl_ds = load_ntmerl(max_samples=None, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    # 2. Small-Student
    ss = load_small_student_corpus()

    # 3. Объединяем только TRAIN
    from datasets import concatenate_datasets
    combined_train = concatenate_datasets([ntmerl_ds["train"], ss])
    combined_train = combined_train.shuffle(seed=seed)

    if max_samples and len(combined_train) > max_samples:
        combined_train = combined_train.select(range(max_samples))

    print(f"\nОбъединённый train: {len(combined_train)} (ntmerl={len(ntmerl_ds['train'])} + ss={len(ss)})")

    return DatasetDict({
        "train": combined_train,
        "validation": ntmerl_ds["validation"],   # из ntmerl
        "test": ntmerl_ds["test"],               # из ntmerl
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ntmerl", "gazeta", "custom", "combined"], default="ntmerl")
    parser.add_argument("--path", help="Путь к JSONL (для source=custom)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.source == "ntmerl":
        ds = load_ntmerl(args.max_samples)
    elif args.source == "gazeta":
        ds = load_gazeta(args.max_samples)
    elif args.source == "combined":
        ds = load_combined(args.max_samples)
    else:
        ds = load_custom_jsonl(args.path)

    print(ds)
    if "train" in ds and len(ds["train"]) > 0:
        ex = ds["train"][0]
        print(f"\n=== Пример из train ===")
        print(f"text ({len(ex['text'])} симв.): {ex['text'][:300]}...")
        print(f"summary ({len(ex['summary'])} симв.): {ex['summary'][:300]}")
