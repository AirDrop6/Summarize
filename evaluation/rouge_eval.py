"""ROUGE-1/2/L + BLEU + BERTScore поверх лемматизированных токенов.

ROUGE — самописный через Counter и LCS (не зависим от токенизатора rouge_score).
BLEU — NLTK corpus_bleu с лемматизацией.
BERTScore — DeepPavlov/rubert-base-cased, опционально (--with-bertscore).

Запуск:
    # Baseline на gazeta
    python -m evaluation.rouge_eval --model IlyaGusev/rut5_base_sum_gazeta \
        --dataset gazeta --max-samples 200

    # Baseline на ru_sci_bench (out-of-domain)
    python -m evaluation.rouge_eval --model IlyaGusev/rut5_base_sum_gazeta \
        --dataset ru_sci_bench --max-samples 200

    # Зафайнтьюненная модель + BERTScore
    python -m evaluation.rouge_eval --model ./models/ruT5-ntmerl \
        --dataset ru_sci_bench --max-samples 200 --with-bertscore
"""
import argparse
import re
import json
from collections import Counter
from pathlib import Path

import pymorphy3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

_morph = pymorphy3.MorphAnalyzer()
_WORD_RE = re.compile(r"[а-яёА-ЯЁa-zA-Z0-9]+")


def lemmatize_tokens(text: str) -> list[str]:
    """Список лемм (в нижнем регистре)."""
    tokens = _WORD_RE.findall(text.lower())
    return [_morph.parse(t)[0].normal_form for t in tokens]


# ============ ROUGE ============

def _ngrams(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(matches: int, pred_total: int, ref_total: int) -> float:
    if pred_total == 0 or ref_total == 0:
        return 0.0
    p = matches / pred_total
    r = matches / ref_total
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def rouge_n(pred: list[str], ref: list[str], n: int) -> float:
    pred_ng = _ngrams(pred, n)
    ref_ng = _ngrams(ref, n)
    overlap = sum((pred_ng & ref_ng).values())
    return _f1(overlap, sum(pred_ng.values()), sum(ref_ng.values()))


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[len(b)]


def rouge_l(pred: list[str], ref: list[str]) -> float:
    return _f1(_lcs_length(pred, ref), len(pred), len(ref))


# ============ BLEU ============

def compute_bleu(predictions: list[list[str]], references: list[list[str]]) -> float:
    """Corpus-BLEU через NLTK. Считается на лемматизированных токенах."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    # references должны быть списком списков ссылок: [[ref1_tokens], [ref2_tokens], ...]
    refs_wrapped = [[r] for r in references]
    smoothing = SmoothingFunction().method1
    return corpus_bleu(refs_wrapped, predictions, smoothing_function=smoothing)


# ============ BERTScore ============

def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "DeepPavlov/rubert-base-cased",
) -> dict:
    """BERTScore P/R/F1, среднее по корпусу."""
    from bert_score import score as bert_score_fn

    P, R, F1 = bert_score_fn(
        predictions, references,
        model_type=model_type,
        num_layers=12,
        lang="ru",
        verbose=False,
    )
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


# ============ Основной цикл ============

def evaluate_model(
    model_name: str,
    dataset_name: str,
    max_samples: int = 200,
    with_bertscore: bool = False,
):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Device: {device}")

    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "gazeta":
        ds = load_dataset("IlyaGusev/gazeta", revision="v2.0", split="test")
        text_col, sum_col = "text", "summary"
    elif dataset_name == "ntmerl":
        from training.prepare_dataset import load_ntmerl
        ds = load_ntmerl()["test"]
        text_col, sum_col = "text", "summary"
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")

    ds = ds.select(range(min(max_samples, len(ds))))

    preds_text, refs_text = [], []
    rows = []

    for i, item in enumerate(ds):
        inputs = tokenizer(
            item[text_col], max_length=1024, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=200, min_length=50,
                num_beams=4, no_repeat_ngram_size=3,
                early_stopping=True,
            )
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        ref = item[sum_col]

        pred_lemmas = lemmatize_tokens(pred)
        ref_lemmas = lemmatize_tokens(ref)

        rows.append({
            "rouge1_f": rouge_n(pred_lemmas, ref_lemmas, 1),
            "rouge2_f": rouge_n(pred_lemmas, ref_lemmas, 2),
            "rougeL_f": rouge_l(pred_lemmas, ref_lemmas),
            "pred_len": len(pred_lemmas),
            "ref_len": len(ref_lemmas),
        })
        preds_text.append(pred)
        refs_text.append(ref)

        if (i + 1) % 20 == 0:
            print(f"  обработано {i + 1}/{len(ds)}")

    df = pd.DataFrame(rows)

    # BLEU поверх лемматизированных токенов
    pred_token_lists = [lemmatize_tokens(p) for p in preds_text]
    ref_token_lists = [lemmatize_tokens(r) for r in refs_text]
    bleu = compute_bleu(pred_token_lists, ref_token_lists)

    print("\n=== Усреднённые метрики ===")
    print(df.mean().to_string())
    print(f"BLEU (lemmatized, corpus): {bleu:.4f}")

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "n_samples": len(df),
        "rouge1_f": float(df["rouge1_f"].mean()),
        "rouge2_f": float(df["rouge2_f"].mean()),
        "rougeL_f": float(df["rougeL_f"].mean()),
        "bleu": float(bleu),
        "avg_pred_len": float(df["pred_len"].mean()),
        "avg_ref_len": float(df["ref_len"].mean()),
    }

    if with_bertscore:
        print("\n=== Считаю BERTScore (это медленно на CPU)... ===")
        bs = compute_bertscore(preds_text, refs_text)
        summary.update(bs)
        print(f"BERTScore: P={bs['bertscore_p']:.4f}, R={bs['bertscore_r']:.4f}, F1={bs['bertscore_f1']:.4f}")

    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="IlyaGusev/rut5_base_sum_gazeta")
    parser.add_argument("--dataset", choices=["gazeta", "ntmerl"], default="gazeta")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--with-bertscore", action="store_true",
                        help="Считать BERTScore (медленно на CPU, ~10-15 мин на 200 примерах)")
    parser.add_argument("--output-csv", default="rouge_results.csv")
    parser.add_argument("--output-json", default=None,
                        help="Куда сохранить агрегированную сводку (по умолчанию рядом с CSV)")
    args = parser.parse_args()

    df, summary = evaluate_model(
        args.model, args.dataset, args.max_samples, args.with_bertscore
    )
    df.to_csv(args.output_csv, index=False)

    json_path = args.output_json or args.output_csv.replace(".csv", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nДетализированно: {args.output_csv}")
    print(f"Сводка: {json_path}")