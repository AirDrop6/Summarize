"""Fine-tune seq2seq модели на парах (text, summary).

Локально на CPU:
    python -m training.finetune \
        --base-model IlyaGusev/rut5_base_sum_gazeta \
        --source ntmerl \
        --output-dir ./models/ruT5-smoke \
        --epochs 1 --batch-size 2 --max-train-samples 50

В Kaggle/Colab с GPU:
    python -m training.finetune \
        --base-model IlyaGusev/rut5_base_sum_gazeta \
        --source ntmerl \
        --output-dir ./models/ruT5-ntmerl \
        --epochs 3 --batch-size 4 --fp16
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from training.prepare_dataset import load_ntmerl, load_gazeta, load_custom_jsonl


def build_compute_metrics(tokenizer):
    """Лёгкий ROUGE-1 на леммах — только для мониторинга обучения.
    Полные метрики (ROUGE-1/2/L + BLEU + BERTScore) считаются отдельно через
    evaluation/rouge_eval.py после обучения.
    """
    # Импорт внутри функции, чтобы не тащить лемматизатор без необходимости
    from evaluation.rouge_eval import lemmatize_tokens, rouge_n

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        # Заменяем -100 в labels на pad_token_id перед декодом
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        scores = []
        for pred, ref in zip(decoded_preds, decoded_labels):
            scores.append(rouge_n(lemmatize_tokens(pred), lemmatize_tokens(ref), 1))

        # Также средняя длина — полезно мониторить, не схлопывается ли модель в ""
        avg_pred_len = float(np.mean([len(p.split()) for p in decoded_preds]))
        return {
            "rouge1_train": float(np.mean(scores)),
            "gen_avg_len": avg_pred_len,
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="IlyaGusev/rut5_base_sum_gazeta")
    parser.add_argument("--output-dir", default="./models/ruT5-finetuned")
    parser.add_argument("--source", choices=["ntmerl", "gazeta", "custom"], default="ntmerl")
    parser.add_argument("--data-path", help="JSONL для source=custom")
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="Mixed precision (только GPU)")
    parser.add_argument("--save-steps-only-best", action="store_true",
                        help="Сохранять только лучший чекпоинт (экономит место в Kaggle)")
    args = parser.parse_args()

    device_info = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Fine-tune на устройстве: {device_info} ===")
    print(f"Base model: {args.base_model}")
    print(f"Dataset: {args.source}")
    print(f"Output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    if args.source == "ntmerl":
        raw = load_ntmerl(args.max_train_samples)
    elif args.source == "gazeta":
        raw = load_gazeta(args.max_train_samples)
    else:
        raw = load_custom_jsonl(args.data_path)

    print(f"\nРазмеры сплитов:")
    for k, v in raw.items():
        print(f"  {k}: {len(v)}")

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["text"],
            max_length=args.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["summary"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    use_fp16 = args.fp16 and torch.cuda.is_available()
    if args.fp16 and not torch.cuda.is_available():
        print("⚠ --fp16 запрошен, но GPU не найден. Игнорирую.")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2 if not args.save_steps_only_best else 1,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1_train",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
        logging_steps=50,
        fp16=use_fp16,
        report_to="none",
        # Чтобы Kaggle не подсасывал в W&B/HF Hub:
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Финальный eval
    print("\n=== Финальный eval на validation ===")
    eval_metrics = trainer.evaluate()
    print(eval_metrics)

    # Сохраняем сводку
    summary = {
        "base_model": args.base_model,
        "source": args.source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_train_samples": args.max_train_samples,
        "device": device_info,
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "final_eval": eval_metrics,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Модель сохранена в {args.output_dir}")
    print(f"✅ Сводка в {args.output_dir}/training_summary.json")
    print("\nЧтобы использовать модель в Streamlit-приложении:")
    print(f"  1) Скопируй папку {args.output_dir} в ./models/ внутри проекта")
    print(f"  2) В .env установи LOCAL_MODEL_PATH=/app/models/<имя_папки>")
    print(f"  3) docker compose restart app")


if __name__ == "__main__":
    main()