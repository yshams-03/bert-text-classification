"""Fine-tune bert-base-uncased on DBpedia 14 with Hugging Face Trainer.

Full training on GPU takes roughly 2-4 hours depending on hardware; on CPU expect
~10x longer.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and return the YAML config as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_setting: str) -> str:
    """Resolve training device from config (auto / cuda / cpu)."""
    setting = device_setting.lower()
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        return "cuda"
    if setting == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device setting: {device_setting}")


def compute_metrics(eval_pred) -> dict:
    """Compute accuracy and macro F1 for Trainer evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def build_model(config: dict) -> AutoModelForSequenceClassification:
    """Load AutoModelForSequenceClassification from config hyperparameters."""
    model_cfg = config["model"]
    return AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model"],
        num_labels=model_cfg["num_labels"],
        hidden_dropout_prob=model_cfg["dropout"],
    )


def main() -> None:
    """Fine-tune the model and save the best checkpoint."""
    config = load_config()
    train_cfg = config["training"]
    set_seed(train_cfg["seed"])

    dataset = load_from_disk(config["data"]["processed_data_dir"])
    model = build_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model"])

    device = get_device(train_cfg["device"])
    print(f"Training device: {device}")
    model.to(device)

    args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        num_train_epochs=train_cfg["num_epochs"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["early_stopping_metric"],
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg["early_stopping_patience"],
            )
        ],
    )

    trainer.train()

    best_dir = f"{train_cfg['output_dir']}/best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    metrics = trainer.evaluate()
    print(f"Final validation accuracy: {metrics.get('eval_accuracy')}")
    print(f"Final validation macro_f1: {metrics.get('eval_macro_f1')}")


if __name__ == "__main__":
    main()
