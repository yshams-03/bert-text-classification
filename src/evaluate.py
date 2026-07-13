"""Evaluate the best DBpedia BERT checkpoint on the held-out test set."""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and return the YAML config as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_path: str) -> tuple:
    """Load model and tokenizer from ``model_path``; put model in eval mode on device."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def run_predictions(
    model,
    test_dataset,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run batched inference; return (y_true, y_pred) as numpy arrays."""
    loader = DataLoader(test_dataset, batch_size=batch_size)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].numpy()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    return np.array(y_true), np.array(y_pred)


def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list,
    report_path: str,
) -> str:
    """Build, save, and return the sklearn classification report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4,
    )
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list,
    save_path: str,
) -> None:
    """Plot and save a labeled 14x14 confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("DBpedia 14 Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    """Evaluate best_model on the test split and write reports."""
    config = load_config()
    eval_cfg = config["evaluation"]
    label_names = config["model"]["label_names"]

    dataset = load_from_disk(config["data"]["processed_data_dir"])
    test_dataset = dataset["test"]

    model, _tokenizer, device = load_model_and_tokenizer(
        config["inference"]["model_path"],
    )
    y_true, y_pred = run_predictions(
        model,
        test_dataset,
        device=device,
        batch_size=eval_cfg["test_batch_size"],
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro_f1: {macro_f1:.4f}")

    acc_status = "PASS" if accuracy >= eval_cfg["target_accuracy"] else "FAIL"
    f1_status = "PASS" if macro_f1 >= eval_cfg["target_macro_f1"] else "FAIL"
    print(
        f"Accuracy target ({eval_cfg['target_accuracy']}): {acc_status} "
        f"(got {accuracy:.4f})"
    )
    print(
        f"Macro-F1 target ({eval_cfg['target_macro_f1']}): {f1_status} "
        f"(got {macro_f1:.4f})"
    )

    report = generate_report(
        y_true,
        y_pred,
        label_names,
        eval_cfg["classification_report_path"],
    )
    print(report)
    generate_confusion_matrix(
        y_true,
        y_pred,
        label_names,
        eval_cfg["confusion_matrix_path"],
    )
    print(f"Saved report to {eval_cfg['classification_report_path']}")
    print(f"Saved confusion matrix to {eval_cfg['confusion_matrix_path']}")


if __name__ == "__main__":
    main()
