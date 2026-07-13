"""Reusable DBpedia BERT classifier for CLI and FastAPI serving."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and return the YAML config as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class DBpediaClassifier:
    """Single- and batch-prediction wrapper around a fine-tuned BERT checkpoint."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        config_path: str = "configs/config.yaml",
    ) -> None:
        """Load model, tokenizer, and label names from config."""
        config = load_config(config_path)
        self.label_names: list[str] = list(config["model"]["label_names"])
        self.text_template: str = config["data"]["text_template"]
        self.max_length: int = config["data"]["max_length"]

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, title: str, content: str, top_k: int = 3) -> list[dict]:
        """Predict top-k labels for one title/content pair."""
        text = self.text_template.format(title=title, content=content)
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1)[0]
            k = min(top_k, probs.shape[0])
            values, indices = torch.topk(probs, k)

        return [
            {"label": self.label_names[int(idx)], "confidence": float(val)}
            for val, idx in zip(values.tolist(), indices.tolist())
        ]

    def predict_batch(
        self,
        items: list[dict],
        top_k: int = 3,
    ) -> list[list[dict]]:
        """Batched forward pass over items with ``title`` and ``content`` keys."""
        texts = [
            self.text_template.format(title=item["title"], content=item["content"])
            for item in items
        ]
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1)
            k = min(top_k, probs.shape[1])
            values, indices = torch.topk(probs, k, dim=-1)

        results: list[list[dict]] = []
        for row_vals, row_idxs in zip(values, indices):
            results.append(
                [
                    {
                        "label": self.label_names[int(idx)],
                        "confidence": float(val),
                    }
                    for val, idx in zip(row_vals.tolist(), row_idxs.tolist())
                ]
            )
        return results


def main() -> None:
    """CLI entry point: print top-k predictions as JSON."""
    config = load_config()
    parser = argparse.ArgumentParser(description="DBpedia BERT inference")
    parser.add_argument("--title", required=True, help="Entity title")
    parser.add_argument("--content", required=True, help="Entity abstract/content")
    parser.add_argument(
        "--top_k",
        type=int,
        default=config["inference"]["default_top_k"],
        help="Number of top predictions",
    )
    parser.add_argument(
        "--model_path",
        default=config["inference"]["model_path"],
        help="Path to fine-tuned model directory",
    )
    args = parser.parse_args()

    classifier = DBpediaClassifier(model_path=args.model_path)
    predictions = classifier.predict(args.title, args.content, top_k=args.top_k)
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
