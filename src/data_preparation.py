"""Prepare DBpedia 14: merge text, stratified val split, tokenize, save to disk."""

from __future__ import annotations

from typing import Any

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BertTokenizer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load and return the YAML config as a dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_dataset(dataset_name: str) -> DatasetDict:
    """Load the raw Hugging Face dataset (train/test splits)."""
    return load_dataset(dataset_name)


def merge_text_fields(example: dict, template: str) -> dict:
    """Create a ``text`` field from title/content using the given template."""
    example["text"] = template.format(title=example["title"], content=example["content"])
    return example


def create_stratified_val_split(
    train_dataset: Dataset,
    val_split: float,
    seed: int,
) -> DatasetDict:
    """Split train into stratified train/validation DatasetDict."""
    split = train_dataset.train_test_split(
        test_size=val_split,
        stratify_by_column="label",
        seed=seed,
    )
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: BertTokenizer,
    max_length: int,
) -> DatasetDict:
    """Tokenize text fields and format columns for the Hugging Face Trainer."""

    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset_dict.map(_tokenize, batched=True)
    # Trainer expects a "labels" column.
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized


def main() -> None:
    """Run the full data preparation pipeline and save to disk."""
    config = load_config()
    data_cfg = config["data"]
    model_cfg = config["model"]

    raw = load_raw_dataset(data_cfg["dataset_name"])
    template = data_cfg["text_template"]

    raw = DatasetDict(
        {
            "train": raw["train"].map(
                lambda ex: merge_text_fields(ex, template),
            ),
            "test": raw["test"].map(
                lambda ex: merge_text_fields(ex, template),
            ),
        }
    )

    splits = create_stratified_val_split(
        raw["train"],
        val_split=data_cfg["val_split"],
        seed=data_cfg["val_seed"],
    )
    dataset_dict = DatasetDict(
        {
            "train": splits["train"],
            "validation": splits["validation"],
            "test": raw["test"],
        }
    )

    tokenizer = BertTokenizer.from_pretrained(model_cfg["base_model"])
    tokenized = tokenize_dataset(dataset_dict, tokenizer, data_cfg["max_length"])

    out_dir = data_cfg["processed_data_dir"]
    tokenized.save_to_disk(out_dir)

    print(f"train: {len(tokenized['train'])}")
    print(f"validation: {len(tokenized['validation'])}")
    print(f"test: {len(tokenized['test'])}")


if __name__ == "__main__":
    main()
