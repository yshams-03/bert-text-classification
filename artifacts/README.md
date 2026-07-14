# Model and dataset artifacts

Large checkpoints and processed datasets are intentionally excluded from Git. Store them in an artifact registry, object store, or release asset and record their SHA-256 values in `artifacts/manifest.json`.

Generate or refresh the manifest from the repository root with:

```bash
python -m src.create_manifest
```

The manifest records relative paths, byte sizes, SHA-256 checksums, and the generated timestamp. Verify an artifact before deployment by comparing its checksum with the manifest.

The trained model also contains `checkpoints/run_manifest.json`, which records the dataset, tokenizer text format, hyperparameters, random seeds, and library versions used for training.
