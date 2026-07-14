# DBpedia 14 BERT Text Classification

BERT-based multi-class classifier for the DBpedia Ontology Classification dataset (14 classes). Target performance is ≥98.5% accuracy and macro-F1 on the held-out test set. Serving uses FastAPI, with an optional Docker image for deployment.

## Project structure

```
bert-text-classification/
├── configs/
│   └── config.yaml          # Single source of truth for hyperparameters and paths
├── src/
│   ├── data_preparation.py  # Load DBpedia 14, split, tokenize, save Arrow dataset
│   ├── train.py             # Fine-tune bert-base-uncased with early stopping
│   ├── evaluate.py          # Test-set metrics, report, confusion matrix
│   └── inference.py         # DBpediaClassifier + CLI predictions
├── deployment/
│   ├── api.py               # FastAPI /health and /predict
│   ├── Dockerfile           # Canonical container image for the API
│   └── requirements.txt     # Pinned training + serving dependencies
├── notebooks/               # Optional exploration notebooks
├── data/processed/          # Created by data_preparation.py (Arrow DatasetDict)
├── checkpoints/best_model/  # Created by train.py
└── reports/                 # Created by evaluate.py
```

## Setup

- Python 3.10 recommended (3.12 also works with the pinned stack)
- Install dependencies:

```bash
pip install -r deployment/requirements.txt
```

GPU (CUDA) is recommended for training. CPU works but is roughly ~10x slower.

## Usage

### a. Prepare data

```bash
python src/data_preparation.py
```

Downloads DBpedia 14, merges `title` and `content` with `{title} [SEP] {content}`, creates a stratified 90/10 train/validation split (seed 42), tokenizes with `bert-base-uncased` (max length 128), and saves to `data/processed/`. Expect ~504000 train / ~56000 validation / 70000 test.

### b. Train

```bash
python src/train.py
```

Fine-tunes for up to 3 epochs with AdamW (`lr=2e-5`), batch size 32, 10% linear warmup, linear decay, and early stopping (patience 2 on validation macro-F1). Runtime: ~2–4 hours on GPU; much longer on CPU. Early stopping may finish before 3 full epochs. Best weights + tokenizer are written to `checkpoints/best_model/`.

### c. Evaluate

```bash
python src/evaluate.py
```

Scores the test set, prints overall accuracy / macro-F1 with PASS/FAIL against the 98.5% targets, and writes:

- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

### d. Inference (CLI)

```bash
python src/inference.py --title "Apple Inc." --content "Apple Inc. is an American multinational technology company headquartered in Cupertino, California."
```

Prints top-3 predictions as JSON (`Company` expected first with high confidence).

### e. Serve via API

From the repo root:

```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Apple Inc.","abstract":"Apple Inc. is an American technology company.","top_k":3}'
```

The API field `abstract` is mapped to the classifier's `content` argument.

### f. Docker (volume-mount checkpoints — do not bake weights into the image)

```bash
docker build -t dbpedia-bert .
docker run -p 8000:8000 -v "$(pwd)/checkpoints:/app/checkpoints" dbpedia-bert
```

Windows PowerShell:

```powershell
docker build -t dbpedia-bert .
docker run -p 8000:8000 -v "${PWD}/checkpoints:/app/checkpoints" dbpedia-bert
```

`-v .../checkpoints:/app/checkpoints` mounts host `checkpoints/best_model` into the container. Rebuild the image after code changes only; swap weights without rebuilding. The root [`Dockerfile`](Dockerfile) is the canonical image definition. To bake weights in, uncomment its checkpoint `COPY` line.

Health / predict (PowerShell):

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -ContentType "application/json" -Body '{"title":"Apple Inc.","abstract":"Apple Inc. is an American technology company.","top_k":3}'
```

Or use `curl.exe` (not PowerShell's `curl` alias). Also note API: `py -m uvicorn deployment.api:app --host 0.0.0.0 --port 8000` if `uvicorn` is not on PATH.

## Configuration

All hyperparameters and paths live in `configs/config.yaml`:

| Section | Purpose |
|---------|---------|
| `data` | Dataset name, max length, text template, val split, processed path |
| `model` | Base model, num labels, dropout, label names |
| `training` | Device, LR, batch sizes, epochs, warmup, early stopping, checkpoints |
| `evaluation` | Test batch size, report paths, accuracy / macro-F1 targets |
| `inference` | Default model path and top-k for CLI |
| `api` | Host, port, model path for FastAPI |

## Expected results

| Metric | Target | Your results |
|--------|--------|--------------|
| Accuracy | ≥ 0.985 | **0.9934** (PASS) |
| Macro-F1 | ≥ 0.985 | **0.9934** (PASS) |

Val (best checkpoint, epoch 2): accuracy / macro-F1 **0.9928**. Train ~4h on RTX 4070.

## Troubleshooting

- **CUDA out of memory** — lower `training.batch_size` (and optionally `eval_batch_size`) in `configs/config.yaml`.
- **`stratify_by_column` error** — upgrade `datasets` so stratified `train_test_split` is supported (pinned version in `deployment/requirements.txt` includes it).
- **Slow CPU training** — confirm `torch.cuda.is_available()`; for experiments only, try a smaller `data.max_length` or fewer epochs (final runs should keep the locked config).
- **SSL / Hugging Face download fails** (`CERTIFICATE_VERIFY_FAILED`) — common on Windows with a corporate proxy. Install `pip-system-certs` so Python trusts the system CA store, then re-run `data_preparation.py`.
- **CUDA torch vs pinned `torch==`** — if you already have a CUDA build (e.g. `2.6.0+cu124`), install that from [pytorch.org](https://pytorch.org) first; then `pip install -r deployment/requirements.txt` so the remaining pins resolve against it without replacing the GPU wheel.
- **`uvicorn` / `docker` not recognized** — use `py -m uvicorn ...`. Install/start [Docker Desktop](https://www.docker.com/products/docker-desktop/) for `docker build` / `docker run` with the checkpoints volume mount.

## Reproducibility and project checks

- Training writes `checkpoints/run_manifest.json` with the dataset, text format, seeds, hyperparameters, and library versions.
- Run `python -m src.create_manifest` to create SHA-256 metadata for checkpoints and processed data in `artifacts/manifest.json` before publishing them to artifact storage.
- The canonical container definition is the root `Dockerfile`; `docker-compose.yml` uses it directly.
- Project metadata is defined in `pyproject.toml`, and GitHub Actions runs compilation, Ruff linting, and tests on every push and pull request.
