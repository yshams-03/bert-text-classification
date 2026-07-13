# Project progress — DBpedia 14 BERT

Last updated: 2026-07-14

## Pipeline status

| Step | Status | Notes |
|------|--------|-------|
| 1. `configs/config.yaml` | Done | Locked hyperparameters |
| 2. `deployment/requirements.txt` | Done | `torch==2.6.0` / `transformers==4.48.3` / `accelerate==1.2.1` |
| 3. `src/data_preparation.py` | Done | Splits: **504000 / 56000 / 70000** |
| 4. `src/train.py` | Done | CUDA RTX 4070; ~4h; best val **0.9928** |
| 5. `src/evaluate.py` | Done | Test **0.9934 / 0.9934 PASS** |
| 6. `src/inference.py` | Done | Apple Inc. → Company @ ~0.999 |
| 7. `deployment/api.py` | Done | `/health` + `/predict` PASS |
| 8. `deployment/Dockerfile` | Done | Volume-mount checkpoints (default) |
| 9. `README.md` | Done | Results table filled |

## Test results

| Metric | Target | Result |
|--------|--------|--------|
| Accuracy | ≥ 0.985 | **0.9934 PASS** |
| Macro-F1 | ≥ 0.985 | **0.9934 PASS** |

Reports: `reports/classification_report.txt`, `reports/confusion_matrix.png`.

## Docker

```powershell
docker build -t dbpedia-bert -f deployment/Dockerfile .
docker run -p 8000:8000 -v "${PWD}/checkpoints:/app/checkpoints" dbpedia-bert
```

## Known environment notes

- Local torch: `2.6.0+cu124`; Docker: PyPI CPU `torch==2.6.0`
- HF TLS: `pip-system-certs`
- Serve locally: `py -m uvicorn deployment.api:app --host 0.0.0.0 --port 8000`
- PowerShell: use `Invoke-RestMethod` or `curl.exe` (not `curl` alias)
