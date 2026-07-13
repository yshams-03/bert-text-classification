"""FastAPI service wrapping DBpediaClassifier.

Import path: repo root is added to sys.path so `from src.inference import ...`
works both via `uvicorn deployment.api:app` from the repo root and inside Docker
(WORKDIR /app with the same layout).
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure repo root is on sys.path (deployment/ sits one level below root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.inference import DBpediaClassifier  # noqa: E402

logger = logging.getLogger(__name__)

classifier: DBpediaClassifier | None = None


def load_config(config_path: str | None = None) -> dict:
    """Load YAML config from repo-root configs/config.yaml by default."""
    path = config_path or str(_REPO_ROOT / "configs" / "config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    title: str
    abstract: str
    top_k: int = Field(default=3, ge=1, le=14)


class Prediction(BaseModel):
    """Single label/confidence pair."""

    label: str
    confidence: float


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the classifier once at startup; clear on shutdown."""
    global classifier
    config = load_config()
    model_path = config["api"]["model_path"]
    # Resolve relative model path against repo root.
    resolved = Path(model_path)
    if not resolved.is_absolute():
        resolved = _REPO_ROOT / resolved
    try:
        classifier = DBpediaClassifier(
            model_path=str(resolved),
            device="auto",
            config_path=str(_REPO_ROOT / "configs" / "config.yaml"),
        )
        logger.info("Loaded DBpediaClassifier from %s", resolved)
    except Exception:
        logger.exception(
            "Failed to load model from %s. /predict will return 503 until fixed.",
            resolved,
        )
        classifier = None
    yield
    classifier = None


app = FastAPI(
    title="DBpedia BERT Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check; confirms the service is up and the model object exists."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok", model="dbpedia-bert-base")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify a title + abstract; maps abstract -> content for the classifier."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if request.title.strip() == "" and request.abstract.strip() == "":
        raise HTTPException(
            status_code=422,
            detail="title and abstract must not both be empty",
        )
    results = classifier.predict(
        title=request.title,
        content=request.abstract,
        top_k=request.top_k,
    )
    return PredictResponse(predictions=[Prediction(**r) for r in results])
