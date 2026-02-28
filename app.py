"""FastAPI serving application — /health and /predict endpoints."""
import logging
import pickle
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import BEST_MODEL_PATH, MLFLOW_MODEL_NAME, MLFLOW_TRACKING_URI
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spaceship Titanic Predictor",
    description=(
        "Serving flow: client → FastAPI (preprocess + infer) "
        "→ MLflow model → response (prediction + probability)"
    ),
    version="1.0.0",
)

# ── Model registry ────────────────────────────────────────────────────────────
_model = None


def _load_model():
    """Load the model from the MLflow registry (Production stage) or local pkl."""
    global _model
    if _model is not None:
        return _model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/Production"
        _model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded from MLflow registry: %s", model_uri)
    except Exception:
        logger.info("MLflow registry unavailable — loading local model from %s", BEST_MODEL_PATH)
        with open(BEST_MODEL_PATH, "rb") as fh:
            _model = pickle.load(fh)
    return _model


# ── Schemas ───────────────────────────────────────────────────────────────────
class PassengerFeatures(BaseModel):
    PassengerId: Optional[str] = None
    HomePlanet: Optional[str] = None
    CryoSleep: Optional[bool] = None
    Cabin: Optional[str] = None
    Destination: Optional[str] = None
    Age: Optional[float] = None
    VIP: Optional[bool] = None
    RoomService: Optional[float] = None
    FoodCourt: Optional[float] = None
    ShoppingMall: Optional[float] = None
    Spa: Optional[float] = None
    VRDeck: Optional[float] = None
    Name: Optional[str] = None


class PredictionResponse(BaseModel):
    passenger_id: Optional[str]
    transported: bool
    probability: float


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health():
    """Liveness / readiness check."""
    model_ready = Path(BEST_MODEL_PATH).exists()
    return {"status": "ok", "model_ready": model_ready}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(passenger: PassengerFeatures):
    """Predict whether a passenger was transported to another dimension.

    Send a JSON body with any subset of passenger fields.  Missing fields
    are handled by the preprocessing pipeline's imputation logic.
    """
    try:
        model = _load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    df = pd.DataFrame([passenger.dict()])

    try:
        processed = preprocess(df, is_train=False)
        processed = engineer_features(processed)

        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in processed.columns:
                    processed[col] = 0
            X = processed[list(model.feature_names_in_)]
        else:
            X = processed

        pred = bool(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        return PredictionResponse(
            passenger_id=passenger.PassengerId,
            transported=pred,
            probability=prob,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
