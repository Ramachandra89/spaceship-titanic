"""Model training, evaluation, and MLflow logging."""
import logging
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    LEARNING_RATE,
    MAX_DEPTH,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    N_ESTIMATORS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != TARGET_COL]


def _save_confusion_matrix(cm: np.ndarray) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def _save_feature_importance(model: GradientBoostingClassifier, feature_cols: List[str]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    top_n = min(20, len(feature_cols))
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_cols[i] for i in indices[::-1]],
        importances[indices[::-1]],
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    plt.close()


def train_model(
    df: pd.DataFrame,
) -> Tuple[GradientBoostingClassifier, List[str], float]:
    """Train a GradientBoostingClassifier and log the run with MLflow.

    Returns
    -------
    model       : Fitted classifier.
    feature_cols: List of feature column names used.
    accuracy    : Validation accuracy.
    """
    logger.info("Starting model training (shape=%s)", df.shape)
    feature_cols = _feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    params = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "learning_rate": LEARNING_RATE,
        "random_state": RANDOM_STATE,
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        logger.info("Validation accuracy: %.4f", accuracy)

        _save_confusion_matrix(confusion_matrix(y_val, y_pred))
        mlflow.log_artifact(str(CONFUSION_MATRIX_PATH))

        _save_feature_importance(model, feature_cols)
        mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BEST_MODEL_PATH, "wb") as fh:
            pickle.dump(model, fh)

        logger.info("Model saved to %s  (run_id=%s)", BEST_MODEL_PATH, run.info.run_id)

    return model, feature_cols, accuracy
