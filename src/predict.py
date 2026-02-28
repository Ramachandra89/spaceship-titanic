"""Generate predictions on the test set and write a Kaggle submission file."""
import logging
import pickle
from typing import List

import pandas as pd

from src.config import BEST_MODEL_PATH, SUBMISSION_PATH, TARGET_COL
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess

logger = logging.getLogger(__name__)


def load_model():
    """Load the locally saved model."""
    with open(BEST_MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def predict_test(test_df: pd.DataFrame, train_columns: List[str]) -> pd.DataFrame:
    """Run predictions on the raw test DataFrame.

    Parameters
    ----------
    test_df      : Raw test data (must contain ``PassengerId``).
    train_columns: All column names present in the processed training set
                   (including the target column).

    Returns
    -------
    submission : DataFrame with columns [PassengerId, Transported].
    """
    logger.info("Generating test predictions")
    passenger_ids = test_df["PassengerId"].copy()

    processed = preprocess(test_df, is_train=False)
    processed = engineer_features(processed)

    feature_cols = [c for c in train_columns if c != TARGET_COL]
    for col in feature_cols:
        if col not in processed.columns:
            processed[col] = 0

    model = load_model()

    if hasattr(model, "feature_names_in_"):
        X = processed[list(model.feature_names_in_)]
    else:
        X = processed[feature_cols]

    predictions = model.predict(X)

    submission = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Transported": predictions.astype(bool),
        }
    )

    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Submission saved to %s", SUBMISSION_PATH)
    return submission
