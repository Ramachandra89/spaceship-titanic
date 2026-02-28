"""Derive new features from preprocessed data."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def _add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in _SPEND_COLS if c in df.columns]
    if available:
        df["TotalSpend"] = df[available].sum(axis=1)
    return df


def _add_spend_per_age(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalSpend" in df.columns and "Age" in df.columns:
        df["SpendPerAge"] = df["TotalSpend"] / (df["Age"] + 1)
    return df


def _add_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    if "GroupSize" in df.columns:
        df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps."""
    logger.info("Running feature engineering (shape=%s)", df.shape)
    df = df.copy()
    df = _add_total_spend(df)
    df = _add_spend_per_age(df)
    df = _add_is_alone(df)
    return df
