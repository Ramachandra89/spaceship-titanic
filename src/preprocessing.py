"""Cleaning, missing-value imputation, and encoding."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
_CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side"]
_BOOL_COLS = ["CryoSleep", "VIP"]
_NUM_COLS = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNum"]


def _extract_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split Cabin → Deck, CabinNum, Side."""
    df = df.copy()
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cabin_split[0] if 0 in cabin_split.columns else pd.NA
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce") if 1 in cabin_split.columns else pd.NA
    df["Side"] = cabin_split[2] if 2 in cabin_split.columns else pd.NA
    df = df.drop(columns=["Cabin"])
    return df


def _extract_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive GroupId and GroupSize from PassengerId."""
    df = df.copy()
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("GroupId")["GroupId"].transform("count")
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values."""
    df = df.copy()
    for col in _NUM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in _CAT_COLS:
        if col in df.columns:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) else "Unknown")
    for col in _BOOL_COLS:
        if col in df.columns:
            mode = df[col].mode()
            fill_val = bool(mode[0]) if len(mode) else False
            df[col] = df[col].fillna(fill_val).astype(bool)
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode boolean and categorical columns."""
    df = df.copy()
    for col in _BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
    present_cats = [c for c in _CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present_cats, drop_first=False)
    return df


def preprocess(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Full preprocessing pipeline.

    Parameters
    ----------
    df:       Raw passenger DataFrame.
    is_train: When True the ``Transported`` target column is kept.
    """
    logger.info("Running preprocessing (is_train=%s, shape=%s)", is_train, df.shape)
    df = _extract_group_features(df)
    df = _extract_cabin_features(df)
    df = _fill_missing(df)
    df = _encode(df)
    drop_cols = ["PassengerId", "Name", "GroupId"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df
