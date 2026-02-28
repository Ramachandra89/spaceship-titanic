"""Load raw CSV data files and optionally download them from Kaggle."""
import logging
import os
import zipfile

import pandas as pd
from dotenv import load_dotenv

from src.config import RAW_DATA_DIR, SAMPLE_SUBMISSION, TEST_RAW, TRAIN_RAW

logger = logging.getLogger(__name__)

_KAGGLE_COMPETITION = "spaceship-titanic"


def download_raw_data() -> None:
    """Download competition CSVs from Kaggle using credentials in .env.

    Credentials are read from a ``.env`` file (or the environment) via the
    ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` variables.  The files are extracted
    into ``data/raw/`` and the downloaded zip is removed afterwards.

    This function is a no-op when all three raw CSV files already exist.
    """
    if TRAIN_RAW.exists() and TEST_RAW.exists() and SAMPLE_SUBMISSION.exists():
        logger.info("Raw data already present — skipping download.")
        return

    # Load .env so KAGGLE_USERNAME / KAGGLE_KEY are available to the kaggle SDK
    load_dotenv()

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set. "
            "Copy .env.example → .env and fill in your credentials."
        )

    import kaggle

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading '%s' competition files from Kaggle…", _KAGGLE_COMPETITION)
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        _KAGGLE_COMPETITION,
        path=str(RAW_DATA_DIR),
        quiet=False,
    )

    # Unzip the downloaded archive
    zip_path = RAW_DATA_DIR / f"{_KAGGLE_COMPETITION}.zip"
    if zip_path.exists():
        logger.info("Extracting %s → %s", zip_path, RAW_DATA_DIR)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DATA_DIR)
        zip_path.unlink()
        logger.info("Download complete.")


def load_train() -> pd.DataFrame:
    """Load the raw training CSV."""
    logger.info("Loading training data from %s", TRAIN_RAW)
    return pd.read_csv(TRAIN_RAW)


def load_test() -> pd.DataFrame:
    """Load the raw test CSV."""
    logger.info("Loading test data from %s", TEST_RAW)
    return pd.read_csv(TEST_RAW)


def load_sample_submission() -> pd.DataFrame:
    """Load the sample submission CSV."""
    logger.info("Loading sample submission from %s", SAMPLE_SUBMISSION)
    return pd.read_csv(SAMPLE_SUBMISSION)
