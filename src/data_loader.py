"""Load raw CSV data files."""
import logging
import pandas as pd

from src.config import TRAIN_RAW, TEST_RAW, SAMPLE_SUBMISSION

logger = logging.getLogger(__name__)


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
