"""End-to-end pipeline runner: preprocessing → training → prediction."""
import logging

from src.config import FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR
from src.data_loader import load_test, load_train
from src.feature_engineering import engineer_features
from src.model import train_model
from src.predict import predict_test
from src.preprocessing import preprocess
from src.utils import ensure_dirs, setup_logging


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    ensure_dirs(PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, OUTPUTS_DIR)

    logger.info("=== Spaceship Titanic MLOps Pipeline ===")

    # 1. Load data
    logger.info("Step 1/4 — Loading raw data")
    train_df = load_train()
    test_df = load_test()

    # 2. Preprocess + feature engineering
    logger.info("Step 2/4 — Preprocessing & feature engineering")
    train_processed = preprocess(train_df, is_train=True)
    train_processed = engineer_features(train_processed)

    train_processed.to_csv(PROCESSED_DATA_DIR / "train_processed.csv", index=False)
    logger.info("Processed training data saved to %s", PROCESSED_DATA_DIR / "train_processed.csv")

    # 3. Train model (MLflow logging included)
    logger.info("Step 3/4 — Training model")
    _model, _feature_cols, accuracy = train_model(train_processed)

    # 4. Generate test-set predictions / Kaggle submission
    logger.info("Step 4/4 — Generating test predictions")
    predict_test(test_df, list(train_processed.columns))

    logger.info("Pipeline complete — validation accuracy: %.4f", accuracy)


if __name__ == "__main__":
    main()
