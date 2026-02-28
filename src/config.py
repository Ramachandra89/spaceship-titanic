"""Paths, constants, and hyperparameters for the Spaceship Titanic pipeline."""
from pathlib import Path

# ── Base directories ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ── File paths ────────────────────────────────────────────────────────────────
TRAIN_RAW = RAW_DATA_DIR / "train.csv"
TEST_RAW = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION = RAW_DATA_DIR / "sample_submission.csv"
TRAIN_PROCESSED = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_PROCESSED = PROCESSED_DATA_DIR / "test_processed.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SUBMISSION_PATH = OUTPUTS_DIR / "submission.csv"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "spaceship-titanic"
MLFLOW_MODEL_NAME = "spaceship-titanic-model"

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_COL = "Transported"

# ── Hyperparameters ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200
MAX_DEPTH = 5
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
LEARNING_RATE = 0.1
