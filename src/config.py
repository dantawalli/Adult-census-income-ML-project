from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COLUMN = "income"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "adult.csv"

MODEL_DIR = BASE_DIR / "models"

BEST_MODEL_NAME = "best_random_forest_model.joblib"