from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "income"

DATASET_NAME = "wenruliu/adult-income-dataset"
DATA_FILE_NAME = "adult.csv"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

BEST_MODEL_NAME = "best_random_forest_model.joblib"