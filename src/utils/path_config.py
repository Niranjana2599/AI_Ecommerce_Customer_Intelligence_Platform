from pathlib import Path

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================
# DATA DIRECTORIES
# =========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

processed_DATA_DIR = DATA_DIR / "processed"

# =========================================================
# ARTIFACTS
# =========================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"

VECTOR_DB_DIR = ARTIFACTS_DIR / "vector_db"

FORECAST_DIR = ARTIFACTS_DIR / "forecast"

# =========================================================
# CREATE DIRECTORIES
# =========================================================

for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    processed_DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    VECTOR_DB_DIR,
    FORECAST_DIR
]:
    path.mkdir(parents=True, exist_ok=True)