import os
import json
import pandas as pd

# -------------------------------
# Setup Kaggle API credentials
# -------------------------------
def setup_kaggle_auth(kaggle_json_path="kaggle.json"):
    if not os.path.exists(kaggle_json_path):
        raise FileNotFoundError(
            f"{kaggle_json_path} not found. Please download it from Kaggle → Account → Create New API Token."
        )
    
    with open(kaggle_json_path, "r") as f:
        creds = json.load(f)
    
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]

# -------------------------------
# Paths & dataset info
# -------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_PATH, "datareq")
os.makedirs(DATA_FOLDER, exist_ok=True)

DATASET_ID = "svaningelgem/crypto-currencies-daily-prices"

# -------------------------------
# Download dataset (skips if exists)
# -------------------------------
def download_dataset(force=False):
    setup_kaggle_auth(os.path.join(BASE_PATH, "kaggle.json"))
    
    # Skip download if data already exists
    if not force and any(fname.endswith(".csv") for fname in os.listdir(DATA_FOLDER)):
        print("✅ Dataset already available. Skipping download.")
        return
    
    print("⬇️ Downloading dataset from Kaggle...")
    os.system(f'kaggle datasets download -d {DATASET_ID} -p "{DATA_FOLDER}" --unzip')
    print("✅ Download complete.")

# -------------------------------
# Load specific coin
# -------------------------------
def load_coin(coin: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_FOLDER, f"{coin}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{coin}.csv not found. Run download_dataset() first.")
    
    return pd.read_csv(file_path)
