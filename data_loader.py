import os
import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FULL_DATASET_PATH = os.path.join(DATA_DIR, "steam_games.csv")
REVENUE_DATASET_PATH = os.path.join(DATA_DIR, "steam_revenue.csv")


def _check_files():
    missing = []
    if not os.path.exists(FULL_DATASET_PATH):
        missing.append("data/steam_games.csv")
    if not os.path.exists(REVENUE_DATASET_PATH):
        missing.append("data/steam_revenue.csv")
    if missing:
        raise FileNotFoundError(
            f"\nMissing data files: {missing}\n"
            "Download them from Kaggle and place them in the data/ folder.\n"
            "  steam_games.csv  -> https://www.kaggle.com/datasets/fronkongames/steam-games-dataset\n"
            "  steam_revenue.csv -> https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024\n"
        )


def load_revenue_dataset():
    _check_files()
    df = pd.read_csv(REVENUE_DATASET_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    for col in ["steamid", "price", "revenue", "copiessold", "avgplaytime", "reviewscore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Revenue dataset loaded: {len(df)} rows")
    return df


def load_full_dataset():
    _check_files()
    df = pd.read_csv(FULL_DATASET_PATH, index_col=False, low_memory=False)

    if str(df.columns[0]).lower().startswith("unnamed"):
        df = df.drop(columns=df.columns[0])

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    numeric_cols = ["appid", "price", "positive", "negative", "peak_ccu", "average_playtime_forever"]
    for col in numeric_cols:
        df[col] = (
            df[col].astype(str).str.replace(",", "", regex=False).replace("nan", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Full Steam dataset loaded: {len(df)} rows")
    return df