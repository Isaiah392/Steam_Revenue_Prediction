import numpy as np
import pandas as pd


FEATURES = [
    "price",
    "log_positive",
    "log_negative",
    "log_peak_ccu",
    "review_score",
    "release_year",
]


def parse_owners(x):
    try:
        x = str(x).strip()
        if x.lower() == "nan":
            return np.nan
        low, high = x.split("-")
        return (float(low.replace(",", "").strip()) + float(high.replace(",", "").strip())) / 2
    except Exception:
        return np.nan


def engineer_features(df):
    df = df.copy()

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year

    df["review_score"] = df["positive"] / (df["positive"] + df["negative"] + 1)
    df["owners_mid"] = df["estimated_owners"].apply(parse_owners)

    df["log_positive"] = np.log1p(df["positive"])
    df["log_negative"] = np.log1p(df["negative"])
    df["log_peak_ccu"] = np.log1p(df["peak_ccu"])

    return df


def merge_and_label(df_full, df_rev):
    df_full = engineer_features(df_full)

    df_labeled = df_full.merge(
        df_rev[["steamid", "revenue"]],
        left_on="appid",
        right_on="steamid",
        how="inner"
    )

    df_labeled = df_labeled[df_labeled["revenue"] >= 0].copy()
    df_labeled["log_revenue"] = np.log1p(df_labeled["revenue"])

    print(f"Matched labeled rows after merge: {len(df_labeled)}")
    return df_labeled


def get_train_data(df_labeled):
    df_train = df_labeled.dropna(subset=FEATURES + ["log_revenue"]).copy()
    print(f"Training rows after dropping nulls: {len(df_train)}")

    X = df_train[FEATURES]
    y = df_train["log_revenue"]
    return X, y, df_train


def get_prediction_data(df_full, labeled_ids):
    df_full = engineer_features(df_full)
    df_pred = df_full[~df_full["appid"].isin(labeled_ids)].copy()
    df_pred = df_pred.dropna(subset=FEATURES)
    print(f"Unlabeled games available for prediction: {len(df_pred)}")
    return df_pred