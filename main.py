import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def evaluate(model, name, X_test, y_test):
    preds = model.predict(X_test)
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))


def clip_predictions_from_log(preds_log):
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0, None)
    return preds


def parse_owners(x):
    try:
        x = str(x).strip()
        if x.lower() == "nan":
            return np.nan
        low, high = x.split("-")
        low = float(low.replace(",", "").strip())
        high = float(high.replace(",", "").strip())
        return (low + high) / 2
    except Exception:
        return np.nan


# -----------------------------
# LOAD REVENUE DATASET
# -----------------------------
rev_path = kagglehub.dataset_download(
    "alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024"
)
rev_file = [f for f in os.listdir(rev_path) if f.endswith(".csv")][0]
df_rev = pd.read_csv(os.path.join(rev_path, rev_file))

# -----------------------------
# LOAD FULL STEAM DATASET
# -----------------------------
full_path = kagglehub.dataset_download("fronkongames/steam-games-dataset")
full_file = [f for f in os.listdir(full_path) if f.endswith(".csv")][0]
df_full = pd.read_csv(os.path.join(full_path, full_file), index_col=False)

if str(df_full.columns[0]).lower().startswith("unnamed"):
    df_full = df_full.drop(columns=df_full.columns[0])

# -----------------------------
# CLEAN COLUMN NAMES
# -----------------------------
df_rev = clean_column_names(df_rev)
df_full = clean_column_names(df_full)

# -----------------------------
# CLEAN REVENUE DATASET
# -----------------------------
df_rev["steamid"] = pd.to_numeric(df_rev["steamid"], errors="coerce")
df_rev["price"] = pd.to_numeric(df_rev["price"], errors="coerce")
df_rev["revenue"] = pd.to_numeric(df_rev["revenue"], errors="coerce")
df_rev["copiessold"] = pd.to_numeric(df_rev["copiessold"], errors="coerce")
df_rev["avgplaytime"] = pd.to_numeric(df_rev["avgplaytime"], errors="coerce")
df_rev["reviewscore"] = pd.to_numeric(df_rev["reviewscore"], errors="coerce")

# -----------------------------
# CLEAN FULL DATASET
# -----------------------------
numeric_cols = [
    "appid",
    "price",
    "positive",
    "negative",
    "peak_ccu",
    "average_playtime_forever",
]

for col in numeric_cols:
    df_full[col] = (
        df_full[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("nan", np.nan)
    )
    df_full[col] = pd.to_numeric(df_full[col], errors="coerce")

df_full["release_date"] = pd.to_datetime(df_full["release_date"], errors="coerce")
df_full["release_year"] = df_full["release_date"].dt.year

df_full["review_score"] = df_full["positive"] / (
    df_full["positive"] + df_full["negative"] + 1
)

df_full["owners_mid"] = df_full["estimated_owners"].apply(parse_owners)

# log features from the full steam dataset
df_full["log_positive"] = np.log1p(df_full["positive"])
df_full["log_negative"] = np.log1p(df_full["negative"])
df_full["log_peak_ccu"] = np.log1p(df_full["peak_ccu"])

# -----------------------------
# MERGE REVENUE LABELS ONTO FULL DATASET
# -----------------------------
df_labeled = df_full.merge(
    df_rev[["steamid", "revenue"]],
    left_on="appid",
    right_on="steamid",
    how="inner"
)

print("Revenue rows:", len(df_rev))
print("Full rows:", len(df_full))
print("Matched labeled rows:", len(df_labeled))
print(df_labeled[["appid", "name", "revenue"]].head())

# -----------------------------
# TARGET
# -----------------------------
df_labeled = df_labeled[df_labeled["revenue"] >= 0].copy()
df_labeled["log_revenue"] = np.log1p(df_labeled["revenue"])

# -----------------------------
# FEATURES
# -----------------------------
features = [
    "price",
    "log_positive",
    "log_negative",
    "log_peak_ccu",
    "review_score",
    "release_year",
]

df_train = df_labeled.dropna(subset=features + ["log_revenue"]).copy()

print("\nTraining rows after dropna:", len(df_train))
print(df_train[features + ["log_revenue"]].head())

X = df_train[features]
y = df_train["log_revenue"]

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODELS
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
xgb.fit(X_train, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
evaluate(xgb, "XGBoost", X_test, y_test)
evaluate(rf, "Random Forest", X_test, y_test)
evaluate(gb, "Gradient Boosting", X_test, y_test)

# choose best model manually here
best_model = gb

# -----------------------------
# PREDICT ON UNLABELED FULL DATASET
# -----------------------------
labeled_ids = set(df_labeled["appid"].dropna().astype(int))

df_pred = df_full[~df_full["appid"].isin(labeled_ids)].copy()
df_pred = df_pred.dropna(subset=features)

print("\nUnlabeled prediction rows:", len(df_pred))

preds_log = best_model.predict(df_pred[features])
preds_revenue = clip_predictions_from_log(preds_log)

df_pred["predicted_revenue"] = preds_revenue

print("\nSample predictions:")
print(df_pred[["appid", "name", "predicted_revenue"]].head(20))

df_pred[["appid", "name", "predicted_revenue"]].to_csv(
    "predicted_steam_revenues.csv",
    index=False
)

print("\nSaved predictions to predicted_steam_revenues.csv")

# -----------------------------
# VISUALS
# -----------------------------

# 1) Correlation heatmap on labeled training data
corr_cols = [
    "price",
    "log_positive",
    "log_negative",
    "log_peak_ccu",
    "review_score",
    "release_year",
    "log_revenue",
]
corr = df_train[corr_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlation Heatmap (Labeled Training Data)")
plt.tight_layout()
plt.show()

# 2) Price vs real revenue
plt.figure(figsize=(7, 5))
plt.scatter(df_train["price"], np.expm1(df_train["log_revenue"]), alpha=0.5)
plt.xlabel("Price")
plt.ylabel("Real Revenue")
plt.title("Price vs Real Revenue (Top Revenue Games)")
plt.tight_layout()
plt.show()

# 3) Positive reviews vs real revenue
plt.figure(figsize=(7, 5))
plt.scatter(df_train["positive"], np.expm1(df_train["log_revenue"]), alpha=0.5)
plt.xlabel("Positive Reviews")
plt.ylabel("Real Revenue")
plt.title("Positive Reviews vs Real Revenue")
plt.tight_layout()
plt.show()

# 4) Peak CCU vs real revenue
plt.figure(figsize=(7, 5))
plt.scatter(df_train["peak_ccu"], np.expm1(df_train["log_revenue"]), alpha=0.5)
plt.xlabel("Peak CCU")
plt.ylabel("Real Revenue")
plt.title("Peak CCU vs Real Revenue")
plt.tight_layout()
plt.show()

# 5) Price vs predicted revenue
plt.figure(figsize=(7, 5))
plt.scatter(df_pred["price"], df_pred["predicted_revenue"], alpha=0.3)
plt.xlabel("Price")
plt.ylabel("Predicted Revenue")
plt.title("Price vs Predicted Revenue (Unlabeled Steam Games)")
plt.tight_layout()
plt.show()

# 6) Positive reviews vs predicted revenue
plt.figure(figsize=(7, 5))
plt.scatter(df_pred["positive"], df_pred["predicted_revenue"], alpha=0.3)
plt.xlabel("Positive Reviews")
plt.ylabel("Predicted Revenue")
plt.title("Positive Reviews vs Predicted Revenue")
plt.tight_layout()
plt.show()

# 7) Peak CCU vs predicted revenue
plt.figure(figsize=(7, 5))
plt.scatter(df_pred["peak_ccu"], df_pred["predicted_revenue"], alpha=0.3)
plt.xlabel("Peak CCU")
plt.ylabel("Predicted Revenue")
plt.title("Peak CCU vs Predicted Revenue")
plt.tight_layout()
plt.show()

# 8) Distribution comparison
plt.figure(figsize=(8, 5))
plt.hist(df_train["log_revenue"], bins=40, alpha=0.5, label="Real Log Revenue")
plt.hist(np.log1p(df_pred["predicted_revenue"]), bins=40, alpha=0.5, label="Predicted Log Revenue")
plt.legend()
plt.title("Distribution: Real vs Predicted Log Revenue")
plt.tight_layout()
plt.show()

# 9) Top predicted games bar chart
top_games = df_pred.sort_values("predicted_revenue", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_games["name"], top_games["predicted_revenue"])
plt.xlabel("Predicted Revenue")
plt.title("Top 10 Predicted Revenue Games")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 10) Feature importance for Random Forest
rf_importances = rf.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(features, rf_importances)
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()