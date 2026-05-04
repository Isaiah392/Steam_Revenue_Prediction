import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def error_by_price_tier(df_test, y_test, preds):
    df = df_test.copy()
    df["actual"] = y_test.values
    df["predicted"] = preds
    df["residual"] = df["actual"] - df["predicted"]
    df["abs_error"] = np.abs(df["residual"])

    bins = [0, 5, 15, 30, 60, 999]
    labels = ["Free-$5", "$5-$15", "$15-$30", "$30-$60", "$60+"]
    df["price_tier"] = pd.cut(df["price"], bins=bins, labels=labels)

    tier_stats = df.groupby("price_tier", observed=True)["abs_error"].agg(["mean", "median", "count"])
    tier_stats.columns = ["Mean MAE", "Median MAE", "Count"]

    print("\nError breakdown by price tier:")
    print(tier_stats.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(tier_stats.index.astype(str), tier_stats["Mean MAE"])
    ax.set_xlabel("Price Tier")
    ax.set_ylabel("Mean Absolute Error (Log Revenue)")
    ax.set_title("Prediction Error by Price Tier")
    plt.tight_layout()
    plt.savefig("error_by_price_tier.png", dpi=150)
    plt.show()

    return tier_stats


def error_by_revenue_quartile(y_test, preds):
    actual = y_test.values
    residuals = actual - preds

    quartiles = pd.qcut(actual, q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    df = pd.DataFrame({"actual": actual, "residual": residuals, "quartile": quartiles})

    q_stats = df.groupby("quartile", observed=True)["residual"].agg(["mean", "std", "count"])
    q_stats.columns = ["Mean Residual", "Std", "Count"]

    print("\nResidual breakdown by actual revenue quartile:")
    print(q_stats.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(q_stats.index.astype(str), q_stats["Mean Residual"])
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Revenue Quartile")
    ax.set_ylabel("Mean Residual (Log Scale)")
    ax.set_title("Systematic Bias by Revenue Quartile")
    plt.tight_layout()
    plt.savefig("error_by_revenue_quartile.png", dpi=150)
    plt.show()

    return q_stats


def worst_predictions(df_test, y_test, preds, n=10):
    df = df_test[["name", "price"]].copy()
    df["actual_log_rev"] = y_test.values
    df["predicted_log_rev"] = preds
    df["abs_error"] = np.abs(df["actual_log_rev"] - df["predicted_log_rev"])
    df["actual_revenue"] = np.expm1(df["actual_log_rev"])
    df["predicted_revenue"] = np.expm1(df["predicted_log_rev"])

    worst = df.nlargest(n, "abs_error")[
        ["name", "price", "actual_revenue", "predicted_revenue", "abs_error"]
    ]

    print(f"\nTop {n} worst predictions:")
    print(worst.to_string(index=False))
    return worst


def run_full_analysis(df_test, y_test, preds):
    error_by_price_tier(df_test, y_test, preds)
    error_by_revenue_quartile(y_test, preds)
    worst_predictions(df_test, y_test, preds)