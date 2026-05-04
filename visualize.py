import numpy as np
import matplotlib.pyplot as plt
from preprocessing import FEATURES


def plot_correlation_heatmap(df_train):
    cols = FEATURES + ["log_revenue"]
    corr = df_train[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    ax.set_title("Correlation Heatmap (Labeled Training Data)")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()


def plot_feature_importance(model, model_name="Random Forest"):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(FEATURES, importances)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_title(f"Feature Importance ({model_name})")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()


def plot_predicted_vs_actual(y_test, preds, model_name="Best Model"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.4, s=20)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Log Revenue")
    ax.set_ylabel("Predicted Log Revenue")
    ax.set_title(f"Predicted vs Actual ({model_name})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", dpi=150)
    plt.show()


def plot_residuals(y_test, preds, model_name="Best Model"):
    residuals = y_test - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(preds, residuals, alpha=0.4, s=20)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted Log Revenue")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residual Plot ({model_name})")

    axes[1].hist(residuals, bins=40, edgecolor="black", linewidth=0.4)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig("residuals.png", dpi=150)
    plt.show()


def plot_revenue_distribution(df_train, df_pred):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_train["log_revenue"], bins=40, alpha=0.55, label="Real (labeled)")
    ax.hist(
        np.log1p(df_pred["predicted_revenue"]),
        bins=40, alpha=0.55, label="Predicted (unlabeled)"
    )
    ax.set_xlabel("Log Revenue")
    ax.legend()
    ax.set_title("Distribution: Real vs Predicted Log Revenue")
    plt.tight_layout()
    plt.savefig("revenue_distribution.png", dpi=150)
    plt.show()


def plot_top_predicted_games(df_pred, n=10):
    top = df_pred.sort_values("predicted_revenue", ascending=False).head(n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["name"], top["predicted_revenue"])
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Revenue (USD)")
    ax.set_title(f"Top {n} Predicted Revenue Games")
    plt.tight_layout()
    plt.savefig("top_predicted_games.png", dpi=150)
    plt.show()


def plot_cv_comparison(cv_results):
    names = list(cv_results.keys())
    means = [cv_results[n]["mean_r2"] for n in names]
    stds = [cv_results[n]["std_r2"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5)
    ax.set_ylabel("Cross-Validated R2")
    ax.set_title("Model Comparison (5-Fold CV)")
    ax.set_xticklabels(names, rotation=30, ha="right")
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9
        )
    plt.tight_layout()
    plt.savefig("model_comparison_cv.png", dpi=150)
    plt.show()