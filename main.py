import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import load_revenue_dataset, load_full_dataset
from preprocessing import merge_and_label, get_train_data, get_prediction_data
from models import get_models, train_all
from evaluate import (
    compute_metrics,
    run_cross_validation,
    print_comparison_table,
    select_best_model,
)
from tuning import tune_all
from analysis import run_full_analysis
from visualize import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_revenue_distribution,
    plot_top_predicted_games,
    plot_cv_comparison,
)


def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    df_rev = load_revenue_dataset()
    df_full = load_full_dataset()

    # ── Preprocess & merge ───────────────────────────────────────────────────
    print("\nMerging and engineering features...")
    df_labeled = merge_and_label(df_full, df_rev)
    X, y, df_train = get_train_data(df_labeled)

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── Baseline cross-validation ─────────────────────────────────────────────
    models = get_models()
    cv_results = run_cross_validation(models, X, y, cv=5)

    # ── Train baselines on training split ────────────────────────────────────
    print("\nTraining all baseline models...")
    trained = train_all(models, X_train, y_train)

    print("\nBaseline test-set results:")
    baseline_results = {}
    for name, model in trained.items():
        baseline_results[name] = compute_metrics(model, X_test, y_test)
    print_comparison_table(baseline_results)

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    tuned_info = tune_all(X_train, y_train, n_iter=30)

    print("\nTuned model test-set results:")
    tuned_results = {}
    for name, info in tuned_info.items():
        tuned_results[name] = compute_metrics(info["model"], X_test, y_test)
    print_comparison_table(tuned_results)

    # ── Select best overall model ─────────────────────────────────────────────
    all_tuned_models = {name: info["model"] for name, info in tuned_info.items()}
    all_tuned_cv = {name: {"mean_r2": info["cv_r2"], "std_r2": 0.0} for name, info in tuned_info.items()}
    best_name, best_model = select_best_model(all_tuned_models, all_tuned_cv)

    # ── Deep error analysis ───────────────────────────────────────────────────
    best_preds = tuned_results[best_name]["predictions"]
    df_test_rows = df_train.iloc[X_test.index] if hasattr(X_test, "index") else df_train.iloc[:len(y_test)]

    # Rebuild df_test with proper index alignment
    df_test_aligned = df_train.loc[X_test.index]
    print("\nRunning error analysis...")
    run_full_analysis(df_test_aligned, y_test, best_preds)

    # ── Predict on unlabeled games ────────────────────────────────────────────
    labeled_ids = set(df_labeled["appid"].dropna().astype(int))
    df_pred = get_prediction_data(df_full, labeled_ids)
    preds_log = best_model.predict(df_pred[list(X.columns)])
    df_pred["predicted_revenue"] = np.clip(np.expm1(preds_log), 0, None)

    output_file = "predicted_steam_revenues.csv"
    df_pred[["appid", "name", "predicted_revenue"]].to_csv(output_file, index=False)
    print(f"\nSaved {len(df_pred)} predictions to {output_file}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_correlation_heatmap(df_train)
    plot_feature_importance(trained["Random Forest"], "Random Forest (Baseline)")

    if hasattr(tuned_info["Random Forest (Tuned)"]["model"], "feature_importances_"):
        plot_feature_importance(
            tuned_info["Random Forest (Tuned)"]["model"],
            "Random Forest (Tuned)"
        )

    plot_predicted_vs_actual(y_test, best_preds, best_name)
    plot_residuals(y_test, best_preds, best_name)
    plot_revenue_distribution(df_train, df_pred)
    plot_top_predicted_games(df_pred)
    plot_cv_comparison(cv_results)

    print("\nDone.")


if __name__ == "__main__":
    main()