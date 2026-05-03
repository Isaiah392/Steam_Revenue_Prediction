import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def compute_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "predictions": preds}


def run_cross_validation(models, X, y, cv=5):
    print(f"\nRunning {cv}-fold cross-validation on all models...")
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        cv_results[name] = {
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
            "scores": scores
        }
        print(f"  {name}: R2 = {scores.mean():.4f} (+/- {scores.std():.4f})")
    return cv_results


def print_comparison_table(results):
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("=" * 55)
    for name, m in results.items():
        print(f"{name:<25} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['R2']:>8.4f}")
    print("=" * 55)


def select_best_model(trained_models, cv_results):
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_r2"])
    print(f"\nBest model by cross-validation R2: {best_name}")
    return best_name, trained_models[best_name]