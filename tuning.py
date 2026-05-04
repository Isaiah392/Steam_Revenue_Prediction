import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [5, 8, 10, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.8],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 8],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2],
}

GB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.7, 0.85, 1.0],
    "min_samples_leaf": [1, 2, 4],
}


def tune_model(name, model, param_grid, X_train, y_train, n_iter=30, cv=5):
    print(f"  Tuning {name} ({n_iter} iterations, {cv}-fold CV)...")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    Best R2: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_all(X_train, y_train, n_iter=30):
    print("\nStarting hyperparameter tuning (this will take several minutes)...")

    tuned = {}

    rf_model, rf_params, rf_score = tune_model(
        "Random Forest",
        RandomForestRegressor(random_state=42),
        RF_PARAM_GRID,
        X_train, y_train,
        n_iter=n_iter,
    )
    tuned["Random Forest (Tuned)"] = {"model": rf_model, "params": rf_params, "cv_r2": rf_score}

    xgb_model, xgb_params, xgb_score = tune_model(
        "XGBoost",
        XGBRegressor(random_state=42, verbosity=0),
        XGB_PARAM_GRID,
        X_train, y_train,
        n_iter=n_iter,
    )
    tuned["XGBoost (Tuned)"] = {"model": xgb_model, "params": xgb_params, "cv_r2": xgb_score}

    gb_model, gb_params, gb_score = tune_model(
        "Gradient Boosting",
        GradientBoostingRegressor(random_state=42),
        GB_PARAM_GRID,
        X_train, y_train,
        n_iter=n_iter,
    )
    tuned["Gradient Boosting (Tuned)"] = {"model": gb_model, "params": gb_params, "cv_r2": gb_score}

    return tuned