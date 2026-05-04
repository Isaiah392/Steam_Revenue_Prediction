# Steam Game Revenue Prediction

**CSCI 4420/5420 — Applied Machine Learning**
Team: Isaiah Malcolm, Luis Escamilla

## Overview

This project predicts the revenue of Steam games using publicly available metadata. We train ensemble regression models on ~1,500 games with known revenue figures and use those models to estimate revenue for the broader Steam catalog (~122,000 games).

## Project Structure

```
steam_revenue/
├── data/                        # CSV datasets (not tracked by git — see below)
│   ├── steam_games.csv
│   └── steam_revenue.csv
├── main.py                      # Entry point — runs the full pipeline
├── data_loader.py               # Loads and validates the CSV files
├── preprocessing.py             # Feature engineering and dataset merging
├── models.py                    # Baseline model definitions
├── tuning.py                    # RandomizedSearch hyperparameter tuning
├── evaluate.py                  # Cross-validation, metrics, model selection
├── analysis.py                  # Error analysis and residual breakdowns
├── visualize.py                 # All plots
├── predicted_steam_revenues.csv # Output — generated after running main.py
├── README.md
└── .gitignore
```

## Dataset Setup

The datasets are not included in this repo due to file size. Download them from Kaggle and place them in the `data/` folder:

| File | Source |
|------|--------|
| `steam_games.csv` | https://www.kaggle.com/datasets/fronkongames/steam-games-dataset |
| `steam_revenue.csv` | https://www.kaggle.com/datasets/alicemtopcu/top-1500-games-on-steam-by-revenue-09-09-2024 |

After downloading, unzip if needed and rename the files exactly as shown above.

## Installation

Python 3.8 or higher is required.

```bash
pip install scikit-learn xgboost pandas numpy matplotlib
```

## Running the Code

```bash
cd steam_revenue
python main.py
```

The pipeline will:
1. Load and validate both datasets
2. Merge them using Steam App ID as the join key
3. Engineer features (log transforms, review score ratio, etc.)
4. Run 5-fold cross-validation on all baseline models
5. Tune Random Forest, XGBoost, and Gradient Boosting with RandomizedSearch
6. Evaluate all models on a held-out test set and print a comparison table
7. Run error analysis broken down by price tier and revenue quartile
8. Generate all plots and save them as PNG files
9. Save revenue predictions for unlabeled games to `predicted_steam_revenues.csv`

## Features Used

| Feature | Description |
|---------|-------------|
| `price` | Game price in USD |
| `log_positive` | Log of positive review count |
| `log_negative` | Log of negative review count |
| `log_peak_ccu` | Log of peak concurrent users |
| `review_score` | Positive / (Positive + Negative) ratio |
| `release_year` | Year the game was released |

## Models

Baselines: Linear Regression, Ridge, Lasso
Ensemble: Random Forest, Gradient Boosting, XGBoost (all tuned via RandomizedSearch)

## Output Files

After running `main.py`, the following files are created in the project root:

- `predicted_steam_revenues.csv` — revenue predictions for unlabeled Steam games
- `correlation_heatmap.png`
- `feature_importance.png`
- `predicted_vs_actual.png`
- `residuals.png`
- `revenue_distribution.png`
- `top_predicted_games.png`
- `model_comparison_cv.png`
- `error_by_price_tier.png`
- `error_by_revenue_quartile.png`