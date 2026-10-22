"""
Trains a demand forecasting model across all store/SKU combinations.

Uses scikit-learn's HistGradientBoostingRegressor (a native-categorical
gradient boosting model, similar in spirit to XGBoost/LightGBM) so the
pipeline runs without extra native dependencies. If xgboost is installed in
your environment, set USE_XGBOOST = True below to swap it in; the feature
table and evaluation code don't change.
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.append(".")
from src.features import FEATURE_COLS, TARGET_COL, build_feature_table

USE_XGBOOST = False

TRAIN_END = "2025-06-30"  # everything after this date is the test window


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error - the standard accuracy metric
    used in supply chain forecasting (more stable than MAPE when actual
    demand can be zero on slow days)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom


def load_and_prepare(path="data/daily_demand.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = build_feature_table(df)
    # Early rows for each store/SKU don't have 28-day lag history yet; drop them.
    df = df.dropna(subset=FEATURE_COLS)
    return df


def split(df):
    train = df[df["date"] <= TRAIN_END]
    test = df[df["date"] > TRAIN_END]
    return train, test


def train_model(train_df):
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    if USE_XGBOOST:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=400,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
        )

    model.fit(X_train, y_train)
    return model


def evaluate(model, test_df):
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    results = test_df.copy()
    results["forecast"] = preds
    results["residual"] = results[TARGET_COL] - results["forecast"]

    overall_wape = wape(y_test, preds)
    overall_mae = np.mean(np.abs(y_test - preds))

    return results, {"wape": overall_wape, "mae": overall_mae}


def per_series_residual_std(results: pd.DataFrame) -> pd.DataFrame:
    """Residual std per store/SKU on the test window - this feeds directly
    into the safety stock calculation in inventory_calc.py."""
    return (
        results.groupby(["store", "sku_id"])["residual"]
        .std()
        .reset_index()
        .rename(columns={"residual": "residual_std"})
    )


def main():
    print("Loading data and building features...")
    df = load_and_prepare()
    train_df, test_df = split(df)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    print("Training model...")
    model = train_model(train_df)

    print("Evaluating on holdout window...")
    results, metrics = evaluate(model, test_df)
    print(f"WAPE: {metrics['wape']:.3f} | MAE: {metrics['mae']:.3f} units/day")

    joblib.dump(model, "data/model.joblib")
    results.to_csv("data/forecast_results.csv", index=False)

    residual_std = per_series_residual_std(results)
    residual_std.to_csv("data/residual_std.csv", index=False)

    print("Saved model to data/model.joblib")
    print("Saved forecast results to data/forecast_results.csv")
    print("Saved per-series residual std to data/residual_std.csv")


if __name__ == "__main__":
    main()
