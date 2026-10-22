"""Feature engineering for the daily store/SKU demand data."""

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    """Adds lagged demand features, computed per store/SKU series."""
    df = df.sort_values(["store", "sku_id", "date"]).copy()
    group = df.groupby(["store", "sku_id"])["units_sold"]
    for lag in lags:
        df[f"lag_{lag}"] = group.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows=(7, 28)) -> pd.DataFrame:
    """Rolling mean/std of demand, shifted by 1 day to avoid leaking the
    current day's actual sales into its own features."""
    df = df.sort_values(["store", "sku_id", "date"]).copy()
    shifted = df.groupby(["store", "sku_id"])["units_sold"].shift(1)
    for w in windows:
        df[f"roll_mean_{w}"] = (
            shifted.groupby([df["store"], df["sku_id"]])
            .transform(lambda s: s.rolling(w, min_periods=3).mean())
        )
        df[f"roll_std_{w}"] = (
            shifted.groupby([df["store"], df["sku_id"]])
            .transform(lambda s: s.rolling(w, min_periods=3).std())
        )
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # store/sku as categorical codes for the model
    df["store_code"] = df["store"].astype("category").cat.codes
    df["sku_code"] = df["sku_id"].astype("category").cat.codes

    return df


FEATURE_COLS = [
    "store_code",
    "sku_code",
    "day_of_week",
    "is_weekend",
    "month",
    "is_holiday_season",
    "on_promo",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_28",
    "roll_std_28",
]

TARGET_COL = "units_sold"
