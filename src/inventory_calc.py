"""
Turns a demand forecast into an actual inventory decision: safety stock and
reorder point per store/SKU. This is the piece that makes the project an
SCM tool rather than just a forecasting model.

Formulas used (standard in inventory management):
    safety_stock   = z * sigma_demand * sqrt(lead_time_days)
    reorder_point  = (avg_daily_demand * lead_time_days) + safety_stock

sigma_demand is estimated from the model's forecast residuals on the holdout
window (data/residual_std.csv), rather than from raw demand variance, so it
reflects how uncertain the FORECAST is, not just how variable sales are.
"""

import pandas as pd
from scipy.stats import norm

LEAD_TIME_DAYS = 4  # distributor restock lead time, in days
SERVICE_LEVEL = 0.95  # probability of not stocking out during lead time


def z_for_service_level(service_level: float) -> float:
    return norm.ppf(service_level)


def compute_recommendations(
    forecast_results: pd.DataFrame,
    residual_std: pd.DataFrame,
    lead_time_days: int = LEAD_TIME_DAYS,
    service_level: float = SERVICE_LEVEL,
) -> pd.DataFrame:
    """
    forecast_results: output of forecast_model.evaluate(), must have
        store, sku_id, date, forecast, units_sold columns
    residual_std: output of forecast_model.per_series_residual_std()
    """
    z = z_for_service_level(service_level)

    latest = (
        forecast_results.sort_values("date")
        .groupby(["store", "sku_id"])
        .tail(7)  # last week of the test window as "current" conditions
        .groupby(["store", "sku_id"])
        .agg(avg_daily_forecast=("forecast", "mean"), last_date=("date", "max"))
        .reset_index()
    )

    rec = latest.merge(residual_std, on=["store", "sku_id"], how="left")
    rec["residual_std"] = rec["residual_std"].fillna(rec["residual_std"].median())

    rec["safety_stock"] = z * rec["residual_std"] * (lead_time_days**0.5)
    rec["reorder_point"] = (
        rec["avg_daily_forecast"] * lead_time_days
    ) + rec["safety_stock"]

    rec["safety_stock"] = rec["safety_stock"].round(1)
    rec["reorder_point"] = rec["reorder_point"].round(1)
    rec["avg_daily_forecast"] = rec["avg_daily_forecast"].round(1)

    return rec


def flag_status(current_on_hand: float, reorder_point: float) -> str:
    """Simple status label used by the dashboard. current_on_hand is
    whatever the store/warehouse reports as current stock."""
    if current_on_hand <= reorder_point * 0.5:
        return "Critical - reorder now"
    elif current_on_hand <= reorder_point:
        return "Reorder soon"
    elif current_on_hand > reorder_point * 3:
        return "Possible overstock"
    else:
        return "Sufficient stock"


if __name__ == "__main__":
    results = pd.read_csv("data/forecast_results.csv", parse_dates=["date"])
    residual_std = pd.read_csv("data/residual_std.csv")

    rec = compute_recommendations(results, residual_std)
    rec.to_csv("data/inventory_recommendations.csv", index=False)
    print(f"Wrote {len(rec)} store/SKU reorder recommendations")
    print(rec.head(10))
