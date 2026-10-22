"""Generates a static PNG preview of the dashboard's core charts, so the
results can be sanity-checked without running Streamlit."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

results = pd.read_csv("data/forecast_results.csv", parse_dates=["date"])
rec = pd.read_csv("data/inventory_recommendations.csv")

store, sku = "Store_01", "SKU_004"
series = results[(results.store == store) & (results.sku_id == sku)].sort_values("date")

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(series["date"], series["units_sold"], label="Actual", alpha=0.6)
axes[0].plot(series["date"], series["forecast"], label="Forecast", linewidth=2)
axes[0].set_title(f"{store} / {sku}: actual vs. forecast demand (holdout window)")
axes[0].legend()
axes[0].set_ylabel("Units sold")

top = rec.sort_values("safety_stock", ascending=False).head(10)
axes[1].barh(top["store"] + " / " + top["sku_id"], top["safety_stock"])
axes[1].set_title("Top 10 store/SKU combinations by safety stock (most volatile demand)")
axes[1].invert_yaxis()
axes[1].set_xlabel("Safety stock (units)")

plt.tight_layout()
plt.savefig("data/preview.png", dpi=130)
print("Saved data/preview.png")
