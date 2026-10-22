"""
Generates a synthetic daily demand dataset for a franchise food/grocery
distributor supplying multiple store locations.

This is NOT scraped or copied from a real company. It's built to match the
statistical shape of real retail demand data (trend, weekly seasonality,
holiday spikes, promotions, and per-store/per-SKU variation) the way public
datasets like the Kaggle "Store Item Demand Forecasting" challenge or the
M5/Walmart dataset are structured, so the modeling and inventory logic that
follows is realistic even though the raw numbers are simulated.

Run this once to produce data/daily_demand.csv.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_STORES = 8
N_SKUS = 12
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

STORE_NAMES = [f"Store_{i+1:02d}" for i in range(N_STORES)]

SKU_CATALOG = [
    ("SKU_001", "Frozen Dumplings 1lb", 4.0),
    ("SKU_002", "Frozen Dumplings 2lb", 6.5),
    ("SKU_003", "Kimchi 16oz", 2.5),
    ("SKU_004", "Kimchi 32oz", 4.0),
    ("SKU_005", "Instant Ramen 5-pack", 3.0),
    ("SKU_006", "Instant Ramen 10-pack", 5.5),
    ("SKU_007", "Frozen Rice Cakes 1lb", 3.5),
    ("SKU_008", "Soy Sauce 16oz", 2.0),
    ("SKU_009", "Gochujang 16oz", 3.0),
    ("SKU_010", "Frozen Dumplings Veggie 1lb", 4.5),
    ("SKU_011", "Seaweed Snacks 12-pack", 5.0),
    ("SKU_012", "Frozen Bulgogi 1lb", 7.0),
]

# Each store gets its own baseline traffic level, so demand isn't identical
# across locations.
STORE_BASELINE = RNG.normal(loc=1.0, scale=0.22, size=N_STORES)
STORE_BASELINE = np.clip(STORE_BASELINE, 0.55, 1.6)

# Each SKU gets its own popularity and volatility.
SKU_BASE_DEMAND = RNG.uniform(6, 40, size=N_SKUS)
SKU_VOLATILITY = RNG.uniform(0.15, 0.45, size=N_SKUS)

# A handful of "promotion weeks" per SKU per year create demand spikes,
# the kind a distributor has to plan extra stock around.
def promo_calendar(dates, n_promos_per_year=3):
    promo_days = set()
    for year in dates.year.unique():
        year_dates = dates[dates.year == year]
        if len(year_dates) == 0:
            continue
        starts = RNG.choice(year_dates, size=n_promos_per_year, replace=False)
        for s in starts:
            promo_days.update(pd.date_range(s, periods=5, freq="D"))
    return promo_days


def build():
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    rows = []
    for store_idx, store in enumerate(STORE_NAMES):
        store_mult = STORE_BASELINE[store_idx]
        for sku_idx, (sku_id, sku_name, unit_cost) in enumerate(SKU_CATALOG):
            base = SKU_BASE_DEMAND[sku_idx] * store_mult
            vol = SKU_VOLATILITY[sku_idx]
            promo_days = promo_calendar(dates)

            # Mild upward trend over the 3 years (the business is growing).
            trend = np.linspace(0, 0.18, len(dates))

            # Weekly seasonality: weekends higher for most SKUs.
            dow = dates.dayofweek.values
            weekly = np.where(dow >= 5, 0.22, 0.0)

            # Winter holiday bump in Nov/Dec for most SKUs.
            month = dates.month.values
            holiday = np.where(np.isin(month, [11, 12]), 0.28, 0.0)

            promo_flag = np.array([d in promo_days for d in dates])
            promo_bump = np.where(promo_flag, 0.55, 0.0)

            noise = RNG.normal(0, vol, size=len(dates))

            demand = base * (1 + trend + weekly + holiday + promo_bump + noise)
            demand = np.clip(demand, 0, None)
            demand = RNG.poisson(demand)  # counts, not fractional units

            for d, q, promo in zip(dates, demand, promo_flag):
                rows.append(
                    {
                        "date": d,
                        "store": store,
                        "sku_id": sku_id,
                        "sku_name": sku_name,
                        "unit_cost": unit_cost,
                        "units_sold": int(q),
                        "on_promo": bool(promo),
                    }
                )

    df = pd.DataFrame(rows)
    df = df.sort_values(["store", "sku_id", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build()
    out_path = "data/daily_demand.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(df.head())
