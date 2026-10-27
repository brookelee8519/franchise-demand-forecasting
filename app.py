"""
Streamlit dashboard for the franchise demand forecasting + reorder tool.

Run with:
    streamlit run app.py

Expects the pipeline to have already been run once:
    python3 src/generate_data.py
    python3 src/forecast_model.py
    python3 src/inventory_calc.py
"""

import pandas as pd
import streamlit as st

from src.inventory_calc import flag_status

st.set_page_config(page_title="Franchise Demand & Reorder Dashboard", layout="wide")


@st.cache_data
def load_data():
    results = pd.read_csv("data/forecast_results.csv", parse_dates=["date"])
    rec = pd.read_csv("data/inventory_recommendations.csv")
    return results, rec


results, rec = load_data()

st.title("Franchise Demand & Reorder Dashboard")
st.caption(
    "Demand forecasts and reorder recommendations across franchise store "
    "locations, built on simulated SKU-level sales data."
)

tab1, tab2 = st.tabs(["Store / SKU detail", "Portfolio overview"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Store", sorted(results["store"].unique()))
    with col2:
        sku = st.selectbox(
            "SKU",
            sorted(results[results["store"] == store]["sku_id"].unique()),
        )

    series = results[(results["store"] == store) & (results["sku_id"] == sku)].sort_values(
        "date"
    )
    rec_row = rec[(rec["store"] == store) & (rec["sku_id"] == sku)]

    st.subheader(f"{store} - {sku}")
    st.line_chart(series.set_index("date")[["units_sold", "forecast"]])

    if not rec_row.empty:
        r = rec_row.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg daily forecast", f"{r['avg_daily_forecast']:.1f} units")
        m2.metric("Safety stock", f"{r['safety_stock']:.1f} units")
        m3.metric("Reorder point", f"{r['reorder_point']:.1f} units")

        on_hand = st.number_input(
            "Current stock on hand (enter what the store reports)",
            min_value=0.0,
            value=float(r["reorder_point"]),
            step=1.0,
        )
        status = flag_status(on_hand, r["reorder_point"])
        st.info(f"Status: **{status}**")

with tab2:
    st.subheader("Portfolio-wide reorder status")
    st.caption(
        "Assumes current stock equals the reorder point for every item, "
        "since this demo has no live inventory feed - swap in real on-hand "
        "quantities to make this view actionable."
    )
    rec_display = rec.copy()
    rec_display["status"] = rec_display["reorder_point"].apply(
        lambda rp: flag_status(rp, rp)
    )
    st.dataframe(
        rec_display[
            [
                "store",
                "sku_id",
                "avg_daily_forecast",
                "safety_stock",
                "reorder_point",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Highest safety-stock items")
    st.caption("These are the items with the most volatile/uncertain demand.")
    top = rec.sort_values("safety_stock", ascending=False).head(10).copy()
    top["item"] = top["store"] + " / " + top["sku_id"]
    st.bar_chart(top.set_index("item")["safety_stock"])
