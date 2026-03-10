import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("📈 Future Sales Prediction")

# --- Executive Summary Panel ---
st.subheader("📰 Executive Summary")
summary_placeholder = st.empty()  # will fill after KPI calculation

st.markdown("""
This module analyzes past sales data and predicts future sales trends.

Businesses can use this to:
• Plan inventory  
• Set revenue targets  
• Adjust marketing strategies  
""")

# Load Data
products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")

data = orders.merge(products, on="ProductID")
data["Revenue"] = data["Quantity"] * data["Price"]
data["OrderDate"] = pd.to_datetime(data["OrderDate"])

# --- Multi-category selection ---
st.subheader("🎯 Compare Multiple Categories")
categories = products["Category"].unique()
selected_categories = st.multiselect("Select categories to compare:", categories, default=categories[:2])

# --- Forecast Horizon Slider ---
st.subheader("⏳ Forecast Horizon")
forecast_horizon = st.slider("Select forecast horizon (months):", min_value=3, max_value=12, value=6, step=1)

comparison_df = pd.DataFrame()
kpi_data = []

for cat in selected_categories:
    cat_data = data[data["Category"] == cat]
    monthly = cat_data.groupby(cat_data["OrderDate"].dt.to_period("M"))["Revenue"].sum()
    monthly.index = monthly.index.to_timestamp()
    
    if len(monthly) > forecast_horizon + 3:  # ensure enough data points
        train = monthly[:-forecast_horizon]
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        
        combined = pd.concat([monthly, forecast])
        comparison_df[cat] = combined
        
        # KPIs
        total_rev = monthly.sum()
        avg_rev = monthly.mean()
        best_month = monthly.idxmax().strftime("%B %Y")
        growth = ((forecast.iloc[-1] - train.iloc[-1]) / train.iloc[-1]) * 100
        
        kpi_data.append({
            "Category": cat,
            "Total Revenue": f"{total_rev:,.0f}",
            "Avg Monthly": f"{avg_rev:,.0f}",
            "Best Month": best_month,
            "Predicted Growth": f"{growth:.2f}%"
        })

# --- Executive Summary Text ---
if kpi_data:
    summary_lines = []
    for kpi in kpi_data:
        growth_val = float(kpi["Predicted Growth"].replace("%",""))
        if growth_val > 0:
            summary_lines.append(f"**{kpi['Category']}** is projected to grow (+{kpi['Predicted Growth']}).")
        elif growth_val < 0:
            summary_lines.append(f"**{kpi['Category']}** is projected to decline ({kpi['Predicted Growth']}).")
        else:
            summary_lines.append(f"**{kpi['Category']}** is flat ({kpi['Predicted Growth']}).")
    summary_placeholder.markdown(" • ".join(summary_lines))

# --- Comparison Chart ---
st.subheader("📊 Multi-Category Sales Comparison")
if not comparison_df.empty:
    fig, ax = plt.subplots()
    for cat in comparison_df.columns:
        comparison_df[cat].plot(ax=ax, label=cat)
    plt.legend()
    plt.ylabel("Revenue")
    plt.title(f"Sales Forecast Comparison ({forecast_horizon}-Month Horizon)")
    st.pyplot(fig)
else:
    st.warning("Not enough data to generate forecasts for selected categories.")

# --- KPI Cards per Category ---
st.subheader("📌 Category Performance KPIs")
if kpi_data:
    for kpi in kpi_data:
        st.markdown(f"### {kpi['Category']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", kpi["Total Revenue"])
        col2.metric("Avg Monthly", kpi["Avg Monthly"])
        col3.metric("Best Month", kpi["Best Month"])
        col4.metric("Predicted Growth", kpi["Predicted Growth"])

# --- Download KPI Summary ---
if kpi_data:
    kpi_df = pd.DataFrame(kpi_data)
    csv_kpi = kpi_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download KPI Summary as CSV",
        data=csv_kpi,
        file_name=f"category_kpi_summary_{forecast_horizon}m.csv",
        mime="text/csv"
    )

    # --- Heatmap Visualization ---
    st.subheader("🔥 Category Growth Heatmap")
    # Clean '%' before converting to float
    kpi_df["Predicted Growth (Numeric)"] = kpi_df["Predicted Growth"].str.replace('%','').astype(float)
    heatmap_data = kpi_df.set_index("Category")[["Predicted Growth (Numeric)"]]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", center=0, fmt=".2f", ax=ax)
    plt.title(f"Predicted Growth Rates by Category ({forecast_horizon}-Month Horizon)")
    st.pyplot(fig)

# --- Seasonality Detection ---
st.subheader("📅 Seasonality Detection")
monthly_all = data.groupby(data["OrderDate"].dt.month)["Revenue"].mean()
fig3, ax3 = plt.subplots()
monthly_all.plot(kind="bar", color="skyblue", ax=ax3)
plt.xlabel("Month")
plt.ylabel("Average Revenue")
plt.title("Average Monthly Revenue (Seasonality Pattern)")
st.pyplot(fig3)

# --- Decomposition Chart with Toggle + Summary ---
st.subheader("🔎 Trend, Seasonality & Residuals")
monthly_total = data.groupby(data["OrderDate"].dt.to_period("M"))["Revenue"].sum()
monthly_total.index = monthly_total.index.to_timestamp()

decomp_type = st.radio("Select decomposition model:", ["additive", "multiplicative"])

if len(monthly_total) > 24:  # need at least 2 years of data
    decomposition = seasonal_decompose(monthly_total, model=decomp_type, period=12)
    fig4 = decomposition.plot()
    fig4.set_size_inches(10, 8)
    st.pyplot(fig4)

    # --- Quick Summary Table ---
    st.subheader("📌 Decomposition Summary")
    trend_growth = decomposition.trend.dropna().mean()
    strongest_season = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean().idxmax()
    residual_var = decomposition.resid.dropna().var()

    summary_df = pd.DataFrame({
        "Metric": ["Avg Trend Growth", "Strongest Seasonal Month", "Residual Variance"],
        "Value": [f"{trend_growth:,.2f}", strongest_season, f"{residual_var:,.2f}"]
    })
    st.table(summary_df)

    # --- Download Decomposition Summary ---
    csv_decomp = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Decomposition Summary as CSV",
        data=csv_decomp,
        file_name="decomposition_summary.csv",
        mime="text/csv"
    )

else:
    st.warning("Not enough data points for decomposition (need at least 2 years). Showing moving average trend instead.")
    fig5, ax5 = plt.subplots()
    monthly_total.rolling(window=6).mean().plot(ax=ax5, color="green")
    plt.title("6-Month Moving Average Trend (Fallback)")
    plt.ylabel("Revenue")
    st.pyplot(fig5)

st.info("This decomposition chart separates long-term trend, recurring seasonality, and random residuals.")

# --- Business Insight ---
st.subheader("💡 Business Insight")
for kpi in kpi_data:
    growth_val = float(kpi["Predicted Growth"].replace("%",""))
    if growth_val > 0:
        st.success(f"{kpi['Category']} shows positive growth — increase inventory and marketing.")
    elif growth_val < 0:
        st.warning(f"{kpi['Category']} shows decline — run promotions and manage stock carefully.")
    else:
        st.info(f"{kpi['Category']} is flat — maintain current strategy but monitor closely.")
