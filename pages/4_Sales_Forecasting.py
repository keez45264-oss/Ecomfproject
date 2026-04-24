import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("📈 Future Sales Prediction")

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
        
        # --- FIX: assign proper future date index ---
        last_date = monthly.index[-1]
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1),
                                     periods=forecast_horizon, freq='MS')
        forecast.index = future_dates
        
        combined = pd.concat([monthly, forecast])
        comparison_df[cat] = combined
        
        # KPIs
        total_rev = monthly.sum()
        avg_rev = monthly.mean()
        best_month = monthly.idxmax().strftime("%B %Y")
        growth = ((forecast.iloc[-1] - train.iloc[-1]) / train.iloc[-1]) * 100
        
        kpi_data.append({
            "Category": cat,
            "Total Revenue": total_rev,
            "Avg Monthly": avg_rev,
            "Best Month": best_month,
            "Predicted Growth": growth
        })

# --- Comparison Chart ---
st.subheader("📊 Multi-Category Sales Comparison")
if not comparison_df.empty:
    fig, ax = plt.subplots()
    for cat in comparison_df.columns:
        monthly = comparison_df[cat].iloc[:-forecast_horizon]
        forecast = comparison_df[cat].iloc[-forecast_horizon:]
        monthly.plot(ax=ax, label=f"{cat} Actual")
        forecast.plot(ax=ax, label=f"{cat} Forecast", linestyle="--")
    plt.legend()
    plt.ylabel("Revenue")
    plt.title(f"Sales Forecast Comparison ({forecast_horizon}-Month Horizon)")
    st.pyplot(fig)
else:
    st.warning("Not enough data to generate forecasts for selected categories.")

# --- KPI Cards per Category ---
st.subheader("📌 Category Performance KPIs")
for kpi in kpi_data:
    st.markdown(f"### {kpi['Category']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"{kpi['Total Revenue']:,.0f}")
    col2.metric("Avg Monthly", f"{kpi['Avg Monthly']:,.0f}")
    col3.metric("Best Month", kpi["Best Month"])
    col4.metric("Predicted Growth", f"{kpi['Predicted Growth']:.2f}%")

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
    heatmap_data = kpi_df.set_index("Category")[["Predicted Growth"]]
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

# --- Business Insight ---
st.subheader("💡 Business Insight")
st.markdown(f"""
- **Category Prioritization**: Compare growth trends across categories to decide where to allocate inventory and marketing spend.  
- **Risk Management**: Identify categories with declining forecasts and plan promotions.  
- **Strategic Planning**: Focus on categories with strong growth potential for expansion.  
- **Forecast Horizon Advantage**: Adjust between short-term ({forecast_horizon} months) and long-term planning to align with business cycles.  
- **Seasonality Detection**: Peaks in certain months indicate recurring demand cycles — plan campaigns and stock accordingly.  
- **Decomposition Insight**: Trend shows long-term growth, seasonality highlights recurring cycles, residuals capture unexpected fluctuations.  
""")
