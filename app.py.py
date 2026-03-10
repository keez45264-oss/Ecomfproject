import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

st.title("🛒 E-Commerce Sales Forecasting & Customer Insights")

st.write("Upload your CSV file to begin analysis.")

# Upload CSV
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Convert OrderDate
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])

    # Create Revenue column (if not present)
    if 'Revenue' not in df.columns:
        df['Revenue'] = df['TotalAmount']

    # -----------------------------
    # Monthly Sales Analysis
    # -----------------------------
    df['YearMonth'] = df['OrderDate'].dt.to_period('M')
    monthly_sales = df.groupby('YearMonth')['Revenue'].sum().reset_index()
    monthly_sales['YearMonth'] = monthly_sales['YearMonth'].dt.to_timestamp()

    st.subheader("📈 Monthly Revenue Trend")

    fig, ax = plt.subplots()
    ax.plot(monthly_sales['YearMonth'], monthly_sales['Revenue'])
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.set_title("Monthly Revenue")
    st.pyplot(fig)

    # -----------------------------
    # Forecasting using ARIMA
    # -----------------------------
    st.subheader("🔮 Sales Forecast (Next 6 Months)")

    monthly_sales.set_index('YearMonth', inplace=True)

    model = ARIMA(monthly_sales['Revenue'], order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)

    st.write(forecast)

    # -----------------------------
    # RFM Customer Segmentation
    # -----------------------------
    st.subheader("👥 Customer Segmentation (RFM + KMeans)")

    today = df['OrderDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'OrderDate': lambda x: (today - x.max()).days,
        'OrderID': 'count',
        'Revenue': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.write("Cluster Distribution")
    st.write(rfm['Cluster'].value_counts())

    st.write("Cluster Summary")
    st.write(rfm.groupby('Cluster').mean())

else:
    st.info("Please upload a CSV file to continue.")