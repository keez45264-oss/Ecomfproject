import streamlit as st

st.set_page_config(page_title="E-Commerce Sales Forecasting", layout="wide")

# Title & Subtitle
st.title("🛒 E-Commerce Sales Forecasting")
st.markdown("### Forecasting • Segmentation • Demand Prediction")

# Project Modules
st.markdown("""
### 📌 Project Modules
1. 📊 Business Dashboard – Key metrics & trends  
2. 👥 RFM Segmentation – Customer value analysis  
3. 🤖 Customer Insights – Customer grouping  
4. 📈 Sales Forecasting – Time series predictions  
5. 📦 Demand Prediction – Machine learning models  
""")

# Navigation Note
st.info("Use the sidebar to navigate between modules.")
