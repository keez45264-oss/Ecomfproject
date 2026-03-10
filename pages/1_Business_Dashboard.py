import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Business Intelligence Dashboard")

# Load Data
customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")

data = orders.merge(customers, on="CustomerID")
data = data.merge(products, on="ProductID")

data["Revenue"] = data["Quantity"] * data["Price"]
data["OrderDate"] = pd.to_datetime(data["OrderDate"])

# --- Executive Summary ---
st.subheader("📰 Executive Summary")
total_revenue = data["Revenue"].sum()
total_orders = data["OrderID"].nunique()
total_customers = data["CustomerID"].nunique()
avg_order_value = total_revenue / total_orders
rev_per_customer = total_revenue / total_customers

st.markdown(f"""
- **Total Revenue**: ₹ {total_revenue:,.2f}  
- **Total Orders**: {total_orders}  
- **Total Customers**: {total_customers}  
- **Avg Order Value**: ₹ {avg_order_value:,.2f}  
- **Revenue per Customer**: ₹ {rev_per_customer:,.2f}  
""")

# --- KPI Cards ---
st.subheader("📌 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"₹ {total_revenue:,.2f}")
col2.metric("Total Orders", total_orders)
col3.metric("Total Customers", total_customers)

# --- Monthly Sales Trend ---
st.subheader("📈 Monthly Sales Trend")
monthly = data.groupby(data["OrderDate"].dt.to_period("M"))["Revenue"].sum()
monthly.index = monthly.index.to_timestamp()

fig, ax = plt.subplots()
monthly.plot(ax=ax, color="blue", label="Monthly Revenue")
monthly.rolling(3).mean().plot(ax=ax, color="orange", linestyle="--", label="3-Month Avg")
plt.ylabel("Revenue")
plt.title("Monthly Sales Trend")
plt.legend()
st.pyplot(fig)

# --- Top Products & Customers ---
st.subheader("🏆 Top Products by Revenue")
top_products = data.groupby("ProductName")["Revenue"].sum().sort_values(ascending=False).head(5)
st.table(top_products)

st.subheader("🏆 Top Customers by Revenue")
top_customers = data.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False).head(5)
st.table(top_customers)
