import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("👥 Customer Insights Dashboard")

st.markdown("""
This section analyzes customer purchasing behavior.
It helps businesses identify:
            
• Loyal customers  
• High-value customers  
• Customers who may stop purchasing  
""")

# Load Data
customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")

# Merge data
data = orders.merge(products, on="ProductID")
data["Revenue"] = data["Quantity"] * data["Price"]
data["OrderDate"] = pd.to_datetime(data["OrderDate"])

# --- Date Range Filter ---
st.subheader("📅 Filter by Date Range")
min_date = data["OrderDate"].min()
max_date = data["OrderDate"].max()
start_date, end_date = st.date_input(
    "Select date range:",
    [min_date, max_date]
)

# Apply filter
data = data[(data["OrderDate"] >= pd.to_datetime(start_date)) & 
            (data["OrderDate"] <= pd.to_datetime(end_date))]

# Create RFM Table
snapshot_date = data["OrderDate"].max() + pd.Timedelta(days=1)
rfm = data.groupby("CustomerID").agg({
    "OrderDate": lambda x: (snapshot_date - x.max()).days,
    "OrderID": "count",
    "Revenue": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]

# Scale data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Map clusters to labels
cluster_labels = {
    0: "Loyal Customers",
    1: "At-Risk Customers",
    2: "High-Value Customers",
    3: "New Customers"
}
rfm["Segment"] = rfm["Cluster"].map(cluster_labels)

# --- Summary Stats Panel ---
st.subheader("📌 Summary Statistics")
total_customers = rfm.shape[0]
total_revenue = rfm["Monetary"].sum()
avg_frequency = rfm["Frequency"].mean()
segment_counts = rfm["Segment"].value_counts()

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Total Revenue", f"{total_revenue:,.0f}")
col3.metric("Avg Purchase Frequency", f"{avg_frequency:.2f}")

st.write("### Customers per Segment")
st.bar_chart(segment_counts)

# Visualization
st.subheader("📊 Customer Segments")

fig, ax = plt.subplots()
scatter = ax.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Cluster"], cmap="viridis")
legend1 = ax.legend(*scatter.legend_elements(), title="Segments")
ax.add_artist(legend1)
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spending")
plt.title("Customer Segmentation")
st.pyplot(fig)

# Interactive filter
st.subheader("🔍 Explore Customers by Segment")
segment_choice = st.selectbox("Select a segment:", rfm["Segment"].unique())
filtered_customers = rfm[rfm["Segment"] == segment_choice]
st.dataframe(filtered_customers.sort_values("Monetary", ascending=False).head(20))

# --- Download filtered customers ---
csv = filtered_customers.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered Customers as CSV",
    data=csv,
    file_name=f"{segment_choice}_customers.csv",
    mime="text/csv"
)

# Top customers
st.subheader("🏆 Top Customers")
top_customers = rfm.sort_values("Monetary", ascending=False).head(10)
top_customers["AvgOrderValue"] = top_customers["Monetary"] / top_customers["Frequency"]
st.dataframe(top_customers[["Recency", "Frequency", "Monetary", "AvgOrderValue", "Segment"]])

# At risk customers
st.subheader("⚠️ At-Risk Customers")
at_risk = rfm[rfm["Segment"] == "At-Risk Customers"].sort_values("Recency", ascending=False).head(10)
at_risk["RecommendedAction"] = "Send re-engagement offer"
st.dataframe(at_risk[["Recency", "Frequency", "Monetary", "Segment", "RecommendedAction"]])

# Business Recommendations
st.subheader("💡 Business Recommendations")
st.markdown("""
- **Retention**: Offer discounts or loyalty points to at-risk customers.  
- **Upsell**: Target top customers with premium products.  
- **Acquisition**: Identify patterns from loyal customers to attract similar profiles.  
- **Forecasting Link**: Loyal customers provide stable revenue, at-risk customers may cause drops, new customers drive growth.  
""")
