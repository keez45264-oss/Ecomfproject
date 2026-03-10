import streamlit as st
import pandas as pd

st.title("👥 RFM Customer Segmentation")

products = pd.read_csv("products.csv")
orders = pd.read_csv("orders.csv")

data = orders.merge(products, on="ProductID")
data["Revenue"] = data["Quantity"] * data["Price"]
data["OrderDate"] = pd.to_datetime(data["OrderDate"])

snapshot_date = data["OrderDate"].max() + pd.Timedelta(days=1)

rfm = data.groupby("CustomerID").agg({
    "OrderDate": lambda x: (snapshot_date - x.max()).days,
    "OrderID": "count",
    "Revenue": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# --- RFM Scoring ---
rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)

rfm["RFM_Segment"] = rfm["R_Score"].map(str) + rfm["F_Score"].map(str) + rfm["M_Score"].map(str)
rfm["RFM_Score"] = rfm[["R_Score","F_Score","M_Score"]].sum(axis=1)

# --- Segment Labels ---
def segment_customer(row):
    if row["RFM_Score"] >= 12:
        return "Champions"
    elif row["RFM_Score"] >= 9:
        return "Loyal"
    elif row["RFM_Score"] >= 6:
        return "At Risk"
    else:
        return "Lost"

rfm["Segment"] = rfm.apply(segment_customer, axis=1)

# --- Display ---
st.subheader("📌 RFM Table with Segments")
st.dataframe(rfm.head())

# --- Segment Distribution ---
st.subheader("📊 Segment Distribution")
segment_counts = rfm["Segment"].value_counts()
st.bar_chart(segment_counts)

# --- Executive Summary ---
st.subheader("📰 Executive Summary")
champions = rfm[rfm["Segment"]=="Champions"]
at_risk = rfm[rfm["Segment"]=="At Risk"]
lost = rfm[rfm["Segment"]=="Lost"]

st.markdown(f"""
- **Champions**: {len(champions)} customers driving {champions['Monetary'].sum():,.0f} revenue.  
- **At Risk**: {len(at_risk)} customers need re‑engagement.  
- **Lost**: {len(lost)} customers inactive — consider win‑back campaigns.  
""")
# --- Sidebar Controls ---
st.sidebar.header("⚙️ RFM Thresholds")

recency_cutoff = st.sidebar.slider("Recency cutoff (days)", 30, 365, 90)
frequency_cutoff = st.sidebar.slider("Frequency cutoff (orders)", 1, 10, 3)
monetary_cutoff = st.sidebar.slider("Monetary cutoff (₹)", 1000, 50000, 20000)

def segment_customer(row):
    if row['Recency'] <= recency_cutoff and row['Frequency'] >= frequency_cutoff and row['Monetary'] >= monetary_cutoff:
        return "Champions"
    elif row['Recency'] <= recency_cutoff*2 and row['Frequency'] >= frequency_cutoff:
        return "Loyal"
    elif row['Recency'] > recency_cutoff*2 and row['Frequency'] <= frequency_cutoff:
        return "At Risk"
    elif row['Recency'] > 365:
        return "Lost"
    else:
        return "Others"

rfm["Segment"] = rfm.apply(segment_customer, axis=1)
# --- Download Segmented RFM Table ---
st.subheader("⬇️ Download Segmented RFM Table")

csv = rfm.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name="RFM_Segmentation.csv",
    mime="text/csv"
)

excel = rfm.to_excel("RFM_Segmentation.xlsx", index=False)
with open("RFM_Segmentation.xlsx", "rb") as f:
    st.download_button(
        label="Download as Excel",
        data=f,
        file_name="RFM_Segmentation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

