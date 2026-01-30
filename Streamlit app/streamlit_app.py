# Streamlit app start

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Superstore Sales Dashboard")
st.caption("Interactive analytics dashboard for exploring sales performance, profitability drivers, "
    "and discount risk across regions, categories, and customer segments. "
    "All metrics and visualisations update dynamically based on sidebar filters, "
    "supporting data-driven decision-making and stakeholder insight.")


with st.expander("How to use this dashboard"):
    st.markdown("""
    **How to use:**
    - Adjust filters in the sidebar to narrow the dataset  
    - Metrics update automatically based on your selections  
    - Charts reflect the current filtered view  
    - Use different date ranges and segments to compare performance  
    """)

st.markdown(
    "### Story flow\n"
    "Start with the **Overview** KPIs, then compare **Category** and **Regional** performance. "
    "Next, explore **Discount vs Profit Risk** to spot margin pressure, and use **Time Trends** "
    "and **Distribution View** to validate patterns over time and identify outliers."
)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    # Resolve path relative to this file (robust regardless of working directory)
    app_dir = Path(__file__).resolve().parent
    csv_path = (app_dir.parent / "data" / "processed" / "superstore_cleaned.csv").resolve()

    if not csv_path.exists():
        st.error(f"CSV not found at: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)

# Ensure dates are datetime
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

    # If ship_days missing, create it
    if "ship_days" not in df.columns and {"ship_date", "order_date"}.issubset(df.columns):
        df["ship_days"] = (df["ship_date"] - df["order_date"]).dt.days

    # If order_year/month missing, create them
    if "order_year" not in df.columns and "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year

    if "order_month" not in df.columns and "order_date" in df.columns:
        df["order_month"] = df["order_date"].dt.month

    # Ensure key categoricals are strings for filters (Streamlit-friendly)
    for c in ["region", "category", "sub_category", "segment", "ship_mode", "state", "city"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


df = load_data()

# -----------------------------
# Basic validation
# -----------------------------
required_cols = {"sales", "profit", "discount", "quantity", "order_date", "region", "category", "segment"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns in processed dataset: {sorted(list(missing))}")
    st.stop()

# -----------------------------
# Sidebar filters of min and max date
# -----------------------------
st.sidebar.header("Filters")

min_date = df["order_date"].min()
max_date = df["order_date"].max()

date_range = st.sidebar.date_input(
    "Order date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
    help="Filter dashboard metrics and charts by order date."
)

regions = sorted(df["region"].unique())
categories = sorted(df["category"].unique())
segments = sorted(df["segment"].unique())

region_sel = st.sidebar.multiselect("Region", regions, default=regions, help="Filter by region.")
category_sel = st.sidebar.multiselect("Category", categories, default=categories, help="Filter by product category.")
segment_sel = st.sidebar.multiselect("Segment", segments, default=segments, help="Filter by customer segment.")

# Apply filters
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask = (
    (df["order_date"] >= start_date) &
    (df["order_date"] <= end_date) &
    (df["region"].isin(region_sel)) &
    (df["category"].isin(category_sel)) &
    (df["segment"].isin(segment_sel))
)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No data matches the current filters. Try expanding the date range or selections.")
    st.stop()

# -----------------------------
# Overview Metrics (Section)
# -----------------------------
st.subheader("Overview")

total_sales = fdf["sales"].sum()
total_profit = fdf["profit"].sum()
orders = fdf["order_id"].nunique() if "order_id" in fdf.columns else len(fdf)
avg_discount = fdf["discount"].mean()
profit_margin = (total_profit / total_sales) if total_sales else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Sales", f"${total_sales:,.0f}", help="Sum of sales for selected filters.")
c2.metric("Total Profit", f"${total_profit:,.0f}", help="Sum of profit for selected filters.")
c3.metric("Orders", f"{orders:,}", help="Number of unique orders (or rows if order_id missing).")
c4.metric("Avg Discount", f"{avg_discount:.1%}", help="Average discount across filtered data.")
c5.metric(
    "Profit Margin",
    f"{profit_margin:.1%}" if np.isfinite(profit_margin) else "N/A",
    help="Total Profit / Total Sales for selected filters."
)

st.divider()

# -----------------------------
# Helper: AI-style Summary
# -----------------------------
def generate_rule_based_summary(d: pd.DataFrame) -> str:
    # Top category by profit
    cat_profit = d.groupby("category")["profit"].sum().sort_values(ascending=False)
    top_cat = cat_profit.index[0]
    top_cat_profit = cat_profit.iloc[0]

    # Region performance
    reg_profit = d.groupby("region")["profit"].sum().sort_values(ascending=False)
    top_reg = reg_profit.index[0]
    bottom_reg = reg_profit.index[-1]

    # Discount-risk relationship quick signal
    corr = d[["discount", "profit"]].corr(numeric_only=True).iloc[0, 1]
    corr_txt = "negative" if corr < -0.05 else "positive" if corr > 0.05 else "weak/unclear"

    return (
        f"**Summary (auto-generated):** In the selected data, **{top_cat}** is the top category by total profit "
        f"(${top_cat_profit:,.0f}). The strongest region by profit is **{top_reg}**, while **{bottom_reg}** trails. "
        f"The relationship between discount and profit appears **{corr_txt}** (corr ≈ {corr:.2f}), suggesting discounting "
        f"may impact profitability depending on segment/category mix."
    )


with st.expander("AI-Generated Summary (for stakeholders)", expanded=True):
    st.write(generate_rule_based_summary(fdf))
    st.caption("This is a rule-based ‘AI-style’ narrative generated from the filtered dataset.")

st.divider()

# -----------------------------
# Category Performance (Bar chart)
# -----------------------------
st.caption(
    "Compare categories by **total profit** and **total sales**. "
    "Look for categories with **high sales but lower profit**, which can indicate margin pressure "
    "or discount-heavy selling."
)
st.subheader("Category Performance")

cat_summary = (
    fdf.groupby("category", as_index=False)
    .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"))
    .sort_values("total_profit", ascending=False)
)

colA, colB = st.columns([1, 1])
with colA:
    fig_cat_profit = px.bar(
        cat_summary,
        x="category",
        y="total_profit",
        title="Total Profit by Category (Bar Chart)",
        labels={"category": "Category", "total_profit": "Total Profit"}
    )
    fig_cat_profit.update_traces(hovertemplate="Category=%{x}<br>Total Profit=%{y:$,.0f}<extra></extra>")
    st.plotly_chart(fig_cat_profit, use_container_width=True)

with colB:
    fig_cat_sales = px.bar(
        cat_summary,
        x="category",
        y="total_sales",
        title="Total Sales by Category (Bar Chart)",
        labels={"category": "Category", "total_sales": "Total Sales"}
    )
    fig_cat_sales.update_traces(hovertemplate="Category=%{x}<br>Total Sales=%{y:$,.0f}<extra></extra>")
    st.plotly_chart(fig_cat_sales, use_container_width=True)

st.divider()

# -----------------------------
# Regional Performance (Bar)
# -----------------------------
st.subheader("Regional Performance")
st.caption(
    "This view highlights which regions contribute most to overall profitability. "
    "Use it to identify **strong-performing regions** and areas that may require pricing, mix, "
    "or operational attention."
)
reg_summary = (
    fdf.groupby("region", as_index=False)
    .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"))
    .sort_values("total_profit", ascending=False)
)

fig_reg = px.bar(
    reg_summary,
    x="region",
    y="total_profit",
    title="Total Profit by Region (Bar Chart)",
    labels={"region": "Region", "total_profit": "Total Profit"}
)
fig_reg.update_traces(hovertemplate="Region=%{x}<br>Total Profit=%{y:$,.0f}<extra></extra>")
st.plotly_chart(fig_reg, use_container_width=True)

st.divider()

# -----------------------------
# Discount vs Profit Risk (Scatter) + optional Box plot
# -----------------------------
st.subheader("Discount vs Profit Risk")

col1, col2 = st.columns([2, 1])

with col1:
    fig_scatter = px.scatter(
        fdf,
        x="discount",
        y="profit",
        color="category",
        hover_data=["segment", "region", "sales", "quantity"],
        title="Profit vs Discount (Scatter Plot)",
        labels={"discount": "Discount", "profit": "Profit"}
    )
    fig_scatter.update_traces(hovertemplate="Discount=%{x:.0%}<br>Profit=%{y:$,.2f}<extra></extra>")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    show_box = st.checkbox(
        "Show box plot by discount band",
        value=True,
        help="Summarises profit spread across discount ranges."
    )

    if show_box:
        bins = [-0.001, 0.0, 0.1, 0.2, 0.3, 0.4, 1.0]
        labels = ["0%", "0–10%", "10–20%", "20–30%", "30–40%", "40%+"]

        temp = fdf.copy()
        temp["discount_band"] = pd.cut(temp["discount"], bins=bins, labels=labels)

        fig_box = px.box(
            temp,
            x="discount_band",
            y="profit",
            title="Profit Distribution by Discount Band (Box Plot)",
            labels={"discount_band": "Discount Band", "profit": "Profit"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# -----------------------------
# Time Trends (Line chart)
# -----------------------------
st.subheader("Time Trends")
st.caption(
    "Track performance over time to spot **seasonality**, growth/decline periods, and moments where "
    "**profit diverges from sales** (potential discounting or cost effects)."
)
monthly = (
    fdf.dropna(subset=["order_date"])
    .assign(order_month_start=lambda d: d["order_date"].dt.to_period("M").dt.to_timestamp())
    .groupby("order_month_start", as_index=False)
    .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"))
    .sort_values("order_month_start")
)

colL, colR = st.columns(2)

with colL:
    fig_line_sales = px.line(
        monthly,
        x="order_month_start",
        y="total_sales",
        title="Total Sales Over Time (Line Chart)",
        labels={"order_month_start": "Month", "total_sales": "Total Sales"}
    )
    fig_line_sales.update_traces(hovertemplate="Month=%{x|%b %Y}<br>Total Sales=%{y:$,.0f}<extra></extra>")
    st.plotly_chart(fig_line_sales, use_container_width=True)

with colR:
    fig_line_profit = px.line(
        monthly,
        x="order_month_start",
        y="total_profit",
        title="Total Profit Over Time (Line Chart)",
        labels={"order_month_start": "Month", "total_profit": "Total Profit"}
    )
    fig_line_profit.update_traces(hovertemplate="Month=%{x|%b %Y}<br>Total Profit=%{y:$,.0f}<extra></extra>")
    st.plotly_chart(fig_line_profit, use_container_width=True)

st.divider()

# -----------------------------
# Histogram
# -----------------------------
st.subheader("Distribution View")
st.caption(
    "Histograms help identify **skewness** and **outliers**. "
    "Skewed distributions mean the **median** may represent a more typical order than the mean."
)
metric_choice = st.selectbox(
    "Choose a metric to view distribution",
    options=["sales", "profit", "discount", "quantity", "ship_days"],
    help="Histograms help identify skewness and outliers."
)

if metric_choice in fdf.columns:
    fig_hist = px.histogram(
        fdf,
        x=metric_choice,
        nbins=30,
        title=f"Distribution of {metric_choice} (Histogram)",
        labels={metric_choice: metric_choice}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info(f"'{metric_choice}' is not available in the filtered dataset.")

# -----------------------------
# Footer / Notes
# -----------------------------
st.caption(
    "Notes: This dashboard is built for exploration. Use filters to compare groups and timespans. "
    "All metrics update based on selection."
)
st.caption(
    "Data source: Public anonymised retail dataset — "
    "[Superstore Dataset (Kaggle)](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)"
)