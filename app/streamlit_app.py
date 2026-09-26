"""Customer analytics dashboard: KPIs, cohort retention and RFM segments for the Olist marketplace."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
BLUE, ORANGE, INK_2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"

st.set_page_config(page_title="Customer Analytics · Olist", page_icon="📊", layout="wide")


@st.cache_data
def load():
    orders = pd.read_parquet(DATA / "orders.parquet")
    customers = pd.read_parquet(DATA / "customers.parquet")
    orders["order_month"] = pd.to_datetime(orders["order_month"])
    customers["cohort_month"] = pd.to_datetime(customers["cohort_month"])
    return orders, customers


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13), hoverlabel=dict(font_size=13),
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


orders, customers = load()

st.title("Customer Analytics Dashboard")
st.caption("Olist Brazilian e-commerce marketplace · ~98K orders from ~95K customers · Jan 2017 – Aug 2018 · "
           "[source code](https://github.com/VineethVadlapalli/olist-customer-analytics)")

# Filters: one row above the charts
f1, f2 = st.columns([2, 3])
states = sorted(orders.customer_state.dropna().unique())
chosen_states = f1.multiselect("Customer state", states, placeholder="All states")
months = sorted(orders.order_month.dt.date.unique())
start, end = f2.select_slider("First-purchase month", options=months,
                              value=(pd.Timestamp("2017-01-01").date(), pd.Timestamp("2018-08-01").date()),
                              format_func=lambda d: d.strftime("%b %Y"))

cust = customers[(customers.cohort_month.dt.date >= start) & (customers.cohort_month.dt.date <= end)]
if chosen_states:
    cust = cust[cust.customer_state.isin(chosen_states)]
# Sep/Oct 2018 are partial months in the source data; showing them would fake a collapse.
ords = orders[orders.customer_unique_id.isin(cust.customer_unique_id) & (orders.order_month <= "2018-08-01")]

if cust.empty:
    st.warning("No customers match these filters.")
    st.stop()

# KPI tiles
k = st.columns(5)
k[0].metric("Customers", f"{len(cust):,}")
k[1].metric("Revenue (BRL)", f"{ords.order_value.sum() / 1e6:.2f}M")
k[2].metric("Avg order value", f"{ords.order_value.mean():.0f} BRL")
k[3].metric("Repeat customers", f"{(cust.frequency > 1).mean():.1%}")
k[4].metric("Avg review score", f"{ords.review_score.mean():.2f} / 5")

tab_trend, tab_cohort, tab_segment, tab_delivery = st.tabs(
    ["Revenue trend", "Cohort retention", "Customer segments", "Delivery vs reviews"])

with tab_trend:
    monthly = (ords.groupby("order_month").agg(revenue=("order_value", "sum"), orders=("order_id", "count"))
               .reset_index())
    fig = px.line(monthly, x="order_month", y="revenue", markers=True,
                  labels={"order_month": "", "revenue": "Revenue (BRL)"})
    fig.update_traces(line=dict(color=BLUE, width=2), marker=dict(size=7),  # complete months only
                      hovertemplate="%{x|%b %Y}<br>Revenue: %{y:,.0f} BRL<extra></extra>")
    st.subheader("Monthly revenue")
    st.plotly_chart(style(fig), use_container_width=True)

with tab_cohort:
    act = ords.merge(cust[["customer_unique_id", "cohort_month"]], on="customer_unique_id")
    act["m"] = ((act.order_month.dt.year - act.cohort_month.dt.year) * 12
                + act.order_month.dt.month - act.cohort_month.dt.month)
    sizes = cust.groupby("cohort_month").size()
    ret = act.groupby(["cohort_month", "m"]).customer_unique_id.nunique().unstack().div(sizes, axis=0)
    ret = ret.loc[:, [c for c in ret.columns if 1 <= c <= 12]] * 100
    fig = go.Figure(go.Heatmap(
        z=ret.values, x=[f"M{c}" for c in ret.columns], y=ret.index.strftime("%b %Y"),
        colorscale="Blues", zmin=0, colorbar=dict(title="% returning"),
        hovertemplate="Cohort %{y}<br>%{x}: %{z:.2f}% bought again<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    st.subheader("Share of each first-purchase cohort buying again, by month")
    st.plotly_chart(style(fig, 520), use_container_width=True)
    st.caption("Fewer than 1 in 100 customers buys again in any later month: this is a one-purchase business, "
               "so growth depends on acquisition and the second order is the biggest untapped lever.")

with tab_segment:
    seg = (cust.groupby("segment").agg(customers=("customer_unique_id", "count"), revenue=("monetary", "sum"),
                                       avg_spend=("monetary", "mean"), avg_recency_days=("recency_days", "mean"))
           .assign(pct_customers=lambda d: d.customers / d.customers.sum() * 100,
                   pct_revenue=lambda d: d.revenue / d.revenue.sum() * 100)
           .sort_values("revenue").reset_index())
    fig = go.Figure()
    fig.add_bar(y=seg.segment, x=seg.pct_customers, orientation="h", name="% of customers", marker_color=BLUE,
                hovertemplate="%{y}<br>%{x:.1f}% of customers<extra></extra>")
    fig.add_bar(y=seg.segment, x=seg.pct_revenue, orientation="h", name="% of revenue", marker_color=ORANGE,
                hovertemplate="%{y}<br>%{x:.1f}% of revenue<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    st.subheader("RFM segments: share of customers vs share of revenue")
    st.plotly_chart(style(fig, 400), use_container_width=True)
    st.dataframe(
        seg.sort_values("revenue", ascending=False).set_index("segment")
        .style.format({"customers": "{:,}", "revenue": "{:,.0f}", "avg_spend": "{:.0f}",
                       "avg_recency_days": "{:.0f}", "pct_customers": "{:.1f}%", "pct_revenue": "{:.1f}%"}),
        use_container_width=True)

with tab_delivery:
    d = ords.dropna(subset=["days_vs_estimate", "review_score"]).copy()
    d["bucket"] = pd.cut(d.days_vs_estimate, [-999, -10, -1, 0, 7, 999],
                         labels=["10+ days early", "1-9 days early", "on the estimated day", "1-7 days late",
                                 "8+ days late"])
    b = (d.groupby("bucket", observed=True).review_score
         .agg(bad_rate=lambda s: (s <= 2).mean() * 100, orders="count").reset_index())
    fig = px.bar(b, x="bucket", y="bad_rate", labels={"bucket": "", "bad_rate": "% of 1-2 star reviews"},
                 custom_data=["orders"])
    fig.update_traces(marker_color=[ORANGE if "late" in x else BLUE for x in b.bucket], width=0.5,
                      hovertemplate="%{x}<br>%{y:.1f}% bad reviews<br>%{customdata[0]:,} orders<extra></extra>")
    st.subheader("Late deliveries drive bad reviews")
    st.plotly_chart(style(fig), use_container_width=True)
