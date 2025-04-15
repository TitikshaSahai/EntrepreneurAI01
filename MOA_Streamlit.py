# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re
import json
import urllib.request

# ------------------------
# Header & Configuration
# ------------------------
st.set_page_config(page_title="Canadian Retail Market Intelligence", layout="wide")
st.title("Entrepreneur AI")
st.markdown("This dashboard uses retail sales, growth trends, and opportunity analytics to identify high-potential markets across Canada.")

# ------------------------
# Load & Preprocess Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_sales_data.csv")
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df['Year'] = df['REF_DATE'].dt.year
    df['Month'] = df['REF_DATE'].dt.month
    df['NAICS_Code'] = df['Industry'].apply(lambda x: re.search(r"\[(\d{2,6})\]", str(x)).group(1) if re.search(r"\[(\d{2,6})\]", str(x)) else None)
    df['NAICS_3Digit'] = df['NAICS_Code'].str[:3]
    return df

df = load_data()
latest_year = df['Year'].max()

# ------------------------
# Sidebar Filters
# ------------------------
st.sidebar.header("Filter Options")
sector_options = sorted(df['NAICS_3Digit'].dropna().unique())
province_options = sorted(df['Province'].dropna().unique())

selected_sector = st.sidebar.selectbox("Select Industry Sector (NAICS 3-Digit)", options=["All"] + sector_options)
selected_province = st.sidebar.selectbox("Select Province", options=["All"] + province_options)
rel_score_thresh = st.sidebar.slider("Relative Sales Score Threshold", 0.5, 1.5, 1.0, 0.05)
min_cagr = st.sidebar.slider("Minimum CAGR (%)", 0.0, 25.0, 5.0, 0.5)

# ------------------------
# National Overview
# ------------------------
st.subheader("National-Level Retail Trends")
col1, col2 = st.columns(2)

with col1:
    national_sales = df[(df['Province'] == 'Canada') & (df['Sales'] == 'Total retail sales') & (df['Adjustments'] == 'Unadjusted')]
    monthly = national_sales.groupby('REF_DATE')['Sales_Value'].sum().reset_index()
    fig1 = px.line(monthly, x='REF_DATE', y='Sales_Value', title='Monthly National Retail Sales', template='plotly_white')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    ecom_trend = df[(df['Province'] == 'Canada') & (df['Sales'] == 'Retail e-commerce sales') & (df['Adjustments'] == 'Unadjusted')]
    ecom_monthly = ecom_trend.groupby('REF_DATE')['Sales_Value'].sum().reset_index()
    fig2 = px.line(ecom_monthly, x='REF_DATE', y='Sales_Value', title='National E-commerce Sales Trend', template='plotly_white', color_discrete_sequence=['green'])
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------
# National Opportunity Visualizations
# ------------------------
st.subheader("National Opportunity Visualizations")

# CAGR
cagr_df = df[(df['Province'] == 'Canada') & (df['Sales'] == 'Total retail sales') & (df['Adjustments'] == 'Unadjusted') & (df['NAICS_3Digit'].notna())]
pivot_df = cagr_df.pivot_table(index='NAICS_3Digit', columns='Year', values='Sales_Value', aggfunc='sum').dropna(thresh=2)
start, end = pivot_df.columns.min(), pivot_df.columns.max()
cagr_vals = ((pivot_df[end] / pivot_df[start]) ** (1 / (end - start)) - 1) * 100
cagr_vals = cagr_vals.reset_index().rename(columns={0: 'CAGR (%)'})

# Growth vs Volatility Matrix
with st.expander(" Growth vs Volatility Matrix"):
    yoy = pivot_df.pct_change(axis=1) * 100
    trend_df = pd.DataFrame({
        'NAICS_3Digit': yoy.index,
        'Mean YoY Growth (%)': yoy.mean(axis=1).round(2),
        'Volatility (Std Dev %)': yoy.std(axis=1).round(2),
        'Latest Sales (Billion CAD)': (pivot_df[end] / 1e9).round(2)
    })
    mean_growth = trend_df['Mean YoY Growth (%)'].mean()
    mean_volatility = trend_df['Volatility (Std Dev %)'].mean()
    trend_df['Quadrant'] = trend_df.apply(
        lambda r: "Stable Growth" if r['Mean YoY Growth (%)'] >= mean_growth and r['Volatility (Std Dev %)'] <= mean_volatility
        else "Risky Boom" if r['Mean YoY Growth (%)'] >= mean_growth
        else "Slow but Stable" if r['Volatility (Std Dev %)'] <= mean_volatility
        else "Volatile Decline", axis=1)
    fig = px.scatter(
        trend_df, x='Volatility (Std Dev %)', y='Mean YoY Growth (%)',
        size='Latest Sales (Billion CAD)', color='Quadrant', hover_name='NAICS_3Digit',
        title="Growth vs Volatility Matrix (Bubble = Market Size)", template='plotly_white', size_max=60
    )
    st.plotly_chart(fig, use_container_width=True)

# Choropleth Map
with st.expander(" Map of Under-Served High-Growth Industries by Province"):
    url = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/canada.geojson"
    geo_path = "canada_provinces.geojson"
    urllib.request.urlretrieve(url, geo_path)
    with open(geo_path) as f:
        canada_geojson = json.load(f)
    all_provinces = [feature['properties']['name'] for feature in canada_geojson['features']]
    all_df = pd.DataFrame({'Province': all_provinces})
    prov_df = df[(df['Year'] == latest_year) & (df['Sales'] == 'Total retail sales') & (df['Adjustments'] == 'Unadjusted') & (df['NAICS_3Digit'].notna())]
    prov_sales = prov_df.groupby(['Province', 'NAICS_3Digit'])['Sales_Value'].sum().reset_index()
    national_avg = prov_sales.groupby('NAICS_3Digit')['Sales_Value'].mean().reset_index(name='Avg_Sales')
    opp_by_prov = prov_sales.merge(national_avg, on='NAICS_3Digit')
    opp_by_prov = opp_by_prov.merge(cagr_vals, on='NAICS_3Digit')
    opp_by_prov['Relative_Sales_Score'] = opp_by_prov['Sales_Value'] / opp_by_prov['Avg_Sales']
    filtered = opp_by_prov[(opp_by_prov['Relative_Sales_Score'] < rel_score_thresh) & (opp_by_prov['CAGR (%)'] > min_cagr)]
    industry_summary = filtered.groupby('Province')['NAICS_3Digit'].nunique().reset_index(name='Under_Served_Industries')
    industry_summary = pd.merge(all_df, industry_summary, on='Province', how='left').fillna(0)
    fig_map = px.choropleth(
        industry_summary, geojson=canada_geojson, featureidkey="properties.name",
        locations="Province", color="Under_Served_Industries",
        color_continuous_scale="Oranges", scope="north america",
        title="Under-Served High-Growth Industries by Province",
        labels={"Under_Served_Industries": "# of Industries"})
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

# ------------------------
# Drill-down to Province
# ------------------------
st.subheader(" Provincial Opportunity Analysis")

prov_df = df[(df['Year'] == latest_year) & (df['Sales'] == 'Total retail sales') & (df['Adjustments'] == 'Unadjusted') & (df['NAICS_3Digit'].notna())]
prov_sales = prov_df.groupby(['Province', 'NAICS_3Digit'])['Sales_Value'].sum().reset_index()
national_avg = prov_sales.groupby('NAICS_3Digit')['Sales_Value'].mean().reset_index(name='Avg_Sales')

if selected_province != "All":
    selected_prov_df = prov_sales[prov_sales['Province'] == selected_province].copy()
    selected_prov_df = selected_prov_df.merge(national_avg, on='NAICS_3Digit')
    selected_prov_df['Relative_Sales_Score'] = selected_prov_df['Sales_Value'] / selected_prov_df['Avg_Sales']
    selected_prov_df = selected_prov_df.merge(cagr_vals, on='NAICS_3Digit', how='left')
    if selected_sector != "All":
        selected_prov_df = selected_prov_df[selected_prov_df['NAICS_3Digit'] == selected_sector]
    result = selected_prov_df[(selected_prov_df['Relative_Sales_Score'] < rel_score_thresh) & (selected_prov_df['CAGR (%)'] > min_cagr)]

    st.markdown(f"### Opportunity Summary for **{selected_province}**")
    if result.empty:
        st.warning("No matching industries found with current filters.")
    else:
        st.dataframe(result[['NAICS_3Digit', 'Sales_Value', 'Avg_Sales', 'Relative_Sales_Score', 'CAGR (%)']].round(2))
        fig = px.bar(result.head(10), y='NAICS_3Digit', x='CAGR (%)', orientation='h', template='plotly_white', title='Top Under-Served Industries')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select a province from the sidebar to see specific opportunities.")

