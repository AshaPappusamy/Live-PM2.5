import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
# Import your trained pipeline and functions
# from model_pipeline import pipeline
# from feature_engineering import add_lag_features
# from forecast_functions import recursive_forecast
# from data_fetch import fetch_openaq_pm25, fetch_open_meteo_pm25, fetch_weather

st.set_page_config(page_title="Live PM2.5 Forecast", layout="wide")
st.title("🌫️ Live PM2.5 Nowcast & Forecast Dashboard")

# -------------------------------
# 1️⃣ Fetch Live PM2.5 + Weather Data
# -------------------------------
@st.cache_data(ttl=3600)
def fetch_latest_data():
    LAT, LON = 12.9716, 77.5946  # Bangalore example
    HOURS_HISTORY = 48
    try:
        pm25 = fetch_openaq_pm25(LAT, LON, radius_km=25, hours_history=HOURS_HISTORY)
    except:
        pm25 = fetch_open_meteo_pm25(LAT, LON, HOURS_HISTORY)
    weather = fetch_weather(LAT, LON, HOURS_HISTORY)
    # Merge and process features
    df = pd.merge(pm25, weather, on='datetime', how='inner')
    df = add_lag_features(df.set_index('datetime'))
    return df

pm25_features = fetch_latest_data()

# -------------------------------
# 2️⃣ Generate Forecast
# -------------------------------
FORECAST_HORIZON = st.sidebar.slider("Forecast Horizon (hours)", 1, 48, 24)
last_ts = pm25_features.index.max()

forecast_df = recursive_forecast(
    pipeline,        # your trained pipeline variable
    pm25_features,
    last_ts,
    horizon=FORECAST_HORIZON
)

# -------------------------------
# 3️⃣ Combine Observed + Forecast
# -------------------------------
obs_recent = pm25_features[['pm25']].copy()
obs_recent = obs_recent.loc[pm25_features.index.max() - timedelta(hours=48):]

forecast_df.index = pd.to_datetime(forecast_df.index)

combined = pd.concat([
    obs_recent.rename(columns={'pm25': 'Observed'}),
    forecast_df.rename(columns={'pm25_pred': 'Forecast'})
], axis=1)

# -------------------------------
# 4️⃣ Visualization: Observed vs Forecast
# -------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=combined.index, y=combined['Observed'], mode='lines+markers',
    name='Observed PM2.5', line=dict(color='royalblue', width=2)
))
fig.add_trace(go.Scatter(
    x=combined.index, y=combined['Forecast'], mode='lines+markers',
    name='Forecast PM2.5', line=dict(color='orange', width=2, dash='dot')
))
# Highlight anomalies
threshold = 90
anomalies = forecast_df[forecast_df['pm25_pred'] > threshold]
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies.index, y=anomalies['pm25_pred'], mode='markers',
        name='High PM2.5 (>90)', marker=dict(color='red', size=8, symbol='x')
    ))

st.subheader("Observed vs Forecast PM2.5")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 5️⃣ Summary Metrics
# -------------------------------
overlap = combined.dropna(subset=['Observed', 'Forecast'])
if len(overlap) > 0:
    mae = mean_absolute_error(overlap['Observed'], overlap['Forecast'])
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} µg/m³")

st.write("📈 Forecast Summary Statistics:")
st.write(f"- Average Forecast: {forecast_df['pm25_pred'].mean():.2f} µg/m³")
st.write(f"- Max Forecast: {forecast_df['pm25_pred'].max():.2f} µg/m³")
st.write(f"- Min Forecast: {forecast_df['pm25_pred'].min():.2f} µg/m³")

# -------------------------------
# 6️⃣ Feature Importance
# -------------------------------
importances = pipeline.named_steps['gradientboostingregressor'].feature_importances_
feature_cols = pm25_features.drop(columns=['pm25']).columns.tolist()
feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': importances})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)

st.subheader("Top Features Affecting PM2.5 Prediction")
st.bar_chart(feature_importance.set_index('feature').head(10))

# -------------------------------
# 7️⃣ Download Forecast CSV
# -------------------------------
st.download_button(
    label="Download Forecast CSV",
    data=forecast_df.to_csv(index=True),
    file_name='pm25_forecast.csv',
    mime='text/csv'
)
