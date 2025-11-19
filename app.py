# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error
import requests

st.set_page_config(page_title="PM2.5 Forecast Dashboard", layout="wide")
st.title("🌆 PM2.5 Forecast Dashboard")
st.markdown("Live PM2.5 + Weather data with 24–48h forecast and anomaly detection")

# -------------------------------
# 1️⃣ Define cities
cities_info = {
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867),
    "Madurai": (9.9252, 78.1198),
    "Mysore": (12.2958, 76.6394)
}

HOURS_HISTORY = 168  # last 7 days
FORECAST_HOURS = 48  # 24–48h forecast

# -------------------------------
# 2️⃣ Fetch PM2.5 + Weather from Open-Meteo
def fetch_open_meteo_city(lat, lon, hours=HOURS_HISTORY):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&hourly=pm2_5,temperature_2m,relative_humidity_2m,"
        "windspeed_10m,winddirection_10m,precipitation"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"], utc=True),
            "pm2_5": data["hourly"]["pm2_5"],
            "temperature_2m": data["hourly"]["temperature_2m"],
            "relative_humidity_2m": data["hourly"]["relative_humidity_2m"],
            "windspeed_10m": data["hourly"]["windspeed_10m"],
            "winddirection_10m": data["hourly"]["winddirection_10m"],
            "precipitation": data["hourly"]["precipitation"]
        })
        return df.tail(hours)
    except:
        return pd.DataFrame()

# -------------------------------
# 3️⃣ Collect all cities data
all_cities_df = []
for city, (lat, lon) in cities_info.items():
    st.write(f"Fetching PM2.5 + weather for {city}...")
    df_city = fetch_open_meteo_city(lat, lon)
    if df_city.empty:
        st.warning(f"⚠️ Failed to fetch data for {city}")
        continue
    df_city["city"] = city
    all_cities_df.append(df_city)

final_df = pd.concat(all_cities_df, ignore_index=True)

# -------------------------------
# 4️⃣ Sidebar: select city
selected_city = st.sidebar.selectbox("Select City:", list(cities_info.keys()))
city_df = final_df[final_df["city"] == selected_city].reset_index(drop=True)

# -------------------------------
# 5️⃣ Feature Engineering
# Lag features for model
lags = [1, 2, 3, 6, 12, 24]
for lag in lags:
    city_df[f"pm2_5_lag_{lag}"] = city_df["pm2_5"].shift(lag)
city_df = city_df.dropna().reset_index(drop=True)

feature_cols = ["pm2_5_lag_1","pm2_5_lag_2","pm2_5_lag_3","pm2_5_lag_6","pm2_5_lag_12","pm2_5_lag_24",
                "temperature_2m","relative_humidity_2m","windspeed_10m","winddirection_10m","precipitation"]
target_col = "pm2_5"

X = city_df[feature_cols].fillna(0)
y = city_df[target_col]

# -------------------------------
# 6️⃣ Train Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
model.fit(X, y)

# Feature importance
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# -------------------------------
# 7️⃣ Forecasting next 48h recursively
last_row = city_df.iloc[-1:].copy()
forecast_list = []

for h in range(1, FORECAST_HOURS + 1):
    X_pred = last_row[feature_cols].fillna(0)
    y_pred = model.predict(X_pred)[0]
    forecast_list.append(y_pred)
    
    # Update lags for next hour
    for lag in reversed(lags):
        if lag == 1:
            last_row[f"pm2_5_lag_{lag}"] = y_pred
        else:
            last_row[f"pm2_5_lag_{lag}"] = last_row[f"pm2_5_lag_{lag-1}"]

forecast_df = pd.DataFrame({
    "datetime": [city_df["datetime"].iloc[-1] + timedelta(hours=i) for i in range(1, FORECAST_HOURS+1)],
    "pm2_5_forecast": forecast_list
})

# -------------------------------
# 8️⃣ Anomaly detection on PM2.5
iso = IsolationForest(contamination=0.05, random_state=42)
city_df["anomaly"] = iso.fit_predict(city_df[["pm2_5"]])
city_df["anomaly"] = city_df["anomaly"].apply(lambda x: True if x==-1 else False)

# -------------------------------
# 9️⃣ Display last observed PM2.5
st.subheader(f"{selected_city} - Last Observed PM2.5")
st.metric(label="PM2.5 (µg/m³)", value=round(city_df["pm2_5"].iloc[-1], 1))

# -------------------------------
# 10️⃣ Plot PM2.5 + Forecast + Anomalies
fig = px.line(city_df, x="datetime", y="pm2_5", title=f"{selected_city} PM2.5 History & Forecast")
fig.add_scatter(x=forecast_df["datetime"], y=forecast_df["pm2_5_forecast"], mode="lines", name="Forecast")
fig.add_scatter(x=city_df[city_df["anomaly"]]["datetime"], 
                y=city_df[city_df["anomaly"]]["pm2_5"], 
                mode="markers", name="Anomaly", marker=dict(color="red", size=8))
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 11️⃣ Feature importance chart
st.subheader("Feature Importance")
fig_feat = px.bar(feature_importance, x="importance", y="feature", orientation="h")
st.plotly_chart(fig_feat, use_container_width=True)

# -------------------------------
# 12️⃣ Validation Metrics (MAE, coverage %)
y_pred_train = model.predict(X)
mae = mean_absolute_error(y, y_pred_train)
coverage = round(100 * (~X.isna()).all(axis=1).mean(), 2)

st.subheader("Model Validation Metrics")
st.write(f"MAE on training set: {mae:.2f}")
st.write(f"Coverage of valid PM2.5 readings: {coverage}%")
st.write(f"Forecast horizon: {FORECAST_HOURS} hours")
