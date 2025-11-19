# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PM2.5 Forecast Dashboard", layout="wide")

# --- Cities ---
CITIES = {
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867),
    "Madurai": (9.9252, 78.1198),
    "Mysore": (12.2958, 76.6394)
}
HOURS_HISTORY = 168  # Last 7 days

# --- Fetch PM2.5 + weather from Open-Meteo API ---
@st.cache_data(ttl=3600)
def fetch_pm25_weather(lat, lon, hours=HOURS_HISTORY):
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&hourly=pm2_5,temperature_2m,relative_humidity_2m,"
        f"windspeed_10m,winddirection_10m,precipitation&timezone=UTC"
    )
    
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()["hourly"]
        
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data["time"], utc=True),
            "pm2_5": data["pm2_5"],
            "temperature_2m": data["temperature_2m"],
            "relative_humidity_2m": data["relative_humidity_2m"],
            "windspeed_10m": data["windspeed_10m"],
            "winddirection_10m": data["winddirection_10m"],
            "precipitation": data["precipitation"]
        })
        
        df = df.tail(hours).reset_index(drop=True)
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch data: {e}")
        return pd.DataFrame()

# --- Load data for all cities ---
all_data = []
for city, (lat, lon) in CITIES.items():
    st.write(f"Fetching PM2.5 + weather for {city}...")
    city_df = fetch_pm25_weather(lat, lon)
    if not city_df.empty:
        city_df["city"] = city
        all_data.append(city_df)

if not all_data:
    st.error("❌ No data available for any city. Exiting.")
    st.stop()

df_all = pd.concat(all_data, ignore_index=True)

# --- Feature engineering ---
def create_lag_features(df, lags=[1,2,3]):
    df = df.sort_values("datetime").reset_index(drop=True)
    for lag in lags:
        df[f"pm2_5_lag{lag}"] = df["pm2_5"].shift(lag)
    return df.dropna().reset_index(drop=True)

df_all = df_all.groupby("city").apply(create_lag_features).reset_index(drop=True)

# --- Split features/target ---
features = ["pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3", "temperature_2m", 
            "relative_humidity_2m", "windspeed_10m", "winddirection_10m", "precipitation"]
target = "pm2_5"

X = df_all[features]
y = df_all[target]

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train Gradient Boosting Regressor ---
st.write("Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
gb_model.fit(X_scaled, y)
st.success("✅ Model trained.")

# --- Predict next hour ---
next_input = X_scaled[-1].reshape(1,-1)
next_hour_forecast = gb_model.predict(next_input)[0]

# --- Recursive 24h forecast ---
forecast_hours = 24
forecast_list = []
last_row = X_scaled[-1,:].copy()
for h in range(forecast_hours):
    pred = gb_model.predict(last_row.reshape(1,-1))[0]
    forecast_list.append(pred)
    # update lag features
    last_row[0] = last_row[1]
    last_row[1] = last_row[2]
    last_row[2] = pred

# --- Anomaly detection ---
iso = IsolationForest(contamination=0.05)
df_all["anomaly"] = iso.fit_predict(df_all[["pm2_5"]])
anomalies = df_all[df_all["anomaly"]==-1]

# --- Streamlit Dashboard ---
st.title("🌫️ PM2.5 Forecast Dashboard")
st.write(f"Last observed PM2.5: {df_all['pm2_5'].iloc[-1]:.1f}")
st.write(f"Next hour forecast: {next_hour_forecast:.1f}")

st.subheader("24h Forecast")
forecast_df = pd.DataFrame({
    "datetime": pd.date_range(df_all["datetime"].iloc[-1]+pd.Timedelta(hours=1), periods=forecast_hours, freq="H"),
    "pm2_5_forecast": forecast_list
})
st.line_chart(forecast_df.set_index("datetime"))

st.subheader("Anomalies in past data")
st.dataframe(anomalies[["datetime","city","pm2_5"]])

st.subheader("Feature Importance")
feat_imp = pd.DataFrame({
    "feature": features,
    "importance": gb_model.feature_importances_
}).sort_values("importance", ascending=False)
st.bar_chart(feat_imp.set_index("feature"))
