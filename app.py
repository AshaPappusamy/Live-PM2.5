# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PM2.5 Forecast Dashboard", layout="wide")

# -------------------- Cities --------------------
CITIES = {
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.7041, 77.1025),
    "Hyderabad": (17.3850, 78.4867),
    "Madurai": (9.9252, 78.1198),
    "Mysore": (12.2958, 76.6394)
}

HOURS_HISTORY = 168  # 7 days

# -------------------- Data Fetch --------------------
@st.cache_data(ttl=3600)
def fetch_pm25_weather(lat, lon, hours=HOURS_HISTORY):
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=pm2_5,temperature_2m,relative_humidity_2m,"
        "windspeed_10m,winddirection_10m,precipitation"
        "&timezone=UTC"
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

        return df.tail(hours).reset_index(drop=True)

    except Exception as e:
        st.warning(f"⚠️ Data fetch failed: {e}")
        return pd.DataFrame()


# -------------------- Load Data --------------------
all_data = []

for city, (lat, lon) in CITIES.items():
    st.write(f"Fetching data for {city}...")
    city_df = fetch_pm25_weather(lat, lon)

    if not city_df.empty:
        city_df["city"] = city
        all_data.append(city_df)

if not all_data:
    st.error("❌ No data available from API")
    st.stop()

df_all = pd.concat(all_data, ignore_index=True)


# -------------------- Feature Engineering --------------------
def create_lag_features(df):
    df = df.sort_values("datetime")
    df["pm2_5_lag1"] = df["pm2_5"].shift(1)
    df["pm2_5_lag2"] = df["pm2_5"].shift(2)
    df["pm2_5_lag3"] = df["pm2_5"].shift(3)
    return df.dropna()

df_all = (
    df_all
    .groupby("city", group_keys=False)
    .apply(create_lag_features)
    .reset_index(drop=True)
)

# -------------------- Features --------------------
features = [
    "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3",
    "temperature_2m", "relative_humidity_2m",
    "windspeed_10m", "winddirection_10m", "precipitation"
]

target = "pm2_5"

X = df_all[features]
y = df_all[target]

# -------------------- 🔴 CRITICAL CLOUD FIX --------------------
# Force numeric (API sometimes returns strings/nulls)
X = X.apply(pd.to_numeric, errors="coerce")

# Remove invalid rows
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna()

# Align target
y = y.loc[X.index]

# Stop if no usable data
if X.empty or len(X) < 10:
    st.error("❌ Not enough clean data to train model. Please try later.")
    st.stop()

# -------------------- Scaling --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- Model Training --------------------
st.write("Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_scaled, y)
st.success("✅ Model trained")

# -------------------- Forecast --------------------
next_hour = model.predict(X_scaled[-1].reshape(1, -1))[0]

forecast_hours = 24
forecast = []
last_row = X_scaled[-1].copy()

for _ in range(forecast_hours):
    pred = model.predict(last_row.reshape(1, -1))[0]
    forecast.append(pred)
    last_row[0] = last_row[1]
    last_row[1] = last_row[2]
    last_row[2] = pred

# -------------------- Anomaly Detection --------------------
iso = IsolationForest(contamination=0.05, random_state=42)
df_all["anomaly"] = iso.fit_predict(df_all[["pm2_5"]])
anomalies = df_all[df_all["anomaly"] == -1]

# -------------------- Dashboard --------------------
st.title("🌫️ PM2.5 Forecast Dashboard")

st.metric("Last Observed PM2.5", f"{df_all['pm2_5'].iloc[-1]:.1f}")
st.metric("Next Hour Forecast", f"{next_hour:.1f}")

st.subheader("📈 24-Hour Forecast")
forecast_df = pd.DataFrame({
    "datetime": pd.date_range(
        df_all["datetime"].iloc[-1] + pd.Timedelta(hours=1),
        periods=forecast_hours,
        freq="H"
    ),
    "PM2.5": forecast
})
st.line_chart(forecast_df.set_index("datetime"))

st.subheader("🚨 Anomalies")
st.dataframe(anomalies[["datetime", "city", "pm2_5"]])

st.subheader("🧠 Feature Importance")
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
st.bar_chart(importance.set_index("Feature"))
