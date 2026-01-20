# ============================================================
# 🌫️ PM2.5 Nowcast & 24h Forecast Dashboard
# Environmental Data Science | Air Quality Monitoring
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="PM2.5 Forecast Dashboard", layout="wide")
st.title("🌫️ PM2.5 Nowcast & 24h Forecast Dashboard")
st.markdown("**Environmental Data Science | Air Quality Monitoring**")

# ------------------------------------------------------------
# CITY COORDINATES (INDIA)
# ------------------------------------------------------------
CITY_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867)
}

st.sidebar.header("🌍 Location Settings")
city = st.sidebar.selectbox("Select City", list(CITY_COORDS.keys()))
lat, lon = CITY_COORDS[city]

# ------------------------------------------------------------
# DATA FETCH WITH FALLBACK (OpenAQ → Open-Meteo)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_pm25_data(lat, lon, city):
    # -------------------- Try OpenAQ --------------------
    try:
        url = "https://api.openaq.org/v2/measurements"
        params = {
            "parameter": "pm25",
            "limit": 500,
            "sort": "desc",
            "order_by": "datetime"
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            records = data.get("results") or data.get("data")
            if records:
                rows = []
                for item in records:
                    rows.append({
                        "datetime": item.get("date", {}).get("utc"),
                        "pm25": item.get("value"),
                        "region": item.get("location", city)
                    })
                df = pd.DataFrame(rows)
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna()
                if not df.empty:
                    return df
    except:
        pass

    # -------------------- Fallback: Open-Meteo --------------------
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5",
            "timezone": "auto"
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        times = data["hourly"]["time"]
        pm25 = data["hourly"]["pm2_5"]
        df = pd.DataFrame({
            "datetime": pd.to_datetime(times),
            "pm25": pm25,
            "region": city
        })
        return df.dropna()
    except:
        return pd.DataFrame()

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
def create_features(df):
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Time features
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Lag features
    for lag in [1, 3, 6, 12, 24]:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)
    
    # Rolling features
    for w in [3, 6, 12, 24]:
        df[f"pm25_roll_mean_{w}"] = df["pm25"].rolling(w).mean()

    df = df.dropna()
    return df

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
raw_df = fetch_pm25_data(lat, lon, city)
if raw_df.empty:
    st.error("⚠️ Unable to load PM2.5 data from APIs.")
    st.stop()

feature_df = create_features(raw_df)

# ------------------------------------------------------------
# ANOMALY DETECTION USING ISOLATION FOREST
# ------------------------------------------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
feature_df["anomaly_flag"] = iso.fit_predict(feature_df[["pm25"]])
feature_df["anomaly_flag"] = feature_df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

region_df = feature_df[feature_df["region"] == city]

# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------
feature_columns = [
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_3", "pm25_roll_mean_6", "pm25_roll_mean_12", "pm25_roll_mean_24",
    "hour", "dayofweek", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]

X = region_df[feature_columns]
y = region_df["pm25"]

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X, y)

# ------------------------------------------------------------
# RECURSIVE FORECAST FUNCTION
# ------------------------------------------------------------
def recursive_forecast(df, horizon=24):
    current = df.iloc[-1:].copy()
    forecasts = []

    for _ in range(horizon):
        pred = model.predict(current[feature_columns])[0]
        next_time = current["datetime"].iloc[0] + pd.Timedelta(hours=1)

        forecasts.append({
            "datetime": next_time,
            "predicted_pm25": pred
        })

        # Update current row for next prediction
        current["pm25"] = pred
        current["hour"] = next_time.hour
        current["dayofweek"] = next_time.dayofweek
        current["is_weekend"] = int(current["dayofweek"].iloc[0] in [5,6])
        current["hour_sin"] = np.sin(2 * np.pi * current["hour"] / 24)
        current["hour_cos"] = np.cos(2 * np.pi * current["hour"] / 24)
        current["dow_sin"] = np.sin(2 * np.pi * current["dayofweek"] / 7)
        current["dow_cos"] = np.cos(2 * np.pi * current["dayofweek"] / 7)
        for lag in [1, 3, 6, 12, 24]:
            current[f"pm25_lag_{lag}"] = pred
        for w in [3, 6, 12, 24]:
            current[f"pm25_roll_mean_{w}"] = pred
        current["datetime"] = next_time

    return pd.DataFrame(forecasts)

forecast_df = recursive_forecast(region_df)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
last_row = region_df.iloc[-1]
col1, col2 = st.columns(2)
col1.metric("Last Observed PM2.5 (µg/m³)", f"{last_row.pm25:.2f}")
col2.metric("Last Updated", last_row.datetime.strftime("%Y-%m-%d %H:%M"))

# ------------------------------------------------------------
# FORECAST VISUALIZATION
# ------------------------------------------------------------
st.subheader("📈 24-Hour PM2.5 Forecast")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(region_df.tail(24)["datetime"], region_df.tail(24)["pm25"], label="Last 24h Actual")
ax.plot(forecast_df["datetime"], forecast_df["predicted_pm25"], "--", label="Next 24h Forecast")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_xlabel("Datetime")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ------------------------------------------------------------
# RECENT ANOMALIES
# ------------------------------------------------------------
st.subheader("🚨 Recent Anomalies")
anomalies = region_df[region_df["anomaly_flag"] == 1].tail(10)
if anomalies.empty:
    st.success("No recent anomalies detected")
else:
    st.dataframe(anomalies[["datetime", "pm25"]])

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------
st.subheader("🔍 Feature Importance")
importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))

# ------------------------------------------------------------
# INFO
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
**City:** {city}  
**Model:** Gradient Boosting Regressor  
**Primary Source:** OpenAQ  
**Fallback Source:** Open-Meteo  
**Forecast Horizon:** 24 hours
"""
)
