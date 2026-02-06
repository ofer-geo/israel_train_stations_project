"""
Train Station Activations Dashboard (Israel, 2025)

A Streamlit dashboard that visualizes:
- Monthly average daily passenger activations (working days only)
- Station location on a map
- Time-of-day distribution (7 service periods)
- Weekday distribution (Sun–Thu)

Data source: data.gov.il (official public datasets)
"""

# ==================== Imports ====================


import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from data.ckan_client import get_station_activations_info
from data.gtfs import get_trains_stations_info
from  calendar_utils.calendar_helpers import service_days_dict, get_month_day_weekday_dict
from data_agg.data_aggregators import station_avg_daily_activations, get_daily_pattern_df, weekday_pattern_df
from visualization.plots import (plot_station_daily_avg_by_month, plot_daily_pattern_percent,plot_weekday_percent_pie)
from visualization.maps import station_location_on_map
from config import year

# ==================== Streamlit app ====================

st.set_page_config(page_title="Train Station Activations", layout="wide")
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #f2f2f2;
    }

    /* Containers (st.container, st.columns, etc.) */
    div[data-testid="stVerticalBlock"] > div {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
    }

    /* Optional: remove extra padding around page */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Israeli Train Stations Departures Dashboard")
st.markdown(
    """
    This dashboard presents **average daily passenger departures** at **Israel Railways train stations**.

    The analysis is based on **official public data from data.gov.il** and covers **the year 2025**.
    All figures reflect **working days only**, excluding **weekends (Fridays and Saturdays)** and **public holidays**.

    Select a station to explore:
    - 📊 **Monthly average daily departures**
    - 🗺️ **Geographic location** of the selected station
    - ⏰ **Distribution of departures by time of day**, grouped into **7 service periods**
    - 📅 **Weekday distribution of departures** across working days

    The dashboard is intended for **exploratory analysis and comparison between stations**.
    """
)



service_days = service_days_dict(year)
weekdays_dict = get_month_day_weekday_dict(year)

# Load GTFS stops (project-relative path)
stations = pd.read_csv("data/stops.txt")

# Build station selector options
train_stations_dict, train_station_names = get_trains_stations_info(stations)

# --- Station selector (top of page)
station_name = st.selectbox("Choose a station", train_station_names, index=None)
if station_name is None:
    st.stop()

stop_code = train_stations_dict[station_name]

# --- Fetch station data from API (updates on selection change)
station_df = get_station_activations_info(stop_code)

if station_df is None or station_df.empty:
    st.warning("Couldn't fetch station data - check spelling or try another station.")
else:
    # 1) Monthly averages (working days only)
    monthly_avg = {
        month: station_avg_daily_activations(station_df, year, month, service_days)
        for month in range(1, 13)
    }
    monthly_avg_df = pd.DataFrame.from_dict(monthly_avg, orient="index", columns=["value"])
    monthly_avg_df.index.name = "month"

    # 2) Time-of-day distribution (percent)
    daily_pattern_df = get_daily_pattern_df(station_df, year, service_days)

    # 3) Weekday distribution (percent)
    df_weekday = weekday_pattern_df(station_df, year, service_days, weekdays_dict)

    # --- Layout: two main columns (charts left, map+pie right)
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container(border=True):
            fig = plot_station_daily_avg_by_month(monthly_avg_df, station_name)
            st.plotly_chart(fig, use_container_width=True)

        with st.container(border=True):
            fig2 = plot_daily_pattern_percent(daily_pattern_df, station_name=station_name, height=490)
            st.plotly_chart(fig2, use_container_width=True, key="daily_pattern")

    with col2:
        with st.container(border=True):
            st.subheader("Station location")
            m = station_location_on_map(stations, stop_code, station_name)
            st_folium(m, width=450, height=375, returned_objects=[], key="map")

        with st.container(border=True):
            st.subheader("Weekday distribution")
            fig3 = plot_weekday_percent_pie(df_weekday, figsize=(7, 7))
            st.pyplot(fig3, clear_figure=True)

            # Extra padding at the bottom of this panel (purely visual balance)
            st.markdown('<div style="padding-bottom:40px;"></div>', unsafe_allow_html=True)