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
from datetime import date
from data.ckan_client import fetch_station_df_cached
from data.gtfs import get_trains_stations_info
from  calendar_utils.calendar_helpers import service_days_dict, get_month_day_weekday_dict, filter_df_by_dates
from data_agg.data_aggregators import station_avg_daily_activations_timeseries, get_daily_pattern_df, weekday_pattern_df
from visualization.plots import (plot_station_daily_avg_by_month, plot_daily_pattern_percent,plot_station_timeseries,plot_weekday_percent_pie)
from visualization.maps import station_location_on_map
from config import BASE, RESOURCE_IDS, YEARS, DEFAULT_TIMEOUT, TemporalInterval, train_station_names_eng

# ==================== Streamlit app ====================

st.set_page_config(page_title="Train Station Activations", layout="wide")
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #f2f2f2;
    }

    /* --- Sidebar styling --- */
    section[data-testid="stSidebar"] {
        background-color: #808080;   /* grey */
        border-right: 2px solid #5f5f5f;  /* darkgrey */
    }

    /* Optional: add a little padding inside the sidebar */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
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

st.title("Israeli Train Stations Departures Dashboard 🚆")

with st.expander("About this dashboard"):
    st.markdown(
        """
            This dashboard presents **passenger departures at Israel Railways stations**
            based on **official Ministry of Transport data** published via **data.gov.il**.
    
            The analysis includes **working days only** (excluding weekends and public holidays).
    
            **What you can explore:**
            - 📈 Average daily departures  
            - ⏰ Distribution of departures across the day  
            - 📅 Weekday patterns  
            """
    )


service_days = service_days_dict(YEARS)
weekdays_dict = get_month_day_weekday_dict(YEARS)

# Load GTFS stops (project-relative path)
stations = pd.read_csv("data/stops.txt")

# Build station selector options
train_stations_dict, train_station_names = get_trains_stations_info(stations)

# --- Sidebar controls
st.sidebar.header("Controls")

station_name = st.sidebar.selectbox(
    "Choose a station",
    sorted(list(train_station_names_eng.keys())),
    index=None
)


# Date range controls (only show if station is selected)
st.sidebar.markdown("***From Date***")
from_col1, from_col2 = st.sidebar.columns([1.1, 1])
with from_col1:
    from_year = st.selectbox("Year", YEARS, index=0, key="from_year")
with from_col2:
    from_month = st.selectbox("Month", list(range(1, 13)), index=0, key="from_month")

st.sidebar.markdown("***To Date***")
to_col1, to_col2 = st.sidebar.columns([1.1, 1])
with to_col1:
    to_year = st.selectbox("Year", YEARS, index=len(YEARS) - 1, key="to_year")
with to_col2:
    to_month = st.selectbox("Month", list(range(1, 13)), index=0, key="to_month")



if station_name is None:
    st.info("Select a station from the sidebar to display results.")
    st.stop()

# validate date range
from_date = date(int(from_year), int(from_month), 1)
to_date = date(int(to_year), int(to_month), 1)

if (to_year, to_month) < (from_year, from_month):
    st.warning("Date range isn't valid (To Date is earlier than From Date).")
    st.stop()


stop_code = train_station_names_eng[station_name]

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if st.sidebar.button("Load", type="primary"):
    st.session_state.loaded = True

if not st.session_state.loaded:
    st.info("Choose a date range and click **Load**.")
    st.stop()

# --- Fetch station data from API (updates on selection change)
station_df = fetch_station_df_cached(stop_code, RESOURCE_IDS, BASE, DEFAULT_TIMEOUT)


if station_df is None or station_df.empty:
    st.warning("Couldn't fetch station data - check spelling or try another station.")
    st.stop()


# --- Filter by dates (your helper)
station_df = filter_df_by_dates(station_df, from_date, to_date)


if station_df is None or station_df.empty:
    st.warning("Date range isn't available for this station (no data in selected period).")
    st.stop()

# 1) Station time series (month/day/year) + first chart
with st.container(border=True):
    interval = st.radio(
        "Temporal interval",
        options=TemporalInterval,
        horizontal=True,
        index=TemporalInterval.index("month"),
        key="temporal_interval",
    )

    ts_df = station_avg_daily_activations_timeseries(
        station_df,
        YEARS,
        service_days,
        temporal_interval=interval
    )
    # Trim to selected date range (works for all intervals)
    from_ts = pd.Timestamp(from_date)
    to_ts = pd.Timestamp(to_date)

    if interval != "year":
        ts_df = ts_df[(ts_df["date"] >= from_ts) & (ts_df["date"] <= to_ts)].reset_index(drop=True)

    if ts_df.empty:
        st.warning("No data available for the selected date range.")
        st.stop()


    fig = plot_station_timeseries(ts_df, station_name, temporal_interval=interval)
    st.plotly_chart(fig, use_container_width=True, key=f"ts_{stop_code}_{interval}")

# 2) Time-of-day distribution (percent)
daily_pattern_df = get_daily_pattern_df(station_df, YEARS, service_days)


fig2 = plot_daily_pattern_percent(daily_pattern_df, station_name=station_name, height=490)
st.plotly_chart(fig2, use_container_width=True, key="daily_pattern")

# 3) Weekday distribution (percent)
df_weekday = weekday_pattern_df(station_df, YEARS, service_days, weekdays_dict)


MAP_H = 430  # try 430–480 depending on your pie
PIE_SIZE = (5, 5)  # reduce pie height a bit

col1, col2 = st.columns([2, 1])

with col1:
    with st.container(border=True):
        st.subheader("Weekday distribution")
        fig3 = plot_weekday_percent_pie(df_weekday, figsize=PIE_SIZE)
        st.pyplot(fig3, clear_figure=True)

with col2:
    with st.container(border=True):
        st.subheader("Station location")
        m = station_location_on_map(stations, stop_code, station_name)
        st_folium(m, width=450, height=MAP_H, returned_objects=[], key="map")

