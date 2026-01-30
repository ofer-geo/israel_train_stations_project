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

import numpy as np
import pandas as pd
import json
import requests
import calendar
from datetime import date
from pyluach.dates import GregorianDate
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium
from collections import defaultdict


# ==================== Calendar helpers ====================

def non_service_days_by_month_israel(year: int) -> dict[int, list[int]]:
    """
    Build a {month: [non_service_day_numbers]} mapping for Israel public transport.

    Non-service days include:
    - Fridays + Saturdays
    - Selected Jewish holidays (Israel) + their eves
    """
    holidays_names = ["Rosh Hashana", "Yom Kippur", "Succos", "Pesach", "Shavuos"]
    out: dict[int, list[int]] = {}

    for month in range(1, 13):
        days = set()

        # 1) Weekends (Fri/Sat)
        cal = calendar.monthcalendar(year, month)
        for week in cal:
            for wd in (calendar.FRIDAY, calendar.SATURDAY):
                d = week[wd]
                if d:
                    days.add(d)

        # 2) Holidays + eves within this Gregorian month
        last_day = calendar.monthrange(year, month)[1]
        for day in range(1, last_day + 1):
            g = GregorianDate(year, month, day)
            hname = g.to_heb().holiday(israel=True)
            if hname in holidays_names:
                days.add(day)
                if day != 1:
                    days.add(day - 1)

        out[month] = sorted(days)

    return out


def get_month_day_weekday_dict(year: int) -> dict[int, dict[str, str]]:
    """
    Returns:
    {
        month (1-12): {
            "day_1": "Monday",
            "day_2": "Tuesday",
            ...
        }
    }
    """
    out: dict[int, dict[str, str]] = {}

    for month in range(1, 13):
        days_in_month = calendar.monthrange(year, month)[1]
        month_dict: dict[str, str] = {}

        for day in range(1, days_in_month + 1):
            weekday_name = date(year, month, day).strftime("%A")
            month_dict[f"day_{day}"] = weekday_name

        out[month] = month_dict

    return out


# ==================== Data access ====================

def get_trains_stations_info(df_stations: pd.DataFrame):
    """
    Filter GTFS stops to (approx.) Israel Railways station stop_code range,
    then return:
    - dict: {stop_name -> stop_code}
    - list: station names (for UI selector)
    """
    df_train_stations = df_stations[df_stations["stop_code"].between(17000, 17200, inclusive="both")]
    train_stations_dict = dict(zip(df_train_stations["stop_name"], df_train_stations["stop_code"]))
    train_station_names = list(train_stations_dict.keys())
    return train_stations_dict, train_station_names


def get_station_activations_info(stop_code: int) -> pd.DataFrame | None:
    """
    Fetch station activations from data.gov.il CKAN Datastore API using StationId filter.
    Returns a DataFrame on success, otherwise None.
    """
    BASE = "https://data.gov.il/api/3/action/datastore_search"
    resource_id = "b2c6b258-4638-4f8e-bcad-600f0cdfb449"

    params = {
        "resource_id": resource_id,
        "filters": json.dumps({"StationId": stop_code}),
    }

    try:
        r = requests.get(BASE, params=params, timeout=30)
        r.raise_for_status()

        data = r.json()

        # CKAN-level failure
        if not data.get("success", False):
            print("Couldn't fetch station data - check spelling or try another name")
            return None

        records = data["result"]["records"]

        # Valid request but no matching rows
        if not records:
            print("Couldn't fetch station data - check spelling or try another name")
            return None

        return pd.DataFrame(records)

    except requests.exceptions.RequestException:
        print("Couldn't fetch station data - check spelling or try another name")
        return None


# ==================== Aggregations ====================

def station_avg_daily_activations(df: pd.DataFrame, year: int, month: int, non_service_days: dict) -> float:
    """
    Compute mean daily activations for a given month,
    excluding weekends/holidays (non_service_days).
    """
    number_of_days = calendar.monthrange(year, month)[1]
    days = [f"day_{i}" for i in range(1, number_of_days + 1) if i not in non_service_days[month]]
    day_sums = np.array([df.loc[df.month_key == month, day].sum() for day in days])
    return float(round(day_sums.mean(), 1))


def summing_activations_time_of_day(df: pd.DataFrame, year: int, month: int, day: str) -> dict:
    """
    Sum station activations for a specific month/day column, grouped by service period (LowOrPeakDescFull).
    Returns: {time_period_label -> total}
    """
    # NOTE: This uses the globally available station_df categories in your current setup.
    day_dict = {time: 0 for time in station_df["LowOrPeakDescFull"].unique()}

    sums = (
        df.loc[df["month_key"] == month]
          .groupby("LowOrPeakDescFull")[day]
          .sum()
    )

    for k in day_dict:
        day_dict[k] += int(sums.get(k, 0))

    return day_dict


def get_daily_pattern_df(df: pd.DataFrame, year: int, non_service_days: dict) -> pd.DataFrame:
    """
    Build a time-of-day distribution for the whole year:
    - aggregates across all working days
    - returns a DF with counts + percentage share
    """
    general_dict = defaultdict(int)

    for month in range(1, 13):
        number_of_days = calendar.monthrange(2025, month)[1]
        days = [f"day_{i}" for i in range(1, number_of_days + 1) if i not in non_service_days[month]]

        for day in days:
            day_dict = summing_activations_time_of_day(df, year, month, day)
            for k, v in day_dict.items():
                general_dict[k] += v

    daily_pattern_df = pd.DataFrame(general_dict.items(), columns=["Time of day", "Activations"])
    daily_pattern_df["Percent"] = daily_pattern_df["Activations"] / daily_pattern_df["Activations"].sum() * 100
    return daily_pattern_df


def weekday_pattern_df(df: pd.DataFrame, year: int, non_service_days: dict) -> pd.DataFrame:
    """
    Aggregate activations by weekday (Sun–Thu) for working days only.
    Returns a DF with total + percent.
    """
    weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    weekday_activations = {d: 0 for d in weekdays}

    for month in range(1, 13):
        number_of_days = calendar.monthrange(2025, month)[1]
        days = [f"day_{i}" for i in range(1, number_of_days + 1) if i not in non_service_days[month]]

        for day in days:
            # NOTE: This also references global station_df in your current code.
            total_weekday = int(df[(df["year_key"] == year) & (station_df["month_key"] == month)][day].sum(skipna=True))
            weekday_activations[weekdays_dict[month][day]] += total_weekday

    weekday_df = pd.DataFrame([weekday_activations]).T.rename(columns={0: "total"})
    weekday_df["percent"] = round(weekday_df["total"] / weekday_df["total"].sum() * 100, 1)
    return weekday_df


# ==================== Visualizations ====================

def plot_station_daily_avg_by_month(df: pd.DataFrame, station_name: str):
    """
    Plotly bar chart: monthly average daily departures (working days only).
    """
    df_plot = df.reset_index()
    df_plot.columns = ["month", "value"]
    df_plot["month"] = df_plot["month"].apply(lambda m: calendar.month_abbr[int(m)])

    fig = px.bar(
        df_plot,
        x="month",
        y="value",
        labels={"month": "Month", "value": "Average daily departures"},
        color_discrete_sequence=["#7ED957"],
    )
    fig.update_traces(marker_line_color="#2E7D32", marker_line_width=1)

    fig.update_layout(
        height=490,
        margin=dict(t=80, b=30),
        title={
            "text": "<b>Monthly Average Daily Departures</b>",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 22},
        },
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Average daily departures</b>",
    )
    return fig


def plot_daily_pattern_percent(daily_pattern_df: pd.DataFrame, station_name=None, height=600, sort_by="time"):
    """
    Plotly bar chart: distribution of departures by time-of-day (percentage share).
    """
    dfp = daily_pattern_df.copy()

    if sort_by == "percent_desc":
        dfp = dfp.sort_values("Percent", ascending=False)
    else:
        dfp["_start"] = dfp["Time of day"].str.slice(0, 5)
        dfp = dfp.sort_values("_start").drop(columns=["_start"])

    fig = px.bar(
        dfp,
        x="Time of day",
        y="Percent",
        labels={"Percent": "Share of activations (%)", "Time of day": ""},
        title="<b>Passenger Activations by Time of Day (%)</b>",
        color_discrete_sequence=["#F28B82"],
    )
    fig.update_traces(marker_line_color="#B71C1C", marker_line_width=1)

    fig.update_traces(
        text=dfp["Percent"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        cliponaxis=False,
    )

    fig.update_layout(
        height=height,
        margin=dict(t=80, b=80, l=40, r=40),
        yaxis=dict(ticksuffix="%", rangemode="tozero"),
        xaxis_tickangle=-25,
        title={
            "text": "<b>Passenger Departures by Time of Day (%)</b>",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 22},
        },
        xaxis_title="<b>Time of the day</b>",
        yaxis_title="<b>Share of departures</b>",
    )
    return fig


def plot_weekday_percent_pie(df: pd.DataFrame, percent_col="percent", figsize=(7, 7)):
    """
    Matplotlib pie chart: weekday distribution (percent labels inside, legend on the right).
    """
    labels = df.index.astype(str).tolist()
    percents = df[percent_col].astype(float).tolist()

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)

    wedges, _, _ = ax.pie(
        percents,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 14},
    )

    ax.legend(
        wedges,
        labels,
        title="Day of week",
        loc="center left",
        bbox_to_anchor=(1, 0.3),
        fontsize=13,
        title_fontsize=14,
    )

    plt.tight_layout()
    return fig


def station_location_on_map(df: pd.DataFrame, stop_code: int, station_name: str):
    """
    Folium map with a single marker at the selected station coordinates.
    """
    lat, lon = df.loc[df["stop_code"] == stop_code, ["stop_lat", "stop_lon"]].iloc[0]

    m = folium.Map(location=[lat, lon], zoom_start=15, tiles="OpenStreetMap")
    folium.Marker(
        location=[lat, lon],
        popup=station_name,
        tooltip=station_name,
        icon=folium.Icon(color="red", icon="train", prefix="fa"),
    ).add_to(m)

    return m


# ==================== Streamlit app ====================

st.set_page_config(page_title="Train Station Activations", layout="wide")

st.title("Israeli Train Stations Departures Dashboard")
st.markdown(
    """
    This dashboard presents **average daily passenger activations** at **Israel Railways train stations**.

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

# --- Global app settings/data that do not depend on station selection
year = 2025
non_service_days = non_service_days_by_month_israel(year)
weekdays_dict = get_month_day_weekday_dict(year)

# Load GTFS stops (project-relative path)
stations = pd.read_csv("files/stops.txt")

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
        month: station_avg_daily_activations(station_df, year, month, non_service_days)
        for month in range(1, 13)
    }
    monthly_avg_df = pd.DataFrame.from_dict(monthly_avg, orient="index", columns=["value"])
    monthly_avg_df.index.name = "month"

    # 2) Time-of-day distribution (percent)
    daily_pattern_df = get_daily_pattern_df(station_df, year, non_service_days)

    # 3) Weekday distribution (percent)
    df_weekday = weekday_pattern_df(station_df, year, non_service_days)

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
            st_folium(m, width=450, height=420, returned_objects=[], key="map")

        with st.container(border=True):
            st.subheader("Weekday distribution")
            fig3 = plot_weekday_percent_pie(df_weekday)
            st.pyplot(fig3, clear_figure=True)

            # Extra padding at the bottom of this panel (purely visual balance)
            st.markdown('<div style="padding-bottom:70px;"></div>', unsafe_allow_html=True)