import numpy as np
import pandas as pd
import json
import requests
import calendar
from datetime import date, timedelta
from pyluach.dates import GregorianDate
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit.components.v1 import html as st_html
from collections import defaultdict
import matplotlib.patheffects as pe




def non_service_days_by_month_israel(year: int) -> dict[int, list[int]]:
    """
    Returns: {month_number: [day_of_month, ...]}
    Includes: Fridays + Saturdays + selected Jewish holidays (Israel) + their eves.
    """
    # Holiday name matching: pyluach uses English holiday names like "Rosh Hashana", "Yom Kippur", etc. :contentReference[oaicite:3]{index=3}
    holidays_names = ['Rosh Hashana', 'Yom Kippur', 'Succos', 'Pesach', 'Shavuos']

    out = {}

    for month in range(1, 13):
        days = set()

        # 1) Fridays + Saturdays (weekday: 0=Mon ... 6=Sun)
        cal = calendar.monthcalendar(year, month)
        for week in cal:
            for wd in (calendar.FRIDAY, calendar.SATURDAY):
                d = week[wd]
                if d:
                    days.add(d)

        # 2) Jewish holidays (and eves) within this Gregorian month
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


def station_avg_daily_activations(df, year: int, month: int, non_service_days: dict):
    number_of_days = calendar.monthrange(year, month)[1]
    days = [f"day_{i}" for i in range(1, number_of_days+1) if i not in non_service_days[month]]
    day_sums = np.array([df.loc[df.month_key == month, day].sum() for day in days])

    return float(round(day_sums.mean(), 1))


def get_trains_stations_info(df_stations):
    df_train_stations = df_stations[
        df_stations["stop_code"].between(17000, 17200, inclusive="both")
    ]
    train_stations_dict = dict(zip(df_train_stations["stop_name"], df_train_stations["stop_code"]))
    train_station_names = list(train_stations_dict.keys())
    return train_stations_dict, train_station_names


def get_station_activations_info(stop_code):
    BASE = "https://data.gov.il/api/3/action/datastore_search"
    resource_id = "b2c6b258-4638-4f8e-bcad-600f0cdfb449"

    params = {
        "resource_id": resource_id,
        "filters": json.dumps({
            "StationId": stop_code
        }),
    }

    try:
        r = requests.get(BASE, params=params, timeout=30)
        r.raise_for_status()  # HTTP errors (4xx, 5xx)

        data = r.json()

        # CKAN-level failure
        if not data.get("success", False):
            print("Couldn't fetch station data - check spelling or try another name")
            return None

        records = data["result"]["records"]

        # Valid request but no data
        if not records:
            print("Couldn't fetch station data - check spelling or try another name")
            return None

        return pd.DataFrame(records)

    except requests.exceptions.RequestException as e:
        # Network / timeout / HTTP error
        print("Couldn't fetch station data - check spelling or try another name")
        return None


def plot_station_daily_avg_by_month(df, station_name):
    df_plot = df.reset_index()
    df_plot.columns = ["month", "value"]

    df_plot["month"] = df_plot["month"].apply(lambda m: calendar.month_abbr[int(m)])

    fig = px.bar(
        df_plot,
        x="month",
        y="value",
        labels={"month": "Month", "value": "Average daily departures"},
        color_discrete_sequence=["#7ED957"]
    )
    fig.update_traces(
        marker_line_color="#2E7D32",  # darker green
        marker_line_width=1
    )

    fig.update_layout(
        height=490,
        margin=dict(t=80, b=30),
        title={
            "text": f"<b>Monthly Average Daily Departures</b>",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 22},
        },
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Average daily departures</b>"
    )

    return fig  # <-- return instead of fig.show()


def station_location_on_map(df, stop_code, station_name):
    lat, lon = df.loc[df["stop_code"] == stop_code, ["stop_lat", "stop_lon"]].iloc[0]
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    folium.Marker(
        location=[lat, lon],
        popup=station_name,
        tooltip=station_name,
        icon=folium.Icon(color="red", icon="train", prefix="fa")
    ).add_to(m)

    return m


def summing_activations_time_of_day(df, year: int, month: int, day: str):
    day_dict = {time: 0 for time in station_df['LowOrPeakDescFull'].unique()}
    # 1) filter to January (month_key = 1) and sum day_1
    sums = (
        df
        .loc[df["month_key"] == month]
        .groupby("LowOrPeakDescFull")[day]
        .sum()
    )

    # 2) assign safely into the dict
    for k in day_dict:
        day_dict[k] += int(sums.get(k, 0))

    return day_dict


def get_daily_pattern_df(df, year: int, non_service_days: dict):
    general_dict = defaultdict(int)

    for month in range(1, 13):
        number_of_days = calendar.monthrange(2025, month)[1]
        days = [f"day_{i}" for i in range(1, number_of_days+1) if i not in non_service_days[month]]

        for day in days:
            day_dict = summing_activations_time_of_day(df, year, month, day)
            for k, v in day_dict.items():
                general_dict[k] += v

        general_dict = dict(general_dict)

    daily_pattern_df = pd.DataFrame(general_dict.items(), columns=["Time of day", "Activations"])

    daily_pattern_df["Percent"] = daily_pattern_df["Activations"] / daily_pattern_df["Activations"].sum() * 100

    return daily_pattern_df


def plot_daily_pattern_percent(daily_pattern_df, station_name=None, height=600, sort_by="time"):
    """
    Plot daily activation pattern as a percent bar chart.

    Parameters
    ----------
    daily_pattern_df : pd.DataFrame
        Must include columns: ['Time of day', 'Percent'] (and optionally 'Activations').
    station_name : str | None
        If provided, added to the title.
    height : int
        Plot height in pixels (useful for Streamlit layout tuning).
    sort_by : str
        'time' keeps the original time-of-day order (recommended),
        'percent_desc' sorts by percent descending.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    dfp = daily_pattern_df.copy()

    # ordering
    if sort_by == "percent_desc":
        dfp = dfp.sort_values("Percent", ascending=False)
    else:
        # keep original order as provided; if it got shuffled, you can re-sort by time prefix
        # assumes the label starts with "HH:MM - HH:MM"
        dfp["_start"] = dfp["Time of day"].str.slice(0, 5)
        dfp = dfp.sort_values("_start")
        dfp = dfp.drop(columns=["_start"])

    title = "<b>Passenger Activations by Time of Day (%)</b>"

    fig = px.bar(
        dfp,
        x="Time of day",
        y="Percent",
        labels={"Percent": "Share of activations (%)", "Time of day": ""},
        title=title,
        color_discrete_sequence=["#F28B82"]
    )
    fig.update_traces(
        marker_line_color="#B71C1C",  # darker red
        marker_line_width=1
    )

    fig.update_traces(
        text=dfp["Percent"].round(1).astype(str) + "%",
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
        cliponaxis=False,  # helps show labels above bars
    )

    fig.update_layout(
        height=height,
        margin=dict(t=80, b=80, l=40, r=40),
        yaxis=dict(ticksuffix="%", rangemode="tozero"),
        xaxis_tickangle=-25,

    )

    fig.update_layout(
        title={
            "text": f"<b>Passenger Departures by Time of Day (%)</b>",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 22},
        },
        xaxis_title="<b>Time of the day</b>",
        yaxis_title="<b>Share of departures</b>"
    )

    return fig

def weekday_pattern_df(df,year:int,non_service_days:dict):
    weekdays = ["Sunday","Monday","Tuesday","Wednesday","Thursday"]
    weekday_activations = {day:0 for day in weekdays}

    for month in range(1,13):
        number_of_days = calendar.monthrange(2025, month)[1]
        days = [f"day_{i}" for i in range(1,number_of_days+1) if i not in non_service_days[month]]
        for day in days:
            total_weekday = int(df[(df["year_key"]==year) & (station_df["month_key"]==month)][day].sum(skipna=True))
            weekday_activations[weekdays_dict[month][day]] += total_weekday

    weekday_df = pd.DataFrame([weekday_activations]).T
    weekday_df = weekday_df.rename(columns={0:'total'})
    weekday_df['percent'] = round(weekday_df['total'] / weekday_df['total'].sum() *100,1)

    return weekday_df


def plot_weekday_percent_pie(
    df,
    percent_col="percent",
    title="Share of Passenger Activations by Day of Week",
    figsize=(7, 7)
):
    """
    df:
        index  -> weekday names (Sunday, Monday, ...)
        column -> percent values
    """
    labels = df.index.astype(str).tolist()
    percents = df[percent_col].astype(float).tolist()

    sns.set_style("white")

    fig, ax = plt.subplots(figsize=figsize)

    wedges, _, autotexts = ax.pie(
        percents,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 14}
    )

    ax.legend(
        wedges,
        labels,
        title="Day of week",
        loc="center left",
        bbox_to_anchor=(1, 0.3),
        fontsize = 13,
        title_fontsize=14
    )

    plt.tight_layout()

    return fig

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
    out = {}

    for month in range(1, 13):
        days_in_month = calendar.monthrange(year, month)[1]
        month_dict = {}

        for day in range(1, days_in_month + 1):
            weekday_name = date(year, month, day).strftime("%A")
            month_dict[f"day_{day}"] = weekday_name

        out[month] = month_dict

    return out

st.set_page_config(page_title="Train Station Activations", layout="wide")

st.title("Train Station Activations")
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


year = 2025
non_service_days = non_service_days_by_month_israel(year)
weekdays_dict = get_month_day_weekday_dict(2025)

# Load stations once
stations = pd.read_csv("files/stops.txt")

train_stations_dict, train_station_names = get_trains_stations_info(stations)

# --- TOP: selector ---

station_name = st.selectbox("Choose a station", train_station_names, index=None)
if station_name is None:
    st.stop()
stop_code = train_stations_dict[station_name]

# --- BELOW: chart updates automatically when selection changes ---
station_df = get_station_activations_info(stop_code)

if station_df is None or station_df.empty:
    st.warning("Couldn't fetch station data - check spelling or try another station.")
else:
    monthly_avg = {
        month: station_avg_daily_activations(station_df, year, month, non_service_days)
        for month in range(1, 13)
    }

    monthly_avg_df = pd.DataFrame.from_dict(monthly_avg, orient="index", columns=["value"])
    monthly_avg_df.index.name = "month"

    daily_pattern_df = get_daily_pattern_df(station_df, 2025, non_service_days)

    df_weekday = weekday_pattern_df(station_df, 2025, non_service_days)


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
            st.markdown(
                """
                <div style="padding-bottom:70px;">
                """,
                unsafe_allow_html=True,
            )