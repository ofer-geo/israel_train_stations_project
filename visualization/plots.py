import pandas as pd
import seaborn as sns
import calendar
import plotly.express as px
import matplotlib.pyplot as plt
from config import TemporalInterval

def plot_station_daily_avg_by_month_old(df: pd.DataFrame, station_name: str):
    """
    Plotly bar chart: monthly average daily departures (working days only).
    """
    df_plot = df.reset_index()
    df_plot.columns = ["month", "value"]
    df_plot["month"] = df_plot["month"].apply(lambda m: calendar.month_abbr[int(m)])


    fig = px.line(
        df_plot,
        x="month",
        y="value",
        labels={"month": "Month", "value": "Average daily departures"},
        color_discrete_sequence=["#7ED957"],
    )
    # Line + marker styling
    fig.update_traces(
        line=dict(color="#2E7D32", width=3),
        marker=dict(size=8, color="#7ED957", line=dict(width=1, color="#2E7D32"))
    )

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

def plot_station_daily_avg_by_month(df: pd.DataFrame, station_name: str):
    """
    Plotly line chart: monthly average daily departures (working days only).
    Expects columns: ['date', 'avg_daily_activations'].
    """
    y_min = df["avg_daily_activations"].min()
    y_max = df["avg_daily_activations"].max()

    # Add 20% padding
    padding = 0.2 * (y_max - y_min) if y_max > y_min else 0
    y_lower = max(0, y_min-(y_min/5))
    y_upper = y_max + (y_max/5)

    x_min = df["date"].min()
    x_max = df["date"].max()

    fig = px.line(
        df,
        x="date",
        y="avg_daily_activations",
        markers=True,
        labels={
            "date": "Month",
            "avg_daily_activations": "Average daily departures",
        },
    )

    fig.update_layout(
        height=490,
        margin=dict(t=80, b=30),
        title={
            "text": f"<b>Monthly Average Daily Departures – {station_name}</b>",
            "x": 0,
            "xanchor": "left",
            "font": {"size": 22},
        },
        xaxis_title="<b>Month</b>",
        yaxis_title="<b>Average daily departures</b>",
        yaxis=dict(range=[y_lower, y_upper]),
        xaxis=dict(range=[x_min, x_max]),
    )

    return fig


def plot_station_timeseries(
        df: pd.DataFrame,
        station_name: str,
        temporal_interval: TemporalInterval = "month",
        height: int = 490,
):
    """
    Plot station activations timeseries.

    - day/month -> line chart
    - year      -> bar chart
    """

    interval_meta = {
        "day": {
            "x_label": "Date",
            "title": f"<b>Daily Departures – {station_name}</b>",
            "y_label": "Departures",
            "tickformat": "%Y-%m-%d",
            "hoverformat": "%Y-%m-%d",
        },
        "month": {
            "x_label": "Month",
            "title": f"<b>Monthly Average Daily Departures – {station_name}</b>",
            "y_label": "Average daily departures",
            "tickformat": "%Y-%m",
            "hoverformat": "%Y-%m",
        },
        "year": {
            "x_label": "Year",
            "title": f"<b>Yearly Average Daily Departures – {station_name}</b>",
            "y_label": "Average daily departures",
            "tickformat": "%Y",
            "hoverformat": "%Y",
        },
    }

    meta = interval_meta[temporal_interval]

    # --- y range with padding
    y_min = float(df["avg_daily_activations"].min())
    y_max = float(df["avg_daily_activations"].max())

    y_lower = max(0, y_min - (y_min / 2 if y_min > 0 else 0))
    y_upper = y_max + (y_max / 3 if y_max > 0 else 1)

    # --- x range
    x_min = df["date"].min()
    x_max = df["date"].max()

    # ==============================
    # CHART TYPE SWITCH
    # ==============================
    if temporal_interval == "year":
        df_plot = df.copy()
        df_plot["year"] = df_plot["date"].dt.year.astype(str)

        fig = px.bar(
            df_plot,
            x="year",
            y="avg_daily_activations",
            labels={
                "year": meta["x_label"],
                "avg_daily_activations": meta["y_label"],
            },
            color_discrete_sequence=["#7ED957"],
        )

        fig.update_traces(marker_line_color="#2E7D32", width = 0.3,marker_line_width=1)
        fig.update_xaxes(type="category", categoryorder="category ascending")

    else:
        fig = px.line(
            df,
            x="date",
            y="avg_daily_activations",
            markers=True,
            labels={
                "date": meta["x_label"],
                "avg_daily_activations": meta["y_label"],
            },
        )

    # ==============================
    # COMMON LAYOUT
    # ==============================
    fig.update_layout(
        height=height,
        margin=dict(t=80, b=30),
        title={
            "text": meta["title"],
            "x": 0,
            "xanchor": "left",
            "font": {"size": 22},
        },
        xaxis_title=f"<b>{meta['x_label']}</b>",
        yaxis_title=f"<b>{meta['y_label']}</b>",
    )

    if temporal_interval != "year":
        fig.update_xaxes(range=[x_min, x_max], autorange=False, tickformat=meta["tickformat"])
    else:
        # don't set date range/tickformat on categorical axis
        pass

    fig.update_yaxes(range=[y_lower, y_upper], autorange=False)

    fig.update_traces(
        hovertemplate=(
            f"{meta['x_label']}: %{{x|{meta['hoverformat']}}}<br>"
            f"{meta['y_label']}: %{{y}}<extra></extra>"
        )
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


