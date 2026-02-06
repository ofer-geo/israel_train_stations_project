import pandas as pd
import seaborn as sns
import calendar
import plotly.express as px
import matplotlib.pyplot as plt

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


