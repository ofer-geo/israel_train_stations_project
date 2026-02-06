import pandas as pd
import numpy as np
from collections import defaultdict

def station_avg_daily_activations(df: pd.DataFrame, year: int, month: int, service_days: dict) -> float:
    """
    Compute mean daily activations for a given month,
    excluding weekends/holidays (non_service_days).
    """
    days = service_days[month]
    day_sums = np.array([df.loc[df.month_key == month, day].sum() for day in days])
    return float(round(day_sums.mean(), 1))

def station_avg_daily_activations_timeseries(
    df: pd.DataFrame,
    years: list[int],
    service_days_by_year: dict[int, dict[int, list[str]]],
) -> pd.DataFrame:
    """
    Compute mean daily activations per month across multiple years.

    Returns DataFrame with columns:
        date | avg_daily_activations
    Where `date` is the first day of each month (YYYY-MM-01).
    """
    rows = []

    for year in years:
        for month in range(1, 13):
            days = service_days_by_year[year][month]

            month_df = df.loc[
                (df["year_key"] == year) & (df["month_key"] == month)
            ]

            if month_df.empty or not days:
                avg_val = None
            else:
                day_sums = np.array(
                    [month_df[day].sum() for day in days],
                    dtype=float
                )
                avg_val = round(day_sums.mean(), 1)

            rows.append({
                "date": pd.Timestamp(year=year, month=month, day=1),
                "avg_daily_activations": avg_val
            })

    ts_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Trim rows outside the actual date range in df
    df_min = pd.Timestamp(int(df["year_key"].min()), int(df["month_key"].min()), 1)
    df_max = pd.Timestamp(int(df["year_key"].max()), int(df["month_key"].max()), 1)

    ts_df = ts_df[(ts_df["date"] >= df_min) & (ts_df["date"] <= df_max)]

    return ts_df


def _sum_time_of_day_for_day(
    df: pd.DataFrame,
    month: int,
    day: str
) -> dict:
    """
    Sum activations for one day column, grouped by time-of-day.
    Internal helper.
    """
    sums = (
        df.loc[df["month_key"] == month]
          .groupby("LowOrPeakDescFull")[day]
          .sum()
    )

    return sums.fillna(0).astype(int).to_dict()


def get_daily_pattern_df(
    df: pd.DataFrame,
    years: list[int],
    service_days_by_year: dict[int, dict[int, list[str]]]
) -> pd.DataFrame:
    """
    Build a time-of-day distribution across multiple years.
    Returns a DataFrame with total activations and percent share.
    """
    general_dict = defaultdict(int)

    for year in years:
        year_df = df[df["year_key"] == year]

        for month in range(1, 13):
            for day in service_days_by_year[year][month]:
                day_dict = _sum_time_of_day_for_day(
                    year_df,
                    month,
                    day
                )
                for k, v in day_dict.items():
                    general_dict[k] += v

    daily_pattern_df = pd.DataFrame(
        general_dict.items(),
        columns=["Time of day", "Activations"]
    )

    daily_pattern_df["Percent"] = (
        daily_pattern_df["Activations"]
        / daily_pattern_df["Activations"].sum()
        * 100
    )

    return daily_pattern_df


def weekday_pattern_df(df: pd.DataFrame, year: int, service_days: dict,weekdays_dict: dict) -> pd.DataFrame:
    """
    Aggregate activations by weekday (Sun–Thu) for working days only.
    Returns a DF with total + percent.
    """
    weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    weekday_activations = {d: 0 for d in weekdays}

    for month in range(1, 13):

        days = service_days[month]

        for day in days:
            # NOTE: This also references global station_df in your current code.
            total_weekday = int(df[(df["year_key"] == year) & (df["month_key"] == month)][day].sum(skipna=True))
            weekday_activations[weekdays_dict[month][day]] += total_weekday

    weekday_df = pd.DataFrame([weekday_activations]).T.rename(columns={0: "total"})
    weekday_df["percent"] = round(weekday_df["total"] / weekday_df["total"].sum() * 100, 1)
    return weekday_df