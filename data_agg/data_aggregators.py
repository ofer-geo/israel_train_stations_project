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


def summing_activations_time_of_day(df: pd.DataFrame, year: int, month: int, day: str) -> dict:
    """
    Sum station activations for a specific month/day column, grouped by service period (LowOrPeakDescFull).
    Returns: {time_period_label -> total}
    """
    # NOTE: This uses the globally available station_df categories in your current setup.
    day_dict = {time: 0 for time in df["LowOrPeakDescFull"].unique()}

    sums = (
        df.loc[df["month_key"] == month]
          .groupby("LowOrPeakDescFull")[day]
          .sum()
    )

    for k in day_dict:
        day_dict[k] += int(sums.get(k, 0))

    return day_dict


def get_daily_pattern_df(df: pd.DataFrame, year: int, service_days: dict) -> pd.DataFrame:
    """
    Build a time-of-day distribution for the whole year:
    - aggregates across all working days
    - returns a DF with counts + percentage share
    """
    general_dict = defaultdict(int)

    for month in range(1, 13):


        days = service_days[month]

        for day in days:
            day_dict = summing_activations_time_of_day(df, year, month, day)
            for k, v in day_dict.items():
                general_dict[k] += v

    daily_pattern_df = pd.DataFrame(general_dict.items(), columns=["Time of day", "Activations"])
    daily_pattern_df["Percent"] = daily_pattern_df["Activations"] / daily_pattern_df["Activations"].sum() * 100
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