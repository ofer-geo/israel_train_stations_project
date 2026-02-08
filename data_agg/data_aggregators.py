import pandas as pd
import numpy as np
from collections import defaultdict
from config import TemporalInterval
import calendar

def station_avg_daily_activations(df: pd.DataFrame, year: int, month: int, service_days: dict) -> float:
    """
    Compute mean daily activations for a given month,
    excluding weekends/holidays (non_service_days).
    """
    days = service_days[month]
    day_sums = np.array([df.loc[df.month_key == month, day].sum() for day in days])
    return float(round(day_sums.mean(), 1))

def station_avg_daily_activations_timeseries_old(
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

def station_avg_daily_activations_timeseries(
    df: pd.DataFrame,
    years: list[int],
    service_days_by_year: dict[int, dict[int, list[str]]],
    temporal_interval: TemporalInterval = "month",
) -> pd.DataFrame:
    """
    Compute average daily activations at different temporal intervals.

    Parameters
    ----------
    df : DataFrame
        Station dataframe with columns: year_key, month_key, day_1..day_31
    years : list[int]
        Years to consider
    service_days_by_year : dict
        {year: {month: ["day_1", "day_2", ...]}} (working days only)
    temporal_interval : "month" | "day" | "year"
        Output granularity.

    Returns
    -------
    DataFrame with columns:
        date | avg_daily_activations

    - month: date is YYYY-MM-01, value is avg daily activations in that month
    - day:   date is YYYY-MM-DD, value is activations on that day
    - year:  date is YYYY-01-01, value is avg daily activations in that year
    """
    rows = []

    # Only work on requested years (optional but helps)
    df = df[df["year_key"].isin(years)].copy()

    if df.empty:
        return pd.DataFrame(columns=["date", "avg_daily_activations"])

    if temporal_interval == "month":
        for year in years:
            for month in range(1, 13):
                days = service_days_by_year[year][month]

                month_df = df.loc[(df["year_key"] == year) & (df["month_key"] == month)]

                if month_df.empty or not days:
                    avg_val = None
                else:
                    day_sums = np.array(
                        [
                            pd.to_numeric(month_df[day], errors="coerce").sum()
                            for day in days
                        ],
                        dtype=float
                    )
                    avg_val = round(day_sums.mean(), 1)

                rows.append({
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "avg_daily_activations": avg_val
                })

        ts_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

        # Trim to actual available range in df (safe min/max using constructed dates)
        df_dates = pd.to_datetime(dict(year=df["year_key"], month=df["month_key"], day=1))
        df_min, df_max = df_dates.min(), df_dates.max()
        ts_df = ts_df[(ts_df["date"] >= df_min) & (ts_df["date"] <= df_max)]

        return ts_df

    # -----------------------
    # DAY: one row per service day (exact date)
    # -----------------------
    if temporal_interval == "day":
        for year in years:
            for month in range(1, 13):
                days = service_days_by_year[year][month]
                month_df = df.loc[(df["year_key"] == year) & (df["month_key"] == month)]

                if month_df.empty or not days:
                    continue

                # For each service day, sum across rows for that day column
                for day_col in days:
                    day_num = int(day_col.split("_")[1])

                    # skip invalid dates (e.g., day_31 in a 30-day month)
                    last_day = calendar.monthrange(year, month)[1]
                    if day_num > last_day:
                        continue

                    val = float(pd.to_numeric(month_df[day_col], errors="coerce").sum())

                    rows.append({
                        "date": pd.Timestamp(year=year, month=month, day=day_num),
                        "avg_daily_activations": round(val, 1)
                    })

        ts_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return ts_df

    # -----------------------
    # YEAR: avg daily activations across all service days in that year
    # -----------------------
    if temporal_interval == "year":
        for year in years:
            year_vals = []

            for month in range(1, 13):
                days = service_days_by_year[year][month]
                month_df = df.loc[(df["year_key"] == year) & (df["month_key"] == month)]

                if month_df.empty or not days:
                    continue

                # daily totals inside the month (only service days)
                day_sums = np.array(
                    [
                        pd.to_numeric(month_df[day], errors="coerce").sum()
                        for day in days
                    ],
                    dtype=float
                )
                year_vals.extend(day_sums.tolist())

            avg_val = round(float(np.mean(year_vals)), 1) if year_vals else None

            rows.append({
                "date": pd.Timestamp(year=year, month=1, day=1),
                "avg_daily_activations": avg_val
            })

        ts_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


        return ts_df

    raise ValueError("temporal_interval must be one of: 'month', 'day', 'year'")


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

    daily_pattern_df["Time of day"] = daily_pattern_df["Time of day"].str[:13]

    return daily_pattern_df


def weekday_pattern_df_old(df: pd.DataFrame, year: int, service_days: dict,weekdays_dict: dict) -> pd.DataFrame:
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


def weekday_pattern_df(
    df: pd.DataFrame,
    years: list[int],
    service_days_by_year: dict[int, dict[int, list[str]]],
    weekdays_by_year: dict[int, dict[int, dict[str, str]]],
) -> pd.DataFrame:
    """
    Aggregate activations by weekday (Sun–Thu) across multiple years,
    using service days only (no Fri/Sat/holidays).

    Returns a DataFrame:
        index: weekday name
        columns: total, percent
    """
    weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    weekday_activations = {d: 0 for d in weekdays}

    for year in years:
        for month in range(1, 13):
            days = service_days_by_year[year][month]

            # Filter once per (year, month) for speed
            month_df = df.loc[(df["year_key"] == year) & (df["month_key"] == month)]
            if month_df.empty:
                continue

            for day_col in days:
                # Sum that day across all service periods / rows
                total_day = month_df[day_col].sum(skipna=True)

                # Map day->weekday using your precomputed dict
                wd = weekdays_by_year[year][month][day_col]

                # Keep only Sun–Thu (in case something slips in)
                if wd in weekday_activations:
                    weekday_activations[wd] += int(total_day)

    weekday_df = (
        pd.DataFrame.from_dict(weekday_activations, orient="index", columns=["total"])
        .sort_index(key=lambda s: s.map({d: i for i, d in enumerate(weekdays)}))
    )
    total_sum = weekday_df["total"].sum()
    weekday_df["percent"] = (weekday_df["total"] / total_sum * 100).round(1) if total_sum else 0.0

    return weekday_df