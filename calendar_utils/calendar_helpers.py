import calendar
from datetime import date
from pyluach.dates import GregorianDate
import pandas as pd

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

def service_days_dict(years: list[int])-> dict[int, list[str]]:
    """
    Build a mapping of service days (working days) for each month.

    Returns:
        {
            month (1–12): ["day_1", "day_2", ...]
        }

    Service days exclude weekends and public holidays,
    as defined by `non_service_days_by_month_israel`.
    """
    # Get non-service days (weekends + holidays) per month
    service_days_by_year = {}

    for year in years:

        non_service_days = non_service_days_by_month_israel(year)
        service_days = {}
        # Iterate through all months of the year
        for month in range(1, 13):
            # Number of days in the current month
            number_of_days = calendar.monthrange(year, month)[1]
            # Keep only working days in "day_X" format
            days = [
                f"day_{i}"
                for i in range(1, number_of_days + 1)
                if i not in non_service_days[month]
            ]
            service_days[month] = days
        service_days_by_year[year] = service_days

    return service_days_by_year


def get_month_day_weekday_dict(years:list) -> dict[int, dict[str, str]]:
    """
        Returns:
        {
            year: {
                month (1–12): {
                    "day_1": "Monday",
                    "day_2": "Tuesday",
                    ...
                }
            }
        }
    """
    out_by_year: dict[int, dict[int, dict[str, str]]] = {}

    for year in years:
        year_dict: dict[int, dict[str, str]] = {}

        for month in range(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            month_dict: dict[str, str] = {}

            for day in range(1, days_in_month + 1):
                weekday_name = date(year, month, day).strftime("%A")
                month_dict[f"day_{day}"] = weekday_name

            year_dict[month] = month_dict
        out_by_year[year] = year_dict

    return out_by_year

def filter_df_by_dates(df: pd.DataFrame, from_date, to_date):
    from_ts = pd.Timestamp(from_date)
    to_ts   = pd.Timestamp(to_date)

    df = df[
        (df["date"] >= from_ts) &
        (df["date"] <= to_ts)
    ]
    return df