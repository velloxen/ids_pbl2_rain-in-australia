#!/usr/bin/env python3
"""Add engineered columns to the weatherAUS.csv dataset."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# NOTE: from the U.S. Department of Commerce National Oceanic and Atmospheric Administration
NINO_INDICES_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
# La Niña: NINO3.4 sea-surface temperature anomaly below -0.5 °C (NOAA/CPC convention).
LA_NINA_NINO34_THRESHOLD = -0.5

# Locations omitted from this dict are always "No" (no monsoon climate).
# TODO: source?
LOCATION_MONSOON_MONTHS: dict[str, tuple[int, int]] = {
    # Tropical north — wet/monsoon season roughly Nov–Apr
    "Darwin": (11, 4),
    "Katherine": (11, 4),
    "Cairns": (11, 4),
    "Townsville": (11, 4),
    "Uluru": (11, 4),
    # Sub-tropical 
    "Brisbane": (12, 3),
    "GoldCoast": (12, 3),
}

NEW_COLUMNS = [
    "TempRange",
    "MonsoonSeason",
    "WindSpeedDiff",
    "HumidityDiff",
    "TempDiff9am3pm",
    "PressureDiff",
    "DewPoint9am",
    "Season",
    "DaysSinceRain",
    "ConsecutiveRainDays",
    "LaNina",
]

CATEGORICAL_COLUMNS = {"MonsoonSeason", "Season", "LaNina"}


def is_monsoon_month(month: int, start_month: int, end_month: int) -> bool:
    """Return True if month falls within an inclusive range, including wrap-around."""
    if start_month <= end_month:
        return start_month <= month <= end_month
    return month >= start_month or month <= end_month


def monsoon_season(location: str, month: int) -> str:
    if location not in LOCATION_MONSOON_MONTHS:
        return "No"
    start_month, end_month = LOCATION_MONSOON_MONTHS[location]
    return "Yes" if is_monsoon_month(month, start_month, end_month) else "No"

# TODO: source?
def southern_hemisphere_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"

# TODO: source?
def dew_point_celsius(temp_c: float, humidity_pct: float) -> float:
    """Approximate dew point (°C) from temperature and relative humidity (Magnus formula)."""
    rh = humidity_pct / 100.0
    if rh <= 0:
        return math.nan
    alpha = math.log(rh) + (17.625 * temp_c) / (243.04 + temp_c)
    return (243.04 * alpha) / (17.625 - alpha)


def fetch_nino34_anomalies(cache_path: Path | None = None) -> pd.DataFrame:
    """Load monthly NINO3.4 anomalies from NOAA CPC (optionally cached on disk)."""
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["YearMonth"])
        return cached

    with urlopen(NINO_INDICES_URL, timeout=30) as response:
        raw = response.read().decode("utf-8")

    rows: list[dict[str, float | int]] = []
    for line in raw.splitlines():
        if not re.match(r"^\s*\d{4}\s+\d{1,2}\s", line):
            # Ensure lines start with year yyyy and month m
            # Original dataset is actually clean so this doesn't matter
            continue
        parts = line.split()
        year, month = int(parts[0]), int(parts[1])
        nino34_anom = float(parts[9])
        rows.append({"Year": year, "Month": month, "Nino34Anom": nino34_anom})

    indices = pd.DataFrame(rows)
    indices["YearMonth"] = pd.to_datetime(
        indices["Year"].astype(str) + "-" + indices["Month"].astype(str)
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        indices.to_csv(cache_path, index=False)

    return indices


def la_nina_flag(nino34_anom: float) -> str | None:
    if pd.isna(nino34_anom):
        # Original dataset is actually clean so this doesn't matter
        return None
    return "Yes" if nino34_anom < LA_NINA_NINO34_THRESHOLD else "No"

# TODO: (low-priority) why not just pd.isna(value)?
def _is_missing_rain(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))

# NOTE: this can generally be thought of as checking days since rain at the very end of the day (so that today counts as one of those days)
# NOTE: counts today as a day since rain. So every day on which there is rain has DaysSinceRain = 0
# NOTE: should we start the data with 1+ no-rain days, we set those to DaysSinceRain = NaN
#       so  Rainfall = [0, 0.6, 0, 0, 1, 0.2, ...] -> DaysSinceRain = [NaN, 0, 1, 2, 0, 0, ...]
#       but Rainfall = [1, 0] -> DaysSinceRain = [0, 1]
def _days_since_rain(rained: pd.Series) -> pd.Series:
    # e.g. rained = [0, 0.6, 0, 0, 1, 1.2, 3.6, ...]
    # seen_rain = True and days = 0 upon value = 0.6
    # results <- [NaN, 0, 1, 2, 0, 0, 0, ...]
    result: list[float] = []
    seen_rain = False
    days = math.nan
    for value in rained:
        if _is_missing_rain(value):
            # Propagate NaN and do NOT increase days...
            # TODO: should days be incremented anyway? I think so...
            # You could reconsider this as noting the start of a dry spell and each day subtracting the current date
            # So NaN propagates but days keeps incrementing
            # I think that gets the minimal amount of wrongness
            result.append(math.nan)
            continue
        # TODO: this tests value == 0, dataset defines RainedToday as rainfall >= 1mm
        if value:
            result.append(0.0)
            days = 0.0
            seen_rain = True
        elif not seen_rain:
            result.append(math.nan)
        else:
            days = days + 1.0
            result.append(days)
    return pd.Series(result, index=rained.index, dtype="Float64")


# NOTE: this ignores NaN from `rained` and perpetuates them
# e.g. Rainfall = [0, 0.6, 0, 0, 1, 1.2, 3.6, ...] -> ConsecutiveRainDays = [0, 0, 1, 0, 0, 1, 2, ...]
# TODO: that feels... wrong. We randomly have days of no rain with non-zero consecutive rain and days of rain with zero consecutive rain...
def _consecutive_rain_before(rained: pd.Series) -> pd.Series:
    result: list[float] = []
    streak = 0.0
    for value in rained:
        if _is_missing_rain(value):
            # Again, does NOT increase streak
            # TODO: should it? I think so...
            result.append(math.nan)
            continue
        # NOTE: update streak AFTER appending, so we do NOT consider today in the streak.
        # This is opposed to _days_since_rain where we could think of the streak as counting from the end of the day
        # Here we are considering counting from the very start of the day
        # TODO: should the two be normalized/unified?
        result.append(streak)
        streak = streak + 1.0 if value else 0.0
    return pd.Series(result, index=rained.index, dtype="Float64")


def add_rain_history_features(df: pd.DataFrame) -> pd.DataFrame:
    #  ~~TODO: [x] need to ensure df is sorted by date~~
    # It already was, just not in this function so it wasn't clear.
    # Moved it in.
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)
    # NOTE: (low-priority) you can just `df['Rainfall'].apply` (or something like that) this rather than all these `row['Rainfall']`s
    def rained_today(row: pd.Series) -> bool | None:
        if pd.notna(row["Rainfall"]):
            # TODO: [ ] the original dataset defines RainToday as "1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0"
            # TODO: [ ] (medium-priority) verify that df['RainToday'] == df['Rainfall'] >= 1
            return bool(row["Rainfall"] > 0)
        if pd.notna(row["RainToday"]):
            return row["RainToday"] == "Yes"
        return None

    df = df.copy()
    # e.g. [0.6, 0, 0, 0, 1, 0.2, ...]
    df["_rained"] = df.apply(rained_today, axis=1)

    days_since = []
    consecutive = []
    for _, location in df.groupby("Location", sort=False):
        location_rained = location["_rained"]
        days_since.append(_days_since_rain(location_rained))
        consecutive.append(_consecutive_rain_before(location_rained))

    df["DaysSinceRain"] = pd.concat(days_since).sort_index()
    df["ConsecutiveRainDays"] = pd.concat(consecutive).sort_index()
    return df.drop(columns="_rained")


def enrich(df: pd.DataFrame, climate_cache: Path | None = None) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]

    months = df["Date"].dt.month
    df["MonsoonSeason"] = [
        monsoon_season(location, month)
        for location, month in zip(df["Location"], months, strict=True)
    ]

    # Signed 3pm - 9am: positive = increase, negative = decrease
    df["WindSpeedDiff"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
    df["HumidityDiff"] = df["Humidity3pm"] - df["Humidity9am"]
    df["TempDiff9am3pm"] = df["Temp3pm"] - df["Temp9am"]
    df["PressureDiff"] = df["Pressure3pm"] - df["Pressure9am"]

    df["DewPoint9am"] = [
        dew_point_celsius(temp, humidity)
        if pd.notna(temp) and pd.notna(humidity)
        else math.nan
        for temp, humidity in zip(df["Temp9am"], df["Humidity9am"], strict=True)
    ]

    df["Season"] = df["Date"].dt.month.map(southern_hemisphere_season)

    df = add_rain_history_features(df)

    """
    The script downloads monthly NINO3.4 anomalies from NOAA CPC 
    and caches them in nino34_indices.csv. Rows are tagged Yes/No by month.
    """
    nino = fetch_nino34_anomalies(climate_cache)
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df = df.merge(
        nino[["YearMonth", "Nino34Anom"]],
        on="YearMonth",
        how="left",
    )
    df["LaNina"] = df["Nino34Anom"].map(la_nina_flag)
    df = df.drop(columns=["YearMonth", "Nino34Anom"])

    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich weatherAUS.csv with engineered columns.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("weatherAUS.csv"),
        help="Input CSV path (default: weatherAUS.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weatherAUS_enriched.csv"),
        help="Output CSV path (default: weatherAUS_enriched.csv)",
    )
    parser.add_argument(
        "--climate-cache",
        type=Path,
        default=Path("nino34_indices.csv"),
        help="Cache path for downloaded NINO3.4 index (default: nino34_indices.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, na_values="NA")
    row_count = len(df)

    df = enrich(df, climate_cache=args.climate_cache)
    df.to_csv(args.output, index=False, na_rep="NA")

    print(f"Wrote {row_count:,} rows to {args.output}")
    print(f"Added columns: {', '.join(NEW_COLUMNS)}")
    print("NA counts in new columns:")
    for col in NEW_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            continue
        print(f"  {col}: {df[col].isna().sum():,}")
    if "LaNina" in df.columns:
        print(f"  LaNina (missing index): {df['LaNina'].isna().sum():,}")
        print(f"  LaNina Yes: {(df['LaNina'] == 'Yes').sum():,}")
        print(f"  LaNina No: {(df['LaNina'] == 'No').sum():,}")


if __name__ == "__main__":
    main()
