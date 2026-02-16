"""FastF1 data ingestion and cleaning for GrandPrixAI."""

from typing import Optional

import fastf1
import pandas as pd

from config import RACE_NAME, RACES, TOTAL_LAPS, YEAR


def _load_single_race(
    race_name: str,
    year: int,
    total_laps: int,
    top_n: int,
) -> pd.DataFrame:
    """Load and clean lap data from a single F1 race."""
    session = fastf1.get_session(year, race_name, "R")
    session.load(telemetry=False, laps=True, weather=True)

    results = session.results
    if results is None or len(results) == 0:
        raise ValueError(f"No results found for {year} {race_name}")

    top_results = results.head(top_n) if hasattr(results, "head") else results.iloc[:top_n]

    if "Abbreviation" in top_results.columns:
        top_abbrevs = top_results["Abbreviation"].astype(str).tolist()
    elif "DriverNumber" in top_results.columns:
        top_abbrevs = [str(d) for d in top_results["DriverNumber"].tolist()]
    else:
        driver_nums = list(session.drivers)[:top_n]
        top_abbrevs = [session.get_driver(str(d))["Abbreviation"] for d in driver_nums]

    laps = session.laps
    laps = laps.pick_drivers(top_abbrevs)
    laps = laps.pick_wo_box()
    laps = laps.pick_not_deleted()

    try:
        laps = laps.pick_track_status("1", how="equals")
    except Exception:
        if "Status" in laps.columns:
            laps = laps[laps["Status"] == "1"].copy()
        elif "TrackStatus" in laps.columns:
            laps = laps[laps["TrackStatus"] == "1"].copy()

    try:
        weather = laps.get_weather_data()
        if weather is not None and not weather.empty:
            laps = laps.reset_index(drop=True)
            weather = weather.reset_index(drop=True)
            temp_col = "TrackTemp" if "TrackTemp" in weather.columns else "AirTemp"
            if temp_col in weather.columns:
                laps = pd.concat(
                    [laps, weather[[temp_col]].rename(columns={temp_col: "TrackTemp"})],
                    axis=1,
                )
    except Exception:
        pass

    df = pd.DataFrame(laps)
    df = df.copy()

    if "LapTime" in df.columns and pd.api.types.is_timedelta64_dtype(df["LapTime"]):
        df["LapTime"] = df["LapTime"].dt.total_seconds()

    required = ["Driver", "LapNumber", "LapTime", "TyreLife", "Compound"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=required)

    if "TrackTemp" not in df.columns:
        df["TrackTemp"] = 40.0
    else:
        df["TrackTemp"] = df["TrackTemp"].fillna(df["TrackTemp"].median())

    df["FuelLoad"] = total_laps - df["LapNumber"].astype(int)

    median_lap = df.groupby("Driver")["LapTime"].transform("median")
    df = df[df["LapTime"] <= 1.07 * median_lap].copy()

    # Contextual outlier filter: drop laps >3.5s slower than lap-average (traffic/noise)
    lap_avg = df.groupby("LapNumber")["LapTime"].transform("mean")
    df = df[df["LapTime"] <= lap_avg + 3.5].copy()

    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].str.upper()
        df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]

    output_cols = ["Driver", "LapNumber", "LapTime", "TyreLife", "Compound", "TrackTemp", "FuelLoad"]
    df = df[[c for c in output_cols if c in df.columns]]

    return df.reset_index(drop=True)


def load_race_data(
    race_name: str = RACE_NAME,
    year: int = YEAR,
    total_laps: int = TOTAL_LAPS,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Load and clean lap data from a specific F1 race for the top N finishers.

    Features extracted: TyreLife, Compound, TrackTemp, FuelLoad (NOT LapNumber).
    Cleaning: TrackStatus='1' only (green flag), remove laps 7% slower than
    driver median, exclude in/out laps and deleted laps.

    Returns:
        Cleaned DataFrame with columns: Driver, LapNumber, LapTime, TyreLife,
        Compound, TrackTemp, FuelLoad. LapTime in seconds.
    """
    return _load_single_race(race_name, year, total_laps, top_n)


def load_all_race_data(
    races: Optional[list[tuple[str, int]]] = None,
    year: int = YEAR,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Load and concatenate lap data from multiple races for compound rebalancing.

    British GP is MEDIUM-heavy; Spanish GP adds more SOFT/HARD usage.
    Uses FuelLoad only (not LapNumber) to avoid multicollinearity.

    Args:
        races: List of (race_name, total_laps). Defaults to config.RACES.
        year: Championship year.
        top_n: Number of top finishers per race.

    Returns:
        Concatenated cleaned DataFrame.
    """
    if races is None:
        races = RACES
    dfs = []
    for race_name, total_laps in races:
        df = _load_single_race(race_name, year, total_laps, top_n)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
