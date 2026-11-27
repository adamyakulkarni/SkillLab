"""Data loading, inspection, cleaning, and preparation pipeline (Slides 05-07)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import urllib.request

import numpy as np
import pandas as pd

from src.aqi_calculator import calculate_naqi, get_aqi_category

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HOURLY_AQI_CSV = DATA_DIR / "bengaluru_cpcb_hourly_aqi.csv"
POLLUTANT_CSV = DATA_DIR / "cpcb_bangalore_raw.csv"

SEASON_MAP = {
    12: "Winter (Dec-Feb)",
    1: "Winter (Dec-Feb)",
    2: "Winter (Dec-Feb)",
    3: "Pre-Monsoon (Mar-May)",
    4: "Pre-Monsoon (Mar-May)",
    5: "Pre-Monsoon (Mar-May)",
    6: "Monsoon (Jun-Sep)",
    7: "Monsoon (Jun-Sep)",
    8: "Monsoon (Jun-Sep)",
    9: "Monsoon (Jun-Sep)",
    10: "Post-Monsoon / Festive (Oct-Nov)",
    11: "Post-Monsoon / Festive (Oct-Nov)",
}

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# Physical plausibility caps used to flag sensor glitches (e.g. 999.99 / 65574 in raw CPCB logs)
POLLUTANT_VALID_BOUNDS: Dict[str, Tuple[float, float]] = {
    "PM25": (0.0, 500.0),
    "PM10": (0.0, 800.0),
    "NO2": (0.0, 500.0),
    "NO": (0.0, 500.0),
    "NOX": (0.0, 800.0),
    "SO2": (0.0, 500.0),
    "CO": (0.0, 50.0),
    "O3": (0.0, 500.0),
    "NH3": (0.0, 500.0),
    "RH": (0.0, 100.0),
    "TEMP": (5.0, 48.0),
    "WS": (0.0, 40.0),
}


@dataclass
class CleaningReport:
    """Captures Slide 07 inspection & before/after cleaning telemetry."""

    dataset_name: str
    dataset_type: str  # "hourly_aqi" | "pollutants"
    raw_rows: int
    raw_cols: int
    raw_missing_total: int
    raw_duplicates: int
    sensor_glitches_flagged: int
    cleaned_rows: int
    cleaned_missing_total: int
    date_start: str
    date_end: str
    stations: List[str]
    missing_by_col: Dict[str, int] = field(default_factory=dict)
    dtypes_summary: Dict[str, str] = field(default_factory=dict)
    glitch_examples: List[str] = field(default_factory=list)


def assign_season_features(df: pd.DataFrame, date_col: str = "Date_dt") -> pd.DataFrame:
    """Derive standardized Year, Month, DayOfWeek, Season features from datetime."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")
    out["Date_dt"] = dt
    out["Year"] = dt.dt.year
    out["Month"] = dt.dt.month
    out["Month_Name"] = out["Month"].map(MONTH_NAMES)
    out["DayOfWeek"] = dt.dt.day_name()
    out["Is_Weekend"] = dt.dt.day_of_week >= 5
    out["DayOfYear"] = dt.dt.day_of_year
    out["Season"] = out["Month"].map(SEASON_MAP)
    return out


def inspect_dataframe(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, str], int]:
    """Run Slide 07 Step 02 (Inspect): missing values, dtypes, duplicate count."""
    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    dupes = int(df.duplicated().sum())
    return missing, dtypes, dupes


def load_and_clean_hourly_aqi(
    filepath: Optional[Path] = None,
    drop_duplicates: bool = True,
    clip_invalid_aqi: bool = True,
    impute_missing_hours: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Load and clean the 2017-2023 CPCB/KSPCB Bengaluru 24-hour AQI dataset."""
    path = Path(filepath) if filepath else HOURLY_AQI_CSV
    raw_df = pd.read_csv(path)
    hour_cols = [c for c in raw_df.columns if ":" in c]

    missing_by_col, dtypes_summary, raw_dupes = inspect_dataframe(raw_df)
    raw_missing_total = int(raw_df.isna().sum().sum())

    # Detect out-of-bounds AQI readings
    numeric_hours = raw_df[hour_cols].apply(pd.to_numeric, errors="coerce")
    glitches_mask = (numeric_hours < 0) | (numeric_hours > 500)
    glitches_count = int(glitches_mask.sum().sum())

    df = raw_df.copy()
    for col in hour_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if drop_duplicates:
        df = df.drop_duplicates(subset=["Station Name", "Date"], keep="first")

    if clip_invalid_aqi:
        for col in hour_cols:
            df.loc[(df[col] < 0) | (df[col] > 500), col] = np.nan

    df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date_dt"]).sort_values(["Station Name", "Date_dt"]).reset_index(drop=True)

    # Keep raw daily mean before imputation for before/after comparison
    df["Raw_Daily_Mean_AQI"] = raw_df[hour_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    if impute_missing_hours:
        # Interpolate short intra-day gaps across hours within the same day
        df[hour_cols] = df[hour_cols].interpolate(axis=1, limit=4, limit_direction="both")

    df["Daily_Mean_AQI"] = df[hour_cols].mean(axis=1)
    df["Daily_Max_AQI"] = df[hour_cols].max(axis=1)
    df["Morning_Peak_AQI"] = df[["07:00:00", "08:00:00", "09:00:00", "10:00:00"]].mean(axis=1)
    df["Afternoon_AQI"] = df[["13:00:00", "14:00:00", "15:00:00", "16:00:00"]].mean(axis=1)
    df["Evening_Night_AQI"] = df[["19:00:00", "20:00:00", "21:00:00", "22:00:00", "23:00:00"]].mean(axis=1)

    df = df.dropna(subset=["Daily_Mean_AQI"]).reset_index(drop=True)
    df = assign_season_features(df, "Date_dt")
    df["Station"] = df["Station Name"].str.replace(" Bengaluru CPCB", "", regex=False).str.replace(" Bengaluru KSPCB", "", regex=False)
    df["AQI_Category"] = df["Daily_Mean_AQI"].apply(lambda x: get_aqi_category(x).label)

    cleaned_missing_total = int(df[hour_cols].isna().sum().sum())
    stations = sorted(df["Station"].dropna().unique().tolist())

    report = CleaningReport(
        dataset_name="Bengaluru CPCB/KSPCB Hourly AQI (2017-2023)",
        dataset_type="hourly_aqi",
        raw_rows=len(raw_df),
        raw_cols=len(raw_df.columns),
        raw_missing_total=raw_missing_total,
        raw_duplicates=raw_dupes,
        sensor_glitches_flagged=glitches_count,
        cleaned_rows=len(df),
        cleaned_missing_total=cleaned_missing_total,
        date_start=df["Date_dt"].min().strftime("%Y-%m-%d"),
        date_end=df["Date_dt"].max().strftime("%Y-%m-%d"),
        stations=stations,
        missing_by_col=missing_by_col,
        dtypes_summary=dtypes_summary,
        glitch_examples=[],
    )
    return raw_df, df, report


def _row_naqi(row: pd.Series) -> Tuple[float, str, str]:
    conc = {
        "PM2.5": row.get("PM25"),
        "PM10": row.get("PM10"),
        "NO2": row.get("NO2"),
        "SO2": row.get("SO2"),
        "CO": row.get("CO"),
        "O3": row.get("O3"),
        "NH3": row.get("NH3"),
    }
    aqi, dom, _, cat = calculate_naqi(conc, strict_min_pollutants=False)
    return (
        float(aqi) if aqi is not None else np.nan,
        dom if dom is not None else "Unknown",
        cat.label if aqi is not None else "Unknown",
    )


def load_and_clean_pollutants(
    filepath: Optional[Path] = None,
    drop_duplicates: bool = True,
    filter_sensor_glitches: bool = True,
    missing_strategy: str = "station_median",  # "station_median" | "dropna_core" | "none"
) -> Tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Load and clean the 2015-2018 CPCB Bengaluru multi-pollutant station dataset."""
    path = Path(filepath) if filepath else POLLUTANT_CSV
    raw_df = pd.read_csv(path)
    missing_by_col, dtypes_summary, raw_dupes = inspect_dataframe(raw_df)
    raw_missing_total = int(raw_df.isna().sum().sum())

    df = raw_df.copy()
    raw_dt_strings = df["date"].astype(str).str.strip() + " " + df["from_time"].astype(str).str.strip()
    df["Date_dt"] = pd.to_datetime(
        raw_dt_strings,
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=["Date_dt"]).copy()

    # Keep raw uncleaned PM2.5 copy for Slide 07 before/after diagnostic charts
    df["PM25_Raw"] = pd.to_numeric(df["PM25"], errors="coerce")

    glitch_examples: List[str] = []
    glitches_count = 0
    for col, (lo, hi) in POLLUTANT_VALID_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            mask = (df[col] < lo) | (df[col] > hi)
            cnt = int(mask.sum())
            glitches_count += cnt
            if cnt > 0 and len(glitch_examples) < 8:
                bad_vals = df.loc[mask, ["station", "date", col]].head(2)
                for _, brow in bad_vals.iterrows():
                    glitch_examples.append(f"{brow['station']} ({brow['date']}): {col}={brow[col]}")
            if filter_sensor_glitches and cnt > 0:
                df.loc[mask, col] = np.nan

    if drop_duplicates:
        df = df.drop_duplicates(subset=["station", "Date_dt"], keep="first")

    df = assign_season_features(df, "Date_dt")
    df["Hour"] = df["Date_dt"].dt.hour
    df["Station"] = df["station"].astype(str)

    core_pollutants = [c for c in ["PM25", "PM10", "NO2", "SO2", "CO", "O3"] if c in df.columns]

    if missing_strategy == "station_median":
        for col in core_pollutants:
            med_sm = df.groupby(["Station", "Month"])[col].transform("median")
            med_s = df.groupby("Station")[col].transform("median")
            df[col] = df[col].fillna(med_sm).fillna(med_s).fillna(df[col].median())
    elif missing_strategy == "dropna_core":
        df = df.dropna(subset=["PM25", "NO2", "SO2", "CO"], how="any")

    # Compute CPCB National AQI vectorized via apply on unique rounded or direct rows
    aqi_res = df.apply(_row_naqi, axis=1, result_type="expand")
    df["AQI"] = aqi_res[0]
    df["Dominant_Pollutant"] = aqi_res[1]
    df["AQI_Category"] = aqi_res[2]

    df = df.dropna(subset=["AQI"]).sort_values(["Station", "Date_dt"]).reset_index(drop=True)

    cleaned_missing_total = int(df[core_pollutants].isna().sum().sum())
    stations = sorted(df["Station"].dropna().unique().tolist())

    report = CleaningReport(
        dataset_name="CPCB Bengaluru Multi-Pollutant Stations (2015-2018)",
        dataset_type="pollutants",
        raw_rows=len(raw_df),
        raw_cols=len(raw_df.columns),
        raw_missing_total=raw_missing_total,
        raw_duplicates=raw_dupes,
        sensor_glitches_flagged=glitches_count,
        cleaned_rows=len(df),
        cleaned_missing_total=cleaned_missing_total,
        date_start=df["Date_dt"].min().strftime("%Y-%m-%d"),
        date_end=df["Date_dt"].max().strftime("%Y-%m-%d"),
        stations=stations,
        missing_by_col=missing_by_col,
        dtypes_summary=dtypes_summary,
        glitch_examples=glitch_examples,
    )
    return raw_df, df, report


def load_custom_csv(filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Load and clean an arbitrary user CSV containing Date + pollutant/AQI columns."""
    raw_df = pd.read_csv(filepath)
    missing_by_col, dtypes_summary, raw_dupes = inspect_dataframe(raw_df)
    raw_missing_total = int(raw_df.isna().sum().sum())

    df = raw_df.copy().drop_duplicates()
    date_col = next((c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None)
    if date_col is None:
        raise ValueError("Custom CSV must contain a Date or Timestamp column.")

    df["Date_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["Date_dt"]).copy()
    df = assign_season_features(df, "Date_dt")

    # Normalize common pollutant column aliases
    col_map = {}
    for c in df.columns:
        cu = c.strip().upper().replace(".", "")
        if cu == "PM25":
            col_map[c] = "PM25"
        elif cu == "PM10":
            col_map[c] = "PM10"
        elif cu == "NO2":
            col_map[c] = "NO2"
        elif cu == "SO2":
            col_map[c] = "SO2"
        elif cu == "CO":
            col_map[c] = "CO"
        elif cu == "O3":
            col_map[c] = "O3"
        elif cu in ("STATION", "LOCATION", "CITY", "AREA"):
            col_map[c] = "Station"
        elif cu == "AQI":
            col_map[c] = "AQI"
    df = df.rename(columns=col_map)
    if "Station" not in df.columns:
        df["Station"] = "Bengaluru (Custom)"

    for col in ["PM25", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "AQI" not in df.columns or df["AQI"].isna().all():
        aqi_res = df.apply(_row_naqi, axis=1, result_type="expand")
        df["AQI"] = aqi_res[0]
        df["Dominant_Pollutant"] = aqi_res[1]
    else:
        df["Dominant_Pollutant"] = "AQI"

    df = df.dropna(subset=["AQI"]).sort_values("Date_dt").reset_index(drop=True)
    df["AQI_Category"] = df["AQI"].apply(lambda x: get_aqi_category(x).label)

    report = CleaningReport(
        dataset_name=f"Custom CSV: {Path(filepath).name}",
        dataset_type="pollutants",
        raw_rows=len(raw_df),
        raw_cols=len(raw_df.columns),
        raw_missing_total=raw_missing_total,
        raw_duplicates=raw_dupes,
        sensor_glitches_flagged=0,
        cleaned_rows=len(df),
        cleaned_missing_total=int(df.isna().sum().sum()),
        date_start=df["Date_dt"].min().strftime("%Y-%m-%d"),
        date_end=df["Date_dt"].max().strftime("%Y-%m-%d"),
        stations=sorted(df["Station"].astype(str).unique().tolist()),
        missing_by_col=missing_by_col,
        dtypes_summary=dtypes_summary,
    )
    return raw_df, df, report


def fetch_live_open_meteo_bengaluru(days_past: int = 7, days_forecast: int = 5) -> pd.DataFrame:
    """Fetch live & forecast hourly air quality for Bengaluru via Open-Meteo (Slide 10 Future Scope)."""
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        "?latitude=12.9716&longitude=77.5946"
        "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust"
        f"&past_days={days_past}&forecast_days={days_forecast}&timezone=Asia%2FKolkata"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "InfiniteLoop-Bengaluru-AQI/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "Date_dt": pd.to_datetime(hourly["time"]),
        "PM25": pd.to_numeric(hourly["pm2_5"], errors="coerce"),
        "PM10": pd.to_numeric(hourly["pm10"], errors="coerce"),
        # Open-Meteo CO is in ug/m3; convert to mg/m3 for CPCB NAQI consistency
        "CO": pd.to_numeric(hourly["carbon_monoxide"], errors="coerce") / 1000.0,
        "NO2": pd.to_numeric(hourly["nitrogen_dioxide"], errors="coerce"),
        "SO2": pd.to_numeric(hourly["sulphur_dioxide"], errors="coerce"),
        "O3": pd.to_numeric(hourly["ozone"], errors="coerce"),
        "Dust": pd.to_numeric(hourly["dust"], errors="coerce"),
    })
    df["Station"] = "Bengaluru (Open-Meteo Regional Grid)"
    aqi_res = df.apply(_row_naqi, axis=1, result_type="expand")
    df["AQI"] = aqi_res[0]
    df["Dominant_Pollutant"] = aqi_res[1]
    df["AQI_Category"] = aqi_res[2]
    return df.dropna(subset=["AQI"]).reset_index(drop=True)
