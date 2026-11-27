"""Seasonal pattern discovery, Diwali festival analysis, and Bengaluru cause engine."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

DIWALI_DATES: Dict[int, str] = {
    2015: "2015-11-11",
    2016: "2016-10-30",
    2017: "2017-10-19",
    2018: "2018-11-07",
    2019: "2019-10-27",
    2020: "2020-11-14",
    2021: "2021-11-04",
    2022: "2022-10-24",
    2023: "2023-11-12",
    2024: "2024-10-31",
    2025: "2025-10-20",
    2026: "2026-11-08",
}

SEASON_ORDER = [
    "Winter (Dec-Feb)",
    "Pre-Monsoon (Mar-May)",
    "Monsoon (Jun-Sep)",
    "Post-Monsoon / Festive (Oct-Nov)",
]

# Scientific breakdown of firework-borne heavy metals & compounds (Oct-Nov spikes)
FIREWORK_HEAVY_METALS: List[Dict[str, str]] = [
    {
        "element": "Barium (Ba)",
        "source": "Barium nitrate / chlorate (green fireworks & oxidizer)",
        "impact": "Forms dense sub-micron PM2.5 aerosols; causes airway irritation & cardiovascular stress.",
    },
    {
        "element": "Strontium (Sr)",
        "source": "Strontium carbonate / nitrate (deep red flames)",
        "impact": "Surges 20x-40x above baseline during Diwali nights; lingers in fine respirable dust.",
    },
    {
        "element": "Potassium (K)",
        "source": "Potassium nitrate / perchlorate (black powder propellant)",
        "impact": "Primary chemical tracer of firecracker & biomass smoke; drives acute PM2.5 mass peaks.",
    },
    {
        "element": "Copper (Cu)",
        "source": "Copper chloride / oxide (blue sparkles)",
        "impact": "Catalyzes reactive oxygen species (ROS) in lung tissue, worsening asthma & bronchitis.",
    },
    {
        "element": "Aluminium (Al) & Magnesium (Mg)",
        "source": "Metallic flash powder & white sparklers",
        "impact": "Creates persistent ultra-fine metallic haze that stays suspended during calm nights.",
    },
    {
        "element": "Lead (Pb) & Trace Metals",
        "source": "Low-grade cracker stabilizers, casings & detonators",
        "impact": "Neurotoxic heavy metal tracked in India's National AQI; accumulates via surface dust resuspension.",
    },
]

SEASONAL_CAUSE_PROFILES: List[Dict[str, str]] = [
    {
        "season": "Post-Monsoon / Festive (Oct-Nov)",
        "severity": "High Spikes (Poor to Severe during Festivals)",
        "key_pollutants": "PM2.5, PM10, SO2, NO2 + Heavy Metals (Ba, Sr, K, Cu, Al, Pb)",
        "why_it_happens": (
            "As southwest monsoon rains withdraw, wind speeds drop and cooler nights compress the "
            "planetary boundary layer. Festival fireworks (especially Diwali) inject sudden bursts of "
            "sub-micron PM2.5, SO2, and toxic heavy metals (Barium, Strontium, Potassium, Copper, Lead) "
            "that remain trapped near ground level for days."
        ),
    },
    {
        "season": "Winter (Dec-Feb)",
        "severity": "Sustained Worst Baseline (Moderate to Very Poor mornings/nights)",
        "key_pollutants": "PM2.5, CO, NO2, Organic Carbon",
        "why_it_happens": (
            "Cooler night temperatures (12-16 C) create thermal inversions that cap vertical dispersion. "
            "Zero rainfall washout means traffic exhaust and localized burning of dry leaves, municipal "
            "garbage, and roadside wood/coal fires accumulate sharply between 06:00-09:00 and 20:00-23:00."
        ),
    },
    {
        "season": "Pre-Monsoon / Dry Summer (Mar-May)",
        "severity": "High Coarse Dust & Ozone (Moderate-Poor in Mar-Apr; drops in May)",
        "key_pollutants": "PM10 (road/construction dust), O3 (photochemical), NO2",
        "why_it_happens": (
            "Dry soil and intense traffic resuspend coarse road and construction dust (PM10), while "
            "strong solar radiation converts vehicular NOx and VOCs into ground-level Ozone (O3). "
            "Pre-monsoon thunderstorms in late May start washing particulates out."
        ),
    },
    {
        "season": "Monsoon (Jun-Sep)",
        "severity": "Cleanest Period (Good to Satisfactory)",
        "key_pollutants": "Low PM2.5 / PM10; localized traffic NO2/CO persists near junctions",
        "why_it_happens": (
            "Frequent Southwest Monsoon rainfall wet-scavenges airborne particles and suppresses road dust, "
            "while stronger westerly winds disperse gaseous emissions rapidly across the Bengaluru plateau."
        ),
    },
]

BENGALURU_AREA_PROFILES: Dict[str, Dict[str, str]] = {
    "City Railway Station": {
        "zone": "Transit & Dense Urban Core (Majestic)",
        "drivers": "Heavy diesel buses, railway hub, chronic traffic idling, street-canyon wind trapping (CO, NO2, PM2.5).",
    },
    "CRS": {
        "zone": "City Railway Station (Majestic Core)",
        "drivers": "Heavy diesel buses, railway transit, chronic traffic congestion (CO, NO2, PM10).",
    },
    "Silk Board": {
        "zone": "Major Traffic Bottleneck (ORR / Hosur Rd)",
        "drivers": "Severe stop-and-go congestion, heavy diesel freight, metro/flyover construction dust (CO, NO2, PM2.5, PM10).",
    },
    "SILKBOARD": {
        "zone": "Silk Board Junction (ORR / Hosur Rd)",
        "drivers": "Severe stop-and-go traffic congestion, diesel exhaust, construction dust (CO, NO2, PM10).",
    },
    "Peenya": {
        "zone": "Major Industrial Belt (North-West Bengaluru)",
        "drivers": "Industrial boilers, diesel generator sets, manufacturing/metal-finishing units & freight trucks (SO2, PM2.5, NO2).",
    },
    "Shivapura Peenya": {
        "zone": "Peenya Industrial Periphery",
        "drivers": "Small/medium industrial fuel burning, workshops, diesel cargo movement (PM2.5, SO2, PM10).",
    },
    "Bapuji Nagar": {
        "zone": "Mysuru Road Transit & Mixed Commercial/Industrial",
        "drivers": "Highway bus/truck traffic, small-scale fuel burning, commercial waste burning (PM2.5, CO, NO2).",
    },
    "RVCE-Mailasandra": {
        "zone": "Mysuru Road / Outer Peripheral Corridor",
        "drivers": "Inter-city highway diesel trucks, stone/construction haulage & dry road dust (PM10, PM2.5, NO2).",
    },
    "Jayanagar 5th Block": {
        "zone": "Dense Residential South Bengaluru",
        "drivers": "Evening dry-leaf/waste burning, festive firecracker clusters, arterial traffic (PM2.5, heavy metals during Diwali).",
    },
    "JAYANAGAR": {
        "zone": "Jayanagar Residential Hub",
        "drivers": "Festive fireworks, evening leaf/waste burning, urban traffic (PM2.5, NO2, CO).",
    },
    "BWSSB Kadabesanahalli": {
        "zone": "Outer Ring Road (IT & Construction Corridor)",
        "drivers": "High-rise/metro construction dust, tech-park commuter traffic, backup DG sets (PM10, PM2.5, NO2).",
    },
    "BWSSB": {
        "zone": "Kadabesanahalli / ORR Corridor",
        "drivers": "Construction dust, ORR traffic, backup diesel generators.",
    },
    "BTM Layout": {
        "zone": "Residential & South Arterial Corridor",
        "drivers": "Overflow Silk Board/Bannerghatta traffic, evening biomass/waste burning, festive fireworks (PM2.5, NO2).",
    },
    "BTM": {
        "zone": "BTM Layout",
        "drivers": "Arterial commuter traffic, localized waste burning, festive firework smoke.",
    },
    "Hebbal": {
        "zone": "North Gateway / Airport Highway Corridor",
        "drivers": "Airport expressway & ORR heavy vehicles, flyover bottleneck, road dust (PM10, NO2).",
    },
    "HEBBAL": {
        "zone": "Hebbal Flyover / Bellary Rd",
        "drivers": "Highway diesel traffic, construction dust, NO2 & PM10 spikes.",
    },
    "Jigani": {
        "zone": "South Industrial Estate",
        "drivers": "Pharmaceutical/manufacturing boilers, industrial fuel combustion, cargo trucks (SO2, PM2.5).",
    },
    "KAVIKA": {
        "zone": "Mysuru Road (KAVIKA Industrial/Traffic)",
        "drivers": "Dense Mysuru Road diesel traffic & nearby industrial fuel combustion (high NO2 & PM10).",
    },
    "NIHMANS": {
        "zone": "Hosur Road Institutional / Park Buffer",
        "drivers": "Moderate arterial traffic buffered by green cover; photochemical O3 & regional PM2.5.",
    },
    "Hombegowda Nagar": {
        "zone": "Central-South Institutional / Residential",
        "drivers": "Moderate urban background with arterial traffic and winter inversion accumulation.",
    },
    "Sanegurava Halli": {
        "zone": "West Bengaluru Residential / Mixed",
        "drivers": "Lower baseline aided by local dispersion, though winter & Diwali spikes still elevate PM2.5.",
    },
    "SGHALLI": {
        "zone": "Sanegurava Halli",
        "drivers": "Lower baseline residential zone affected seasonally by winter inversions & festival smoke.",
    },
    "Kasturi Nagar": {
        "zone": "East Bengaluru / Old Madras Rd Corridor",
        "drivers": "Commuter traffic & ORR/NGEF construction dust (PM10, NO2).",
    },
}

# Approximate geographic coordinates (latitude, longitude) of Bengaluru CPCB/KSPCB stations
BENGALURU_STATION_COORDS: Dict[str, Tuple[float, float]] = {
    "Peenya": (13.0285, 77.5197),
    "Shivapura Peenya": (13.0230, 77.5060),
    "Hebbal": (13.0354, 77.5988),
    "HEBBAL": (13.0354, 77.5988),
    "Sanegurava Halli": (12.9900, 77.5440),
    "SGHALLI": (12.9900, 77.5440),
    "Kasturi Nagar": (13.0040, 77.6530),
    "City Railway Station": (12.9767, 77.5713),
    "CRS": (12.9767, 77.5713),
    "Bapuji Nagar": (12.9568, 77.5397),
    "BWSSB Kadabesanahalli": (12.9473, 77.6983),
    "BWSSB": (12.9473, 77.6983),
    "Hombegowda Nagar": (12.9375, 77.5949),
    "NIHMANS": (12.9375, 77.5949),
    "RVCE-Mailasandra": (12.9250, 77.5030),
    "Jayanagar 5th Block": (12.9208, 77.5842),
    "JAYANAGAR": (12.9208, 77.5842),
    "BTM Layout": (12.9166, 77.6101),
    "BTM": (12.9166, 77.6101),
    "Silk Board": (12.9170, 77.6210),
    "SILKBOARD": (12.9170, 77.6210),
    "KAVIKA": (12.9560, 77.5430),
    "Jigani": (12.7930, 77.6350),
}


def compute_monthly_summary(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Compute monthly mean, median, p95, max, and sample counts."""
    if metric_col not in df.columns or "Month" not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby("Month")[metric_col]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            P95=lambda s: np.nanpercentile(s.dropna(), 95) if len(s.dropna()) else np.nan,
            Max="max",
        )
        .reset_index()
    )
    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    grouped["Month_Name"] = grouped["Month"].map(month_labels)
    return grouped.sort_values("Month").reset_index(drop=True)


def compute_seasonal_summary(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Compute seasonal statistics ordered chronologically through the year."""
    if metric_col not in df.columns or "Season" not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby("Season")[metric_col]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            P95=lambda s: np.nanpercentile(s.dropna(), 95) if len(s.dropna()) else np.nan,
            Max="max",
        )
        .reset_index()
    )
    order_idx = {name: i for i, name in enumerate(SEASON_ORDER)}
    grouped["_order"] = grouped["Season"].map(lambda s: order_idx.get(s, 99))
    return grouped.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)


def compute_diwali_window_analysis(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Compare festival window (D-1 to D+2) vs non-Diwali same-month baseline."""
    if metric_col not in df.columns or "Date_dt" not in df.columns:
        return pd.DataFrame()

    rows = []
    peak_col = "Daily_Max_AQI" if "Daily_Max_AQI" in df.columns else metric_col
    for year, date_str in sorted(DIWALI_DATES.items()):
        dt = pd.to_datetime(date_str)
        yr_df = df[df["Date_dt"].dt.year == year]
        if yr_df.empty:
            continue
        win_mask = (yr_df["Date_dt"] >= dt - pd.Timedelta(days=1)) & (
            yr_df["Date_dt"] <= dt + pd.Timedelta(days=2)
        )
        win_df = yr_df[win_mask]
        month_base_df = yr_df[(yr_df["Date_dt"].dt.month == dt.month) & (~win_mask)]
        if win_df.empty or month_base_df.empty:
            continue
        win_mean = float(win_df[metric_col].mean())
        base_mean = float(month_base_df[metric_col].mean())
        peak_val = float(win_df[peak_col].max())
        surge_pct = ((win_mean - base_mean) / base_mean * 100.0) if base_mean > 0 else 0.0
        rows.append({
            "Year": year,
            "Diwali_Date": date_str,
            "Festival_Window_Mean": round(win_mean, 1),
            "Month_Baseline_Mean": round(base_mean, 1),
            "Peak_Recorded": round(peak_val, 1),
            "Surge_vs_Baseline_Pct": round(surge_pct, 1),
        })
    return pd.DataFrame(rows)


def compute_station_summary(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Rank Bengaluru stations by mean pollution and attach area driver profiles."""
    if metric_col not in df.columns or "Station" not in df.columns:
        return pd.DataFrame()
    grouped = (
        df.groupby("Station")[metric_col]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            P90=lambda s: np.nanpercentile(s.dropna(), 90) if len(s.dropna()) else np.nan,
            Max="max",
        )
        .reset_index()
        .sort_values("Mean", ascending=False)
        .reset_index(drop=True)
    )
    grouped["Zone_Type"] = grouped["Station"].map(
        lambda s: BENGALURU_AREA_PROFILES.get(s, {}).get("zone", "Bengaluru Urban Monitoring Site")
    )
    grouped["Primary_Causes"] = grouped["Station"].map(
        lambda s: BENGALURU_AREA_PROFILES.get(s, {}).get(
            "drivers", "Mixed urban traffic, construction dust, and seasonal inversions."
        )
    )
    return grouped


def generate_dynamic_insights(df: pd.DataFrame, metric_col: str) -> Dict[str, str]:
    """Produce data-backed narrative answers for worst months, cleanest months, and hotspots."""
    monthly = compute_monthly_summary(df, metric_col)
    seasonal = compute_seasonal_summary(df, metric_col)
    stations = compute_station_summary(df, metric_col)
    diwali = compute_diwali_window_analysis(df, metric_col)

    worst_months_str = "N/A"
    cleanest_months_str = "N/A"
    if not monthly.empty:
        top3 = monthly.sort_values("Mean", ascending=False).head(3)
        bot3 = monthly.sort_values("Mean", ascending=True).head(3)
        worst_months_str = ", ".join(f"{r['Month_Name']} ({r['Mean']:.1f})" for _, r in top3.iterrows())
        cleanest_months_str = ", ".join(f"{r['Month_Name']} ({r['Mean']:.1f})" for _, r in bot3.iterrows())

    worst_season = seasonal.sort_values("Mean", ascending=False).iloc[0]["Season"] if not seasonal.empty else "N/A"
    best_season = seasonal.sort_values("Mean", ascending=True).iloc[0]["Season"] if not seasonal.empty else "N/A"

    top_station = stations.iloc[0]["Station"] if not stations.empty else "N/A"
    top_station_val = f"{stations.iloc[0]['Mean']:.1f}" if not stations.empty else "N/A"
    clean_station = stations.iloc[-1]["Station"] if not stations.empty else "N/A"
    clean_station_val = f"{stations.iloc[-1]['Mean']:.1f}" if not stations.empty else "N/A"

    diwali_summary = "No Diwali dates in selected range."
    if not diwali.empty:
        avg_surge = diwali["Surge_vs_Baseline_Pct"].mean()
        max_row = diwali.sort_values("Festival_Window_Mean", ascending=False).iloc[0]
        diwali_summary = (
            f"Across {len(diwali)} Diwali years, festival window mean averaged {avg_surge:+.1f}% vs. rest-of-month baseline "
            f"(highest window mean: {max_row['Festival_Window_Mean']} in {int(max_row['Year'])}, "
            f"peak recorded: {max_row['Peak_Recorded']})."
        )

    return {
        "worst_months": worst_months_str,
        "cleanest_months": cleanest_months_str,
        "worst_season": worst_season,
        "best_season": best_season,
        "top_station": f"{top_station} ({top_station_val})",
        "clean_station": f"{clean_station} ({clean_station_val})",
        "diwali_summary": diwali_summary,
    }
