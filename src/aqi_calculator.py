"""CPCB Indian National Air Quality Index (NAQI) calculation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass(frozen=True)
class AQICategory:
    """Represents a CPCB AQI category with visual styling and health guidance."""

    label: str
    min_aqi: int
    max_aqi: int
    color: str
    text_color: str
    health_impact: str
    advisory: str


CPCB_CATEGORIES: List[AQICategory] = [
    AQICategory(
        label="Good",
        min_aqi=0,
        max_aqi=50,
        color="#16A34A",
        text_color="#FFFFFF",
        health_impact="Minimal impact.",
        advisory="Ideal air quality for outdoor exercise and ventilation.",
    ),
    AQICategory(
        label="Satisfactory",
        min_aqi=51,
        max_aqi=100,
        color="#65A30D",
        text_color="#FFFFFF",
        health_impact="Minor breathing discomfort to sensitive people.",
        advisory="Safe for general public; asthma-prone groups should monitor symptoms.",
    ),
    AQICategory(
        label="Moderately Polluted",
        min_aqi=101,
        max_aqi=200,
        color="#D97706",
        text_color="#FFFFFF",
        health_impact="Breathing discomfort to people with lung/heart disease, children and older adults.",
        advisory="Reduce prolonged outdoor exertion near heavy-traffic corridors.",
    ),
    AQICategory(
        label="Poor",
        min_aqi=201,
        max_aqi=300,
        color="#EA580C",
        text_color="#FFFFFF",
        health_impact="Breathing discomfort to most people on prolonged exposure.",
        advisory="Limit outdoor morning/evening workouts; consider N95 masks in traffic hotspots.",
    ),
    AQICategory(
        label="Very Poor",
        min_aqi=301,
        max_aqi=400,
        color="#DC2626",
        text_color="#FFFFFF",
        health_impact="Respiratory illness on prolonged exposure; pronounced effect on heart/lung patients.",
        advisory="Avoid outdoor physical activity; keep windows closed during evening inversions or fireworks.",
    ),
    AQICategory(
        label="Severe",
        min_aqi=401,
        max_aqi=500,
        color="#7F1D1D",
        text_color="#FFFFFF",
        health_impact="Affects healthy people and seriously impacts those with existing diseases.",
        advisory="Stay indoors, use air purifiers if available, and avoid all outdoor exertion.",
    ),
]

# Official CPCB NAQI Breakpoints: (B_LO, B_HI, I_LO, I_HI)
# Units: PM2.5, PM10, NO2, SO2, O3, NH3 in ug/m3; CO in mg/m3.
CPCB_BREAKPOINTS: Dict[str, List[Tuple[float, float, int, int]]] = {
    "PM2.5": [
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500),
    ],
    "PM10": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, 600.0, 401, 500),
    ],
    "NO2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 180.0, 101, 200),
        (180.0, 280.0, 201, 300),
        (280.0, 400.0, 301, 400),
        (400.0, 600.0, 401, 500),
    ],
    "SO2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 380.0, 101, 200),
        (380.0, 800.0, 201, 300),
        (800.0, 1600.0, 301, 400),
        (1600.0, 2400.0, 401, 500),
    ],
    "CO": [
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500),
    ],
    "O3": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 168.0, 101, 200),
        (168.0, 208.0, 201, 300),
        (208.0, 748.0, 301, 400),
        (748.0, 1000.0, 401, 500),
    ],
    "NH3": [
        (0.0, 200.0, 0, 50),
        (200.0, 400.0, 51, 100),
        (400.0, 800.0, 101, 200),
        (800.0, 1200.0, 201, 300),
        (1200.0, 1800.0, 301, 400),
        (1800.0, 2400.0, 401, 500),
    ],
}

POLLUTANT_ALIASES: Dict[str, str] = {
    "PM25": "PM2.5",
    "PM2.5": "PM2.5",
    "PM_25": "PM2.5",
    "PM10": "PM10",
    "NO2": "NO2",
    "SO2": "SO2",
    "CO": "CO",
    "O3": "O3",
    "NH3": "NH3",
}


def compute_sub_index(pollutant: str, concentration: Optional[float]) -> Optional[float]:
    """Calculate CPCB linear-interpolated sub-index I_p for a single pollutant."""
    canonical = POLLUTANT_ALIASES.get(pollutant.upper(), pollutant)
    if canonical not in CPCB_BREAKPOINTS or concentration is None:
        return None
    try:
        val = float(concentration)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or val < 0:
        return None

    ranges = CPCB_BREAKPOINTS[canonical]
    for b_lo, b_hi, i_lo, i_hi in ranges:
        if val <= b_hi:
            sub = ((i_hi - i_lo) / (b_hi - b_lo)) * (val - b_lo) + i_lo
            return round(min(max(sub, 0.0), 500.0), 2)
    return 500.0


def get_aqi_category(aqi_value: Optional[float]) -> AQICategory:
    """Return the CPCB AQI category matching a numeric AQI value."""
    if aqi_value is None or (isinstance(aqi_value, float) and math.isnan(aqi_value)):
        return CPCB_CATEGORIES[0]
    val = min(max(float(aqi_value), 0.0), 500.0)
    for cat in CPCB_CATEGORIES:
        if val <= cat.max_aqi:
            return cat
    return CPCB_CATEGORIES[-1]


def calculate_naqi(
    concentrations: Dict[str, Optional[float]],
    strict_min_pollutants: bool = False,
) -> Tuple[Optional[float], Optional[str], Dict[str, float], AQICategory]:
    """Compute overall CPCB National AQI = max(sub-indices).

    Args:
        concentrations: Mapping of pollutant name -> measured concentration.
        strict_min_pollutants: If True, enforce CPCB's rule (>= 3 pollutants
            with at least one of PM2.5 or PM10).

    Returns:
        (overall_aqi, dominant_pollutant, sub_indices_dict, AQICategory)
    """
    sub_indices: Dict[str, float] = {}
    for raw_name, conc in concentrations.items():
        canonical = POLLUTANT_ALIASES.get(raw_name.upper(), raw_name)
        sub = compute_sub_index(canonical, conc)
        if sub is not None:
            sub_indices[canonical] = sub

    if not sub_indices:
        return None, None, {}, CPCB_CATEGORIES[0]

    has_pm = ("PM2.5" in sub_indices) or ("PM10" in sub_indices)
    if strict_min_pollutants and (len(sub_indices) < 3 or not has_pm):
        return None, None, sub_indices, CPCB_CATEGORIES[0]

    dominant = max(sub_indices, key=lambda k: sub_indices[k])
    overall = round(sub_indices[dominant], 1)
    return overall, dominant, sub_indices, get_aqi_category(overall)
