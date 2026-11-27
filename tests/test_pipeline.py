"""Automated tests for CPCB NAQI math, cleaning pipeline, seasonal causes, and ML models."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aqi_calculator import calculate_naqi, compute_sub_index, get_aqi_category
from src.data_pipeline import load_and_clean_hourly_aqi, load_and_clean_pollutants
from src.ml_models import predict_single_scenario, train_cross_pollutant_models, train_next_day_aqi_models
from src.seasonal_analysis import (
    BENGALURU_STATION_COORDS,
    compute_diwali_window_analysis,
    compute_monthly_summary,
    compute_seasonal_summary,
    compute_station_summary,
)


class TestAirQualityAppPipeline(unittest.TestCase):
    def test_cpcb_naqi_breakpoints(self) -> None:
        self.assertEqual(compute_sub_index("PM2.5", 30.0), 50.0)
        self.assertEqual(compute_sub_index("PM2.5", 60.0), 100.0)
        aqi, dom, subs, cat = calculate_naqi({"PM2.5": 90.0, "PM10": 80.0, "NO2": 40.0})
        self.assertEqual(dom, "PM2.5")
        self.assertAlmostEqual(aqi, 200.0, places=1)
        self.assertEqual(cat.label, "Moderately Polluted")
        self.assertIn("PM10", subs)
        self.assertEqual(get_aqi_category(350).label, "Very Poor")

    def test_hourly_aqi_cleaning_and_seasonal_patterns(self) -> None:
        raw_df, clean_df, report = load_and_clean_hourly_aqi()
        self.assertEqual(len(raw_df), 21413)
        self.assertGreater(len(clean_df), 21000)
        self.assertGreaterEqual(len(report.stations), 12)

        monthly = compute_monthly_summary(clean_df, "Daily_Mean_AQI")
        self.assertEqual(len(monthly), 12)
        monsoon_mean = monthly[monthly["Month"].isin([6, 7, 8, 9])]["Mean"].mean()
        winter_mean = monthly[monthly["Month"].isin([12, 1, 2])]["Mean"].mean()
        self.assertGreater(winter_mean, monsoon_mean)

        diwali = compute_diwali_window_analysis(clean_df, "Daily_Mean_AQI")
        self.assertGreaterEqual(len(diwali), 5)
        self.assertGreater(diwali["Surge_vs_Baseline_Pct"].mean(), 0.0)

        stations = compute_station_summary(clean_df, "Daily_Mean_AQI")
        self.assertIn("Primary_Causes", stations.columns)
        for st in stations["Station"]:
            self.assertIn(st, BENGALURU_STATION_COORDS)

    def test_pollutant_dataset_mixed_timestamps_and_glitches(self) -> None:
        raw_df, clean_df, report = load_and_clean_pollutants()
        self.assertEqual(len(raw_df), 34849)
        # Mixed H:M timestamps from 2018 (HEBBAL, JAYANAGAR, etc.) must be retained
        self.assertGreater(len(clean_df), 34000)
        self.assertGreater(report.sensor_glitches_flagged, 200)
        self.assertLessEqual(clean_df["PM25"].max(), 500.0)
        self.assertIn("HEBBAL", report.stations)
        self.assertIn("Peenya", report.stations)

        seasonal = compute_seasonal_summary(clean_df, "AQI")
        self.assertEqual(len(seasonal), 4)

    def test_ml_workflows_and_single_prediction(self) -> None:
        _, df_h, _ = load_and_clean_hourly_aqi()
        bundle_h = train_next_day_aqi_models(df_h, test_ratio=0.2, chronological_split=True)
        self.assertEqual(len(bundle_h.results), 4)
        self.assertGreater(bundle_h.results[0].r2, 0.5)

        pred = predict_single_scenario(
            bundle_h,
            bundle_h.results[0].model_name,
            bundle_h.stations[0],
            month=11,
            overrides={"AQI_lag1": 140.0, "AQI_roll7_mean": 115.0, "Night_AQI_lag1": 160.0},
        )
        self.assertTrue(0.0 <= pred <= 500.0)

        _, df_p, _ = load_and_clean_pollutants()
        bundle_p = train_cross_pollutant_models(df_p, target_col="AQI", test_ratio=0.2)
        self.assertEqual(len(bundle_p.results), 4)
        self.assertGreater(bundle_p.results[0].r2, 0.25)


if __name__ == "__main__":
    unittest.main()
