"""Regression modeling and evaluation workflow (Slide 09)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelEvalResult:
    """Stores actual obtained regression metrics and test predictions."""

    model_name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    pipeline: Pipeline
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingBundle:
    """Complete evaluation output for Slide 09 comparison & What-If inference."""

    mode: str  # "next_day_aqi" | "cross_pollutant"
    target_name: str
    train_size: int
    test_size: int
    split_description: str
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    results: List[ModelEvalResult]
    y_test: np.ndarray
    test_dates: Optional[pd.Series]
    predictions_by_model: Dict[str, np.ndarray]
    stations: List[str]
    station_defaults: Dict[str, Dict[str, float]]


def _build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline([
            ("prep", preprocessor),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("prep", preprocessor),
            ("model", Ridge(alpha=5.0)),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocessor),
            ("model", HistGradientBoostingRegressor(max_iter=180, max_depth=6, learning_rate=0.06, random_state=42)),
        ]),
    }


def _extract_feature_importance(pipe: Pipeline, top_k: int = 10) -> Dict[str, float]:
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    try:
        names = list(prep.get_feature_names_out())
    except Exception:
        return {}
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in names]
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_, dtype=float))
        total = vals.sum()
        if total > 0:
            vals = vals / total
    else:
        return {}
    pairs = sorted(zip(clean_names, vals), key=lambda x: x[1], reverse=True)[:top_k]
    return {k: round(float(v), 4) for k, v in pairs}


def prepare_next_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-free lag and cyclical seasonal features for next-day AQI forecasting."""
    work = df.sort_values(["Station", "Date_dt"]).copy()
    g = work.groupby("Station")

    work["AQI_lag1"] = g["Daily_Mean_AQI"].shift(1)
    work["AQI_lag2"] = g["Daily_Mean_AQI"].shift(2)
    work["AQI_lag7"] = g["Daily_Mean_AQI"].shift(7)
    work["AQI_roll7_mean"] = g["Daily_Mean_AQI"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    work["AQI_roll7_std"] = g["Daily_Mean_AQI"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std()).fillna(0.0)
    work["Morning_AQI_lag1"] = g["Morning_Peak_AQI"].shift(1)
    work["Night_AQI_lag1"] = g["Evening_Night_AQI"].shift(1)

    work["Month_sin"] = np.sin(2 * np.pi * work["Month"] / 12.0)
    work["Month_cos"] = np.cos(2 * np.pi * work["Month"] / 12.0)
    work["DayOfYear_sin"] = np.sin(2 * np.pi * work["DayOfYear"] / 365.25)
    work["DayOfYear_cos"] = np.cos(2 * np.pi * work["DayOfYear"] / 365.25)
    work["Is_Weekend_Num"] = work["Is_Weekend"].astype(int)

    needed = [
        "Station", "Date_dt", "Daily_Mean_AQI",
        "AQI_lag1", "AQI_lag2", "AQI_lag7", "AQI_roll7_mean", "AQI_roll7_std",
        "Morning_AQI_lag1", "Night_AQI_lag1",
        "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos", "Is_Weekend_Num",
    ]
    return work[needed].dropna().reset_index(drop=True)


def train_next_day_aqi_models(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    chronological_split: bool = True,
) -> TrainingBundle:
    """Train & evaluate models predicting Daily_Mean_AQI from lagged + seasonal features."""
    feat_df = prepare_next_day_features(df)
    if len(feat_df) < 100:
        raise ValueError("Not enough rows with lag history to train time-series models.")

    num_cols = [
        "AQI_lag1", "AQI_lag2", "AQI_lag7", "AQI_roll7_mean", "AQI_roll7_std",
        "Morning_AQI_lag1", "Night_AQI_lag1",
        "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos", "Is_Weekend_Num",
    ]
    cat_cols = ["Station"]
    feature_cols = num_cols + cat_cols
    target_col = "Daily_Mean_AQI"

    if chronological_split:
        feat_df = feat_df.sort_values("Date_dt").reset_index(drop=True)
        split_idx = int(len(feat_df) * (1.0 - test_ratio))
        train_df = feat_df.iloc[:split_idx]
        test_df = feat_df.iloc[split_idx:]
        split_desc = (
            f"Chronological Split ({train_df['Date_dt'].min().strftime('%Y-%m-%d')} to "
            f"{train_df['Date_dt'].max().strftime('%Y-%m-%d')} train | "
            f"{test_df['Date_dt'].min().strftime('%Y-%m-%d')} to "
            f"{test_df['Date_dt'].max().strftime('%Y-%m-%d')} test)"
        )
    else:
        train_df, test_df = train_test_split(feat_df, test_size=test_ratio, random_state=42)
        split_desc = f"Random Split ({int((1 - test_ratio) * 100)}% train / {int(test_ratio * 100)}% test)"

    X_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])

    models = _build_models(preprocessor)
    results: List[ModelEvalResult] = []
    preds_map: Dict[str, np.ndarray] = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = np.clip(pipe.predict(X_test), 0.0, 500.0)
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        fi = _extract_feature_importance(pipe)
        results.append(ModelEvalResult(name, mae, mse, rmse, r2, pipe, fi))
        preds_map[name] = y_pred

    results.sort(key=lambda r: r.r2, reverse=True)

    station_defaults: Dict[str, Dict[str, float]] = {}
    for st, grp in feat_df.groupby("Station"):
        station_defaults[str(st)] = {
            "AQI_lag1": float(grp["AQI_lag1"].median()),
            "AQI_roll7_mean": float(grp["AQI_roll7_mean"].median()),
            "AQI_roll7_std": float(grp["AQI_roll7_std"].median()),
            "Morning_AQI_lag1": float(grp["Morning_AQI_lag1"].median()),
            "Night_AQI_lag1": float(grp["Night_AQI_lag1"].median()),
        }

    return TrainingBundle(
        mode="next_day_aqi",
        target_name="Next-Day Daily Mean AQI",
        train_size=len(train_df),
        test_size=len(test_df),
        split_description=split_desc,
        feature_columns=feature_cols,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        results=results,
        y_test=y_test,
        test_dates=test_df["Date_dt"].reset_index(drop=True),
        predictions_by_model=preds_map,
        stations=sorted(feat_df["Station"].unique().tolist()),
        station_defaults=station_defaults,
    )


def train_cross_pollutant_models(
    df: pd.DataFrame,
    target_col: str = "AQI",
    test_ratio: float = 0.2,
    chronological_split: bool = False,
) -> TrainingBundle:
    """Train & evaluate models estimating AQI or PM2.5 from co-measured gases & weather."""
    work = df.copy()
    work["Month_sin"] = np.sin(2 * np.pi * work["Month"] / 12.0)
    work["Month_cos"] = np.cos(2 * np.pi * work["Month"] / 12.0)

    candidate_num = ["NO2", "SO2", "CO", "O3", "RH", "TEMP", "WS", "Month_sin", "Month_cos"]
    if "Hour" in work.columns:
        work["Hour_sin"] = np.sin(2 * np.pi * work["Hour"] / 24.0)
        work["Hour_cos"] = np.cos(2 * np.pi * work["Hour"] / 24.0)
        candidate_num.extend(["Hour_sin", "Hour_cos"])

    # Only include numeric features that actually have >=40% non-null values
    num_cols = [c for c in candidate_num if c in work.columns and work[c].notna().mean() >= 0.4]
    cat_cols = ["Station"] if "Station" in work.columns else []
    if not num_cols:
        raise ValueError("Dataset lacks sufficient numeric pollutant/weather features for regression.")

    # Impute remaining sparse weather columns using median for modeling
    for c in num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(work[c].median())

    if target_col not in work.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    work = work.dropna(subset=[target_col]).copy()
    feature_cols = num_cols + cat_cols

    if chronological_split and "Date_dt" in work.columns:
        work = work.sort_values("Date_dt").reset_index(drop=True)
        split_idx = int(len(work) * (1.0 - test_ratio))
        train_df = work.iloc[:split_idx]
        test_df = work.iloc[split_idx:]
        split_desc = f"Chronological Split ({len(train_df)} train / {len(test_df)} test)"
    else:
        train_df, test_df = train_test_split(work, test_size=test_ratio, random_state=42)
        split_desc = f"Random Split ({int((1 - test_ratio) * 100)}% train / {int(test_ratio * 100)}% test)"

    X_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy()

    transformers = [("num", StandardScaler(), num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    preprocessor = ColumnTransformer(transformers)

    models = _build_models(preprocessor)
    results: List[ModelEvalResult] = []
    preds_map: Dict[str, np.ndarray] = {}

    upper_cap = 500.0
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = np.clip(pipe.predict(X_test), 0.0, upper_cap)
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        fi = _extract_feature_importance(pipe)
        results.append(ModelEvalResult(name, mae, mse, rmse, r2, pipe, fi))
        preds_map[name] = y_pred

    results.sort(key=lambda r: r.r2, reverse=True)

    station_defaults: Dict[str, Dict[str, float]] = {}
    if "Station" in work.columns:
        for st, grp in work.groupby("Station"):
            station_defaults[str(st)] = {c: float(grp[c].median()) for c in num_cols}

    return TrainingBundle(
        mode="cross_pollutant",
        target_name=target_col,
        train_size=len(train_df),
        test_size=len(test_df),
        split_description=split_desc,
        feature_columns=feature_cols,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        results=results,
        y_test=y_test,
        test_dates=test_df["Date_dt"].reset_index(drop=True) if "Date_dt" in test_df.columns else None,
        predictions_by_model=preds_map,
        stations=sorted(work["Station"].astype(str).unique().tolist()) if "Station" in work.columns else [],
        station_defaults=station_defaults,
    )


def predict_single_scenario(
    bundle: TrainingBundle,
    model_name: str,
    station: str,
    month: int,
    overrides: Dict[str, float],
) -> float:
    """Run inference for a single What-If scenario using a trained pipeline."""
    result = next((r for r in bundle.results if r.model_name == model_name), bundle.results[0])
    defaults = bundle.station_defaults.get(station, {})

    row: Dict[str, Any] = {}
    for col in bundle.numeric_features:
        row[col] = overrides.get(col, defaults.get(col, 0.0))

    # Refresh cyclical calendar encodings for chosen month
    day_of_year = (month - 1) * 30.4 + 15.0
    if "Month_sin" in bundle.numeric_features:
        row["Month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
    if "Month_cos" in bundle.numeric_features:
        row["Month_cos"] = float(np.cos(2 * np.pi * month / 12.0))
    if "DayOfYear_sin" in bundle.numeric_features:
        row["DayOfYear_sin"] = float(np.sin(2 * np.pi * day_of_year / 365.25))
    if "DayOfYear_cos" in bundle.numeric_features:
        row["DayOfYear_cos"] = float(np.cos(2 * np.pi * day_of_year / 365.25))

    if "Station" in bundle.categorical_features:
        row["Station"] = station

    X = pd.DataFrame([row])[bundle.feature_columns]
    pred = float(result.pipeline.predict(X)[0])
    return round(min(max(pred, 0.0), 500.0), 1)
