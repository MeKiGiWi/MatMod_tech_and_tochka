from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CalibrationConfig:
    years: list[int]
    target_mmdd: list[str]
    strength: float = 0.22
    min_factor: float = 0.75
    max_factor: float = 1.25
    spring_start: str = "04-01"
    spring_end: str = "05-20"


DEFAULT_CALIBRATION_CONFIG = CalibrationConfig(
    years=[2019, 2021, 2022, 2023, 2024, 2025],
    target_mmdd=["04-27", "04-28", "04-29", "04-30", "05-01", "05-02", "05-03"],
)


def calibrate_forecast(
    raw_history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG,
) -> pd.DataFrame:
    hist = raw_history_df.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist["sale_date"] = pd.to_datetime(hist["sale_date"])
    hist["sale_hour"] = hist["timestamp"].dt.hour
    hist["year"] = hist["sale_date"].dt.year
    hist["mmdd"] = hist["sale_date"].dt.strftime("%m-%d")
    hist["dow"] = hist["sale_date"].dt.dayofweek
    hist = hist[hist["year"].isin(config.years)].copy()

    hist_target = hist[hist["mmdd"].isin(config.target_mmdd)].copy()
    spring = hist[
        (hist["mmdd"] >= config.spring_start)
        & (hist["mmdd"] <= config.spring_end)
        & (~hist["mmdd"].isin(config.target_mmdd))
    ].copy()

    target_profile = (
        hist_target.groupby(["mmdd", "sale_hour"])["guests_count"]
        .mean()
        .reset_index(name="target_mean")
    )
    spring_profile = (
        spring.groupby(["dow", "sale_hour"])["guests_count"]
        .mean()
        .reset_index(name="spring_mean")
    )

    calibrated = forecast_df.copy()
    calibrated["timestamp"] = pd.to_datetime(calibrated["timestamp"])
    calibrated["sale_hour"] = calibrated["timestamp"].dt.hour
    calibrated["mmdd"] = calibrated["timestamp"].dt.strftime("%m-%d")
    calibrated["dow"] = calibrated["timestamp"].dt.dayofweek

    calibrated = calibrated.merge(target_profile, on=["mmdd", "sale_hour"], how="left")
    calibrated = calibrated.merge(spring_profile, on=["dow", "sale_hour"], how="left")

    calibrated["raw_factor"] = calibrated["target_mean"] / calibrated["spring_mean"]
    calibrated["raw_factor"] = (
        calibrated["raw_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )
    calibrated["raw_factor"] = calibrated["raw_factor"].clip(
        config.min_factor,
        config.max_factor,
    )
    calibrated["calibration_factor"] = (
        1.0 + config.strength * (calibrated["raw_factor"] - 1.0)
    )
    calibrated["prediction_calibrated"] = (
        calibrated["prediction"] * calibrated["calibration_factor"]
    ).clip(lower=0)
    calibrated["guests_count"] = calibrated["prediction_calibrated"].round().astype(int)

    return calibrated

