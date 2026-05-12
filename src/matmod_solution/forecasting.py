from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .notebook_runtime import load_notebook_namespace
from .postprocess import DEFAULT_CALIBRATION_CONFIG, calibrate_forecast


FORECAST_NOTEBOOK_CELL_INDEXES = [4, 6, 8, 20]


@dataclass
class ForecastArtifacts:
    forecast_df: pd.DataFrame
    submission_df: pd.DataFrame
    calibrated_df: pd.DataFrame | None


def run_forecast_pipeline(
    notebook_path: str | Path,
    csv_path: str | Path,
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    forecast_start: str,
    use_calibration: bool = True,
) -> ForecastArtifacts:
    runtime = load_notebook_namespace(
        notebook_path=notebook_path,
        code_cell_indexes=FORECAST_NOTEBOOK_CELL_INDEXES,
    )

    forecast_df, submission_df = runtime.predict_future_week_with_checkpoint(
        csv_path=csv_path,
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        forecast_start=forecast_start,
        config=runtime.config,
    )

    calibrated_df = None
    if use_calibration:
        raw_history_df = runtime.load_raw_guests(csv_path)
        calibrated_df = calibrate_forecast(
            raw_history_df=raw_history_df,
            forecast_df=forecast_df,
            config=DEFAULT_CALIBRATION_CONFIG,
        )

    return ForecastArtifacts(
        forecast_df=forecast_df,
        submission_df=submission_df,
        calibrated_df=calibrated_df,
    )


def build_distribution_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    prepared = forecast_df.copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"])
    prepared["sale_date"] = prepared["timestamp"].dt.normalize()
    prepared["sale_hour"] = prepared["timestamp"].dt.hour
    prepared["predicted_guests"] = prepared["guests_count"]
    return prepared[["sale_date", "sale_hour", "predicted_guests", "guests_count"]].copy()

