from __future__ import annotations

import argparse
from pathlib import Path

from matmod_solution.forecasting import build_distribution_forecast, run_forecast_pipeline
from matmod_solution.scheduling import SchedulingInputs, build_staff_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference and staff scheduling pipeline.",
    )
    parser.add_argument(
        "--history-csv",
        default="data/clean_guests.csv",
        help="CSV with historical guest counts.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/present/tft_weather_model_final_20260427.ckpt",
        help="Path to TFT checkpoint.",
    )
    parser.add_argument(
        "--metadata",
        default="checkpoints/present/tft_weather_model_final_20260427_metadata.json",
        help="Path to checkpoint metadata JSON.",
    )
    parser.add_argument(
        "--forecast-start",
        default="2026-04-27",
        help="Forecast start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--forecast-notebook",
        default="notebooks/present_v2/02_clean_previous_runs_test_finetune_submission.ipynb",
        help="Notebook with the working TFT pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/latest_run",
        help="Directory for forecast and scheduling outputs.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable May calibration step.",
    )
    parser.add_argument("--reqlabor", default="data/distribution/reqlabor.csv")
    parser.add_argument("--sched", default="data/distribution/sched.csv")
    parser.add_argument("--shifts", default="data/distribution/shifts.csv")
    parser.add_argument(
        "--station-priorities",
        default="data/distribution/station_priorities.csv",
    )
    parser.add_argument(
        "--staff-limits",
        default="data/distribution/staff_limits.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = run_forecast_pipeline(
        notebook_path=args.forecast_notebook,
        csv_path=args.history_csv,
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        forecast_start=args.forecast_start,
        use_calibration=not args.no_calibration,
    )

    if artifacts.calibrated_df is not None:
        forecast_df = artifacts.calibrated_df.copy()
    else:
        forecast_df = artifacts.forecast_df.assign(
            guests_count=artifacts.forecast_df["prediction"].clip(lower=0).round().astype(int)
        )

    forecast_csv = output_dir / "forecast_for_distribution.csv"
    submission_csv = output_dir / "submission.csv"
    distribution_forecast_csv = output_dir / "distribution_forecast.csv"

    artifacts.submission_df.to_csv(submission_csv, index=False)
    forecast_df.to_csv(forecast_csv, index=False)
    build_distribution_forecast(forecast_df).to_csv(distribution_forecast_csv, index=False)

    scheduling_inputs = SchedulingInputs(
        reqlabor_path=Path(args.reqlabor),
        sched_path=Path(args.sched),
        shifts_path=Path(args.shifts),
        station_priorities_path=Path(args.station_priorities),
        staff_limits_path=Path(args.staff_limits),
        forecast_path=distribution_forecast_csv,
    )
    _, _, summary = build_staff_schedule(
        inputs=scheduling_inputs,
        output_dir=output_dir,
    )

    print(f"Saved submission: {submission_csv}")
    print(f"Saved forecast: {forecast_csv}")
    print(f"Saved distribution forecast: {distribution_forecast_csv}")
    print(f"Scheduling summary: {summary}")


if __name__ == "__main__":
    main()
