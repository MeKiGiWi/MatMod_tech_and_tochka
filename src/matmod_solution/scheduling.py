from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
from pulp import HiGHS, LpMinimize, LpProblem, LpStatus, LpVariable, lpSum


DEFAULT_STATIONS = ["BVR", "C", "FF", "K", "TS"]


@dataclass
class SchedulingInputs:
    reqlabor_path: Path
    sched_path: Path
    shifts_path: Path
    station_priorities_path: Path
    staff_limits_path: Path
    forecast_path: Path


def get_version(weekday_0based: int, hour: int) -> str:
    if weekday_0based < 5:
        return "будни/утр." if hour < 10 else "будни/осн."
    return "вых/утр." if hour < 10 else "вых/осн."


def build_staff_schedule(
    inputs: SchedulingInputs,
    output_dir: str | Path,
    stations: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    stations = stations or DEFAULT_STATIONS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reqlabor_df = pd.read_csv(inputs.reqlabor_path)
    sched_df = pd.read_csv(inputs.sched_path)
    shifts_df = pd.read_csv(inputs.shifts_path)
    station_prior_df = pd.read_csv(inputs.station_priorities_path)
    staff_limits_df = pd.read_csv(inputs.staff_limits_path)
    forecast_df = pd.read_csv(inputs.forecast_path)

    if "predicted_guests" in forecast_df.columns and "guests_count" not in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"predicted_guests": "guests_count"})

    forecast_df["sale_date"] = pd.to_datetime(forecast_df["sale_date"])
    base_date = forecast_df["sale_date"].min()

    reqlabor_df["version"] = reqlabor_df["version"].astype(str).str.strip()
    reqlabor_index: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for _, row in reqlabor_df.iterrows():
        reqlabor_index[(row["station_key"], row["version"])].append(
            (float(row["guests_count"]), float(row["reqlabor"]))
        )
    for key in reqlabor_index:
        reqlabor_index[key].sort(key=lambda item: item[0])

    def required_labor(station: str, version: str, guests: float) -> float:
        pairs = reqlabor_index.get((station, version), [])
        if not pairs:
            return 0.0
        for guests_count, req in pairs:
            if guests <= guests_count:
                return req
        return pairs[-1][1]

    req: dict[tuple[int, int, str], float] = {}
    for _, row in forecast_df.iterrows():
        weekday = int(pd.Timestamp(row["sale_date"]).weekday())
        hour = int(row["sale_hour"])
        version = get_version(weekday, hour)
        for station in stations:
            req[(weekday, hour, station)] = required_labor(
                station=station,
                version=version,
                guests=float(row["guests_count"]),
            )

    avail: dict[int, dict[int, tuple[int, int]]] = defaultdict(dict)
    for _, row in sched_df.iterrows():
        avail[int(row["employee_id"])][int(row["day"]) - 1] = (
            int(row["starttime"]),
            int(row["finishtime"]),
        )

    worktime_limit = {
        int(row["employee_id"]): int(row["worktime_limit"])
        for _, row in staff_limits_df.iterrows()
    }
    shift_limit = {
        int(row["employee_id"]): int(row["shift_limit"])
        for _, row in staff_limits_df.iterrows()
    }

    station_prior = defaultdict(dict)
    for _, row in station_prior_df.iterrows():
        station_prior[int(row["employee_id"])][row["station_key"]] = int(
            row["station_priority"]
        )

    shift_prior = dict(
        zip(
            shifts_df["shift_duration"].astype(int),
            shifts_df["shift_priority"].astype(int),
        )
    )
    allowed_lengths = sorted(shift_prior)
    employees = sorted(staff_limits_df["employee_id"].astype(int).unique())

    problem = LpProblem("Staff_Scheduling", LpMinimize)
    valid_shifts: list[tuple[int, int, int, int, str, LpVariable]] = []
    covering_index: dict[tuple[int, int, str], list[LpVariable]] = defaultdict(list)

    for employee in employees:
        for day in range(7):
            if day not in avail[employee]:
                continue
            start_ok, end_ok = avail[employee][day]
            max_length = min(shift_limit[employee], 9)

            for length in allowed_lengths:
                if length > max_length:
                    continue
                min_start = max(7, start_ok)
                max_start = min(23 - length, end_ok - length)
                if min_start > max_start:
                    continue

                for start_hour in range(min_start, max_start + 1):
                    for station in stations:
                        var = LpVariable(
                            f"x_{employee}_{day}_{start_hour}_{length}_{station}",
                            cat="Binary",
                        )
                        valid_shifts.append(
                            (employee, day, start_hour, length, station, var)
                        )
                        for hour in range(start_hour, start_hour + length):
                            if 7 <= hour < 23:
                                covering_index[(day, hour, station)].append(var)

    for employee in employees:
        for day in range(7):
            vars_day = [
                var
                for emp, d, _, _, _, var in valid_shifts
                if emp == employee and d == day
            ]
            if vars_day:
                problem += lpSum(vars_day) <= 1, f"one_shift_{employee}_{day}"

    shortage: dict[tuple[int, int, str], LpVariable] = {}
    surplus: dict[tuple[int, int, str], LpVariable] = {}
    for day in range(7):
        for hour in range(7, 23):
            for station in stations:
                req_val = req.get((day, hour, station), 0)
                if req_val > 0 or covering_index[(day, hour, station)]:
                    shortage[(day, hour, station)] = LpVariable(
                        f"sh_{day}_{hour}_{station}",
                        lowBound=0,
                        cat="Integer",
                    )
                    surplus[(day, hour, station)] = LpVariable(
                        f"su_{day}_{hour}_{station}",
                        lowBound=0,
                        cat="Integer",
                    )
                    if covering_index[(day, hour, station)]:
                        problem += (
                            lpSum(covering_index[(day, hour, station)]) >= 1,
                            f"min_one_{day}_{hour}_{station}",
                        )
                    problem += (
                        lpSum(covering_index[(day, hour, station)])
                        + shortage[(day, hour, station)]
                        >= req_val,
                        f"min_cover_{day}_{hour}_{station}",
                    )
                    problem += (
                        lpSum(covering_index[(day, hour, station)])
                        - surplus[(day, hour, station)]
                        <= req_val + 2,
                        f"max_cover_{day}_{hour}_{station}",
                    )

    day_work = {}
    for employee in employees:
        emp_vars = [
            (length, var)
            for emp, _, _, length, _, var in valid_shifts
            if emp == employee
        ]
        if emp_vars:
            problem += (
                lpSum(length * var for length, var in emp_vars) <= worktime_limit[employee],
                f"weekly_h_{employee}",
            )

        for day in range(7):
            vars_day = [
                var
                for emp, d, _, _, _, var in valid_shifts
                if emp == employee and d == day
            ]
            y = LpVariable(f"dw_{employee}_{day}", cat="Binary")
            if vars_day:
                problem += lpSum(vars_day) <= y
                problem += lpSum(vars_day) >= y
            else:
                problem += y == 0
            day_work[(employee, day)] = y

        problem += lpSum(day_work[(employee, day)] for day in range(7)) <= 5
        problem += lpSum(day_work[(employee, day)] for day in range(7)) >= 1

    objective = []
    for var in shortage.values():
        objective.append(100_000 * var)
    for var in surplus.values():
        objective.append(500 * var)
    for employee, _, _, length, station, var in valid_shifts:
        objective.append((shift_prior[length] + station_prior[employee].get(station, 2)) * var)
    problem += lpSum(objective)

    problem.solve(HiGHS(msg=True, timeLimit=300, mip_rel_gap=0, threads=4, presolve="on"))

    shift_rows = []
    for employee, day, start_hour, length, station, var in valid_shifts:
        if (var.varValue or 0) >= 0.5:
            shift_rows.append(
                {
                    "employee_id": employee,
                    "day": day + 1,
                    "date": (base_date + timedelta(days=day)).date().isoformat(),
                    "start_hour": start_hour,
                    "end_hour": start_hour + length,
                    "shift_duration": length,
                    "station_key": station,
                }
            )

    shifts_out = pd.DataFrame(shift_rows).sort_values(
        ["date", "start_hour", "station_key", "employee_id"]
    )
    coverage_rows = []
    for day in range(7):
        for hour in range(7, 23):
            for station in stations:
                assigned = sum(
                    int(round(var.varValue or 0))
                    for var in covering_index[(day, hour, station)]
                )
                coverage_rows.append(
                    {
                        "date": (base_date + timedelta(days=day)).date().isoformat(),
                        "day": day + 1,
                        "hour": hour,
                        "station_key": station,
                        "required_people": req.get((day, hour, station), 0),
                        "assigned_people": assigned,
                        "shortage": int(round(shortage.get((day, hour, station), 0).varValue or 0))
                        if (day, hour, station) in shortage
                        else 0,
                        "surplus": int(round(surplus.get((day, hour, station), 0).varValue or 0))
                        if (day, hour, station) in surplus
                        else 0,
                    }
                )
    coverage_out = pd.DataFrame(coverage_rows)

    shifts_out.to_csv(output_dir / "shift_table.csv", index=False)
    coverage_out.to_csv(output_dir / "coverage_table.csv", index=False)

    summary = {
        "status": LpStatus[problem.status],
        "total_shortage": int(sum(round(var.varValue or 0) for var in shortage.values())),
        "total_surplus": int(sum(round(var.varValue or 0) for var in surplus.values())),
        "assigned_shifts": int(len(shifts_out)),
    }
    return shifts_out, coverage_out, summary

