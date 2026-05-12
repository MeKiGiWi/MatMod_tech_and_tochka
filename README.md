# MatMod Solution

Проект собирает единый пайплайн:

1. Загружает TFT-модель из чекпоинта.
2. Делает прогноз гостей на будущую неделю.
3. Выполняет постобработку прогноза.
4. Передаёт прогноз в алгоритм формирования распределения.
5. Сохраняет итоговые таблицы смен и покрытия.

## Структура

- `main.py` — единая точка входа.
- `src/matmod_solution/forecasting.py` — запуск инференса через проверенный ноутбук.
- `src/matmod_solution/postprocess.py` — калибровка прогноза.
- `src/matmod_solution/scheduling.py` — MILP-распределение смен.
- `notebooks/` — перенесённые ноутбуки из `present_v2` и ноутбук распределения.

## Подготовка

Положите данные в такие пути:

- `data/clean_guests.csv`
- `data/distribution/reqlabor.csv`
- `data/distribution/sched.csv`
- `data/distribution/shifts.csv`
- `data/distribution/station_priorities.csv`
- `data/distribution/staff_limits.csv`
- `checkpoints/present/*.ckpt`
- `checkpoints/present/*metadata.json`

## Запуск

```bash
PYTHONPATH=src python3 main.py \
  --history-csv data/clean_guests.csv \
  --checkpoint checkpoints/present/tft_weather_model_final_20260427.ckpt \
  --metadata checkpoints/present/tft_weather_model_final_20260427_metadata.json \
  --forecast-start 2026-04-27 \
  --output-dir outputs/latest_run
```

После запуска появятся:

- `outputs/latest_run/submission.csv`
- `outputs/latest_run/forecast_for_distribution.csv`
- `outputs/latest_run/distribution_forecast.csv`
- `outputs/latest_run/shift_table.csv`
- `outputs/latest_run/coverage_table.csv`

