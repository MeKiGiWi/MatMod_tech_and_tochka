# Vkusno i Tochka Workforce Forecasting & Scheduling

![Hackathon Winner](https://img.shields.io/badge/Hackathon-Winner-success)
![Forecasting TFT + Prophet](https://img.shields.io/badge/Forecasting-TFT%20%2B%20Prophet-orange)
![Optimization MILP](https://img.shields.io/badge/Optimization-MILP-blue)
![Solver PuLP + HiGHS](https://img.shields.io/badge/Solver-PuLP%20%2B%20HiGHS-2f855a)
![Visualization Pygame](https://img.shields.io/badge/Visualization-Pygame-6b46c1)

Проект-победитель хакатона МАИ x ФУ VI Весенняя школа ИТ и ИИ.

Система прогнозирования количества клиентов в ПБО «Вкусно и Точка» по часам и автоматического составления смен сотрудников на следующую неделю с учетом ограничений, доступности и пожеланий персонала.

## Ключевые особенности

- Проект-победитель хакатона МАИ x ФУ VI Весенняя школа ИТ и ИИ.
- Почасовой прогноз клиентского потока для конкретной точки.
- Использование временных рядов с погодными и календарными признаками.
- Сравнение моделей `Prophet` и `Temporal Fusion Transformer`.
- MILP-оптимизация расписания сотрудников через `PuLP`.
- Учет рабочих ограничений, лимитов часов и доступности сотрудников.
- Подготовка таблиц покрытия и итогового сменного расписания.

## Проблема

Для ресторанов быстрого питания важно заранее понимать ожидаемый поток клиентов, чтобы формировать рабочие смены без нехватки персонала и без лишних затрат на избыточное покрытие. Ручное планирование плохо масштабируется и слабо учитывает одновременно исторический спрос, погодные факторы, праздники и ограничения сотрудников.

## Решение

Система состоит из двух основных частей:

1. Forecasting pipeline
   - подготовка временного ряда по посетителям;
   - сбор погодных, календарных и праздничных агрегатов;
   - исследование влияния признаков на target;
   - обучение `Prophet` и `Temporal Fusion Transformer`;
   - почасовой прогноз трафика на следующую неделю;
   - постобработка и калибровка прогноза.

2. Scheduling pipeline
   - преобразование прогноза в demand для станций;
   - расчет требуемого количества сотрудников по часам;
   - оптимизация смен методом целочисленного линейного программирования;
   - учет доступности, лимитов часов, длительностей смен и приоритетов станций;
   - выгрузка расписания и таблицы покрытия.

## Архитектура

```text
Исторические данные по гостям
  -> погодные и календарные признаки
  -> Prophet / TFT forecasting
  -> постобработка прогноза
  -> расчет требуемой нагрузки по часам
  -> MILP scheduling
  -> таблица смен и покрытие по станции/часу
```

## Machine Learning

- В качестве признаков используются временные, погодные и праздничные агрегаты и их производные.
- Для baseline и сравнительного анализа применялся `Prophet`.
- Основная продвинутая модель прогноза построена на `Temporal Fusion Transformer`.
- Модели предсказывают количество посетителей в конкретный час.
- После инференса прогноз дополнительно калибруется перед передачей в scheduling pipeline.

Репозиторий сфокусирован на демонстрации полного аналитического пайплайна для hackathon prototype: от прогноза спроса до оптимизации расписания. Он не претендует на production-grade систему кадрового планирования.

## Scheduling

- Расписание строится как задача целочисленного линейного программирования.
- Для оптимизации используется `PuLP` с solver `HiGHS`.
- Ограничения учитывают доступность сотрудника по дням и часам.
- В модели присутствуют недельные лимиты часов, лимиты на длину смены и число рабочих дней.
- Целевая функция штрафует нехватку покрытия и избыточную нагрузку, а также учитывает приоритеты смен и станций.

## Структура репозитория

```text
src/matmod_solution/   Python-модули прогноза, постобработки и scheduling
notebooks/             Исследования, обучение моделей и эксперименты
data/                  Плейсхолдеры для входных данных
checkpoints/           Плейсхолдеры для model checkpoints
outputs/               Плейсхолдеры для результатов запусков
main.py                Единая точка входа в forecasting + scheduling pipeline
```

## Технологии

- Python
- pandas
- NumPy
- PyTorch
- PyTorch Forecasting
- Lightning
- Prophet
- PuLP
- HiGHS
- matplotlib
- Pygame

## Запуск

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Подготовка данных

Положите входные файлы в следующие пути:

```text
data/clean_guests.csv
data/distribution/reqlabor.csv
data/distribution/sched.csv
data/distribution/shifts.csv
data/distribution/station_priorities.csv
data/distribution/staff_limits.csv
checkpoints/present/*.ckpt
checkpoints/present/*metadata.json
```

### Основной запуск

```bash
PYTHONPATH=src python3 main.py \
  --history-csv data/clean_guests.csv \
  --checkpoint checkpoints/present/tft_weather_model_final_20260427.ckpt \
  --metadata checkpoints/present/tft_weather_model_final_20260427_metadata.json \
  --forecast-start 2026-04-27 \
  --output-dir outputs/latest_run
```

### Артефакты запуска

После выполнения пайплайна формируются:

- `outputs/latest_run/submission.csv`
- `outputs/latest_run/forecast_for_distribution.csv`
- `outputs/latest_run/distribution_forecast.csv`
- `outputs/latest_run/shift_table.csv`
- `outputs/latest_run/coverage_table.csv`

## Ключевые файлы

- [main.py](/Users/daniil/code/mai/MatMod_tech_and_tochka/main.py)
- [src/matmod_solution/forecasting.py](/Users/daniil/code/mai/MatMod_tech_and_tochka/src/matmod_solution/forecasting.py)
- [src/matmod_solution/postprocess.py](/Users/daniil/code/mai/MatMod_tech_and_tochka/src/matmod_solution/postprocess.py)
- [src/matmod_solution/scheduling.py](/Users/daniil/code/mai/MatMod_tech_and_tochka/src/matmod_solution/scheduling.py)
- [src/matmod_solution/notebook_runtime.py](/Users/daniil/code/mai/MatMod_tech_and_tochka/src/matmod_solution/notebook_runtime.py)
- [notebooks/training_and_eval/01_train_weather_tft_prod_calendar_8weeks.ipynb](/Users/daniil/code/mai/MatMod_tech_and_tochka/notebooks/training_and_eval/01_train_weather_tft_prod_calendar_8weeks.ipynb)
- [notebooks/training_and_eval/02_clean_previous_runs_test_finetune_submission.ipynb](/Users/daniil/code/mai/MatMod_tech_and_tochka/notebooks/training_and_eval/02_clean_previous_runs_test_finetune_submission.ipynb)
- [notebooks/распределение_fixed.ipynb](/Users/daniil/code/mai/MatMod_tech_and_tochka/notebooks/распределение_fixed.ipynb)

## Ограничения

- Это hackathon prototype, а не production-ready система планирования.
- Для воспроизводимости нужны исходные данные и чекпоинты моделей, которые не включены в репозиторий.
- Прогнозная часть зависит от сохраненного notebook-based pipeline.
- Для industrial deployment потребуются дополнительные проверки качества, мониторинг и формализация бизнес-ограничений.

## Результат

Проект занял первое место на хакатоне МАИ x ФУ VI Весенняя школа ИТ и ИИ.

## Disclaimer

Проект предназначен как исследовательский и hackathon prototype. Он демонстрирует связку ML-прогнозирования трафика и математической оптимизации расписания, но не является production-решением без дополнительной инженерной и бизнес-проработки.
