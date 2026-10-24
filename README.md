# Retail Decision Lab

Retail sales forecasting and inventory analysis, built from an Operations & Decisions coursework project.

The app compares daily sales forecasts and shows how different stock policies affect shortages, spoilage and inventory costs. It uses data from one store, covering December 2017 to December 2019.

[Results](reports/RESULTS.md) · [Method](docs/METHODOLOGY.md) · [Data notes](docs/DATA.md) · [Project walkthrough](docs/WALKTHROUGH.md)

![Final test MAE by product](docs/assets/forecast_performance.svg)

## Features

- Compare yesterday's sales, last week's sales and Random Forest forecasts.
- Evaluate models across three validation periods and a final test period.
- Simulate daily orders, deliveries, stock expiration and lost sales.
- Change lead time, shelf life and cost assumptions in the dashboard.
- Inspect repeated records, missing dates and conflicting discount flags.
- Download forecast metrics and daily inventory records as CSV files.

## Run locally

Use **Python 3.12**.

```bash
git clone https://github.com/sudhanshudubey/Operation_decision.git
cd Operation_decision
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. Open the local URL printed in the terminal. The first load trains the models; later product selections reuse the cached results.

## Results

Eight of the ten initially selected products have enough data for evaluation. Their final test MAE is about **3.3%–24.3% lower** than the baseline chosen during validation. Two products are excluded because they lack enough recent observations. Full metrics, dates and exclusions are in [the results report](reports/RESULTS.md).

Lower forecast error does not always improve inventory service. The simulation reports fill rate, lost sales, waste and cost separately. Its 90% and 95% buffer settings are calculated from validation errors; they are not guaranteed service levels.

## Data notes

The CSV contains 204,135 rows and 72,452 distinct `C520_*` sales records. Some sales values repeat under different article codes. By default, the analysis counts each identical sales record once and keeps distinct records on the same date. The dashboard also offers the original row-summing method for comparison.

This interpretation still needs confirmation against the original data dictionary. Missing dates remain unknown rather than being filled with zero sales. Dataset provenance and redistribution permission have not been confirmed. See [the data notes](docs/DATA.md).

Inventory lead time, shelf life and costs are assumptions. The results describe a historical simulation, not measured savings from a live store.

## Reproduce the analysis

```bash
python -m scripts.evaluate
python -m unittest discover -s tests -v
python -m scripts.evaluate --check
```

The evaluation command saves the report, CSV metrics, input hash and chart. `--check` reruns the analysis and compares the seven saved evidence tables. Tests cover forecast timing, repeated records, missing observations, inventory calculations and dashboard interactions.

The [analysis notebook](notebooks/retail_decision_analysis.ipynb) uses the same Python modules. Open it in a Jupyter-enabled editor with the project environment and a notebook kernel.

## Files

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard |
| `retail/data.py` | Data validation and daily sales preparation |
| `retail/forecast.py` | Forecasting models and evaluation |
| `retail/inventory.py` | Stock simulation |
| `scripts/evaluate.py` | Generate and check results |
| `tests/` | Analysis and dashboard tests |
| `reports/` | Results, metrics and assumptions |
| `notebooks/` | Current analysis and archived coursework |

The [original coursework notebook](notebooks/archive/original_coursework.ipynb) is kept in the archive. Use the current notebook or dashboard to run the updated analysis.
