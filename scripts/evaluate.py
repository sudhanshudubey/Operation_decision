"""Reproduce the committed evidence: python -m scripts.evaluate [--check]."""

import argparse
import hashlib
import json
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retail.data import DATA_PATH, ROOT, audit_data, daily_sales, load_data
from retail.forecast import run_evaluation
from retail.inventory import InventoryAssumptions, compare_inventory


def markdown_table(frame):
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.2f}"
        return str(value).replace("|", "/")
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *("| " + " | ".join(format_value(v) for v in row) + " |"
          for row in frame.itertuples(index=False, name=None)),
    ])


def evidence_table(result):
    rows = []
    for row in result["selection"].itertuples():
        metrics = result["test"].query("product_id == @row.product_id").set_index("model")
        selected, baseline = metrics.loc[row.selected_model], metrics.loc[row.baseline_model]
        reduction = (baseline.mae - selected.mae) / baseline.mae * 100 if baseline.mae else np.nan
        rows.append({"Product": row.product, "Selected model": row.selected_model,
                     "Baseline": row.baseline_model, "Baseline MAE": baseline.mae,
                     "Selected MAE": selected.mae, "MAE reduction (%)": reduction,
                     "Selected bias": selected.bias, "Test days": int(selected.n_days)})
    return pd.DataFrame(rows)


def write_chart(table, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "retail-decision-lab"})
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor="#f6f8fc")
    ax.set_facecolor("#f6f8fc")
    y = np.arange(len(table))
    ax.barh(y - .18, table["Baseline MAE"], .34, color="#a8b8ca", label="Validation-selected baseline")
    ax.barh(y + .18, table["Selected MAE"], .34, color="#2266ad", label="Validation-selected model")
    ax.set_yticks(y, table.Product)
    ax.invert_yaxis()
    ax.set_xlabel("Final test MAE · lower is better · units differ by product")
    ax.set_title("Retail forecasting: test the model against a credible baseline", loc="left", pad=22, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.text(.02, .01, "Distinct-record interpretation is provisional. Three expanding validation folds; final 20% held out.", fontsize=9, color="#526278")
    fig.tight_layout(rect=[0, .03, 1, 1])
    fig.savefig(path, format="svg", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recompute and compare committed metrics without rewriting files.")
    args = parser.parse_args()
    raw = load_data()
    audit, product_audit = audit_data(raw)
    result = run_evaluation(daily_sales(raw))
    inventory = []
    for pid in result["selection"].product_id:
        metrics, _ = compare_inventory(result, pid)
        metrics.insert(0, "product_id", pid)
        metrics.insert(1, "product", result["selection"].set_index("product_id").loc[pid, "product"])
        inventory.append(metrics)
    inventory = pd.concat(inventory, ignore_index=True)
    tables = {"test_metrics.csv": result["test"], "validation_metrics.csv": result["validation"],
              "coverage.csv": result["coverage"], "excluded_products.csv": result["excluded"],
              "model_selection.csv": result["selection"], "sales_grain_audit.csv": product_audit,
              "inventory_scenarios.csv": inventory}
    output = ROOT / "reports"
    if args.check:
        for filename, actual in tables.items():
            path = output / filename
            expected = pd.read_csv(path)
            # Normalize datetime columns in exactly the same way as CSV output.
            from io import StringIO
            normalized = pd.read_csv(StringIO(actual.to_csv(index=False)))
            pd.testing.assert_frame_equal(expected, normalized, check_exact=False, rtol=1e-7, atol=1e-8)
        print(f"Verified {len(tables)} committed evidence tables against the supplied dataset.")
        return
    output.mkdir(exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    result["predictions"].to_csv(output / "test_predictions.csv", index=False)
    manifest = {"data_file": "dataset.csv", "sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                "sales_interpretation": "distinct C520 records (provisional)", "audit": audit,
                "inventory_assumptions": vars(InventoryAssumptions()),
                "versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "streamlit", "matplotlib"]},
                "splits": result["splits"]}
    (output / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
    summary = evidence_table(result)
    chart_path = ROOT / "docs/assets/forecast_performance.svg"
    chart_path.parent.mkdir(exist_ok=True, parents=True)
    write_chart(summary, chart_path)
    selected_inventory = inventory.merge(result["selection"][["product_id", "selected_model"]],
                                         on="product_id", validate="many_to_one")
    selected_inventory = selected_inventory[selected_inventory.model == selected_inventory.selected_model]
    text = f"""# Reproduced results

Generated by `python -m scripts.evaluate`. This is a historical experiment, not measured business impact.

## Observation audit

- Raw rows: {audit['raw_rows']:,}; distinct C520 records: {audit['distinct_sales_records']:,}.
- Repeated sales records: {audit['repeated_sales_records']:,}; records with conflicting discount metadata: {audit['discount_conflict_records']:,}.
- Observation window: {audit['start']} through {audit['end']}; {audit['stores']} store; {audit['products']} parent products.
- Default grain is provisional. Confirm against the original data dictionary before using quantities operationally.

## Final forecasting test

The candidate assortment is selected before the first validation fold. All eligible models use the same test days per product. A baseline and a model are selected using validation MAE, then refit before the final test. The three folds end before {result['splits']['test_start'].date()}; the final test ends {result['splits']['test_end'].date()}.

{markdown_table(summary)}

MAE is in the product's recorded sales units. Positive bias means overprediction. In particular, Cucumber has substantial positive test bias despite a lower MAE; lower average error does not establish a well-calibrated inventory policy.

## Coverage exclusions

{markdown_table(result['excluded'])}

These products are retained in the audit. No backfill with a better-performing product is performed. Missing dates are unknown, and subsequent incomplete lag windows are excluded.

## Inventory scenarios

Defaults: lead time 2 days; fresh shelf life 3 days; holding cost 0.01 / unit / day; spoilage cost 1 / unit; lost-sale penalty 2 / unit. These are assumptions, not source measurements. Initial inventory is fresh and equal to the first policy target; terminal stock and pipeline are disclosed in the CSV. The daily review policy forecasts lead time + one day recursively from each origin. Buffer errors are calibrated on validation data only.

{markdown_table(selected_inventory[['product', 'buffer_quantile', 'days', 'fill_rate_pct', 'lost_sales_units', 'waste_units', 'scenario_cost']])}

90% / 95% are error-buffer quantiles, not achieved service guarantees. Fill rate is units served / historical sales in the selected consecutive test run. The full CSV also contains the validation-selected baseline policies. Do not pool these costs across products with different evaluation windows. Lower forecast MAE does not guarantee better inventory service: inspect Broccoli against its baseline.

## Practical limits

No verified source attribution, redistribution permission, stockout observations, actual shelf life or cost data are available. Recorded sales may be censored demand. The simulation uses historical demand to update forecasts even after simulated lost sales (a full-information assumption). Overlapping calibration horizons are dependent; there is no independent coverage guarantee or uncertainty interval. Results cover one store and a limited historical period. They do not establish a causal price effect, live forecasting performance, or real savings.

See `manifest.json` for the input hash, versions, dates and scenario assumptions. `test_predictions.csv` is generated locally and excluded from version control; the compact evidence tables are committed.
"""
    (output / "RESULTS.md").write_text(text)
    print(summary.to_string(index=False))
    print(f"Saved evidence tables, report and chart to {output}.")


if __name__ == "__main__":
    main()
