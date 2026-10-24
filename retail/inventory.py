"""A daily FIFO, lost-sales simulation under explicit scenario assumptions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retail.forecast import forecast_horizon


@dataclass(frozen=True)
class InventoryAssumptions:
    lead_days: int = 2
    shelf_days: int = 3
    holding_cost: float = 0.01
    waste_cost: float = 1.0
    lost_sale_cost: float = 2.0

    def __post_init__(self):
        if self.lead_days < 1 or self.shelf_days < 1:
            raise ValueError("Lead time and shelf life must be positive whole days.")
        if int(self.lead_days) != self.lead_days or int(self.shelf_days) != self.shelf_days:
            raise ValueError("Lead time and shelf life must be whole days.")
        costs = [self.holding_cost, self.waste_cost, self.lost_sale_cost]
        if not all(np.isfinite(v) and v >= 0 for v in costs):
            raise ValueError("Scenario costs must be finite and non-negative.")


def simulate_inventory(demand, targets, assumptions=InventoryAssumptions()):
    """Receive, expire, order, then serve demand FIFO each day.

    Orders placed on t arrive before demand on t+lead_days. A fresh batch
    received on t is usable for shelf_days days, including t. Initial stock
    is a fresh batch equal to the first target. Unfilled demand is lost.
    """
    if demand.empty or not demand.index.equals(targets.index):
        raise ValueError("Demand and targets must share a non-empty daily index.")
    expected = pd.date_range(demand.index.min(), periods=len(demand))
    if not demand.index.equals(expected):
        raise ValueError("Inventory simulation requires consecutive observed days.")
    if not np.isfinite(demand).all() or not np.isfinite(targets).all():
        raise ValueError("Inventory inputs must be finite; unknown demand cannot be zero-filled.")
    if (demand < 0).any() or (targets < 0).any():
        raise ValueError("Inventory inputs must be non-negative.")
    initial = float(np.ceil(targets.iloc[0]))
    batches = [[assumptions.shelf_days, initial]]
    pipeline, rows = [], []
    for t, (day, actual) in enumerate(demand.items()):
        received = sum(q for arrival, q in pipeline if arrival == t)
        pipeline = [(arrival, q) for arrival, q in pipeline if arrival > t]
        if received:
            batches.append([t + assumptions.shelf_days, received])
        waste = sum(q for expiry, q in batches if expiry <= t)
        batches = [[expiry, q] for expiry, q in batches if expiry > t]
        opening = sum(q for _, q in batches)
        position = opening + sum(q for _, q in pipeline)
        order = float(np.ceil(max(0, targets.iloc[t] - position)))
        if order:
            pipeline.append((t + assumptions.lead_days, order))
        remaining = float(actual)
        for batch in batches:
            taken = min(batch[1], remaining)
            batch[1] -= taken
            remaining -= taken
        closing = sum(q for _, q in batches)
        cost = (closing * assumptions.holding_cost + waste * assumptions.waste_cost
                + remaining * assumptions.lost_sale_cost)
        rows.append({"date": day, "demand": actual, "opening_stock": opening,
                     "received": received, "ordered": order, "fulfilled": actual - remaining,
                     "lost_sales": remaining, "waste": waste, "closing_stock": closing,
                     "target": targets.iloc[t], "scenario_cost": cost})
    trace = pd.DataFrame(rows).set_index("date")
    total = trace.demand.sum()
    supply = initial + trace.received.sum()
    metrics = {"days": len(trace), "demand_units": float(total),
               "fill_rate_pct": float(trace.fulfilled.sum() / total * 100) if total else np.nan,
               "stockout_free_days_pct": float((trace.lost_sales == 0).mean() * 100),
               "lost_sales_units": float(trace.lost_sales.sum()),
               "waste_units": float(trace.waste.sum()),
               "waste_pct_of_received": float(trace.waste.sum() / supply * 100) if supply else np.nan,
               "mean_closing_stock": float(trace.closing_stock.mean()),
               "scenario_cost": float(trace.scenario_cost.sum()),
               "initial_stock": initial, "received_units": float(trace.received.sum()),
               "ending_stock": float(trace.closing_stock.iloc[-1]),
               "ending_pipeline": float(sum(q for _, q in pipeline))}
    return trace, metrics


def calibrate_buffer(result, product_id, model, horizon, quantile):
    """Calibrate horizon error on validation only; never use final test errors."""
    sales = result["series"][product_id].sales
    errors = []
    for fold in result["splits"]["folds"]:
        origins = pd.date_range(fold["start"], fold["end"] - pd.Timedelta(days=horizon - 1))
        predicted = forecast_horizon(sales, origins, horizon, model,
                                     result["validation_forests"][product_id, fold["fold"]])
        actual = pd.DataFrame({step + 1: sales.reindex(origins + pd.Timedelta(days=step)).to_numpy()
                               for step in range(horizon)}, index=origins)
        valid = predicted.notna().all(axis=1) & actual.notna().all(axis=1)
        errors.extend((actual.loc[valid].sum(axis=1) - predicted.loc[valid].sum(axis=1)).tolist())
    if len(errors) < 20:
        raise ValueError("Not enough validation horizons to calibrate inventory buffers.")
    return max(0.0, float(np.quantile(errors, quantile))), len(errors)


def compare_inventory(result, product_id, assumptions=InventoryAssumptions()):
    selection = result["selection"].set_index("product_id").loc[product_id]
    models = list(dict.fromkeys([selection.selected_model, selection.baseline_model]))
    sales = result["series"][product_id].sales
    start = result["splits"]["test_start"]
    # Use the longest consecutive TEST run with known demand and seven known
    # prior days. Report the window so gaps are never silently treated as sales=0.
    eligible = sales.notna() & sales.shift(1).rolling(7).count().eq(7)
    eligible = eligible.loc[start:]
    groups = eligible.ne(eligible.shift()).cumsum()
    runs = [part.index for _, part in eligible.groupby(groups) if part.all()]
    if not runs or max(map(len, runs)) < 14:
        raise ValueError("Need at least 14 consecutive eligible test days for the simulation.")
    dates = max(runs, key=len)
    horizon = assumptions.lead_days + 1  # lead time plus daily review period
    metrics, traces = [], {}
    for model in models:
        forecasts = forecast_horizon(sales, dates, horizon, model, result["forests"][product_id])
        for quantile in [0.90, 0.95]:
            buffer, n = calibrate_buffer(result, product_id, model, horizon, quantile)
            targets = forecasts.sum(axis=1, min_count=horizon) + buffer
            trace, row = simulate_inventory(sales.loc[dates], targets, assumptions)
            policy = f"{model} / {quantile:.0%} buffer"
            traces[policy] = trace
            metrics.append({"policy": policy, "model": model, "buffer_quantile": quantile,
                            "buffer_units": buffer, "calibration_horizons": n,
                            "start": dates[0], "end": dates[-1], **row})
    return pd.DataFrame(metrics), traces
