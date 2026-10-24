# Evaluation and inventory protocol

## Forecast origin and inputs

At the start of date t, predict recorded sales on t using observations up to t-1. Inputs are sales at t-1 and t-7, mean sales over t-7..t-1, weekday and month. There is no current-day sales, price or discount input. All forecasts are non-negative. Missing history invalidates the feature row; zeros remain valid.

## Temporal experiment

1. Construct a global daily calendar. Reserve its last 20% for the final test.
2. Within the first 80%, create three adjacent validation blocks, each 15% of the development calendar, using all earlier observations for training. The first training window is therefore approximately 44% of the complete calendar.
3. Rank the ten candidate products using only observations before the first validation block. Do not rerank or backfill exclusions using later outcomes.
4. Require at least 60 training and seven validation observations in every fold, and at least seven final test observations. Disclose excluded candidates and all coverage counts.
5. Compare yesterday's sales, sales on the same day last week, and a Random Forest (200 trees, max depth 10, min leaf size 2, seed 42). These hyperparameters are fixed; no exhaustive tuning is claimed.
6. Select the model with the lowest validation MAE, weighting folds by evaluated days. Separately select the stronger of the two simple baselines using the same validation rule. Ties favor the earlier-listed simpler model. Forest computation uses one worker to avoid non-deterministic floating-point accumulation during prediction.
7. Refit on the development period. Report all three models on common eligible final test dates. Do not change the selection after observing test performance.

Models are frozen during each validation/test block, but one-day evaluation may use actual sales from earlier dates within that block. This represents rolling daily observation, not a single multi-month forecast. The final test has been inspected during development; a fresh independent period is needed for an unbiased claim about future performance. No statistical-significance claim is made.

## Metrics

| Metric | Definition | Interpretation |
| --- | --- | --- |
| MAE | Mean absolute prediction error | Recorded sales units; lower is better |
| WAPE | Sum absolute errors / sum absolute actual sales × 100 | Undefined if actual sales sum to zero |
| Seasonal MASE | MAE / mean training absolute seven-day difference | Training-scaled error; not a guarantee of beating a baseline in another period |
| Bias | Mean(prediction − actual) | Positive means overprediction |
| MAE reduction | (Selected baseline MAE − selected model MAE) / baseline MAE × 100 | Baseline selection occurs on validation, not test; negative values are retained |

## Inventory policy and timing

The policy reviews stock daily. For a fixed lead time L, it forecasts **L+1 days from a single order origin**, predicting t through t+L. Predictions are fed back recursively: future actual sales cannot enter the same-origin horizon. Each model uses its own validation horizon errors to calibrate a non-negative buffer at the 90th or 95th percentile. Only horizons whose actuals are entirely inside their validation block are used. Overlapping errors are dependent; these quantiles are not formal service or coverage guarantees.

Order-up-to target = max(0, forecast horizon sum + validation-error buffer). Orders are rounded up to whole recorded units. Inventory position is on-hand stock plus outstanding orders. The order quantity is max(0, target − position), rounded up.

On each day, the simulation receives due deliveries, expires unusable batches, places a new order, then serves demand FIFO. An order placed on t arrives before demand on t+L. A batch received on t is usable through t+shelf_life−1; it expires at the start of the next day. There are no backorders, capacity limits, minimum order quantities, variable lead times or supplier shortages. Outstanding units are included in position even if an earlier batch is expected to expire; this intentionally simple policy can perform poorly for short shelf lives.

Each policy starts with fresh stock at its first target. Initial stock, deliveries, terminal stock and pipeline are exported. Cost is holding on end-of-day stock + spoilage penalty on expired units + lost-sale penalty on unmet units. These are scenario cost units; procurement costs, fixed ordering costs, age-at-delivery variability and terminal valuation are excluded. At each new origin, historical observed demand is available even if the simulated policy could not serve all of it. This is a full-information assumption; the simulation does not model the feedback from policy-induced sales censoring into future forecasts.

Fill rate = total fulfilled units / observed demand units. Stockout-free days = fraction of days with no lost units. Spoilage rate uses initial inventory plus received units as its denominator. Different product windows are not pooled into a business cost claim. A 95% buffer can increase stock and waste without achieving 95% service; inspect the trace rather than assuming the target is attained.

## Price analysis

Daily observed prices are grouped using up to four quantile bins. Tied boundaries are dropped and labels are inferred from the resulting intervals. The dashboard reports bin counts and mean/median sales, alongside a scatterplot. It estimates neither a demand elasticity coefficient nor a causal price response. Calendar effects, availability, selection and promotion confounding remain unresolved.

## Verification

The automated checks perturb future observations to check temporal isolation, test recursive forecasting and validation-only calibration, preserve distinct sales events during deduplication, and check delivery timing, expiration, cost arithmetic, lost sales and mass conservation. Streamlit's test framework exercises every eligible product in the default interpretation, the raw-sum interpretation and a longer lead time. A separate reproduction check compares seven committed evidence tables against a fresh computation using the pinned environment.
