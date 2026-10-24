# Data provenance and observation grain

## What is available

The repository already contained `dataset.csv` and an identical compressed copy. The revised project retains the CSV as its sole input. It contains one store, 188 parent products, dates from 2017-12-26 to 2019-12-24, and 204,135 joined rows. The input SHA-256 is recorded in [the manifest](../reports/manifest.json).

The original publisher, source URL, data dictionary and redistribution permission are not documented in the repository. These details still need confirmation.

## Why the unit of observation matters

There are 72,452 distinct records when comparing every `C520_*` column. The remaining 131,683 rows repeat one of those sales records while article metadata may change. For example, Aubergine has nine article mappings for many dates, with the same sales quantity repeated. Row-summing can therefore multiply observed quantities. This is consistent with a one-to-many join, but requires confirmation from the source owner.

There are also 43 date/item groups with two distinct sales records, so deduplicating blindly on date and product would discard potentially valid observations. The default instead retains each distinct **complete C520 record**, then aggregates those records by date and parent product. It preserves differing prices, sales quantities, statuses and change fields.

The default and original interpretations are both available in the dashboard. Switching interpretation also changes assortment ranking; it is a sensitivity analysis of the entire pipeline, not a controlled comparison on an identical assortment. Neither interpretation is a verified ground truth.

## Metadata conflicts

12,717 distinct sales records have both 0 and 1 discount flags across their metadata mappings. The daily descriptive flag uses the maximum of the associated flags, but **discount is excluded from forecasting**. Same-day price is also excluded because its availability at the forecast origin is undocumented. Descriptive daily price is the unweighted mean of distinct sales-record prices, not a proven transaction-weighted selling price.

## Missing observations and demand

- Missing product-date observations remain `NaN`. They may represent closure, discontinued assortment, missing reporting, or another process; they are not assumed to be zero demand.
- An explicitly recorded zero is kept as zero.
- Forecast evaluation requires an observed outcome and a complete seven-day historical feature window.
- Inventory uses the longest eligible consecutive run within the final test period, with the exact dates and number of days disclosed.
- Recorded sales are treated as demand for the simulation. If historical stockouts constrained sales, true demand and lost sales are not identifiable here.
- Units are the recorded `C520_SOLDPIECES` values. Parent pack sizes can differ; do not treat aggregate MAE or inventory costs across products as a comparable monetary business KPI.

## Information still needed

Confirm the sales-record definitions and obtain the assortment calendar, store opening days, stockout history, promotion timing, delivery lead times, shelf lives, costs and usage permissions. A newer evaluation period and a live trial would be needed to measure business impact.
