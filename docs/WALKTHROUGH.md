# Project walkthrough

## The problem

A store needs enough fresh produce to meet demand, but extra stock can expire. The project compares sales forecasts and tests the stock policies they produce.

## 1. Check the data

Open the Data audit tab. Some sales records repeat under several article codes. Compare the distinct-record totals with the original row sums. The original data dictionary is needed to confirm which interpretation is correct.

Missing dates are kept as unknown observations. Products without enough data are listed as exclusions.

## 2. Compare forecasts

Choose a product in Forecast quality. Compare yesterday's sales, last week's sales and Random Forest predictions.

The model is selected using three validation periods, then evaluated on the final test period. Check MAE and bias together: Cucumber has lower MAE but substantial overprediction. The charts show where errors occur.

## 3. Follow the inventory

Open Inventory scenarios and inspect a daily stock trace. Orders arrive after the selected lead time, older stock is used first, and expired units are counted as waste.

Change shelf life or a cost assumption and compare fill rate, lost sales, waste and cost. The buffer percentile is a setting based on validation errors; achieved service is measured separately.

## 4. Explain the limits

The analysis covers one historical store-period. Shelf life and costs are assumptions, and recorded sales may understate demand when the store was out of stock. Price comparisons show association rather than a causal effect.

The next steps are confirming the data definitions, obtaining actual inventory and cost information, and testing on a newer period. Refer to [the results report](../reports/RESULTS.md) for exact figures and dates.
