# Presenting the project in an interview

## Explain the decision

Start with the store decision: fresh produce needs enough stock to meet demand, but excess stock can expire. Forecast quality is useful only when you understand what the resulting order policy does. Explain the observed store-period and distinguish this historical experiment from a deployed replenishment system.

## Show the evidence

Open the Data audit tab first. Demonstrate how a one-to-many article mapping can repeat identical sales values. Explain why you kept distinct sales records and preserved the original interpretation as sensitivity. Be explicit that confirming this interpretation requires the original data dictionary.

In Forecast quality, explain a lag feature with an actual product example. Walk through the training, three validation windows and final test. Explain why last week is a stronger baseline for some products than yesterday. Show signed bias as well as MAE; Cucumber's positive bias is a useful example of a metric hiding an operational risk.

In Inventory scenarios, follow a single delivery and batch expiration in the exported trace. Change shelf life or a cost assumption and explain why the preferred scenario may change. Compare measured fill rate with the buffer label. Explain why lower forecast error need not improve every inventory outcome.

Finish with a limitation you can investigate: the original sales grain, unknown stockouts, actual shelf life, or evaluation on a new period. A credible next experiment is more useful than claiming unmeasured savings.

## Be precise about your contribution

The original notebook references group coursework. Identify which original sections you personally completed, what your teammates contributed, and which portfolio extensions you have since understood and verified. Do not describe all group work as sole authorship. AI coding assistance was used for this refactor; be ready to explain and modify the code yourself.

## Claims the evidence can support

- Implemented or contributed to chronological forecasting evaluation, data-quality auditing, and a Streamlit decision-support prototype—only for the parts you can explain and personally own.
- Compared two simple baselines with a Random Forest across three validation windows and a disclosed final test.
- Analyzed simulated inventory trade-offs under explicit lead-time, shelf-life and cost assumptions.

Use the exact baseline, product, dates and interpretation when citing a forecasting number from [the report](../reports/RESULTS.md). Do not convert MAE reduction into "accuracy," claim real waste/cost reductions, imply a causal price effect, or describe a historical dashboard as a production deployment.
