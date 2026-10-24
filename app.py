"""Run from any directory: streamlit run /path/to/Operation_decision/app.py."""

import pandas as pd
import streamlit as st

from retail.data import DATA_PATH, audit_data, daily_sales, load_data
from retail.forecast import run_evaluation
from retail.inventory import InventoryAssumptions, compare_inventory

st.set_page_config(page_title="Retail Decision Lab", page_icon="📦", layout="wide")
st.markdown("""<style>
    .block-container {padding-top: 2rem; max-width: 1440px;}
    [data-testid="stMetric"] {background: #f3f6fa; padding: 16px; border-radius: 10px; color: #16253a;}
    h1, h2, h3 {letter-spacing: -0.025em;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Auditing sales and evaluating three chronological validation windows…")
def get_analysis(mode, input_version):
    raw = load_data()
    audit, products = audit_data(raw)
    daily = daily_sales(raw, mode=mode)
    return run_evaluation(daily), daily, audit, products


st.sidebar.title("Retail Decision Lab")
st.sidebar.caption("Forecast → evaluate → compare decisions")
mode_label = st.sidebar.selectbox("Sales interpretation", [
    "Distinct sales records (provisional)", "Original row sums (sensitivity)"
])
mode = "distinct" if mode_label.startswith("Distinct") else "raw"
try:
    analysis, daily, audit, product_audit = get_analysis(mode, DATA_PATH.stat().st_mtime_ns)
except (ValueError, FileNotFoundError) as exc:
    st.error(f"Cannot evaluate this dataset: {exc}")
    st.stop()

selection = analysis["selection"]
names = dict(zip(selection.product_id, selection["product"]))
pid = st.sidebar.selectbox("Product", selection.product_id.tolist(), format_func=lambda value: names[value])
st.sidebar.divider()
st.sidebar.subheader("Inventory assumptions")
lead = st.sidebar.slider("Lead time (days)", 1, 7, 2)
shelf = st.sidebar.slider("Shelf life on arrival (days)", 1, 14, 3)
holding = st.sidebar.number_input("Holding cost / unit / day", min_value=0.0, value=0.01, step=0.01)
waste = st.sidebar.number_input("Spoilage cost / unit", min_value=0.0, value=1.0, step=0.10)
lost = st.sidebar.number_input("Lost-sale penalty / unit", min_value=0.0, value=2.0, step=0.10)
st.sidebar.caption("Costs are illustrative cost units, not measured business expenses.")

st.title("Retail Decision Lab")
st.write("How much confidence should a store place in a forecast—and what happens when that forecast drives inventory?")
st.caption(f"Historical case study · one store · {audit['start']} to {audit['end']} · {names[pid]}")
st.warning("Data interpretation is provisional. Repeated sales records and conflicting discount flags need source-owner verification. All results are historical experiments.")

selected = selection.set_index("product_id").loc[pid]
test = analysis["test"].query("product_id == @pid").set_index("model")
chosen = test.loc[selected.selected_model]
baseline = test.loc[selected.baseline_model]
skill = (baseline.mae - chosen.mae) / baseline.mae * 100 if baseline.mae else float("nan")
coverage = analysis["coverage"].set_index("product_id").loc[pid]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected on validation", selected.selected_model)
c2.metric("Final test MAE", f"{chosen.mae:.2f} units")
c3.metric("MAE reduction vs baseline", f"{skill:+.1f}%", help=f"Baseline selected using validation: {selected.baseline_model}. Negative means worse on the final test.")
c4.metric("Test days evaluated", f"{int(coverage.test_scored_days)} / {int(coverage.test_calendar_days)}")

forecast_tab, inventory_tab, price_tab, audit_tab, method_tab = st.tabs([
    "Forecast quality", "Inventory scenarios", "Price & sales", "Data audit", "Method & limits"
])

with forecast_tab:
    st.subheader("An honest final test")
    st.write(f"Compare yesterday, last week, and a Random Forest. The selected model is **{selected.selected_model}**; the baseline is **{selected.baseline_model}**. Both choices use validation data only.")
    predictions = analysis["predictions"].query("product_id == @pid")
    curves = predictions.pivot(index="date", columns="model", values="prediction")
    curves["Observed sales"] = predictions.drop_duplicates("date").set_index("date").actual
    st.line_chart(curves[["Observed sales", selected.selected_model] +
                         ([selected.baseline_model] if selected.baseline_model != selected.selected_model else [])])
    st.caption("One-day-ahead retrospective evaluation: each day may use earlier observed test sales. This is not a single forecast for the whole test period.")
    st.dataframe(test[["selected", "validation_baseline", "mae", "wape_pct", "mase", "bias", "n_days"]].round(3), width="stretch")
    st.caption("MAE: units of average absolute error. WAPE: total absolute error / total observed sales. MASE: error scaled by training-period weekly-naive error. Bias > 0 means overprediction.")
    folds = analysis["validation"].query("product_id == @pid")
    st.subheader("Performance across validation windows")
    st.dataframe(folds[["fold", "model", "train_end", "validation_start", "validation_end", "mae", "n_days"]].round(3), hide_index=True, width="stretch")
    st.download_button("Download final test metrics", analysis["test"].to_csv(index=False), "test_metrics.csv", "text/csv")

with inventory_tab:
    st.subheader("Follow the stock, not just a percentile")
    st.write("Daily orders arrive after the assumed lead time. Stock expires after its shelf life; demand is served FIFO, and unmet units are lost sales. Each policy starts with fresh stock equal to its first order-up-to target.")
    try:
        scenarios, traces = compare_inventory(analysis, pid, InventoryAssumptions(lead, shelf, holding, waste, lost))
        first = scenarios.iloc[0]
        st.caption(f"Simulation window: {first['start'].date()} to {first['end'].date()} ({int(first.days)} days). This is the longest consecutive eligible test run; unknown observations are excluded.")
        st.info("90% and 95% refer to validation-error buffer quantiles. They do not guarantee a 90% or 95% service level. Achieved fill rate is measured below.")
        st.dataframe(scenarios[["policy", "fill_rate_pct", "stockout_free_days_pct", "lost_sales_units", "waste_units", "mean_closing_stock", "scenario_cost"]].round(2), hide_index=True, width="stretch")
        policy = st.selectbox("Inspect a policy's daily stock movement", list(traces))
        st.line_chart(traces[policy][["demand", "closing_stock", "lost_sales", "waste"]])
        st.download_button("Download daily inventory trace", traces[policy].to_csv(), "inventory_trace.csv", "text/csv")
        st.caption("Cost = closing stock × holding cost + expired units × spoilage cost + lost units × lost-sale penalty. Excludes procurement, fixed order costs and terminal-stock valuation. Historical sales are treated as demand; pre-existing stockouts may censor them. The lowest scenario cost is not a proven business optimum.")
    except ValueError as exc:
        st.warning(str(exc))

with price_tab:
    st.subheader("Historical association, with uncertainty")
    st.write("Price tiers summarize observed sales; they do not estimate causal price elasticity. Seasonality, product availability, and promotions may confound the relationship.")
    product = daily[daily.product_id == pid].copy()
    st.scatter_chart(product, x="price", y="sales")
    if product.price.nunique() >= 2:
        product["price_tier"] = pd.qcut(product.price, q=4, duplicates="drop")
        summary = product.groupby("price_tier", observed=True).agg(
            days=("sales", "size"), mean_sales=("sales", "mean"), median_sales=("sales", "median")
        )
        summary.index = summary.index.astype(str)
        st.dataframe(summary.round(2), width="stretch")
        st.caption("Tied price boundaries can produce fewer than four tiers. Daily price is an unweighted mean across distinct records; no price is imputed on missing dates.")
    else:
        st.info("Only one observed price: price-tier comparison is unavailable.")

with audit_tab:
    st.subheader("Check the observation before trusting the metric")
    a, b, c = st.columns(3)
    a.metric("Raw rows", f"{audit['raw_rows']:,}")
    b.metric("Repeated sales records", f"{audit['repeated_sales_records']:,}")
    c.metric("Conflicting discount records", f"{audit['discount_conflict_records']:,}")
    st.write("The default counts identical C520 sales records once. Distinct prices, quantities, change records and statuses remain separate. This addresses a suspected join expansion; it is an assumption pending the source data dictionary.")
    st.dataframe(product_audit.sort_values("raw_sales", ascending=False).round(2), hide_index=True, width="stretch")
    st.subheader("Coverage of the selected products")
    st.dataframe(analysis["coverage"], hide_index=True, width="stretch")
    if not analysis["excluded"].empty:
        st.write("Products excluded by the predeclared coverage rules:")
        st.dataframe(analysis["excluded"], hide_index=True, width="stretch")
    st.caption("Missing dates remain unknown. Evaluation excludes missing outcomes and days whose seven-day feature window is incomplete; this can bias results if missingness is systematic. A recorded zero sale remains a valid observation.")
    st.download_button("Download sales-grain audit", product_audit.to_csv(index=False), "sales_grain_audit.csv", "text/csv")

with method_tab:
    st.subheader("What this experiment establishes")
    st.markdown("""
    - Products are selected using sales before the first validation window.
    - Three expanding chronological validation folds select a model per product.
    - Models are refit on the first 80% of calendar days, then scored on the final 20%.
    - Forecast inputs are yesterday's sales, last week's sales, a shifted seven-day mean, weekday, and month.
    - Same-day prices and disputed discount flags are excluded from forecasting.
    - Inventory uses recursive forecasts from each order origin over lead time + one review day. Buffers use validation horizon errors only.
    - Data source, redistribution permission, record grain, stockout history, shelf life and actual costs are unverified. No real savings or production deployment are claimed.
    """)
    st.caption("This case study demonstrates reproducible analysis and decision trade-offs. It does not prescribe a universal service target or a live ordering policy.")
