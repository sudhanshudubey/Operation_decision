"""Tests for the failure modes that change analytical conclusions."""

import unittest

import numpy as np
import pandas as pd

from retail.data import audit_data, calendar_series, daily_sales, validate_data
from retail.forecast import forecast_horizon, make_features, make_splits, run_evaluation, score
from retail.inventory import InventoryAssumptions, calibrate_buffer, simulate_inventory


def raw_fixture():
    rows = []
    for article, discount in [(10001, 0), (10002, 1)]:
        rows.append({"C520_DATE": "2020-01-01", "C520_STORENR": 1, "C520_ITEMNR": 100,
                     "C520_SOLDPIECES": 10, "C520_RETAILPRICE": 2,
                     "ARTNRVIER": 100, "BEZ_LANG": "Product", "DISCOUNT": discount,
                     "ARTIKELNR": article})
    return validate_data(pd.DataFrame(rows))


class DataTests(unittest.TestCase):
    def test_join_expansion_is_counted_once_and_discount_conflict_is_visible(self):
        raw = raw_fixture()
        audit, _ = audit_data(raw)
        self.assertEqual(audit["repeated_sales_records"], 1)
        self.assertEqual(audit["discount_conflict_records"], 1)
        self.assertEqual(daily_sales(raw).sales.iloc[0], 10)
        self.assertEqual(daily_sales(raw).any_discount_flag.iloc[0], 1)
        self.assertEqual(daily_sales(raw, "raw").sales.iloc[0], 20)

    def test_distinct_sales_records_on_same_date_are_preserved(self):
        raw = raw_fixture()
        extra = raw.iloc[[0]].copy()
        extra["C520_SOLDPIECES"] = 3
        self.assertEqual(daily_sales(pd.concat([raw, extra])).sales.iloc[0], 13)

    def test_missing_date_is_unknown_and_recorded_zero_is_valid(self):
        daily = daily_sales(raw_fixture())
        daily.loc[0, "sales"] = 0
        ts = calendar_series(daily, 100, "2020-01-01", "2020-01-02")
        self.assertEqual(ts.sales.iloc[0], 0)
        self.assertTrue(pd.isna(ts.sales.iloc[1]))

    def test_multiple_stores_are_not_silently_pooled(self):
        raw = raw_fixture()
        raw.loc[1, "C520_STORENR"] = 2
        with self.assertRaisesRegex(ValueError, "one store"):
            validate_data(raw)

    def test_bad_quantity_fails_before_training(self):
        for bad in [-1, np.inf, np.nan]:
            raw = raw_fixture()
            raw["C520_SOLDPIECES"] = raw.C520_SOLDPIECES.astype(float)
            raw.loc[0, "C520_SOLDPIECES"] = bad
            with self.assertRaises(ValueError):
                validate_data(raw)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dates = pd.date_range("2020-01-01", periods=400)
        part = pd.DataFrame({"date": dates, "product_id": 1, "product": "Established",
                             "sales": 20 + dates.dayofweek * 3 + np.arange(400) * 0.01,
                             "price": 1.0, "any_discount_flag": 0})
        late = part.copy()
        late["product_id"], late["product"] = 2, "Future bestseller"
        late["sales"] = np.where(np.arange(400) < 250, 1.0, 10000.0)
        cls.daily = pd.concat([part, late], ignore_index=True)
        cls.result = run_evaluation(cls.daily, top_n=1)

    def test_products_are_selected_before_future_sales_spike(self):
        self.assertEqual(self.result["selection"].product_id.tolist(), [1])

    def test_folds_and_final_test_are_ordered_and_disjoint(self):
        split = self.result["splits"]
        previous_end = None
        for fold in split["folds"]:
            self.assertLess(fold["train_end"], fold["start"])
            self.assertLessEqual(fold["end"], split["development_end"])
            if previous_end is not None:
                self.assertGreater(fold["start"], previous_end)
            previous_end = fold["end"]
        self.assertLess(split["development_end"], split["test_start"])

    def test_features_do_not_read_current_or_future_sales(self):
        sales = self.result["series"][1].sales.copy()
        date = sales.index[100]
        before = make_features(sales).loc[date]
        sales.loc[date:] += 100000
        after = make_features(sales).loc[date]
        pd.testing.assert_series_equal(before.drop("actual"), after.drop("actual"))

    def test_gap_invalidates_lag_window_but_zero_does_not(self):
        sales = pd.Series(0.0, index=pd.date_range("2020-01-01", periods=30))
        sales.iloc[10] = np.nan
        features = make_features(sales)
        self.assertNotIn(sales.index[10], features.index)
        self.assertNotIn(sales.index[17], features.index)
        self.assertIn(sales.index[18], features.index)
        self.assertEqual(features.loc[sales.index[18], "actual"], 0)

    def test_recursive_horizon_cannot_see_future_actuals(self):
        sales = self.result["series"][1].sales.copy()
        date = self.result["splits"]["test_start"]
        forest = self.result["forests"][1]
        for model in ["Yesterday", "Last week", "Random Forest"]:
            before = forecast_horizon(sales, [date], 8, model, forest)
            changed = sales.copy()
            changed.loc[date:] = 999999
            after = forecast_horizon(changed, [date], 8, model, forest)
            pd.testing.assert_frame_equal(before, after)
        weekly = forecast_horizon(sales, [date], 8, "Last week", forest)
        self.assertEqual(weekly.iloc[0, 7], weekly.iloc[0, 0])

    def test_inventory_calibration_cannot_read_test_outcomes(self):
        result = dict(self.result)
        result["series"] = {key: value.copy() for key, value in result["series"].items()}
        before = calibrate_buffer(result, 1, "Random Forest", 3, .95)
        result["series"][1].loc[result["splits"]["test_start"]:, "sales"] = 999999
        after = calibrate_buffer(result, 1, "Random Forest", 3, .95)
        self.assertAlmostEqual(before[0], after[0], places=10)
        self.assertEqual(before[1], after[1])

    def test_changing_test_demand_cannot_change_model_selection(self):
        changed = self.daily.copy()
        changed.loc[changed.date >= self.result["splits"]["test_start"], "sales"] *= 100
        updated = run_evaluation(changed, top_n=1)
        pd.testing.assert_frame_equal(self.result["selection"], updated["selection"])

    def test_scores_have_expected_units_and_bias_direction(self):
        scores = score([1, 3], [2, 4], pd.Series(np.arange(20.0)))
        self.assertEqual(scores["mae"], 1)
        self.assertEqual(scores["bias"], 1)
        self.assertEqual(scores["wape_pct"], 50)
        self.assertAlmostEqual(scores["mase"], 1 / 7)

    def test_all_models_are_evaluated_on_the_same_final_days(self):
        days = self.result["predictions"].groupby("model").date.apply(list)
        self.assertEqual(days.iloc[0], days.iloc[1])
        self.assertEqual(days.iloc[0], days.iloc[2])


class InventoryTests(unittest.TestCase):
    def series(self, values):
        return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values)), dtype=float)

    def test_arrivals_obey_lead_time(self):
        demand = self.series([0, 0, 0, 5])
        targets = self.series([0, 10, 10, 10])
        trace, _ = simulate_inventory(demand, targets, InventoryAssumptions(lead_days=2))
        self.assertEqual(trace.ordered.iloc[1], 10)
        self.assertEqual(trace.received.iloc[2], 0)
        self.assertEqual(trace.received.iloc[3], 10)
        self.assertEqual(trace.fulfilled.iloc[3], 5)

    def test_batches_expire_at_end_of_shelf_life(self):
        trace, metrics = simulate_inventory(self.series([0, 0]), self.series([10, 0]),
                                             InventoryAssumptions(shelf_days=1))
        self.assertEqual(trace.closing_stock.iloc[0], 10)
        self.assertEqual(trace.waste.iloc[1], 10)
        self.assertEqual(metrics["waste_units"], 10)

    def test_lost_sales_are_not_backordered(self):
        trace, metrics = simulate_inventory(self.series([10, 0]), self.series([0, 0]))
        self.assertEqual(trace.lost_sales.iloc[0], 10)
        self.assertEqual(trace.lost_sales.iloc[1], 0)
        self.assertEqual(metrics["fill_rate_pct"], 0)

    def test_stock_and_demand_are_conserved(self):
        trace, m = simulate_inventory(self.series([5, 0, 20, 0, 3, 0, 7]),
                                     self.series([10, 10, 20, 20, 10, 10, 10]),
                                     InventoryAssumptions(shelf_days=2))
        self.assertAlmostEqual(m["initial_stock"] + m["received_units"],
                               trace.fulfilled.sum() + m["waste_units"] + m["ending_stock"])
        np.testing.assert_allclose(trace.demand, trace.fulfilled + trace.lost_sales)
        self.assertTrue((trace[["closing_stock", "waste", "lost_sales"]] >= 0).all().all())

    def test_cost_matches_explicit_assumptions(self):
        trace, metrics = simulate_inventory(self.series([3, 8]), self.series([5, 0]),
                                             InventoryAssumptions(shelf_days=1, holding_cost=1,
                                                                  waste_cost=2, lost_sale_cost=3))
        # Day 0: 2 held. Day 1: 2 expire, 8 sales lost.
        self.assertEqual(metrics["scenario_cost"], 2 + 2 * 2 + 8 * 3)

    def test_unknown_or_nonconsecutive_demand_is_rejected(self):
        with self.assertRaises(ValueError):
            simulate_inventory(self.series([1, np.nan]), self.series([1, 1]))
        demand = self.series([1, 1, 1]).drop(pd.Timestamp("2020-01-02"))
        with self.assertRaises(ValueError):
            simulate_inventory(demand, demand)

    def test_invalid_assumptions_are_rejected(self):
        for kwargs in [{"lead_days": 0}, {"shelf_days": 1.5}, {"holding_cost": -1}]:
            with self.assertRaises(ValueError):
                InventoryAssumptions(**kwargs)


if __name__ == "__main__":
    unittest.main()
