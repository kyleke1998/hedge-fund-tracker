import math
import unittest
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from app.backtest.study import (
    asof_prices,
    daily_equity,
    double_sort_table,
    forward_returns,
    neutralize,
    quantile_table,
)

INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
CLOSE = pd.DataFrame(
    {
        "AAA": [100.0, 110.0, 120.0, 130.0, 200.0],
        "BBB": [50.0, 50.0, 50.0, 50.0, 50.0],
        "CCC": [np.nan, np.nan, np.nan, 10.0, 12.0],
    },
    index=INDEX,
)


class TestAsofPrices(unittest.TestCase):
    """
    Entry and exit dates are calendar dates that often are not trading days, so
    every lookup resolves to the last bar at or before the date.
    """

    def test_a_trading_day_resolves_to_its_own_bar(self):
        self.assertEqual(asof_prices(CLOSE, date(2024, 1, 4))["AAA"], 120.0)

    def test_a_non_trading_day_resolves_to_the_previous_bar(self):
        self.assertEqual(asof_prices(CLOSE, date(2024, 1, 6))["AAA"], 130.0)

    def test_a_date_before_any_bar_has_no_price(self):
        self.assertTrue(math.isnan(asof_prices(CLOSE, date(2023, 12, 31))["AAA"]))

    def test_a_ticker_whose_history_starts_later_has_no_price_yet(self):
        self.assertTrue(math.isnan(asof_prices(CLOSE, date(2024, 1, 3))["CCC"]))


class TestForwardReturns(unittest.TestCase):
    """
    The holding-period return is entry-to-exit on as-of prices, and must be
    missing rather than zero when either leg is unavailable.
    """

    def test_the_return_spans_entry_to_exit(self):
        returns = forward_returns(CLOSE, date(2024, 1, 2), date(2024, 1, 8))
        self.assertAlmostEqual(returns["AAA"], 1.0)
        self.assertAlmostEqual(returns["BBB"], 0.0)

    def test_a_missing_entry_price_yields_no_return(self):
        returns = forward_returns(CLOSE, date(2024, 1, 2), date(2024, 1, 8))
        self.assertTrue(math.isnan(returns["CCC"]))


class TestDailyEquity(unittest.TestCase):
    """
    A daily curve is what makes the drawdown honest: a quarterly curve cannot
    see a fall that recovered before the next rebalance.
    """

    def test_a_buy_and_hold_basket_compounds_its_constituents(self):
        curve = daily_equity(CLOSE, {"AAA": 0.5, "BBB": 0.5}, date(2024, 1, 2), date(2024, 1, 8))
        self.assertAlmostEqual(curve.iloc[0], 1.0)
        self.assertAlmostEqual(curve.iloc[-1], 1.5)

    def test_the_curve_covers_the_holding_window_only(self):
        curve = daily_equity(CLOSE, {"AAA": 1.0}, date(2024, 1, 3), date(2024, 1, 5))
        self.assertEqual(list(curve.index), list(INDEX[1:4]))

    def test_weights_are_renormalised_over_the_priceable_names(self):
        curve = daily_equity(CLOSE, {"AAA": 0.5, "CCC": 0.5}, date(2024, 1, 2), date(2024, 1, 8))
        self.assertAlmostEqual(curve.iloc[-1], 2.0)

    def test_a_basket_with_no_priceable_name_has_no_curve(self):
        self.assertTrue(daily_equity(CLOSE, {"ZZZ": 1.0}, date(2024, 1, 2), date(2024, 1, 8)).empty)


class TestQuantileTable(unittest.TestCase):
    """
    Monotonicity across score buckets is the evidence that a signal is graded
    rather than driven by a handful of names at one extreme.
    """

    def setUp(self):
        self.df = pd.DataFrame({"score": np.arange(1.0, 21.0), "fwd": np.arange(1.0, 21.0) / 100.0})

    def test_buckets_are_ordered_worst_to_best_by_mean_outcome(self):
        table = quantile_table(self.df, "score", "fwd", bins=4)
        self.assertEqual(len(table), 4)
        self.assertTrue(table["mean_return"].is_monotonic_increasing)

    def test_each_bucket_reports_its_size_and_hit_rate(self):
        table = quantile_table(self.df, "score", "fwd", bins=4)
        self.assertEqual(table["n"].tolist(), [5, 5, 5, 5])
        self.assertTrue((table["hit_rate"] == 1.0).all())

    def test_rows_missing_a_score_or_an_outcome_are_ignored(self):
        df = self.df.copy()
        df.loc[0, "fwd"] = np.nan
        df.loc[1, "score"] = np.nan
        self.assertEqual(quantile_table(df, "score", "fwd", bins=2)["n"].sum(), 18)

    def test_too_few_observations_produce_an_empty_table(self):
        self.assertTrue(quantile_table(self.df.head(2), "score", "fwd", bins=5).empty)


if __name__ == "__main__":
    unittest.main()


class TestDoubleSort(unittest.TestCase):
    """
    A conditional table is the direct test of the divergence thesis: it asks what
    accumulation is worth *given* the price trend, which a single blended score
    cannot separate.
    """

    def setUp(self):
        grid = [
            {"acc": acc, "px": px, "fwd": 0.01 * acc - 0.02 * px}
            for acc in range(1, 5)
            for px in range(1, 5)
            for _ in range(3)
        ]
        self.df = pd.DataFrame(grid)

    def test_the_table_is_a_grid_of_the_requested_size(self):
        table = double_sort_table(self.df, "acc", "px", "fwd", bins=2)
        self.assertEqual(table["mean_return"].shape, (2, 2))
        self.assertEqual(table["count"].to_numpy().sum(), len(self.df))

    def test_cells_move_with_both_sort_dimensions(self):
        table = double_sort_table(self.df, "acc", "px", "fwd", bins=2)
        means = table["mean_return"]
        self.assertGreater(cast(float, means.iloc[1, 0]), cast(float, means.iloc[0, 0]))
        self.assertLess(cast(float, means.iloc[0, 1]), cast(float, means.iloc[0, 0]))

    def test_rows_missing_any_input_are_ignored(self):
        df = self.df.copy()
        df.loc[0, "fwd"] = np.nan
        self.assertEqual(
            double_sort_table(df, "acc", "px", "fwd", bins=2)["count"].to_numpy().sum(),
            len(df) - 1,
        )

    def test_too_few_observations_produce_an_empty_result(self):
        self.assertEqual(double_sort_table(self.df.head(3), "acc", "px", "fwd", bins=5), {})


class TestNeutralize(unittest.TestCase):
    """
    Sector-demeaning separates a genuine stock-selection edge from a bet that one
    sector happened to work.
    """

    def test_values_are_expressed_relative_to_their_group_mean(self):
        df = pd.DataFrame({"sector": ["A", "A", "B", "B"], "x": [1.0, 3.0, 10.0, 20.0]})
        out = neutralize(df, ["x"], "sector")
        self.assertEqual(out["x"].tolist(), [-1.0, 1.0, -5.0, 5.0])

    def test_a_group_of_one_collapses_to_zero(self):
        df = pd.DataFrame({"sector": ["A"], "x": [7.0]})
        self.assertEqual(neutralize(df, ["x"], "sector")["x"].tolist(), [0.0])

    def test_rows_without_a_group_are_pooled_rather_than_dropped(self):
        df = pd.DataFrame({"sector": [None, None, "A"], "x": [1.0, 3.0, 9.0]})
        out = neutralize(df, ["x"], "sector")
        self.assertEqual(len(out), 3)
        self.assertEqual(out["x"].iloc[0], -1.0)
