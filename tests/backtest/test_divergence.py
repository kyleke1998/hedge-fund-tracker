import math
import unittest
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from app.backtest.divergence import (
    ACCUMULATION_COLUMNS,
    PRICE_STRENGTH_COLUMNS,
    accumulation_features,
    balanced_share_delta,
    buy_gate,
    divergence_score,
    hit_rate,
    max_drawdown,
    price_features,
    rank_ic,
    rank_z,
    sell_gate,
    split_adjust_factor,
    summarize_windows,
    trend_tstat,
)


class TestSplitAdjustment(unittest.TestCase):
    """
    Reported share counts must be restated in post-split units before any
    quarter-over-quarter comparison, or a split reads as accumulation.
    """

    def test_factor_is_product_of_splits_after_the_reference_date(self):
        splits = [(date(2024, 6, 10), 10.0), (date(2025, 3, 3), 2.0)]
        self.assertEqual(split_adjust_factor(date(2024, 3, 31), splits), 20.0)

    def test_splits_on_or_before_the_reference_date_are_already_reflected(self):
        splits = [(date(2024, 6, 10), 10.0)]
        self.assertEqual(split_adjust_factor(date(2024, 6, 30), splits), 1.0)
        self.assertEqual(split_adjust_factor(date(2024, 6, 10), splits), 1.0)

    def test_no_splits_leaves_shares_untouched(self):
        self.assertEqual(split_adjust_factor(date(2024, 3, 31), []), 1.0)

    def test_reverse_split_shrinks_the_factor(self):
        self.assertAlmostEqual(
            split_adjust_factor(date(2023, 1, 1), [(date(2023, 6, 1), 0.1)]), 0.1
        )


class TestBalancedShareDelta(unittest.TestCase):
    """
    Accumulation must be measured only across funds that filed in both
    quarters: the tracked-fund list grows over time, and counting a newly
    tracked fund's legacy position as a purchase fabricates accumulation.
    """

    def setUp(self):
        self.prev = pd.DataFrame(
            [
                {"Fund": "A", "Ticker": "AAA", "Adj_Shares": 100.0},
                {"Fund": "B", "Ticker": "AAA", "Adj_Shares": 100.0},
                {"Fund": "D", "Ticker": "AAA", "Adj_Shares": 900.0},
            ]
        )
        self.curr = pd.DataFrame(
            [
                {"Fund": "A", "Ticker": "AAA", "Adj_Shares": 150.0},
                {"Fund": "B", "Ticker": "AAA", "Adj_Shares": 50.0},
                {"Fund": "C", "Ticker": "AAA", "Adj_Shares": 5000.0},
            ]
        )

    def test_only_funds_present_in_both_quarters_contribute(self):
        row = balanced_share_delta(self.prev, self.curr).set_index("Ticker").loc["AAA"]
        self.assertEqual(row["Prev_Shares"], 200.0)
        self.assertEqual(row["Curr_Shares"], 200.0)
        self.assertEqual(row["Share_Chg_Pct"], 0.0)

    def test_breadth_counts_common_fund_buyers_against_sellers(self):
        row = balanced_share_delta(self.prev, self.curr).set_index("Ticker").loc["AAA"]
        self.assertEqual(row["Buyers"], 1)
        self.assertEqual(row["Sellers"], 1)
        self.assertEqual(row["Holders"], 2)
        self.assertEqual(row["Breadth"], 0.0)

    def test_position_new_to_every_common_fund_has_no_percentage_change(self):
        prev = pd.DataFrame([{"Fund": "A", "Ticker": "ZZZ", "Adj_Shares": 0.0}])
        curr = pd.DataFrame([{"Fund": "A", "Ticker": "ZZZ", "Adj_Shares": 500.0}])
        row = balanced_share_delta(prev, curr).set_index("Ticker").loc["ZZZ"]
        self.assertTrue(math.isnan(cast(float, row["Share_Chg_Pct"])))
        self.assertTrue(bool(row["All_New"]))

    def test_no_overlapping_funds_yields_an_empty_frame(self):
        prev = pd.DataFrame([{"Fund": "X", "Ticker": "AAA", "Adj_Shares": 10.0}])
        curr = pd.DataFrame([{"Fund": "Y", "Ticker": "AAA", "Adj_Shares": 10.0}])
        self.assertTrue(balanced_share_delta(prev, curr).empty)


class TestAccumulationFeatures(unittest.TestCase):
    """
    Persistence and multi-quarter intensity are what separate "steady
    accumulation" from a single opportunistic add.
    """

    def _panel(self) -> pd.DataFrame:
        rows = []
        # Fund A holds every quarter and adds steadily; fund B holds flat.
        for i, quarter in enumerate(["2024Q1", "2024Q2", "2024Q3", "2024Q4"]):
            rows.append(
                {"Quarter": quarter, "Fund": "A", "Ticker": "UP", "Adj_Shares": 100.0 * (i + 1)}
            )
            rows.append({"Quarter": quarter, "Fund": "B", "Ticker": "UP", "Adj_Shares": 100.0})
            rows.append(
                {"Quarter": quarter, "Fund": "A", "Ticker": "DOWN", "Adj_Shares": 400.0 - 100.0 * i}
            )
            rows.append({"Quarter": quarter, "Fund": "B", "Ticker": "DOWN", "Adj_Shares": 100.0})
        return pd.DataFrame(rows)

    def test_persistence_counts_quarters_of_positive_share_change(self):
        feats = accumulation_features(self._panel(), "2024Q4", lookback=4)
        row = feats.set_index("Ticker").loc["UP"]
        self.assertEqual(row["Acc_Persistence"], 3)
        self.assertEqual(feats.set_index("Ticker").loc["DOWN"]["Acc_Persistence"], 0)

    def test_multi_quarter_change_compounds_the_lookback_window(self):
        feats = accumulation_features(self._panel(), "2024Q4", lookback=4).set_index("Ticker")
        # UP: common-fund shares 200 -> 500 over three transitions.
        self.assertAlmostEqual(cast(float, feats.loc["UP", "Share_Chg_Pct_4q"]), 1.5)
        self.assertAlmostEqual(cast(float, feats.loc["DOWN", "Share_Chg_Pct_4q"]), -0.6)

    def test_the_one_quarter_change_matches_the_balanced_delta(self):
        feats = accumulation_features(self._panel(), "2024Q4", lookback=4).set_index("Ticker")
        self.assertAlmostEqual(cast(float, feats.loc["UP", "Share_Chg_Pct"]), 100.0 / 400.0)

    def test_a_quarter_without_a_predecessor_produces_no_rows(self):
        self.assertTrue(accumulation_features(self._panel(), "2024Q1", lookback=4).empty)


class TestPriceFeatures(unittest.TestCase):
    """
    Every price feature must be computable from bars strictly at or before the
    entry date, or the backtest is looking ahead.
    """

    def setUp(self):
        idx = pd.bdate_range("2023-01-02", "2025-06-30")
        self.prices = pd.DataFrame(
            {
                "AAA": np.linspace(100.0, 200.0, len(idx)),
                "BBB": np.linspace(200.0, 100.0, len(idx)),
            },
            index=idx,
        )
        self.volume = pd.DataFrame(1_000_000.0, index=idx, columns=["AAA", "BBB"])

    def test_features_ignore_bars_after_the_entry_date(self):
        entry = date(2024, 5, 15)
        full = price_features(self.prices, self.volume, entry)
        truncated = price_features(
            self.prices.loc[: pd.Timestamp(entry)], self.volume.loc[: pd.Timestamp(entry)], entry
        )
        pd.testing.assert_frame_equal(full, truncated)

    def test_a_rising_series_scores_positive_trend_and_a_falling_one_negative(self):
        feats = price_features(self.prices, self.volume, date(2024, 5, 15)).set_index("Ticker")
        self.assertGreater(feats.loc["AAA"]["Trend_Tstat"], 0)
        self.assertLess(feats.loc["BBB"]["Trend_Tstat"], 0)
        self.assertGreater(feats.loc["AAA"]["Ret_6m"], 0)
        self.assertLess(feats.loc["BBB"]["Ret_6m"], 0)

    def test_price_versus_moving_average_has_the_sign_of_the_trend(self):
        feats = price_features(self.prices, self.volume, date(2024, 5, 15)).set_index("Ticker")
        self.assertGreater(feats.loc["AAA"]["Px_vs_MA200"], 0)
        self.assertLess(feats.loc["BBB"]["Px_vs_MA200"], 0)

    def test_a_ticker_without_bars_before_the_entry_date_is_dropped(self):
        feats = price_features(self.prices, self.volume, date(2022, 1, 3))
        self.assertTrue(feats.empty)


class TestTrendTstat(unittest.TestCase):
    """
    The trend statistic is a volatility-normalised slope, so "flat" and "noisy
    but flat" both read near zero regardless of the price level.
    """

    def test_a_flat_series_has_no_trend(self):
        self.assertEqual(trend_tstat(pd.Series([50.0] * 40)), 0.0)

    def test_the_statistic_is_invariant_to_price_scale(self):
        base = pd.Series(np.linspace(10.0, 20.0, 60)) + np.sin(np.arange(60))
        self.assertAlmostEqual(trend_tstat(base), trend_tstat(base * 137.0), places=6)

    def test_too_few_observations_return_nan(self):
        self.assertTrue(math.isnan(trend_tstat(pd.Series([1.0, 2.0]))))


class TestRankZ(unittest.TestCase):
    """
    Cross-sectional scores are rank-based so a single outlier cannot dominate
    a composite.
    """

    def test_scores_are_centred_and_ordered(self):
        z = rank_z(pd.Series([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(z.mean(), 0.0)
        self.assertTrue(z.is_monotonic_increasing)

    def test_an_extreme_outlier_does_not_dominate(self):
        z = rank_z(pd.Series([1.0, 2.0, 3.0, 1e12]))
        self.assertAlmostEqual(z.iloc[-1] - z.iloc[-2], z.iloc[1] - z.iloc[0])

    def test_missing_values_stay_missing(self):
        z = rank_z(pd.Series([1.0, np.nan, 3.0]))
        self.assertTrue(math.isnan(z.iloc[1]))
        self.assertFalse(math.isnan(z.iloc[0]))

    def test_a_constant_column_scores_flat_zero(self):
        self.assertTrue((rank_z(pd.Series([5.0, 5.0, 5.0])) == 0.0).all())


class TestDivergenceScore(unittest.TestCase):
    """
    The score is accumulation minus price strength: it must rise when funds
    buy harder and fall when the stock has already run.
    """

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Ticker": ["A", "B", "C", "D"],
                **{col: [1.0, 2.0, 3.0, 4.0] for col in ACCUMULATION_COLUMNS},
                **{col: [1.0, 2.0, 3.0, 4.0] for col in PRICE_STRENGTH_COLUMNS},
            }
        )

    def test_matched_accumulation_and_price_strength_cancel_out(self):
        score = divergence_score(self._frame())
        self.assertTrue(np.allclose(score.to_numpy(), 0.0))

    def test_weak_price_with_strong_accumulation_ranks_highest(self):
        df = self._frame()
        for col in PRICE_STRENGTH_COLUMNS:
            df[col] = [4.0, 3.0, 2.0, 1.0]
        score = divergence_score(df)
        self.assertEqual(score.idxmax(), 3)
        self.assertEqual(score.idxmin(), 0)

    def test_a_row_missing_every_accumulation_input_scores_nan(self):
        df = self._frame()
        for col in ACCUMULATION_COLUMNS:
            df.loc[0, col] = np.nan
        self.assertTrue(math.isnan(divergence_score(df).iloc[0]))


class TestGates(unittest.TestCase):
    """
    The literal reading of the thesis: a buy needs weak-to-flat price *and*
    steady accumulation; both conditions, not a blended score.
    """

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Ticker": ["QUIET_ACC", "RUNNER_ACC", "QUIET_DIST", "RUNNER_DIST"],
                "Ret_6m": [-0.10, 0.60, -0.10, 0.60],
                "Acc_Persistence": [3, 3, 0, 0],
                "Breadth": [0.5, 0.5, -0.5, -0.5],
                "Share_Chg_Pct": [0.20, 0.20, -0.20, -0.20],
                "Holders": [10, 10, 10, 10],
            }
        )

    def test_buy_requires_both_a_soft_price_and_steady_accumulation(self):
        flags = buy_gate(self._frame()).tolist()
        self.assertEqual(flags, [True, False, False, False])

    def test_sell_requires_both_a_strong_price_and_steady_distribution(self):
        flags = sell_gate(self._frame()).tolist()
        self.assertEqual(flags, [False, False, False, True])

    def test_thin_ownership_is_excluded_from_both_gates(self):
        df = self._frame()
        df["Holders"] = 1
        self.assertFalse(buy_gate(df).any())
        self.assertFalse(sell_gate(df).any())


class TestMetrics(unittest.TestCase):
    """
    Reported performance statistics.
    """

    def test_max_drawdown_measures_the_worst_peak_to_trough_fall(self):
        self.assertAlmostEqual(max_drawdown([0.5, -0.4, 0.1]), -0.4)

    def test_a_never_losing_curve_has_no_drawdown(self):
        self.assertEqual(max_drawdown([0.1, 0.2]), 0.0)

    def test_an_empty_track_record_has_no_drawdown(self):
        self.assertEqual(max_drawdown([]), 0.0)

    def test_hit_rate_is_the_share_of_positive_outcomes(self):
        self.assertAlmostEqual(hit_rate([1.0, -1.0, 2.0, 0.0]), 0.5)

    def test_rank_ic_is_one_for_a_monotone_relationship(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(rank_ic(x, x**3), 1.0)
        self.assertAlmostEqual(rank_ic(x, -(x**3)), -1.0)

    def test_rank_ic_needs_at_least_three_paired_observations(self):
        self.assertTrue(math.isnan(rank_ic(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))))

    def test_summary_reports_win_rate_and_benchmark_comparison(self):
        summary = summarize_windows([0.10, -0.05, 0.20], [0.05, 0.05, 0.05], periods_per_year=4)
        self.assertAlmostEqual(summary["win_rate"], 2 / 3)
        self.assertAlmostEqual(summary["beat_benchmark_rate"], 2 / 3)
        self.assertAlmostEqual(summary["cum_return"], 1.10 * 0.95 * 1.20 - 1)
        self.assertAlmostEqual(summary["max_drawdown"], -0.05)
        self.assertEqual(summary["n_windows"], 3)


if __name__ == "__main__":
    unittest.main()
