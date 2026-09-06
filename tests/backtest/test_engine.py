import unittest
from datetime import date

import pandas as pd

from app.analysis.smart_scores import score_core
from app.backtest.engine import (
    Benchmark,
    build_screen,
    min_holders_for_quarter,
    quarter_entry_date,
    quarter_smart_scores,
    run_backtest,
)
from app.backtest.strategies import strategy_by_id


def _analysis(rows: list[dict]) -> pd.DataFrame:
    """
    Build a stock-level analysis frame with the columns strategies rank on.
    """
    return pd.DataFrame(rows)


ANALYSIS = {
    "2025Q1": _analysis(
        [
            {
                "Ticker": "AAA",
                "Holder_Count": 20,
                "Avg_Portfolio_Pct": 10.0,
                "Max_Portfolio_Pct": 40.0,
            },
            {
                "Ticker": "BBB",
                "Holder_Count": 16,
                "Avg_Portfolio_Pct": 5.0,
                "Max_Portfolio_Pct": 60.0,
            },
            {
                "Ticker": "CCC",
                "Holder_Count": 10,
                "Avg_Portfolio_Pct": 8.0,
                "Max_Portfolio_Pct": 90.0,
            },
        ]
    ),
    "2025Q2": _analysis(
        [
            {
                "Ticker": "AAA",
                "Holder_Count": 18,
                "Avg_Portfolio_Pct": 6.0,
                "Max_Portfolio_Pct": 30.0,
            },
            {
                "Ticker": "DDD",
                "Holder_Count": 15,
                "Avg_Portfolio_Pct": 6.0,
                "Max_Portfolio_Pct": 20.0,
            },
        ]
    ),
}

# Derived from the engine rather than restated, so the fixture follows
# FILING_LAG_DAYS instead of pinning yesterday's entry dates.
ENTRY_Q1 = quarter_entry_date("2025Q1").isoformat()
ENTRY_Q2 = quarter_entry_date("2025Q2").isoformat()
ENTRY_Q3 = quarter_entry_date("2025Q3").isoformat()

PRICES = {
    ("AAA", ENTRY_Q1): 100.0,
    ("BBB", ENTRY_Q1): 50.0,
    ("SPY", ENTRY_Q1): 400.0,
    ("QQQ", ENTRY_Q1): 300.0,
    ("AAA", ENTRY_Q2): 120.0,  # +20%
    ("BBB", ENTRY_Q2): 55.0,  # +10%
    ("SPY", ENTRY_Q2): 420.0,  # +5%
    ("QQQ", ENTRY_Q2): 330.0,  # +10%
    ("AAA", ENTRY_Q3): 132.0,  # +10% over window 2
    ("DDD", ENTRY_Q2): 60.0,
    ("DDD", ENTRY_Q3): 66.0,  # +10%
    ("SPY", ENTRY_Q3): 441.0,  # +5%
    ("QQQ", ENTRY_Q3): 363.0,  # +10%
}

BENCHES = [Benchmark("SPY", "S&P 500"), Benchmark("QQQ", "Nasdaq 100")]


def _price(ticker: str, day: date) -> float | None:
    """
    Deterministic price lookup over the PRICES table.
    """
    return PRICES.get((ticker, day.isoformat()))


def _analysis_fn(quarter: str) -> pd.DataFrame:
    """
    Return the canned analysis frame for a quarter.
    """
    return ANALYSIS[quarter]


def _run(as_of: date, strategy_ids=("avg_portfolio",)):
    """
    Run the backtest over the canned data with a fixed 150-fund universe (threshold 15).
    """
    return run_backtest(
        price_fn=_price,
        as_of=as_of,
        analysis_fn=_analysis_fn,
        quarters=["2025Q1", "2025Q2", "2025Q3"],
        strategies=[strategy_by_id(sid) for sid in strategy_ids],
        benchmarks=BENCHES,
        fund_count_fn=lambda _q: 150,  # -> threshold 15
    )


class TestBuildScreen(unittest.TestCase):
    """
    Tests for per-strategy screen reconstruction + weighting.
    """

    def test_avg_portfolio_weights_normalized(self):
        """
        Avg Portfolio screen filters by holders and weights by Avg_Portfolio_Pct.
        """
        weights = build_screen(
            "2025Q1", strategy_by_id("avg_portfolio"), threshold=15, analysis_fn=_analysis_fn
        )
        self.assertEqual(set(weights), {"AAA", "BBB"})
        self.assertAlmostEqual(weights["AAA"], 10.0 / 15.0)
        self.assertAlmostEqual(weights["BBB"], 5.0 / 15.0)

    def test_big_bets_selects_by_max_pct(self):
        """
        Big Bets (top-N, no min holders) ranks by Max_Portfolio_Pct.
        """
        weights = build_screen(
            "2025Q1", strategy_by_id("big_bets"), threshold=15, analysis_fn=_analysis_fn, top_n=2
        )
        self.assertEqual(set(weights), {"CCC", "BBB"})  # 90, 60


class TestHelpers(unittest.TestCase):
    """
    Tests for the date + threshold helpers.
    """

    def test_entry_is_the_day_after_the_filing_deadline(self):
        """
        Entry is quarter-end + 46 days, i.e. the first session *after* the day-45
        13F deadline. Buying on day 45 itself would transact on a screen that
        includes filings which may only have been published after that close.
        """
        self.assertEqual(quarter_entry_date("2025Q1"), date(2025, 5, 16))

    def test_min_holders_is_ceil_of_ten_percent(self):
        """
        Threshold is ceil(funds/10), matching the QuarterlyTrends UI default.
        """
        self.assertEqual(min_holders_for_quarter("q", fund_count_fn=lambda _q: 125), 13)
        self.assertEqual(min_holders_for_quarter("q", fund_count_fn=lambda _q: 123), 13)
        self.assertEqual(min_holders_for_quarter("q", fund_count_fn=lambda _q: 120), 12)


class TestRunBacktest(unittest.TestCase):
    """
    Tests for the windowed multi-series backtest.
    """

    def test_consolidated_only(self):
        """
        Windows whose exit has not matured are excluded.
        """
        rows = _run(date(2025, 6, 1))
        self.assertEqual(rows, [])

    def test_emits_benchmark_and_strategy_rows(self):
        """
        One matured window emits both benchmark series plus each strategy.
        """
        rows = _run(date(2025, 9, 1))
        kinds = {(r["series_type"], r["series_id"]) for r in rows}
        self.assertIn(("benchmark", "SPY"), kinds)
        self.assertIn(("benchmark", "QQQ"), kinds)
        self.assertIn(("strategy", "avg_portfolio"), kinds)

    def test_strategy_return_and_excess(self):
        """
        Strategy window return is conviction-weighted; excess is vs the anchor (SPY).
        """
        rows = _run(date(2025, 9, 1))
        strat = next(r for r in rows if r["series_type"] == "strategy")
        self.assertAlmostEqual(strat["window_return"], 0.16667, places=4)
        self.assertAlmostEqual(strat["excess_return"], 0.11667, places=4)
        self.assertEqual(strat["n_stocks"], 2)
        spy = next(r for r in rows if r["series_id"] == "SPY")
        self.assertAlmostEqual(spy["window_return"], 0.05, places=4)
        self.assertIsNone(spy["excess_return"])

    def test_cumulative_compounds(self):
        """
        Cumulative returns compound per series across windows.
        """
        rows = _run(date(2026, 1, 1))
        strat_w2 = [r for r in rows if r["series_type"] == "strategy"][-1]
        # window 2: AAA(+10%) only priced (BBB absent in Q2 screen) -> conv2 = 0.10
        self.assertAlmostEqual(strat_w2["window_return"], 0.10, places=4)
        expected = (1 + 0.16667) * (1 + 0.10) - 1
        self.assertAlmostEqual(strat_w2["cum_return"], expected, places=3)

    def test_multi_strategy_distinct_rows(self):
        """
        Each requested strategy produces its own series rows.
        """
        rows = _run(date(2025, 9, 1), strategy_ids=("avg_portfolio", "big_bets"))
        strat_ids = {r["series_id"] for r in rows if r["series_type"] == "strategy"}
        self.assertEqual(strat_ids, {"avg_portfolio", "big_bets"})


if __name__ == "__main__":
    unittest.main()


class TestQuarterSmartScores(unittest.TestCase):
    """
    The smart score the divergence study evaluates has to be the *published*
    score, which means computing it on the whole quarter's universe. Its three
    components are percentile ranks, so scoring a filtered subset would silently
    produce a different number for the same stock in the same quarter.
    """

    FRAME = pd.DataFrame(
        [
            {"Ticker": "AAA", "Holder_Count": 40, "Net_Buyers": 20, "Avg_Portfolio_Pct": 8.0},
            {"Ticker": "BBB", "Holder_Count": 30, "Net_Buyers": 10, "Avg_Portfolio_Pct": 4.0},
            {"Ticker": "CCC", "Holder_Count": 20, "Net_Buyers": 0, "Avg_Portfolio_Pct": 2.0},
            {"Ticker": "DDD", "Holder_Count": 10, "Net_Buyers": -10, "Avg_Portfolio_Pct": 1.0},
        ]
    )

    def test_scores_are_returned_per_ticker_in_the_one_to_ten_band(self):
        scores = quarter_smart_scores("2025Q1", analysis_fn=lambda _: self.FRAME)
        self.assertEqual(sorted(scores.index), ["AAA", "BBB", "CCC", "DDD"])
        self.assertTrue(((scores >= 1.0) & (scores <= 10.0)).all())

    def test_ranking_follows_the_institutional_components(self):
        scores = quarter_smart_scores("2025Q1", analysis_fn=lambda _: self.FRAME)
        self.assertEqual(scores.idxmax(), "AAA")
        self.assertEqual(scores.idxmin(), "DDD")

    def test_the_score_reflects_the_full_universe_not_a_subset(self):
        full = quarter_smart_scores("2025Q1", analysis_fn=lambda _: self.FRAME)
        subset = quarter_smart_scores(
            "2025Q1", analysis_fn=lambda _: self.FRAME[self.FRAME["Ticker"].isin(["CCC", "DDD"])]
        )
        self.assertNotAlmostEqual(full["CCC"], subset["CCC"])

    def test_an_empty_quarter_produces_an_empty_series(self):
        empty = quarter_smart_scores("2025Q1", analysis_fn=lambda _: self.FRAME.iloc[0:0])
        self.assertTrue(empty.empty)


class TestQuarterSmartScoresDeduplication(unittest.TestCase):
    """
    The stock-level frame groups by ticker *and* company, so one ticker carried
    under two company spellings ("DoorDash Inc" / "Doordash Inc") splits into two
    rows that each understate its institutional footprint. The score has to come
    back one row per ticker, and from the row that actually describes the stock.
    """

    SPLIT_NAME = pd.DataFrame(
        [
            {
                "Ticker": "DASH",
                "Company": "DoorDash Inc",
                "Holder_Count": 1,
                "Net_Buyers": 1,
                "Avg_Portfolio_Pct": 0.05,
            },
            {
                "Ticker": "DASH",
                "Company": "Doordash Inc",
                "Holder_Count": 15,
                "Net_Buyers": -4,
                "Avg_Portfolio_Pct": 1.69,
            },
            {
                "Ticker": "AAA",
                "Company": "A Co",
                "Holder_Count": 8,
                "Net_Buyers": 2,
                "Avg_Portfolio_Pct": 0.9,
            },
        ]
    )

    def test_a_ticker_split_across_spellings_returns_one_score(self):
        scores = quarter_smart_scores("2026Q2", analysis_fn=lambda _: self.SPLIT_NAME)
        self.assertEqual(scores.index.tolist().count("DASH"), 1)

    def test_the_surviving_score_is_the_stronger_of_the_split_rows(self):
        frame = self.SPLIT_NAME
        both = score_core(frame)[frame["Ticker"].to_numpy() == "DASH"]
        scores = quarter_smart_scores("2026Q2", analysis_fn=lambda _: frame)
        self.assertAlmostEqual(scores["DASH"], float(both.max()))
        self.assertNotAlmostEqual(scores["DASH"], float(both.min()))
