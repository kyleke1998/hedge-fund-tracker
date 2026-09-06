import unittest
from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd

from app.analysis.cost_basis import (
    Bar,
    build_position_panel,
    estimate_cost_basis,
    normalise_share_basis,
)

QUARTER_STARTS = {
    "2024Q1": date(2024, 1, 1),
    "2024Q2": date(2024, 4, 1),
    "2024Q3": date(2024, 7, 1),
    "2024Q4": date(2024, 10, 1),
}


def bars_for(
    quarter: str,
    prices: list[float] | float,
    volume: float = 1_000_000.0,
    days: int = 60,
) -> list[Bar]:
    """
    Business-day bars covering one quarter, at a flat price or a given ramp.
    """
    start = QUARTER_STARTS[quarter]
    series = prices if isinstance(prices, list) else [float(prices)] * days
    out: list[Bar] = []
    day = start
    for price in series:
        while day.weekday() >= 5:
            day += timedelta(days=1)
        out.append(Bar(day=day, price=price, volume=volume))
        day += timedelta(days=1)
    return out


def ramp(low: float, high: float, days: int = 60) -> list[float]:
    """
    A linear price path from `low` to `high` over `days` bars.
    """
    return [low + (high - low) * i / (days - 1) for i in range(days)]


def panel(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """
    Position panel from (quarter, fund, shares, implied price) tuples.
    """
    return pd.DataFrame(rows, columns=["Quarter", "Fund", "Shares", "Price"])


class TestEstimateCostBasis(unittest.TestCase):
    def test_empty_panel_yields_no_points(self):
        self.assertEqual(estimate_cost_basis(panel([]), []), [])

    def test_flat_quarter_prices_collapse_the_band_to_that_price(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0)]),
            bars_for("2024Q1", 10.0),
            n_sims=200,
        )
        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertAlmostEqual(point.low, 10.0, places=6)
        self.assertAlmostEqual(point.mid, 10.0, places=6)
        self.assertAlmostEqual(point.high, 10.0, places=6)
        self.assertEqual(point.shares, 1000.0)
        self.assertEqual(point.holders, 1)
        self.assertEqual(point.as_of, "2024-03-31")

    def test_band_stays_inside_the_quarter_price_range(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 20.0)]),
            bars_for("2024Q1", ramp(10.0, 20.0)),
            n_sims=400,
        )
        point = points[0]
        self.assertLess(point.low, point.mid)
        self.assertLess(point.mid, point.high)
        self.assertGreaterEqual(point.low, 10.0)
        self.assertLessEqual(point.high, 20.0)

    def test_holding_through_a_rally_leaves_the_basis_untouched(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q2", "A", 1000.0, 50.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 50.0),
            n_sims=200,
        )
        self.assertAlmostEqual(points[1].mid, 10.0, places=6)

    def test_a_second_vintage_averages_into_the_basis(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q2", "A", 2000.0, 20.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 20.0),
            n_sims=200,
        )
        self.assertAlmostEqual(points[1].mid, 15.0, places=6)
        self.assertEqual(points[1].shares, 2000.0)

    def test_selling_does_not_move_the_average_cost(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q2", "A", 400.0, 50.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 50.0),
            n_sims=200,
        )
        self.assertAlmostEqual(points[1].mid, 10.0, places=6)
        self.assertEqual(points[1].shares, 400.0)

    def test_aggregate_weights_each_fund_by_its_surviving_shares(self):
        points = estimate_cost_basis(
            panel(
                [
                    ("2024Q1", "A", 1000.0, 10.0),
                    ("2024Q2", "A", 1000.0, 20.0),
                    ("2024Q2", "B", 3000.0, 20.0),
                ]
            ),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 20.0),
            n_sims=200,
        )
        self.assertAlmostEqual(points[1].mid, 17.5, places=6)
        self.assertEqual(points[1].holders, 2)

    def test_a_position_too_large_to_rush_prices_near_the_quarter_vwap(self):
        bars = bars_for("2024Q1", 15.0) + bars_for("2024Q2", ramp(10.0, 20.0), volume=1_000_000.0)
        quarters = ["2024Q1", "2024Q2"]
        small = estimate_cost_basis(
            panel([("2024Q2", "A", 50_000.0, 20.0)]), bars, quarters, n_sims=400
        )[0]
        large = estimate_cost_basis(
            panel([("2024Q2", "A", 6_000_000.0, 20.0)]), bars, quarters, n_sims=400
        )[0]
        self.assertLess(large.high - large.low, small.high - small.low)

    def test_a_large_seeded_position_keeps_a_diffuse_vintage_band(self):
        points = estimate_cost_basis(
            panel([("2024Q2", "A", 6_000_000.0, 15.0)]),
            bars_for("2024Q1", ramp(10.0, 20.0), volume=1_000_000.0) + bars_for("2024Q2", 50.0),
            quarters=["2024Q2"],
            n_sims=400,
        )
        point = points[0]
        self.assertGreater(point.high - point.low, 4.0)
        self.assertGreater(point.mid, 13.0)
        self.assertLess(point.mid, 18.0)

    def test_buys_carry_a_timing_bias_above_the_plain_vwap(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q2", "A", 6_001_000.0, 15.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", ramp(10.0, 20.0), volume=1_000_000.0),
            n_sims=200,
        )
        self.assertGreater(points[1].mid, 15.2)
        self.assertLess(points[1].mid, 16.5)

    def test_pre_history_holdings_are_reported_as_seeded(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q2", "A", 2000.0, 20.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 20.0),
            n_sims=200,
        )
        self.assertAlmostEqual(points[0].seeded_pct, 100.0, places=6)
        self.assertAlmostEqual(points[1].seeded_pct, 50.0, places=6)

    def test_seed_vintage_is_drawn_from_prices_before_the_first_quarter(self):
        points = estimate_cost_basis(
            panel([("2024Q2", "A", 1000.0, 50.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 50.0),
            quarters=["2024Q2"],
            n_sims=200,
        )
        self.assertAlmostEqual(points[0].mid, 10.0, places=6)

    def test_a_quarter_with_no_bars_falls_back_to_the_filed_price(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q2", "A", 2000.0, 30.0)]),
            bars_for("2024Q1", 10.0),
            quarters=["2024Q1", "2024Q2"],
            n_sims=200,
        )
        self.assertAlmostEqual(points[1].mid, 20.0, places=6)

    def test_a_fund_that_skipped_a_filing_keeps_its_position(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q3", "A", 1000.0, 30.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 20.0) + bars_for("2024Q3", 30.0),
            quarters=["2024Q1", "2024Q2", "2024Q3"],
            n_sims=200,
        )
        self.assertEqual([p.quarter for p in points], ["2024Q1", "2024Q2", "2024Q3"])
        self.assertEqual([p.shares for p in points], [1000.0, 1000.0, 1000.0])
        self.assertAlmostEqual(points[2].mid, 10.0, places=6)

    def test_a_close_row_retires_the_position(self):
        points = estimate_cost_basis(
            panel(
                [
                    ("2024Q1", "A", 1000.0, 10.0),
                    ("2024Q2", "A", 0.0, 0.0),
                    ("2024Q3", "A", 1000.0, 30.0),
                ]
            ),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 20.0) + bars_for("2024Q3", 30.0),
            n_sims=200,
        )
        self.assertEqual([p.quarter for p in points], ["2024Q1", "2024Q3"])
        self.assertAlmostEqual(points[1].mid, 30.0, places=6)

    def test_quarters_with_no_holders_are_omitted(self):
        points = estimate_cost_basis(
            panel([("2024Q1", "A", 1000.0, 10.0), ("2024Q2", "A", 0.0, 0.0)]),
            bars_for("2024Q1", 10.0) + bars_for("2024Q2", 20.0),
            n_sims=200,
        )
        self.assertEqual([p.quarter for p in points], ["2024Q1"])

    def test_estimates_are_reproducible_for_a_given_seed(self):
        args = (panel([("2024Q1", "A", 1000.0, 20.0)]), bars_for("2024Q1", ramp(10.0, 20.0)))
        first = estimate_cost_basis(*args, n_sims=200, seed=7)
        second = estimate_cost_basis(*args, n_sims=200, seed=7)
        self.assertEqual([p.mid for p in first], [p.mid for p in second])


class TestNormaliseShareBasis(unittest.TestCase):
    def test_a_pre_split_quarter_is_restated_onto_the_market_basis(self):
        out = normalise_share_basis(
            panel(
                [
                    ("2024Q1", "A", 100.0, 500.0),
                    ("2024Q1", "B", 200.0, 500.0),
                    ("2024Q1", "C", 300.0, 500.0),
                    ("2024Q2", "A", 1000.0, 55.0),
                ]
            ),
            {"2024Q1": 50.0, "2024Q2": 55.0},
        )
        q1 = out[out["Quarter"] == "2024Q1"]
        self.assertEqual(sorted(q1["Shares"].tolist()), [1000.0, 2000.0, 3000.0])
        self.assertTrue((q1["Price"] == 50.0).all())
        self.assertEqual(out.loc[out["Quarter"] == "2024Q2", "Shares"].tolist(), [1000.0])

    def test_ordinary_price_drift_is_left_alone(self):
        out = normalise_share_basis(
            panel(
                [
                    ("2024Q1", "A", 100.0, 52.0),
                    ("2024Q1", "B", 200.0, 52.0),
                    ("2024Q1", "C", 300.0, 52.0),
                ]
            ),
            {"2024Q1": 50.0},
        )
        self.assertEqual(out["Shares"].tolist(), [100.0, 200.0, 300.0])

    def test_a_single_fund_on_the_wrong_basis_is_harmonised(self):
        out = normalise_share_basis(
            panel(
                [
                    ("2024Q1", "A", 100.0, 500.0),
                    ("2024Q1", "B", 200.0, 50.0),
                    ("2024Q1", "C", 300.0, 50.0),
                    ("2024Q1", "D", 400.0, 50.0),
                ]
            ),
            {"2024Q1": 50.0},
        )
        shares = dict(zip(out["Fund"], out["Shares"], strict=True))
        self.assertEqual(shares["A"], 1000.0)
        self.assertEqual(shares["B"], 200.0)


QUARTER_ROWS = {
    "2024Q1": [
        ("037833100", "AAPL", 100.0, "1.00M", "Alpha"),
        ("037833100", "AAPL", 50.0, "0.50M", "Beta"),
        ("67066G104", "NVDA", 10.0, "0.10M", "Alpha"),
    ],
    "2024Q2": [
        ("03783310X", "AAPL", 200.0, "2.00M", "Alpha"),
        ("037833100", "AAPL", 100.0, "1.00M", "Alpha"),
    ],
}


def fake_quarterly_data(quarter: str) -> pd.DataFrame:
    """
    Stand-in for the on-disk quarter loader, in the raw CSV column layout.
    """
    return pd.DataFrame(
        QUARTER_ROWS.get(quarter, []), columns=["CUSIP", "Ticker", "Shares", "Value", "Fund"]
    )


def fake_stocks() -> pd.DataFrame:
    """
    CUSIP-indexed ticker master, with two CUSIPs mapping to the same ticker.
    """
    return pd.DataFrame(
        {"Ticker": ["AAPL", "AAPL", "NVDA"]},
        index=pd.Index(["037833100", "03783310X", "67066G104"], name="CUSIP"),
    )


class TestBuildPositionPanel(unittest.TestCase):
    def setUp(self):
        patcher = patch.multiple(
            "app.analysis.cost_basis",
            load_quarterly_data=fake_quarterly_data,
            load_stocks=fake_stocks,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_one_row_per_fund_and_quarter_with_the_filed_price(self):
        out = build_position_panel("AAPL", ["2024Q1", "2024Q2"])
        rows = {(q, f): (s, p) for q, f, s, p in out.itertuples(index=False)}
        self.assertEqual(rows[("2024Q1", "Alpha")], (100.0, 10_000.0))
        self.assertEqual(rows[("2024Q1", "Beta")], (50.0, 10_000.0))

    def test_multiple_cusips_of_one_ticker_collapse_into_a_single_position(self):
        out = build_position_panel("AAPL", ["2024Q2"])
        self.assertEqual(out["Shares"].tolist(), [300.0])
        self.assertEqual(out["Price"].tolist(), [10_000.0])

    def test_other_tickers_are_excluded(self):
        self.assertTrue(build_position_panel("TSLA", ["2024Q1"]).empty)


if __name__ == "__main__":
    unittest.main()
