import unittest
from datetime import date
from typing import cast

import pandas as pd

from app.backtest.holdings_panel import (
    apply_split_adjustment,
    build_holdings_panel,
    universe_tickers,
)

STOCKS = pd.DataFrame(
    {"Ticker": ["AAA", "BBB", None], "Company": ["A Co", "B Co", "Untracked"]},
    index=pd.Index(["111111111", "222222222", "999999999"], name="CUSIP"),
)

QUARTERS = {
    "2024Q1": pd.DataFrame(
        [
            {
                "CUSIP": "111111111",
                "Ticker": "OLD",
                "Company": "x",
                "Fund": "F1",
                "Shares": 100,
                "Value": "1.00M",
                "Portfolio%": "2.0%",
            },
            {
                "CUSIP": "222222222",
                "Ticker": "OLD",
                "Company": "x",
                "Fund": "F1",
                "Shares": 0,
                "Value": "0",
                "Portfolio%": "0%",
            },
            {
                "CUSIP": "999999999",
                "Ticker": "OLD",
                "Company": "x",
                "Fund": "F1",
                "Shares": 50,
                "Value": "0.50M",
                "Portfolio%": "1.0%",
            },
        ]
    ),
    "2024Q2": pd.DataFrame(
        [
            {
                "CUSIP": "111111111",
                "Ticker": "OLD",
                "Company": "x",
                "Fund": "F1",
                "Shares": 60,
                "Value": "0.60M",
                "Portfolio%": "1.0%",
            },
            {
                "CUSIP": "111111111",
                "Ticker": "OLD",
                "Company": "x",
                "Fund": "F2",
                "Shares": 40,
                "Value": "0.40M",
                "Portfolio%": "3.0%",
            },
            {
                "CUSIP": "222222222",
                "Ticker": "OLD",
                "Company": "x",
                "Fund": "F2",
                "Shares": 10,
                "Value": "0.10M",
                "Portfolio%": "0.5%",
            },
        ]
    ),
}


def _panel() -> pd.DataFrame:
    return build_holdings_panel(
        quarters=["2024Q1", "2024Q2"],
        load_fn=lambda q: QUARTERS[q].copy(),
        stocks_fn=lambda: STOCKS,
    )


class TestBuildHoldingsPanel(unittest.TestCase):
    """
    The panel is the study's raw material: one row per fund/ticker/quarter with
    share counts that survive CUSIP changes and formatted-value parsing.
    """

    def test_tickers_come_from_the_master_stock_list_not_the_filing(self):
        panel = _panel()
        self.assertEqual(set(panel["Ticker"]), {"AAA", "BBB"})

    def test_zero_share_and_unmapped_rows_are_excluded(self):
        panel = _panel()
        self.assertEqual(len(panel[panel["Quarter"] == "2024Q1"]), 1)

    def test_multiple_cusips_for_one_ticker_collapse_into_a_single_row(self):
        stocks = STOCKS.copy()
        stocks.loc["222222222", "Ticker"] = "AAA"
        panel = build_holdings_panel(
            quarters=["2024Q2"], load_fn=lambda q: QUARTERS[q].copy(), stocks_fn=lambda: stocks
        )
        indexed = panel.set_index(["Fund", "Ticker"])
        self.assertEqual(cast(float, indexed.loc[("F2", "AAA"), "Shares"]), 50.0)

    def test_formatted_values_are_parsed_into_numbers(self):
        panel = _panel()
        row = panel[(panel["Quarter"] == "2024Q2") & (panel["Fund"] == "F1")].iloc[0]
        self.assertAlmostEqual(row["Value"], 600_000.0)
        self.assertAlmostEqual(row["Portfolio_Pct"], 1.0)


class TestUniverse(unittest.TestCase):
    """
    The download universe is capped by ownership breadth: a name one fund owns
    cannot support a consensus-accumulation signal and only costs a fetch.
    """

    def test_only_tickers_reaching_the_holder_floor_in_some_quarter_qualify(self):
        panel = pd.DataFrame(
            [
                {"Quarter": "2024Q1", "Fund": f, "Ticker": "WIDE", "Shares": 1.0}
                for f in ["F1", "F2", "F3"]
            ]
            + [{"Quarter": "2024Q1", "Fund": "F1", "Ticker": "THIN", "Shares": 1.0}]
        )
        self.assertEqual(universe_tickers(panel, min_holders=3), ["WIDE"])

    def test_the_floor_may_be_met_in_any_single_quarter(self):
        panel = pd.DataFrame(
            [{"Quarter": "2024Q1", "Fund": "F1", "Ticker": "T", "Shares": 1.0}]
            + [
                {"Quarter": "2024Q2", "Fund": f, "Ticker": "T", "Shares": 1.0}
                for f in ["F1", "F2", "F3"]
            ]
        )
        self.assertEqual(universe_tickers(panel, min_holders=3), ["T"])


class TestSplitAdjustment(unittest.TestCase):
    """
    Share counts are restated post-split so a 10-for-1 does not masquerade as a
    900% accumulation in the quarter it lands.
    """

    def setUp(self):
        self.panel = pd.DataFrame(
            [
                {"Quarter": "2024Q1", "Fund": "F1", "Ticker": "SPLIT", "Shares": 100.0},
                {"Quarter": "2024Q3", "Fund": "F1", "Ticker": "SPLIT", "Shares": 1000.0},
                {"Quarter": "2024Q1", "Fund": "F1", "Ticker": "PLAIN", "Shares": 100.0},
            ]
        )
        self.splits = pd.DataFrame([{"ticker": "SPLIT", "date": "2024-06-10", "ratio": 10.0}])

    def test_pre_split_quarters_are_scaled_up_to_post_split_units(self):
        adjusted = apply_split_adjustment(self.panel, self.splits).set_index(["Ticker", "Quarter"])
        self.assertEqual(cast(float, adjusted.loc[("SPLIT", "2024Q1"), "Adj_Shares"]), 1000.0)
        self.assertEqual(cast(float, adjusted.loc[("SPLIT", "2024Q3"), "Adj_Shares"]), 1000.0)

    def test_tickers_without_splits_are_left_alone(self):
        adjusted = apply_split_adjustment(self.panel, self.splits).set_index(["Ticker", "Quarter"])
        self.assertEqual(cast(float, adjusted.loc[("PLAIN", "2024Q1"), "Adj_Shares"]), 100.0)

    def test_an_empty_split_history_is_a_no_op(self):
        adjusted = apply_split_adjustment(
            self.panel, pd.DataFrame(columns=["ticker", "date", "ratio"])
        )
        self.assertTrue((adjusted["Adj_Shares"] == adjusted["Shares"]).all())

    def test_a_split_after_every_quarter_scales_them_all_equally(self):
        splits = pd.DataFrame([{"ticker": "SPLIT", "date": "2030-01-01", "ratio": 2.0}])
        adjusted = apply_split_adjustment(self.panel, splits).set_index(["Ticker", "Quarter"])
        self.assertEqual(cast(float, adjusted.loc[("SPLIT", "2024Q1"), "Adj_Shares"]), 200.0)
        self.assertEqual(cast(float, adjusted.loc[("SPLIT", "2024Q3"), "Adj_Shares"]), 2000.0)


class TestSplitAdjustmentUsesQuarterEnd(unittest.TestCase):
    """
    A split lands relative to the quarter-end reporting date, not the filing date.
    """

    def test_a_split_inside_the_reported_quarter_is_already_in_the_filing(self):
        panel = pd.DataFrame([{"Quarter": "2024Q2", "Fund": "F1", "Ticker": "S", "Shares": 100.0}])
        splits = pd.DataFrame([{"ticker": "S", "date": date(2024, 5, 1).isoformat(), "ratio": 4.0}])
        adjusted = apply_split_adjustment(panel, splits)
        self.assertEqual(cast(float, adjusted.iloc[0]["Adj_Shares"]), 100.0)


if __name__ == "__main__":
    unittest.main()
