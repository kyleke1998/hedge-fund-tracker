import unittest

import pandas as pd

from app.analysis.regime import (
    build_regime_rows,
    detect_splits,
    drop_anomalous_funds,
    harmonise_basis,
    quarter_levels,
    transition_metrics,
)


def holding(fund: str, cusip: str, shares: float, price: float, ticker: str = "") -> dict:
    """
    One position row in the shape the regime loaders produce.
    """
    return {
        "Fund": fund,
        "CUSIP": cusip,
        "Ticker": ticker or cusip,
        "SharesN": float(shares),
        "ValueN": float(shares) * float(price),
        "Px": float(price),
        "Sector": "Technology",
    }


def frame(rows: list[dict]) -> pd.DataFrame:
    """
    Build a positions frame from holding() rows.
    """
    return pd.DataFrame(rows)


class TestDetectSplits(unittest.TestCase):
    def test_detects_share_multiplication_matched_by_a_price_fall(self):
        """
        A 20:1 split multiplies every holder's shares and divides the price.
        """
        prev = frame([holding(f"F{i}", "AMZN0", 1_000, 2000.0) for i in range(5)])
        curr = frame([holding(f"F{i}", "AMZN0", 20_000, 100.0) for i in range(5)])
        self.assertEqual(detect_splits(prev, curr), {"AMZN0": 20.0})

    def test_tolerates_holders_trading_around_the_split(self):
        """
        Funds trade through a split, so only a cluster - not every holder - lands
        exactly on the factor. AMZN's real Q2'22 ratios ranged from 0.96 to 39.
        """
        prev = frame([holding(f"F{i}", "AMZN0", 1_000, 2000.0) for i in range(6)])
        traded = [16_000, 20_000, 20_000, 33_000, 39_000, 1_000]
        curr = frame([holding(f"F{i}", "AMZN0", s, 100.0) for i, s in enumerate(traded)])
        self.assertEqual(detect_splits(prev, curr), {"AMZN0": 20.0})

    def test_ignores_accumulation_at_a_flat_price(self):
        """
        Doubling a share count while the price barely moves is buying, not a split.
        This is the BIDU Q2'23 case that a share-ratio-only rule misreads.
        """
        prev = frame([holding(f"F{i}", "BIDU0", 1_000, 150.0) for i in range(5)])
        curr = frame([holding(f"F{i}", "BIDU0", 2_000, 137.0) for i in range(5)])
        self.assertEqual(detect_splits(prev, curr), {})

    def test_ignores_names_with_too_few_paired_holders(self):
        """
        Two holders cannot establish a consensus price basis.
        """
        prev = frame([holding(f"F{i}", "THIN0", 1_000, 200.0) for i in range(2)])
        curr = frame([holding(f"F{i}", "THIN0", 2_000, 100.0) for i in range(2)])
        self.assertEqual(detect_splits(prev, curr), {})

    def test_detects_a_reverse_split(self):
        """
        A 1:2 reverse split halves shares and doubles the price.
        """
        prev = frame([holding(f"F{i}", "RVRS0", 2_000, 5.0) for i in range(5)])
        curr = frame([holding(f"F{i}", "RVRS0", 1_000, 10.0) for i in range(5)])
        self.assertEqual(detect_splits(prev, curr), {"RVRS0": 0.5})


class TestHarmoniseBasis(unittest.TestCase):
    def test_rescales_a_fund_filing_on_the_pre_split_basis(self):
        """
        SHOP split one day before 2022 quarter-end, so some funds filed pre-split
        and some post-split in the SAME quarter. The outlier must be rescaled onto
        the consensus basis or its delta is a phantom multi-billion-dollar sale.
        """
        rows = [holding(f"F{i}", "SHOP0", 10_000, 31.25) for i in range(4)]
        rows.append(holding("Laggard", "SHOP0", 1_000, 312.5))
        out = harmonise_basis(frame(rows))
        laggard = out[out["Fund"] == "Laggard"].iloc[0]
        self.assertAlmostEqual(laggard["SharesN"], 10_000.0, places=6)
        self.assertAlmostEqual(laggard["Px"], 31.25, places=6)

    def test_preserves_position_value_when_rescaling(self):
        """
        Rescaling changes the basis, never the dollars.
        """
        rows = [holding(f"F{i}", "SHOP0", 10_000, 31.25) for i in range(4)]
        rows.append(holding("Laggard", "SHOP0", 1_000, 312.5))
        out = harmonise_basis(frame(rows))
        laggard = out[out["Fund"] == "Laggard"].iloc[0]
        self.assertAlmostEqual(laggard["SharesN"] * laggard["Px"], 312_500.0, places=4)

    def test_leaves_a_consistent_quarter_untouched(self):
        """
        Ordinary price dispersion must not trigger a rescale.
        """
        rows = [holding(f"F{i}", "AAPL0", 1_000, 100.0 + i) for i in range(5)]
        out = harmonise_basis(frame(rows))
        self.assertEqual(list(out["SharesN"]), [1_000.0] * 5)


class TestAnomalyScreen(unittest.TestCase):
    def test_drops_a_book_priced_orders_of_magnitude_from_consensus(self):
        """
        Arena filed $183B against a $0.66B book - a denomination error. The tell
        is its pricing of securities other funds hold at a normal price.
        """
        rows = [holding(f"Good{i}", cusip, 100, 10.0) for i in range(4) for cusip in ("A0", "B0")]
        rows += [holding("Arena", cusip, 100, 10_000.0) for cusip in ("A0", "B0", "C0")]
        rows += [holding(f"Good{i}", "C0", 100, 10.0) for i in range(4)]
        out = drop_anomalous_funds({"2022Q4": frame(rows)})
        self.assertNotIn("Arena", set(out["2022Q4"]["Fund"]))
        self.assertIn("Good0", set(out["2022Q4"]["Fund"]))

    def test_keeps_a_fund_that_merely_grows_over_years(self):
        """
        Four years of ordinary growth must not read as an anomaly - the reason
        this screen tests pricing rather than a fund's own AUM history.
        """
        panels = {
            "2022Q1": frame([holding(f"F{i}", "A0", 100, 10.0) for i in range(4)]),
            "2026Q2": frame(
                [holding("F0", "A0", 5_000, 10.0)]
                + [holding(f"F{i}", "A0", 100, 10.0) for i in range(1, 4)]
            ),
        }
        out = drop_anomalous_funds(panels)
        self.assertIn("F0", set(out["2026Q2"]["Fund"]))

    def test_does_not_judge_a_fund_with_too_few_comparable_positions(self):
        """
        One overlapping holding is not enough evidence to discard a whole book.
        """
        rows = [holding(f"Good{i}", "A0", 100, 10.0) for i in range(4)]
        rows.append(holding("Lonely", "A0", 100, 10_000.0))
        out = drop_anomalous_funds({"2022Q4": frame(rows)})
        self.assertIn("Lonely", set(out["2022Q4"]["Fund"]))


class TestTransitionMetrics(unittest.TestCase):
    def test_counts_new_closed_and_untouched_on_the_common_fund_panel(self):
        """
        A fund present in only one of the two quarters must not register as
        wholesale opening or closing - that would measure roster growth.
        """
        prev = frame(
            [
                holding("A", "KEEP", 100, 10.0),
                holding("A", "GONE", 100, 10.0),
                holding("B", "KEEP", 100, 10.0),
            ]
        )
        curr = frame(
            [
                holding("A", "KEEP", 100, 10.0),
                holding("A", "FRESH", 50, 10.0),
                holding("B", "KEEP", 100, 10.0),
                holding("NewFund", "KEEP", 999, 10.0),
            ]
        )
        m = transition_metrics(prev, curr, {})
        self.assertEqual(m["new_positions"], 1)
        self.assertEqual(m["closed_positions"], 1)
        self.assertEqual(m["new_close"], 1.0)
        # Union over the two common funds is KEEP/GONE/FRESH for A plus KEEP for
        # B; the two KEEPs are unchanged.
        self.assertAlmostEqual(m["untouched_pct"], 50.0)

    def test_untouched_share_counts_unchanged_positions(self):
        """
        Two of four surviving positions unchanged -> 50%.
        """
        prev = frame([holding("A", f"S{i}", 100, 10.0) for i in range(4)])
        curr = frame(
            [
                holding("A", "S0", 100, 10.0),
                holding("A", "S1", 100, 10.0),
                holding("A", "S2", 150, 10.0),
                holding("A", "S3", 50, 10.0),
            ]
        )
        m = transition_metrics(prev, curr, {})
        self.assertAlmostEqual(m["untouched_pct"], 50.0)

    def test_recycling_is_new_capital_over_released_capital(self):
        """
        Dollars into newly opened names divided by dollars released by exits.
        """
        prev = frame([holding("A", "OUT", 100, 10.0)])
        curr = frame([holding("A", "IN", 200, 10.0)])
        m = transition_metrics(prev, curr, {})
        self.assertAlmostEqual(m["recycle"], 2.0)

    def test_split_adjustment_removes_a_phantom_purchase(self):
        """
        Without the split factor a 20:1 split reads as a 20x buy.
        """
        prev = frame([holding("A", "SPLIT", 1_000, 2000.0)])
        curr = frame([holding("A", "SPLIT", 20_000, 100.0)])
        m = transition_metrics(prev, curr, {"SPLIT": 20.0})
        self.assertAlmostEqual(m["untouched_pct"], 100.0)
        self.assertAlmostEqual(m["gross_buy"], 0.0)


class TestQuarterLevels(unittest.TestCase):
    def test_sector_weights_sum_to_one_hundred(self):
        """
        Weights are percentages of classified value.
        """
        rows = [
            {**holding("A", "T0", 100, 10.0), "Sector": "Technology"},
            {**holding("A", "H0", 100, 10.0), "Sector": "Healthcare"},
        ]
        levels = quarter_levels(frame(rows))
        self.assertAlmostEqual(sum(levels["sectors"].values()), 100.0)
        self.assertAlmostEqual(levels["sectors"]["Technology"], 50.0)

    def test_unclassified_value_is_excluded_from_sector_weights(self):
        """
        Unclassified value is dropped, never bucketed into a residual sector.
        """
        rows = [
            {**holding("A", "T0", 100, 10.0), "Sector": "Technology"},
            {**holding("A", "U0", 100, 10.0), "Sector": "Unclassified"},
        ]
        levels = quarter_levels(frame(rows))
        self.assertEqual(set(levels["sectors"]), {"Technology"})
        self.assertAlmostEqual(levels["sectors"]["Technology"], 100.0)

    def test_mega_cap_weight_uses_the_basket(self):
        """
        Mega-cap weight is the basket's share of total reported value.
        """
        rows = [
            {**holding("A", "C1", 100, 10.0, ticker="NVDA")},
            {**holding("A", "C2", 300, 10.0, ticker="OTHER")},
        ]
        levels = quarter_levels(frame(rows))
        self.assertAlmostEqual(levels["mega_weight_pct"], 25.0)


class TestBuildRegimeRows(unittest.TestCase):
    def test_emits_long_format_rows_for_every_quarter(self):
        """
        Output is long format so one CSV carries metrics, mega-cap and sectors.
        """
        panels = {
            "2022Q1": frame(
                [holding("A", "C1", 100, 10.0, ticker="NVDA"), holding("A", "C2", 100, 10.0)]
            ),
            "2022Q2": frame(
                [holding("A", "C1", 120, 10.0, ticker="NVDA"), holding("A", "C2", 100, 10.0)]
            ),
        }
        rows = build_regime_rows(panels)
        kinds = {r["kind"] for r in rows}
        self.assertEqual(kinds, {"metric", "mega", "sector"})
        quarters = {r["quarter"] for r in rows}
        self.assertEqual(quarters, {"2022Q1", "2022Q2"})

    def test_mega_share_index_is_chain_linked_through_a_roster_change(self):
        """
        A fund joining must not move the share index. The index chains
        quarter-over-quarter growth measured on funds common to both quarters.
        """
        panels = {
            "2022Q1": frame([holding("A", "C1", 100, 10.0, ticker="NVDA")]),
            "2022Q2": frame(
                [
                    holding("A", "C1", 200, 10.0, ticker="NVDA"),
                    holding("Joiner", "C1", 5_000, 10.0, ticker="NVDA"),
                ]
            ),
        }
        rows = build_regime_rows(panels)
        idx = {
            r["quarter"]: r["value"]
            for r in rows
            if r["kind"] == "mega" and r["key"] == "share_index"
        }
        self.assertAlmostEqual(idx["2022Q1"], 100.0)
        self.assertAlmostEqual(idx["2022Q2"], 200.0)


if __name__ == "__main__":
    unittest.main()
