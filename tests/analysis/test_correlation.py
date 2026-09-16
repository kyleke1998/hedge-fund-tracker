import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.analysis.correlation import (
    average_pairwise_correlation,
    build_correlation_rows,
    build_return_panel,
    rolling_average_correlation,
    usable_block,
)
from app.stocks.bar_history import Bar


def sessions(count: int, start: date = date(2024, 1, 1)) -> list[date]:
    """
    A run of consecutive calendar days standing in for trading sessions.
    """
    return [start + timedelta(days=i) for i in range(count)]


def bars(prices: list[float], days: list[date] | None = None) -> list[Bar]:
    """
    Bars for a price path, one per session.
    """
    days = days or sessions(len(prices))
    return [
        Bar(day=day, price=price, volume=1_000.0) for day, price in zip(days, prices, strict=True)
    ]


def walk(steps: list[float], base: float = 100.0) -> list[float]:
    """
    A price path whose log returns are exactly `steps`.
    """
    prices = [base]
    for step in steps:
        prices.append(prices[-1] * float(np.exp(step)))
    return prices


def correlation(block: np.ndarray) -> float:
    """
    The average pairwise correlation of a block, failing when it is undefined.
    """
    value = average_pairwise_correlation(block)
    if value is None:
        raise AssertionError("expected a correlation for this block")
    return value


def cell(panel: pd.DataFrame, day: date, ticker: str) -> float:
    """
    One return out of a panel.
    """
    return dict(zip(panel.index, panel[ticker].to_numpy(dtype=float), strict=True))[day]


class TestAveragePairwiseCorrelation(unittest.TestCase):
    def test_identical_movers_correlate_perfectly(self):
        """
        Columns that move together every session average to 1.0.
        """
        moves = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        block = np.column_stack([moves, moves, moves])
        self.assertAlmostEqual(correlation(block), 1.0, places=9)

    def test_opposite_movers_correlate_negatively(self):
        """
        A perfectly hedged pair averages to -1.0.
        """
        moves = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        block = np.column_stack([moves, -moves])
        self.assertAlmostEqual(correlation(block), -1.0, places=9)

    def test_scale_does_not_matter(self):
        """
        Correlation is scale-free: a high-beta name is still perfectly correlated.
        """
        moves = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        block = np.column_stack([moves, moves * 3.0])
        self.assertAlmostEqual(correlation(block), 1.0, places=9)

    def test_independent_movers_average_near_zero(self):
        """
        Unrelated names leave no common factor behind.
        """
        block = np.random.default_rng(11).normal(size=(250, 40))
        self.assertLess(abs(correlation(block)), 0.05)

    def test_matches_the_off_diagonal_mean_of_the_correlation_matrix(self):
        """
        The standardized-sum identity must equal the explicit pairwise mean.
        """
        block = np.random.default_rng(3).normal(size=(63, 12))
        matrix = np.corrcoef(block, rowvar=False)
        expected = (float(matrix.sum()) - 12) / (12 * 11)
        self.assertAlmostEqual(correlation(block), expected, places=12)

    def test_a_single_column_has_no_pair_to_measure(self):
        """
        One name is not a correlation.
        """
        self.assertIsNone(average_pairwise_correlation(np.array([[0.01], [0.02], [-0.03]])))


class TestUsableBlock(unittest.TestCase):
    def test_drops_a_column_with_a_gap_in_the_window(self):
        """
        A name that did not trade every session of the window cannot be paired
        with the ones that did without inventing returns for the missing days.
        """
        window = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.01, np.nan, -0.02]})
        self.assertEqual(usable_block(window).shape, (3, 1))

    def test_drops_a_column_that_never_moved(self):
        """
        A flat series has no variance, so its correlation is undefined.
        """
        window = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.0, 0.0]})
        self.assertEqual(usable_block(window).shape, (3, 1))

    def test_keeps_complete_moving_columns(self):
        """
        Everything else survives.
        """
        window = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, -0.01, 0.03]})
        self.assertEqual(usable_block(window).shape, (3, 2))


class TestBuildReturnPanel(unittest.TestCase):
    def test_returns_log_changes_on_the_session_calendar(self):
        """
        One row per session after the first, one column per ticker.
        """
        days = sessions(4)
        panel = build_return_panel({"AAA": bars([100.0, 110.0, 121.0, 121.0], days)}, days)
        self.assertEqual(list(panel.index), days[1:])
        self.assertAlmostEqual(cell(panel, days[1], "AAA"), float(np.log(1.1)), places=12)
        self.assertAlmostEqual(cell(panel, days[3], "AAA"), 0.0, places=12)

    def test_a_ticker_missing_a_session_leaves_a_gap_not_a_flat_day(self):
        """
        Reindexing onto the calendar must not carry a stale price forward: an
        untraded session would read as a 0% move and dampen the correlation.
        """
        days = sessions(4)
        partial = bars([100.0, 110.0, 121.0], [days[0], days[1], days[3]])
        panel = build_return_panel({"AAA": partial}, days, min_session_coverage=0.0)
        self.assertTrue(np.isnan(cell(panel, days[2], "AAA")))

    def test_drops_a_session_most_of_the_basket_did_not_trade(self):
        """
        A date only a handful of names have a bar for is a hole in the price
        source, not a market day. Keeping it would knock every absent name out
        of ~three months of windows; the return simply spans the gap instead.
        """
        days = sessions(4)
        full = {f"T{i}": bars(walk([0.01, -0.02, 0.03]), days) for i in range(9)}
        gapped = bars(walk([0.01, 0.03]), [days[0], days[1], days[3]])
        panel = build_return_panel({**full, "GAP": gapped}, days, min_session_coverage=0.95)
        self.assertEqual(list(panel.index), [days[1], days[3]])
        self.assertAlmostEqual(cell(panel, days[3], "GAP"), 0.03, places=12)

    def test_keeps_a_session_almost_the_whole_basket_traded(self):
        """
        One absent name is that name's gap, not the session's.
        """
        days = sessions(4)
        full = {f"T{i}": bars(walk([0.01, -0.02, 0.03]), days) for i in range(19)}
        gapped = bars(walk([0.01, 0.03]), [days[0], days[1], days[3]])
        panel = build_return_panel({**full, "GAP": gapped}, days, min_session_coverage=0.9)
        self.assertEqual(list(panel.index), days[1:])
        self.assertTrue(np.isnan(cell(panel, days[2], "GAP")))

    def test_a_name_that_had_not_listed_yet_does_not_condemn_a_session(self):
        """
        Coverage is judged against the names trading at the time; otherwise the
        earliest sessions would look empty simply because half the basket IPO'd
        later.
        """
        days = sessions(4)
        listed = {f"T{i}": bars(walk([0.01, -0.02, 0.03]), days) for i in range(3)}
        newcomer = bars(walk([0.02]), days[2:])
        panel = build_return_panel({**listed, "NEW": newcomer}, days, min_session_coverage=0.9)
        self.assertEqual(list(panel.index), days[1:])

    def test_ignores_sessions_outside_the_calendar(self):
        """
        The calendar ticker defines the sessions; a bar on any other day is noise.
        """
        days = sessions(3)
        extra = bars([100.0, 110.0, 121.0, 130.0], [*days, date(2024, 1, 4)])
        panel = build_return_panel({"AAA": extra}, days)
        self.assertEqual(list(panel.index), days[1:])


class TestRollingAverageCorrelation(unittest.TestCase):
    def test_emits_nothing_until_the_window_is_full(self):
        """
        A 3-month reading needs three months of sessions behind it.
        """
        days = sessions(6)
        moves = [0.01, -0.02, 0.03, -0.01, 0.02]
        panel = build_return_panel(
            {"AAA": bars(walk(moves), days), "BBB": bars(walk(moves), days)}, days
        )
        rows = rolling_average_correlation(panel, window=4, min_stocks=2)
        self.assertEqual([row["date"] for row in rows], days[4:])

    def test_reports_how_many_names_the_reading_came_from(self):
        """
        Coverage varies over time, so each reading carries its own name count.
        """
        days = sessions(5)
        moves = [0.01, -0.02, 0.03, -0.01]
        panel = build_return_panel(
            {
                "AAA": bars(walk(moves), days),
                "BBB": bars(walk([-m for m in moves]), days),
                "CCC": bars(walk(moves), days),
            },
            days,
        )
        rows = rolling_average_correlation(panel, window=4, min_stocks=2)
        self.assertEqual([row["stocks"] for row in rows], [3])
        self.assertAlmostEqual(rows[0]["correlation"], -1 / 3, places=9)

    def test_skips_a_window_with_too_few_usable_names(self):
        """
        Two names is a pair, not a market: a thin window is no reading at all.
        """
        days = sessions(5)
        moves = [0.01, -0.02, 0.03, -0.01]
        panel = build_return_panel(
            {"AAA": bars(walk(moves), days), "BBB": bars(walk(moves), days)}, days
        )
        self.assertEqual(rolling_average_correlation(panel, window=4, min_stocks=3), [])


class TestBuildCorrelationRows(unittest.TestCase):
    def loader(self, paths: dict[str, list[float]], days: list[date]):
        """
        A bar loader over canned price paths, ignoring tickers it does not know.
        """

        def load(ticker: str, start: date) -> list[Bar]:
            prices = paths.get(ticker)
            if prices is None:
                return []
            return [bar for bar in bars(prices, days) if bar.day >= start]

        return load

    def test_builds_iso_dated_rows_from_the_calendar_ticker_sessions(self):
        """
        The rows are what the CSV writer persists: ISO date, value, name count.
        """
        days = sessions(6)
        moves = [0.01, -0.02, 0.03, -0.01, 0.02]
        paths = {
            "QQQ": walk(moves),
            "AAA": walk(moves),
            "BBB": walk(moves),
            "CCC": walk([-m for m in moves]),
        }
        rows = build_correlation_rows(
            start=days[0],
            tickers=("AAA", "BBB", "CCC"),
            loader=self.loader(paths, days),
            window=4,
            min_stocks=3,
        )
        self.assertEqual([row["date"] for row in rows], [d.isoformat() for d in days[4:]])
        self.assertEqual({row["stocks"] for row in rows}, {3})
        self.assertAlmostEqual(rows[0]["correlation"], -1 / 3, places=6)

    def test_drops_a_ticker_with_no_history_anywhere(self):
        """
        A delisted or misspelled ticker returns no bars and must not sink the run.
        """
        days = sessions(6)
        moves = [0.01, -0.02, 0.03, -0.01, 0.02]
        paths = {"QQQ": walk(moves), "AAA": walk(moves), "BBB": walk(moves)}
        rows = build_correlation_rows(
            start=days[0],
            tickers=("AAA", "BBB", "GONE"),
            loader=self.loader(paths, days),
            window=4,
            min_stocks=2,
        )
        self.assertEqual({row["stocks"] for row in rows}, {2})

    def test_no_calendar_means_no_rows(self):
        """
        Without the index's own sessions there is nothing to align returns to.
        """
        days = sessions(6)
        rows = build_correlation_rows(
            start=days[0],
            tickers=("AAA",),
            loader=self.loader({"AAA": walk([0.01] * 5)}, days),
            window=4,
            min_stocks=2,
        )
        self.assertEqual(rows, [])

    def test_emits_only_readings_from_the_requested_start(self):
        """
        Sessions before `start` are warm-up for the first window, not output.
        """
        days = sessions(8)
        moves = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01]
        paths = {"QQQ": walk(moves), "AAA": walk(moves), "BBB": walk([-m for m in moves])}
        rows = build_correlation_rows(
            start=days[5],
            tickers=("AAA", "BBB"),
            loader=self.loader(paths, days),
            window=4,
            min_stocks=2,
            warmup=timedelta(days=30),
        )
        self.assertEqual([row["date"] for row in rows], [d.isoformat() for d in days[5:]])


if __name__ == "__main__":
    unittest.main()
