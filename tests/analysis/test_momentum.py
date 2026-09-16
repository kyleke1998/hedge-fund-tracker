import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.analysis.momentum import build_momentum_rows, rate_of_change
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


class TestRateOfChange(unittest.TestCase):
    def test_percent_change_over_the_window(self):
        """
        The reading is (price_t / price_{t-window} - 1) x 100.
        """
        prices = pd.Series([100.0, 110.0, 120.0, 132.0], index=pd.Index(sessions(4)))
        roc = rate_of_change(prices, window=2)
        self.assertAlmostEqual(roc.iloc[2], 20.0, places=9)
        self.assertAlmostEqual(roc.iloc[3], 20.0, places=9)

    def test_the_first_window_readings_have_no_lookback(self):
        """
        Nothing is published until a full window sits behind the session.
        """
        prices = pd.Series([100.0, 101.0, 102.0], index=pd.Index(sessions(3)))
        roc = rate_of_change(prices, window=2)
        self.assertTrue(np.isnan(roc.iloc[0]))
        self.assertTrue(np.isnan(roc.iloc[1]))

    def test_a_fall_reads_negative(self):
        """
        A lower price than a quarter ago is a negative rate of change.
        """
        prices = pd.Series([100.0, 90.0], index=pd.Index(sessions(2)))
        self.assertAlmostEqual(rate_of_change(prices, window=1).iloc[1], -10.0, places=9)


class TestBuildMomentumRows(unittest.TestCase):
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

    def test_builds_iso_dated_rows_from_the_loaded_sessions(self):
        """
        The rows are what the CSV writer persists: ISO date and the reading.
        """
        days = sessions(5)
        rows = build_momentum_rows(
            start=days[0],
            ticker="SPY",
            loader=self.loader({"SPY": [100.0, 110.0, 120.0, 90.0, 132.0]}, days),
            window=2,
            warmup=timedelta(days=0),
        )
        self.assertEqual([row["date"] for row in rows], [d.isoformat() for d in days[2:]])
        self.assertAlmostEqual(rows[0]["roc"], 20.0, places=6)
        self.assertAlmostEqual(rows[1]["roc"], -18.181818, places=4)

    def test_sessions_before_the_start_are_lookback_only(self):
        """
        Warm-up sessions feed the first window but are never emitted.
        """
        days = sessions(6)
        rows = build_momentum_rows(
            start=days[4],
            ticker="SPY",
            loader=self.loader({"SPY": [100.0, 101.0, 102.0, 103.0, 110.0, 120.0]}, days),
            window=2,
            warmup=timedelta(days=30),
        )
        self.assertEqual([row["date"] for row in rows], [d.isoformat() for d in days[4:]])
        self.assertAlmostEqual(rows[0]["roc"], 100.0 * (110.0 / 102.0 - 1.0), places=6)

    def test_no_bars_means_no_rows(self):
        """
        A ticker the source cannot serve yields an empty series, not a crash.
        """
        days = sessions(4)
        rows = build_momentum_rows(
            start=days[0],
            ticker="SPY",
            loader=self.loader({}, days),
            window=2,
            warmup=timedelta(days=0),
        )
        self.assertEqual(rows, [])

    def test_unordered_and_duplicated_bars_are_normalised(self):
        """
        The loader's order is not trusted: the series is sorted and de-duped.
        """
        days = sessions(3)
        scrambled = [
            Bar(day=days[2], price=120.0, volume=1.0),
            Bar(day=days[0], price=100.0, volume=1.0),
            Bar(day=days[0], price=999.0, volume=1.0),
            Bar(day=days[1], price=110.0, volume=1.0),
            Bar(day=days[0], price=100.0, volume=1.0),
        ]
        rows = build_momentum_rows(
            start=days[0],
            ticker="SPY",
            loader=lambda _ticker, _start: scrambled,
            window=2,
            warmup=timedelta(days=0),
        )
        self.assertEqual([row["date"] for row in rows], [days[2].isoformat()])
        self.assertAlmostEqual(rows[0]["roc"], 20.0, places=6)


if __name__ == "__main__":
    unittest.main()
