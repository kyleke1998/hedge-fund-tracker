import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from app.stocks.bar_history import Bar, load_daily_bars

ROWS = [
    ("AAA", "2024-01-02", 10.0, 12.0, 8.0, 11.0, 1000.0),
    ("AAA", "2024-01-03", 11.0, 13.0, 9.0, 12.0, 2000.0),
    ("BBB", "2024-01-02", 50.0, 50.0, 50.0, 50.0, 500.0),
]


def write_cache(folder: Path) -> Path:
    """
    A minimal bars.parquet in the layout of the external OHLCV cache.
    """
    frame = pd.DataFrame(ROWS, columns=["symbol", "date", "open", "high", "low", "close", "volume"])
    frame.to_parquet(folder / "bars.parquet")
    return folder


SPLIT_ROWS = [
    ("AAA", "2024-01-03", 12.0, 12.0, 12.0, 12.0, 100.0),
    ("AAA", "2024-01-04", 1.0, 1.0, 1.0, 1.0, 1000.0),
]


def write_splits(folder: Path, rows: list[tuple[str, str, float]]) -> None:
    """
    A splits.parquet alongside the bars, in the external cache's layout.
    """
    pd.DataFrame(rows, columns=["symbol", "date", "split_factor"]).to_parquet(
        folder / "splits.parquet"
    )


class TestSplitAdjustment(unittest.TestCase):
    def test_an_as_traded_series_is_restated_onto_the_post_split_basis(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pd.DataFrame(
                ROWS[:1] + SPLIT_ROWS,
                columns=["symbol", "date", "open", "high", "low", "close", "volume"],
            ).to_parquet(folder / "bars.parquet")
            write_splits(folder, [("AAA", "2024-01-04", 12.0)])
            bars = load_daily_bars("AAA", date(2024, 1, 1), cache_dir=folder)
        self.assertAlmostEqual(bars[1].price, 1.0)
        self.assertAlmostEqual(bars[1].volume, 1200.0)
        self.assertAlmostEqual(bars[2].price, 1.0)
        self.assertAlmostEqual(bars[2].volume, 1000.0)

    def test_an_already_adjusted_series_is_left_alone(self):
        with TemporaryDirectory() as tmp:
            folder = write_cache(Path(tmp))
            write_splits(folder, [("AAA", "2024-01-03", 7.0)])
            bars = load_daily_bars("AAA", date(2024, 1, 1), cache_dir=folder)
        self.assertAlmostEqual(bars[0].price, (12.0 + 8.0 + 11.0) / 3)
        self.assertEqual(bars[0].volume, 1000.0)

    def test_another_symbols_splits_are_ignored(self):
        with TemporaryDirectory() as tmp:
            folder = write_cache(Path(tmp))
            write_splits(folder, [("BBB", "2024-01-03", 7.0)])
            bars = load_daily_bars("AAA", date(2024, 1, 1), cache_dir=folder)
        self.assertAlmostEqual(bars[0].price, (12.0 + 8.0 + 11.0) / 3)


class TestLoadDailyBars(unittest.TestCase):
    def test_reads_the_cache_and_prices_each_bar_at_its_typical_price(self):
        with TemporaryDirectory() as tmp:
            bars = load_daily_bars("AAA", date(2024, 1, 1), cache_dir=write_cache(Path(tmp)))
        self.assertEqual([b.day for b in bars], [date(2024, 1, 2), date(2024, 1, 3)])
        self.assertAlmostEqual(bars[0].price, (12.0 + 8.0 + 11.0) / 3)
        self.assertEqual(bars[0].volume, 1000.0)

    def test_bars_before_the_start_date_are_dropped(self):
        with TemporaryDirectory() as tmp:
            bars = load_daily_bars("AAA", date(2024, 1, 3), cache_dir=write_cache(Path(tmp)))
        self.assertEqual([b.day for b in bars], [date(2024, 1, 3)])

    def test_a_ticker_missing_from_the_cache_falls_back_to_the_price_source(self):
        fallback = [Bar(day=date(2024, 2, 1), price=7.0, volume=0.0)]
        with (
            TemporaryDirectory() as tmp,
            patch("app.stocks.bar_history._fetch_bars", return_value=fallback) as fetch,
        ):
            bars = load_daily_bars("CCC", date(2024, 1, 1), cache_dir=write_cache(Path(tmp)))
        self.assertEqual(bars, fallback)
        fetch.assert_called_once()

    def test_an_absent_cache_falls_back_without_raising(self):
        with (
            TemporaryDirectory() as tmp,
            patch("app.stocks.bar_history._fetch_bars", return_value=[]) as fetch,
        ):
            self.assertEqual(load_daily_bars("AAA", date(2024, 1, 1), cache_dir=Path(tmp)), [])
        fetch.assert_called_once()

    def test_the_cache_directory_defaults_to_the_configured_one(self):
        with TemporaryDirectory() as tmp:
            write_cache(Path(tmp))
            with patch.dict("os.environ", {"BARS_CACHE_DIR": tmp}):
                bars = load_daily_bars("BBB", date(2024, 1, 1))
        self.assertEqual([b.price for b in bars], [50.0])


if __name__ == "__main__":
    unittest.main()
