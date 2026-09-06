import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from app.backtest.price_cache import PriceCache


class TestPriceCache(unittest.TestCase):
    """
    Tests for the persistent (ticker, date) -> price cache.
    """

    def setUp(self):
        """
        Create a temporary cache file path and a counting fake fetcher.
        """
        self.tmp = tempfile.mkdtemp(prefix="hft_price_cache_")
        self.path = Path(self.tmp) / "prices.csv"
        self.calls: list[tuple[str, date]] = []

    def tearDown(self):
        """
        Remove the temporary directory.
        """
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fetch(self, ticker: str, day: date):
        """
        Record the call and return a deterministic price (or None for MISS).
        """
        self.calls.append((ticker, day))
        if ticker == "MISS":
            return None
        return 100.0

    def test_miss_fetches_and_persists(self):
        """
        A cache miss calls the fetcher, returns the value, and writes it to disk.
        """
        cache = PriceCache(path=self.path, fetch_fn=self._fetch)
        price = cache.get("AAA", date(2025, 5, 15))
        self.assertEqual(price, 100.0)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.path.exists())

    def test_hit_does_not_refetch(self):
        """
        A second lookup of the same (ticker, date) is served from memory.
        """
        cache = PriceCache(path=self.path, fetch_fn=self._fetch)
        cache.get("AAA", date(2025, 5, 15))
        cache.get("AAA", date(2025, 5, 15))
        self.assertEqual(len(self.calls), 1)

    def test_persisted_cache_reloads_without_fetch(self):
        """
        A fresh instance over the same file returns cached prices with no fetch.
        """
        PriceCache(path=self.path, fetch_fn=self._fetch).get("AAA", date(2025, 5, 15))
        self.calls.clear()
        reloaded = PriceCache(path=self.path, fetch_fn=self._fetch)
        price = reloaded.get("AAA", date(2025, 5, 15))
        self.assertEqual(price, 100.0)
        self.assertEqual(len(self.calls), 0)

    def test_none_result_is_not_persisted(self):
        """
        A failed lookup (None) is returned but not cached, so it is retried.
        """
        cache = PriceCache(path=self.path, fetch_fn=self._fetch)
        self.assertIsNone(cache.get("MISS", date(2025, 5, 15)))
        self.assertIsNone(cache.get("MISS", date(2025, 5, 15)))
        self.assertEqual(len(self.calls), 2)


class TestPriceCacheSplitInvalidation(unittest.TestCase):
    """
    Yahoo's OHLC series is split-adjusted at source, so a split retroactively
    rewrites every earlier price. A cache that assumes prices are immutable
    serves pre-split values next to freshly fetched post-split ones and
    fabricates a return of the split ratio in whichever window straddles it.
    """

    def setUp(self):
        """
        Create a temporary cache path and a counting fake fetcher.
        """
        self.tmp = tempfile.mkdtemp(prefix="hft_price_split_")
        self.path = Path(self.tmp) / "prices.csv"
        self.calls: list[tuple[str, date]] = []
        self.splits: dict[str, list[date]] = {}

    def tearDown(self):
        """
        Remove the temporary directory.
        """
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fetch(self, ticker: str, day: date):
        """
        Record the call and return a deterministic price.
        """
        self.calls.append((ticker, day))
        return 25.0

    def _splits(self, ticker: str) -> list[date]:
        """
        Return the configured split ex-dates for a ticker.
        """
        return self.splits.get(ticker, [])

    def _seed(self, ticker: str, day: date, price: float, fetched_at: date) -> None:
        """
        Write one cache row with an explicit fetch date.
        """
        cache = PriceCache(
            path=self.path, fetch_fn=lambda _t, _d: price, splits_fn=self._splits, today=fetched_at
        )
        cache.get(ticker, day)

    def test_row_fetched_before_a_later_split_is_refetched(self):
        """
        The price was cached before the split, so the stored value is on the
        old (pre-split) scale and must be discarded.
        """
        self._seed("AAA", date(2025, 5, 16), 100.0, fetched_at=date(2025, 6, 1))
        self.splits["AAA"] = [date(2025, 7, 1)]

        cache = PriceCache(
            path=self.path,
            fetch_fn=self._fetch,
            splits_fn=self._splits,
            today=date(2025, 8, 15),
        )
        price = cache.get("AAA", date(2025, 5, 16))

        self.assertEqual(price, 25.0)
        self.assertEqual(self.calls, [("AAA", date(2025, 5, 16))])

    def test_row_fetched_after_the_split_is_kept(self):
        """
        A price cached after the ex-date was already adjusted at source, so it
        is still valid and must not trigger a refetch.
        """
        self._seed("AAA", date(2025, 5, 16), 25.0, fetched_at=date(2025, 8, 1))
        self.splits["AAA"] = [date(2025, 7, 1)]

        cache = PriceCache(
            path=self.path,
            fetch_fn=self._fetch,
            splits_fn=self._splits,
            today=date(2025, 8, 15),
        )
        price = cache.get("AAA", date(2025, 5, 16))

        self.assertEqual(price, 25.0)
        self.assertEqual(self.calls, [])

    def test_split_before_the_priced_date_does_not_invalidate(self):
        """
        A split with an ex-date at or before the priced day never changes that
        day's quote, whenever it was fetched.
        """
        self._seed("AAA", date(2025, 5, 16), 100.0, fetched_at=date(2025, 6, 1))
        self.splits["AAA"] = [date(2025, 4, 1)]

        cache = PriceCache(
            path=self.path,
            fetch_fn=self._fetch,
            splits_fn=self._splits,
            today=date(2025, 8, 15),
        )

        self.assertEqual(cache.get("AAA", date(2025, 5, 16)), 100.0)
        self.assertEqual(self.calls, [])

    def test_splits_are_not_looked_up_for_uncached_tickers(self):
        """
        A cold run has nothing to invalidate, so it must not pay a splits
        lookup per ticker.
        """
        looked_up: list[str] = []

        def splits_fn(ticker: str) -> list[date]:
            """
            Record the lookup and report no splits.
            """
            looked_up.append(ticker)
            return []

        cache = PriceCache(path=self.path, fetch_fn=self._fetch, splits_fn=splits_fn)
        cache.get("FRESH", date(2025, 5, 16))

        self.assertEqual(looked_up, [])

    def test_splits_are_looked_up_once_per_ticker(self):
        """
        The check runs once per ticker per process, not once per lookup.
        """
        self._seed("AAA", date(2025, 5, 16), 100.0, fetched_at=date(2025, 6, 1))
        looked_up: list[str] = []

        def splits_fn(ticker: str) -> list[date]:
            """
            Record the lookup and report no splits.
            """
            looked_up.append(ticker)
            return []

        cache = PriceCache(path=self.path, fetch_fn=self._fetch, splits_fn=splits_fn)
        cache.get("AAA", date(2025, 5, 16))
        cache.get("AAA", date(2025, 8, 14))

        self.assertEqual(looked_up, ["AAA"])

    def test_legacy_file_is_upgraded_to_the_current_header(self):
        """
        Appending to a cache written before ``fetched_at`` existed must not leave
        a ragged file whose rows carry more fields than its header.
        """
        self.path.write_text(
            '"ticker","date","price"\n"AAA","2025-05-16","100.0"\n', encoding="utf-8"
        )

        cache = PriceCache(
            path=self.path,
            fetch_fn=self._fetch,
            splits_fn=self._splits,
            today=date(2025, 8, 15),
        )
        cache.get("BBB", date(2025, 5, 16))

        rows = list(csv.reader(self.path.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(rows[0], ["ticker", "date", "price", "fetched_at"])
        self.assertTrue(all(len(row) == 4 for row in rows[1:]), rows)
        self.assertEqual(len(rows), 3)

    def test_legacy_row_without_fetch_date_is_invalidated_by_any_later_split(self):
        """
        A cache written before the fetch date was tracked cannot be proven
        current, so a split after the priced day forces a refetch.
        """
        self.path.write_text(
            '"ticker","date","price"\n"AAA","2025-05-16","100.0"\n', encoding="utf-8"
        )
        self.splits["AAA"] = [date(2025, 7, 1)]

        cache = PriceCache(
            path=self.path,
            fetch_fn=self._fetch,
            splits_fn=self._splits,
            today=date(2025, 8, 15),
        )
        price = cache.get("AAA", date(2025, 5, 16))

        self.assertEqual(price, 25.0)
        self.assertEqual(self.calls, [("AAA", date(2025, 5, 16))])


if __name__ == "__main__":
    unittest.main()
