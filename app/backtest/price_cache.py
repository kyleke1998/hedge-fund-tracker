import csv
import math
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import TypeIs

from app.utils.logger import get_logger, log_safe

logger = get_logger(__name__)

CACHE_DIR = "__pricecache__"
CACHE_FILE = "prices.csv"
_FIELDNAMES = ["ticker", "date", "price", "fetched_at"]
# A lookup still empty this long after its date is treated as permanent (a
# delisted or unknown symbol), recorded as a blank-price row and never retried.
MISS_SETTLE_DAYS = 7


def _is_valid_price(price: float | None) -> TypeIs[float]:
    """
    True for a finite, strictly positive quote; anything else is a failed lookup.
    """
    return price is not None and math.isfinite(price) and price > 0


class PriceCache:
    """
    Persistent (ticker, date) -> price cache backing the backtest price lookups.

    Caching makes a full regeneration near-instant: changing the tracked-fund
    list reshuffles screen membership but reuses every cached price, fetching
    only genuinely new (ticker, date) pairs. Misses fall through to the injected
    fetcher. A failed lookup for a date older than MISS_SETTLE_DAYS is recorded
    too (blank price), because a historical price that no provider has will not
    appear later; a recent miss is retried on the next run.

    Historical prices are *not* immutable, which is the subtlety this class has
    to handle. Yahoo's OHLC series is split-adjusted at source, so a split
    retroactively rescales every earlier bar. A row cached before the ex-date
    therefore holds a pre-split price; served next to a freshly fetched
    post-split one, it fabricates a return of the split ratio in whichever
    window straddles the split. Each row records when it was fetched, and the
    first lookup of a ticker that has cached rows drops the ones a later split
    has since invalidated. Tickers with nothing cached cost no splits lookup, so
    a cold run is unaffected.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        fetch_fn: Callable[[str, date], float | None] | None = None,
        range_fetch_fn: Callable[[str, date, date], float | None] | None = None,
        splits_fn: Callable[[str], list[date]] | None = None,
        today: date | None = None,
        today_fn: Callable[[], date] | None = None,
    ) -> None:
        """
        Load any existing cache file and store the fallback price fetchers.

        ``today`` pins the date stamped into ``fetched_at``; ``today_fn`` drives the
        miss-settling window and defaults to the pinned ``today``.
        """
        self._path = Path(path) if path is not None else Path(CACHE_DIR) / CACHE_FILE
        self._fetch_fn = fetch_fn or self._default_fetch_fn
        self._range_fetch_fn = range_fetch_fn or self._default_range_fetch_fn
        self._splits_fn = splits_fn or self._default_splits_fn
        self._today = today or date.today()
        self._today_fn = today_fn or (lambda: self._today)
        self._misses: set[tuple[str, str]] = set()
        self._cache: dict[tuple[str, str], tuple[float, str]] = self._load()
        self._splits_checked: set[str] = set()
        if self._legacy_header:
            # Upgrade in place, or appended rows would carry more fields than the header.
            self._rewrite()

    @staticmethod
    def _default_fetch_fn(ticker: str, day: date) -> float | None:
        """
        Default single-day lookup via the project's free price-fetch chain.
        """
        from app.stocks.price_fetcher import PriceFetcher

        return PriceFetcher.get_avg_price(ticker, day)

    @staticmethod
    def _default_range_fetch_fn(ticker: str, start: date, end: date) -> float | None:
        """
        Default last-bar-in-range lookup via the project's free price-fetch chain.
        """
        from app.stocks.price_fetcher import PriceFetcher

        return PriceFetcher.get_last_price_in_range(ticker, start, end)

    @staticmethod
    def _default_splits_fn(ticker: str) -> list[date]:
        """
        Default split-history lookup via yfinance (ex-dates only).

        A failed lookup yields no dates: a failed check must not abort the caller.
        """
        from app.stocks.libraries import YFinance

        return [ex_date for ex_date, _factor in YFinance.get_splits(ticker) or []]

    def _load(self) -> dict[tuple[str, str], tuple[float, str]]:
        """
        Read the persisted cache file into memory, skipping malformed rows.

        A row written before ``fetched_at`` existed loads with an empty fetch
        date, which the split check treats as "cannot be proven current".
        """
        cache: dict[tuple[str, str], tuple[float, str]] = {}
        self._legacy_header = False
        if not self._path.exists():
            return cache
        with self._path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self._legacy_header = bool(reader.fieldnames) and list(reader.fieldnames) != _FIELDNAMES
            for row in reader:
                value = row.get("price")
                if value == "" and row.get("ticker") and row.get("date"):
                    self._misses.add((row["ticker"], row["date"]))
                    continue
                if not value:
                    continue
                try:
                    price = float(value)
                    key = (row["ticker"], row["date"])
                except ValueError, KeyError:
                    continue
                if _is_valid_price(price):
                    cache[key] = (price, row.get("fetched_at") or "")
        return cache

    def _rewrite(self) -> None:
        """
        Rewrite the whole cache file from memory (used after an invalidation).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for (ticker, day_iso), (price, fetched_at) in self._cache.items():
                writer.writerow(
                    {
                        "ticker": ticker,
                        "date": day_iso,
                        "price": price,
                        "fetched_at": fetched_at,
                    }
                )
            for ticker, day_iso in sorted(self._misses):
                writer.writerow({"ticker": ticker, "date": day_iso, "price": "", "fetched_at": ""})

    def _append(self, ticker: str, day_iso: str, price: float | None) -> None:
        """
        Append one resolved price, or a blank settled miss, to the on-disk cache.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._path.exists()
        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES, quoting=csv.QUOTE_ALL)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "ticker": ticker,
                    "date": day_iso,
                    "price": "" if price is None else price,
                    "fetched_at": "" if price is None else self._today.isoformat(),
                }
            )

    def _record_miss(self, ticker: str, key_day: str, last_day: date) -> None:
        """
        Remember a failed lookup once its date is old enough to be final.
        """
        if (self._today_fn() - last_day) <= timedelta(days=MISS_SETTLE_DAYS):
            return
        self._misses.add((ticker, key_day))
        self._append(ticker, key_day, None)

    def _drop_split_stale(self, ticker: str) -> None:
        """
        Discard this ticker's rows that a split has since rescaled.

        A row for day D is invalid when some split ex-date S satisfies
        ``D < S`` (the split rewrites that day's quote) and the row was fetched
        before S (so it never saw the adjustment). Runs once per ticker per
        process, and only for tickers that actually have cached rows.
        """
        if ticker in self._splits_checked:
            return
        self._splits_checked.add(ticker)

        rows = [key for key in self._cache if key[0] == ticker]
        if not rows:
            return

        try:
            splits = self._splits_fn(ticker)
        except Exception:
            logger.error("Split lookup failed for %s", log_safe(ticker), exc_info=True)
            return
        if not splits:
            return

        stale = [
            key
            for key in rows
            if any(
                key[1] < split.isoformat()
                and (not self._cache[key][1] or self._cache[key][1] < split.isoformat())
                for split in splits
            )
        ]
        if not stale:
            return

        for key in stale:
            del self._cache[key]
        logger.progress(
            "Refetching %d cached price(s) for %s after a split", len(stale), log_safe(ticker)
        )
        self._rewrite()

    def get(self, ticker: str, day: date) -> float | None:
        """
        Return the price for (ticker, day), using the cache before fetching.
        """
        self._drop_split_stale(ticker)
        day_iso = day.isoformat()
        key = (ticker, day_iso)
        if key in self._cache:
            return self._cache[key][0]
        if key in self._misses:
            return None
        price = self._fetch_fn(ticker, day)
        if not _is_valid_price(price):
            self._record_miss(ticker, day_iso, day)
            return None
        self._cache[key] = (price, self._today.isoformat())
        self._append(ticker, day_iso, price)
        return price

    def last_in_range(self, ticker: str, start: date, end: date) -> float | None:
        """
        Return the last bar's price in (start, end], cached per (ticker, window).
        """
        self._drop_split_stale(ticker)
        # The window key never parses as a date, so it cannot shadow a daily price.
        key_day = f"{start.isoformat()}..{end.isoformat()}"
        key = (ticker, key_day)
        if key in self._cache:
            return self._cache[key][0]
        if key in self._misses:
            return None
        price = self._range_fetch_fn(ticker, start, end)
        if not _is_valid_price(price):
            self._record_miss(ticker, key_day, end)
            return None
        self._cache[key] = (price, self._today.isoformat())
        self._append(ticker, key_day, price)
        return price
