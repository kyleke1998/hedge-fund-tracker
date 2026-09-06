import csv
from collections.abc import Callable
from datetime import date
from pathlib import Path

from app.utils.logger import get_logger, log_safe

logger = get_logger(__name__)

CACHE_DIR = "__pricecache__"
CACHE_FILE = "prices.csv"
_FIELDNAMES = ["ticker", "date", "price", "fetched_at"]


class PriceCache:
    """
    Persistent (ticker, date) -> price cache backing the backtest price lookups.

    Caching makes a full regeneration near-instant: changing the tracked-fund
    list reshuffles screen membership but reuses every cached price, fetching
    only genuinely new (ticker, date) pairs. Misses fall through to the injected
    fetcher; only successful (non-None) lookups are persisted, so a transient
    failure is retried on the next run.

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
        splits_fn: Callable[[str], list[date]] | None = None,
        today: date | None = None,
    ) -> None:
        """
        Load any existing cache file and store the fallback price fetcher.
        """
        self._path = Path(path) if path is not None else Path(CACHE_DIR) / CACHE_FILE
        self._fetch_fn = fetch_fn or self._default_fetch_fn
        self._splits_fn = splits_fn or self._default_splits_fn
        self._today = today or date.today()
        self._cache: dict[tuple[str, str], tuple[float, str]] = self._load()
        self._splits_checked: set[str] = set()
        if self._legacy_header:
            # Upgrade in place, or appended rows would carry more fields than the header.
            self._rewrite()

    @staticmethod
    def _default_fetch_fn(ticker: str, day: date) -> float | None:
        """
        Default lookup via the project's free price-fetch chain.
        """
        from app.stocks.price_fetcher import PriceFetcher

        return PriceFetcher.get_avg_price(ticker, day)

    @staticmethod
    def _default_splits_fn(ticker: str) -> list[date]:
        """
        Default split-history lookup via yfinance.
        """
        from app.stocks.libraries import YFinance

        return YFinance.get_splits(ticker)

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
                if not value:
                    continue
                try:
                    cache[(row["ticker"], row["date"])] = (
                        float(value),
                        row.get("fetched_at") or "",
                    )
                except (ValueError, KeyError):
                    continue
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

    def _append(self, ticker: str, day_iso: str, price: float) -> None:
        """
        Append a single resolved price to the on-disk cache (creating it first).
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
                    "price": price,
                    "fetched_at": self._today.isoformat(),
                }
            )

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
        price = self._fetch_fn(ticker, day)
        if price is not None:
            self._cache[key] = (price, self._today.isoformat())
            self._append(ticker, day_iso, price)
        return price
