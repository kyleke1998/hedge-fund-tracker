"""
Bulk-download the daily bars and split history the divergence study needs.

Written as a separate step because it is the only part that touches the network:
it caches close prices, volume and split ex-dates for the whole tracked universe
into gitignored ``__pricecache__/`` so the research driver can be re-run offline
and instantly. Yahoo rate-limits wide sweeps, so batches are small, threaded no
harder than the rest of the project allows, and paced between requests.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from app.backtest.divergence_cache import (  # noqa: E402
    CLOSE_FILE,
    HISTORY_START,
    SPLITS_FILE,
    VOLUME_FILE,
)
from app.backtest.price_cache import CACHE_DIR  # noqa: E402
from app.utils.logger import get_logger, log_safe  # noqa: E402

logger = get_logger(__name__)

BATCH_SIZE = 40
PAUSE_SECONDS = 1.0
MAX_WORKERS = 2


def sanitize(ticker: str) -> str:
    """
    Map a 13F ticker to Yahoo's symbol convention (share classes use a hyphen).
    """
    from app.stocks.libraries import YFinance

    return YFinance._sanitize_ticker(ticker)


def fetch(tickers: list[str], end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download close, volume and splits for every ticker, batched and paced.

    Returns date-indexed close and volume frames keyed by the *original* ticker
    plus a long ``ticker``/``date``/``ratio`` split frame. Symbols Yahoo does not
    know are simply absent from the result.
    """
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    splits: list[dict] = []

    symbol_map = {sanitize(t): t for t in tickers}
    symbols = sorted(symbol_map)
    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start : start + BATCH_SIZE]
        logger.progress("Fetching bars %d-%d of %d", start + 1, start + len(batch), len(symbols))
        try:
            data = yf.download(
                tickers=batch,
                start=HISTORY_START,
                end=end,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                progress=False,
                threads=MAX_WORKERS,
            )
        except Exception:
            logger.error("Batch download failed at offset %d", start, exc_info=True)
            time.sleep(PAUSE_SECONDS * 10)
            continue

        if data is None or data.empty:
            time.sleep(PAUSE_SECONDS)
            continue

        for symbol in batch:
            original = symbol_map[symbol]
            try:
                frame = data[symbol] if len(batch) > 1 else data
            except KeyError:
                continue
            close = frame["Close"].dropna()
            if close.empty:
                continue
            closes[original] = close
            volumes[original] = frame["Volume"].reindex(close.index)
            if "Stock Splits" in frame.columns:
                events = frame["Stock Splits"].dropna()
                for when, ratio in events[events > 0].items():
                    splits.append(
                        {
                            "ticker": original,
                            "date": pd.Timestamp(when).date().isoformat(),
                            "ratio": float(ratio),
                        }
                    )
        time.sleep(PAUSE_SECONDS)

    logger.success("Resolved bars for %d of %d tickers", len(closes), len(tickers))
    return (
        pd.DataFrame(closes).sort_index(),
        pd.DataFrame(volumes).sort_index(),
        pd.DataFrame(splits, columns=["ticker", "date", "ratio"]),
    )


def main() -> None:
    """
    Resolve the tracked universe, download its bars and persist the cache.
    """
    from app.backtest.holdings_panel import build_holdings_panel, universe_tickers

    panel = build_holdings_panel()
    tickers = universe_tickers(panel)
    logger.info("Universe: %d tickers across %d quarters", len(tickers), panel["Quarter"].nunique())

    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).date().isoformat()
    close, volume, splits = fetch([*tickers, "SPY", "IWM", "RSP"], end)

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    close.to_parquet(CLOSE_FILE)
    volume.to_parquet(VOLUME_FILE)
    splits.to_csv(SPLITS_FILE, index=False)
    logger.success(
        "Cached %s bars for %s tickers and %s splits",
        log_safe(str(len(close))),
        log_safe(str(close.shape[1])),
        log_safe(str(len(splits))),
    )


if __name__ == "__main__":
    main()
