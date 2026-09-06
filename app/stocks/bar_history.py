"""
Dense daily OHLCV history for one ticker.

`PriceFetcher` answers "what was this worth on that day"; this module answers
"how did this trade over that stretch", which is a different question with
different needs — every session, not a sampled one, and volume alongside price
so an analysis can tell a day the whole market traded from a day nobody did.
Cost-basis estimation is the caller that needs both.

Bars come from a local OHLCV cache when one is configured (`BARS_CACHE_DIR`,
holding a `bars.parquet` of symbol/date/open/high/low/close/volume rows plus a
`splits.parquet` ledger) and from yfinance otherwise. Everything leaves here on
one share basis — today's — because a caller comparing prices with filed share
counts has no way to tell which side of a split a bar sits on. Each bar is priced at its typical price, (H+L+C)/3 —
the standard single-number stand-in for a session's VWAP, and far closer to
where volume actually traded than the close alone.
"""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from app.utils.logger import get_logger, log_safe

logger = get_logger(__name__)

BARS_CACHE_ENV = "BARS_CACHE_DIR"
BARS_FILE = "bars.parquet"
SPLITS_FILE = "splits.parquet"
_COLUMNS = ["date", "high", "low", "close", "volume"]
_PRICE_COLUMNS = ["high", "low", "close"]


@dataclass(frozen=True)
class Bar:
    """
    One session: a representative price and the volume traded that day.
    """

    day: date
    price: float
    volume: float


def _typical_price(frame: pd.DataFrame) -> pd.Series:
    """
    (High + Low + Close) / 3 for every row.
    """
    return (frame["high"] + frame["low"] + frame["close"]) / 3.0


def _to_bars(frame: pd.DataFrame, start: date) -> list[Bar]:
    """
    Convert a cleaned OHLCV frame into ordered bars from `start` onwards.
    """
    if frame.empty:
        return []
    frame = frame.dropna(subset=["date", "high", "low", "close"]).copy()
    frame["day"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame[frame["day"].notna() & (frame["day"] >= start)].sort_values("day")
    prices = _typical_price(frame)
    volumes = (
        frame["volume"].fillna(0.0)
        if "volume" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    return [
        Bar(day=day, price=float(price), volume=float(volume))
        for day, price, volume in zip(frame["day"], prices, volumes, strict=True)
        if price > 0
    ]


def _load_splits(cache_dir: Path, ticker: str) -> list[tuple[str, float]]:
    """
    A ticker's (ex-date, factor) split ledger from the cache, newest first.
    """
    path = cache_dir / SPLITS_FILE
    if not path.is_file():
        return []
    try:
        frame = pd.read_parquet(path, filters=[("symbol", "==", ticker)])
    except Exception:
        logger.warning("Unusable split ledger at %s", log_safe(str(path)), exc_info=True)
        return []
    rows = frame.dropna(subset=["date", "split_factor"]).sort_values("date", ascending=False)
    return [(str(d), float(f)) for d, f in zip(rows["date"], rows["split_factor"], strict=True)]


def _split_adjust(frame: pd.DataFrame, splits: list[tuple[str, float]]) -> pd.DataFrame:
    """
    Put every bar on the latest share basis, whatever basis it arrived on.

    Whether a source hands back as-traded or already-adjusted prices is not
    something to assume — this cache does both, depending on the symbol — so
    each split is settled by the series itself: if the close jumps by roughly
    the split factor across the ex-date the earlier bars are still as-traded and
    get rescaled (prices down, volume up), and if it does not they are already
    adjusted and are left alone. "Roughly" is decided in log space, by whether
    the observed jump sits nearer the factor than it does to no move at all.
    Splits are applied newest first, so rescaling a prefix never disturbs the
    jump an older split is judged by.
    """
    if frame.empty or not splits:
        return frame

    out = frame.sort_values("date").reset_index(drop=True)
    for ex_date, factor in splits:
        if factor <= 0 or factor == 1:
            continue
        before = out[out["date"] < ex_date]
        after = out[out["date"] >= ex_date]
        if before.empty or after.empty:
            continue
        last, first = float(before["close"].iloc[-1]), float(after["close"].iloc[0])
        if last <= 0 or first <= 0:
            continue
        jump = np.log(last / first)
        if abs(jump - np.log(factor)) >= abs(jump):
            continue
        out.loc[before.index, _PRICE_COLUMNS] /= factor
        if "volume" in out.columns:
            out.loc[before.index, "volume"] *= factor
    return out


def _load_from_cache(ticker: str, start: date, cache_dir: Path) -> list[Bar]:
    """
    Read a ticker's bars out of the local OHLCV parquet cache.

    A cache that is absent, unreadable or simply missing the ticker is not an
    error — the caller falls back to the live source.
    """
    path = cache_dir / BARS_FILE
    if not path.is_file():
        return []
    try:
        frame = pd.read_parquet(path, filters=[("symbol", "==", ticker)], columns=_COLUMNS)
    except Exception:
        logger.warning("Unusable bar cache at %s", log_safe(str(path)), exc_info=True)
        return []
    return _to_bars(_split_adjust(frame, _load_splits(cache_dir, ticker)), start)


def _fetch_bars(ticker: str, start: date) -> list[Bar]:
    """
    Daily bars from the live price source, oldest first.
    """
    from app.stocks.libraries import YFinance

    rows = YFinance.get_ohlcv(ticker, start)
    if not rows:
        return []
    return _to_bars(pd.DataFrame(rows), start)


def load_daily_bars(ticker: str, start: date, cache_dir: Path | str | None = None) -> list[Bar]:
    """
    Daily bars for a ticker from `start` onwards, oldest first.

    Args:
        ticker: Stock ticker, as spelled in the cache and at the price source.
        start: First session to return.
        cache_dir: Local OHLCV cache; defaults to `$BARS_CACHE_DIR`, and the
            live source is used when neither is set or the ticker is absent.

    Returns:
        list[Bar]: Ordered bars, empty when no source has the ticker.
    """
    if cache_dir is None:
        load_dotenv()
    folder = cache_dir if cache_dir is not None else os.environ.get(BARS_CACHE_ENV)
    if folder:
        cached = _load_from_cache(ticker, start, Path(folder))
        if cached:
            return cached
    return _fetch_bars(ticker, start)
