"""
SPY three-month rate of change — the market's own momentum.

Every fund dial on the regime page is about what the tracked managers did; this
line is about the tape they did it on. It is the percent change in SPY over a
rolling 63-session window (about one quarter), the same "3-month ROC" a desk
would pull up on a terminal: above zero the index is higher than it was a
quarter ago and the trend is with you, below zero it is not.

The series is deliberately plain. One ticker, one lookback, no basket
bookkeeping — the only real choices are:

  * The price is `app.stocks.bar_history`'s typical price ((H+L+C)/3), already
    put on today's split basis by the loader, so a split in the window can never
    read as a crash or a melt-up.
  * 63 sessions, not 3 calendar months. A fixed session count keeps every
    reading built from the same amount of data; the calendar drifts with
    holidays.
  * Warm-up sessions before `HISTORY_START` are loaded for the first window's
    lookback but never emitted.
"""

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.stocks.bar_history import Bar, load_daily_bars
from app.utils.logger import get_logger

logger = get_logger(__name__)

# The index proxy. SPY has the deepest, cleanest daily history of the S&P 500 ETFs.
TICKER = "SPY"

# Roughly one quarter of sessions - the standard "3-month rate of change".
WINDOW_SESSIONS = 63

# First reading published. Earlier sessions are only ever window warm-up.
HISTORY_START = date(2016, 1, 1)

# Calendar span loaded before HISTORY_START so the first reading has a full
# window of lookback behind it: 63 sessions is about three months, taken with slack.
WARMUP = timedelta(days=140)

BarLoader = Callable[[str, date], list[Bar]]


def rate_of_change(prices: pd.Series, window: int = WINDOW_SESSIONS) -> pd.Series:
    """
    Percent change of a price series over `window` sessions.

    Args:
        prices: Session-indexed price series, oldest first.
        window: Sessions between the two prices compared.

    Returns:
        pd.Series: `(price_t / price_{t-window} - 1) * 100`, NaN for the first
            `window` sessions where there is no lookback.
    """
    return (prices / prices.shift(window) - 1.0) * 100.0


def _price_series(bars: list[Bar]) -> pd.Series:
    """
    A ticker's bars as a date-indexed price series, one row per session.
    """
    frame = pd.DataFrame({"day": [bar.day for bar in bars], "price": [bar.price for bar in bars]})
    frame = frame.drop_duplicates(subset="day", keep="last").sort_values("day")
    return pd.Series(frame["price"].to_numpy(), index=pd.Index(frame["day"]))


def build_momentum_rows(
    start: date = HISTORY_START,
    ticker: str = TICKER,
    loader: BarLoader = load_daily_bars,
    window: int = WINDOW_SESSIONS,
    warmup: timedelta = WARMUP,
) -> list[dict]:
    """
    Build the persisted rate-of-change series from the index proxy's daily bars.

    Args:
        start: First session to publish a reading for.
        ticker: The index proxy to measure.
        loader: Bar source, taking a ticker and a first session.
        window: Sessions per reading.
        warmup: Calendar span loaded before `start` to fill the first window.

    Returns:
        list[dict]: CSV-ready rows with an ISO date and the reading, oldest first.
    """
    bars = loader(ticker, start - warmup)
    if not bars:
        logger.warning("No %s bars available; momentum series skipped", ticker)
        return []
    roc = rate_of_change(_price_series(bars), window)
    rows: list[dict] = []
    for day, value in zip(roc.index, roc.to_numpy(dtype=float), strict=True):
        if isinstance(day, date) and day >= start and np.isfinite(value):
            rows.append({"date": day.isoformat(), "roc": float(value)})
    return rows
