"""
Realized average stock correlation inside the QQQ basket.

Every other signal on the regime page is about what the tracked funds did; this
one is about the market they did it in. When average pairwise correlation is
high the index moves as one block and stock picking earns little — dispersion
has collapsed and the whole basket is trading on one macro factor. When it is
low, names move on their own news, which is the regime a concentrated book of
individual positions is built for.

The reading is the mean of every pairwise correlation of daily log returns
across the basket over a rolling three-month window (63 sessions), equally
weighted over pairs rather than by market cap: it is a statement about how
uniformly the constituents move, not about how the index performed.

Three things shape the implementation:

  * The basket is fixed, not re-picked each quarter, so a change in the line is
    a change in behaviour rather than a change in membership (same reasoning as
    the mega-cap basket in `app.analysis.regime`). It is a membership snapshot
    of the Nasdaq-100 and carries the survivorship bias that implies.
  * QQQ's own bars define the session calendar. A name is measured on the
    index's trading days or not at all, so a thin foreign holiday can never
    invent a shared 0% move across half the basket.
  * A name missing any session inside a window is dropped from that window
    rather than forward-filled. A carried-forward price reads as a flat day,
    and flat days pull correlation toward zero for a purely clerical reason.
  * A session that most of the then-listed basket has no bar for is a hole in
    the price source rather than a market day, and is removed before returns
    are taken. Left in, one such date would evict every absent name from three
    months of windows, quietly halving the basket the readings come from.
"""

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.stocks.bar_history import Bar, load_daily_bars
from app.utils.logger import get_logger, log_safe

logger = get_logger(__name__)

# QQQ tracks the Nasdaq-100, and its own bars supply the session calendar.
CALENDAR_TICKER = "QQQ"

# Nasdaq-100 membership snapshot, held fixed so the series stays comparable.
QQQ_BASKET = (
    "AAPL",
    "ABNB",
    "ADBE",
    "ADI",
    "ADP",
    "ADSK",
    "AEP",
    "AMAT",
    "AMD",
    "AMGN",
    "AMZN",
    "APP",
    "ARM",
    "AVGO",
    "AXON",
    "AZN",
    "BIIB",
    "BKNG",
    "BKR",
    "CCEP",
    "CDNS",
    "CDW",
    "CEG",
    "CHTR",
    "CMCSA",
    "COST",
    "CPRT",
    "CRWD",
    "CSCO",
    "CSGP",
    "CSX",
    "CTAS",
    "CTSH",
    "DASH",
    "DDOG",
    "DXCM",
    "EA",
    "EXC",
    "FANG",
    "FAST",
    "FTNT",
    "GEHC",
    "GFS",
    "GILD",
    "GOOG",
    "GOOGL",
    "HON",
    "IDXX",
    "INTC",
    "INTU",
    "ISRG",
    "KDP",
    "KHC",
    "KLAC",
    "LIN",
    "LRCX",
    "LULU",
    "MAR",
    "MCHP",
    "MDB",
    "MDLZ",
    "MELI",
    "META",
    "MNST",
    "MRVL",
    "MSFT",
    "MSTR",
    "MU",
    "NFLX",
    "NVDA",
    "NXPI",
    "ODFL",
    "ON",
    "ORLY",
    "PANW",
    "PAYX",
    "PCAR",
    "PDD",
    "PEP",
    "PLTR",
    "PYPL",
    "QCOM",
    "REGN",
    "ROP",
    "ROST",
    "SBUX",
    "SNPS",
    "TEAM",
    "TMUS",
    "TSLA",
    "TTD",
    "TTWO",
    "TXN",
    "VRSK",
    "VRTX",
    "WBD",
    "WDAY",
    "XEL",
    "ZS",
)

# Roughly one quarter of sessions - the standard "3-month realized" window.
WINDOW_SESSIONS = 63

# Share of the then-listed basket that must have a bar for a date to count as a
# session. Set high: real absences are rare, so a date many names are missing is
# almost always the source's gap rather than theirs.
MIN_SESSION_COVERAGE = 0.9

# Below this many usable names the mean is a handful of pairs, not a market.
MIN_STOCKS = 30

# First reading published. Earlier sessions are only ever window warm-up.
HISTORY_START = date(2016, 1, 1)

# Calendar span loaded before HISTORY_START so the first reading has a full
# window behind it: 63 sessions is about three months, taken with slack.
WARMUP = timedelta(days=200)

BarLoader = Callable[[str, date], list[Bar]]


def average_pairwise_correlation(returns: np.ndarray) -> float | None:
    """
    Mean correlation over every distinct pair of columns in a return block.

    Computed from the standardized columns rather than an explicit correlation
    matrix: with z-scored columns z_i, the sum of the whole correlation matrix
    is ||sum_i z_i||^2 / (T - 1), so removing the N diagonal ones leaves the
    pair total in one pass instead of N^2.

    Args:
        returns: Sessions-by-names block of returns, complete and non-constant.

    Returns:
        float | None: The average pairwise correlation, or None when there is
            no pair to measure.
    """
    rows, names = returns.shape
    if names < 2 or rows < 2:
        return None
    centred = returns - returns.mean(axis=0)
    standardized = centred / centred.std(axis=0, ddof=1)
    total = float(np.square(standardized.sum(axis=1)).sum()) / (rows - 1)
    return (total - names) / (names * (names - 1))


def usable_block(window: pd.DataFrame) -> np.ndarray:
    """
    The columns of a window that can be correlated: complete and moving.
    """
    block = window.to_numpy(dtype=float)
    block = block[:, ~np.isnan(block).any(axis=0)]
    if block.size == 0:
        return block
    return block[:, block.std(axis=0, ddof=1) > 0]


def _price_series(bars: Sequence[Bar]) -> pd.Series:
    """
    A ticker's bars as a date-indexed price series, one row per session.
    """
    frame = pd.DataFrame({"day": [bar.day for bar in bars], "price": [bar.price for bar in bars]})
    frame = frame.drop_duplicates(subset="day", keep="last").sort_values("day")
    return pd.Series(frame["price"].to_numpy(), index=pd.Index(frame["day"]))


def _covered_sessions(prices: pd.DataFrame, min_coverage: float) -> pd.Index:
    """
    The sessions enough of the then-listed basket actually traded on.

    A ticker counts towards a session only between its own first and last bar,
    so names that had not listed yet - or have since been taken private - never
    make a session look uncovered.
    """
    present = prices.notna()
    listed = present.cummax() & present[::-1].cummax()[::-1]
    expected = listed.sum(axis=1)
    covered = present.sum(axis=1) >= expected * min_coverage
    return prices.index[covered & (expected > 0)]


def build_return_panel(
    bars_by_ticker: Mapping[str, Sequence[Bar]],
    sessions: Sequence[date],
    min_session_coverage: float = MIN_SESSION_COVERAGE,
) -> pd.DataFrame:
    """
    Daily log returns for every ticker, aligned on one session calendar.

    Args:
        bars_by_ticker: Loaded bars keyed by ticker.
        sessions: The trading days to measure on, oldest first.
        min_session_coverage: Share of the then-listed basket that must have a
            bar for a session to be kept.

    Returns:
        pd.DataFrame: One row per kept session after the first, one column per
            ticker, with NaN wherever a ticker did not trade that session.
    """
    index = pd.Index(list(sessions))
    prices = pd.DataFrame(
        {ticker: _price_series(bars).reindex(index) for ticker, bars in bars_by_ticker.items()},
        index=index,
    )
    prices = prices.loc[_covered_sessions(prices, min_session_coverage)]
    logs = pd.DataFrame(
        np.log(prices.to_numpy(dtype=float)), index=prices.index, columns=prices.columns
    )
    return logs.diff().iloc[1:]


def rolling_average_correlation(
    returns: pd.DataFrame,
    window: int = WINDOW_SESSIONS,
    min_stocks: int = MIN_STOCKS,
) -> list[dict]:
    """
    The rolling average pairwise correlation series, one reading per session.

    Args:
        returns: Return panel from `build_return_panel`.
        window: Sessions per reading.
        min_stocks: Fewest usable names a reading may be built from.

    Returns:
        list[dict]: Rows of date, correlation and contributing name count, for
            every session with a full window and enough coverage.
    """
    rows: list[dict] = []
    for end in range(window, len(returns) + 1):
        block = usable_block(returns.iloc[end - window : end])
        if block.shape[1] < min_stocks:
            continue
        value = average_pairwise_correlation(block)
        if value is None:
            continue
        rows.append(
            {"date": returns.index[end - 1], "correlation": value, "stocks": int(block.shape[1])}
        )
    return rows


def build_correlation_rows(
    start: date = HISTORY_START,
    tickers: Sequence[str] = QQQ_BASKET,
    loader: BarLoader = load_daily_bars,
    window: int = WINDOW_SESSIONS,
    min_stocks: int = MIN_STOCKS,
    warmup: timedelta = WARMUP,
    min_session_coverage: float = MIN_SESSION_COVERAGE,
) -> list[dict]:
    """
    Build the persisted correlation series from the basket's daily bars.

    Args:
        start: First session to publish a reading for.
        tickers: The basket to correlate.
        loader: Bar source, taking a ticker and a first session.
        window: Sessions per reading.
        min_stocks: Fewest usable names a reading may be built from.
        warmup: Calendar span loaded before `start` to fill the first window.
        min_session_coverage: Share of the then-listed basket that must have a
            bar for a session to be kept.

    Returns:
        list[dict]: CSV-ready rows with an ISO date, the correlation and the
            number of names behind it, oldest first.
    """
    first = start - warmup
    calendar = loader(CALENDAR_TICKER, first)
    if not calendar:
        logger.warning("No %s bars available; correlation series skipped", CALENDAR_TICKER)
        return []
    sessions = [bar.day for bar in calendar]

    bars_by_ticker: dict[str, Sequence[Bar]] = {}
    for ticker in tickers:
        bars = loader(ticker, first)
        if not bars:
            logger.warning("No bars for %s; excluded from the basket", log_safe(ticker))
            continue
        bars_by_ticker[ticker] = bars
    logger.progress("Loaded %d of %d basket names", len(bars_by_ticker), len(tickers))

    panel = build_return_panel(bars_by_ticker, sessions, min_session_coverage)
    readings = rolling_average_correlation(panel, window=window, min_stocks=min_stocks)
    return [
        {
            "date": row["date"].isoformat(),
            "correlation": row["correlation"],
            "stocks": row["stocks"],
        }
        for row in readings
        if row["date"] >= start
    ]
