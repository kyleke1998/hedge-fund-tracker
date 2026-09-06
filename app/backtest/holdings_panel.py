"""
Long-format tracked-fund holdings panel used by the divergence study.

One row per (quarter, fund, ticker) with share counts restated into post-split
units, which is the input every accumulation feature is derived from. Ticker
resolution goes through ``stocks.csv`` rather than the filing's own symbol so a
company reported under two CUSIPs collapses into one position, matching what the
production stock-level aggregation does.
"""

from collections.abc import Callable, Sequence
from datetime import date, datetime

import pandas as pd

from app.backtest.divergence import split_adjust_factor
from app.utils.pd import get_numeric_series, get_percentage_number_series
from app.utils.strings import get_quarter_date

# Fewest funds that must hold a name in some quarter for it to enter the study.
# One or two holders cannot express a consensus, and the universe size drives the
# price download.
DEFAULT_MIN_HOLDERS = 3


def build_holdings_panel(
    quarters: Sequence[str] | None = None,
    *,
    load_fn: Callable[[str], pd.DataFrame] | None = None,
    stocks_fn: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Assemble every tracked quarter into ``Quarter``/``Fund``/``Ticker`` holdings.

    Rows without a resolved ticker, and closed positions, are dropped; formatted
    ``Value`` / ``Portfolio%`` strings are parsed to numbers. Dependencies are
    injectable so the tests do not touch the CSV database.
    """
    from app.database import get_all_quarters, load_quarterly_data, load_stocks

    load = load_fn or load_quarterly_data
    stocks = (stocks_fn or load_stocks)()
    resolved = sorted(quarters if quarters is not None else get_all_quarters())

    frames = []
    for quarter in resolved:
        df = load(quarter)
        if df.empty:
            continue
        df = (
            df.drop(columns=["Ticker", "Company"], errors="ignore")
            .set_index("CUSIP")
            .join(stocks[["Ticker"]], how="left")
            .reset_index()
        )
        df = df[df["Ticker"].notna()]
        df["Value"] = get_numeric_series(df["Value"])
        df["Portfolio_Pct"] = get_percentage_number_series(df["Portfolio%"])
        grouped = (
            df.groupby(["Fund", "Ticker"])
            .agg(
                Shares=("Shares", "sum"),
                Value=("Value", "sum"),
                Portfolio_Pct=("Portfolio_Pct", "sum"),
            )
            .reset_index()
        )
        grouped.insert(0, "Quarter", quarter)
        frames.append(grouped[grouped["Shares"] > 0])

    if not frames:
        return pd.DataFrame(
            columns=["Quarter", "Fund", "Ticker", "Shares", "Value", "Portfolio_Pct"]
        )
    return pd.concat(frames, ignore_index=True)


def universe_tickers(panel: pd.DataFrame, *, min_holders: int = DEFAULT_MIN_HOLDERS) -> list[str]:
    """
    Tickers reaching the holder floor in at least one quarter, sorted.
    """
    if panel.empty:
        return []
    holders = panel.groupby(["Quarter", "Ticker"])["Fund"].nunique()
    return sorted(holders[holders >= min_holders].index.get_level_values("Ticker").unique())


def apply_split_adjustment(panel: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``Adj_Shares``: reported shares restated into current, post-split units.

    A filing reports the share count as of quarter-end, so only splits with an
    ex-date after that quarter-end still have to be applied. Doing this before any
    quarter-over-quarter comparison is what keeps a 10-for-1 split from reading as
    a 900% accumulation, and it also keeps shares consistent with Yahoo's bars,
    which are split-adjusted at source.
    """
    result = panel.copy()
    if result.empty:
        result["Adj_Shares"] = pd.Series(dtype="float64")
        return result

    by_ticker: dict[str, list[tuple[date, float]]] = {}
    if not splits.empty:
        for ticker, when, ratio in zip(
            splits["ticker"].astype(str),
            splits["date"].astype(str),
            splits["ratio"].astype(float),
            strict=True,
        ):
            ex_date = datetime.strptime(when, "%Y-%m-%d").date()
            by_ticker.setdefault(ticker, []).append((ex_date, ratio))

    quarter_ends = {
        q: datetime.strptime(get_quarter_date(q), "%Y-%m-%d").date()
        for q in result["Quarter"].unique()
    }
    factors = [
        split_adjust_factor(quarter_ends[quarter], by_ticker.get(ticker, []))
        for quarter, ticker in zip(result["Quarter"], result["Ticker"], strict=True)
    ]
    result["Adj_Shares"] = result["Shares"].astype(float) * pd.Series(factors, index=result.index)
    return result
