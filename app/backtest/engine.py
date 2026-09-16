import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from app.analysis.smart_scores import score_core
from app.analysis.stocks import (
    _aggregate_stock_data,
    _calculate_derived_metrics,
    _calculate_fund_level_flags,
    aggregate_quarter_by_fund,
)
from app.backtest.price_cache import PriceCache
from app.backtest.strategies import DEFAULT_TOP_N, STRATEGIES, StrategySpec, select_screen
from app.database import count_funds_in_quarter, get_all_quarters, load_quarterly_data
from app.utils.logger import get_logger, log_safe
from app.utils.pd import get_numeric_series, get_percentage_number_series
from app.utils.strings import get_quarter_date

logger = get_logger(__name__)

# 13F is due 45 days after quarter-end, and most funds file on the deadline itself —
# often after the close. Entering on day 46 keeps the screen strictly inside what was
# public before the trade, at the cost of one day of drift.
FILING_LAG_DAYS = 46


@dataclass(frozen=True)
class Benchmark:
    """
    A reference index for comparison. The first in ``BENCHMARKS`` is the anchor
    used for the per-strategy ``excess_return``.
    """

    ticker: str
    label: str


BENCHMARKS: list[Benchmark] = [Benchmark("SPY", "S&P 500")]


def _constant_count(n: int) -> Callable[[str], int]:
    """
    A fund-count callable that always answers `n`, whatever the quarter.
    """

    def count(_quarter: str) -> int:
        return n

    return count


def published_funds(
    quarter: str,
    as_of: date,
    *,
    filing_dates_fn: Callable[[], pd.DataFrame] | None = None,
) -> set[str] | None:
    """
    Funds whose 13F for `quarter` was on EDGAR by `as_of`.

    Returns None when the ledger cannot answer for this quarter — no file, or no
    rows for it — so a database without filing dates behaves exactly as before
    rather than screening every fund out. A row whose date will not parse counts
    as unpublished: an unreadable date is not evidence the filing was public.
    """
    from app.database import load_filing_dates

    frame = (filing_dates_fn or load_filing_dates)()
    if frame.empty:
        return None
    rows = frame[frame["Quarter"] == quarter]
    if rows.empty:
        return None
    filed = pd.to_datetime(rows["Filing_Date"], errors="coerce")
    return set(rows.loc[filed.notna() & (filed <= pd.Timestamp(as_of)), "Fund"])


def gate_to_published(
    df: pd.DataFrame,
    quarter: str,
    as_of: date | None,
    *,
    filing_dates_fn: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Drop holdings rows from funds that had not filed for `quarter` by `as_of`.

    A fixed filing lag is not point-in-time: the 45-day deadline slides to the
    next business day, and funds file late or in catch-up batches months
    afterwards. For 2025Q4 the deadline landed on the Presidents' Day weekend,
    so nearly half the tracked funds were still unpublished on the entry date a
    fixed lag assumes they had met. Passing `as_of` as None leaves the frame
    untouched, which is what the non-backtest callers want.
    """
    if as_of is None:
        return df
    allowed = published_funds(quarter, as_of, filing_dates_fn=filing_dates_fn)
    if allowed is None:
        return df
    return df[df["Fund"].isin(allowed)].copy()


def _prepare_quarter_pit(quarter: str, as_of: date | None = None) -> pd.DataFrame:
    """
    Build the point-in-time stock-level analysis frame for a quarter.

    Mirrors the production aggregation but deliberately omits the non-quarterly
    (13D/G, Form 4) merge that ``get_quarter_data`` applies to the latest quarter
    — a backtest must only see data known at filing time. With `as_of` set, funds
    that had not yet filed for the quarter are dropped as well.
    """
    df = gate_to_published(load_quarterly_data(quarter), quarter, as_of)
    df["Delta_Value_Num"] = get_numeric_series(df["Delta_Value"])
    df["Value_Num"] = get_numeric_series(df["Value"])
    df["Portfolio_Pct"] = get_percentage_number_series(df["Portfolio%"])
    df_fund = _calculate_fund_level_flags(aggregate_quarter_by_fund(df))
    return _calculate_derived_metrics(_aggregate_stock_data(df_fund))


def min_holders_for_quarter(
    quarter: str, *, divisor: int = 10, fund_count_fn: Callable[[str], int] | None = None
) -> int:
    """
    Membership threshold for a quarter: ceil(funds / divisor).

    The default divisor (10 → ~10% of funds) matches the QuarterlyTrends UI; a
    strategy can widen the net with a larger divisor (e.g. Increasing uses 20 →
    ~5%). Both sides compute it the same way, so the backtest screens match the UI.
    """
    count = (fund_count_fn or count_funds_in_quarter)(quarter)
    return math.ceil(count / divisor)


def quarter_entry_date(quarter: str) -> date:
    """
    Return the strategy's entry date for a quarter (quarter-end + filing lag) —
    the first session after the 13F deadline, so no filing published that day
    can influence a trade made at that day's price.
    """
    quarter_end = datetime.strptime(get_quarter_date(quarter), "%Y-%m-%d").date()
    return quarter_end + timedelta(days=FILING_LAG_DAYS)


def quarter_smart_scores(
    quarter: str, *, analysis_fn: Callable[[str], pd.DataFrame] | None = None
) -> pd.Series:
    """
    Ticker -> published smart score for a quarter, as known at filing time.

    Computed on the quarter's *whole* point-in-time universe. The score's three
    components are percentile ranks, so a caller that scores an already-filtered
    frame would get a different number for the same stock in the same quarter —
    this is the value the site and the smart-score screen actually show.
    """
    df = (analysis_fn or _prepare_quarter_pit)(quarter)
    if df.empty:
        return pd.Series(dtype="float64", name="Smart_Score")
    scores = pd.Series(
        score_core(df).to_numpy(), index=pd.Index(df["Ticker"], name="Ticker"), name="Smart_Score"
    )
    # The frame groups by ticker *and* company, so one ticker carried under two
    # company spellings splits into rows that each understate its footprint.
    # Keeping the strongest collapses it back to the row describing the stock.
    return scores.groupby(level="Ticker").max()


def build_screen(
    quarter: str,
    spec: StrategySpec,
    *,
    threshold: int,
    analysis_fn: Callable[[str], pd.DataFrame] | None = None,
    top_n: int = DEFAULT_TOP_N,
    as_of: date | None = None,
) -> dict[str, float]:
    """
    Reconstruct a strategy's screen for a quarter as ticker -> weight.

    Selection follows the strategy spec; weights are each name's
    ``Avg_Portfolio_Pct`` normalized to sum 1.0. Returns an empty dict when no
    stock qualifies. `as_of` gates the universe to filings already public on
    that date; it is ignored when `analysis_fn` supplies the frame.
    """
    df = analysis_fn(quarter) if analysis_fn else _prepare_quarter_pit(quarter, as_of=as_of)
    if spec.sort_column == "Smart_Score" and "Smart_Score" not in df.columns:
        # Derived lazily from the frame itself (works for injected frames too).
        df = df.assign(Smart_Score=score_core(df))
    selected = select_screen(df, spec, threshold=threshold, top_n=top_n)
    tickers = [str(t) for t in selected["Ticker"].tolist()]
    weights = [float(w) for w in selected["Avg_Portfolio_Pct"].tolist()]
    total = sum(weights)
    if total <= 0:
        return {}
    return {ticker: weight / total for ticker, weight in zip(tickers, weights, strict=True)}


def _window_return(
    screen: dict[str, float],
    entry: date,
    exit_date: date,
    price_fn: Callable[[str, date], float | None],
    label: str,
) -> tuple[float, set[str]] | None:
    """
    Conviction-weighted return for one window plus the set of priced names.

    Constituents missing an entry/exit price are dropped (logged) and the rest
    renormalized. Returns None if no constituent has prices.
    """
    kept: dict[str, tuple[float, float]] = {}
    for ticker, weight in screen.items():
        p_entry = price_fn(ticker, entry)
        p_exit = price_fn(ticker, exit_date)
        if p_entry and p_exit and p_entry > 0:
            kept[ticker] = (weight, p_exit / p_entry - 1)
        else:
            logger.warning(
                "Backtest excluded %s in %s: missing price", log_safe(ticker), log_safe(label)
            )
    if not kept:
        return None
    total_weight = sum(w for w, _ in kept.values())
    conviction = sum((w / total_weight) * r for w, r in kept.values())
    return conviction, set(kept)


def run_backtest(
    *,
    price_fn: Callable[[str, date], float | None] | None = None,
    as_of: date | None = None,
    analysis_fn: Callable[[str], pd.DataFrame] | None = None,
    quarters: list[str] | None = None,
    strategies: list[StrategySpec] = STRATEGIES,
    benchmarks: list[Benchmark] = BENCHMARKS,
    fund_count_fn: Callable[[str], int] | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """
    Backtest every strategy and benchmark over consecutive consolidated quarters.

    Returns a flat (long) list of rows — one per (series, window) — where a
    series is a strategy or a benchmark. Each row carries the per-window and
    cumulative return; strategy rows also carry stock count, excess vs the anchor
    benchmark, and turnover. Dependencies are injectable for testing.
    """
    price = price_fn or PriceCache().get
    resolved_as_of = as_of or date.today()
    resolved_quarters = quarters if quarters is not None else sorted(get_all_quarters())
    anchor = benchmarks[0].ticker if benchmarks else None

    rows: list[dict] = []
    bench_cum = {b.ticker: 1.0 for b in benchmarks}
    strat_cum = {s.strategy_id: 1.0 for s in strategies}
    prev_screen: dict[str, set[str]] = {}

    for quarter_in, quarter_out in zip(resolved_quarters, resolved_quarters[1:], strict=False):
        entry = quarter_entry_date(quarter_in)
        exit_date = quarter_entry_date(quarter_out)
        if exit_date > resolved_as_of:
            continue  # window not yet consolidated

        common = {
            "quarter_in": quarter_in,
            "quarter_out": quarter_out,
            "entry_date": entry.isoformat(),
            "exit_date": exit_date.isoformat(),
        }

        bench_ret: dict[str, float] = {}
        for bench in benchmarks:
            p_entry = price(bench.ticker, entry)
            p_exit = price(bench.ticker, exit_date)
            if not p_entry or not p_exit or p_entry <= 0:
                logger.warning("Backtest missing benchmark price for %s", log_safe(bench.ticker))
                continue
            ret = p_exit / p_entry - 1
            bench_ret[bench.ticker] = ret
            bench_cum[bench.ticker] *= 1 + ret
            rows.append(
                {
                    **common,
                    "series_type": "benchmark",
                    "series_id": bench.ticker,
                    "label": bench.label,
                    "n_stocks": None,
                    "window_return": ret,
                    "cum_return": bench_cum[bench.ticker] - 1,
                    "excess_return": None,
                    "turnover": None,
                }
            )

        anchor_ret = bench_ret.get(anchor) if anchor else None
        # The holder threshold has to count the funds the screen can actually see,
        # or a quarter where many funds have yet to file sets a bar its own
        # shrunken universe cannot clear.
        visible = fund_count_fn
        if visible is None:
            published = published_funds(quarter_in, entry)
            if published is not None:
                visible = _constant_count(len(published))
        for spec in strategies:
            threshold = min_holders_for_quarter(
                quarter_in, divisor=spec.min_holders_divisor, fund_count_fn=visible
            )
            screen = build_screen(
                quarter_in,
                spec,
                threshold=threshold,
                analysis_fn=analysis_fn,
                top_n=top_n,
                as_of=entry,
            )
            window = _window_return(screen, entry, exit_date, price, spec.strategy_id)
            if window is None:
                continue
            conviction, kept = window
            strat_cum[spec.strategy_id] *= 1 + conviction
            prev = prev_screen.get(spec.strategy_id)
            turnover = 0.0 if prev is None else 1.0 - len(kept & prev) / len(kept)
            rows.append(
                {
                    **common,
                    "series_type": "strategy",
                    "series_id": spec.strategy_id,
                    "label": spec.label,
                    "n_stocks": len(kept),
                    "window_return": conviction,
                    "cum_return": strat_cum[spec.strategy_id] - 1,
                    "excess_return": (conviction - anchor_ret) if anchor_ret is not None else None,
                    "turnover": turnover,
                }
            )
            prev_screen[spec.strategy_id] = kept

    return rows
