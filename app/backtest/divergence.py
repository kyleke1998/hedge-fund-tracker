"""
Price / institutional-accumulation divergence features and the statistics used
to judge them.

The thesis under test: a stock whose price is falling, flat, or only drifting up
while the tracked funds keep adding shares is being accumulated quietly, and the
mirror image (a stock that has already run while funds distribute) is being sold
into strength. Everything here is a pure function over frames so the research
driver and the tests exercise identical code.

Two measurement rules make the features honest and are enforced throughout:

* **Split adjustment.** Reported share counts are restated into post-split units
  before any quarter-over-quarter comparison; a 10-for-1 split otherwise reads as
  a 900% accumulation.
* **Balanced fund panel.** Share changes are computed only across funds that
  filed in *both* quarters. The tracked-fund list grew from 88 to 147 funds over
  the sample, and counting a newly tracked fund's pre-existing position as a
  purchase fabricates accumulation in exactly the names most funds already hold.
"""

from collections.abc import Iterable, Sequence
from datetime import date

import numpy as np
import pandas as pd

# Inputs to the accumulation half of the divergence composite.
ACCUMULATION_COLUMNS: tuple[str, ...] = (
    "Share_Chg_Pct_Capped",
    "Breadth_Avg",
    "Acc_Persistence",
    "Acc_Days_Of_Volume",
)

# Inputs to the price-strength half. The composite subtracts these, so a name
# scores well by being accumulated *and* unloved.
PRICE_STRENGTH_COLUMNS: tuple[str, ...] = ("Ret_6m", "Trend_Tstat", "Px_vs_MA200")

# A quarter-on-quarter share change is unbounded to the upside (a fund quadrupling
# a stub position) but floored at -100%. Capping keeps the raw value usable as a
# gate input; the composite itself ranks, so the cap only matters for reporting.
SHARE_CHANGE_CAP = 2.0

# Trailing windows in trading days.
BARS_QUARTER = 63
BARS_HALF_YEAR = 126
BARS_YEAR = 252
BARS_MA_LONG = 200

# Fewest bars before the entry date for a ticker to get price features at all.
MIN_PRICE_BARS = 60

# Fewest observations for a trend regression to mean anything.
MIN_TREND_BARS = 10


def split_adjust_factor(reference: date, splits: Iterable[tuple[date, float]]) -> float:
    """
    Multiplier restating a share count reported on ``reference`` into today's units.

    Yahoo's historical OHLC is split-adjusted at source, so a price from before a
    split is already expressed post-split while the 13F share count is not. Every
    split with an ex-date strictly after the reference date has to be applied to
    the share side to put the two on the same footing.
    """
    factor = 1.0
    for ex_date, ratio in splits:
        if ex_date > reference and ratio > 0:
            factor *= ratio
    return factor


def balanced_share_delta(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ticker share change between two quarters across funds present in both.

    Both frames carry ``Fund``, ``Ticker`` and ``Adj_Shares`` (split-adjusted).
    Returns one row per ticker held by a common fund in either quarter, with the
    common-fund share totals, the percentage change, buyer/seller/holder counts
    and their breadth. ``All_New`` marks a ticker no common fund held before, for
    which a percentage change is undefined.
    """
    common = sorted(set(prev["Fund"]) & set(curr["Fund"]))
    if not common:
        return _empty_delta_frame()

    prev_c = prev[prev["Fund"].isin(common)]
    curr_c = curr[curr["Fund"].isin(common)]
    tickers = sorted(set(prev_c["Ticker"]) | set(curr_c["Ticker"]))
    if not tickers:
        return _empty_delta_frame()

    index = pd.MultiIndex.from_product([common, tickers], names=["Fund", "Ticker"])
    prev_s = prev_c.groupby(["Fund", "Ticker"])["Adj_Shares"].sum().reindex(index, fill_value=0.0)
    curr_s = curr_c.groupby(["Fund", "Ticker"])["Adj_Shares"].sum().reindex(index, fill_value=0.0)

    by_ticker = pd.DataFrame(
        {
            "Prev_Shares": prev_s.groupby("Ticker").sum(),
            "Curr_Shares": curr_s.groupby("Ticker").sum(),
            "Buyers": (curr_s > prev_s).groupby("Ticker").sum(),
            "Sellers": (curr_s < prev_s).groupby("Ticker").sum(),
            "Holders": (curr_s > 0).groupby("Ticker").sum(),
        }
    )
    by_ticker["All_New"] = by_ticker["Prev_Shares"] <= 0
    by_ticker["Share_Chg_Pct"] = np.where(
        by_ticker["Prev_Shares"] > 0,
        by_ticker["Curr_Shares"] / by_ticker["Prev_Shares"].replace(0.0, np.nan) - 1.0,
        np.nan,
    )
    by_ticker["Breadth"] = np.where(
        by_ticker["Holders"] > 0,
        (by_ticker["Buyers"] - by_ticker["Sellers"]) / by_ticker["Holders"].replace(0, np.nan),
        np.nan,
    )
    return by_ticker.reset_index()[list(_empty_delta_frame().columns)]


def _empty_delta_frame() -> pd.DataFrame:
    """
    Column template for ``balanced_share_delta`` so callers can rely on the shape.
    """
    return pd.DataFrame(
        {
            "Ticker": pd.Series(dtype="object"),
            "Prev_Shares": pd.Series(dtype="float64"),
            "Curr_Shares": pd.Series(dtype="float64"),
            "Share_Chg_Pct": pd.Series(dtype="float64"),
            "Buyers": pd.Series(dtype="int64"),
            "Sellers": pd.Series(dtype="int64"),
            "Holders": pd.Series(dtype="int64"),
            "Breadth": pd.Series(dtype="float64"),
            "All_New": pd.Series(dtype="bool"),
        }
    )


def accumulation_features(panel: pd.DataFrame, quarter: str, *, lookback: int = 4) -> pd.DataFrame:
    """
    Institutional-accumulation features for one quarter, as known at its filing.

    ``panel`` is long-format ``Quarter``/``Fund``/``Ticker``/``Adj_Shares``. The
    latest transition gives the one-quarter change; the ``lookback``-quarter
    window gives the multi-quarter change, the count of consecutive-ish quarters
    of net buying (``Acc_Persistence``) and the average breadth over the window —
    the features that separate steady accumulation from a single opportunistic
    add. Returns an empty frame when the quarter has no predecessor in the panel.
    """
    quarters = sorted(panel["Quarter"].unique())
    if quarter not in quarters:
        return _empty_delta_frame()
    idx = quarters.index(quarter)
    if idx == 0:
        return _empty_delta_frame()

    slices = {
        q: panel[panel["Quarter"] == q] for q in quarters[max(0, idx - lookback + 1) : idx + 1]
    }
    window = sorted(slices)

    base = balanced_share_delta(slices[window[-2]], slices[quarter])
    if base.empty:
        return base

    transitions = [
        balanced_share_delta(slices[a], slices[b]) for a, b in zip(window, window[1:], strict=False)
    ]
    accumulating = [
        t.set_index("Ticker")["Curr_Shares"] > t.set_index("Ticker")["Prev_Shares"]
        for t in transitions
    ]
    breadths = [t.set_index("Ticker")["Breadth"] for t in transitions]

    base = base.set_index("Ticker")
    base["Acc_Persistence"] = (
        pd.concat(accumulating, axis=1).reindex(base.index).sum(axis=1).astype("int64")
    )
    base["Acc_Quarters"] = len(transitions)
    base["Breadth_Avg"] = pd.concat(breadths, axis=1).reindex(base.index).mean(axis=1)

    span = balanced_share_delta(slices[window[0]], slices[quarter]).set_index("Ticker")
    base[f"Share_Chg_Pct_{lookback}q"] = span["Share_Chg_Pct"].reindex(base.index)
    base["Share_Chg_Pct_Capped"] = base["Share_Chg_Pct"].clip(upper=SHARE_CHANGE_CAP)
    return base.reset_index()


def trend_tstat(prices: pd.Series) -> float:
    """
    Volatility-normalised slope of log price against time.

    The t-statistic of the OLS slope, which reads "flat" the same way whether the
    stock trades at $5 or $500 and penalises a drift that is small relative to
    its own noise. Returns NaN below ``MIN_TREND_BARS`` observations.
    """
    series = pd.Series(prices).dropna()
    series = series[series > 0]
    if len(series) < MIN_TREND_BARS:
        return float("nan")

    y = np.log(series.to_numpy(dtype=float))
    x = np.arange(len(y), dtype=float)
    x_centred = x - x.mean()
    denominator = float((x_centred**2).sum())
    slope = float((x_centred * (y - y.mean())).sum() / denominator)
    residuals = y - (y.mean() + slope * x_centred)
    dof = len(y) - 2
    residual_var = float((residuals**2).sum()) / dof
    if residual_var <= 0:
        return 0.0 if slope == 0 else float(np.sign(slope) * np.inf)
    return slope / float(np.sqrt(residual_var / denominator))


def price_features(prices: pd.DataFrame, volume: pd.DataFrame, entry: date) -> pd.DataFrame:
    """
    Trailing price features per ticker, using only bars at or before ``entry``.

    ``prices`` and ``volume`` are date-indexed frames with one column per ticker.
    Truncating to the entry date is what keeps the study point-in-time: the
    result is identical whether or not later bars are in the input. Tickers with
    fewer than ``MIN_PRICE_BARS`` bars are dropped.
    """
    cutoff = pd.Timestamp(entry)
    px = prices.loc[prices.index <= cutoff]
    vol = volume.loc[volume.index <= cutoff]

    rows = []
    for ticker in px.columns:
        series = px[ticker].dropna()
        if len(series) < MIN_PRICE_BARS:
            continue
        last = float(series.iloc[-1])
        dollar_volume = (px[ticker] * vol[ticker]).dropna().tail(BARS_QUARTER)
        rows.append(
            {
                "Ticker": ticker,
                "Px_Entry": last,
                "Ret_1q": _trailing_return(series, BARS_QUARTER),
                "Ret_6m": _trailing_return(series, BARS_HALF_YEAR),
                "Ret_12m": _trailing_return(series, BARS_YEAR),
                "Trend_Tstat": trend_tstat(series.tail(BARS_HALF_YEAR)),
                "Px_vs_MA200": last / float(series.tail(BARS_MA_LONG).mean()) - 1.0,
                "Drawdown_52w": last / float(series.tail(BARS_YEAR).max()) - 1.0,
                "Vol_63": _realised_vol(series.tail(BARS_QUARTER + 1)),
                "Dollar_Vol_63": float(dollar_volume.median()) if len(dollar_volume) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=_price_feature_columns())


def _price_feature_columns() -> list[str]:
    """
    Column order emitted by ``price_features`` (also the empty-frame template).
    """
    return [
        "Ticker",
        "Px_Entry",
        "Ret_1q",
        "Ret_6m",
        "Ret_12m",
        "Trend_Tstat",
        "Px_vs_MA200",
        "Drawdown_52w",
        "Vol_63",
        "Dollar_Vol_63",
    ]


def _trailing_return(series: pd.Series, bars: int) -> float:
    """
    Simple return over the last ``bars`` bars, or NaN when history is too short.
    """
    if len(series) <= bars:
        return float("nan")
    start = float(series.iloc[-bars - 1])
    return float(series.iloc[-1]) / start - 1.0 if start > 0 else float("nan")


def _realised_vol(series: pd.Series) -> float:
    """
    Annualised standard deviation of daily log returns.
    """
    ratios = (series / series.shift(1)).dropna()
    if len(ratios) < 2:
        return float("nan")
    log_returns = np.log(ratios.to_numpy(dtype=float))
    return float(log_returns.std(ddof=1) * np.sqrt(BARS_YEAR))


def rank_z(values: pd.Series) -> pd.Series:
    """
    Cross-sectional rank score, centred at 0 with unit standard deviation.

    Ranking rather than z-scoring the raw values is what stops one stub position
    that quintupled from swamping a composite. Missing values stay missing; a
    column with no dispersion scores flat zero.
    """
    series = pd.Series(values, dtype="float64")
    count = int(series.notna().sum())
    if count == 0:
        return pd.Series(np.nan, index=series.index, dtype="float64")
    uniform = (series.rank(method="average") - 0.5) / count
    return (uniform - 0.5) * np.sqrt(12.0)


def divergence_score(
    df: pd.DataFrame,
    *,
    accumulation_columns: Sequence[str] = ACCUMULATION_COLUMNS,
    price_columns: Sequence[str] = PRICE_STRENGTH_COLUMNS,
) -> pd.Series:
    """
    Cross-sectional divergence: mean accumulation rank minus mean price-strength rank.

    Positive means the tracked funds are buying a name the market has not bid up
    — the buy side of the thesis; negative means funds are trimming a name that
    has already run. A row with no accumulation input at all scores NaN rather
    than defaulting to the middle of the pack.
    """
    accumulation = pd.concat([rank_z(df[c]) for c in accumulation_columns], axis=1).mean(axis=1)
    strength = pd.concat([rank_z(df[c]) for c in price_columns], axis=1).mean(axis=1)
    return accumulation - strength


def buy_gate(
    df: pd.DataFrame,
    *,
    max_trailing_return: float = 0.05,
    min_persistence: int = 3,
    min_holders: int = 5,
) -> pd.Series:
    """
    Literal reading of the buy thesis, as a conjunction rather than a blend.

    Price down, flat or only slightly up over the trailing six months, while the
    common-fund panel has been net-buying for most of the lookback, is net-buying
    this quarter, and more funds added than trimmed.
    """
    return (
        (df["Ret_6m"] <= max_trailing_return)
        & (df["Acc_Persistence"] >= min_persistence)
        & (df["Breadth"] > 0)
        & (df["Share_Chg_Pct"] > 0)
        & (df["Holders"] >= min_holders)
    ).fillna(False)


def sell_gate(
    df: pd.DataFrame,
    *,
    min_trailing_return: float = 0.15,
    max_persistence: int = 1,
    min_holders: int = 5,
) -> pd.Series:
    """
    Mirror of ``buy_gate``: a name that has run while the funds distribute into it.
    """
    return (
        (df["Ret_6m"] >= min_trailing_return)
        & (df["Acc_Persistence"] <= max_persistence)
        & (df["Breadth"] < 0)
        & (df["Share_Chg_Pct"] < 0)
        & (df["Holders"] >= min_holders)
    ).fillna(False)


def max_drawdown(returns: Sequence[float]) -> float:
    """
    Worst peak-to-trough fall of the equity curve implied by period returns.

    Returned as a non-positive number; a curve that never falls scores 0.0.
    """
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for period_return in returns:
        equity *= 1.0 + float(period_return)
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def hit_rate(values: Sequence[float]) -> float:
    """
    Share of observations that are strictly positive (NaN when there are none).
    """
    series = pd.Series(list(values), dtype="float64").dropna()
    if series.empty:
        return float("nan")
    return float((series > 0).mean())


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    """
    Spearman rank correlation between a signal and its forward outcome.

    The information coefficient is the properly powered test here: it uses every
    stock-quarter rather than the handful of portfolio windows the sample allows.
    """
    paired = pd.DataFrame(
        {"x": pd.Series(x).astype(float), "y": pd.Series(y).astype(float)}
    ).dropna()
    if len(paired) < 3 or paired["x"].nunique() < 2 or paired["y"].nunique() < 2:
        return float("nan")
    # Pearson correlation of average ranks — the definition of Spearman's rho, and
    # pandas' own "spearman" method needs SciPy, which this project does not ship.
    return float(paired["x"].rank().corr(paired["y"].rank()))


def summarize_windows(
    returns: Sequence[float],
    benchmark: Sequence[float] | None = None,
    *,
    periods_per_year: int = 4,
) -> dict:
    """
    Performance statistics for a sequence of holding-period returns.

    ``win_rate`` is the share of windows that made money and
    ``beat_benchmark_rate`` the share that beat the benchmark over the same
    window; ``max_drawdown`` is measured on the quarterly curve, so it is a lower
    bound on what an intra-quarter mark would have shown.
    """
    series = pd.Series(list(returns), dtype="float64").dropna()
    n = len(series)
    if n == 0:
        return {"n_windows": 0}

    cumulative = float(np.prod(1.0 + series.to_numpy(dtype=float))) - 1.0
    years = n / periods_per_year
    volatility = (
        float(np.std(series.to_numpy(dtype=float), ddof=1) * np.sqrt(periods_per_year))
        if n > 1
        else float("nan")
    )
    summary = {
        "n_windows": n,
        "cum_return": cumulative,
        "cagr": (1.0 + cumulative) ** (1.0 / years) - 1.0 if cumulative > -1.0 else -1.0,
        "avg_window": float(series.mean()),
        "median_window": float(series.median()),
        "best_window": float(series.max()),
        "worst_window": float(series.min()),
        "win_rate": hit_rate(series.tolist()),
        "volatility": volatility,
        "sharpe": float(series.mean() * periods_per_year / volatility)
        if volatility
        else float("nan"),
        "max_drawdown": max_drawdown(series.tolist()),
    }

    if benchmark is not None:
        bench = pd.Series(list(benchmark), dtype="float64").reindex(series.index)
        excess = (series - bench).dropna()
        summary["beat_benchmark_rate"] = hit_rate(excess.tolist())
        summary["avg_excess"] = float(excess.mean()) if len(excess) else float("nan")
        tracking_error = (
            float(np.std(excess.to_numpy(dtype=float), ddof=1) * np.sqrt(periods_per_year))
            if len(excess) > 1
            else float("nan")
        )
        summary["tracking_error"] = tracking_error
        summary["information_ratio"] = (
            float(excess.mean() * periods_per_year / tracking_error)
            if tracking_error
            else float("nan")
        )
        summary["bench_cum_return"] = (
            float(np.prod(1.0 + bench.dropna().to_numpy(dtype=float))) - 1.0
        )
    return summary
