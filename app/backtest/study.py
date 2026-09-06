"""
Portfolio mechanics shared by the divergence study.

Price lookups, holding-period returns, daily mark-to-market curves and quantile
tables — the pieces that turn a cross-sectional score into a track record. Kept
apart from ``divergence`` so the feature definitions stay independent of how a
screen is traded.
"""

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd

# Smallest number of observations that can support a quantile breakdown.
MIN_QUANTILE_OBSERVATIONS = 4


def asof_prices(close: pd.DataFrame, day: date) -> pd.Series:
    """
    Last close at or before ``day`` for every ticker.

    Entry and exit dates are calendar dates (a quarter-end plus the filing lag),
    so they routinely land on weekends and holidays; resolving backwards keeps
    the trade at a price that existed.
    """
    window = close.loc[close.index <= pd.Timestamp(day)]
    if window.empty:
        return pd.Series(np.nan, index=close.columns, dtype="float64")
    return window.ffill().iloc[-1].astype("float64")


def forward_returns(close: pd.DataFrame, entry: date, exit_date: date) -> pd.Series:
    """
    Holding-period return per ticker between two dates, NaN where either leg is missing.
    """
    entry_px = asof_prices(close, entry)
    exit_px = asof_prices(close, exit_date)
    return (exit_px / entry_px.where(entry_px > 0) - 1.0).astype("float64")


def daily_equity(
    close: pd.DataFrame, weights: Mapping[str, float], entry: date, exit_date: date
) -> pd.Series:
    """
    Daily buy-and-hold curve for a weighted basket, starting at 1.0 on the entry bar.

    Weights are renormalised over the names that actually have an entry price, so
    an unpriceable constituent dilutes nothing. Positions then drift with the
    market until the next rebalance, which is what a quarterly screen really does.
    """
    entry_px = asof_prices(close, entry)
    priced = {
        ticker: weight
        for ticker, weight in weights.items()
        if ticker in close.columns and pd.notna(entry_px.get(ticker)) and entry_px[ticker] > 0
    }
    total = sum(priced.values())
    if not priced or total <= 0:
        return pd.Series(dtype="float64")

    window = close.loc[
        (close.index >= pd.Timestamp(entry)) & (close.index <= pd.Timestamp(exit_date)),
        list(priced),
    ]
    if window.empty:
        return pd.Series(dtype="float64")

    normalized = window.ffill().div(pd.Series({t: entry_px[t] for t in priced}))
    shares = pd.Series({t: w / total for t, w in priced.items()})
    return normalized.mul(shares).sum(axis=1, min_count=1)


def quantile_table(
    df: pd.DataFrame, score_column: str, return_column: str, *, bins: int = 5
) -> pd.DataFrame:
    """
    Mean/median forward return and hit rate per score bucket, worst bucket first.

    A graded signal shows a monotone progression across buckets; a signal driven
    by a few names at one extreme does not, which is why the whole table is
    reported rather than just the top bucket.
    """
    paired = df[[score_column, return_column]].dropna()
    if len(paired) < max(bins, MIN_QUANTILE_OBSERVATIONS):
        return pd.DataFrame(columns=["bucket", "n", "mean_return", "median_return", "hit_rate"])

    labels = pd.qcut(paired[score_column].rank(method="first"), bins, labels=False)
    grouped = paired.groupby(labels)[return_column]
    sizes = grouped.size()
    return pd.DataFrame(
        {
            "bucket": np.asarray(sizes.index, dtype=int) + 1,
            "n": sizes.to_numpy(),
            "mean_return": grouped.mean().to_numpy(),
            "median_return": grouped.median().to_numpy(),
            "hit_rate": grouped.apply(lambda s: float((s > 0).mean())).to_numpy(),
        }
    ).reset_index(drop=True)


def double_sort_table(
    df: pd.DataFrame, row_column: str, column_column: str, value_column: str, *, bins: int = 5
) -> dict[str, pd.DataFrame]:
    """
    Independent double sort: mean outcome and cell count on a ``bins`` x ``bins`` grid.

    A blended score cannot say whether accumulation pays *because of* or *in spite
    of* the price trend. Sorting on both at once can: the thesis predicts the
    strongest cell sits where accumulation is highest and price strength lowest.
    Returns ``{"mean_return": frame, "count": frame}``, empty when the sample
    cannot fill the grid.
    """
    paired = df[[row_column, column_column, value_column]].dropna()
    if len(paired) < bins * bins:
        return {}

    row_bin = pd.qcut(paired[row_column].rank(method="first"), bins, labels=False)
    col_bin = pd.qcut(paired[column_column].rank(method="first"), bins, labels=False)
    grouped = paired.groupby([row_bin, col_bin])[value_column]
    labels = list(range(1, bins + 1))
    return {
        "mean_return": grouped.mean()
        .unstack()
        .reindex(index=range(bins), columns=range(bins))
        .set_axis(labels)
        .set_axis(labels, axis=1),
        "count": grouped.size()
        .unstack()
        .reindex(index=range(bins), columns=range(bins))
        .fillna(0)
        .astype(int)
        .set_axis(labels)
        .set_axis(labels, axis=1),
    }


def neutralize(df: pd.DataFrame, columns: Sequence[str], group_column: str) -> pd.DataFrame:
    """
    Express the given columns relative to their group mean, in place of the raw value.

    Sector-demeaning is what separates stock selection from a sector bet: a screen
    that only found last year's best sector scores zero once its sector's own
    return is removed. Rows with no group are pooled into one residual bucket
    rather than dropped, so the sample size is unchanged.
    """
    result = df.copy()
    groups = result[group_column].fillna("__ungrouped__")
    for column in columns:
        result[column] = result[column] - result.groupby(groups)[column].transform("mean")
    return result
