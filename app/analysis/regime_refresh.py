"""
Unattended refresh of every series behind the Regime Signals page.

regime.csv is derived from the local 13F database; correlation.csv and
momentum.csv are rebuilt from live daily bars. In CI there is no bar cache, so a
Yahoo outage or rate limit makes those builders return nothing, a truncated
history or a thinned basket - all without raising. Writing that would silently
wipe or degrade the published chart, so a fresh series only replaces the one on
disk when it is at least as complete; otherwise the old file is kept and the
series is reported as refused.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from app.utils.logger import get_logger, log_safe

logger = get_logger(__name__)

# Delistings shave a name or two off the basket; a fetch that loses more than
# this share of it is broken, not the market.
MAX_COUNT_DROP = 0.10


@dataclass(frozen=True)
class Series:
    """
    One regime-page CSV: how to rebuild it, read the current copy and write it.
    """

    name: str
    build: Callable[[], list[dict]]
    load: Callable[[], pd.DataFrame]
    write: Callable[[list[dict]], object]
    date_field: str | None = None
    count_field: str | None = None


def accept_series(
    existing: pd.DataFrame,
    fresh: list[dict],
    date_field: str | None = None,
    count_field: str | None = None,
) -> tuple[bool, str]:
    """
    Whether a freshly built series may replace the one on disk.

    Args:
        existing: The series currently published (empty on first generation).
        fresh: The rows just built.
        date_field: Column holding the ISO session date, for dated series.
        count_field: Column holding how many names a reading was built from.

    Returns:
        tuple[bool, str]: The verdict and a human-readable reason.
    """
    if not fresh:
        return False, "fresh build is empty"
    if existing.empty:
        return True, "first generation"
    if date_field is None:
        return True, "non-empty rebuild"

    if len(fresh) < len(existing):
        return False, f"history shrank from {len(existing)} to {len(fresh)} rows"
    old_last = str(existing[date_field].iloc[-1])
    new_last = str(fresh[-1][date_field])
    if new_last < old_last:
        return False, f"series ends {new_last}, before the published {old_last}"

    if count_field is not None:
        old_count = float(existing[count_field].iloc[-1])
        new_count = float(fresh[-1][count_field])
        if new_count < old_count * (1 - MAX_COUNT_DROP):
            return False, f"{count_field} fell from {old_count:.0f} to {new_count:.0f}"
    return True, f"extends to {new_last}"


def refresh_series(series: Sequence[Series]) -> list[str]:
    """
    Rebuild each series and write the ones that pass `accept_series`.

    Returns:
        list[str]: Names of the series left untouched because the rebuild
            failed or was refused.
    """
    refused: list[str] = []
    for item in series:
        logger.progress("Refreshing %s...", item.name)
        try:
            fresh = item.build()
        except Exception:
            logger.error("Building %s failed; keeping the published copy", item.name, exc_info=True)
            refused.append(item.name)
            continue
        ok, reason = accept_series(item.load(), fresh, item.date_field, item.count_field)
        if not ok:
            logger.warning("Keeping the published %s: %s", item.name, log_safe(reason))
            refused.append(item.name)
            continue
        item.write(fresh)
        logger.success("%s refreshed (%d rows, %s)", item.name, len(fresh), log_safe(reason))
    return refused


def regime_page_series() -> list[Series]:
    """
    The three CSVs the Regime Signals page reads, wired to their builders.
    """
    from app.analysis.correlation import build_correlation_rows
    from app.analysis.momentum import build_momentum_rows
    from app.analysis.regime import compute_regime_signals
    from app.database.correlation import load_correlation_rows, write_correlation_rows
    from app.database.momentum import load_momentum_rows, write_momentum_rows
    from app.database.regime import load_regime_rows, write_regime_rows

    return [
        Series("regime", compute_regime_signals, load_regime_rows, write_regime_rows),
        Series(
            "correlation",
            build_correlation_rows,
            load_correlation_rows,
            write_correlation_rows,
            date_field="date",
            count_field="stocks",
        ),
        Series(
            "momentum",
            build_momentum_rows,
            load_momentum_rows,
            write_momentum_rows,
            date_field="date",
        ),
    ]
