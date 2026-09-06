"""
Generate database/regime.csv — the market-regime signal table.

Derives position churn, capital recycling, mega-cap crowding and sector weights
from every quarter of the 13F database. Compute is offline (splits have to be
detected across consecutive quarters, which the browser has no cheap way to do),
so the CSV is bundled into the static GitHub Pages build and the Regime Signals
page only reads it.

Regenerate:
    pipenv run gen-regime
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.regime import compute_regime_signals  # noqa: E402
from app.database.regime import write_regime_rows  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """
    Rebuild the regime-signal CSV from the available quarters.
    """
    logger.progress("Computing regime signals across all quarters...")
    rows = compute_regime_signals()
    path = write_regime_rows(rows)
    logger.success("Regime signals written to %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    main()
