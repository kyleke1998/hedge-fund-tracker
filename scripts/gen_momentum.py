"""
Generate database/momentum.csv — SPY three-month rate of change.

Rolls a 63-session window across SPY's daily bars and writes one reading per
session. Compute is offline (it needs a decade of daily bars), so the CSV is
bundled into the static GitHub Pages build and the Regime Signals page only
reads it.

Regenerate:
    pipenv run gen-momentum
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.momentum import build_momentum_rows  # noqa: E402
from app.database.momentum import write_momentum_rows  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """
    Rebuild the momentum CSV from SPY's daily price history.
    """
    logger.progress("Computing SPY three-month rate of change...")
    rows = build_momentum_rows()
    path = write_momentum_rows(rows)
    logger.success("Momentum series written to %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    main()
