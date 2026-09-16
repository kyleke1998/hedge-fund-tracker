"""
Generate database/correlation.csv — realized average stock correlation.

Rolls a three-month window of daily returns across the QQQ basket and writes one
reading per session. Compute is offline (it needs a decade of daily bars for a
hundred names), so the CSV is bundled into the static GitHub Pages build and the
Regime Signals page only reads it.

Regenerate:
    pipenv run gen-correlation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.correlation import build_correlation_rows  # noqa: E402
from app.database.correlation import write_correlation_rows  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """
    Rebuild the correlation CSV from the basket's daily price history.
    """
    logger.progress("Computing realized average stock correlation...")
    rows = build_correlation_rows()
    path = write_correlation_rows(rows)
    logger.success("Correlation series written to %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    main()
