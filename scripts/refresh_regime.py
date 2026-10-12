"""
Refresh every CSV behind the Regime Signals page in one go.

Rebuilds database/regime.csv, correlation.csv and momentum.csv, keeping the
published copy of any series whose rebuild fails or comes back less complete
(see app/analysis/regime_refresh.py). Run on a schedule by
.github/workflows/regime-refresh.yml; exits non-zero when a series was kept so
the run is flagged.

Run:
    pipenv run refresh-regime
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.regime_refresh import refresh_series, regime_page_series  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    """
    Refresh all regime-page series and report which were left untouched.
    """
    refused = refresh_series(regime_page_series())
    if refused:
        logger.warning("Not refreshed: %s", ", ".join(refused))
        return 1
    logger.success("All regime signals refreshed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
