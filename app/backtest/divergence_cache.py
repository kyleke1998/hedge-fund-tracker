"""
Filesystem locations for the divergence study's cached market data.

Kept in its own module so the fetch script and the research driver agree on the
paths without either importing the other. The files live under the same
gitignored directory as the backtest price cache: they are large, rebuildable,
and must never reach the repository.
"""

from pathlib import Path

from app.backtest.price_cache import CACHE_DIR

CLOSE_FILE = Path(CACHE_DIR) / "divergence_close.parquet"
VOLUME_FILE = Path(CACHE_DIR) / "divergence_volume.parquet"
SPLITS_FILE = Path(CACHE_DIR) / "divergence_splits.csv"

# Bars start well before the first entry date so the trailing 200-day average and
# 12-month return are defined for the earliest quarter in the study.
HISTORY_START = "2020-12-01"
