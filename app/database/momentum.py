"""
Persistence for database/momentum.csv — SPY three-month rate of change.

One row per session: the date and the percent change in SPY over the trailing
63 sessions. Written offline by `pipenv run gen-momentum` and bundled into the
static build, because the browser has neither the price history nor the budget
to roll the window itself.
"""

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd

import app.database as _db
from app.database import MOMENTUM_FILE
from app.utils.logger import get_logger

logger = get_logger(__name__)

CSV_FIELDS = ["date", "roc"]

# Four decimals is finer than the chart can show and keeps a regeneration from
# rewriting every line over float noise in the last digits.
_PRECISION = 4


def _round(value: object) -> object:
    """
    Round float values so repeated regeneration produces a stable diff.
    """
    return round(value, _PRECISION) if isinstance(value, float) else value


def write_momentum_rows(rows: list[dict], path: Path | str | None = None) -> Path:
    """
    Atomically write momentum rows to a QUOTE_ALL CSV (header even when empty).
    """
    target = Path(path) if path else Path(_db.DB_FOLDER) / MOMENTUM_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _round(row.get(field)) for field in CSV_FIELDS})
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def load_momentum_rows(path: Path | str | None = None) -> pd.DataFrame:
    """
    Read the momentum CSV, returning an empty frame when it has not been generated.
    """
    target = Path(path) if path else Path(_db.DB_FOLDER) / MOMENTUM_FILE
    if not target.exists():
        logger.warning("Momentum series not generated yet; run 'pipenv run gen-momentum'")
        return pd.DataFrame(columns=CSV_FIELDS)
    return pd.read_csv(target)
