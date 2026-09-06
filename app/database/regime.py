"""
Persistence for database/regime.csv — the market-regime signal table.

Long format, one row per (kind, quarter, key): `kind` is "metric" for the
per-quarter scalars, "mega" for the mega-cap basket series, and "sector" for
sector weights. One file carries all three so the page makes a single fetch.
"""

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd

import app.database as _db
from app.database import REGIME_FILE
from app.utils.logger import get_logger

logger = get_logger(__name__)

CSV_FIELDS = ["kind", "quarter", "key", "value"]


def write_regime_rows(rows: list[dict], path: Path | str | None = None) -> Path:
    """
    Atomically write regime rows to a QUOTE_ALL CSV (header even when empty).
    """
    target = Path(path) if path else Path(_db.DB_FOLDER) / REGIME_FILE
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
                writer.writerow({field: row.get(field) for field in CSV_FIELDS})
        tmp_path.replace(target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def load_regime_rows(path: Path | str | None = None) -> pd.DataFrame:
    """
    Read the regime CSV, returning an empty frame when it has not been generated.
    """
    target = Path(path) if path else Path(_db.DB_FOLDER) / REGIME_FILE
    if not target.exists():
        logger.warning("Regime signals not generated yet; run 'pipenv run gen-regime'")
        return pd.DataFrame(columns=CSV_FIELDS)
    return pd.read_csv(target)
