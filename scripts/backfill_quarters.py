"""Backfill database quarter folders from SEC EDGAR 13F filings.

Usage:
  pipenv run python -X utf8 scripts/backfill_quarters.py --from 2022Q1

Notes:
- Requires network access and a valid SEC_USER_AGENT env var (SEC policy).
- Run from the repo root inside the project's pipenv: `pipenv run python ...`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Iterable

import app.database as _db
from app.database.quarters import save_comparison
from app.patterns import QUARTER_CAPTURE_RE
from app.scraper.sec_scraper import fetch_latest_two_13f_filings
from app.scraper.xml_processor import xml_to_dataframe_13f
from app.analysis.quarterly_report import generate_comparison
from app.utils.strings import parse_quarter, get_quarter_date, get_quarter
from app.utils.logger import get_logger

logger = get_logger(__name__)


def iter_quarters(start: str, end: str) -> Iterable[str]:
    """Yield quarters from start (inclusive) to end (inclusive) in ascending order."""
    y1, q1 = parse_quarter(start)
    y2, q2 = parse_quarter(end)
    quarters = []
    y, q = y1, q1
    while (y < y2) or (y == y2 and q <= q2):
        quarters.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return quarters


def find_filing_for_quarter(cik: str, target_date: str, max_offset: int = 500):
    """Search EDGAR filings for a CIK and return the filing whose reference_date == target_date.

    Returns the filings list where filings[0] matches target_date, or None.
    """
    offset = 0
    while offset <= max_offset:
        filings = fetch_latest_two_13f_filings(cik, offset)
        if not filings:
            return None
        # filings[0] is the page's first filing; check its reference_date
        if filings[0].get("reference_date") == target_date:
            return filings
        # Advance by 1 (the search page is paginated by start param); caller
        # may adjust step size but keep it conservative to avoid skipping.
        offset += 1
    return None


def backfill_quarter_for_fund(fund: dict, quarter: str) -> bool:
    """Attempt to backfill one fund for the given quarter.

    Returns True on success (file written), False otherwise.
    """
    cik = fund.get("CIK")
    if not cik:
        logger.warning("Skipping fund with no CIK: %s", fund)
        return False

    target_date = get_quarter_date(quarter)
    logger.info("Searching %s filings for %s -> %s", fund.get("Fund"), cik, target_date)
    filings = find_filing_for_quarter(cik, target_date)
    if not filings:
        logger.warning("No filing found for %s / %s", fund.get("Fund"), quarter)
        return False

    # Build dataframe for the target filing and its previous filing (if present)
    try:
        df_latest = xml_to_dataframe_13f(filings[0]["xml_content"])
        df_prev = None
        if len(filings) > 1 and filings[1].get("xml_content"):
            df_prev = xml_to_dataframe_13f(filings[1]["xml_content"])
        comparison = generate_comparison(df_latest, df_prev)
        save_comparison(
            comparison,
            filings[0]["reference_date"],
            fund.get("Fund"),
            filing_date=filings[0].get("date"),
        )
        return True
    except Exception as exc:
        logger.error(
            "Error processing %s for %s: %s", fund.get("Fund"), quarter, exc, exc_info=True
        )
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill 13F quarter folders from EDGAR")
    parser.add_argument("--from", dest="start", required=True, help="Start quarter, e.g. 2022Q1")
    parser.add_argument(
        "--to", dest="end", help="End quarter (inclusive). Defaults to current quarter"
    )
    parser.add_argument(
        "--funds",
        nargs="*",
        help="Optional list of fund names to restrict (exact Fund column values)",
    )
    args = parser.parse_args(argv)

    start_q = args.start
    end_q = args.end or datetime.utcnow().strftime("%YQ%q")
    # If end not provided, compute current quarter from today
    if args.end is None:
        now = datetime.utcnow()
        q = (now.month - 1) // 3 + 1
        end_q = f"{now.year}Q{q}"

    # Basic validation of quarter formats
    if not QUARTER_CAPTURE_RE.match(start_q) or not QUARTER_CAPTURE_RE.match(end_q):
        logger.error("Quarter format invalid; use YYYYQn (e.g. 2022Q1)")
        return 2

    quarters = list(iter_quarters(start_q, end_q))
    logger.info("Backfilling quarters: %s", ", ".join(quarters))

    funds = _db.load_hedge_funds()
    if args.funds:
        funds = [f for f in funds if f.get("Fund") in args.funds]

    total = len(funds) * len(quarters)
    logger.info("Processing %d funds × %d quarters = %d tasks", len(funds), len(quarters), total)

    completed = 0
    with __import__("app.scraper.sec_scraper").scraper_session():
        for quarter in quarters:
            for fund in funds:
                try:
                    ok = backfill_quarter_for_fund(fund, quarter)
                    completed += 1
                    logger.info(
                        "Progress: %d/%d (last: %s / %s -> %s)",
                        completed,
                        total,
                        fund.get("Fund"),
                        quarter,
                        "OK" if ok else "SKIP",
                    )
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    return 1

    logger.info("Backfill complete: %d/%d tasks finished", completed, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
