"""
Tests for the per-fund-quarter 13F filing-date ledger (database/filing_dates.csv).

The ledger records when each fund's filing for a reporting period was actually
published by EDGAR, which is what makes an amendment look-ahead gate possible:
a comparison rebuilt from a 13F-HR/A published months after the quarter's entry
date is not information the strategy could have acted on.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.database import (
    FILING_DATES_FILE,
    load_filing_dates,
    record_filing_date,
    save_comparison,
)


class TestFilingDates(unittest.TestCase):
    def setUp(self):
        """
        Redirect the database folder at a temporary directory.
        """
        self.db = tempfile.mkdtemp(prefix="hft_filing_dates_")
        self.patcher = patch("app.database.DB_FOLDER", self.db)
        self.patcher.start()
        self.path = Path(self.db) / FILING_DATES_FILE

    def tearDown(self):
        """
        Stop the patch and remove the temporary directory.
        """
        self.patcher.stop()
        shutil.rmtree(self.db, ignore_errors=True)

    def test_record_creates_ledger_with_header(self):
        """
        The first record on a fresh database produces a parseable CSV.
        """
        record_filing_date("2025Q1", "Fund A", "2025-05-14")

        df = load_filing_dates()
        self.assertEqual(list(df.columns), ["Quarter", "Fund", "Filing_Date"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Filing_Date"], "2025-05-14")

    def test_record_upserts_rather_than_appending_duplicates(self):
        """
        Re-recording the same (quarter, fund) overwrites the date — a regenerate
        run that picks up a newer amendment must not leave two rows behind.
        """
        record_filing_date("2025Q1", "Fund A", "2025-05-14")
        record_filing_date("2025Q1", "Fund A", "2025-09-02")

        df = load_filing_dates()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Filing_Date"], "2025-09-02")

    def test_distinct_quarters_and_funds_coexist(self):
        """
        Rows are keyed by (quarter, fund), not by either alone.
        """
        record_filing_date("2025Q1", "Fund A", "2025-05-14")
        record_filing_date("2025Q2", "Fund A", "2025-08-13")
        record_filing_date("2025Q1", "Fund B", "2025-05-15")

        df = load_filing_dates()
        self.assertEqual(len(df), 3)

    def test_ledger_is_sorted_for_stable_diffs(self):
        """
        Rows are ordered by quarter then fund so the committed file does not
        churn on every regeneration.
        """
        record_filing_date("2025Q2", "Zeta", "2025-08-13")
        record_filing_date("2025Q1", "Alpha", "2025-05-14")
        record_filing_date("2025Q1", "Beta", "2025-05-15")

        df = load_filing_dates()
        self.assertEqual(
            list(zip(df["Quarter"], df["Fund"], strict=True)),
            [("2025Q1", "Alpha"), ("2025Q1", "Beta"), ("2025Q2", "Zeta")],
        )

    def test_load_returns_empty_frame_when_absent(self):
        """
        A database with no ledger yet loads as an empty typed frame, not an error.
        """
        df = load_filing_dates()
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), ["Quarter", "Fund", "Filing_Date"])

    def test_save_comparison_records_the_filing_date(self):
        """
        The single write path: saving a comparison with its publication date
        also lands the row in the ledger, keyed by the derived quarter.
        """
        save_comparison(
            pd.DataFrame({"CUSIP": ["123"], "Company": ["A"]}),
            "2025-03-31",
            "Fund A",
            filing_date="2025-05-14",
        )

        df = load_filing_dates()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Quarter"], "2025Q1")
        self.assertEqual(df.iloc[0]["Fund"], "Fund A")
        self.assertEqual(df.iloc[0]["Filing_Date"], "2025-05-14")

    def test_save_comparison_without_filing_date_writes_no_ledger(self):
        """
        The parameter stays optional so existing callers and tests are unaffected.
        """
        save_comparison(pd.DataFrame({"CUSIP": ["123"]}), "2025-03-31", "Fund A")

        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
