import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from app.database.correlation import CSV_FIELDS, load_correlation_rows, write_correlation_rows


class TestCorrelationCsv(unittest.TestCase):
    def setUp(self):
        """
        Create a temp database folder.
        """
        self.tmp = Path(tempfile.mkdtemp(prefix="hft_corr_"))
        self.path = self.tmp / "correlation.csv"

    def tearDown(self):
        """
        Remove the temp folder.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_the_canonical_header_and_one_row_per_reading(self):
        """
        The CSV is the page's only input, so its shape is part of the contract.
        """
        write_correlation_rows(
            [{"date": "2024-01-02", "correlation": 0.312345678, "stocks": 95}], self.path
        )
        with self.path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()), CSV_FIELDS)
        self.assertEqual(rows[0]["date"], "2024-01-02")
        self.assertEqual(rows[0]["stocks"], "95")

    def test_rounds_the_correlation_so_regeneration_is_a_stable_diff(self):
        """
        Full float precision would rewrite every line on any harmless change.
        """
        write_correlation_rows(
            [{"date": "2024-01-02", "correlation": 0.312345678, "stocks": 95}], self.path
        )
        with self.path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["correlation"], "0.3123")

    def test_writes_a_header_even_with_no_readings(self):
        """
        An empty series must still parse as a CSV, not 404 the page.
        """
        write_correlation_rows([], self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8").strip(), '"date","correlation","stocks"')

    def test_loads_back_what_was_written(self):
        """
        Round-trip through the reader.
        """
        write_correlation_rows(
            [
                {"date": "2024-01-02", "correlation": 0.31, "stocks": 95},
                {"date": "2024-01-03", "correlation": 0.29, "stocks": 95},
            ],
            self.path,
        )
        frame = load_correlation_rows(self.path)
        self.assertEqual(list(frame.columns), CSV_FIELDS)
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(float(frame["correlation"].iloc[1]), 0.29)

    def test_an_ungenerated_file_reads_as_empty_rather_than_raising(self):
        """
        The series is optional: a missing file is a not-yet-generated series.
        """
        frame = load_correlation_rows(self.tmp / "missing.csv")
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), CSV_FIELDS)


if __name__ == "__main__":
    unittest.main()
