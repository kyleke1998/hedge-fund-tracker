import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from app.database.momentum import CSV_FIELDS, load_momentum_rows, write_momentum_rows


class TestMomentumCsv(unittest.TestCase):
    def setUp(self):
        """
        Create a temp database folder.
        """
        self.tmp = Path(tempfile.mkdtemp(prefix="hft_mom_"))
        self.path = self.tmp / "momentum.csv"

    def tearDown(self):
        """
        Remove the temp folder.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_the_canonical_header_and_one_row_per_reading(self):
        """
        The CSV is the page's only input, so its shape is part of the contract.
        """
        write_momentum_rows(
            [{"date": "2024-01-02", "roc": 4.691234}, {"date": "2024-01-03", "roc": -1.2}],
            self.path,
        )
        with self.path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()), CSV_FIELDS)
        self.assertEqual(rows[0]["date"], "2024-01-02")
        self.assertEqual(len(rows), 2)

    def test_rounds_the_reading_so_regeneration_is_a_stable_diff(self):
        """
        Full float precision would rewrite every line on any harmless change.
        """
        write_momentum_rows([{"date": "2024-01-02", "roc": 4.69123456}], self.path)
        with self.path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["roc"], "4.6912")

    def test_writes_a_header_even_with_no_readings(self):
        """
        An empty series must still parse as a CSV, not 404 the page.
        """
        write_momentum_rows([], self.path)
        self.assertEqual(self.path.read_text(encoding="utf-8").strip(), '"date","roc"')

    def test_loads_back_what_was_written(self):
        """
        Round-trip through the reader.
        """
        write_momentum_rows(
            [{"date": "2024-01-02", "roc": 4.69}, {"date": "2024-01-03", "roc": -2.1}],
            self.path,
        )
        frame = load_momentum_rows(self.path)
        self.assertEqual(list(frame.columns), CSV_FIELDS)
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(float(frame["roc"].iloc[1]), -2.1)

    def test_an_ungenerated_file_reads_as_empty_rather_than_raising(self):
        """
        The series is optional: a missing file is a not-yet-generated series.
        """
        frame = load_momentum_rows(self.tmp / "missing.csv")
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns), CSV_FIELDS)


if __name__ == "__main__":
    unittest.main()
