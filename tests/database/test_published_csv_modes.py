import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from app.backtest.report import write_rows
from app.database.correlation import write_correlation_rows
from app.database.momentum import write_momentum_rows
from app.database.regime import write_regime_rows


class TestPublishedCsvModes(unittest.TestCase):
    """
    The Docker app reads database/ as another UID, so a writer that leaves its
    mkstemp 0600 on the file makes the endpoint serving it answer 500.
    """

    WRITERS = {
        "regime": write_regime_rows,
        "momentum": write_momentum_rows,
        "correlation": write_correlation_rows,
        "performance": write_rows,
    }

    def setUp(self):
        """
        Create a temp database folder.
        """
        self.tmp = Path(tempfile.mkdtemp(prefix="hft_modes_"))

    def tearDown(self):
        """
        Remove the temp folder.
        """
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_file_is_world_readable(self):
        """
        A first write must not leave the temp file's 0600 behind.
        """
        for name, write in self.WRITERS.items():
            with self.subTest(writer=name):
                target = self.tmp / f"{name}_new.csv"
                write([], target)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_rewrite_keeps_the_existing_mode(self):
        """
        A regeneration over a published file keeps its permissions.
        """
        for name, write in self.WRITERS.items():
            with self.subTest(writer=name):
                target = self.tmp / f"{name}_existing.csv"
                target.write_text("", encoding="utf-8")
                target.chmod(0o664)
                write([], target)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o664)


if __name__ == "__main__":
    unittest.main()
