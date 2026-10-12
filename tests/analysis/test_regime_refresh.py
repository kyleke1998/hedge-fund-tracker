import unittest

import pandas as pd

from app.analysis.regime_refresh import Series, accept_series, refresh_series


def corr_frame(dates: list[str], stocks: list[int]) -> pd.DataFrame:
    """
    An existing correlation CSV as the loader returns it.
    """
    return pd.DataFrame({"date": dates, "correlation": [0.2] * len(dates), "stocks": stocks})


def corr_rows(dates: list[str], stocks: list[int]) -> list[dict]:
    """
    Freshly built correlation rows.
    """
    return [
        {"date": d, "correlation": 0.2, "stocks": n} for d, n in zip(dates, stocks, strict=True)
    ]


class TestAcceptSeries(unittest.TestCase):
    def test_accepts_a_series_that_extends_the_existing_one(self):
        """
        The normal daily run appends a session.
        """
        existing = corr_frame(["2026-08-06", "2026-08-07"], [98, 98])
        fresh = corr_rows(["2026-08-06", "2026-08-07", "2026-08-10"], [98, 98, 98])
        ok, _ = accept_series(existing, fresh, date_field="date", count_field="stocks")
        self.assertTrue(ok)

    def test_accepts_the_first_generation(self):
        """
        With nothing on disk there is nothing to protect.
        """
        ok, _ = accept_series(pd.DataFrame(), corr_rows(["2026-08-06"], [98]), date_field="date")
        self.assertTrue(ok)

    def test_refuses_an_empty_build(self):
        """
        The builders return [] when the price source is down; writing that
        would wipe the page.
        """
        ok, reason = accept_series(corr_frame(["2026-08-07"], [98]), [], date_field="date")
        self.assertFalse(ok)
        self.assertIn("empty", reason)

    def test_refuses_a_series_that_ends_earlier(self):
        """
        A stale source must not roll the published series back.
        """
        existing = corr_frame(["2026-08-06", "2026-08-07"], [98, 98])
        fresh = corr_rows(["2026-08-05", "2026-08-06"], [98, 98])
        ok, _ = accept_series(existing, fresh, date_field="date")
        self.assertFalse(ok)

    def test_refuses_a_shorter_history(self):
        """
        Losing the early years means the source returned a truncated history.
        """
        existing = corr_frame(["2026-08-05", "2026-08-06", "2026-08-07"], [98, 98, 98])
        fresh = corr_rows(["2026-08-07", "2026-08-10"], [98, 98])
        ok, _ = accept_series(existing, fresh, date_field="date")
        self.assertFalse(ok)

    def test_refuses_a_reading_built_from_a_thinned_basket(self):
        """
        A partial rate-limit drops names silently; a correlation over half the
        basket is not comparable to the published one.
        """
        existing = corr_frame(["2026-08-07"], [98])
        fresh = corr_rows(["2026-08-07", "2026-08-10"], [60, 60])
        ok, reason = accept_series(existing, fresh, date_field="date", count_field="stocks")
        self.assertFalse(ok)
        self.assertIn("stocks", reason)

    def test_tolerates_a_constituent_or_two_dropping_out(self):
        """
        Delistings are real; only a large drop signals a broken fetch.
        """
        existing = corr_frame(["2026-08-07"], [98])
        fresh = corr_rows(["2026-08-07", "2026-08-10"], [97, 96])
        ok, _ = accept_series(existing, fresh, date_field="date", count_field="stocks")
        self.assertTrue(ok)

    def test_undated_series_only_needs_to_be_non_empty(self):
        """
        regime.csv is long-format by quarter and computed locally, so only the
        empty case is suspicious.
        """
        existing = pd.DataFrame({"kind": ["metric"] * 3})
        ok, _ = accept_series(existing, [{"kind": "metric"}])
        self.assertTrue(ok)


class TestRefreshSeries(unittest.TestCase):
    def test_writes_accepted_series_and_skips_refused_ones(self):
        """
        One broken source must not block the others from updating.
        """
        written: dict[str, list[dict]] = {}

        def writer(name: str):
            """
            Record what would have been written.
            """
            return lambda rows: written.__setitem__(name, rows)

        good = Series(
            name="momentum",
            build=lambda: [{"date": "2026-08-10", "roc": 1.0}],
            load=lambda: pd.DataFrame({"date": ["2026-08-07"], "roc": [0.5]}),
            write=writer("momentum"),
            date_field="date",
        )
        broken = Series(
            name="correlation",
            build=lambda: [],
            load=lambda: corr_frame(["2026-08-07"], [98]),
            write=writer("correlation"),
            date_field="date",
            count_field="stocks",
        )
        refused = refresh_series([good, broken])
        self.assertEqual(refused, ["correlation"])
        self.assertEqual(set(written), {"momentum"})

    def test_a_crashing_builder_is_refused_not_fatal(self):
        """
        A network exception in one builder leaves its CSV untouched.
        """

        def boom() -> list[dict]:
            """
            Simulate the price source raising.
            """
            raise ConnectionError("yahoo down")

        written: list[str] = []
        series = Series(
            name="momentum",
            build=boom,
            load=pd.DataFrame,
            write=lambda rows: written.append("momentum"),
        )
        self.assertEqual(refresh_series([series]), ["momentum"])
        self.assertEqual(written, [])


if __name__ == "__main__":
    unittest.main()
