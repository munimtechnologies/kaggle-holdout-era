"""Tests for holdout_era.py (stdlib only). MV2 of Munim, Inc. (made with AI agents). Apache-2.0."""
import datetime as dt
import unittest

import holdout_era as he


def synth(n=200, is_max=True, top_gap=0.6, rest_gap=0.1):
    teams = []
    for i in range(1, n + 1):
        if is_max:
            priv = 1000.0 - i
            pub = priv + (top_gap if i <= 10 else rest_gap)
        else:
            priv = float(i)
            pub = priv - (top_gap if i <= 10 else rest_gap)
        teams.append({"pub_rank": i, "priv_rank": i, "pub": pub, "priv": priv, "kernel": i % 2 == 0})
    return teams


class TestHelpers(unittest.TestCase):
    def test_parse_date(self):
        self.assertEqual(he.parse_date("03/15/2021 23:59:00"), dt.date(2021, 3, 15))
        self.assertEqual(he.parse_date("2025-01-02 00:00:00"), dt.date(2025, 1, 2))
        self.assertEqual(he.parse_date("2025-01-02T00:00:00"), dt.date(2025, 1, 2))
        self.assertEqual(he.parse_date(dt.datetime(2024, 5, 6, 7, 8)), dt.date(2024, 5, 6))
        self.assertIsNone(he.parse_date(""))
        self.assertIsNone(he.parse_date(None))
        self.assertIsNone(he.parse_date("garbage"))
        self.assertIsNone(he.parse_date("nan"))

    def test_era(self):
        self.assertIsNone(he.era_of(dt.date(2019, 12, 31)))
        self.assertEqual(he.era_of(dt.date(2020, 1, 1)), "E0")
        self.assertEqual(he.era_of(dt.date(2022, 11, 30)), "E0")
        self.assertEqual(he.era_of(dt.date(2022, 12, 1)), "E1")
        self.assertEqual(he.era_of(dt.date(2024, 12, 31)), "E1")
        self.assertEqual(he.era_of(dt.date(2025, 1, 1)), "E2")
        self.assertEqual(he.era_of(dt.date(2026, 11, 5)), "E2")
        self.assertIsNone(he.era_of(dt.date(2026, 11, 6)))
        self.assertIsNone(he.era_of(None))

    def test_quantile_median(self):
        xs = list(range(900, 1000))
        self.assertAlmostEqual(he.quantile(xs, 0.75) - he.quantile(xs, 0.25), 49.5)
        self.assertEqual(he.median([3, 1, 2]), 2)
        self.assertEqual(he.median([4, 1, 2, 3]), 2.5)

    def test_spearman(self):
        self.assertAlmostEqual(he.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(he.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(he.avg_ranks([5, 5, 1]), [2.5, 2.5, 1.0])
        self.assertIsNone(he.spearman([1, 1, 1], [1, 2, 3]))

    def test_kernel_and_timeshift(self):
        self.assertTrue(he.is_kernel("12345"))
        self.assertFalse(he.is_kernel(""))
        self.assertFalse(he.is_kernel(None))
        self.assertFalse(he.is_kernel("nan"))
        self.assertTrue(he.is_time_shifted("After that a Forecasting Phase follows."))
        self.assertFalse(he.is_time_shifted("A standard image classification task."))
        self.assertFalse(he.is_time_shifted(None))

    def test_pick_public_best(self):
        rows = [
            (1, 1, False, False, 0.5, 0.4, ""),
            (2, 1, False, True, 0.7, 0.3, "k9"),
            (3, 1, True, False, 0.9, 0.9, ""),       # after deadline: ignored
            (4, 2, False, False, 0.2, 0.25, ""),
            (5, 2, False, False, 0.1, None, ""),     # lower is better for team 2
            (6, 3, False, False, 0.1, 0.1, ""),      # team 3 has no sign: ignored
            (7, 1, False, False, float("nan"), 0.1, ""),
        ]
        best = he.pick_public_best(rows, {1: 1.0, 2: -1.0})
        self.assertEqual(best[1], (0.7, 0.3, True))
        self.assertEqual(best[2], (0.1, None, False))
        self.assertNotIn(3, best)

    def test_pick_public_best_tie(self):
        rows = [(10, 1, False, False, 0.5, 0.1, ""), (11, 1, False, True, 0.5, 0.2, ""),
                (12, 1, False, True, 0.5, 0.3, "")]
        self.assertEqual(he.pick_public_best(rows, {1: 1.0})[1], (0.5, 0.2, False))


class TestCompMetrics(unittest.TestCase):
    def test_max(self):
        m = he.comp_metrics(synth(is_max=True), True)
        self.assertIsNone(m["excluded"])
        self.assertEqual(m["n"], 200)
        self.assertEqual(m["k"], 10)
        self.assertEqual(m["n_ref"], 50)
        self.assertAlmostEqual(m["scale"], 49.5)
        self.assertAlmostEqual(m["O"], 0.5 / 49.5, places=9)
        self.assertAlmostEqual(m["kernel_share"], 0.5)
        self.assertEqual(m["ret10"], 1.0)
        self.assertAlmostEqual(m["rho100"], 1.0)

    def test_min(self):
        m = he.comp_metrics(synth(is_max=False), False)
        self.assertIsNone(m["excluded"])
        self.assertAlmostEqual(m["O"], 0.5 / 49.5, places=9)

    def test_no_selection_effect(self):
        m = he.comp_metrics(synth(top_gap=0.1, rest_gap=0.1), True)
        self.assertAlmostEqual(m["O"], 0.0, places=9)

    def test_k_grows(self):
        m = he.comp_metrics(synth(n=1000), True)
        self.assertEqual(m["k"], 20)

    def test_retention(self):
        t = synth()
        t[9]["priv_rank"], t[10]["priv_rank"] = 11, 10
        m = he.comp_metrics(t, True)
        self.assertAlmostEqual(m["ret10"], 0.9)

    def test_exclusions(self):
        self.assertEqual(he.comp_metrics(synth(n=50), True)["excluded"], "few_teams")
        same = synth(top_gap=0.0, rest_gap=0.0)
        self.assertEqual(he.comp_metrics(same, True)["excluded"], "no_separate_private")
        flat = synth()
        for x in flat:
            x["priv"] = 1.0
            x["pub"] = 1.5
        self.assertEqual(he.comp_metrics(flat, True)["excluded"], "zero_scale")


class TestStats(unittest.TestCase):
    def test_perm_clear_difference(self):
        a = [i * 0.01 for i in range(20)]
        b = [1.0 + i * 0.01 for i in range(20)]
        obs, p = he.perm_test_median_diff(a, b, n_perm=2000)
        self.assertAlmostEqual(obs, 1.0)
        self.assertLess(p, 0.01)

    def test_perm_null(self):
        a = [1.0, 2.0, 3.0] * 5
        obs, p = he.perm_test_median_diff(a, list(a), n_perm=2000)
        self.assertEqual(obs, 0.0)
        self.assertGreater(p, 0.3)

    def test_perm_stratified(self):
        a = [0.0] * 10 + [5.0] * 10
        b = [1.0] * 10 + [6.0] * 10
        sa = ["x"] * 10 + ["y"] * 10
        obs, p = he.perm_test_median_diff(a, b, n_perm=1000, strata_a=sa, strata_b=list(sa))
        self.assertEqual(obs, 1.0)
        self.assertLess(p, 0.01)

    def test_boot_ci(self):
        lo, hi = he.boot_ci_median_diff([0.0] * 10, [1.0] * 10, n_boot=200)
        self.assertEqual((lo, hi), (1.0, 1.0))

    def test_manifest_sha(self):
        m1 = {"files": [{"path": "data/nodes_team.parquet", "sha256": "abc"}]}
        m2 = {"files": {"data/nodes_team.parquet": {"bytes": 1, "sha256": "def"}}}
        self.assertEqual(he.find_manifest_sha(m1, "nodes_team.parquet"), "abc")
        self.assertEqual(he.find_manifest_sha(m2, "nodes_team.parquet"), "def")
        self.assertIsNone(he.find_manifest_sha({"x": 1}, "nodes_team.parquet"))


if __name__ == "__main__":
    unittest.main()
