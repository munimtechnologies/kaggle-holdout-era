"""Tests for he2_excess.py (stdlib only). MV2 of Munim, Inc. (made with AI agents). Apache-2.0."""
import random
import unittest

import he2_excess as h2


def line(n=200, is_max=True, slope=0.1, top_extra=0.0, flat_mid=False):
    """Oriented public x = 1000 - i, gap d = slope * (x - 800) (+ top_extra for the top 10)."""
    teams = []
    for i in range(1, n + 1):
        x = 1000.0 - i
        if flat_mid and i > 50:
            x = 900.0
            d = 0.1
        else:
            d = slope * (x - 800.0) + (top_extra if i <= 10 else 0.0)
        pub, priv = x, x - d
        if not is_max:
            pub, priv = -pub, -priv
        teams.append({"pub_rank": i, "pub": pub, "priv": priv, "kernel": i % 4 == 0})
    return teams


class TestExcess(unittest.TestCase):
    def test_straight_line_gives_zero_excess(self):
        m = h2.excess_metrics(line(), True)
        self.assertIsNone(m["excluded"])
        self.assertEqual(m["n"], 200)
        self.assertEqual(m["k"], 10)
        self.assertEqual((m["n_b1"], m["n_b2"]), (50, 50))
        self.assertAlmostEqual(m["scale"], 44.55, places=9)
        self.assertAlmostEqual(m["b_raw"], 0.1, places=9)
        self.assertAlmostEqual(m["O"], 7.0 / 44.55, places=9)
        self.assertAlmostEqual(m["X"], 0.0, places=9)
        self.assertAlmostEqual(m["kernel_share"], 0.25)

    def test_top_excess(self):
        m = h2.excess_metrics(line(top_extra=0.5), True)
        self.assertAlmostEqual(m["scale"], 44.55, places=9)
        self.assertAlmostEqual(m["O"], 7.5 / 44.55, places=9)
        self.assertAlmostEqual(m["X"], 0.5 / 44.55, places=9)

    def test_lower_is_better(self):
        m = h2.excess_metrics(line(is_max=False, top_extra=0.5), False)
        self.assertIsNone(m["excluded"])
        self.assertAlmostEqual(m["X"], 0.5 / 44.55, places=9)
        self.assertAlmostEqual(m["O"], 7.5 / 44.55, places=9)

    def test_clamp_high(self):
        m = h2.excess_metrics(line(slope=2.0), True)
        self.assertAlmostEqual(m["scale"], 49.5, places=9)
        self.assertAlmostEqual(m["b_raw"], 2.0, places=9)
        self.assertEqual(m["b"], 1.0)
        self.assertAlmostEqual(m["O"], 140.0 / 49.5, places=9)
        self.assertAlmostEqual(m["X"], 70.0 / 49.5, places=9)
        self.assertAlmostEqual(m["X_raw"], 0.0, places=9)

    def test_clamp_low(self):
        m = h2.excess_metrics(line(slope=-0.1), True)
        self.assertEqual(m["b"], 0.0)
        self.assertAlmostEqual(m["X"], m["O"], places=12)

    def test_exclusions(self):
        self.assertEqual(h2.excess_metrics(line(n=50), True)["excluded"], "few_teams")
        self.assertEqual(h2.excess_metrics(line(slope=0.0), True)["excluded"], "no_separate_private")
        self.assertEqual(h2.excess_metrics(line(flat_mid=True), True)["excluded"], "flat_mid")


class TestStats(unittest.TestCase):
    def test_sign_test(self):
        self.assertAlmostEqual(h2.sign_test([1, 2, 3, 4, 5])["p_one_sided"], 1.0 / 32)
        self.assertAlmostEqual(h2.sign_test([1, -1, 0])["p_one_sided"], 0.75)
        self.assertAlmostEqual(h2.sign_test([-1, -2])["p_one_sided"], 1.0)
        self.assertIsNone(h2.sign_test([])["p_one_sided"])

    def test_quantile(self):
        self.assertAlmostEqual(h2.quantile(list(range(1, 101)), 0.75) - h2.quantile(list(range(1, 101)), 0.25), 49.5)
        self.assertEqual(h2.median([4, 1, 2, 3]), 2.5)

    def test_sim_leaderboard_shape(self):
        lb = h2.sim_leaderboard(random.Random(1), 50, "skewed", 0.3, 5, 2)
        self.assertEqual(len(lb), 50)
        self.assertEqual([t["pub_rank"] for t in lb], list(range(1, 51)))
        pubs = [t["pub"] for t in lb]
        self.assertEqual(pubs, sorted(pubs, reverse=True))

    def test_calibrate_runs(self):
        st = ({"name": "g", "skill": "normal", "sigma1": 0.6, "m_top": 1, "m_rest": 1, "gate": True},)
        res = h2.calibrate(reps=3, n_teams=300, seed=7, settings=st)
        self.assertEqual(res["settings"]["g"]["n"], 3)
        self.assertIn("C0_pass", res)


if __name__ == "__main__":
    unittest.main()
