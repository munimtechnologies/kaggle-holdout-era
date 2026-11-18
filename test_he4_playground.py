"""Tests for he4_playground.py (stdlib only). MV2 of Munim, Inc. (made with AI agents). Apache-2.0."""
import csv
import datetime as dt
import json
import os
import tempfile
import unittest

import he4_playground as h4

HEADER = ["slug", "deadline", "era", "seg_group", "excluded", "time_shifted", "X", "X_raw", "O"]


def row(slug, deadline, era, x, seg="playground", excluded="", ts="False", o=None):
    return {"slug": slug, "deadline": deadline, "era": era, "seg_group": seg, "excluded": excluded,
            "time_shifted": ts, "X": "" if x is None else repr(x), "X_raw": "" if x is None else repr(x),
            "O": repr(o if o is not None else (x or 0.0) + 0.1)}


def series(n_e0=10, n_e1=10, n_e2=10, x0=0.0, x1=0.5, x2=1.0, jitter=0.001):
    """Monthly episodes in 2021 (E0), 2023 (E1), 2025 (E2); X = era level + jitter * month (distinct
    values, so the median permutation test is not degenerate on ties)."""
    rows = []
    for m in range(1, n_e0 + 1):
        rows.append(row("a%d" % m, dt.date(2021, m, 28).isoformat(), "E0", x0 + jitter * m))
    for m in range(1, n_e1 + 1):
        rows.append(row("b%d" % m, dt.date(2023, m, 28).isoformat(), "E1", x1 + jitter * m))
    for m in range(1, n_e2 + 1):
        rows.append(row("c%d" % m, dt.date(2025, m, 28).isoformat(), "E2", x2 + jitter * m))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestParsing(unittest.TestCase):
    def test_playground_filter(self):
        rows = [row("keep", "2021-03-31", "E0", 0.1),
                row("excl", "2021-03-31", "E0", 0.1, excluded="flat_mid"),
                row("main", "2021-03-31", "E0", 0.1, seg="main"),
                row("nox", "2021-03-31", "E0", None),
                row("nodate", "", "E0", 0.1)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.csv")
            write_csv(p, rows)
            pg = h4.playground_rows(h4.load_rows(p))
        self.assertEqual([r["slug"] for r in pg], ["keep"])
        self.assertAlmostEqual(pg[0]["X"], 0.1)
        self.assertEqual(pg[0]["deadline"], dt.date(2021, 3, 31))
        self.assertFalse(pg[0]["time_shifted"])

    def test_to_float_and_date(self):
        self.assertIsNone(h4.to_float(""))
        self.assertIsNone(h4.to_float("None"))
        self.assertIsNone(h4.to_float("inf"))
        self.assertEqual(h4.to_float("0.25"), 0.25)
        self.assertEqual(h4.parse_date("2025-06-30 23:59:00"), dt.date(2025, 6, 30))
        self.assertIsNone(h4.parse_date("06/30/2025"))


class TestStats(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(h4.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(h4.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(h4.spearman([1, 2, 3], [5, 5, 5]))

    def test_trend_directions(self):
        up = [{"deadline": dt.date(2021, m, 15), "X": 0.1 * m} for m in range(1, 13)]
        t = h4.trend(up, "X", n_perm=199)
        self.assertAlmostEqual(t["spearman"], 1.0)
        self.assertLess(t["p_one_sided"], 0.05)
        down = [{"deadline": dt.date(2021, m, 15), "X": -0.1 * m} for m in range(1, 13)]
        t = h4.trend(down, "X", n_perm=199)
        self.assertAlmostEqual(t["spearman"], -1.0)
        self.assertEqual(t["p_one_sided"], 1.0)
        self.assertEqual(h4.trend(up[:2], "X", n_perm=9)["p_one_sided"], None)


class TestAnalyse(unittest.TestCase):
    def test_rising_series(self):
        res = h4.analyse(series(), n_perm=199, n_boot=50)
        self.assertEqual(res["n_playground_included"], 30)
        self.assertEqual(res["n_by_era"], {"E0": 10, "E1": 10, "E2": 10})
        self.assertTrue(res["G4_pass"])
        self.assertAlmostEqual(res["R1"]["spearman"], 1.0)
        self.assertTrue(res["R1"]["supported"])
        self.assertAlmostEqual(res["R2"]["diff_E2_minus_E0"], 1.0, places=9)
        self.assertTrue(res["R2"]["supported"])
        self.assertEqual(res["R1c_season3_onward"]["n"], 20)
        self.assertEqual(res["R1a_no_time_shifted"]["n"], 30)
        self.assertEqual(sorted(res["exploratory_by_year"]), ["2021", "2023", "2025"])
        self.assertEqual(res["first_deadline"], "2021-01-28")

    def test_gate_blocks_support(self):
        res = h4.analyse(series(n_e0=5, n_e1=0, n_e2=5), n_perm=199, n_boot=20)
        self.assertFalse(res["G4_pass"])
        self.assertLess(res["R1"]["p_one_sided"], 0.05)
        self.assertFalse(res["R1"]["supported"])
        self.assertFalse(res["R2"]["supported"])

    def test_flat_series_not_supported(self):
        res = h4.analyse(series(x0=0.2, x1=0.2, x2=0.2, jitter=0.0), n_perm=99, n_boot=20)
        self.assertIsNone(res["R1"]["spearman"])
        self.assertFalse(res["R1"]["supported"])
        self.assertEqual(res["R2"]["diff_E2_minus_E0"], 0.0)

    def test_main_writes_results(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "he2_competitions.csv")
            write_csv(p, series())
            cal = os.path.join(d, "cal.json")
            with open(cal, "w", encoding="utf-8") as f:
                json.dump({"C0_pass": True}, f)
            out = os.path.join(d, "out")
            self.assertEqual(h4.main(["--he2-csv", p, "--out-dir", out, "--calibration", cal,
                                      "--n-perm", "99", "--n-boot", "20"]), 0)
            with open(os.path.join(out, "he4_results.json"), encoding="utf-8") as f:
                res = json.load(f)
        self.assertTrue(res["he2_C0_pass"])
        self.assertEqual(len(res["input_csv_sha256"]), 64)
        self.assertTrue(res["G4_pass"])


if __name__ == "__main__":
    unittest.main()
