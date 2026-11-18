"""holdout-era he4: Kaggle's Playground Series as a natural experiment for leaderboard overfitting over time.

Pre-registered in PREREGISTRATION-he4.md (read that first; this file implements it). Input is the
per-competition table he2_competitions.csv written by he2_excess.py (pre-registered in
PREREGISTRATION-he2.md); no other data are read. Stdlib only.

MV2 of Munim, Inc. (made with AI agents working in his name). Code: Apache-2.0.
Usage: python he4_playground.py --he2-csv /path/he2_competitions.csv --out-dir /out
       [--calibration /path/he2_calibration.json]
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import time

SEED = 20261118
N_PERM = 10000
N_BOOT = 5000
S3_START = dt.date(2023, 1, 1)   # Playground Series season 3 onward (robustness R1c)
G4_MIN_TOTAL = 30
G4_MIN_ERA = 10


# ----------------------------------------------------------------- parsing
def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan", "null"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def parse_date(v):
    s = str(v or "").strip()
    if len(s) < 10:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def playground_rows(rows):
    """Included Playground competitions from he2_competitions.csv: not excluded by he2, segment group
    'playground', a valid deadline and a finite X."""
    out = []
    for r in rows:
        if (r.get("excluded") or "").strip():
            continue
        if (r.get("seg_group") or "").strip().lower() != "playground":
            continue
        d = parse_date(r.get("deadline"))
        x = to_float(r.get("X"))
        if d is None or x is None:
            continue
        out.append({"slug": r.get("slug"), "deadline": d, "era": (r.get("era") or "").strip(),
                    "X": x, "X_raw": to_float(r.get("X_raw")), "O": to_float(r.get("O")),
                    "time_shifted": (r.get("time_shifted") or "").strip().lower() == "true"})
    out.sort(key=lambda r: (r["deadline"], r["slug"] or ""))
    return out


# ----------------------------------------------------------------- statistics (same definitions as he1)
def quantile(xs, q):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("empty")
    h = (n - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def median(xs):
    return quantile(xs, 0.5)


def avg_ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(avg_ranks(x), avg_ranks(y))


def spearman_perm(x, y, n_perm=N_PERM, seed=SEED):
    """One-sided permutation p that the Spearman correlation is positive."""
    obs = spearman(x, y)
    if obs is None:
        return None, None
    rng = random.Random(seed)
    yy = list(y)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(yy)
        r = spearman(x, yy)
        if r is not None and r >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def perm_test_median_diff(a, b, n_perm=N_PERM, seed=SEED):
    """One-sided p for median(b) - median(a) > 0 (labels shuffled)."""
    obs = median(b) - median(a)
    rng = random.Random(seed + 1)
    pool = list(a) + list(b)
    na = len(a)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if median(pool[na:]) - median(pool[:na]) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def boot_ci_median_diff(a, b, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed + 2)
    diffs = []
    for _ in range(n_boot):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(median(rb) - median(ra))
    return quantile(diffs, 0.025), quantile(diffs, 0.975)


def trend(rows, key, n_perm=N_PERM, seed=SEED):
    pts = [(r["deadline"].toordinal(), r[key]) for r in rows if r.get(key) is not None]
    out = {"n": len(pts), "spearman": None, "p_one_sided": None}
    if len(pts) >= 3:
        rho, p = spearman_perm([a for a, _ in pts], [b for _, b in pts], n_perm=n_perm, seed=seed)
        out.update(spearman=rho, p_one_sided=p)
    return out


# ----------------------------------------------------------------- analysis
def analyse(rows, n_perm=N_PERM, n_boot=N_BOOT, seed=SEED):
    pg = playground_rows(rows)
    by = {e: [r for r in pg if r["era"] == e] for e in ("E0", "E1", "E2")}
    res = {"n_playground_included": len(pg), "n_by_era": {e: len(v) for e, v in by.items()},
           "first_deadline": pg[0]["deadline"].isoformat() if pg else None,
           "last_deadline": pg[-1]["deadline"].isoformat() if pg else None}
    g4 = len(pg) >= G4_MIN_TOTAL and len(by["E0"]) >= G4_MIN_ERA and len(by["E2"]) >= G4_MIN_ERA
    res["G4_pass"] = g4

    r1 = trend(pg, "X", n_perm, seed)
    r1["supported"] = bool(g4 and r1["p_one_sided"] is not None and r1["p_one_sided"] < 0.05)
    res["R1"] = r1
    res["R1a_no_time_shifted"] = trend([r for r in pg if not r["time_shifted"]], "X", n_perm, seed)
    res["R1b_X_raw"] = trend(pg, "X_raw", n_perm, seed)
    res["R1c_season3_onward"] = trend([r for r in pg if r["deadline"] >= S3_START], "X", n_perm, seed)

    x0 = [r["X"] for r in by["E0"]]
    x2 = [r["X"] for r in by["E2"]]
    if len(x0) >= 2 and len(x2) >= 2:
        obs, p = perm_test_median_diff(x0, x2, n_perm, seed)
        lo, hi = boot_ci_median_diff(x0, x2, n_boot, seed)
        res["R2"] = {"median_X_E0": median(x0), "median_X_E2": median(x2), "diff_E2_minus_E0": obs,
                     "p_one_sided": p, "boot95": [lo, hi], "supported": bool(g4 and p < 0.05)}
    res["R3_O_trend"] = trend(pg, "O", n_perm, seed)

    years = sorted({r["deadline"].year for r in pg})
    expl = {}
    for y in years:
        ys = [r for r in pg if r["deadline"].year == y]
        os_ = [r["O"] for r in ys if r["O"] is not None]
        expl[str(y)] = {"n": len(ys), "median_X": median([r["X"] for r in ys]),
                        "median_O": median(os_) if os_ else None}
    res["exploratory_by_year"] = expl
    return res


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--he2-csv", required=True)
    ap.add_argument("--out-dir", default="/out")
    ap.add_argument("--calibration", default=None, help="he2_calibration.json (records he2's C0 gate)")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    res = {"script_sha256": sha256_file(os.path.abspath(__file__)),
           "input_csv_sha256": sha256_file(a.he2_csv), "seed": SEED, "n_perm": a.n_perm, "n_boot": a.n_boot}
    if a.calibration and os.path.exists(a.calibration):
        with open(a.calibration, encoding="utf-8") as f:
            res["he2_C0_pass"] = json.load(f).get("C0_pass")
    else:
        res["he2_C0_pass"] = None
    res.update(analyse(load_rows(a.he2_csv), a.n_perm, a.n_boot, SEED))
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out_dir, "he4_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
