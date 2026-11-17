"""holdout-era he2: excess public-leaderboard optimism beyond what noise selection alone predicts.

Pre-registered in PREREGISTRATION-he2.md (read that first; this file implements it). The he1 data code
(holdout_era.py) is used unchanged, fetched from commit 15dfd9a by fetch_frozen.py and imported from
--frozen-dir. The pure helpers here are stdlib only.

MV2 of Munim, Inc. (made with AI agents working in his name). Code: Apache-2.0.
Usage: python he2_excess.py --data-dir /data --out-dir /out --frozen-dir /out/frozen
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

FROZEN_COMMIT = "15dfd9adeaee03f0fd920a9ac70d96ed9255f5ff"
MIN_TEAMS = 100
SIM_SEED = 20261118
SIM_TEAMS = 2000
SIM_REPS = 400
F_PUBLIC = 0.3
NULL_SIGMAS = (0.1, 0.3, 0.6)
SIM_SETTINGS = (
    {"name": "normal_null_s0.1", "skill": "normal", "sigma1": 0.1, "m_top": 1, "m_rest": 1, "gate": True},
    {"name": "normal_null_s0.3", "skill": "normal", "sigma1": 0.3, "m_top": 1, "m_rest": 1, "gate": True},
    {"name": "normal_null_s0.6", "skill": "normal", "sigma1": 0.6, "m_top": 1, "m_rest": 1, "gate": True},
    {"name": "skewed_null_s0.3", "skill": "skewed", "sigma1": 0.3, "m_top": 1, "m_rest": 1, "gate": False},
    {"name": "normal_adapt_all10_s0.3", "skill": "normal", "sigma1": 0.3, "m_top": 10, "m_rest": 10, "gate": False},
    {"name": "normal_adapt_top30_rest3_s0.3", "skill": "normal", "sigma1": 0.3, "m_top": 30, "m_rest": 3, "gate": False},
    {"name": "skewed_adapt_top30_rest3_s0.3", "skill": "skewed", "sigma1": 0.3, "m_top": 30, "m_rest": 3, "gate": False},
)


# ----------------------------------------------------------------- pure helpers (tested)
def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def excess_metrics(teams, is_max, min_teams=MIN_TEAMS):
    """teams: dicts with pub_rank, pub, priv (scores of the team's public-best submission), kernel.
    Returns a dict with O, X, X_raw, b_raw, b; 'excluded' set when the competition is not usable."""
    out = {"excluded": None}
    sign = 1.0 if is_max else -1.0
    el = [t for t in teams if t.get("pub_rank") and t["pub_rank"] > 0
          and finite(t.get("pub")) and finite(t.get("priv"))]
    n = len(el)
    out["n"] = n
    if n < min_teams:
        out["excluded"] = "few_teams"
        return out
    out["kernel_share"] = sum(1 for t in el if t.get("kernel")) / n
    if sum(1 for t in el if t["pub"] == t["priv"]) >= 0.9 * n:
        out["excluded"] = "no_separate_private"
        return out
    el.sort(key=lambda t: (t["pub_rank"], -sign * t["pub"]))
    k = max(10, int(math.ceil(0.02 * n)))
    x = [sign * t["pub"] for t in el]
    d = [sign * (t["pub"] - t["priv"]) for t in el]
    half_priv = [sign * el[i]["priv"] for i in range(n) if i + 1 <= 0.5 * n]
    scale = quantile(half_priv, 0.75) - quantile(half_priv, 0.25)
    b1 = [i for i in range(n) if 0.25 * n < i + 1 <= 0.5 * n]
    b2 = [i for i in range(n) if 0.5 * n < i + 1 <= 0.75 * n]
    out.update(k=k, scale=scale, n_b1=len(b1), n_b2=len(b2))
    if not (scale > 0) or not b1 or not b2:
        out["excluded"] = "zero_scale"
        return out
    mx_t, md_t = median(x[:k]), median(d[:k])
    mx1, md1 = median([x[i] for i in b1]), median([d[i] for i in b1])
    mx2, md2 = median([x[i] for i in b2]), median([d[i] for i in b2])
    out.update(mx_t=mx_t, md_t=md_t, mx1=mx1, md1=md1, mx2=mx2, md2=md2,
               O=(md_t - md1) / scale)
    if not (mx1 - mx2 > 0):
        out["excluded"] = "flat_mid"
        return out
    b_raw = (md1 - md2) / (mx1 - mx2)
    b = clamp(b_raw, 0.0, 1.0)
    out.update(b_raw=b_raw, b=b,
               X=(md_t - md1 - b * (mx_t - mx1)) / scale,
               X_raw=(md_t - md1 - b_raw * (mx_t - mx1)) / scale)
    return out


def sign_test(xs):
    """Exact one-sided sign test that positives outnumber negatives (zeros dropped)."""
    pos = sum(1 for v in xs if v > 0)
    neg = sum(1 for v in xs if v < 0)
    m = pos + neg
    if m == 0:
        return {"pos": 0, "neg": 0, "p_one_sided": None}
    p = sum(math.comb(m, j) for j in range(pos, m + 1)) / 2.0 ** m
    return {"pos": pos, "neg": neg, "p_one_sided": p}


def sim_leaderboard(rng, n_teams, skill, sigma1, m_top, m_rest, f=F_PUBLIC):
    """One simulated leaderboard (higher is better). Each team keeps its best public score out of m
    variants of equal skill; the private noise of the kept variant is independent of its public noise."""
    sigma2 = sigma1 * math.sqrt(f / (1.0 - f))
    if skill == "normal":
        skills = [rng.gauss(0.0, 1.0) for _ in range(n_teams)]
    else:
        skills = [-rng.expovariate(1.0) for _ in range(n_teams)]
    skills.sort(reverse=True)
    rows = []
    for j, s in enumerate(skills):
        m = m_top if j < 0.1 * n_teams else m_rest
        pub = max(s + rng.gauss(0.0, sigma1) for _ in range(m))
        rows.append((pub, s + rng.gauss(0.0, sigma2)))
    rows.sort(key=lambda r: -r[0])
    return [{"pub_rank": i + 1, "pub": p, "priv": q, "kernel": False} for i, (p, q) in enumerate(rows)]


def calibrate(reps=SIM_REPS, n_teams=SIM_TEAMS, seed=SIM_SEED, settings=SIM_SETTINGS):
    res = {"f_public": F_PUBLIC, "n_teams": n_teams, "reps": reps, "seed": seed, "settings": {}}
    for i, st in enumerate(settings):
        rng = random.Random(seed + i)
        os_, xs = [], []
        for _ in range(reps):
            m = excess_metrics(sim_leaderboard(rng, n_teams, st["skill"], st["sigma1"], st["m_top"], st["m_rest"]),
                               True)
            if m.get("X") is not None:
                os_.append(m["O"])
                xs.append(m["X"])
        res["settings"][st["name"]] = dict(st, n=len(xs), median_O=median(os_) if os_ else None,
                                           median_X=median(xs) if xs else None)
    gate = [v for v in res["settings"].values() if v["gate"]]
    res["C0_pass"] = bool(gate) and all(v["median_X"] is not None and v["median_O"] is not None
                                        and abs(v["median_X"]) < 0.5 * v["median_O"] for v in gate)
    return res


# ----------------------------------------------------------------- analysis on the real data
def run(data_dir, out_dir, frozen_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    cal = calibrate()
    with open(os.path.join(out_dir, "he2_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=1)
    print(json.dumps({"C0_pass": cal["C0_pass"],
                      "settings": {k: (v["median_O"], v["median_X"]) for k, v in cal["settings"].items()}},
                     indent=1))

    sys.path.insert(0, os.path.abspath(frozen_dir))
    import holdout_era as he
    frozen = {"commit": FROZEN_COMMIT, "sha256": he.sha256_file(os.path.join(frozen_dir, "holdout_era.py"))}
    paths, data_rec = he.get_data(data_dir)
    comps = he.build_competitions(paths)

    tcols = he.read_cols(paths["nodes_team.parquet"],
                         ["Id", "CompetitionId", "PublicLeaderboardRank", "PrivateLeaderboardRank", "IsBenchmark"])
    team_comp, team_rank, sign_of_team = {}, {}, {}
    for i, tid in enumerate(tcols["Id"]):
        cid = tcols["CompetitionId"][i]
        if tcols["IsBenchmark"][i]:
            continue
        comp = comps.get(cid)
        if comp is None or comp["is_max"] is None:
            continue
        team_comp[tid] = cid
        team_rank[tid] = he.as_int(tcols["PublicLeaderboardRank"][i])
        sign_of_team[tid] = 1.0 if comp["is_max"] else -1.0
    del tcols
    s = he.read_cols(paths["nodes_submission.parquet"],
                     ["Id", "TeamId", "IsAfterDeadline", "IsSelected", "PublicScoreFullPrecision",
                      "PrivateScoreFullPrecision", "SourceKernelVersionId"])
    rows = zip(s["Id"], s["TeamId"], s["IsAfterDeadline"], s["IsSelected"],
               (he.as_float(v) for v in s["PublicScoreFullPrecision"]),
               (he.as_float(v) for v in s["PrivateScoreFullPrecision"]), s["SourceKernelVersionId"])
    best = he.pick_public_best(rows, sign_of_team)
    del s, rows
    per_comp = {}
    for tid, cid in team_comp.items():
        b = best.get(tid)
        per_comp.setdefault(cid, []).append({"pub_rank": team_rank[tid], "pub": b[0] if b else None,
                                             "priv": b[1] if b else None, "kernel": b[2] if b else False})

    table = []
    for cid, comp in sorted(comps.items(), key=lambda kv: (kv[1]["deadline"] or "", kv[0])):
        row = dict(comp)
        reason = None
        if comp["is_max"] is None:
            reason = "unknown_direction"
        elif (comp["segment"] or "").strip().lower() in he.EXCLUDED_SEGMENTS:
            reason = "segment"
        elif comp["era"] is None:
            reason = "deadline_outside_eras"
        m = excess_metrics(per_comp.get(cid, []), bool(comp["is_max"])) if comp["is_max"] is not None else {}
        row.update(m)
        row["excluded"] = reason or m.get("excluded")
        row["code"] = (m.get("kernel_share") or 0) >= 0.5
        row["seg_group"] = "playground" if (comp["segment"] or "").strip().lower() == "playground" else "main"
        table.append(row)

    inc = [r for r in table if r["excluded"] is None]
    by = {e: [r for r in inc if r["era"] == e] for e in ("E0", "E1", "E2")}
    res = {"frozen": frozen, "data": data_rec, "calibration_C0_pass": cal["C0_pass"],
           "n_competitions_in_data": len(comps), "n_included": len(inc),
           "n_by_era": {e: len(v) for e, v in by.items()}, "exclusions": {}}
    for r in table:
        if r["excluded"]:
            res["exclusions"][r["excluded"]] = res["exclusions"].get(r["excluded"], 0) + 1

    def summ(rs, key):
        xs = [r[key] for r in rs if r.get(key) is not None]
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "median": median(xs), "q25": quantile(xs, 0.25), "q75": quantile(xs, 0.75),
                "share_positive": sum(1 for v in xs if v > 0) / len(xs)}

    res["era_summary"] = {e: {k: summ(v, k) for k in ("X", "X_raw", "O", "b", "b_raw")} for e, v in by.items()}
    res["G2_pass"] = len(inc) >= 60 and len(by["E0"]) >= 15 and len(by["E2"]) >= 15
    res["Q1"] = dict(sign_test([r["X"] for r in inc]), median_X=median([r["X"] for r in inc]) if inc else None)
    res["Q1"]["supported"] = res["Q1"]["p_one_sided"] is not None and res["Q1"]["p_one_sided"] < 0.05
    if res["G2_pass"]:
        x0 = [r["X"] for r in by["E0"]]
        x2 = [r["X"] for r in by["E2"]]
        obs, p = he.perm_test_median_diff(x0, x2)
        lo, hi = he.boot_ci_median_diff(x0, x2)
        res["Q2"] = {"median_X_E0": median(x0), "median_X_E2": median(x2), "diff_E2_minus_E0": obs,
                     "p_one_sided": p, "boot95": [lo, hi], "supported": p < 0.05}
        both = by["E0"] + by["E2"]
        strata = {}
        for r in both:
            strata.setdefault((r["seg_group"], r["code"]), []).append(r["X"])
        smed = {k: median(v) for k, v in strata.items()}
        ra = [r["X"] - smed[(r["seg_group"], r["code"])] for r in by["E0"]]
        rb = [r["X"] - smed[(r["seg_group"], r["code"])] for r in by["E2"]]
        sa = ["%s|%s" % (r["seg_group"], r["code"]) for r in by["E0"]]
        sb = ["%s|%s" % (r["seg_group"], r["code"]) for r in by["E2"]]
        sobs, sp = he.perm_test_median_diff(ra, rb, strata_a=sa, strata_b=sb)
        res["Q2_stratified"] = {"diff_residual_medians": sobs, "p_one_sided": sp,
                                "strata_counts": {"%s|%s" % k: len(v) for k, v in strata.items()}}
        n0 = [r["X"] for r in by["E0"] if not r["time_shifted"]]
        n2 = [r["X"] for r in by["E2"] if not r["time_shifted"]]
        if len(n0) >= 10 and len(n2) >= 10:
            tobs, tp = he.perm_test_median_diff(n0, n2)
            res["Q2_no_time_shifted"] = {"n_E0": len(n0), "n_E2": len(n2), "diff": tobs, "p_one_sided": tp}
        r0 = [r["X_raw"] for r in by["E0"]]
        r2 = [r["X_raw"] for r in by["E2"]]
        robs, rp = he.perm_test_median_diff(r0, r2)
        res["Q2_X_raw"] = {"median_E0": median(r0), "median_E2": median(r2), "diff": robs, "p_one_sided": rp}
    pairs = [(math.log(r["subs_per_team"]), r["X"]) for r in inc if r.get("subs_per_team")]
    if len(pairs) >= 30:
        rho, p = he.spearman_perm([a for a, _ in pairs], [b for _, b in pairs])
        res["Q3"] = {"n": len(pairs), "spearman_logspt_X": rho, "p_one_sided": p,
                     "supported": p is not None and p < 0.05}
    years = sorted({r["year"] for r in inc if r["year"]})
    res["exploratory_by_year"] = {str(y): {k: summ([r for r in inc if r["year"] == y], k) for k in ("X", "O", "b")}
                                  for y in years}
    res["seconds"] = round(time.time() - t0, 1)

    keep = ["id", "slug", "title", "segment", "deadline", "era", "metric", "is_max", "total_teams",
            "total_submissions", "subs_per_team", "time_shifted", "n", "k", "scale", "mx_t", "md_t", "mx1",
            "md1", "mx2", "md2", "b_raw", "b", "O", "X", "X_raw", "kernel_share", "code", "seg_group", "excluded"]
    with open(os.path.join(out_dir, "he2_competitions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keep)
        for r in table:
            w.writerow([r.get(k) for k in keep])
    with open(os.path.join(out_dir, "he2_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("exploratory")}, indent=1, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out-dir", default="/out")
    ap.add_argument("--frozen-dir", default="/out/frozen")
    a = ap.parse_args(argv)
    run(a.data_dir, a.out_dir, a.frozen_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
