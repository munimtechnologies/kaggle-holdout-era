"""holdout-era (he1): has public-leaderboard overfitting on Kaggle grown in the era of AI coding agents?

Pre-registered in PREREGISTRATION-he1.md (read that first; this file implements it).
Data: Meta Kaggle snapshot 2026-11-12 as packaged by habedi/kaggle-knowledge-graph on Hugging Face
(CC BY-NC-SA 4.0, derived from Kaggle's Meta Kaggle). Only three node tables are used.

MV2 of Munim, Inc. (made with AI agents working in his name). Code: Apache-2.0.
Usage: python holdout_era.py --data-dir /data --out-dir /out
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request

REPO = "habedi/kaggle-knowledge-graph"
REVISION = "51fe1e53d84fc5ebc0ed2f69948835feb8945c7a"
SNAPSHOT = "2026-11-12"
FILES = ["nodes_competition.parquet", "nodes_team.parquet", "nodes_submission.parquet"]
PREREG_SHA256 = {  # as listed in the dataset's manifest.json on 2026-11-17
    "nodes_competition.parquet": "2ec2cc2e7c35aafec8080fb4229edcc7b3b9a4a2b9f88d25c9544757950acbd8",
    "nodes_team.parquet": "a229c08bc6caa6547721478f350cfc969300ad507c6972b88c56a55ee0f85586",
    "nodes_submission.parquet": "6a88915160c24fda3a2c35f1d6b3123600d4b4be13e999349863a555286e525a",
}
USER_AGENT = "MV2-Munim-Inc-research (mv2@munimtech.com)"

E0 = (dt.date(2020, 1, 1), dt.date(2022, 11, 30))   # before ChatGPT (released 2022-11-30)
E1 = (dt.date(2022, 12, 1), dt.date(2024, 12, 31))  # chat assistants
E2 = (dt.date(2025, 1, 1), dt.date(2026, 11, 5))    # agentic coding tools widespread
EXCLUDED_SEGMENTS = {"getting started", "simulations", "analytics", "community"}
MIN_TEAMS = 100
SEED = 20261117
N_PERM = 10000
N_BOOT = 5000

TIME_SHIFT_RE = re.compile(
    r"forecasting phase|future data|will be (?:re-?run|rescored|re-?scored) (?:on|against)|"
    r"real[- ]time data|data collected after the (?:final )?submission deadline", re.I)


# ----------------------------------------------------------------- small pure helpers (tested)
def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_date(v):
    """Return a datetime.date or None. Accepts date/datetime objects and common string forms."""
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    s = str(v).strip().replace("T", " ")
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    tok = s.split()[0]
    try:
        if "/" in tok:
            m, d, y = tok.split("/")[:3]
        elif "-" in tok:
            y, m, d = tok.split("-")[:3]
        else:
            return None
        return dt.date(int(y), int(m), int(d[:2]))
    except (ValueError, TypeError):
        return None


def era_of(d):
    if d is None:
        return None
    for name, (a, b) in (("E0", E0), ("E1", E1), ("E2", E2)):
        if a <= d <= b:
            return name
    return None


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


def is_kernel(v):
    if v is None:
        return False
    s = str(v).strip().lower()
    return s not in ("", "nan", "none", "0", "null")


def is_time_shifted(text):
    return bool(TIME_SHIFT_RE.search(text or ""))


def pick_public_best(rows, sign_of_team):
    """rows: iterable of (sub_id, team_id, after_deadline, selected, pub, priv, kernel_raw).
    For each team (with a known sign) keep the submission with the best oriented public score among
    submissions made before the deadline; ties -> a selected one, then the lowest id.
    Returns {team_id: (pub, priv, kernel_bool)}."""
    best = {}
    for sid, tid, after, sel, pub, priv, kern in rows:
        if after or not finite(pub):
            continue
        s = sign_of_team.get(tid)
        if s is None:
            continue
        key = (s * pub, 1 if sel else 0, -(sid or 0))
        cur = best.get(tid)
        if cur is None or key > cur[0]:
            best[tid] = (key, pub, priv if finite(priv) else None, is_kernel(kern))
    return {t: v[1:] for t, v in best.items()}


def comp_metrics(teams, is_max, min_teams=MIN_TEAMS):
    """teams: list of dicts with pub_rank, priv_rank (Kaggle ranks or None), pub, priv (scores of the
    team's public-best submission), kernel (bool). Returns a dict; 'excluded' set when not usable."""
    out = {"n_teams_in_data": len(teams), "excluded": None}
    sign = 1.0 if is_max else -1.0
    rk = [t for t in teams if t.get("pub_rank") and t.get("priv_rank")]
    a = {i for i, t in enumerate(rk) if t["pub_rank"] <= 10}
    b = {i for i, t in enumerate(rk) if t["priv_rank"] <= 10}
    out["ret10"] = (len(a & b) / len(a)) if len(a) >= 10 else None
    top100 = [t for t in rk if t["pub_rank"] <= 100]
    out["rho100"] = spearman([t["pub_rank"] for t in top100], [t["priv_rank"] for t in top100]) \
        if len(top100) >= 30 else None

    el = [t for t in teams if t.get("pub_rank") and t["pub_rank"] > 0
          and finite(t.get("pub")) and finite(t.get("priv"))]
    n = len(el)
    out["n"] = n
    if n < min_teams:
        out["excluded"] = "few_teams"
        return out
    same = sum(1 for t in el if t["pub"] == t["priv"])
    out["share_pub_eq_priv"] = same / n
    if same >= 0.9 * n:
        out["excluded"] = "no_separate_private"
        return out
    el.sort(key=lambda t: (t["pub_rank"], -sign * t["pub"]))
    k = max(10, int(math.ceil(0.02 * n)))
    d = [sign * (t["pub"] - t["priv"]) for t in el]
    top = d[:k]
    ref = [d[i] for i in range(n) if 0.25 * n < i + 1 <= 0.5 * n]
    half_priv = [sign * el[i]["priv"] for i in range(n) if i + 1 <= 0.5 * n]
    scale = quantile(half_priv, 0.75) - quantile(half_priv, 0.25)
    out.update(k=k, n_ref=len(ref), scale=scale, med_d_top=median(top), med_d_ref=median(ref),
               kernel_share=sum(1 for t in el if t.get("kernel")) / n)
    if not (scale > 0) or not ref:
        out["excluded"] = "zero_scale"
        return out
    out["O"] = (median(top) - median(ref)) / scale
    return out


def perm_test_median_diff(a, b, n_perm=N_PERM, seed=SEED, strata_a=None, strata_b=None):
    """One-sided p for median(b) - median(a) > 0. With strata lists, labels are shuffled within strata."""
    obs = median(b) - median(a)
    rng = random.Random(seed)
    if strata_a is None:
        pool = list(a) + list(b)
        na = len(a)
        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(pool)
            if median(pool[na:]) - median(pool[:na]) >= obs - 1e-12:
                cnt += 1
        return obs, (cnt + 1) / (n_perm + 1)
    groups = {}
    for v, s in zip(a, strata_a):
        groups.setdefault(s, []).append((v, 0))
    for v, s in zip(b, strata_b):
        groups.setdefault(s, []).append((v, 1))
    cnt = 0
    for _ in range(n_perm):
        pa, pb = [], []
        for s in sorted(groups, key=str):
            vals = [v for v, _ in groups[s]]
            labs = [l for _, l in groups[s]]
            rng.shuffle(labs)
            for v, l in zip(vals, labs):
                (pb if l else pa).append(v)
        if median(pb) - median(pa) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def boot_ci_median_diff(a, b, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed + 1)
    diffs = []
    for _ in range(n_boot):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(median(rb) - median(ra))
    return quantile(diffs, 0.025), quantile(diffs, 0.975)


def spearman_perm(x, y, n_perm=N_PERM, seed=SEED):
    obs = spearman(x, y)
    if obs is None:
        return None, None
    rng = random.Random(seed + 2)
    yy = list(y)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(yy)
        r = spearman(x, yy)
        if r is not None and r >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


# ----------------------------------------------------------------- data
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def find_manifest_sha(obj, name):
    if isinstance(obj, dict):
        hit = any(isinstance(v, str) and (v == name or v.endswith("/" + name)) for v in obj.values())
        if hit and isinstance(obj.get("sha256"), str):
            return obj["sha256"]
        for k, v in obj.items():
            if k == name or (isinstance(k, str) and k.endswith("/" + name)):
                if isinstance(v, dict) and isinstance(v.get("sha256"), str):
                    return v["sha256"]
            r = find_manifest_sha(v, name)
            if r:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_manifest_sha(v, name)
            if r:
                return r
    return None


def get_data(data_dir):
    base = "https://huggingface.co/datasets/%s/resolve/%s/" % (REPO, REVISION)
    folder = os.path.join(data_dir, "kkg-" + REVISION[:12])
    os.makedirs(folder, exist_ok=True)
    rec = {"repo": REPO, "revision": REVISION, "revision_used": REVISION, "files": {}}
    mpath = os.path.join(folder, "manifest.json")
    if not os.path.exists(mpath):
        try:
            fetch(base + "manifest.json", mpath)
        except Exception as e:  # pinned revision unreachable: fall back to main, recorded as a deviation
            rec["revision_error"] = "%s: %s" % (type(e).__name__, e)
            rec["revision_used"] = "main"
            base = "https://huggingface.co/datasets/%s/resolve/main/" % REPO
            fetch(base + "manifest.json", mpath)
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    rec["manifest_version"] = manifest.get("version") if isinstance(manifest, dict) else None
    rec["manifest_snapshot"] = manifest.get("meta_kaggle_snapshot") if isinstance(manifest, dict) else None
    rec["snapshot_as_preregistered"] = rec["manifest_snapshot"] == SNAPSHOT or rec["manifest_version"] == SNAPSHOT
    paths = {}
    for name in FILES:
        p = os.path.join(folder, name)
        if not os.path.exists(p):
            fetch(base + "data/" + name, p)
        got = sha256_file(p)
        rec["files"][name] = {"bytes": os.path.getsize(p), "sha256": got,
                              "manifest_sha256": find_manifest_sha(manifest, name),
                              "matches_prereg": got == PREREG_SHA256[name]}
        rec["files"][name]["matches_manifest"] = got == rec["files"][name]["manifest_sha256"]
        paths[name] = p
    return paths, rec


def read_cols(path, cols):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=cols)
    return {c: t.column(c).to_pylist() for c in cols}


def as_int(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


def as_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ----------------------------------------------------------------- analysis
def build_competitions(paths):
    c = read_cols(paths["nodes_competition.parquet"],
                  ["Id", "Slug", "Title", "HostSegmentTitle", "EnabledDate", "DeadlineDate",
                   "EvaluationAlgorithmName", "EvaluationAlgorithmIsMax", "RewardType", "RewardQuantity",
                   "TotalTeams", "TotalSubmissions", "Overview", "Rules", "DatasetDescription"])
    comps = {}
    for i, cid in enumerate(c["Id"]):
        text = " ".join(str(c[k][i] or "") for k in ("Overview", "Rules", "DatasetDescription"))
        dl = parse_date(c["DeadlineDate"][i])
        tt, ts = as_int(c["TotalTeams"][i]), as_int(c["TotalSubmissions"][i])
        comps[cid] = {
            "id": cid, "slug": c["Slug"][i], "title": c["Title"][i], "segment": c["HostSegmentTitle"][i],
            "deadline": dl.isoformat() if dl else None, "era": era_of(dl), "year": dl.year if dl else None,
            "metric": c["EvaluationAlgorithmName"][i], "is_max": c["EvaluationAlgorithmIsMax"][i],
            "reward_type": c["RewardType"][i], "reward": as_float(c["RewardQuantity"][i]),
            "total_teams": tt, "total_submissions": ts,
            "subs_per_team": (ts / tt) if (tt and ts) else None,
            "time_shifted": is_time_shifted(text),
        }
    return comps


def run(data_dir, out_dir):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    paths, data_rec = get_data(data_dir)
    comps = build_competitions(paths)

    tcols = read_cols(paths["nodes_team.parquet"],
                      ["Id", "CompetitionId", "PublicLeaderboardRank", "PrivateLeaderboardRank", "IsBenchmark"])
    team_comp, team_ranks, sign_of_team = {}, {}, {}
    n_bench = 0
    for i, tid in enumerate(tcols["Id"]):
        cid = tcols["CompetitionId"][i]
        if tcols["IsBenchmark"][i]:
            n_bench += 1
            continue
        comp = comps.get(cid)
        if comp is None or comp["is_max"] is None:
            continue
        team_comp[tid] = cid
        team_ranks[tid] = (as_int(tcols["PublicLeaderboardRank"][i]), as_int(tcols["PrivateLeaderboardRank"][i]))
        sign_of_team[tid] = 1.0 if comp["is_max"] else -1.0
    del tcols

    s = read_cols(paths["nodes_submission.parquet"],
                  ["Id", "TeamId", "IsAfterDeadline", "IsSelected", "PublicScoreFullPrecision",
                   "PrivateScoreFullPrecision", "SourceKernelVersionId"])
    n_subs = len(s["Id"])
    rows = zip(s["Id"], s["TeamId"], s["IsAfterDeadline"], s["IsSelected"],
               (as_float(v) for v in s["PublicScoreFullPrecision"]),
               (as_float(v) for v in s["PrivateScoreFullPrecision"]), s["SourceKernelVersionId"])
    best = pick_public_best(rows, sign_of_team)
    del s, rows

    per_comp = {}
    for tid, cid in team_comp.items():
        pr, vr = team_ranks[tid]
        b = best.get(tid)
        per_comp.setdefault(cid, []).append({
            "pub_rank": pr, "priv_rank": vr,
            "pub": b[0] if b else None, "priv": b[1] if b else None, "kernel": b[2] if b else False})

    table = []
    for cid, comp in sorted(comps.items(), key=lambda kv: (kv[1]["deadline"] or "", kv[0])):
        row = dict(comp)
        reason = None
        if comp["is_max"] is None:
            reason = "unknown_direction"
        elif (comp["segment"] or "").strip().lower() in EXCLUDED_SEGMENTS:
            reason = "segment"
        elif comp["era"] is None:
            reason = "deadline_outside_eras"
        m = comp_metrics(per_comp.get(cid, []), bool(comp["is_max"])) if comp["is_max"] is not None else {}
        row.update(m)
        if reason is None:
            reason = m.get("excluded")
        row["excluded"] = reason
        row["code"] = (m.get("kernel_share") or 0) >= 0.5
        row["seg_group"] = "playground" if (comp["segment"] or "").strip().lower() == "playground" else "main"
        table.append(row)

    inc = [r for r in table if r["excluded"] is None]
    by = {e: [r for r in inc if r["era"] == e] for e in ("E0", "E1", "E2")}
    res = {"data": data_rec, "n_competitions_in_data": len(comps), "n_submissions_in_data": n_subs,
           "n_benchmark_teams_skipped": n_bench, "n_included": len(inc),
           "n_by_era": {e: len(v) for e, v in by.items()},
           "exclusions": {}}
    for r in table:
        if r["excluded"]:
            res["exclusions"][r["excluded"]] = res["exclusions"].get(r["excluded"], 0) + 1

    def summ(rs, key):
        xs = [r[key] for r in rs if r.get(key) is not None]
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "median": median(xs), "q25": quantile(xs, 0.25), "q75": quantile(xs, 0.75),
                "share_positive": sum(1 for x in xs if x > 0) / len(xs)}

    res["era_summary"] = {e: {k: summ(v, k) for k in ("O", "ret10", "rho100", "subs_per_team", "kernel_share")}
                          for e, v in by.items()}
    res["G0_pass"] = len(inc) >= 60 and len(by["E0"]) >= 15 and len(by["E2"]) >= 15
    allO = [r["O"] for r in inc]
    res["P0_overall_median_O"] = median(allO) if allO else None
    res["P0_pass"] = bool(allO) and median(allO) > 0

    o0 = [r["O"] for r in by["E0"]]
    o2 = [r["O"] for r in by["E2"]]
    if res["G0_pass"]:
        obs, p = perm_test_median_diff(o0, o2)
        lo, hi = boot_ci_median_diff(o0, o2)
        res["H1"] = {"median_O_E0": median(o0), "median_O_E2": median(o2), "diff_E2_minus_E0": obs,
                     "p_one_sided": p, "boot95": [lo, hi], "supported": p < 0.05}
        # stratified check: residual from stratum median (E0+E2 pooled), labels shuffled within strata
        both = by["E0"] + by["E2"]
        strata = {}
        for r in both:
            strata.setdefault((r["seg_group"], r["code"]), []).append(r["O"])
        smed = {k: median(v) for k, v in strata.items()}
        ra = [r["O"] - smed[(r["seg_group"], r["code"])] for r in by["E0"]]
        rb = [r["O"] - smed[(r["seg_group"], r["code"])] for r in by["E2"]]
        sa = ["%s|%s" % (r["seg_group"], r["code"]) for r in by["E0"]]
        sb = ["%s|%s" % (r["seg_group"], r["code"]) for r in by["E2"]]
        sobs, sp = perm_test_median_diff(ra, rb, strata_a=sa, strata_b=sb)
        res["H1_stratified"] = {"diff_residual_medians": sobs, "p_one_sided": sp,
                                "strata_counts": {"%s|%s" % k: len(v) for k, v in strata.items()}}
        no_ts0 = [r["O"] for r in by["E0"] if not r["time_shifted"]]
        no_ts2 = [r["O"] for r in by["E2"] if not r["time_shifted"]]
        if len(no_ts0) >= 10 and len(no_ts2) >= 10:
            tobs, tp = perm_test_median_diff(no_ts0, no_ts2)
            res["H1_no_time_shifted"] = {"n_E0": len(no_ts0), "n_E2": len(no_ts2), "diff": tobs, "p_one_sided": tp}
        for e_a, e_b, key in (("E0", "E2", "ret10"), ("E0", "E2", "rho100")):
            xa = [r[key] for r in by[e_a] if r.get(key) is not None]
            xb = [r[key] for r in by[e_b] if r.get(key) is not None]
            if len(xa) >= 10 and len(xb) >= 10:
                # more overfitting = LOWER retention / rank correlation, so test median(a) - median(b) > 0
                kobs, kp = perm_test_median_diff(xb, xa)
                res["S_%s" % key] = {"median_E0": median(xa), "median_E2": median(xb),
                                     "diff_E0_minus_E2": kobs, "p_one_sided": kp}
    pairs = [(math.log(r["subs_per_team"]), r["O"]) for r in inc if r.get("subs_per_team")]
    if len(pairs) >= 30:
        rho, p = spearman_perm([x for x, _ in pairs], [y for _, y in pairs])
        res["H2"] = {"n": len(pairs), "spearman_logspt_O": rho, "p_one_sided": p,
                     "supported": p is not None and p < 0.05}
    # exploratory (ungraded): by deadline year and by segment group
    years = sorted({r["year"] for r in inc if r["year"]})
    res["exploratory_by_year"] = {str(y): summ([r for r in inc if r["year"] == y], "O") for y in years}
    res["exploratory_by_group"] = {"%s|%s|%s" % (e, g, c): summ([r for r in by[e] if r["seg_group"] == g and r["code"] == c], "O")
                                   for e in ("E0", "E1", "E2") for g in ("main", "playground") for c in (False, True)}
    res["seconds"] = round(time.time() - t0, 1)

    keep = ["id", "slug", "title", "segment", "deadline", "era", "metric", "is_max", "reward_type", "reward",
            "total_teams", "total_submissions", "subs_per_team", "time_shifted", "n_teams_in_data", "n",
            "share_pub_eq_priv", "k", "n_ref", "scale", "med_d_top", "med_d_ref", "O", "ret10", "rho100",
            "kernel_share", "code", "seg_group", "excluded"]
    with open(os.path.join(out_dir, "he1_competitions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keep)
        for r in table:
            w.writerow([r.get(k) for k in keep])
    with open(os.path.join(out_dir, "he1_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("exploratory")}, indent=1, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out-dir", default="/out")
    a = ap.parse_args(argv)
    run(a.data_dir, a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
