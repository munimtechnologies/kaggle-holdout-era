# Pre-registration he3: a prospective replication of he1 on competitions that have not ended yet

MV2 of Munim, Inc. (mv2@munimtech.com). Written 2026-11-17 by AI agents working in his name, **before the
results of he1 exist and before any competition in the he3 window has ended**. This file is not edited
after it is public; any change goes in a new, dated file. Code: Apache-2.0.

## Why a prospective replication

he1 ([PREREGISTRATION-he1.md](PREREGISTRATION-he1.md)) asks whether public-leaderboard optimism at the top
of Kaggle leaderboards (metric O) is higher for competitions that ended in 2025-2026 than for those that
ended in 2020 to November 2022. he1 is pre-registered, but its data already exist, so a reader cannot rule
out that the design was shaped by what is already known about those years. he3 removes that doubt: it
fixes, today, the same test on competitions whose private leaderboards **have not been revealed yet**
(deadlines from 2026-11-06), using a Meta Kaggle snapshot that **will not exist until mid-2027**. Whatever
he1 finds, he3 is the out-of-sample check.

## Frozen code

`holdout_era.py` exactly as committed in munimtechnologies/kaggle-holdout-era at commit
`15dfd9adeaee03f0fd920a9ac70d96ed9255f5ff` (2026-11-17), with only these changes, all mechanical:

1. `REVISION`, `SNAPSHOT` and `PREREG_SHA256` point to the he3 snapshot (below); the hashes are taken
   from that snapshot's own manifest when the run is set up.
2. A new era **E3 = 2026-11-06 to 2027-06-30** (deadline dates, inclusive) is added, and every test that
   compares E0 with E2 compares E0 with E3 instead. E0 stays 2020-01-01 to 2022-11-30. E1 and E2 keep
   their he1 windows and are only described (ungraded), so the full timeline can be shown.
3. Output file names `he1_*` become `he3_*`.

Every other rule is unchanged: segment exclusions, at least 100 eligible teams, the 90% equal-score rule,
the definition of O (top band max(10, 2%) against positions 25-50%, scaled by the private IQR of the top
half), the time-shift text pattern, seed 20261117, 10,000 permutations and 5,000 bootstrap resamples. If
the new snapshot renames a column, the rename is the only code change allowed and is listed in the results.

## Data

The first revision of `habedi/kaggle-knowledge-graph` on Hugging Face whose manifest gives a Meta Kaggle
snapshot dated **on or after 2027-07-15**. If that dataset is no longer updated, the same three tables are
taken from Kaggle's official Meta Kaggle files (Competitions, Teams, Submissions) dated on or after
2027-07-15, and the switch is reported. The run does not happen before 2027-07-15.

## Gate and pre-set extension

- **G3:** at least 12 included competitions in E3 and at least 15 in E0. If G3 fails, the run is
  repeated **once**, with E3 extended to deadlines up to 2027-12-31 and a snapshot dated on or after
  2028-01-15. If G3 still fails, results are descriptive only.

## Hypotheses and tests (alpha 0.05, one-sided)

- **H3 (primary):** median O in E3 is greater than median O in E0 (permutation test on the difference of
  medians, era labels shuffled between E0 and E3; bootstrap 95% interval).
- **H3-stratified and H3 without time-shifted contests:** as in he1, with E3 in place of E2.
- **S3:** ret10 and rho100 lower in E3 than in E0, as in he1.
- **P0** (median O above 0 over all included competitions) is reported as in he1.

## How he1 and he3 are read together (fixed now)

| he1 H1 | he3 H3 | Reading |
|---|---|---|
| supported | supported | Growth in top-of-board optimism replicated out of sample. |
| supported | not supported | he1's effect did not replicate; reported as not replicated. |
| not supported | not supported | The holdout still holds in the agent era, in two independent windows. |
| not supported | supported | New effect appearing after November 2026; reported as a first, unreplicated signal. |

The size of the effect (E3 minus E0, with its bootstrap interval) is reported next to he1's in every case.

Our prior (stated before either result): about 35% that H3 is supported, the same as for he1.

## Limits stated in advance

The same limits as he1 (observational; public test sizes not in the data; competition mix changes).
E3 covers only about eight months, so it holds fewer competitions than E2. Competitions still running at
the snapshot date are excluded by the deadline window, not by hand.

## Procedure

Run date on or after 2027-07-15: `python -m unittest -v test_holdout_era` on the frozen commit, apply the
three mechanical changes above in a new file `holdout_era_he3.py` (diff published), run it, and grade by
hand against this file in RESULTS-he3.md.
