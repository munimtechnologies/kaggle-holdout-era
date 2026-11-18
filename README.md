# kaggle-holdout-era

**Has the public leaderboard on Kaggle become a worse guide to the private one in the era of AI coding
agents?** An open, pre-registered test on Meta Kaggle data from 2020 to November 2026, with a
noise-adjusted second test, a fixed-format natural experiment on the Playground Series, and a
prospective replication on competitions that have not ended yet.

By MV2 of Munim, Inc. (mv2@munimtech.com). Made with AI agents working in his name. Code: Apache-2.0.

## Why

Kaggle shows scores on part of the test data during a competition (the public leaderboard) and ranks
on the hidden rest at the end (the private leaderboard). Roelofs et al. (NeurIPS 2019) found little
adaptive overfitting to public leaderboards up to 2019. Since then, competitors have gained chat
assistants (late 2022) and agentic coding tools that can write and submit many variants (widespread from
2025). If any benchmark shows test-set reuse hurting in the agent era, Kaggle's should.

## What is measured

For each competition, the public-leaderboard top band (top 2%, at least 10 teams) is compared with
mid-table teams: how much more the top teams' public scores overstate their private scores, scaled by the
spread of private scores (metric **O**). Competitions with deadlines before ChatGPT (2020 to Nov 2022)
are compared with 2025 to 2026. Rank shake-up (top-10 retention, top-100 rank correlation) and the link
with submissions per team are secondary. Every rule, threshold and test is fixed in
[PREREGISTRATION-he1.md](PREREGISTRATION-he1.md) before the data were downloaded.

## Noise-adjusted test (he2)

O also grows when a competition's public split is small and noisy, even if nobody overfits (the
winner's curse). [PREREGISTRATION-he2.md](PREREGISTRATION-he2.md), written before any he1 result existed,
fixes a second metric **X**: how far the top band's public-private gap lies above a straight line drawn
through mid-table teams. If skill and noise are roughly normal and nobody adapts to the public board, that
line predicts the top band at any noise level, so X stays near 0; selection on the public board that is
stronger at the top pushes X up. The job first checks this on simulated leaderboards (gate C0), then
tests whether X rose from 2020-2022 to 2025-2026. It also fixes how he1 and he2 are read together.
Code: [he2_excess.py](he2_excess.py), which reuses the he1 data code from commit `15dfd9a` unchanged.

## Playground Series natural experiment (he4)

he1 and he2 compare different competitions in different years, and the competition mix changed. Kaggle's
Playground Series has run a similar short tabular contest almost every month since 2021, right through the
arrival of chat assistants and coding agents. [PREREGISTRATION-he4.md](PREREGISTRATION-he4.md), written
before any he1 or he2 result existed, tests whether X rises with the deadline date within this one stable
format (primary), with robustness checks that drop time-shifted episodes and the 2021-2022 Tabular
Playground format. Code: [he4_playground.py](he4_playground.py), stdlib only; it reads he2's
per-competition table and nothing else.

## Prospective replication (he3)

[PREREGISTRATION-he3.md](PREREGISTRATION-he3.md) fixes the same test, with the code frozen at commit
`15dfd9a`, for competitions with deadlines from 2026-11-06 to 2027-06-30, whose private leaderboards were
not yet revealed when it was written. It runs on a Meta Kaggle snapshot dated on or after 2027-07-15. It
also fixes in advance how the he1 and he3 results are read together.

## Status

- 2026-11-17: he1 pre-registered; analysis queued. No results yet.
- 2026-11-17: he3 (prospective replication) pre-registered, before any he1 result. Runs on or after
  2027-07-15.
- 2026-11-17: he2 (noise-adjusted excess optimism) pre-registered, before any he1 result; queued after he1.
- 2026-11-18: he4 (Playground Series trend) pre-registered, before any he1 or he2 result; runs on he2's
  output once he2 has finished.

## Data

Meta Kaggle snapshot 2026-11-12 as packaged by
[habedi/kaggle-knowledge-graph](https://huggingface.co/datasets/habedi/kaggle-knowledge-graph)
(pinned revision, CC BY-NC-SA 4.0, derived from Kaggle's Meta Kaggle). The per-competition output tables
inherit CC BY-NC-SA 4.0. No personal data are used: only competition-level and team-rank fields and
submission scores.

## Run it

```
pip install -r requirements.txt
python -m unittest -v test_holdout_era test_he2_excess test_he4_playground
python holdout_era.py --data-dir ./data --out-dir ./out
python fetch_frozen.py ./out/frozen
python he2_excess.py --data-dir ./data --out-dir ./out --frozen-dir ./out/frozen
python he4_playground.py --he2-csv ./out/he2_competitions.csv --calibration ./out/he2_calibration.json --out-dir ./out
```

Downloads about 41 MB. Needs about 2 GB of memory and a few minutes of CPU.

## Licence

Code: Apache-2.0 (see LICENSE). Outputs derived from Meta Kaggle: CC BY-NC-SA 4.0.
