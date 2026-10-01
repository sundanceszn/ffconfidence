# Live test of model v0.1 — written 2026-09-30, before any Week 4+ results exist

## What is being tested
`v0.1-ridge-form-usage`: a per-position ridge regression (lambda = 20) that predicts a player's
average half-PPR points over his team's next 4 games from: last-3-game points, season-to-date
PPG, last-3 and last-game carries / targets / pass attempts, games missed in the last 3, and
whether he played last game. Trained on 2018 through the season before the one being predicted.
Scoring: 0.04/pass yd, 4/pass TD, -1/INT, 0.1/rush-rec yd, 6/rush-rec TD, 0.5/reception, -2/fumble lost.

First frozen snapshot: `snapshots/pred_2026_after_wk03.csv` (222 players, made after Week 3,
before any Week 4 game was played). SHA-256 and input-data hash are in its `.manifest.json`.
Commit every snapshot to GitHub the day it is made; the commit timestamp is the proof.

## Rules
1. A snapshot is made each week, after the last game of the week ends and before the next
   week's first kickoff: `python predict_weekly.py --season 2026 --through-week N`.
2. Snapshots are never overwritten. The script refuses to.
3. The model is **not changed** during the test. A changed model gets a new `MODEL_VERSION`,
   its own snapshots, and is judged only on weeks after it was frozen.
4. The thresholds (startable = top 12 QB / 24 RB / 36 WR / 12 TE; "rostered" proxy = top 14 / 30 / 42 / 14
   by season PPG) are fixed now. Changing them after seeing results is not allowed.
5. A row is scored only when all 4 of the player's team's following games are complete.
   Missed games count as 0 points.

## Primary metric
AUC for "finished top-K at his position over the next 4 games", model vs. the season-PPG baseline,
averaged across all scoreable snapshots. RB, WR and TE are the decision positions (the 2021-2025
backtest gains were +0.019, +0.020, +0.029). QB is reported but not gating (backtest gain ~0).

## Verdict, to be applied once snapshots through Week 14 are scoreable (windows complete by Week 18)
- **Pass:** model AUC is above the season-PPG baseline at all three of RB, WR, TE, and the gain is
  at least +0.010 at two or more of them.
- **Fail:** the gain is zero or negative at two or more of RB, WR, TE.
- **Inconclusive:** anything else.

## Secondary metrics (reported, not gating)
Spearman correlation, claim hit@3, drop-right@3, each against last-3-points and season-PPG baselines.

## Limits, stated in advance
- There are at most about 12 snapshots, and their 4-game windows overlap. Differences smaller than
  roughly 0.02 AUC cannot be separated from noise at this sample size. "Pass" means "consistent with
  the backtest", not proof.
- Snapshot 1 (after 3 games) is the weakest: in a backtest at that exact decision point the model
  was only slightly ahead of points-so-far (mean AUC gain over points-so-far: QB +0.015, RB +0.009, WR +0.010, TE +0.012;
  5 seasons each, and the gain was negative in 1-2 of the 5 seasons at every position). Treat early-season picks as low confidence.
- The model has **no injury or news information**. A player who missed his last game (`played_last = 0`)
  gets a prediction that does not know whether he is returning.
- `claim_candidate_top5` and `drop_candidate_bottom5` are **test labels**, not advice. "Rostered" is
  guessed from season PPG, so players who are obviously rostered in a real league (e.g. a star
  with a slow start) can appear as claim candidates. Real claims need your league's actual waiver pool.
- nflverse occasionally corrects past stat lines; scoring uses the data as of scoring time.
