# Addendum 1 to the live test — written 2026-09-30, before any Week 4 game was played

## What was added
FantasyPros Week 4 projections (RB/WR/TE and QB exports), supplied by the user at 17:44 UTC on
2026-09-30, before the first Week 4 kickoff (2026-10-01). Stored unmodified in `fp_weekly/` with
SHA-256 hashes in `fp_weekly/wk04.manifest.json`. The "week 4" label comes from the user; the file
contents fit a 32-team week but cannot prove which week it is.

## Head-to-head design (fixed now, before results)
- **Population:** players at RB/WR/TE/QB present in both our frozen snapshot and the FantasyPros file
  (217 of 222 snapshot players match).
- **Rankings compared:** (1) our `pred_ppg_next4`, (2) FantasyPros Week 4 projected points,
  (3) season-to-date PPG (`std`).
- **Outcome A (like-for-like for FantasyPros):** actual points in Week 4 alone.
- **Outcome B (our model's horizon):** average points over the team's next 4 games (games 4-7).
  FantasyPros' single-week number is used only as a ranking here.
- **Metrics:** Spearman and AUC for "finished top-K at the position" (same K as before), per position.
- **Reported separately:** players with `played_last = 0` in the snapshot. FantasyPros uses injury and
  news information; our model does not, so mixing these cases would hide what is being compared.
- **Status:** descriptive and exploratory. It does NOT change the pass/fail rule in
  PREREGISTRATION.md, which still compares our model with the season-PPG baseline.
- **Blend test** (average of the two rankings vs. each alone) will be run only after at least four
  weekly FantasyPros files exist, and its result will be labeled exploratory.

## Observed before any results (agreement only, not accuracy)
Spearman between our snapshot prediction and FantasyPros Week 4: RB 0.91, WR 0.76, TE 0.69, QB 0.64.
Points-so-far vs FantasyPros: RB 0.84, WR 0.74, TE 0.60, QB 0.64.
The largest disagreements are mostly players who missed their last game (e.g. Nacua and Collins:
FantasyPros top-10 WR, our model far lower) — the injury blind spot already documented. Trevor Lawrence
(FantasyPros rank ~5 at QB vs. our 15) and Breece Hall (FantasyPros ~58 at RB vs. our 17) are the other
notable splits. One snapshot player, Baker Mayfield, is absent from the FantasyPros QB file; the reason is unknown.

## Going forward
One FantasyPros export per week, saved before that week's first kickoff, named `flx_2026_wkNN.csv`
(QB export optional). Never replace a file once saved.
