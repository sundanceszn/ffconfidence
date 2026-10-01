# Addendum 3: lookahead-line log (data collection only)

Written 2026-10-01, before Week 4 kickoff. **Nothing about the predictions changes.** Models v0.2 and v0.3, their
features, their frozen snapshots and the website are exactly as before.

## What is new
Every time the weekly job runs, it records the betting lines already posted for the decision week and the next four
weeks, and when each line first appeared or moved (`lines_log/lookahead_lines_<season>.csv`, a change log: a row is
added only when a game's lines are new or different). The log is data only. It is kept outside `snapshots/` because
the scorer reads every csv there, and it is wrapped so a failure in it cannot affect a run.

## Why
The backtest cannot say how much lines for games 2-4 ahead would help, because history keeps closing lines, which
already include later news (for example a QB injury in the previous game). Best-case ceiling from that flawed data:
QB +0.029 AUC (game 2 alone +0.015), RB about 0, WR +0.004, TE +0.003. The real gain is smaller by an unknown
amount, so the honest answer has to come from lines as actually posted at decision time.

## Tests run before adding it
- Original code vs new code on the full path with identical inputs: every regenerated snapshot is byte-identical,
  core predictions differ by 0.0, website data identical. Only a creation timestamp differs.
- Re-running adds no duplicate rows. A deliberately broken log folder did not change the run's outcome or exit code.
- The commit step works with an empty log folder (a placeholder file is shipped for this) and with a populated one.

## Analysis plan (fixed now, before any data exists)
- **One primary question, QB only:** does adding the average implied points of the team's games 2-3 ahead (lines
  posted at decision time, converted with the model's own implied-points function) improve top-12 AUC over v0.2?
- **Descriptive only, no claims:** RB/WR/TE; how far ahead lines are typically posted; how much lines move between
  first posting and the decision week.
- **When to look:** not before 24 decision snapshots in which the needed lines were posted for at least 70% of QBs.
  This probably means two seasons. "Cannot tell yet" is an acceptable outcome.
- **Shipping rule:** HAC-adjusted p < 0.05 AND gain of at least +0.010 AUC AND positive in a majority of seasons.
  If met, it runs as a parallel series (like v0.3) for a further season. It never replaces v0.2 mid-season.
