# Weekly player outlook (model v0.2)

A static website plus a weekly job that refreshes it. No manual inputs are needed for the model.

## One-time setup (about 10 minutes)
1. Create a new GitHub repository and push **the contents of this folder** to it (so `site/`, `v0.2/` and `.github/` are at the repo root).
2. GitHub: Settings > Actions > General > Workflow permissions > **Read and write permissions**. Save.
3. Netlify: Add new site > Import from Git > pick the repo. Build command: leave empty. Publish directory: `site` (already set in `netlify.toml`).
4. GitHub: Actions tab > **weekly-run** > Run workflow (leave on `auto`). This checks the whole setup end to end. If the week's data is not out yet it exits quietly and the next scheduled attempt picks it up.

## What the site shows
Tabs for QB, RB, WR, TE, FLEX (RB/WR/TE ranked together), and beta tabs for K and DST. Each player has a chance of finishing startable, predicted points, the next opponent and betting total, and injury/flag badges. Under each list, a **Not modeled** card shows injured or returning players the model has no estimate for, with their status. A **Track record** tab fills in as snapshots are graded.

## What runs by itself (UTC)
- **Tuesday 15:00, 21:00, Wednesday 09:00:** Tuesday-type run (form, usage, betting lines). The first attempt that finds the previous week's stats published wins; later ones do nothing.
- **Friday 22:00, Saturday 03:00 and 15:00:** Friday-type run, which adds the final injury report. The last attempt proceeds even if the report looks incomplete.
- Each run saves an immutable snapshot in `snapshots/<season>/`, updates `site/data/`, and grades any older snapshots whose four-game windows are complete (`results/`, Track record tab).
- The commit triggers a Netlify redeploy automatically.
- **If data is still missing at the last attempt** of a run type (Wednesday 09:00 or Saturday 15:00), the job fails on purpose so GitHub can notify you. `site/data/status.json` always records why a run was skipped, and the site shows a warning if the numbers are stale.

## What stays manual (optional, benchmark only)
FantasyPros has no public feed. If you want the FantasyPros comparison, drop that week's FLX export into `v0.1/fp_weekly/` named `flx_2026_wkNN.csv` before the week's first kickoff. The site and model never depend on it.

## Audit and revision
Before deployment a data audit changed six things (probability calibration, TE forecasts, trade handling, scoring, postponed games, version pins). Details and evidence: `v0.2/PREREGISTRATION_ADDENDUM_2_audit.md`. The superseded first Week 4 files are kept in `snapshots/2026/superseded_pre_audit/`.

## Rules that keep the record honest
Never edit or delete a snapshot. Do not change the model during the season (see `v0.2/PREREGISTRATION_v0.2.md`). The pass/fail test is already written down there.

## Files
- `v0.1/` the frozen first model, its Week 4 snapshot, rules and FantasyPros Week 4 files.
- `v0.2/` models (v0.2, v0.3 and the K/DST/FLEX extension), weekly runner, scorer, and all written rules.
- `snapshots/` every weekly prediction, each with a hash manifest.
- `site/` the website (`index.html` + JSON it reads).

## Run it by hand
`pip install -r requirements.txt`, then `python v0.2/run_weekly.py --auto`. Backfill a past week with `--season 2026 --through-week 5 --run tue`.

## Two model versions, one upload
- **v0.2** (main, frozen): form, usage, betting lines, and the final injury report on Fridays.
- **v0.3** (runs alongside, QB/RB/WR/TE only): v0.2 plus last season's points per game. The backtest says it helps WR and TE and early-season QB, and does nothing for RB. The site has a model toggle on those four tabs and the Track record tab compares the two on identical weeks.
Both started with the Week 4 snapshot, before kickoff. Written rules: `v0.2/PREREGISTRATION_v0.2.md`, `..._ADDENDUM_1.md` (FLEX/K/DST), `PREREGISTRATION_v0.3.md`.

## Best use
Backtest (2021-25): the top-3 waiver picks beat a season-average rule by about +1.4 (QB, p=0.003) and +1.6 (RB, p=0.001) points per game over the next four games, ahead in all 5 seasons. WR (+1.1, p=0.11) and TE (+0.5, p=0.29) were ahead on average but not reliably. Choosing starters among players you already own was NOT better than a season average at RB (about -1 point per slot) and no better at WR; treat the app as a claims and hold/drop tool, not a start/sit tool.

## Up/down arrows
From Week 5 the page shows a small ▲/▼ plus the number of places a player moved since the previous week's run of the same type. The weekly job writes `site/data/previous.json` (the earlier run's chances) and the page does the comparison for whichever tab, sort and model you are viewing. Arrows show only when the chance moved 10+ points and only in the "chance startable" sort. Backtest: a typical 3-4 place move is only about 3 chance points, and moves did not predict who beat or missed the forecast, so this is descriptive, not a signal. Week 4 (the first snapshot) has no arrows.

## Lookahead-line log
Each run also records the betting lines already posted for the next few weeks (`lines_log/`, committed automatically). It does not affect any prediction. It exists so we can later test, with honest data, whether knowing lines for games 2-4 ahead helps (a best-case backtest says mainly at QB). Plan in `v0.2/PREREGISTRATION_ADDENDUM_3_lookahead_lines.md`. Do not move it into `snapshots/`: the scorer reads every csv there.

## Honest limits
The backtest edge over simple rules is small (about +0.01 to +0.04 AUC depending on position); K and DST are the hardest to predict. The model knows nothing about scheme, depth charts or breaking news. Early-season outputs (3-4 games of data) are the least reliable. Trust the Track record tab over the backtest as it fills in.

## How data moves (and what happens when it doesn't arrive)
**Sources (all public nflverse files, no keys):** weekly player stats, weekly team stats (for defenses), injury reports, and the schedule with betting lines. Each run downloads fresh copies on a clean runner; nothing is cached between runs, and downloads time out and retry.
**Before anything is frozen, each run checks:**
- the week's stats are published for every team that played (otherwise it waits and retries);
- on Fridays, the injury report is actually final: 90+ players with a game status on 24+ teams. Wednesday practice-only rows do not count (the last Saturday attempt can proceed with a partial report and says so on the site);
- team-level stats are published before the K/DST/FLEX files are written (otherwise only the QB/RB/WR/TE files are written and the rest follow on a retry);
- sanity checks: normal-sized player pools, no missing predictions, betting lines present, all teams present. If a check fails nothing is frozen.
**Where results go:** `snapshots/<season>/` (immutable; written atomically; never overwritten), `site/data/` (what the website reads), `results/` (grading). Every run records the SHA-256 of its input files in the manifest. `site/data/status.json` always says what the latest check did and why.
**If something is wrong:** retries happen automatically; the last attempt of each run type fails the job on purpose so GitHub notifies you. The commit step runs even after a failure, so a snapshot that was already written is never lost.
**Off-season:** GitHub may pause scheduled workflows in a public repo after about 60 days without activity. In September, open the Actions tab and re-enable `weekly-run` if it shows as disabled.
