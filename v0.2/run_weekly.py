#!/usr/bin/env python3
"""
Weekly runner for model v0.2.  Designed to be run by a scheduler (GitHub Actions) with no human input.

    python v0.2/run_weekly.py --auto                      # what the scheduler runs
    python v0.2/run_weekly.py --season 2026 --through-week 3 --run tue     # manual / backfill

What it does
  1. Works out the season and the last COMPLETED week from the public schedule (all games scored).
  2. Checks that nflverse has published that week's stats (and, for the Friday run, the injury report).
     If not, it exits without writing anything (status.json says why); a later scheduled attempt picks it up.
     With --final-attempt it exits with an ERROR instead, so the scheduler shows a failed run.
  3. Trains on earlier seasons only, predicts the next 4 games for every player in the pool.
  4. Writes immutable snapshots (never overwrites), manifests with file hashes, and the website data:
       core snapshot       QB / RB / WR / TE              (frozen spec, written first)
       extension snapshots K and DST models, pooled FLEX  (separate files; the core file is never touched)
       unmodeled list      players with no recent games (injured / returning), shown for hold/drop context

Run types
  tue : form + usage + betting lines.        (waiver-time run; earlier-in-week injury data added nothing in backtests)
  fri : tue + the final injury report.       (backtest showed injury data only helps once statuses are final)
"""
import argparse, datetime as dt, json, os, sys
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_v02 as M
import extras_v02 as X
import v03 as V3

MIN_INJ_TEAMS = 24
MIN_STATUS_ROWS = 90          # a COMPLETE week has ~105-150 players with a game status (Out/Doubtful/Questionable); practice-only reports have none
POOL_BOUNDS = {"QB": (24, 40), "RB": (55, 95), "WR": (65, 110), "TE": (22, 45), "K": (28, 42), "DST": (28, 32)}   # history: QB 27-36, RB 65-82, WR 75-100, TE 25-40, K 31-38, DST 30-32


class NotReady(Exception):
    pass


def save_csv(df, path):
    """Write atomically: a crash can never leave a half-written snapshot behind."""
    tmp = path + ".tmp"
    df.round(4).to_csv(tmp, index=False)
    os.replace(tmp, path)


def sanity_problems(df, with_lines=True):
    """Cheap checks that catch a bad data pull before it is frozen into an immutable snapshot."""
    bad = []
    for pos, (lo, hi) in POOL_BOUNDS.items():
        n = int((df.pos == pos).sum())
        if n and not lo <= n <= hi:
            bad.append(f"{pos} pool has {n} players (expected {lo}-{hi})")
    if df[["pred_ppg_next4", "p_startable"]].isna().any().any():
        bad.append("missing predictions")
    if with_lines and "lines_missing" in df and df.lines_missing.mean() > 0.25:
        bad.append(f"betting lines missing for {df.lines_missing.mean():.0%} of rows")
    if df.team.nunique() < 30:
        bad.append(f"only {df.team.nunique()} teams present")
    return bad
CORE_COLS = ["team", "player_id", "player_name", "pos", "g", "nx_opp_team", "nx_home", "nx_total", "nx_impl", "lines_missing",
             "injury_next", "pred_ppg_next4", "p_startable", "p_pool", "rank_pred_pos", "pts3", "std", "rank_std_ppg_pos",
             "c3", "t3", "a3", "c1", "t1", "a1", "missed3", "played_last"]
KD_COLS = ["team", "player_id", "player_name", "pos", "g", "nx_opp_team", "nx_home", "nx_total", "nx_impl", "nx_opp_impl",
           "nx_indoor", "lines_missing", "pred_ppg_next4", "p_startable", "rank_pred_pos", "pts3", "std", "rank_std_ppg_pos",
           "c3", "t3", "c1", "t1", "missed3", "played_last"]


def log(*a):
    print("[run_weekly]", *a, flush=True)


def detect_context(sched, now=None):
    """Return (season, through_week, next_week) from the schedule, or None if nothing to do.
    A week counts as complete when every game is scored, or when 75%+ are scored and each remaining game has been
    postponed to 3+ days from now (so one postponed game cannot freeze the pipeline)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    scored = sched.assign(done=sched.home_score.notna() & sched.away_score.notna())
    season = int(scored[scored.done].season.max()) if scored.done.any() else None
    if season is None:
        return None
    s = scored[scored.season == season]
    horizon = pd.Timestamp(now.date()) + pd.Timedelta(days=3)

    def complete(g):
        if g.done.all():
            return True
        late = pd.to_datetime(g[~g.done].gameday, errors="coerce")
        return bool(g.done.mean() >= 0.75 and late.notna().all() and (late >= horizon).all())
    wk_done = s.groupby("week").apply(complete)
    full = wk_done[wk_done].index
    if len(full) == 0:
        return None
    through = int(full.max())
    later = sorted(set(s.week) - set(range(1, through + 1)))
    return season, through, (int(later[0]) if later else None)


def first_kickoff_utc(sched, season, week):
    g = sched[(sched.season == season) & (sched.week == week)]
    if g.empty:
        return None, 0
    et = ZoneInfo("America/New_York")
    ks = []
    for _, r in g.iterrows():
        try:
            ks.append(dt.datetime.strptime(f"{r.gameday} {r.gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=et)
                      .astimezone(dt.timezone.utc))
        except Exception:
            pass
    return (min(ks) if ks else None), int((g.home_score.notna() & g.away_score.notna()).sum())


def unmodeled_players(w_cur, pool_keys, season, cache, upto_week):
    """Players with meaningful usage earlier this season but no stat line in their team's last 3 games
    (injured, just back, or benched).  The model gives no estimate for them; show status for hold/drop context."""
    w = w_cur.copy()
    G = w.groupby("team").g.max()
    gp = w.groupby(["player_id", "team", "pos", "player_display_name"]).agg(
        games=("week", "nunique"), carries=("carries", "sum"), targets=("targets", "sum"), attempts=("attempts", "sum"),
        pts=("pts", "sum"), last_g=("g", "max"), last_week=("week", "max")).reset_index()
    skill = (gp.pos != "QB") & ((gp.carries + gp.targets) / gp.games >= 5)
    qb = (gp.pos == "QB") & (gp.attempts / gp.games >= 15)
    gp = gp[skill | qb].copy()
    gp = gp[[(r.player_id, r.team) not in pool_keys for r in gp.itertuples()]].copy()
    gp["team_games_missed"] = gp.team.map(G) - gp.last_g
    gp = gp[gp.team_games_missed >= 3].copy()
    gp["season_ppg"] = gp.pts / gp.games
    gp["injury_status"] = ""
    gp["injury_detail"] = ""
    try:
        raw = pd.read_csv(M.fetch_injuries(season, cache))
        raw = raw[(raw.game_type == "REG") & (raw.week <= upto_week)].sort_values("week")
        last = raw.groupby("gsis_id").tail(1).set_index("gsis_id")
        for i, r in gp.iterrows():
            if r.player_id in last.index:
                x = last.loc[r.player_id]
                st = x.report_status if isinstance(x.report_status, str) else ""
                pr = str(x.practice_status) if isinstance(x.practice_status, str) else ""
                gp.at[i, "injury_status"] = st or pr.replace(" in Practice", "").replace(" In Practice", "")
                gp.at[i, "injury_detail"] = (str(x.report_primary_injury) if isinstance(x.report_primary_injury, str) else "") + \
                                            f" (week {int(x.week)})"
    except Exception as e:
        log(f"injury context for unmodeled players unavailable: {e}")
    out = gp.rename(columns={"player_display_name": "player_name"})[
        ["player_id", "player_name", "pos", "team", "games", "season_ppg", "last_week", "team_games_missed",
         "injury_status", "injury_detail"]]
    return out.sort_values("season_ppg", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--season", type=int)
    ap.add_argument("--through-week", type=int)
    ap.add_argument("--run", choices=["tue", "fri"])
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--asof", action="store_true", help="historical simulation: skip readiness checks, use cached schedule")
    ap.add_argument("--no-site", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="Friday run: proceed even if the injury report looks incomplete")
    ap.add_argument("--final-attempt", action="store_true", help="if data is still not ready, fail loudly (non-zero exit)")
    a = ap.parse_args()
    cache = a.cache or os.path.join(a.root, "data_cache")
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    site_data = os.path.join(a.root, "site", "data")
    ctx = {}

    def finish(outcome, msg, code=0):
        log(msg)
        if not a.no_site:
            os.makedirs(site_data, exist_ok=True)
            with open(os.path.join(site_data, "status.json"), "w") as f:
                json.dump(dict(checked_utc=now.isoformat(), outcome=outcome, message=msg, **ctx), f, indent=1)
        return code

    def not_ready(outcome, msg):
        if a.final_attempt:
            return finish("failed_" + outcome, "FINAL ATTEMPT, data still not ready: " + msg, 1)
        return finish(outcome, msg + "; will retry later")

    sched = M.load_schedule(cache, refresh=not a.asof)
    if a.auto:
        c = detect_context(sched, now)
        if c is None:
            return finish("nothing_to_do", "no completed games found")
        season, through, nxt = c
        if nxt is None:
            return finish("season_complete", f"season {season} complete after week {through}")
    else:
        season, through = a.season, a.through_week
        if season is None or through is None:
            sys.exit("need --auto or both --season and --through-week")
        later = sorted(set(sched[sched.season == season].week) - set(range(1, through + 1)))
        nxt = int(later[0]) if later else None
    run = a.run or ("fri" if now.weekday() in (4, 5) else "tue")
    ctx.update(season=season, through_week=through, decision_for_week=nxt, run=run)
    if through < 3:
        return finish("too_early", f"only {through} completed weeks; the model needs at least 3")
    if nxt is None:
        return finish("nothing_to_do", "no next week in the schedule")

    # Lookahead-line log (data only; see PREREGISTRATION_ADDENDUM_3). Runs before the "already exists" exit so every
    # retry can notice newly posted lines. Wrapped so it can never change or block a run.
    if not a.asof:
        try:
            import lookahead_log
            n_new = lookahead_log.record(sched, season, nxt, run, now, a.root)
            log(f"lookahead-line log: {n_new} new or changed line rows recorded")
        except Exception as e:
            log("lookahead-line log skipped (no effect on snapshots):", repr(e))

    out_dir = os.path.join(a.root, "snapshots", str(season))
    stem = os.path.join(out_dir, f"after_wk{through:02d}_{run}_{M.VERSION}")
    core_csv, kd_csv, flex_csv, um_csv = stem + ".csv", stem + "_kd.csv", stem + "_flex.csv", stem + "_unmodeled.csv"
    core_exists = os.path.exists(core_csv)
    ext_exists = os.path.exists(kd_csv) and os.path.exists(flex_csv)
    stem3 = os.path.join(out_dir, f"after_wk{through:02d}_{run}_{V3.VERSION}")
    v3_csv = stem3 + ".csv"
    v3_exists = os.path.exists(v3_csv)
    if core_exists and ext_exists and v3_exists:
        return finish("already_exists", f"snapshots for week {through} ({run}) already exist, left untouched")

    # ---- readiness: has nflverse published what we need?
    train_seasons = list(range(M.TRAIN_FIRST, season))
    if not a.asof:
        M.fetch_weekly(season, cache, refresh=True)
    w_cur = M.load_weekly([season], cache, through=through)
    if not a.asof:
        played = sched[(sched.season == season) & (sched.week == through) & sched.home_score.notna() & sched.away_score.notna()]
        need = set(played.home_team.replace(M.TEAM_FIX)) | set(played.away_team.replace(M.TEAM_FIX))
        have = set(w_cur[w_cur.week == through].team)
        if not need <= have:
            return not_ready("waiting_for_stats", f"stats for week {through} not fully published (missing teams: {sorted(need - have)})")
    unknown = set(w_cur.team) - set(M.schedule_long(sched).team)
    if unknown:
        sys.exit(f"team codes in stats but not in schedule: {sorted(unknown)}")

    inj_live, inj_teams = None, None
    inj_status_rows = inj_status_teams = None
    inj_final = None
    if run == "fri":
        if not a.asof:
            M.fetch_injuries(season, cache, refresh=True)
        inj_live = M.load_injuries([season], cache, max_week=nxt)
        raw_inj = pd.read_csv(M.fetch_injuries(season, cache))
        nx_rows = raw_inj[(raw_inj.game_type == "REG") & (raw_inj.week == nxt)]
        inj_teams = int(nx_rows.team.nunique())
        inj_status_rows = int(nx_rows.report_status.notna().sum())
        inj_status_teams = int(nx_rows[nx_rows.report_status.notna()].team.nunique())
        inj_final = inj_status_rows >= MIN_STATUS_ROWS and inj_status_teams >= MIN_INJ_TEAMS
        if not a.asof and not inj_final and not a.allow_partial:
            return not_ready("waiting_for_injuries",
                             f"injury report for week {nxt} is not final yet ({inj_status_rows} players with a game status on {inj_status_teams} teams; "
                             f"a final report has {MIN_STATUS_ROWS}+ on {MIN_INJ_TEAMS}+)")
        if not inj_final:
            log(f"WARNING: proceeding with a partial injury report ({inj_status_rows} statuses) because of --allow-partial / asof")
    elif not a.asof:
        try:
            M.fetch_injuries(season, cache, refresh=True)     # display context only for the unmodeled list
        except Exception as e:
            log(f"could not refresh injuries (display only): {e}")

    # ---- core (QB/RB/WR/TE): train on earlier seasons only, then predict
    sl = M.schedule_long(sched)
    w_train = M.load_weekly(train_seasons, cache)
    inj_train = M.load_injuries(train_seasons, cache) if run == "fri" else None
    panel, med = M.build_training(w_train, sl, inj_train)
    models = M.fit_models(panel, run)
    live = M.build_rows(w_cur, "live", False)
    live = M.finish_features(M.attach_context(live, w_cur, sl, inj_live), med)
    res = M.predict(models, live, run)

    problems = sanity_problems(res)
    if problems:
        msg = "; ".join(problems)
        if a.final_attempt:
            return finish("failed_sanity", "FINAL ATTEMPT, data failed sanity checks, nothing frozen: " + msg, 1)
        return finish("sanity_failed", "data failed sanity checks, nothing frozen (will retry): " + msg)

    kick, scored_next = first_kickoff_utc(sched, season, nxt)
    created = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    before_kick = None if a.asof else bool(kick and created < kick)

    def add_meta(df):
        df.insert(0, "model_version", M.VERSION)
        df.insert(1, "run", run)
        df.insert(2, "season", season)
        df.insert(3, "through_week", through)
        df.insert(4, "decision_for_week", nxt)

    snap = res[CORE_COLS].sort_values(["pos", "rank_pred_pos"]).copy()
    add_meta(snap)
    os.makedirs(out_dir, exist_ok=True)
    raw_inputs = {"weekly_stats": M.sha256(M.fetch_weekly(season, cache)), "schedule": M.sha256(M.fetch_schedule(cache, refresh=False))}
    if run == "fri":
        raw_inputs["injuries"] = M.sha256(M.fetch_injuries(season, cache))
    core_diff = None
    if not core_exists:
        save_csv(snap, core_csv)
        manifest = dict(
            model_version=M.VERSION, run=run, season=season, through_week=through, decision_for_week=nxt,
            created_utc=created.isoformat(), first_kickoff_of_decision_week_utc=kick.isoformat() if kick else None,
            created_before_first_kickoff=before_kick, games_of_decision_week_already_scored=scored_next, asof_simulation=a.asof,
            rows=len(snap), rows_by_pos=snap.pos.value_counts().to_dict(), lines_missing_rows=int(snap.lines_missing.sum()),
            injury_report_teams_for_decision_week=inj_teams, injury_status_rows=inj_status_rows, injury_status_teams=inj_status_teams,
            injury_final=inj_final, train_seasons=train_seasons,
            train_rows_by_pos={p: models[p]["n"] for p in models}, features=M.SPECS[run], ridge_lambda=M.RIDGE_LAMBDA,
            logit_C=M.LOGIT_C, startable_top_K=M.K, input_sha256=raw_inputs, snapshot_sha256=M.sha256(core_csv),
            pandas=pd.__version__, numpy=np.__version__)
        with open(stem + ".manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"wrote {os.path.relpath(core_csv, a.root)} ({len(snap)} players)")
    else:
        frozen = pd.read_csv(core_csv)
        m = frozen.merge(snap.round(4), on=["player_id", "team"], suffixes=("_f", "_n"))
        core_diff = float((m.pred_ppg_next4_f - m.pred_ppg_next4_n).abs().max()) if len(m) else None
        log(f"core snapshot already exists and is untouched; recomputed predictions differ from it by at most {core_diff}")
        try:
            manifest = json.load(open(stem + ".manifest.json"))
        except FileNotFoundError:
            manifest = dict(model_version=M.VERSION, run=run, season=season, through_week=through, decision_for_week=nxt,
                            created_utc=None, snapshot_sha256=M.sha256(core_csv), rows=len(frozen))
            log("WARNING: core snapshot had no manifest (interrupted run?); rebuilt a minimal one in memory")

    # ---- v0.3 (adds last season's points per game): its own series, written alongside v0.2
    snap3 = None
    if not v3_exists:
        try:
            pt = V3.prior_table(w_train)
            panel3, med3 = V3.add_prior(panel[panel.season > M.TRAIN_FIRST], pt)
            models3 = V3.fit(panel3, run)
            live3, _ = V3.add_prior(live, pt, med3)
            res3 = V3.predict(models3, live3, run)
            snap3 = res3[CORE_COLS].sort_values(["pos", "rank_pred_pos"]).copy()
            add_meta(snap3)
            snap3["model_version"] = V3.VERSION
            save_csv(snap3, v3_csv)
            m3 = dict(model_version=V3.VERSION, run=run, season=season, through_week=through, decision_for_week=nxt,
                      created_utc=created.isoformat(), created_before_first_kickoff=before_kick, asof_simulation=a.asof,
                      rows=len(snap3), train_seasons=list(range(M.TRAIN_FIRST + 1, season)),
                      train_rows_by_pos={p: models3[p]["n"] for p in models3}, features=V3.spec(run),
                      ridge_lambda=M.RIDGE_LAMBDA, logit_C=M.LOGIT_C, startable_top_K=M.K, input_sha256=raw_inputs,
                      rookie_rows=int(live3.rookie.sum()), snapshot_sha256=M.sha256(v3_csv))
            with open(stem3 + ".manifest.json", "w") as f:
                json.dump(m3, f, indent=2)
            log(f"wrote {os.path.relpath(v3_csv, a.root)} ({len(snap3)} players, {m3['rookie_rows']} without a prior season)")
        except Exception as e:
            log(f"v0.3 FAILED (v0.2 is unaffected): {type(e).__name__}: {e}")
            if a.final_attempt:
                return finish("failed_v03", f"v0.3 failed: {e}", 1)

    # ---- extension: FLEX (pooled RB/WR/TE) and the K / DST models
    ext_ok = False
    ext_pending = None
    kd_snap = fx = kd_models = flex_model = None
    if not ext_exists:
        try:
            sl_kd = X.schedule_long_kd(sched)
            kd_train, cal_train = X.kd_tables(train_seasons, cache, sched)
            kd_panel, kd_med = X.build_training_kd(kd_train, cal_train, sl_kd)
            kd_models = X.fit_kd(kd_panel)
            kd_cur, cal_cur = X.kd_tables([season], cache, sched, refresh_season=None if a.asof else season, through=through)
            if not a.asof:
                have_dst = set(kd_cur[(kd_cur.pos == "DST") & (kd_cur.week == through)].team)
                if not need <= have_dst:
                    raise NotReady(f"team-level stats for week {through} not published yet (missing: {sorted(need - have_dst)})")
            live_kd = X.finish_kd(X.attach_kd(X.build_rows_kd(kd_cur, "live", False), cal_cur, sl_kd), kd_med)
            kd_res = X.predict_kd(kd_models, live_kd)
            flex_model, _ = X.fit_flex(panel, run)
            flex = X.predict_flex(flex_model, res, run)
            kd_bad = sanity_problems(kd_res, with_lines=True)
            if kd_bad:
                raise NotReady("K/DST data failed sanity checks: " + "; ".join(kd_bad))
            kd_snap = kd_res[KD_COLS].sort_values(["pos", "rank_pred_pos"]).copy()
            add_meta(kd_snap)
            fx = res[["team", "player_id", "player_name", "pos"]].merge(flex, on=["team", "player_id"]).sort_values("flex_rank")
            add_meta(fx)
            save_csv(kd_snap, kd_csv)
            save_csv(fx, flex_csv)
            ext_ok = True
            log(f"wrote extension snapshots: {len(kd_snap)} K/DST rows, {len(fx)} FLEX rows")
        except NotReady as e:
            ext_pending = str(e)
            log(f"extension not written yet (core is unaffected): {e}")
        except Exception as e:
            ext_pending = f"{type(e).__name__}: {e}"
            log(f"EXTENSION FAILED (core snapshot is unaffected): {ext_pending}")

    pool_keys = {(r.player_id, r.team) for r in res.itertuples()}
    um = unmodeled_players(w_cur, pool_keys, season, cache, nxt if run == "fri" else through)
    if not os.path.exists(um_csv):
        um.round(3).to_csv(um_csv, index=False)

    if ext_ok:
        ext_manifest = dict(
            model_version=M.VERSION + "-ext", run=run, season=season, through_week=through, decision_for_week=nxt,
            created_utc=created.isoformat(), created_before_first_kickoff=before_kick, asof_simulation=a.asof,
            core_snapshot_sha256=manifest.get("snapshot_sha256"), core_recompute_max_abs_diff=core_diff,
            kd_rows=len(kd_snap), flex_rows=len(fx), kd_train_rows={p: kd_models[p]["n"] for p in kd_models},
            flex_train_rows=flex_model["n"], kd_features=X.KD_SPEC, kd_top_K=X.KD_TOP, flex_top=X.FLEX_TOP,
            scoring_assumptions="K: FG 3/4/5 by distance, PAT 1, miss -1. DST: sack 1, INT 2, fumble rec 2, safety 2, block 2, TD 6, points-allowed tiers 10/7/4/1/0/-1/-4",
            sha256=dict(kd=M.sha256(kd_csv), flex=M.sha256(flex_csv), unmodeled=M.sha256(um_csv)), unmodeled_rows=len(um),
            input_sha256=dict(player_stats_full=M.sha256(X.fetch_sp150(season, cache)), team_stats=M.sha256(X.fetch_team(season, cache))))
        with open(stem + "_ext.manifest.json", "w") as f:
            json.dump(ext_manifest, f, indent=2)

    if not a.no_site:
        os.makedirs(site_data, exist_ok=True)
        core = pd.read_csv(core_csv)
        rows = core.merge(res[["team", "player_id", "wk_next"]], on=["team", "player_id"], how="left")
        if os.path.exists(v3_csv):
            v3 = pd.read_csv(v3_csv)[["team", "player_id", "pred_ppg_next4", "p_startable"]].rename(
                columns={"pred_ppg_next4": "v03_pred", "p_startable": "v03_p"})
            rows = rows.merge(v3, on=["team", "player_id"], how="left")
        if os.path.exists(flex_csv):
            rows = rows.merge(pd.read_csv(flex_csv)[["team", "player_id", "flex_pred", "flex_p", "flex_rank"]],
                              on=["team", "player_id"], how="left")
        sl_played = sched[(sched.season == season) & sched.home_score.notna() & sched.away_score.notna()]
        played_keys = set(zip(sl_played.home_team.replace(M.TEAM_FIX), sl_played.week)) | \
                      set(zip(sl_played.away_team.replace(M.TEAM_FIX), sl_played.week))
        rows["next_game_played"] = [bool((t, w) in played_keys) for t, w in zip(rows.team, rows.wk_next.fillna(-1).astype(int))]
        multi = w_cur.groupby("player_id").team.nunique()
        rows["recent_team_change"] = rows.player_id.map(multi).fillna(1).gt(1)
        parts = [rows]
        if os.path.exists(kd_csv):
            k = pd.read_csv(kd_csv)
            k["injury_next"] = ""
            k["next_game_played"] = False
            k["recent_team_change"] = False
            parts.append(k)
        allrows = pd.concat(parts, ignore_index=True)
        v01 = os.path.join(a.root, "v0.1", "snapshots", f"pred_{season}_after_wk{through:02d}.csv")
        if os.path.exists(v01):
            old = pd.read_csv(v01)[["player_id", "team", "pred_ppg_next4"]].rename(columns={"pred_ppg_next4": "v01_pred"})
            allrows = allrows.merge(old, on=["player_id", "team"], how="left")
        drop = [c for c in ["model_version", "run", "season", "through_week", "decision_for_week", "lines_missing", "wk_next"] if c in allrows]
        allrows = allrows.drop(columns=drop)
        meta = {k: manifest[k] for k in ["model_version", "run", "season", "through_week", "decision_for_week", "created_utc",
                                        "created_before_first_kickoff", "rows", "injury_report_teams_for_decision_week"] if k in manifest}
        meta["extension"] = os.path.exists(kd_csv)
        meta["weeks_left"] = int(sched[sched.season == season].week.max()) - nxt + 1
        meta["injury_final"] = manifest.get("injury_final")
        meta["models"] = ["v0.2"] + (["v0.3"] if os.path.exists(v3_csv) else [])
        payload = {"meta": meta,
                   "players": json.loads(allrows.round(3).to_json(orient="records")),
                   "unmodeled": json.loads(um.round(2).to_json(orient="records"))}
        with open(os.path.join(site_data, "latest.json"), "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        hp = os.path.join(site_data, "history.json")
        hist = json.load(open(hp)) if os.path.exists(hp) else []
        try:   # reference for the page's up/down arrows (data only; wrapped so it can never affect a run)
            import rank_moves
            n_prev = rank_moves.build(a.root, hist, season, nxt, run, V3.VERSION, os.path.join(site_data, "previous.json"))
            log(f"rank-move reference: {n_prev} players from the previous week's {run} run" if n_prev else "rank-move reference: no earlier snapshot, no arrows")
        except Exception as e:
            log("rank-move reference skipped (no effect on predictions):", repr(e))
        if not any(h.get("file") == os.path.relpath(core_csv, a.root) for h in hist):
            hist.append(dict(meta, file=os.path.relpath(core_csv, a.root)))
            json.dump(hist, open(hp, "w"), indent=1)
        try:
            import score_weekly
            score_weekly.score_all(a.root, cache)
        except Exception as e:
            log(f"scoring step skipped: {e}")
    if ext_pending and a.final_attempt:
        return finish("failed_extension", f"FINAL ATTEMPT: core snapshots written, but K/DST/FLEX extension could not be built: {ext_pending}", 1)
    if ext_pending:
        return finish("extension_pending", f"core snapshots written; K/DST/FLEX extension pending ({ext_pending}); will retry later")
    return finish("written" if not core_exists else "written_extension", f"done: week {through} {run} run")


if __name__ == "__main__":
    sys.exit(main())
