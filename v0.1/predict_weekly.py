#!/usr/bin/env python3
"""
Fantasy drop/hold/claim model v0.1  (ridge on form + usage, half-PPR).

Usage:
    python predict_weekly.py --season 2026 --through-week 3
Writes snapshots/pred_{season}_after_wk{NN}.csv and a .manifest.json next to it.

RULES (do not break these, they are what makes the record trustworthy):
  * Run it BEFORE the next week's first kickoff, with --through-week = last COMPLETED week.
  * Never overwrite a snapshot. Change the model => change MODEL_VERSION and start a new log.
"""
import argparse, hashlib, json, os, sys, urllib.request, datetime as dt
import numpy as np, pandas as pd

MODEL_VERSION = "v0.1-ridge-form-usage"
BASE = "https://github.com/nflverse/nflverse-data/releases/download/"
PATHS = ["player_stats/stats_player_week_{y}.csv", "stats_player/stats_player_week_{y}.csv"]
SCHED_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
FEATS = ["pts3", "std", "c3", "t3", "a3", "c1", "t1", "a1", "missed3", "played_last"]
LAMBDA = 20.0
TRAIN_FIRST = 2018
K = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}        # "startable" = top-K at position (12-team assumption)
ROST = {"QB": 14, "RB": 30, "WR": 42, "TE": 14}     # "rostered" proxy = top-N by season-to-date PPG
POS_MAP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def fetch_season(y, cache, refresh=False):
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"weekly_{y}.csv")
    if os.path.exists(f) and not refresh and os.path.getsize(f) > 100000: return f
    last = None
    for p in PATHS:
        try:
            tmp = f + ".tmp"; urllib.request.urlretrieve(BASE + p.format(y=y), tmp)
            if os.path.getsize(tmp) > 100000: os.replace(tmp, f); return f
        except Exception as e: last = e
    raise RuntimeError(f"could not download {y}: {last}")

def load_weekly(seasons, cache, refresh_season=None, through=None):
    frames = []
    for y in seasons:
        d = pd.read_csv(fetch_season(y, cache, refresh=(y == refresh_season)), low_memory=False)
        d = d[d.season_type == "REG"].copy()
        if through is not None and y == seasons[-1]: d = d[d.week <= through]
        fl = d[["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]].fillna(0).sum(axis=1)
        d["pts"] = (0.04*d.passing_yards + 4*d.passing_tds - d.passing_interceptions + 0.1*d.rushing_yards
                    + 6*d.rushing_tds + 0.5*d.receptions + 0.1*d.receiving_yards + 6*d.receiving_tds - 2*fl)
        d["pos"] = d.position.map(POS_MAP)
        frames.append(d[["season","week","team","player_id","player_display_name","pos","pts","carries","targets","attempts"]]
                      .dropna(subset=["pos"]))
    w = pd.concat(frames)
    w[["carries","targets","attempts"]] = w[["carries","targets","attempts"]].fillna(0)
    w["g"] = w.groupby(["season","team"]).week.transform(lambda s: s.rank(method="dense")).astype(int)
    return w

def team_rows(d, g_list, label):
    """Feature rows for one team-season at each decision point g (decision is made AFTER game g)."""
    G = int(d.g.max()); pids = d.player_id.unique(); ix = {p: i for i, p in enumerate(pids)}; n = len(pids)
    P = np.zeros((n, G)); C = np.zeros((n, G)); T = np.zeros((n, G)); A = np.zeros((n, G)); R = np.zeros((n, G), bool)
    i = d.player_id.map(ix).values; gg = d.g.values - 1
    P[i,gg] = d.pts.values; C[i,gg] = d.carries.values; T[i,gg] = d.targets.values; A[i,gg] = d.attempts.values; R[i,gg] = True
    meta = d.drop_duplicates("player_id").set_index("player_id").loc[pids]
    pos = meta.pos.values; names = meta.player_display_name.values; out = []
    for g in g_list:
        if g < 3 or g > G: continue
        l3 = slice(g-3, g)
        c3, t3, a3 = C[:,l3].mean(1), T[:,l3].mean(1), A[:,l3].mean(1)
        pool = R[:,l3].any(1) & np.where(pos == "QB", a3 >= 15, (c3 + t3) >= 3)
        for j in np.where(pool)[0]:
            row = dict(player_id=pids[j], player_name=names[j], pos=pos[j], g=g, pts3=P[j,l3].mean(), std=P[j,:g].mean(),
                       c3=c3[j], t3=t3[j], a3=a3[j], c1=C[j,g-1], t1=T[j,g-1], a1=A[j,g-1],
                       played_last=float(R[j,g-1]), missed3=3-R[j,l3].sum())
            if label: row["y"] = P[j,g:g+4].mean()
            out.append(row)
    return out

def train_panel(w):
    rows = []
    for (s, t), d in w.groupby(["season","team"]):
        G = int(d.g.max())
        for r in team_rows(d, range(3, G-3), True): r.update(season=s, team=t); rows.append(r)
    return pd.DataFrame(rows)

def ridge_fit(X, y, lam=LAMBDA):
    mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd; ym = y.mean()
    return mu, sd, np.linalg.solve(Z.T @ Z + lam*np.eye(Z.shape[1]), Z.T @ (y - ym)), ym
def ridge_pred(m, X): mu, sd, b, ym = m; return ((X - mu) / sd) @ b + ym

def fit_models(panel):
    return {p: ridge_fit(panel[panel.pos == p][FEATS].values.astype(float), panel[panel.pos == p].y.values) for p in K}

def predict_current(w_cur, models):
    rows = []
    for t, d in w_cur.groupby("team"):
        G = int(d.g.max())
        for r in team_rows(d, [G], False): r["team"] = t; r["season"] = int(d.season.iloc[0]); rows.append(r)
    df = pd.DataFrame(rows)
    df["pred_ppg_next4"] = np.nan
    for p in K:
        m = df.pos == p
        df.loc[m, "pred_ppg_next4"] = ridge_pred(models[p], df.loc[m, FEATS].values.astype(float))
    return df

def add_flags(df):
    df = df.copy(); df["rank_pred_pos"] = df.groupby("pos").pred_ppg_next4.rank(ascending=False, method="first").astype(int)
    df["rank_std_ppg_pos"] = df.groupby("pos")["std"].rank(ascending=False, method="first").astype(int)
    df["rostered_proxy"] = df.rank_std_ppg_pos <= df.pos.map(ROST)
    df["rank_pred_waiver"] = np.nan; df["rank_pred_rostered_from_bottom"] = np.nan
    for p in K:
        wv = (df.pos == p) & ~df.rostered_proxy; ro = (df.pos == p) & df.rostered_proxy
        df.loc[wv, "rank_pred_waiver"] = df[wv].pred_ppg_next4.rank(ascending=False, method="first")
        df.loc[ro, "rank_pred_rostered_from_bottom"] = df[ro].pred_ppg_next4.rank(ascending=True, method="first")
    df["claim_candidate_top5"] = df.rank_pred_waiver <= 5
    df["drop_candidate_bottom5"] = df.rank_pred_rostered_from_bottom <= 5
    return df

def schedule_check(season, through):
    try:
        tmp = "/tmp/_games.csv"; urllib.request.urlretrieve(SCHED_URL, tmp)
        g = pd.read_csv(tmp, usecols=["season","game_type","week","gameday","away_score","home_score"])
        n = g[(g.season == season) & (g.game_type == "REG") & (g.week == through + 1)]
        return dict(next_week=through+1, next_week_games=len(n), first_gameday=str(n.gameday.min()),
                    next_week_games_already_scored=int(n.away_score.notna().sum()))
    except Exception as e: return dict(error=str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True); ap.add_argument("--through-week", type=int, required=True)
    ap.add_argument("--cache", default="data_cache"); ap.add_argument("--out", default="snapshots")
    ap.add_argument("--allow-overwrite", action="store_true")
    a = ap.parse_args()
    out_csv = os.path.join(a.out, f"pred_{a.season}_after_wk{a.through_week:02d}.csv")
    if os.path.exists(out_csv) and not a.allow_overwrite: sys.exit(f"REFUSING to overwrite {out_csv}")
    os.makedirs(a.out, exist_ok=True)
    train_seasons = list(range(TRAIN_FIRST, a.season))
    w_train = load_weekly(train_seasons, a.cache)
    w_cur = load_weekly([a.season], a.cache, refresh_season=a.season, through=a.through_week)
    sched = schedule_check(a.season, a.through_week)
    panel = train_panel(w_train); models = fit_models(panel)
    df = add_flags(predict_current(w_cur, models))
    df["snapshot_model"] = MODEL_VERSION; df["through_week"] = a.through_week
    cols = ["snapshot_model","season","through_week","team","player_id","player_name","pos","g","pred_ppg_next4","rank_pred_pos",
            "rostered_proxy","claim_candidate_top5","drop_candidate_bottom5","rank_pred_waiver","rank_pred_rostered_from_bottom",
            "pts3","std","rank_std_ppg_pos","c3","t3","a3","c1","t1","a1","missed3","played_last"]
    df = df.sort_values(["pos","rank_pred_pos"])[cols]
    df.round(4).to_csv(out_csv, index=False)
    raw = os.path.join(a.cache, f"weekly_{a.season}.csv")
    wk = w_cur.groupby("team").week.max()
    manifest = dict(model_version=MODEL_VERSION, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        season=a.season, through_week=a.through_week, teams_with_data_through_that_week=int((wk == a.through_week).sum()),
        max_week_in_input=int(w_cur.week.max()), rows=len(df), rows_by_pos=df.pos.value_counts().to_dict(),
        train_seasons=train_seasons, train_rows_by_pos=panel.pos.value_counts().to_dict(), features=FEATS, ridge_lambda=LAMBDA,
        assumptions=dict(startable_top_K=K, rostered_proxy_top_N_by_season_ppg=ROST, scoring="half-PPR, 4pt pass TD, -1 INT, -2 fumble lost"),
        schedule_check=sched, input_sha256=sha256(raw), output_sha256=sha256(out_csv),
        pandas=pd.__version__, numpy=np.__version__)
    with open(out_csv.replace(".csv", ".manifest.json"), "w") as f: json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
