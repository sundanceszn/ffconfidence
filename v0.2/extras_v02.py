"""
Extensions to model v0.2: kickers (K), team defenses (DST) and a pooled FLEX view.
Kept in its own module so the frozen QB/RB/WR/TE code path in model_v02.py is untouched.

Scoring assumptions (common Yahoo-style defaults; change in one place below if your league differs):
  K   : FG 0-39 = 3, 40-49 = 4, 50+ = 5, PAT = 1, missed FG = -1, missed PAT = -1
  DST : sack 1, INT 2, fumble recovery 2, safety 2, blocked kick 2, defensive/special-teams TD 6,
        points allowed 0 -> +10, 1-6 -> +7, 7-13 -> +4, 14-20 -> +1, 21-27 -> 0, 28-34 -> -1, 35+ -> -4
        (points allowed = opponent score minus points the opponent's own defense/special teams scored)

How the existing model machinery is reused:
  the same 'decision after game g' features (points so far, last-3 points, usage, availability) and the same
  ridge (points) + logistic (chance of finishing top-12 among 32) models.  Usage columns are re-mapped:
    K   : carries -> field-goal attempts, targets -> PAT attempts
    DST : carries -> sacks,               targets -> takeaways (INT + fumble recoveries)
  plus K/DST-specific game-environment features from the betting lines (own implied points, opponent implied points,
  total, home, indoor stadium).
"""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import model_v02 as M

KD_TOP = {"K": 12, "DST": 12}
FLEX_TOP = 72          # 24 RB + 36 WR + 12 TE = the core positions' own top-K, pooled
SP150 = "stats_player/stats_player_week_{y}.csv"
TEAMWK = "stats_team/stats_team_week_{y}.csv"

KD_BASE = M.BASE
KD_LINE = ["nx_total", "nx_impl", "nx_opp_impl", "nx_home", "nx_indoor", "nx_impl_x", "nx_opp_impl_x", "nx_total_x"]
KD_SPEC = KD_BASE + KD_LINE


# ------------------------------------------------------------------ downloads
def fetch_sp150(y, cache, refresh=False):
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"sp150_{y}.csv")
    if os.path.exists(f) and not refresh and os.path.getsize(f) > 100000:
        return f
    if M._download(M.BASE_URL + SP150.format(y=y), f, 100000):
        return f
    raise RuntimeError(f"could not download full player stats for {y}")


def fetch_team(y, cache, refresh=False):
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"team_{y}.csv")
    if os.path.exists(f) and not refresh and os.path.getsize(f) > 5000:
        return f
    if M._download(M.BASE_URL + TEAMWK.format(y=y), f, 5000):
        return f
    raise RuntimeError(f"could not download team stats for {y}")


# ------------------------------------------------------------------ scoring
def k_points(d):
    return (3 * (d.fg_made_0_19 + d.fg_made_20_29 + d.fg_made_30_39) + 4 * d.fg_made_40_49
            + 5 * (d.fg_made_50_59 + d.fg_made_60_) + d.pat_made - d.fg_missed - d.pat_missed)


def pa_points(pa):
    return np.select([pa == 0, pa <= 6, pa <= 13, pa <= 20, pa <= 27, pa <= 34], [10, 7, 4, 1, 0, -1], default=-4)


# ------------------------------------------------------------------ game tables
def kd_tables(seasons, cache, sched, refresh_season=None, through=None):
    """Return (kd, cal): kd has the same columns as model_v02.load_weekly output, for pos in {'K','DST'};
    cal maps (season, team, week) -> team game index g (from the DST rows, one per team-game)."""
    games = sched[["season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    hs = games.rename(columns={"home_team": "team", "away_team": "opp", "home_score": "own_score", "away_score": "opp_score"})
    as_ = games.rename(columns={"away_team": "team", "home_team": "opp", "away_score": "own_score", "home_score": "opp_score"})
    sc = pd.concat([hs, as_])[["season", "week", "team", "opp_score"]]
    sc["team"] = sc.team.replace(M.TEAM_FIX)
    dst_frames, k_frames = [], []
    for y in seasons:
        t = pd.read_csv(fetch_team(y, cache, refresh=(y == refresh_season)), low_memory=False)
        t = t[t.season_type == "REG"].copy()
        t["team"] = t.team.replace(M.TEAM_FIX)
        t["opponent_team"] = t.opponent_team.replace(M.TEAM_FIX)
        if through is not None and y == seasons[-1]:
            t = t[t.week <= through]
        t = t.fillna({c: 0 for c in t.columns if t[c].dtype.kind in "fi"})
        opp = t[["season", "week", "team", "def_tds", "special_teams_tds", "def_safeties"]].rename(
            columns={"team": "opponent_team", "def_tds": "o_dtd", "special_teams_tds": "o_std", "def_safeties": "o_saf"})
        t = t.merge(opp, on=["season", "week", "opponent_team"], how="left").merge(sc, on=["season", "week", "team"], how="left")
        pa = (t.opp_score - 6 * (t.o_dtd.fillna(0) + t.o_std.fillna(0)) - 2 * t.o_saf.fillna(0)).clip(lower=0)
        takeaways = t.def_interceptions + t.fumble_recovery_opp
        t["pts"] = (t.def_sacks + 2 * t.def_interceptions + 2 * t.fumble_recovery_opp + 2 * t.def_safeties
                    + 2 * (t.def_punt_blocks + t.def_pat_blocks + t.def_fg_blocks) + 6 * (t.def_tds + t.special_teams_tds)
                    + pa_points(pa))
        dst_frames.append(pd.DataFrame({"season": t.season, "week": t.week, "team": t.team, "player_id": "DST_" + t.team,
                                        "player_display_name": t.team + " D/ST", "pos": "DST", "pts": t.pts,
                                        "carries": t.def_sacks, "targets": takeaways, "attempts": 0.0}))
        p = pd.read_csv(fetch_sp150(y, cache, refresh=(y == refresh_season)), low_memory=False)
        p = p[(p.season_type == "REG") & (p.position == "K")].copy()
        p["team"] = p.team.replace(M.TEAM_FIX)
        if through is not None and y == seasons[-1]:
            p = p[p.week <= through]
        for c in ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
                  "pat_made", "fg_missed", "pat_missed", "fg_att", "pat_att"]:
            p[c] = p[c].fillna(0)
        k_frames.append(pd.DataFrame({"season": p.season, "week": p.week, "team": p.team, "player_id": p.player_id,
                                      "player_display_name": p.player_display_name, "pos": "K", "pts": k_points(p),
                                      "carries": p.fg_att, "targets": p.pat_att, "attempts": 0.0}))
    dst = pd.concat(dst_frames, ignore_index=True)
    dst["g"] = dst.groupby(["season", "team"]).week.transform(lambda s: s.rank(method="dense")).astype(int)
    cal = dst[["season", "team", "week", "g"]].drop_duplicates()
    k = pd.concat(k_frames, ignore_index=True).merge(cal, on=["season", "team", "week"], how="inner")
    return pd.concat([dst, k], ignore_index=True), cal


def schedule_long_kd(sched):
    """Team-game table with lines plus what K/DST need: opponent implied points and an indoor-stadium flag."""
    sl = M.schedule_long(sched)
    roof = pd.concat([sched[["season", "week", "home_team", "roof"]].rename(columns={"home_team": "team"}),
                      sched[["season", "week", "away_team", "roof"]].rename(columns={"away_team": "team"})])
    roof["team"] = roof.team.replace(M.TEAM_FIX)
    roof["indoor"] = roof.roof.isin(["dome", "closed"]).astype(float)
    sl = sl.merge(roof[["season", "week", "team", "indoor"]], on=["season", "week", "team"], how="left")
    sl["opp_impl"] = sl.total_line - sl.impl
    return sl


# ------------------------------------------------------------------ features (same 'decision after game g' scheme)
def decision_rows_kd(d, g_list, label):
    G = int(d.g.max())
    pids = d.player_id.unique()
    ix = {p: i for i, p in enumerate(pids)}
    n = len(pids)
    P = np.zeros((n, G)); C = np.zeros((n, G)); T = np.zeros((n, G)); A = np.zeros((n, G)); R = np.zeros((n, G), bool)
    i = d.player_id.map(ix).values
    gg = d.g.values - 1
    P[i, gg] = d.pts.values; C[i, gg] = d.carries.values; T[i, gg] = d.targets.values
    A[i, gg] = d.attempts.values; R[i, gg] = True
    meta = d.drop_duplicates("player_id").set_index("player_id").loc[pids]
    pos = meta.pos.values
    names = meta.player_display_name.values
    out = []
    for g in g_list:
        if g < 3 or g > G:
            continue
        l3 = slice(g - 3, g)
        c3, t3, a3 = C[:, l3].mean(1), T[:, l3].mean(1), A[:, l3].mean(1)
        seen = R[:, l3].any(1)
        pool = seen & np.where(pos == "K", (c3 + t3) >= 1, True)
        for j in np.where(pool)[0]:
            row = dict(player_id=pids[j], player_name=names[j], pos=pos[j], g=g, pts3=P[j, l3].mean(), std=P[j, :g].mean(),
                       c3=c3[j], t3=t3[j], a3=a3[j], c1=C[j, g - 1], t1=T[j, g - 1], a1=A[j, g - 1],
                       played_last=float(R[j, g - 1]), missed3=3 - R[j, l3].sum())
            if label:
                row["y"] = P[j, g:g + 4].mean()
            out.append(row)
    return out


def build_rows_kd(kd, mode, label):
    rows = []
    for (s, t), d in kd.groupby(["season", "team"]):
        G = int(d.g.max())
        gl = range(3, G - 3) if mode == "train" else [G]
        for r in decision_rows_kd(d, gl, label):
            r.update(season=s, team=t)
            rows.append(r)
    return pd.DataFrame(rows)


def attach_kd(df, cal, sl):
    df = df.copy()
    df = df.merge(cal.rename(columns={"week": "wk_g"}), on=["season", "team", "g"], how="left").sort_values("wk_g").reset_index(drop=True)
    s2 = sl[["season", "team", "week"]].sort_values("week")
    df = pd.merge_asof(df, s2.rename(columns={"week": "wk_next"}), left_on="wk_g", right_on="wk_next", by=["season", "team"],
                       direction="forward", allow_exact_matches=False)
    nx = sl.rename(columns={"week": "wk_next", "opp": "nx_opp_team", "total_line": "nx_total", "impl": "nx_impl",
                            "home": "nx_home", "opp_impl": "nx_opp_impl", "indoor": "nx_indoor"})
    df = df.merge(nx, on=["season", "team", "wk_next"], how="left")
    df["lines_missing"] = df.nx_total.isna()
    return df


def finish_kd(df, med):
    df = df.copy()
    for c in ["nx_total", "nx_impl", "nx_opp_impl"]:
        df[c] = df[c].fillna(med[c])
    df["nx_home"] = df.nx_home.fillna(0.5)
    df["nx_indoor"] = df.nx_indoor.fillna(0.0)
    for c in ["nx_impl", "nx_opp_impl", "nx_total"]:
        df[c + "_x"] = df[c] * df.pts3
    return df


def build_training_kd(kd_train, cal, sl):
    df = attach_kd(build_rows_kd(kd_train, "train", True), cal, sl)
    med = {c: float(df[c].median()) for c in ["nx_total", "nx_impl", "nx_opp_impl"]}
    df = finish_kd(df, med)
    df["rk_y"] = df.groupby(["season", "g", "pos"]).y.rank(ascending=False, method="first")
    df["top"] = (df.rk_y <= df.pos.map(KD_TOP)).astype(int)
    return df, med


def fit_kd(panel):
    models = {}
    for pos in KD_TOP:
        d = panel[panel.pos == pos]
        X = d[KD_SPEC].values.astype(float)
        sc = StandardScaler().fit(X)
        models[pos] = dict(ridge=M.ridge_fit(X, d.y.values), scaler=sc, n=len(d), base_rate=float(d.top.mean()),
                           logit=LogisticRegression(C=M.LOGIT_C, max_iter=2000).fit(sc.transform(X), d.top.values))
    return models


def predict_kd(models, live):
    live = live.copy()
    live["pred_ppg_next4"] = np.nan
    live["p_startable"] = np.nan
    for pos in KD_TOP:
        m = live.pos == pos
        if not m.any():
            continue
        X = live.loc[m, KD_SPEC].values.astype(float)
        live.loc[m, "pred_ppg_next4"] = M.ridge_pred(models[pos]["ridge"], X)
        live.loc[m, "p_startable"] = models[pos]["logit"].predict_proba(models[pos]["scaler"].transform(X))[:, 1]
    live["rank_pred_pos"] = live.groupby("pos").pred_ppg_next4.rank(ascending=False, method="first").astype(int)
    live["rank_std_ppg_pos"] = live.groupby("pos")["std"].rank(ascending=False, method="first").astype(int)
    return live


# ------------------------------------------------------------------ FLEX (pooled RB/WR/TE)
def flex_feats(df, run):
    df = df.copy()
    df["is_rb"] = (df.pos == "RB").astype(float)
    df["is_wr"] = (df.pos == "WR").astype(float)
    cols = list(M.SPECS[run]) + ["is_rb", "is_wr"]
    for b in ["std", "pts3", "c3", "t3"]:
        for p in ("rb", "wr"):
            df[f"{b}_{p}"] = df[b] * df[f"is_{p}"]
            cols.append(f"{b}_{p}")
    return df, cols


def fit_flex(panel, run):
    d = panel[panel.pos.isin(["RB", "WR", "TE"])].copy()
    d["rk_flex"] = d.groupby(["season", "g"]).y.rank(ascending=False, method="first")
    d["top_flex"] = (d.rk_flex <= FLEX_TOP).astype(int)
    d, cols = flex_feats(d, run)
    X = d[cols].values.astype(float)
    sc = StandardScaler().fit(X)
    return dict(ridge=M.ridge_fit(X, d.y.values), scaler=sc, cols=cols, n=len(d), base_rate=float(d.top_flex.mean()),
                logit=LogisticRegression(C=M.LOGIT_C, max_iter=2000).fit(sc.transform(X), d.top_flex.values)), d


def predict_flex(model, live, run):
    d = live[live.pos.isin(["RB", "WR", "TE"])].copy()
    d, cols = flex_feats(d, run)
    X = d[cols].values.astype(float)
    d["flex_pred"] = M.ridge_pred(model["ridge"], X)
    d["flex_p_pool"] = model["logit"].predict_proba(model["scaler"].transform(X))[:, 1]
    d["flex_p"] = M.recenter(d["flex_p_pool"].values, FLEX_TOP * M.FLEX_RECALL)
    d["flex_rank"] = d.flex_pred.rank(ascending=False, method="first").astype(int)
    return d[["team", "player_id", "flex_pred", "flex_p", "flex_p_pool", "flex_rank"]]
