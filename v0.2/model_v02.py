"""
Fantasy model v0.2 -- form + usage + betting lines (+ final injury report on the Friday run).
Everything here is built from public nflverse data so it can run unattended every week.

Outputs per player, for the next 4 team games:
  pred_ppg_next4 : predicted average half-PPR points per game (ridge)
  p_startable    : probability of finishing top-K at his position over those 4 games (logistic)
Scoring: 0.04/pass yd, 4/pass TD, -1/INT, 0.1/rush-rec yd, 6/rush-rec TD, 0.5/reception, -2/fumble lost.
"""
import hashlib, os, socket, time, urllib.request
import numpy as np, pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

socket.setdefaulttimeout(90)   # a stalled download must fail and retry, not hang the whole job

VERSION = "v0.2"
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/"
WEEKLY_PATHS = ["player_stats/stats_player_week_{y}.csv", "stats_player/stats_player_week_{y}.csv"]
INJ_PATH = "injuries/injuries_{y}.csv"
SCHED_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
TRAIN_FIRST = 2018
K = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}        # "startable" = top-K at position over the next 4 games
ROST = {"QB": 14, "RB": 30, "WR": 42, "TE": 14}     # display context only, never used for advice
POS_MAP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
# franchise moves / alternate codes -> one code per franchise, applied to stats AND schedule so joins never silently fail
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}
RIDGE_LAMBDA = 20.0
RIDGE_LAMBDA_POS = {"TE": 400.0}   # TE point forecasts were over-spread at 20 (out-of-sample slope 0.82 -> 0.90); RMSE and AUC no worse
LOGIT_C = 0.1
# share of the TRUE top-K finishers at each position that are inside our player pool (2021-25 backtest).  The pool misses
# emerging backups and players just back from injury, so a pool player's real chance is a bit lower than the pool-relative
# probability the logistic model learns.  Probabilities are re-centred so they add up to K x recall.
RECALL = {"QB": 0.94, "RB": 0.94, "WR": 0.92, "TE": 0.855}
FLEX_RECALL = 0.918   # K-weighted average of the three above (24 RB, 36 WR, 12 TE)


def lam(pos):
    return RIDGE_LAMBDA_POS.get(pos, RIDGE_LAMBDA)


def recenter(p, target):
    """Shift probabilities on the log-odds scale so they sum to `target`; ranking is unchanged."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    f = lambda d: (1 / (1 + np.exp(-(z + d)))).sum() - target
    try:
        d = brentq(f, -8, 8)
    except ValueError:
        d = 0.0
    return 1 / (1 + np.exp(-(z + d)))

BASE = ["pts3", "std", "c3", "t3", "a3", "c1", "t1", "a1", "missed3", "played_last"]
LINE = ["nx_total", "nx_impl", "nx_home", "nx_impl_x", "nx_total_x"]
INJ = ["cu_out", "cu_dbt", "cu_que", "cu_dnp", "cu_lim", "recent3", "cu_out_x", "cu_dbt_x", "cu_que_x", "cu_dnp_x",
       "nx_out", "nx_dbt", "nx_que", "nx_dnp", "nx_lim", "nx_out_x", "nx_dbt_x", "nx_que_x", "nx_dnp_x"]
SPECS = {"tue": BASE + LINE, "fri": BASE + LINE + INJ}


# ------------------------------------------------------------------ downloads
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download(url, dest, min_bytes, tries=3):
    for i in range(tries):
        try:
            tmp = dest + ".tmp"
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) >= min_bytes:
                os.replace(tmp, dest)
                return True
        except Exception:
            pass
        time.sleep(2 * (i + 1))
    return False


def fetch_weekly(y, cache, refresh=False):
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"weekly_{y}.csv")
    if os.path.exists(f) and not refresh and os.path.getsize(f) > 100000:
        return f
    for p in WEEKLY_PATHS:
        if _download(BASE_URL + p.format(y=y), f, 100000):
            return f
    raise RuntimeError(f"could not download weekly stats for {y}")


def fetch_injuries(y, cache, refresh=False):
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, f"injuries_{y}.csv")
    if os.path.exists(f) and not refresh and os.path.getsize(f) > 1000:
        return f
    if _download(BASE_URL + INJ_PATH.format(y=y), f, 1000):
        return f
    raise RuntimeError(f"could not download injuries for {y}")


def fetch_schedule(cache, refresh=True):
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, "games.csv")
    if os.path.exists(f) and not refresh:
        return f
    if _download(SCHED_URL, f, 100000):
        return f
    if os.path.exists(f):
        return f
    raise RuntimeError("could not download schedule")


# ------------------------------------------------------------------ loaders
def load_weekly(seasons, cache, refresh_season=None, through=None):
    frames = []
    for y in seasons:
        d = pd.read_csv(fetch_weekly(y, cache, refresh=(y == refresh_season)), low_memory=False)
        d = d[d.season_type == "REG"].copy()
        if through is not None and y == seasons[-1]:
            d = d[d.week <= through]
        fl = d[["sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"]].fillna(0).sum(axis=1)
        d["pts"] = (0.04 * d.passing_yards + 4 * d.passing_tds - d.passing_interceptions + 0.1 * d.rushing_yards
                    + 6 * d.rushing_tds + 0.5 * d.receptions + 0.1 * d.receiving_yards + 6 * d.receiving_tds - 2 * fl)
        d["pos"] = d.position.map(POS_MAP)
        d["team"] = d.team.replace(TEAM_FIX)
        frames.append(d[["season", "week", "team", "player_id", "player_display_name", "pos", "pts",
                         "carries", "targets", "attempts"]].dropna(subset=["pos"]))
    w = pd.concat(frames, ignore_index=True)
    w[["carries", "targets", "attempts"]] = w[["carries", "targets", "attempts"]].fillna(0)
    w["g"] = w.groupby(["season", "team"]).week.transform(lambda s: s.rank(method="dense")).astype(int)
    return w


def load_schedule(cache, refresh=True):
    g = pd.read_csv(fetch_schedule(cache, refresh=refresh))
    return g[g.game_type == "REG"].copy()


def schedule_long(g):
    """One row per team-game with opponent, home flag, total line and implied team points.
    nflverse spread_line > 0 means the HOME team is favoured (checked: corr with home margin = +0.45)."""
    h = g[["season", "week", "home_team", "away_team", "total_line", "spread_line"]].rename(
        columns={"home_team": "team", "away_team": "opp"})
    h["home"] = 1
    h["impl"] = (h.total_line + h.spread_line) / 2
    a = g[["season", "week", "away_team", "home_team", "total_line", "spread_line"]].rename(
        columns={"away_team": "team", "home_team": "opp"})
    a["home"] = 0
    a["impl"] = (a.total_line - a.spread_line) / 2
    out = pd.concat([h, a], ignore_index=True)[["season", "week", "team", "opp", "home", "total_line", "impl"]]
    out["team"] = out.team.replace(TEAM_FIX)
    out["opp"] = out.opp.replace(TEAM_FIX)
    return out


def load_injuries(seasons, cache, refresh_season=None, max_week=None):
    fr = []
    for y in seasons:
        d = pd.read_csv(fetch_injuries(y, cache, refresh=(y == refresh_season)), low_memory=False)
        d = d[d.game_type == "REG"]
        if max_week is not None and y == seasons[-1]:
            d = d[d.week <= max_week]
        fr.append(d)
    inj = pd.concat(fr, ignore_index=True)
    inj["out"] = (inj.report_status == "Out").astype(int)
    inj["dbt"] = (inj.report_status == "Doubtful").astype(int)
    inj["que"] = (inj.report_status == "Questionable").astype(int)
    inj["dnp"] = (inj.practice_status == "Did Not Participate In Practice").astype(int)
    inj["lim"] = (inj.practice_status == "Limited Participation in Practice").astype(int)
    inj["any"] = inj[["out", "dbt", "que"]].max(axis=1)
    return (inj.groupby(["season", "week", "gsis_id"])[["out", "dbt", "que", "dnp", "lim", "any"]].max()
            .reset_index().rename(columns={"gsis_id": "player_id"}))


# ------------------------------------------------------------------ features
def decision_rows(d, g_list, label):
    """Feature rows for one team-season; a decision is made AFTER game g using only games <= g."""
    G = int(d.g.max())
    pids = d.player_id.unique()
    ix = {p: i for i, p in enumerate(pids)}
    n = len(pids)
    P = np.zeros((n, G)); C = np.zeros((n, G)); T = np.zeros((n, G)); A = np.zeros((n, G))
    R = np.zeros((n, G), bool)
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
        pool = R[:, l3].any(1) & np.where(pos == "QB", a3 >= 15, (c3 + t3) >= 3)
        for j in np.where(pool)[0]:
            row = dict(player_id=pids[j], player_name=names[j], pos=pos[j], g=g, pts3=P[j, l3].mean(),
                       std=P[j, :g].mean(), c3=c3[j], t3=t3[j], a3=a3[j], c1=C[j, g - 1], t1=T[j, g - 1],
                       a1=A[j, g - 1], played_last=float(R[j, g - 1]), missed3=3 - R[j, l3].sum())
            if label:
                row["y"] = P[j, g:g + 4].mean()
            out.append(row)
    return out


def build_rows(w, mode, label):
    rows = []
    for (s, t), d in w.groupby(["season", "team"]):
        G = int(d.g.max())
        gl = range(3, G - 3) if mode == "train" else [G]
        for r in decision_rows(d, gl, label):
            r.update(season=s, team=t)
            rows.append(r)
    return pd.DataFrame(rows)


def attach_context(df, w, sl, inj):
    df = df.copy()
    tg = w.groupby(["season", "team", "g"]).week.first().rename("wk_g").reset_index()
    df = df.merge(tg, on=["season", "team", "g"], how="left").sort_values("wk_g").reset_index(drop=True)
    s2 = sl[["season", "team", "week"]].sort_values("week")
    df = pd.merge_asof(df, s2.rename(columns={"week": "wk_next"}), left_on="wk_g", right_on="wk_next",
                       by=["season", "team"], direction="forward", allow_exact_matches=False)
    nx = sl.rename(columns={"week": "wk_next", "opp": "nx_opp_team", "total_line": "nx_total",
                            "impl": "nx_impl", "home": "nx_home"})
    df = df.merge(nx, on=["season", "team", "wk_next"], how="left")
    df["lines_missing"] = df.nx_total.isna()
    if inj is not None:
        def flags(weekcol, prefix):
            m = df[["season", "player_id", weekcol]].merge(
                inj.rename(columns={"week": weekcol}), on=["season", "player_id", weekcol], how="left")
            for c in ["out", "dbt", "que", "dnp", "lim"]:
                df[f"{prefix}_{c}"] = m[c].fillna(0).values
        flags("wk_next", "nx")
        flags("wk_g", "cu")
        rec = np.zeros(len(df))
        for off in (0, 1, 2):
            p = df[["season", "player_id"]].copy()
            p["wk"] = df.wk_g - off
            m = p.merge(inj.rename(columns={"week": "wk"}), on=["season", "player_id", "wk"], how="left")
            rec += m["any"].fillna(0).values
        df["recent3"] = rec
    else:
        for pre in ("nx", "cu"):
            for c in ["out", "dbt", "que", "dnp", "lim"]:
                df[f"{pre}_{c}"] = 0.0
        df["recent3"] = 0.0
    return df


def finish_features(df, medians):
    df = df.copy()
    for c in ["nx_total", "nx_impl"]:
        df[c] = df[c].fillna(medians[c])
    df["nx_home"] = df["nx_home"].fillna(0.5)
    for c in ["nx_impl", "nx_total"]:
        df[c + "_x"] = df[c] * df.pts3
    for pre in ("nx", "cu"):
        for s in ("out", "dbt", "que", "dnp"):
            df[f"{pre}_{s}_x"] = df[f"{pre}_{s}"] * df.pts3
    return df


def drop_trade_rows(df, w):
    """Remove training rows that a mid-season trade corrupts: (a) the next-4-games label counts his games for another team as
    zeros, or (b) he joined this team within the last 3 games, so his history here is truncated."""
    mt = w.groupby(["season", "player_id"]).team.nunique()
    multi = mt[mt > 1].reset_index()[["season", "player_id"]]
    sub = df.merge(multi, on=["season", "player_id"])[["season", "team", "player_id", "g", "wk_g"]]
    if sub.empty:
        return df
    other = w[["season", "player_id", "team", "week"]].rename(columns={"team": "team2", "week": "wk2"})
    m = sub.merge(other, on=["season", "player_id"])
    m = m[m.team2 != m.team]
    label_bad = m[(m.wk2 > m.wk_g) & (m.wk2 <= m.wk_g + 5)][["season", "team", "player_id", "g"]].drop_duplicates()
    first = w.groupby(["season", "team", "player_id"]).g.min().rename("first_g").reset_index()
    j = sub.merge(first, on=["season", "team", "player_id"])
    had_other_before = m[m.wk2 <= m.wk_g][["season", "team", "player_id", "g"]].drop_duplicates()
    hist_bad = j[j.first_g > j.g - 3][["season", "team", "player_id", "g"]].merge(had_other_before, on=["season", "team", "player_id", "g"])
    bad = pd.concat([label_bad, hist_bad]).drop_duplicates()
    bad["_bad"] = 1
    out = df.merge(bad, on=["season", "team", "player_id", "g"], how="left")
    return out[out._bad.isna()].drop(columns="_bad").reset_index(drop=True)


def build_training(w_train, sl, inj):
    df = attach_context(build_rows(w_train, "train", True), w_train, sl, inj)
    df = drop_trade_rows(df, w_train)
    med = {c: float(df[c].median()) for c in ["nx_total", "nx_impl"]}
    df = finish_features(df, med)
    grp = ["season", "g", "pos"]
    df["rk_y"] = df.groupby(grp).y.rank(ascending=False, method="first")
    df["top"] = (df.rk_y <= df.pos.map(K)).astype(int)
    return df, med


# ------------------------------------------------------------------ models
def ridge_fit(X, y, lam=RIDGE_LAMBDA):
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Z = (X - mu) / sd
    ym = y.mean()
    return mu, sd, np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym)), ym


def ridge_pred(m, X):
    mu, sd, b, ym = m
    return ((X - mu) / sd) @ b + ym


def fit_models(panel, run):
    feats = SPECS[run]
    models = {}
    for pos in K:
        d = panel[panel.pos == pos]
        X = d[feats].values.astype(float)
        sc = StandardScaler().fit(X)
        models[pos] = dict(ridge=ridge_fit(X, d.y.values, lam(pos)), scaler=sc, n=len(d), base_rate=float(d.top.mean()),
                           logit=LogisticRegression(C=LOGIT_C, max_iter=2000).fit(sc.transform(X), d.top.values))
    return models


def predict(models, live, run):
    feats = SPECS[run]
    live = live.copy()
    live["pred_ppg_next4"] = np.nan
    live["p_startable"] = np.nan
    live["p_pool"] = np.nan
    for pos in K:
        m = live.pos == pos
        if not m.any():
            continue
        X = live.loc[m, feats].values.astype(float)
        live.loc[m, "pred_ppg_next4"] = ridge_pred(models[pos]["ridge"], X)
        live.loc[m, "p_pool"] = models[pos]["logit"].predict_proba(models[pos]["scaler"].transform(X))[:, 1]
        live.loc[m, "p_startable"] = recenter(live.loc[m, "p_pool"].values, K[pos] * RECALL[pos])
    live["rank_pred_pos"] = live.groupby("pos").pred_ppg_next4.rank(ascending=False, method="first").astype(int)
    live["rank_std_ppg_pos"] = live.groupby("pos")["std"].rank(ascending=False, method="first").astype(int)

    def status(r):
        return "Out" if r.nx_out else "Doubtful" if r.nx_dbt else "Questionable" if r.nx_que else ""
    live["injury_next"] = live.apply(status, axis=1) if run == "fri" else ""
    return live
