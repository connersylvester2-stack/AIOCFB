#!/usr/bin/env python3
"""SCRIPT 1 - Power-rank every FBS team from current-season results.

How it works
  * Every completed game gives two observations: "team X scored P points against defense Y".
  * A ridge regression solves for each team's OFFENSE and DEFENSE rating, opponent-adjusted,
    plus league-average scoring and home-field advantage.
  * Home-field advantage is anchored to a prior (default 3.0 pts total) instead of being fully
    free. Early in the year nearly every team has played mostly home games vs weak opponents, and a
    free fit dumps that into home-field advantage (it came out at 7.2 pts in the first live run).
  * Blowouts are dampened (margin beyond 28 counts 25%) while PRESERVING the game total.
  * All FCS opponents are pooled into one "FCS" pseudo-team so those games still carry information.
  * Ratings are in POINTS: overall_rating = how many points better than an average FBS team on a
    neutral field. off_rating = points above average scored vs an average defense;
    def_rating = points saved vs an average offense (higher = better for both).
  * Optional --carryover X (0-1): instead of shrinking each team toward "average", shrink it toward
    X * its LAST-season rating. Default 0 = purely current-year results, as originally requested.
    Early in the season, values around 0.4-0.6 usually track the betting market much better.

Output: output/power_rankings.csv and output/model_params.json (used by the other scripts)
Run:    python power_rankings.py            (optional: --ridge 6 --carryover 0.5 --scale 1.3 --hfa 3.0)
Tune:   python backtest.py                  (finds ridge / carryover / scale from past seasons)
"""
import json

import numpy as np
import pandas as pd

from common import (OUT, apply_std_args, cfbd_get, fbs_team_conferences, fetch_games, std_parser)

BLOWOUT_CAP = 28.0     # margin beyond this is dampened
BLOWOUT_KEEP = 0.25    # ...to this fraction of the excess
DEFAULT_RIDGE = 6.0    # bigger = more shrinkage toward the prior (matters early in the year)
DEFAULT_HFA_TOTAL = 3.0
DEFAULT_HFA_STRENGTH = 1e5    # HFA is effectively FIXED at the prior (use ~0 to estimate it freely)
FCS = "__FCS__"


def dampen(hp, ap):
    """Shrink blowout margins but keep the game total: move points from winner to loser."""
    m = hp - ap
    if abs(m) <= BLOWOUT_CAP:
        return float(hp), float(ap)
    shift = (abs(m) - BLOWOUT_CAP) * (1 - BLOWOUT_KEEP) / 2.0
    return (hp - shift, ap + shift) if m > 0 else (hp + shift, ap - shift)


def fit_ratings(obs, n, ridge, hfa_prior_side, hfa_strength, prior_mean=None):
    """obs rows: (team_idx, opp_idx, home_sign, points, weight). Returns mu, hfa_per_side, off[n], defn[n]."""
    p = 2 + 2 * n
    X = np.zeros((len(obs), p))
    y = np.zeros(len(obs))
    w = np.zeros(len(obs))
    for r, (t, o, s, pts, wt) in enumerate(obs):
        X[r, 0] = 1.0
        X[r, 1] = s
        X[r, 2 + t] = 1.0        # scoring team's offense
        X[r, 2 + n + o] = 1.0    # opponent's defense (positive = allows more)
        y[r], w[r] = pts, wt
    y = y - X[:, 1] * hfa_prior_side            # fit HFA as a deviation from the prior
    pen = np.full(p, 1e-6)
    pen[1] = max(hfa_strength, 1e-6)
    pen[2:] = ridge
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    b = X.T @ (w * y)
    if prior_mean is not None:
        b = b + pen * prior_mean                # shrink toward prior_mean instead of toward 0
    beta = np.linalg.solve(A, b)
    return beta[0], beta[1] + hfa_prior_side, beta[2:2 + n], -beta[2 + n:]   # defense flipped: higher = better


def build_and_fit(games, ridge, half_life, hfa_total, hfa_strength, prior=None):
    conf = fbs_team_conferences(games)
    teams = sorted(conf)
    idx = {t: i for i, t in enumerate(teams)}
    idx[FCS] = len(teams)
    n = len(idx)
    done = [g for g in games if g["completed"] and g["home_pts"] is not None and g["away_pts"] is not None]
    if not done:
        return None
    last_week = max(g["week"] or 0 for g in done)
    obs, per_team = [], {t: [] for t in teams}
    for g in done:
        h = g["home"] if g["home_fbs"] else FCS
        a = g["away"] if g["away_fbs"] else FCS
        hp, ap_ = float(g["home_pts"]), float(g["away_pts"])
        for team, opp, pts, opp_pts, fbs, ofbs in ((g["home"], g["away"], hp, ap_, g["home_fbs"], g["away_fbs"]),
                                                   (g["away"], g["home"], ap_, hp, g["away_fbs"], g["home_fbs"])):
            if fbs:
                per_team[team].append(dict(week=g["week"] or 0, opp=opp if ofbs else FCS, pts=pts, opp_pts=opp_pts))
        dh, da = dampen(hp, ap_)
        wt = 1.0
        if half_life > 0:
            wt = 0.5 ** ((last_week - (g["week"] or 0)) / half_life)
        s = 0.0 if g["neutral"] else 1.0
        obs.append((idx[h], idx[a], s, dh, wt))
        obs.append((idx[a], idx[h], -s, da, wt))
    prior_mean = None
    if prior:
        prior_mean = np.zeros(2 + 2 * n)
        for t, (po, pd_) in prior.items():
            if t in idx:
                prior_mean[2 + idx[t]] = po
                prior_mean[2 + n + idx[t]] = -pd_       # defense coefficient has flipped sign
    mu, hfa, off, defr = fit_ratings(obs, n, ridge, hfa_total / 2.0, hfa_strength, prior_mean)
    return dict(teams=teams, idx=idx, conf=conf, mu=mu, hfa=hfa, off=off, defr=defr,
                per_team=per_team, done=done, last_week=last_week)


def efficiency_table(year):
    """Season advanced stats -> per-team PPA / success rate (informational + efficiency rank)."""
    try:
        raw = cfbd_get("/stats/season/advanced", year=year, excludeGarbageTime="true")
    except RuntimeError:
        return {}
    out = {}
    for r in raw:
        off, de = r.get("offense") or {}, r.get("defense") or {}
        out[r.get("team")] = dict(
            off_ppa=off.get("ppa"), def_ppa=de.get("ppa"),
            off_success=off.get("successRate"), def_success=de.get("successRate"),
            off_explosive=off.get("explosiveness"), def_explosive=de.get("explosiveness"))
    return out


def main():
    ap = std_parser("Power-rank all FBS teams from current-year results")
    ap.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    ap.add_argument("--half-life", type=float, default=0.0,
                    help="recency weighting in weeks (0 = every game counts equally)")
    ap.add_argument("--hfa", type=float, default=DEFAULT_HFA_TOTAL, help="home-field advantage prior, total points")
    ap.add_argument("--hfa-strength", type=float, default=DEFAULT_HFA_STRENGTH,
                    help="how tightly HFA is held to the prior (0 = estimate freely)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply all team ratings (>1 un-compresses over-shrunk ratings; tune with backtest.py)")
    ap.add_argument("--carryover", type=float, default=0.0,
                    help="0-1: shrink toward this fraction of LAST season's ratings (0 = current year only)")
    args = ap.parse_args()
    apply_std_args(args)

    prior = None
    if args.carryover > 0:
        try:
            prev = build_and_fit(fetch_games(args.year - 1), args.ridge, 0.0, args.hfa, args.hfa_strength)
        except RuntimeError:
            prev = None
        if prev:
            prior = {t: (args.carryover * prev["off"][prev["idx"][t]], args.carryover * prev["defr"][prev["idx"][t]])
                     for t in prev["teams"]}
            print(f"Using {args.year - 1} ratings as prior (carryover {args.carryover:g}).")
        else:
            print(f"Could not build {args.year - 1} ratings - falling back to current-year only.")

    res = build_and_fit(fetch_games(args.year), args.ridge, args.half_life, args.hfa, args.hfa_strength, prior)
    if not res:
        raise SystemExit(f"No completed games found for {args.year}.")
    teams, idx, conf = res["teams"], res["idx"], res["conf"]
    mu, hfa, off, defr = res["mu"], res["hfa"], res["off"] * args.scale, res["defr"] * args.scale
    per_team, done, last_week = res["per_team"], res["done"], res["last_week"]
    overall = off + defr
    ov = {t: overall[idx[t]] for t in idx}
    eff = efficiency_table(args.year)

    rows = []
    for t in teams:
        gs = sorted(per_team[t], key=lambda r: r["week"])
        if not gs:
            continue
        w = sum(1 for r in gs if r["pts"] > r["opp_pts"])
        l = sum(1 for r in gs if r["pts"] < r["opp_pts"])
        margins = [r["pts"] - r["opp_pts"] for r in gs]
        e = eff.get(t, {})
        row = dict(
            team=t, conference=conf[t], record=f"{w}-{l}", games=len(gs),
            overall_rating=overall[idx[t]], off_rating=off[idx[t]], def_rating=defr[idx[t]],
            sos=float(np.mean([ov[r["opp"]] for r in gs])),
            ppg=float(np.mean([r["pts"] for r in gs])), papg=float(np.mean([r["opp_pts"] for r in gs])),
            avg_margin=float(np.mean(margins)), last3_margin=float(np.mean(margins[-3:])),
            off_ppa=e.get("off_ppa"), def_ppa=e.get("def_ppa"),
            off_success=e.get("off_success"), def_success=e.get("def_success"),
            off_explosive=e.get("off_explosive"), def_explosive=e.get("def_explosive"))
        if prior:
            row["prior_rating"] = sum(prior.get(t, (0.0, 0.0)))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["overall_rank"] = df["overall_rating"].rank(ascending=False, method="min").astype(int)
    df["off_rank"] = df["off_rating"].rank(ascending=False, method="min").astype(int)
    df["def_rank"] = df["def_rating"].rank(ascending=False, method="min").astype(int)
    df["sos_rank"] = df["sos"].rank(ascending=False, method="min").astype(int)
    if df["off_ppa"].notna().any():
        df["efficiency_score"] = df["off_ppa"] - df["def_ppa"]          # defense PPA: lower allowed = better
        df["efficiency_rank"] = df["efficiency_score"].rank(ascending=False, method="min")
        df["results_vs_efficiency"] = df["efficiency_rank"] - df["overall_rank"]  # + = results better than play
    df = df.sort_values("overall_rank")

    first = ["overall_rank", "team", "conference", "record", "overall_rating", "off_rank", "off_rating",
             "def_rank", "def_rating", "sos_rank", "sos", "games"]
    df = df[first + [c for c in df.columns if c not in first]].round(3)
    df.to_csv(OUT / "power_rankings.csv", index=False)
    (OUT / "model_params.json").write_text(json.dumps(dict(
        season=args.year, league_avg_points=float(mu), home_field_per_side=float(hfa),
        ridge=args.ridge, carryover=args.carryover, scale=args.scale, games_used=len(done), through_week=last_week)))

    print(f"Season {args.year} | {len(done)} games through week {last_week} | ridge={args.ridge} | "
          f"carryover={args.carryover:g} | scale={args.scale:g}")
    print(f"League avg {mu:.1f} pts/team | home-field advantage {2 * hfa:.1f} pts (total)\n")
    show = ["overall_rank", "team", "record", "overall_rating", "off_rank", "def_rank", "sos_rank"]
    print(df[show].head(25).to_string(index=False))
    print(f"\nSaved {len(df)} teams -> {OUT / 'power_rankings.csv'}")


if __name__ == "__main__":
    main()
