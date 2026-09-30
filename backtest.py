#!/usr/bin/env python3
"""BACKTEST - tune the power-rating settings on past seasons instead of guessing.

For each past season and each early-season week (default weeks 4-8) it:
  1. fits the ratings using ONLY games from earlier weeks (exactly what you'd have had that week),
     optionally shrinking toward last season's ratings (carryover),
  2. predicts every FBS-vs-FBS game that week,
  3. compares to the actual final margin, and to the sportsbook closing line.

It grid-searches ridge strength, carryover and scale, then prints the best settings and the exact
command to run. Key columns:
  mae          average miss vs the actual margin (lower = better)
  market_mae   the sportsbooks' miss on the same games - the bar to beat. Expect to land near, not under.
  slope        actual margin regressed on predicted margin. ~1.0 = well calibrated; >1 = predictions too
               compressed (raise scale); <1 = too spread out.
  vs_mkt       average gap between your line and the market line (smaller = you look more like the market)

API use: about 4 CFBD calls per season, all cached for 30 days (default 3 past seasons + current ~ 20 calls).

Run: python backtest.py            (optional: --seasons 4 --weeks 4-8 --hfa 3.0)
"""
import numpy as np
import pandas as pd

from common import OUT, apply_std_args, fetch_games, fetch_lines, std_parser
from power_rankings import build_and_fit

RIDGES = [1.5, 2.5, 4.0, 6.0]
CARRYOVERS = [0.0, 0.6, 0.8, 1.0, 1.2]
SCALES = [1.0, 1.1, 1.2, 1.3, 1.4]
LONG_TTL = 30 * 86400
HFA_STRENGTH = 1e5


def parse_weeks(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = std_parser("Tune ridge / carryover / scale on past seasons")
    ap.add_argument("--seasons", type=int, default=3, help="how many past seasons to test")
    ap.add_argument("--weeks", default="4-8", help="target weeks, e.g. 4-8")
    ap.add_argument("--hfa", type=float, default=3.0, help="total home-field advantage held fixed (points)")
    args = ap.parse_args()
    apply_std_args(args)

    weeks = parse_weeks(args.weeks)
    years = list(range(args.year - args.seasons, args.year + 1))
    print(f"Loading seasons {years[0] - 1}-{years[-1]} (cached after the first run)...")
    games, lines = {}, {}
    for y in range(years[0] - 1, years[-1] + 1):
        try:
            games[y] = fetch_games(y, ttl=LONG_TTL if y < args.year else 3600)
        except RuntimeError as e:
            print(f"  {y}: could not load games ({e})")
            games[y] = []
    for y in years:
        try:
            lines[y] = fetch_lines(y, ttl=LONG_TTL if y < args.year else 3600)
        except RuntimeError:
            lines[y] = {}

    prev_fits = {}
    rows = []
    for y in years:
        G = games.get(y) or []
        if not G:
            continue
        for ridge in RIDGES:
            if (y - 1, ridge) not in prev_fits:
                prev_fits[(y - 1, ridge)] = build_and_fit(games.get(y - 1) or [], ridge, 0.0, args.hfa, HFA_STRENGTH)
            prev = prev_fits[(y - 1, ridge)]
            for carry in CARRYOVERS:
                prior = None
                if carry > 0 and prev:
                    prior = {t: (carry * prev["off"][prev["idx"][t]], carry * prev["defr"][prev["idx"][t]])
                             for t in prev["teams"]}
                elif carry > 0:
                    continue        # no previous season available to carry over
                for w in weeks:
                    test = [g for g in G if g["season_type"] == "regular" and g["week"] == w and g["completed"]
                            and g["home_fbs"] and g["away_fbs"]
                            and g["home_pts"] is not None and g["away_pts"] is not None]
                    if not test:
                        continue
                    train = [g for g in G if g["season_type"] == "regular" and (g["week"] or 0) < w]
                    fit = build_and_fit(train, ridge, 0.0, args.hfa, HFA_STRENGTH, prior)
                    if not fit:
                        continue
                    ov = {t: fit["off"][fit["idx"][t]] + fit["defr"][fit["idx"][t]] for t in fit["teams"]}
                    for g in test:
                        base = 0.0 if g["neutral"] else 2.0 * fit["hfa"]
                        d = ov.get(g["home"], 0.0) - ov.get(g["away"], 0.0)
                        actual = float(g["home_pts"]) - float(g["away_pts"])
                        ln = lines.get(y, {}).get(g["id"])
                        mkt = None if not ln or ln["home_spread"] is None else -ln["home_spread"]
                        mtot = None if not ln else ln["total"]
                        atot = float(g["home_pts"]) + float(g["away_pts"])
                        th = fit["off"][fit["idx"][g["home"]]] - fit["defr"][fit["idx"][g["home"]]] \
                            if g["home"] in fit["idx"] else 0.0
                        ta = fit["off"][fit["idx"][g["away"]]] - fit["defr"][fit["idx"][g["away"]]] \
                            if g["away"] in fit["idx"] else 0.0
                        for sc in SCALES:
                            rows.append((y, w, ridge, carry, sc, base + sc * d, actual, mkt,
                                         2.0 * fit["mu"] + sc * (th + ta), atot, mtot))
    if not rows:
        raise SystemExit("No games could be backtested (check API key / seasons).")

    df = pd.DataFrame(rows, columns=["year", "week", "ridge", "carryover", "scale", "pred", "actual", "mkt",
                                     "ptot", "atot", "mtot"])
    df["err"] = (df["pred"] - df["actual"]).abs()
    mk = df[(df["ridge"] == RIDGES[0]) & (df["carryover"] == df["carryover"].min()) & (df["scale"] == SCALES[0])]
    mk = mk.dropna(subset=["mkt"])
    market_mae = float((mk["mkt"] - mk["actual"]).abs().mean()) if len(mk) else float("nan")
    market_n = len(mk)

    out = []
    for (ridge, carry, sc), g in df.groupby(["ridge", "carryover", "scale"]):
        slope = float(np.polyfit(g["pred"], g["actual"], 1)[0])
        gm = g.dropna(subset=["mkt"])
        out.append(dict(ridge=ridge, carryover=carry, scale=sc, games=len(g),
                        mae=g["err"].mean(), rmse=float(np.sqrt(((g["pred"] - g["actual"]) ** 2).mean())),
                        slope=slope, vs_mkt=(gm["pred"] - gm["mkt"]).abs().mean() if len(gm) else np.nan))
    res = pd.DataFrame(out).sort_values("mae").round(3)
    res["market_mae"] = round(market_mae, 3)
    res.to_csv(OUT / "backtest_results.csv", index=False)

    print(f"\nUp to {int(res['games'].max())} games per setting across {years[0]}-{years[-1]}, "
          f"weeks {weeks[0]}-{weeks[-1]}")
    print(f"Sportsbook closing-line MAE on {market_n} lined games: {market_mae:.2f}\n")
    print("Top 10 settings:")
    print(res.head(10).to_string(index=False))
    best = res.iloc[0]
    print(f"\nBest: ridge={best['ridge']:g}, carryover={best['carryover']:g}, scale={best['scale']:g} "
          f"(MAE {best['mae']:.2f} vs market {market_mae:.2f}, slope {best['slope']:.2f})")
    print("\nPer-year MAE for the best setting:")
    sel = df[(df["ridge"] == best["ridge"]) & (df["carryover"] == best["carryover"]) & (df["scale"] == best["scale"])]
    print(sel.groupby("year")["err"].agg(["mean", "count"]).round(2).to_string())
    # ---- how much should the model be trusted next to the market?
    s2 = sel.dropna(subset=["mkt"])
    if len(s2) > 30:
        x, yv = s2["pred"] - s2["mkt"], s2["actual"] - s2["mkt"]
        w_opt = float(min(1.0, max(0.0, (x * yv).sum() / (x * x).sum())))
        print(f"\nSPREADS - blending model with the market line ({len(s2)} games)")
        print("  weight on model : " + "  ".join(f"{w:>5.1f}" for w in np.arange(0, 1.01, 0.2)))
        print("  MAE             : " + "  ".join(
            f"{(s2['mkt'] + w * x - s2['actual']).abs().mean():>5.2f}" for w in np.arange(0, 1.01, 0.2)))
        print(f"  best weight on the model = {w_opt:.2f}  (0 = market alone is best, 1 = model alone)")
        print(f"  -> for kalshi_picks.py try  --shrink {min(0.95, max(0.5, 1 - w_opt)):.2f}")
    s3 = sel.dropna(subset=["mtot"])
    if len(s3) > 30:
        x, yv = s3["ptot"] - s3["mtot"], s3["atot"] - s3["mtot"]
        w_t = float(min(1.0, max(0.0, (x * yv).sum() / (x * x).sum())))
        print(f"\nTOTALS - model vs market total ({len(s3)} games, ratings only: no weather/pace/injuries)")
        print(f"  market MAE {(s3['mtot'] - s3['atot']).abs().mean():.2f} | model MAE "
              f"{(s3['ptot'] - s3['atot']).abs().mean():.2f} | best weight on the model = {w_t:.2f}")
    print(f"\nRun with:\n  python power_rankings.py --ridge {best['ridge']:g} --carryover {best['carryover']:g} "
          f"--scale {best['scale']:g} --hfa {args.hfa:g}")
    print(f"Full grid saved -> {OUT / 'backtest_results.csv'}")


if __name__ == "__main__":
    main()
