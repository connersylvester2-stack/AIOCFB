#!/usr/bin/env python3
"""SCRIPT 2 - Cover statistics for every FBS team this season.

For each completed game that has a betting line (median across the books CFBD tracks):
  * ATS result           - did the team cover the spread?
  * Total result         - did the game go over/under the total?
  * Points share of line - team points / game total LINE      (25 pts on a 50 total = 50%)
  * Points share of game - team points / ACTUAL game total
  * Implied team total   - total/2 - spread/2 ; then did the team beat its implied number?

Output: output/cover_stats.csv
Run:    python cover_stats.py            (optional: --provider "DraftKings")
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from common import OUT, apply_std_args, fetch_games, fetch_lines, std_parser


def code(x):
    return "W" if x > 0 else "L" if x < 0 else "P"


def record(rs, key):
    w = sum(1 for r in rs if r[key] == "W")
    l = sum(1 for r in rs if r[key] == "L")
    p = sum(1 for r in rs if r[key] == "P")
    return w, l, p


def pct(w, l):
    return round(100.0 * w / (w + l), 1) if (w + l) else np.nan


def main():
    ap = std_parser("Spread/total cover percentages for all FBS teams")
    ap.add_argument("--provider", default=None, help="use one sportsbook instead of the median of all")
    args = ap.parse_args()
    apply_std_args(args)

    games = {g["id"]: g for g in fetch_games(args.year)}
    lines = fetch_lines(args.year, provider=args.provider)

    by_team = defaultdict(list)
    conf = {}
    for gid, ln in lines.items():
        g = games.get(gid)
        if not g or not g["completed"] or g["home_pts"] is None or g["away_pts"] is None:
            continue
        hp, ap_ = float(g["home_pts"]), float(g["away_pts"])
        game_total = hp + ap_
        T, hs = ln["total"], ln["home_spread"]
        for side in ("home", "away"):
            if not g[f"{side}_fbs"]:
                continue
            team = g[side]
            conf[team] = g[f"{side}_conf"]
            pts, opp_pts = (hp, ap_) if side == "home" else (ap_, hp)
            spread = None if hs is None else (hs if side == "home" else -hs)
            r = dict(week=g["week"] or 0, team=team, opp=g["away" if side == "home" else "home"],
                     venue="N" if g["neutral"] else ("H" if side == "home" else "A"),
                     pts=pts, opp_pts=opp_pts, spread=spread, total_line=T, ats=None, ou=None,
                     cover_margin=None, share_line=None, share_actual=None, implied=None, vs_implied=None,
                     team_total=None)
            if spread is not None:
                r["cover_margin"] = (pts - opp_pts) + spread
                r["ats"] = code(r["cover_margin"])
            if T is not None:
                r["ou"] = code(game_total - T)                       # W = over, L = under
                r["share_line"] = pts / T if T else None
                r["share_actual"] = pts / game_total if game_total else None
                if spread is not None:
                    r["implied"] = T / 2.0 - spread / 2.0
                    r["vs_implied"] = pts - r["implied"]
                    r["team_total"] = code(pts - r["implied"])       # W = over own team total
            by_team[team].append(r)

    if not by_team:
        raise SystemExit("No completed games with lines found. (Lines appear once books post them.)")

    rows = []
    for team, rs in by_team.items():
        rs.sort(key=lambda r: r["week"])
        ats = [r for r in rs if r["ats"]]
        ou = [r for r in rs if r["ou"]]
        tt = [r for r in rs if r["team_total"]]
        aw, al, ap_p = record(ats, "ats")
        ow, ol, op = record(ou, "ou")           # W = over, L = under
        tw, tl, _ = record(tt, "team_total")
        fw, fl, _ = record([r for r in ats if r["spread"] < 0], "ats")
        dw, dl, _ = record([r for r in ats if r["spread"] > 0], "ats")
        hw, hl, _ = record([r for r in ats if r["venue"] == "H"], "ats")
        vw, vl, _ = record([r for r in ats if r["venue"] == "A"], "ats")
        l3w, l3l, _ = record(ats[-3:], "ats")
        vals = lambda k: [r[k] for r in rs if r[k] is not None]
        rows.append(dict(
            team=team, conference=conf[team], games_with_lines=len(rs),
            ats_record=f"{aw}-{al}" + (f"-{ap_p}" if ap_p else ""), ats_cover_pct=pct(aw, al),
            avg_cover_margin=np.mean(vals("cover_margin")) if vals("cover_margin") else np.nan,
            ats_last3=f"{l3w}-{l3l}",
            ats_as_fav=f"{fw}-{fl}", ats_as_dog=f"{dw}-{dl}", ats_home=f"{hw}-{hl}", ats_away=f"{vw}-{vl}",
            over_record=f"{ow}-{ol}" + (f"-{op}" if op else ""),
            over_pct=pct(ow, ol), under_pct=pct(ol, ow),
            avg_total_line=np.mean(vals("total_line")) if vals("total_line") else np.nan,
            avg_game_total=np.mean([r["pts"] + r["opp_pts"] for r in rs]),
            avg_pts_share_of_line_pct=100 * np.mean(vals("share_line")) if vals("share_line") else np.nan,
            avg_pts_share_of_game_pct=100 * np.mean(vals("share_actual")) if vals("share_actual") else np.nan,
            avg_pts=np.mean([r["pts"] for r in rs]),
            avg_implied_team_total=np.mean(vals("implied")) if vals("implied") else np.nan,
            avg_pts_vs_implied=np.mean(vals("vs_implied")) if vals("vs_implied") else np.nan,
            team_total_over_pct=pct(tw, tl),
        ))
    df = pd.DataFrame(rows)
    df = df.sort_values(["ats_cover_pct", "avg_cover_margin"], ascending=False).round(2)
    df["ats_rank"] = range(1, len(df) + 1)     # by cover %, ties broken by average cover margin
    df.to_csv(OUT / "cover_stats.csv", index=False)

    show = ["team", "games_with_lines", "ats_record", "ats_cover_pct", "over_record", "over_pct",
            "avg_pts_share_of_line_pct", "avg_pts_vs_implied"]
    enough = df[df["games_with_lines"] >= 3]
    print(f"{len(df)} teams with lined games ({len(lines)} lines found)\n")
    print("Best ATS (3+ games):\n", enough[show].head(10).to_string(index=False))
    print("\nWorst ATS (3+ games):\n", enough[show].tail(10).to_string(index=False))
    print(f"\nSaved -> {OUT / 'cover_stats.csv'}")
    ex = next(iter(by_team.values()))[0]
    print(f"\nSanity check (verify sign convention): {ex['team']} vs {ex['opp']}, team spread {ex['spread']}, "
          f"score {ex['pts']:.0f}-{ex['opp_pts']:.0f}, ATS={ex['ats']}")


if __name__ == "__main__":
    main()
