#!/usr/bin/env python3
"""SCRIPT 3 - Simulate the games YOU choose from the current week.

DEFAULT MODE = the model generates its own line, total and win probabilities from the power ratings,
pace/tendencies, weather and injuries (the sportsbook line is shown for reference only). Optional
--anchor-market starts from the sportsbook consensus line instead (backtest: the ratings alone were
less accurate than the closing line, so that mode is there if you ever want it).

Inputs it combines
  1. Power ratings        (output/power_rankings.csv + model_params.json  <- run script 1 first)
  2. CFBD data            (schedule, kickoff, venue, market line for reference, pace + pass/run tendencies)
  3. Weather              (Open-Meteo forecast, free/no key; skipped for domes and games >15 days out)
  4. Injuries             (injuries.csv that YOU maintain - no free reliable college injury feed exists)
  5. Tendencies           (pace of play + pass-rate x pass/run matchup edges from CFBD PPA)

Picking games
  python simulate_games.py                                  # lists this week's games, you type numbers
  python simulate_games.py --all                            # every FBS-vs-FBS game this week
  python simulate_games.py --games "Georgia @ Alabama" "Ohio State @ Michigan"     # Away @ Home
  python simulate_games.py --games "Oregon @ Ohio State" --neutral                 # hypothetical/neutral site

Output: output/sim_results.csv (one row per game) and output/sim_draws.npz (every simulated score,
used by kalshi_picks.py to price any spread/total/winner market).

injuries.csv columns:  team,player,position,status,points
  status: out / doubtful / questionable / probable   points: optional override (margin points if he's out)
"""
import datetime as dt
import json
import re

import numpy as np
import pandas as pd

from common import (OUT, BASE, apply_std_args, build_matcher, cfbd_get, fetch_games, fetch_lines,
                    find_teams, http_json, local_kickoff, norm_cdf, parse_utc, resolve_team, std_parser)

# ----------------------------------------------------------------------------- tunable constants
N_SIMS_DEFAULT = 20000
SD_MARGIN_BASE, SD_MARGIN_EARLY = 14.5, 6.0     # margin sd = base + early/sqrt(avg games played)
SD_TOTAL_BASE, SD_TOTAL_EARLY = 13.5, 3.0
# Spread of outcomes around the model's own projection, taken from the backtest errors of the ratings model:
# spread RMSE 15.7; total MAE 13.32 / 0.798 = 16.7.
SD_MARGIN_MODEL, SD_TOTAL_MODEL = 15.7, 16.7
# Only used with --anchor-market: backtest blend weights (spread 0.00, total 0.09) and closing-line errors.
W_MARGIN_DEFAULT, W_TOTAL_DEFAULT = 0.0, 0.10
SD_MARGIN_ANCHOR = 14.6   # closing-spread MAE 11.65 / 0.798
SD_TOTAL_ANCHOR = 15.9    # closing-total  MAE 12.70 / 0.798
PACE_WEIGHT = 0.35        # how much of a pace mismatch flows into the total
TEND_K, TEND_CAP = 0.3, 1.5   # pass/run matchup adjustment scale and cap (points per team)
INJ_CAP = 12.0            # max points of injury adjustment per side of the ball
STATUS_MULT = {"out": 1.0, "doubtful": 0.85, "questionable": 0.4, "day-to-day": 0.3, "probable": 0.1}
POS_POINTS = {"QB": 5.5, "RB": 1.0, "WR": 0.8, "TE": 0.5, "OL": 0.6, "OT": 0.6, "OG": 0.6, "C": 0.6,
              "FB": 0.2, "K": 0.6, "P": 0.3, "DL": 0.7, "DE": 0.8, "DT": 0.7, "NT": 0.6, "EDGE": 0.9,
              "LB": 0.6, "CB": 0.8, "S": 0.5, "DB": 0.6}
OFFENSE_POS = {"QB", "RB", "WR", "TE", "OL", "OT", "OG", "C", "FB", "K", "P"}
INJ_PATH = BASE / "injuries.csv"


# ----------------------------------------------------------------------------- inputs
def load_ratings():
    pr, mp = OUT / "power_rankings.csv", OUT / "model_params.json"
    if not pr.exists() or not mp.exists():
        raise SystemExit("Run power_rankings.py first (needs output/power_rankings.csv).")
    return pd.read_csv(pr).set_index("team"), json.loads(mp.read_text())


def load_tendencies(year):
    """pace, pass rate and pass/run PPA (offense and defense) per team, plus league averages."""
    adv, st = {}, {}
    try:
        for r in cfbd_get("/stats/season/advanced", year=year, excludeGarbageTime="true"):
            off, de = r.get("offense") or {}, r.get("defense") or {}
            adv[r["team"]] = dict(off_pass=(off.get("passingPlays") or {}).get("ppa"),
                                  off_rush=(off.get("rushingPlays") or {}).get("ppa"),
                                  def_pass=(de.get("passingPlays") or {}).get("ppa"),
                                  def_rush=(de.get("rushingPlays") or {}).get("ppa"))
    except RuntimeError:
        pass
    try:
        for r in cfbd_get("/stats/season", year=year):
            st.setdefault(r["team"], {})[r["statName"]] = r["statValue"]
    except RuntimeError:
        pass
    tend = {}
    for team in set(adv) | set(st):
        d = dict(adv.get(team, {}))
        s = st.get(team, {})
        gp, pa, ra = s.get("games"), s.get("passAttempts"), s.get("rushingAttempts")
        if gp and pa is not None and ra is not None and (pa + ra) > 0:
            d["plays_pg"] = (pa + ra) / gp
            d["pass_rate"] = pa / (pa + ra)
        tend[team] = d
    lg = {}
    for k in ("off_pass", "off_rush", "def_pass", "def_rush", "plays_pg", "pass_rate"):
        vals = [t[k] for t in tend.values() if t.get(k) is not None]
        lg[k] = float(np.mean(vals)) if vals else None
    return tend, lg


def tendency_adj(t, o, lg):
    """Bonus/penalty (points) for team t's offense vs opponent o's defense from pass-vs-run tendencies.
    = plays * (team pass rate - league pass rate) * (pass-game edge - run-game edge)."""
    try:
        pass_edge = (t["off_pass"] - lg["off_pass"]) + (o["def_pass"] - lg["def_pass"])
        rush_edge = (t["off_rush"] - lg["off_rush"]) + (o["def_rush"] - lg["def_rush"])
        val = TEND_K * t["plays_pg"] * (t["pass_rate"] - lg["pass_rate"]) * (pass_edge - rush_edge)
    except (TypeError, KeyError):
        return 0.0
    return max(-TEND_CAP, min(TEND_CAP, val))


def load_venues():
    try:
        return {v["id"]: v for v in cfbd_get("/venues", ttl=7 * 86400)}
    except RuntimeError:
        return {}


def get_weather(game, venues):
    """Returns (total_adjustment_points, description). Negative = fewer points expected."""
    v = venues.get(game.get("venue_id"))
    if not v:
        return 0.0, "no venue data"
    if v.get("dome"):
        return 0.0, "dome"
    lat, lon = v.get("latitude"), v.get("longitude")
    if lat is None and isinstance(v.get("location"), dict):
        lat, lon = v["location"].get("x"), v["location"].get("y")
    ko = parse_utc(game.get("start"))
    now = dt.datetime.now(dt.timezone.utc)
    if lat is None or lon is None or ko is None or ko < now - dt.timedelta(hours=6) or ko > now + dt.timedelta(days=15):
        return 0.0, "forecast unavailable"
    try:
        d = http_json("https://api.open-meteo.com/v1/forecast", dict(
            latitude=lat, longitude=lon, hourly="temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
            temperature_unit="fahrenheit", wind_speed_unit="mph", precipitation_unit="inch",
            timezone="UTC", forecast_days=16), ttl=1800)
        times = d["hourly"]["time"]
        key = ko.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:00")
        i = times.index(key)
        sl = slice(i, i + 3)
        wind = float(np.mean(d["hourly"]["wind_speed_10m"][sl]))
        gust = float(np.max(d["hourly"]["wind_gusts_10m"][sl]))
        rain = float(np.sum(d["hourly"]["precipitation"][sl]))
        temp = float(np.mean(d["hourly"]["temperature_2m"][sl]))
    except (RuntimeError, KeyError, ValueError, TypeError):
        return 0.0, "forecast unavailable"
    adj = -min(5.0, 0.25 * max(0.0, wind - 10.0))          # wind hurts passing/kicking
    adj += -min(3.0, 6.0 * rain)                            # rain over the 3-hr window
    adj += -min(1.5, 0.05 * max(0.0, 32.0 - temp))          # freezing cold
    return adj, f"{temp:.0f}F, wind {wind:.0f} mph (gusts {gust:.0f}), rain {rain:.2f} in"


def load_injuries(matcher):
    """{team: dict(off, def, notes)} in POINTS lost. Creates a blank template the first time."""
    if not INJ_PATH.exists():
        INJ_PATH.write_text("team,player,position,status,points\n")
        return {}
    try:
        df = pd.read_csv(INJ_PATH)
    except pd.errors.EmptyDataError:
        return {}
    df.columns = [c.strip().lower() for c in df.columns]
    out = {}
    for _, r in df.iterrows():
        team = resolve_team(str(r.get("team", "")), matcher)
        if not team:
            print(f"  [injuries.csv] could not match team '{r.get('team')}'")
            continue
        mult = STATUS_MULT.get(str(r.get("status", "")).strip().lower(), 0.0)
        pos = str(r.get("position", "")).strip().upper()
        pts = r.get("points")
        pts = float(pts) if pts is not None and not pd.isna(pts) else POS_POINTS.get(pos, 0.5)
        d = out.setdefault(team, dict(off=0.0, deff=0.0, notes=[]))
        side = "off" if pos in OFFENSE_POS else "deff"
        d[side] += pts * mult
        if mult > 0:
            d["notes"].append(f"{r.get('player', pos)} ({pos}, {r.get('status')})")
    for d in out.values():
        d["off"], d["deff"] = min(d["off"], INJ_CAP), min(d["deff"], INJ_CAP)
    return out


# ----------------------------------------------------------------------------- game selection
def upcoming_games(year, week=None):
    games = [g for g in fetch_games(year, ttl=1800)
             if not g["completed"] and g["home_fbs"] and g["away_fbs"]]
    if not games:
        return [], None
    if week is None:
        week = min(g["week"] or 99 for g in games)
    games = [g for g in games if g["week"] == week]
    games.sort(key=lambda g: g["start"] or "")
    return games, week


def parse_manual(text, matcher):
    parts = re.split(r"\s+(?:@|at)\s+", text.strip(), flags=re.I)
    if len(parts) != 2:
        raise SystemExit(f"Couldn't read '{text}'. Use  \"Away @ Home\"  (e.g. \"Georgia @ Alabama\").")
    a, h = resolve_team(parts[0], matcher), resolve_team(parts[1], matcher)
    if not a or not h:
        raise SystemExit(f"Couldn't match team names in '{text}' (matched: {a}, {h}).")
    return a, h


def choose_games(args, games, week, matcher):
    if args.games:
        chosen = []
        for text in args.games:
            a, h = parse_manual(text, matcher)
            g = next((g for g in games if g["home"] == h and g["away"] == a), None)
            if g is None:
                g = dict(id=f"manual_{a}_{h}".replace(" ", "_"), week=week, start=None, completed=False,
                         neutral=args.neutral, home=h, away=a, venue_id=None, venue=None)
                print(f"  '{a} @ {h}' isn't on this week's schedule - simulating as a manual matchup"
                      f"{' (neutral site)' if args.neutral else ''}.")
            chosen.append(g)
        return chosen
    if not games:
        if args.all:                       # scheduled runs: nothing to do is not an error
            print("No upcoming FBS-vs-FBS games found.")
            raise SystemExit(0)
        raise SystemExit("No upcoming FBS-vs-FBS games found. Use --games \"Away @ Home\" for a manual matchup.")
    if args.all:
        return games
    print(f"\nWeek {week} - FBS vs FBS\n")
    for i, g in enumerate(games, 1):
        print(f"{i:>3}. {g['away']} @ {g['home']}   {local_kickoff(g['start'])}"
              f"{'  (neutral)' if g['neutral'] else ''}")
    raw = input("\nWhich games? (numbers separated by commas, ranges like 3-6, or 'all'): ").strip().lower()
    if raw in ("all", "a"):
        return games
    picked = set()
    for tok in re.split(r"[,\s]+", raw):
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m:
            picked.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif tok.isdigit():
            picked.add(int(tok))
    chosen = [games[i - 1] for i in sorted(picked) if 1 <= i <= len(games)]
    if not chosen:
        raise SystemExit("No valid games selected.")
    return chosen


# ----------------------------------------------------------------------------- the model
def project(game, R, params, tend, lg, venues, injuries):
    h, a = game["home"], game["away"]
    mu, hfa = params["league_avg_points"], 0.0 if game.get("neutral") else params["home_field_per_side"]
    notes = []

    base_h = mu + hfa + R.at[h, "off_rating"] - R.at[a, "def_rating"]
    base_a = mu - hfa + R.at[a, "off_rating"] - R.at[h, "def_rating"]

    # pace: fast-vs-slow matchups move the total
    pace_mult = 1.0
    ph, pa = tend.get(h, {}).get("plays_pg"), tend.get(a, {}).get("plays_pg")
    if ph and pa and lg.get("plays_pg"):
        pace_mult = 1.0 + PACE_WEIGHT * (((ph + pa) / 2.0) / lg["plays_pg"] - 1.0)
    h1, a1 = base_h * pace_mult, base_a * pace_mult

    # pass/run tendency matchup edges
    th, ta = tendency_adj(tend.get(h, {}), tend.get(a, {}), lg), tendency_adj(tend.get(a, {}), tend.get(h, {}), lg)
    h2, a2 = h1 + th, a1 + ta

    # weather (affects total, split evenly)
    w_adj, w_desc = get_weather(game, venues)
    h3, a3 = h2 + w_adj / 2.0, a2 + w_adj / 2.0

    # injuries
    ih, ia = injuries.get(h, dict(off=0, deff=0, notes=[])), injuries.get(a, dict(off=0, deff=0, notes=[]))
    h4 = h3 - ih["off"] + ia["deff"]
    a4 = a3 - ia["off"] + ih["deff"]
    inj_margin = (h4 - a4) - (h3 - a3)
    inj_total = (h4 + a4) - (h3 + a3)
    inj_notes = []
    if ih["notes"]:
        inj_notes.append(f"{h}: " + ", ".join(ih["notes"]))
    if ia["notes"]:
        inj_notes.append(f"{a}: " + ", ".join(ia["notes"]))

    h4, a4 = max(h4, 3.0), max(a4, 3.0)
    return dict(exp_home=h4, exp_away=a4, hfa_total=2 * hfa, base_margin=base_h - base_a,
                pace_mult=pace_mult, tend_home=th, tend_away=ta, weather_total=w_adj, weather=w_desc,
                inj_margin=inj_margin, inj_total=inj_total, inj_notes="; ".join(inj_notes))


def simulate(exp_home, exp_away, games_played, n, rng, sd_override=None):
    gp = max(1.0, games_played)
    if sd_override:
        sd_m, sd_t = sd_override
    else:
        sd_m = SD_MARGIN_BASE + SD_MARGIN_EARLY / np.sqrt(gp)
        sd_t = SD_TOTAL_BASE + SD_TOTAL_EARLY / np.sqrt(gp)
    margin = rng.normal(exp_home - exp_away, sd_m, n)
    total = rng.normal(exp_home + exp_away, sd_t, n)
    home = np.clip(np.rint((total + margin) / 2.0), 0, None)
    away = np.clip(np.rint((total - margin) / 2.0), 0, None)
    tie = home == away
    if tie.any():                       # overtime: winner gets 1-6 pts, favoring the better team
        p_home = norm_cdf((exp_home - exp_away) / sd_m)
        home_wins = rng.random(tie.sum()) < p_home
        bump = rng.integers(1, 7, tie.sum())
        home[tie] += np.where(home_wins, bump, 0)
        away[tie] += np.where(home_wins, 0, bump)
    return np.column_stack([home, away]).astype(np.int16), sd_m, sd_t


def line_text(home, away, margin_home):
    """'Alabama -3.5' style text for the model's projected line."""
    if abs(margin_home) < 0.05:
        return "PK"
    fav = home if margin_home > 0 else away
    return f"{fav} -{abs(margin_home):.1f}"


def main():
    ap = std_parser("Simulate selected games from the current week")
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--games", nargs="*", help='"Away @ Home" (one or more)')
    ap.add_argument("--all", action="store_true", help="simulate every game this week")
    ap.add_argument("--neutral", action="store_true", help="treat manual --games matchups as neutral site")
    ap.add_argument("--sims", type=int, default=N_SIMS_DEFAULT)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--anchor-market", action="store_true",
                    help="start from the sportsbook consensus line instead of the model's own line")
    ap.add_argument("--w-margin", type=float, default=W_MARGIN_DEFAULT, help="(with --anchor-market) weight on model vs market for the spread")
    ap.add_argument("--w-total", type=float, default=W_TOTAL_DEFAULT, help="(with --anchor-market) weight on model vs market for the total")
    ap.add_argument("--apply-extras", action="store_true",
                    help="(with --anchor-market) also add weather + tendency adjustments on top of the market line")
    args = ap.parse_args()
    apply_std_args(args)

    R, params = load_ratings()
    matcher = build_matcher(R.index.tolist())
    games, week = upcoming_games(args.year, args.week)
    chosen = choose_games(args, games, week, matcher)

    print("\nLoading tendencies, venues, lines, injuries...")
    tend, lg = load_tendencies(args.year)
    venues = load_venues()
    try:
        lines = fetch_lines(args.year, ttl=900)
    except RuntimeError:
        lines = {}
    injuries = load_injuries(matcher)
    if not injuries:
        print("  (no injuries loaded - fill in injuries.csv to include them)")
    rng = np.random.default_rng(args.seed)

    rows, draws = [], {}
    for g in chosen:
        h, a = g["home"], g["away"]
        if h not in R.index or a not in R.index:
            print(f"  skipping {a} @ {h}: no power rating")
            continue
        p = project(g, R, params, tend, lg, venues, injuries)
        gp = (R.at[h, "games"] + R.at[a, "games"]) / 2.0
        ln = lines.get(g["id"], {})
        mkt_s, mkt_t = ln.get("home_spread"), ln.get("total")
        pure_margin = p["exp_home"] - p["exp_away"]
        pure_total = p["exp_home"] + p["exp_away"]
        exp_h, exp_a, sd_override, mode = p["exp_home"], p["exp_away"], None, "model"
        if args.anchor_market:
            if mkt_s is not None and mkt_t is not None:
                mode = "market-anchored"
                margin_mean = -mkt_s + args.w_margin * (pure_margin + mkt_s) + p["inj_margin"]
                total_mean = mkt_t + args.w_total * (pure_total - mkt_t) + p["inj_total"]
                if args.apply_extras:
                    margin_mean += p["tend_home"] - p["tend_away"]
                    total_mean += p["weather_total"] + p["tend_home"] + p["tend_away"]
                exp_h = max((total_mean + margin_mean) / 2.0, 3.0)
                exp_a = max((total_mean - margin_mean) / 2.0, 3.0)
                sd_override = (SD_MARGIN_ANCHOR, SD_TOTAL_ANCHOR)
            else:
                print(f"  note: no sportsbook line for {a} @ {h} - using the model's own line")
        if sd_override is None:
            sd_override = (SD_MARGIN_MODEL, SD_TOTAL_MODEL)
        sims, sd_m, sd_t = simulate(exp_h, exp_a, gp, args.sims, rng, sd_override)
        gid = str(g["id"])
        draws[gid] = sims
        hs, as_ = sims[:, 0].astype(float), sims[:, 1].astype(float)
        margin, total = hs - as_, hs + as_
        model_margin = exp_h - exp_a
        row = dict(
            game_id=gid, week=g["week"], kickoff=local_kickoff(g["start"]), away=a, home=h, neutral=bool(g["neutral"]),
            away_rank=int(R.at[a, "overall_rank"]), home_rank=int(R.at[h, "overall_rank"]),
            mode=mode, exp_home_pts=round(exp_h, 1), exp_away_pts=round(exp_a, 1),
            pure_model_margin_home=round(pure_margin, 1), pure_model_total=round(pure_total, 1),
            model_margin_home=round(model_margin, 1), model_spread_home=round(-model_margin, 1),
            model_line=line_text(h, a, model_margin), model_total=round(exp_h + exp_a, 1),
            home_win_pct=round(100 * float((margin > 0).mean()), 1),
            away_win_pct=round(100 * float((margin < 0).mean()), 1),
            market_spread_home=mkt_s, market_total=mkt_t,
            home_cover_pct=None if mkt_s is None else round(100 * float((margin + mkt_s > 0).mean()), 1),
            away_cover_pct=None if mkt_s is None else round(100 * float((margin + mkt_s < 0).mean()), 1),
            over_pct=None if mkt_t is None else round(100 * float((total > mkt_t).mean()), 1),
            under_pct=None if mkt_t is None else round(100 * float((total < mkt_t).mean()), 1),
            hfa_pts=round(p["hfa_total"], 1), pace_mult=round(p["pace_mult"], 3),
            tend_home_adj=round(p["tend_home"], 2), tend_away_adj=round(p["tend_away"], 2),
            weather=p["weather"], weather_total_adj=round(p["weather_total"], 2),
            injury_margin_adj=round(p["inj_margin"], 2), injury_total_adj=round(p["inj_total"], 2),
            injury_notes=p["inj_notes"], sd_margin=round(sd_m, 1), sd_total=round(sd_t, 1))
        rows.append(row)

        print(f"\n{a} (#{row['away_rank']}) @ {h} (#{row['home_rank']})   {row['kickoff']}")
        print(f"  Model : {row['model_line']}, total {row['model_total']}  ->  "
              f"{h} {exp_h:.1f}, {a} {exp_a:.1f}   |   {h} wins {row['home_win_pct']}%   [{mode}]")
        if mode != "model":
            print(f"  Model's own line was: {line_text(h, a, pure_margin)}, total {pure_total:.1f}")
        if mkt_s is not None or mkt_t is not None:
            ms = line_text(h, a, -mkt_s) if mkt_s is not None else "n/a"
            print(f"  Market: {ms}, total {mkt_t}   |   {h} covers {row['home_cover_pct']}%, "
                  f"{a} covers {row['away_cover_pct']}%, Over {row['over_pct']}%")
        print(f"  Adjustments: weather {p['weather_total']:+.1f} total ({p['weather']}) | "
              f"injuries {p['inj_margin']:+.1f} margin / {p['inj_total']:+.1f} total"
              f"{' [' + p['inj_notes'] + ']' if p['inj_notes'] else ''} | "
              f"tendencies {h} {p['tend_home']:+.1f}, {a} {p['tend_away']:+.1f}")

    if not rows:
        raise SystemExit("Nothing simulated.")
    pd.DataFrame(rows).to_csv(OUT / "sim_results.csv", index=False)
    np.savez_compressed(OUT / "sim_draws.npz", **draws)
    print(f"\nSaved {len(rows)} games -> {OUT / 'sim_results.csv'}  (+ sim_draws.npz)")
    print("Next: python kalshi_picks.py")


if __name__ == "__main__":
    main()
