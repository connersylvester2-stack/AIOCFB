#!/usr/bin/env python3
"""RECORD KEEPER - a public, honest track record for the model's picks.

What it does
  log    After each pipeline run, saves the model's picks (every game, all three market types) to
         ledger/picks_ledger.csv with a timestamp. Only picks logged BEFORE kickoff are accepted, and
         a logged pick is never edited or deleted - only its result is filled in later. Because the
         ledger is committed to git after every run, the commit history is an audit trail.
  grade  Once a game is final, grades each open pick. The outcome comes from Kalshi's own settlement
         when available (that is what actually pays) and is cross-checked against the final score;
         any disagreement is flagged in the `check` column (usually means a parsing bug worth fixing).
  report Builds ledger/record_summary.md (tables), ledger/weekly_recap.txt (tweet-ready text) and
         ledger/recap_card.png (an image for X).

How picks are scored
  * Every pick is a flat 1-unit stake at the logged price, Kalshi fee included (units = profit per $1 risked).
  * The headline record is the BET-tagged picks (edge at or above your --min-edge). LEAN and PASS picks
    are tracked separately so nothing is hidden.
  * The same game can get more than one logged pick if the model changes its mind between runs; all of them
    are kept and graded.
  * Not tracked yet: closing-price value. That needs Kalshi price history and I haven't been able to test it.

Run:  python record_keeper.py            (log + grade + report)
      python record_keeper.py grade      (or: log | report)
      python record_keeper.py report --week 5
"""
import datetime as dt

import numpy as np
import pandas as pd

from common import BASE, OUT, apply_std_args, fetch_games, http_json, parse_utc, std_parser
from kalshi_picks import FEE_RATE, KALSHI

LEDGER_DIR = BASE / "ledger"
LEDGER = LEDGER_DIR / "picks_ledger.csv"
PICK_COLS = ["logged_at", "season", "week", "game_id", "game", "home", "away", "kickoff_utc", "market", "status",
             "pick", "action", "side", "yes_team", "opp", "strike", "total_under", "price_cents", "price_dollars",
             "fee", "model_prob", "edge_cents", "model_line", "model_total", "ticker", "market_title"]
GRADE_COLS = ["home_pts", "away_pts", "yes_outcome", "result", "units", "graded_by", "graded_at", "check"]
ALL_COLS = PICK_COLS + GRADE_COLS
MARKET_NAMES = {"winner": "Winners", "spread": "Spreads", "total": "Totals"}


# ----------------------------------------------------------------------------- ledger io
def load_ledger():
    if not LEDGER.exists():
        return pd.DataFrame(columns=ALL_COLS).astype({c: object for c in ALL_COLS})
    df = pd.read_csv(LEDGER, dtype={"game_id": str, "ticker": str})
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ALL_COLS]
    df[GRADE_COLS] = df[GRADE_COLS].astype(object)
    return df


def save_ledger(df):
    LEDGER_DIR.mkdir(exist_ok=True)
    df[ALL_COLS].to_csv(LEDGER, index=False)


# ----------------------------------------------------------------------------- log
def log_picks(year):
    src = OUT / "model_picks_all_games.csv"
    if not src.exists():
        print("log: no model_picks_all_games.csv this run - nothing to log.")
        return
    df = pd.read_csv(src, dtype={"game_id": str, "ticker": str})
    if "kickoff_utc" not in df.columns or "game_id" not in df.columns:
        print("log: picks file is from an older version of kalshi_picks.py - re-run it first.")
        return
    df = df[df["action"].notna()]
    now = dt.datetime.now(dt.timezone.utc)
    led = load_ledger()
    seen = set(zip(led["game_id"].astype(str), led["market"], led["ticker"].astype(str), led["action"]))
    new, started, dupes = [], 0, 0
    for r in df.to_dict("records"):
        ko = parse_utc(r.get("kickoff_utc"))
        if ko is not None and ko.tzinfo is None:
            ko = ko.replace(tzinfo=dt.timezone.utc)
        if ko is None or ko <= now:
            started += 1                       # already started (or unknown kickoff): can't be verified
            continue
        key = (str(r["game_id"]), r["market"], str(r["ticker"]), r["action"])
        if key in seen:
            dupes += 1
            continue
        seen.add(key)
        row = {c: r.get(c) for c in PICK_COLS}
        row["logged_at"] = now.isoformat(timespec="seconds")
        row["season"] = year
        new.append(row)
    if new:
        add = pd.DataFrame(new)
        for c in ALL_COLS:
            if c not in add.columns:
                add[c] = np.nan
        add[GRADE_COLS] = add[GRADE_COLS].astype(object)
        led = pd.concat([led, add[ALL_COLS]], ignore_index=True) if len(led) else add[ALL_COLS]
        save_ledger(led)
    print(f"log: {len(new)} new picks logged, {dupes} already logged, {started} skipped (game already started).")


# ----------------------------------------------------------------------------- grade
def yes_outcome_from_score(r, hp, ap):
    """True / False / None (push) for the YES side of the market, from the final score."""
    kind = r["market"]
    if kind == "winner":
        pts_y, pts_o = (hp, ap) if r["yes_team"] == r["home"] else (ap, hp)
        return None if pts_y == pts_o else pts_y > pts_o
    strike = float(r["strike"])
    if kind == "spread":
        margin = (hp - ap) if r["yes_team"] == r["home"] else (ap - hp)
        return None if margin == strike else margin > strike
    if kind == "total":
        tot = hp + ap
        if tot == strike:
            return None
        over = tot > strike
        under_market = str(r["total_under"]).lower() in ("true", "1", "1.0")
        return (not over) if under_market else over
    return None


def kalshi_result(ticker):
    """'yes' / 'no' once Kalshi has settled the market, else None."""
    try:
        d = http_json(f"{KALSHI}/markets/{ticker}", {}, ttl=0, retries=1)
    except RuntimeError:
        return None
    m = d.get("market", d) if isinstance(d, dict) else {}
    res = str(m.get("result") or "").lower()
    return res if res in ("yes", "no") else None


def grade_picks():
    led = load_ledger()
    if led.empty:
        print("grade: ledger is empty.")
        return
    todo = led[led["result"].isna()]
    if todo.empty:
        print("grade: nothing waiting to be graded.")
        return
    finals = {}
    for y in todo["season"].dropna().unique():
        finals[int(y)] = {str(g["id"]): g for g in fetch_games(int(y), ttl=900)}
    done = mismatches = 0
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for i, r in todo.iterrows():
        g = finals.get(int(r["season"]), {}).get(str(r["game_id"]))
        if not g or not g["completed"] or g["home_pts"] is None or g["away_pts"] is None:
            continue
        hp, ap = float(g["home_pts"]), float(g["away_pts"])
        by_score = yes_outcome_from_score(r, hp, ap)
        res_k = kalshi_result(r["ticker"])
        check, graded_by = "", "score"
        if res_k is not None:
            yes, graded_by = (res_k == "yes"), "kalshi"
            if by_score is not None and by_score != yes:
                check = f"MISMATCH: score says {'yes' if by_score else 'no'}, Kalshi settled {res_k}"
                mismatches += 1
        else:
            yes = by_score
        if yes is None:
            result, units = "P", 0.0
        else:
            won = yes if r["side"] == "YES" else (not yes)
            cost = float(r["price_dollars"])
            fee = float(r["fee"]) if not pd.isna(r["fee"]) else FEE_RATE * cost * (1 - cost)
            result = "W" if won else "L"
            units = round(1.0 / (cost + fee) - 1.0, 4) if won else -1.0
        led.at[i, "home_pts"], led.at[i, "away_pts"] = hp, ap
        led.at[i, "yes_outcome"] = "" if yes is None else str(bool(yes))
        led.at[i, "result"], led.at[i, "units"] = result, units
        led.at[i, "graded_by"], led.at[i, "graded_at"], led.at[i, "check"] = graded_by, now, check
        done += 1
    save_ledger(led)
    print(f"grade: {done} picks graded, {len(todo) - done} still waiting on final scores"
          + (f", {mismatches} Kalshi/score MISMATCHES (see the `check` column)" if mismatches else "") + ".")


# ----------------------------------------------------------------------------- report
def record(df):
    w, l, p = (df["result"] == "W").sum(), (df["result"] == "L").sum(), (df["result"] == "P").sum()
    n = w + l
    units = float(df["units"].sum())
    return dict(n=int(len(df)), w=int(w), l=int(l), p=int(p), win_pct=100 * w / n if n else np.nan,
                units=units, roi=100 * units / n if n else np.nan,
                avg_price=float(df["price_cents"].mean()) if len(df) else np.nan,
                avg_prob=100 * float(df["model_prob"].mean()) if len(df) else np.nan)


def fmt_rec(r):
    return f"{r['w']}-{r['l']}" + (f"-{r['p']}" if r["p"] else "")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def num(x, spec="{:+.1f}"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else spec.format(x)


def make_card(path, week, wk, season, by_market, footer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, fg, dim, good, bad = "#0f172a", "#f8fafc", "#94a3b8", "#4ade80", "#f87171"
    fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=bg)
    fig.text(0.05, 0.89, f"CFB MODEL  |  WEEK {week} RESULTS", color=fg, fontsize=30, fontweight="bold")
    fig.text(0.05, 0.825, "BET-rated picks  -  flat 1u stake  -  Kalshi fees included", color=dim, fontsize=15)
    for x, title, r in ((0.05, "THIS WEEK", wk), (0.55, f"{season} SEASON", by_market["__season__"])):
        fig.text(x, 0.72, title, color=dim, fontsize=14)
        fig.text(x, 0.58, fmt_rec(r), color=fg, fontsize=54, fontweight="bold")
        col = good if r["units"] >= 0 else bad
        fig.text(x, 0.50, f"{r['units']:+.1f}u   ROI {num(r['roi'], '{:+.0f}')}%", color=col, fontsize=22, fontweight="bold")
    fig.text(0.05, 0.38, "THIS WEEK BY MARKET", color=dim, fontsize=14)
    y = 0.31
    for key in ("spread", "total", "winner"):
        r = by_market.get(key)
        if not r:
            continue
        fig.text(0.05, y, MARKET_NAMES[key], color=fg, fontsize=20)
        fig.text(0.30, y, fmt_rec(r), color=fg, fontsize=20, fontweight="bold")
        fig.text(0.45, y, f"{r['units']:+.1f}u", color=good if r["units"] >= 0 else bad, fontsize=20, fontweight="bold")
        y -= 0.08
    fig.text(0.05, 0.05, footer, color=dim, fontsize=12)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)


def build_report(week=None):
    led = load_ledger()
    g = led[led["result"].isin(["W", "L", "P"])].copy()
    if g.empty:
        print("report: nothing graded yet - picks need to be logged before kickoff and games need to finish.")
        LEDGER_DIR.mkdir(exist_ok=True)
        (LEDGER_DIR / "record_summary.md").write_text("# CFB model record\n\nNothing graded yet.\n")
        return
    for c in ("units", "price_cents", "model_prob", "week"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    bet = g[g["status"] == "BET"]
    season = int(g["season"].max())

    lines = ["# CFB model - track record", "",
             "Every pick is logged before kickoff and never edited. Flat 1-unit stake at the logged Kalshi price, "
             "fees included. Headline = BET-tagged picks; LEAN/PASS are tracked too.", ""]
    n_bet = len(bet)
    if n_bet < 100:
        lines += [f"> Sample size: {n_bet} graded BET picks. Records this small are mostly noise - "
                  "judge the model over hundreds of picks, not dozens.", ""]

    rows = []
    for name, df in (("BET", bet), ("LEAN", g[g["status"] == "LEAN"]), ("PASS", g[g["status"] == "PASS"]), ("ALL", g)):
        r = record(df)
        rows.append([name, r["n"], fmt_rec(r), num(r["win_pct"], "{:.1f}%"), num(r["units"]), num(r["roi"], "{:+.1f}%"),
                     num(r["avg_price"], "{:.0f}c"), num(r["avg_prob"], "{:.0f}%")])
    lines += ["## By status", "", md_table(["Status", "Picks", "W-L", "Win %", "Units", "ROI", "Avg price", "Avg model prob"], rows), ""]

    rows = []
    for key, name in MARKET_NAMES.items():
        r = record(bet[bet["market"] == key])
        if r["n"]:
            rows.append([name, r["n"], fmt_rec(r), num(r["win_pct"], "{:.1f}%"), num(r["units"]), num(r["roi"], "{:+.1f}%")])
    if rows:
        lines += ["## BET picks by market", "", md_table(["Market", "Picks", "W-L", "Win %", "Units", "ROI"], rows), ""]

    rows = []
    for wkno, df in bet.groupby("week"):
        r = record(df)
        rows.append([int(wkno), r["n"], fmt_rec(r), num(r["units"]), num(r["roi"], "{:+.1f}%")])
    if rows:
        lines += ["## BET picks by week", "", md_table(["Week", "Picks", "W-L", "Units", "ROI"], rows), ""]

    rows = []
    g["bucket"] = pd.cut(g["model_prob"], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], right=False, labels=["50-60%", "60-70%", "70-80%", "80-90%", "90%+"])
    for b, df in g.groupby("bucket", observed=True):
        d = df[df["result"].isin(["W", "L"])]
        if len(d):
            rows.append([b, len(d), num(100 * d["model_prob"].mean(), "{:.0f}%"), num(100 * (d["result"] == "W").mean(), "{:.0f}%"),
                         num(d["price_cents"].mean(), "{:.0f}c")])
    if rows:
        lines += ["## Calibration (all graded picks)", "",
                  "If the model is honest, 'Model said' and 'Actually won' should be close.", "",
                  md_table(["Model said", "Picks", "Avg model prob", "Actually won", "Avg price paid"], rows), ""]

    bad = led[led["check"].astype(str).str.startswith("MISMATCH")]
    if len(bad):
        lines += [f"## Needs a look: {len(bad)} pick(s) where Kalshi's settlement and the final score disagree", "",
                  "These are graded by Kalshi's settlement. A mismatch usually means a strike or direction was parsed "
                  "wrong - send them to me.", "", md_table(["Game", "Pick", "Check"], [[r["game"], r["pick"], r["check"]] for _, r in bad.iterrows()]), ""]

    LEDGER_DIR.mkdir(exist_ok=True)
    (LEDGER_DIR / "record_summary.md").write_text("\n".join(lines))

    # ---- weekly recap text + image
    wk_no = week if week is not None else (int(bet["week"].max()) if n_bet else int(g["week"].max()))
    wk_bet = bet[bet["week"] == wk_no]
    wk_rec, season_rec = record(wk_bet), record(bet)
    by_market = {k: record(wk_bet[wk_bet["market"] == k]) for k in MARKET_NAMES if len(wk_bet[wk_bet["market"] == k])}
    by_market["__season__"] = season_rec
    all_wk = record(g[g["week"] == wk_no])
    parts = " | ".join(f"{MARKET_NAMES[k]} {fmt_rec(v)}" for k, v in by_market.items() if k != "__season__")
    footer = "Every pick logged before kickoff  -  not betting advice  -  18+"
    recap = [f"CFB MODEL - WEEK {wk_no} RESULTS", "",
             f"BET picks: {fmt_rec(wk_rec)} ({wk_rec['units']:+.1f}u, ROI {num(wk_rec['roi'], '{:+.0f}')}%)"]
    if parts:
        recap.append(parts)
    recap += [f"All picks on every game: {fmt_rec(all_wk)} ({all_wk['units']:+.1f}u)", "",
              f"{season} season, BET picks: {fmt_rec(season_rec)} ({season_rec['units']:+.1f}u, ROI {num(season_rec['roi'], '{:+.0f}')}%)",
              "", "Every pick is logged before kickoff. Flat 1u stakes, fees included. Not betting advice. 18+."]
    (LEDGER_DIR / "weekly_recap.txt").write_text("\n".join(recap) + "\n")
    try:
        make_card(LEDGER_DIR / "recap_card.png", wk_no, wk_rec, season, by_market, footer)
        card = "recap_card.png"
    except Exception as e:                      # matplotlib missing or font problem - text outputs still written
        card = f"(image skipped: {e})"
    print("\n".join(recap))
    print(f"\nreport: wrote ledger/record_summary.md, weekly_recap.txt, {card}")


# ----------------------------------------------------------------------------- main
def main():
    ap = std_parser("Log, grade and report the model's pick record")
    ap.add_argument("action", nargs="?", default="all", choices=["all", "log", "grade", "report"])
    ap.add_argument("--week", type=int, default=None, help="week for the recap (default: latest graded)")
    args = ap.parse_args()
    apply_std_args(args)
    if args.action in ("all", "log"):
        log_picks(args.year)
    if args.action in ("all", "grade"):
        grade_picks()
    if args.action in ("all", "report"):
        build_report(args.week)


if __name__ == "__main__":
    main()
