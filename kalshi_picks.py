#!/usr/bin/env python3
"""SCRIPT 4 - Turn the simulations into clear Kalshi bets.

Reads   output/sim_results.csv + sim_draws.npz   (script 3)
        output/power_rankings.csv                (script 1, for context)
        output/cover_stats.csv                   (script 2, for context only - NOT used in the math)
Pulls   live open markets from Kalshi's public API (no login needed for market data):
        KXNCAAFGAME (winner), KXNCAAFSPREAD (win-by-over-X), KXNCAAFTOTAL (over/under)
Prices  every market straight from the simulated scores, so any strike is covered.

Method
  * model probability comes from the simulated scores (the model's OWN line - nothing is taken from sportsbooks)
  * for every available market it checks both sides (YES / NO) and keeps the side the model likes
  * edge = model probability - price paid - Kalshi fee; bets are ranked by edge
  * optional --shrink blends the probability toward the Kalshi price (default 0 = pure model)
  * bet size in units: 1u / 2.5u / 5u by edge size (1 unit = $1 by default), capped at 1u and flagged when
    the model and the market disagree by 30+ points (usually means check injuries/QB news)

Output: (1) the model's pick on EVERY simulated game -> output/model_picks_all_games.csv
        (2) the strongest bets only                        -> output/recommendations.csv
Run:    python kalshi_picks.py       (optional: --min-edge 5 --top 10 --shrink 0 --unit 1)
"""
import math
import re

import numpy as np
import pandas as pd

from common import OUT, apply_std_args, build_matcher, find_teams, http_json, std_parser

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
# kind -> Kalshi series tickers to try (first one that returns open markets is used)
SERIES = [("winner", ["KXNCAAFGAME"]), ("spread", ["KXNCAAFSPREAD"]),
          ("total", ["KXNCAAFTOTAL", "KXNCAAFTOTALS", "KXNCAAFOU"])]
FEE_RATE = 0.07            # Kalshi taker fee = 0.07 * P * (1-P) per contract; verify against current fee schedule
MAX_BID_ASK = 0.08         # skip markets whose bid/ask gap is wider than 8 cents
PRICE_RANGE = (0.08, 0.92)  # skip extreme longshots / near-locks (model error dominates there)
BIG_DISAGREE = 0.30
# half / quarter / team-total markets share the same series; we only model full-game outcomes
PARTIAL_RE = re.compile(r"\b(half|quarter|1st|2nd|3rd|4th|1h|2h|q1|q2|q3|q4|team total)\b", re.I)


# ----------------------------------------------------------------------------- Kalshi access
def fetch_series(series):
    markets, cursor = [], None
    while True:
        params = dict(series_ticker=series, status="open", limit=1000)
        if cursor:
            params["cursor"] = cursor
        d = http_json(f"{KALSHI}/markets", params, ttl=0)
        markets += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor:
            return markets


def event_title(event_ticker, cache):
    if event_ticker not in cache:
        try:
            e = http_json(f"{KALSHI}/events/{event_ticker}", {}, ttl=24 * 3600).get("event", {})
            cache[event_ticker] = f"{e.get('title', '')} {e.get('sub_title', '')}"
        except RuntimeError:
            cache[event_ticker] = ""
    return cache[event_ticker]


def px(m, name):
    """Price in dollars (0-1) or None. Handles both *_dollars strings and legacy cents."""
    v = m.get(f"{name}_dollars")
    if v not in (None, ""):
        v = float(v)
    else:
        v = m.get(name)
        v = None if v is None else v / 100.0
    return v if v is not None and 0.0 < v < 1.0 else None


def quote(m):
    """(yes_bid, yes_ask, no_ask, mid) all in dollars where available."""
    yb, ya, na, nb = px(m, "yes_bid"), px(m, "yes_ask"), px(m, "no_ask"), px(m, "no_bid")
    if ya is None and nb is not None:
        ya = 1.0 - nb
    if na is None and yb is not None:
        na = 1.0 - yb
    if yb is None and na is not None:
        yb = 1.0 - na
    mid = (yb + ya) / 2.0 if (yb is not None and ya is not None) else px(m, "last_price")
    return yb, ya, na, mid


# ----------------------------------------------------------------------------- pricing
def num_from(text):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", text or "")
    return float(m.group(1)) if m else None


def model_prob(kind, m, row, draws, matcher):
    """Return (p_yes, meta) or None. meta holds team/strike/direction for plain-English text."""
    home, away = row["home"], row["away"]
    hp, ap = draws[:, 0].astype(float), draws[:, 1].astype(float)
    ysub, title = m.get("yes_sub_title") or "", m.get("title") or ""
    team = None
    for txt in (ysub, title):
        t = [x for x in find_teams(txt, matcher) if x in (home, away)]
        if t:
            team = t[0]
            break
    strike = m.get("floor_strike")
    strike = float(strike) if strike not in (None, "") else num_from(ysub) or num_from(title)

    if kind == "winner":
        if team is None:
            return None
        win = (hp > ap) if team == home else (ap > hp)
        return float(win.mean()), dict(team=team, opp=away if team == home else home)
    if kind == "spread":
        if team is None or strike is None:
            return None
        margin = (hp - ap) if team == home else (ap - hp)
        p = (margin >= strike) if m.get("strike_type") == "greater_or_equal" else (margin > strike)
        return float(p.mean()), dict(team=team, opp=away if team == home else home, strike=strike)
    if kind == "total":
        if strike is None:
            return None
        tot = hp + ap
        text = (ysub or title).lower()
        under = m.get("strike_type") == "less" or ("under" in text and "over" not in text)
        p = (tot < strike) if under else (tot > strike)
        return float(p.mean()), dict(strike=strike, under=under)
    return None


def describe(kind, meta, side):
    """Plain-English bet description for the side we are buying."""
    if kind == "winner":
        return (f"{meta['team']} to WIN the game" if side == "YES"
                else f"{meta['team']} to LOSE the game ({meta['opp']} wins)")
    if kind == "spread":
        s = meta["strike"]
        whole = s == int(s)
        if side == "YES":
            return f"{meta['team']} wins by MORE than {s:g}"
        return (f"{meta['team']} wins by {s:g} or FEWER, or loses" if whole
                else f"{meta['team']} wins by {int(s)} or fewer, or loses")
    over = not meta["under"]
    if side == "NO":
        over = not over
    return f"{'OVER' if over else 'UNDER'} {meta['strike']:g} total points"


def units_for(edge, disagree):
    if edge >= 0.12:
        u = 5.0
    elif edge >= 0.08:
        u = 2.5
    else:
        u = 1.0
    return 1.0 if disagree >= BIG_DISAGREE else u


def context_line(kind, row, cover, ranks, meta):
    h, a = row["home"], row["away"]
    bits = [f"#{ranks.get(a, '?')} {a} at #{ranks.get(h, '?')} {h}"]
    if cover is not None:
        def cs(t):
            r = cover.get(t)
            if not r:
                return None
            return r
        ch, ca = cs(h), cs(a)
        if kind in ("winner", "spread") and ch and ca:
            bits.append(f"ATS: {h} {ch['ats_record']}, {a} {ca['ats_record']}")
        if kind == "total" and ch and ca:
            bits.append(f"Overs: {h} {ch['over_record']}, {a} {ca['over_record']}")
    return " | ".join(bits)


# ----------------------------------------------------------------------------- per-game board
LABEL = {"winner": "WINNER", "spread": "SPREAD", "total": "TOTAL "}


def status_of(c, min_edge):
    e = c["edge"] * 100
    if e >= min_edge and c["ok"]:
        return "BET"
    return "LEAN" if e > 0 else "PASS"


def pick_for(cands, gid, kind):
    cs = [c for c in cands if str(c["row"]["game_id"]) == gid and c["kind"] == kind]
    if not cs:
        return None
    # only sides the model itself believes (>= 50%), then the best-priced rung of that ladder
    believed = [c for c in cs if c["p_adj"] >= 0.5] or cs
    # prefer mid-priced rungs (25-80c) over 88c-type rungs that risk a lot to win a little
    liquid = [c for c in believed if c["ok"] and 0.25 <= c["cost"] <= 0.80]
    return max(liquid or believed, key=lambda c: c["edge"])


def game_board(sim_rows, cands, min_edge):
    """The model's pick for winner / spread / total on EVERY simulated game, with the Kalshi price."""
    print("\n" + "=" * 78)
    print(" THE MODEL'S PICK ON EVERY GAME")
    print("=" * 78)
    print(f" BET  = model edge of {min_edge:g}c+ per contract after fees")
    print(" LEAN = model likes it, but the edge is small")
    print(" PASS = the model's side is priced too high to be worth it")
    out = []
    for r in sim_rows:
        gid = str(r["game_id"])
        print(f"\n{r['away']} @ {r['home']}   ({r['kickoff']})")
        print(f"  Model: {r['model_line']}, total {r['model_total']}   |   "
              f"{r['home']} wins {r['home_win_pct']:.0f}%, {r['away']} wins {r['away_win_pct']:.0f}%")
        for kind in ("winner", "spread", "total"):
            c = pick_for(cands, gid, kind)
            if c is None:
                print(f"  {LABEL[kind]}: no open Kalshi market found")
                out.append(dict(game=f"{r['away']} @ {r['home']}", kickoff=r["kickoff"], model_line=r["model_line"],
                                model_total=r["model_total"], market=kind, pick="no open Kalshi market"))
                continue
            text = describe(kind, c["meta"], c["side"])
            if kind == "winner":               # always name the team the model picks to WIN
                fav = c["meta"]["team"] if c["side"] == "YES" else c["meta"]["opp"]
                text = f"{fav} to WIN the game"
            st = status_of(c, min_edge)
            note = "" if c["ok"] else "   (thin market or extreme price)"
            print(f"  {LABEL[kind]}: {text}")
            print(f"          BUY {c['side']} @ {round(c['cost'] * 100)}c | model {c['p_adj']:.0%} | "
                  f"edge {c['edge'] * 100:+.1f}c   [{st}]{note}")
            print(f"          on Kalshi: \"{c['m'].get('title', '')}\" -> tap {c['side']}")
            meta = c["meta"]
            out.append(dict(game=f"{r['away']} @ {r['home']}", kickoff=r["kickoff"], model_line=r["model_line"],
                            model_total=r["model_total"], home_win_pct=r["home_win_pct"], market=kind, pick=text,
                            action=f"BUY {c['side']}", price_cents=round(c["cost"] * 100),
                            model_prob=round(c["p_adj"], 3), edge_cents=round(c["edge"] * 100, 1), status=st,
                            market_title=c["m"].get("title"), ticker=c["m"].get("ticker"),
                            game_id=gid, week=r.get("week"), kickoff_utc=r.get("kickoff_utc"),
                            home=r["home"], away=r["away"], side=c["side"], yes_team=meta.get("team"),
                            opp=meta.get("opp"), strike=meta.get("strike"), total_under=meta.get("under"),
                            price_dollars=round(c["cost"], 4), fee=round(c["fee"], 4)))
    pd.DataFrame(out).to_csv(OUT / "model_picks_all_games.csv", index=False)
    print(f"\nSaved -> {OUT / 'model_picks_all_games.csv'}")


# ----------------------------------------------------------------------------- main
def main():
    ap = std_parser("Recommend Kalshi college football markets from the simulations")
    ap.add_argument("--min-edge", type=float, default=5.0, help="minimum edge in cents per contract (default 5)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--shrink", type=float, default=0.0, help="0=trust the model fully (default), 1=trust the Kalshi price fully")
    ap.add_argument("--max-disagree", type=float, default=1.0,
                    help="optionally SKIP bets where model and market differ by more than this (default: off, just flag)")
    ap.add_argument("--unit", type=float, default=1.0, help="dollars per unit")
    ap.add_argument("--max-per-game", type=int, default=2)
    args = ap.parse_args()
    apply_std_args(args)

    sp, dp = OUT / "sim_results.csv", OUT / "sim_draws.npz"
    if not sp.exists() or not dp.exists():
        raise SystemExit("Run simulate_games.py first (needs sim_results.csv and sim_draws.npz).")
    sim = pd.read_csv(sp, dtype={"game_id": str})
    draws = np.load(dp)
    ranks, cover = {}, None
    if (OUT / "power_rankings.csv").exists():
        pr = pd.read_csv(OUT / "power_rankings.csv")
        ranks = dict(zip(pr["team"], pr["overall_rank"]))
    if (OUT / "cover_stats.csv").exists():
        cover = pd.read_csv(OUT / "cover_stats.csv").set_index("team").to_dict("index")

    sim_teams = sorted(set(sim["home"]) | set(sim["away"]))
    matcher = build_matcher(sim_teams)
    pair_to_row = {frozenset((r.home, r.away)): r._asdict() for r in sim.itertuples(index=False)}

    print("Fetching Kalshi markets...")
    event_cache, seen, matched, cands = {}, 0, 0, []
    suffix_row = {}          # e.g. "26OCT03UTSARICE" -> that game; links winner/spread/total markets of one game
    diag = []
    for kind, options in SERIES:
        ms, used = [], options[0]
        for series in options:
            try:
                ms = fetch_series(series)
            except RuntimeError as e:
                print(f"  could not fetch {series}: {e}")
                continue
            if ms:
                used = series
                break
        seen += len(ms)
        st = dict(series=used, kind=kind, fetched=len(ms), partial=0, game=0, no_prob=0, unpriced=0, used=0, sample=None)
        for m in ms:
            if PARTIAL_RE.search(f"{m.get('title', '')} {m.get('yes_sub_title', '')} {m.get('no_sub_title', '')}"):
                st["partial"] += 1
                continue
            et = m.get("event_ticker") or ""
            suffix = et.split("-", 1)[1] if "-" in et else None
            blob = f"{m.get('title', '')} {m.get('yes_sub_title', '')} {m.get('no_sub_title', '')}"
            found = find_teams(blob, matcher)
            row = next((pair_to_row[frozenset(c)] for c in _pairs(found) if frozenset(c) in pair_to_row), None)
            if row is None and found:
                found = find_teams(blob + " " + event_title(et, event_cache), matcher)
                row = next((pair_to_row[frozenset(c)] for c in _pairs(found) if frozenset(c) in pair_to_row), None)
            if row is None and suffix in suffix_row:        # totals often don't name the teams
                row = suffix_row[suffix]
            if row is None:
                if st["sample"] is None:
                    st["sample"] = m
                continue
            if suffix:
                suffix_row.setdefault(suffix, row)
            st["game"] += 1
            mp = model_prob(kind, m, row, draws[str(row["game_id"])], matcher)
            if mp is None:
                st["no_prob"] += 1
                if st["sample"] is None:
                    st["sample"] = m
                continue
            matched += 1
            p_model, meta = mp
            yb, ya, na, mid = quote(m)
            if mid is None:
                st["unpriced"] += 1
                continue
            st["used"] += 1
            p_adj = (1 - args.shrink) * p_model + args.shrink * mid
            spread_ok = (ya is None or yb is None) or (ya - yb) <= MAX_BID_ASK
            for side, cost, p in (("YES", ya, p_adj), ("NO", na, 1.0 - p_adj)):
                if cost is None:
                    continue
                ok = PRICE_RANGE[0] <= cost <= PRICE_RANGE[1] and spread_ok
                fee = FEE_RATE * cost * (1 - cost)
                edge = p - cost - fee
                disagree = abs(p_model - mid)
                cands.append(dict(
                    kind=kind, row=row, meta=meta, m=m, side=side, cost=cost, fee=fee, edge=edge, ok=ok,
                    p_model=p_model if side == "YES" else 1 - p_model, p_adj=p,
                    p_market=mid if side == "YES" else 1 - mid, disagree=disagree,
                    volume=m.get("volume") or m.get("volume_24h") or 0))
        diag.append(st)
    for st in diag:
        line = (f"  {st['kind']:<6} ({st['series']}): {st['fetched']} open markets, {st['game']} matched your games, "
                f"{st['used']} priced")
        if st["partial"]:
            line += f", {st['partial']} half/quarter/team-total markets ignored"
        if st["unpriced"]:
            line += f", {st['unpriced']} had no quotes"
        if st["no_prob"]:
            line += f", {st['no_prob']} couldn't be read"
        print(line)
        if st["used"] == 0 and st["sample"]:
            s = st["sample"]
            print(f"         !! nothing usable - sample market: ticker={s.get('ticker')} | title={s.get('title')!r} | "
                  f"yes={s.get('yes_sub_title')!r} | strike={s.get('floor_strike')} | type={s.get('strike_type')}")
        elif st["fetched"] == 0:
            print(f"         !! no open markets returned for {st['series']} - the series ticker may be different")
    print(f"  {seen} open markets checked, {matched} matched to your simulated games.")

    game_board(sim.to_dict("records"), cands, args.min_edge)

    good = [c for c in cands if c["ok"] and c["edge"] * 100 >= args.min_edge]
    skipped = [c for c in good if c["disagree"] > args.max_disagree]
    good = [c for c in good if c["disagree"] <= args.max_disagree]
    if skipped:
        print(f"  Skipped {len(skipped)} bet(s) where model and market differ by more than "
              f"{args.max_disagree:.0%} - almost always a model/input problem, not an edge "
              f"(--max-disagree 1.0 to show them).")
    good.sort(key=lambda c: -c["edge"])
    picks, per_game, per_kind = [], {}, set()
    for c in good:                                   # best rung per game+market type, limited per game
        gk = (c["row"]["game_id"], c["kind"])
        gid = c["row"]["game_id"]
        if gk in per_kind or per_game.get(gid, 0) >= args.max_per_game:
            continue
        per_kind.add(gk)
        per_game[gid] = per_game.get(gid, 0) + 1
        picks.append(c)
        if len(picks) >= args.top:
            break

    if not picks:
        print(f"\nNo bets clear the {args.min_edge:.0f}-cent edge bar right now - see the LEAN picks above, "
              "or try --min-edge 3.")
        return

    out_rows = []
    print("\n" + "=" * 78)
    print(f" STRONGEST BETS (BET-rated only, best per game)   (1 unit = ${args.unit:g})")
    print("=" * 78)
    for i, c in enumerate(picks, 1):
        r, m = c["row"], c["m"]
        text = describe(c["kind"], c["meta"], c["side"])
        units = units_for(c["edge"], c["disagree"])
        contracts = max(1, int(units * args.unit / c["cost"]))
        cost_total = contracts * c["cost"]
        cents = round(c["cost"] * 100)
        flags = []
        if c["disagree"] >= BIG_DISAGREE:
            flags.append("model and market disagree a lot - check injuries/QB news before betting")
        if int(float(c["volume"] or 0)) < 25:
            flags.append("thin volume")
        if c["kind"] == "total":
            why = f"Projected total {r['model_total']} vs Kalshi line {c['meta']['strike']:g}."
        elif c["kind"] == "spread":
            team, opp = c["meta"]["team"], c["meta"]["opp"]
            proj = r["model_margin_home"] if team == r["home"] else -r["model_margin_home"]
            who = f"{team} by {proj:.1f}" if proj >= 0 else f"{opp} by {-proj:.1f}"
            why = f"Projection: {who}; the bet is {team} winning by more than {c['meta']['strike']:g}."
        else:
            why = f"Projected line: {r['model_line']} (home win {r['home_win_pct']}%)."
        ctx = context_line(c["kind"], r, cover, ranks, c["meta"])
        print(f"\n#{i}  BUY {c['side']}:  {text}")
        print(f"    Game     : {r['away']} @ {r['home']}   ({r['kickoff']})")
        print(f"    Price    : {cents}c per contract   ->   buy {contracts} contract(s) = ${cost_total:.2f}"
              f"   ({units:g} unit{'s' if units != 1 else ''})")
        print(f"    Chance   : model {c['p_model']:.0%} | blended {c['p_adj']:.0%} | market {c['p_market']:.0%}"
              f"   ->  edge +{c['edge'] * 100:.1f}c per contract after fees")
        print(f"    Why      : {why}")
        print(f"    Context  : {ctx}")
        print(f"    On Kalshi: find \"{m.get('title', '')}\"  ({m.get('ticker')})  and tap {c['side']}")
        for f in flags:
            print(f"    !! {f}")
        out_rows.append(dict(
            rank=i, action=f"BUY {c['side']}", bet=text, game=f"{r['away']} @ {r['home']}", kickoff=r["kickoff"],
            price_cents=cents, units=units, contracts=contracts, cost_dollars=round(cost_total, 2),
            model_prob=round(c["p_model"], 3), blended_prob=round(c["p_adj"], 3), market_prob=round(c["p_market"], 3),
            edge_cents=round(c["edge"] * 100, 1), why=why, context=ctx, flags="; ".join(flags),
            market_title=m.get("title"), ticker=m.get("ticker")))
    pd.DataFrame(out_rows).to_csv(OUT / "recommendations.csv", index=False)
    print(f"\nSaved -> {OUT / 'recommendations.csv'}")
    print("Reminder: model output, not a guarantee - confirm injuries/lineups before you bet.")


def _pairs(teams):
    return [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]]


if __name__ == "__main__":
    main()
