"""Shared helpers for the SpreadX college football toolkit.

Used by:
  power_rankings.py -> output/power_rankings.csv, output/model_params.json
  cover_stats.py    -> output/cover_stats.csv
  simulate_games.py -> output/sim_results.csv, output/sim_draws.npz
  kalshi_picks.py   -> output/recommendations.csv

Setup: put  CFBD_API_KEY=your_key  in a file named .env next to these scripts
(free key from collegefootballdata.com) or export it in your shell.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from statistics import median

import requests

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
CACHE = BASE / ".cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

CFBD_URL = "https://api.collegefootballdata.com"


# --------------------------------------------------------------------------- env / args
def _load_env():
    p = BASE / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()


def current_season():
    """Season year: Jan-May still belongs to the season that just ended."""
    t = dt.date.today()
    return t.year if t.month >= 6 else t.year - 1


def std_parser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--year", type=int, default=current_season(), help="season (default: current)")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached API responses")
    return ap


def apply_std_args(args):
    if getattr(args, "no_cache", False):
        os.environ["SPREADX_NO_CACHE"] = "1"


# --------------------------------------------------------------------------- HTTP + cache
def http_json(url, params=None, headers=None, ttl=6 * 3600, retries=3):
    """GET json with a small on-disk cache (protects your CFBD monthly call quota)."""
    params = params or {}
    if os.environ.get("SPREADX_NO_CACHE"):
        ttl = 0
    key = hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    cp = CACHE / f"{key}.json"
    if ttl and cp.exists() and time.time() - cp.stat().st_mtime < ttl:
        return json.loads(cp.read_text())
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (i + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if ttl:
                cp.write_text(json.dumps(data))
            return data
        except requests.RequestException as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Request failed: {url} ({last})")


def cfbd_get(path, ttl=6 * 3600, **params):
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise SystemExit("CFBD_API_KEY not set. Add  CFBD_API_KEY=...  to a .env file next to these scripts.")
    params = {k: v for k, v in params.items() if v is not None}
    return http_json(f"{CFBD_URL}{path}", params,
                     {"Authorization": f"Bearer {key}", "accept": "application/json"}, ttl)


def pick(d, *keys, default=None):
    """First non-None value among several possible key spellings (camelCase / snake_case)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- games / lines
def normalize_game(g):
    hc = str(pick(g, "homeClassification", "home_classification", default="") or "").lower()
    ac = str(pick(g, "awayClassification", "away_classification", default="") or "").lower()
    return dict(
        id=pick(g, "id"),
        week=pick(g, "week"),
        season_type=pick(g, "seasonType", "season_type"),
        start=pick(g, "startDate", "start_date"),
        completed=bool(pick(g, "completed", default=False)),
        neutral=bool(pick(g, "neutralSite", "neutral_site", default=False)),
        home=pick(g, "homeTeam", "home_team"),
        away=pick(g, "awayTeam", "away_team"),
        home_pts=pick(g, "homePoints", "home_points"),
        away_pts=pick(g, "awayPoints", "away_points"),
        home_fbs=hc == "fbs",
        away_fbs=ac == "fbs",
        home_conf=pick(g, "homeConference", "home_conference"),
        away_conf=pick(g, "awayConference", "away_conference"),
        venue_id=pick(g, "venueId", "venue_id"),
        venue=pick(g, "venue"),
    )


def fetch_games(year, ttl=3 * 3600):
    """All games involving at least one FBS team (regular season + postseason)."""
    games = []
    for st in ("regular", "postseason"):
        try:
            raw = cfbd_get("/games", ttl=ttl, year=year, seasonType=st)
        except RuntimeError:
            if st == "regular":
                raise
            raw = []
        for g in raw:
            ng = normalize_game(g)
            if ng["home_fbs"] or ng["away_fbs"]:
                games.append(ng)
    return games


def fbs_team_conferences(games):
    teams = {}
    for g in games:
        if g["home_fbs"]:
            teams[g["home"]] = g["home_conf"]
        if g["away_fbs"]:
            teams[g["away"]] = g["away_conf"]
    return teams


_SPREAD_RE = re.compile(r"^(.*?)\s*([+-]?\d+(?:\.\d+)?|PK|pk)\s*$")


def home_spread_from_line(line, home, away):
    """Home-team spread (negative = home favored). Parses formattedSpread when possible
    so we don't depend on one sign convention; falls back to the raw spread field."""
    fs = pick(line, "formattedSpread", "formatted_spread")
    if fs:
        s = str(fs).strip()
        if s.upper() in ("PK", "EVEN", "PICK", "PICK'EM"):
            return 0.0
        m = _SPREAD_RE.match(s)
        if m:
            team, val = m.group(1).strip(), m.group(2)
            v = 0.0 if val.upper() == "PK" else float(val)
            if team == home:
                return v
            if team == away:
                return -v
    sp = line.get("spread")
    return float(sp) if sp is not None else None


def fetch_lines(year, ttl=1800, provider=None):
    """{game_id: {home_spread, total, n_books, home, away}} using the MEDIAN across sportsbooks."""
    out = {}
    for st in ("regular", "postseason"):
        try:
            raw = cfbd_get("/lines", ttl=ttl, year=year, seasonType=st)
        except RuntimeError:
            if st == "regular":
                raise
            raw = []
        for item in raw:
            home = pick(item, "homeTeam", "home_team")
            away = pick(item, "awayTeam", "away_team")
            spreads, totals = [], []
            for ln in item.get("lines") or []:
                if provider and str(ln.get("provider", "")).lower() != provider.lower():
                    continue
                s = home_spread_from_line(ln, home, away)
                t = pick(ln, "overUnder", "over_under")
                if s is not None:
                    spreads.append(float(s))
                if t is not None:
                    totals.append(float(t))
            if spreads or totals:
                out[pick(item, "id", "gameId")] = dict(
                    home_spread=median(spreads) if spreads else None,
                    total=median(totals) if totals else None,
                    n_books=max(len(spreads), len(totals)),
                    home=home, away=away)
    return out


# --------------------------------------------------------------------------- team-name matching
# Extra spellings other sites use. Add to this if the Kalshi matcher reports unmatched teams.
ALT_NAMES = {
    # keys are the names CFBD uses; values are other spellings seen on Kalshi / elsewhere
    "Miami": ["miami fl", "miami florida", "miami hurricanes"],
    "Miami (OH)": ["miami ohio", "miami oh", "miami redhawks"],
    "Ole Miss": ["mississippi"],
    "USC": ["southern california"],
    "UCF": ["central florida"],
    "Pittsburgh": ["pitt"],
    "Massachusetts": ["umass"],
    "UConn": ["connecticut"],
    "Louisiana": ["louisiana lafayette", "ul lafayette"],
    "UL Monroe": ["louisiana monroe", "louisiana-monroe", "ulm"],
    "Florida International": ["fiu"],
    "Florida Atlantic": ["fau"],
    "App State": ["appalachian state"],
    "San Jos\u00e9 State": ["sjsu"],
    "Southern Miss": ["southern mississippi"],
    "Sam Houston": ["sam houston state"],
    "UTSA": ["ut san antonio"],
    "North Carolina": ["unc"],
    "Texas A&M": ["texas a and m", "texas am"],
    "Western Kentucky": ["wku"],
    "Middle Tennessee": ["mtsu", "middle tennessee state"],
    "Northern Illinois": ["niu"],
    "San Diego State": ["sdsu"],
    "Bowling Green": ["bgsu"],
    "Hawai'i": ["hawaii"],
    "Connecticut": [],
}


def norm_team(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bst\b", "state", s)
    return re.sub(r"\s+", " ", s).strip()


def team_variants(team):
    v = {norm_team(team)}
    for a in ALT_NAMES.get(team, []):
        v.add(norm_team(a))
    return v


def build_matcher(teams):
    """List of (variant, team) sorted longest-first so 'michigan state' claims text before 'michigan'."""
    pairs = [(var, t) for t in teams for var in team_variants(t)]
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def find_teams(text, matcher):
    """Teams mentioned in text, in order of appearance."""
    t = " " + norm_team(text) + " "
    found = []
    for var, team in matcher:
        idx = t.find(f" {var} ")
        if idx >= 0:
            found.append((idx, team))
            t = t[:idx + 1] + " " * len(var) + t[idx + 1 + len(var):]
    found.sort()
    seen, out = set(), []
    for _, team in found:
        if team not in seen:
            seen.add(team)
            out.append(team)
    return out


def resolve_team(name, matcher):
    hits = find_teams(name, matcher)
    return hits[0] if hits else None


# --------------------------------------------------------------------------- misc
def parse_utc(s):
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def local_kickoff(s):
    d = parse_utc(s)
    return d.astimezone().strftime("%a %m/%d %I:%M %p").replace(" 0", " ") if d else "TBD"
