"""Fetch and cache data from FPL-Core-Insights GitHub repo and the FPL API."""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

if getattr(sys, "frozen", False):
    _BASE = Path(sys.executable).parent
else:
    _BASE = Path(__file__).resolve().parent.parent

CACHE_DIR = _BASE / "cache"
CACHE_MAX_AGE_SECONDS = 6 * 3600  # 6 hours (static GitHub data)
API_CACHE_MAX_AGE_SECONDS = 30 * 60  # 30 minutes (FPL API — injury/price changes)

GITHUB_BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"

# 2024-2025 structure: {season}/{category}/{category}.csv
SEASON_2425_FILES = {
    "players": "players/players.csv",
    "playermatchstats": "playermatchstats/playermatchstats.csv",
    "matches": "matches/matches.csv",
    "playerstats": "playerstats/playerstats.csv",
    "teams": "teams/teams.csv",
}

# 2025-2026 structure: root-level files + By Gameweek/GW{N}/ per-GW files
SEASON_2526_ROOT_FILES = {
    "players": "players.csv",
    "playerstats": "playerstats.csv",
    "teams": "teams.csv",
}
SEASON_2526_GW_FILES = [
    "playermatchstats.csv",
    "matches.csv",
]

FPL_API_BASE = "https://fantasy.premierleague.com/api"
FPL_API_ENDPOINTS = {
    "bootstrap": f"{FPL_API_BASE}/bootstrap-static/",
    "fixtures": f"{FPL_API_BASE}/fixtures/",
}


def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _is_cache_fresh(path: Path, max_age: int | None = None) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < (max_age if max_age is not None else CACHE_MAX_AGE_SECONDS)


def _fetch_url(url: str, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def _fetch_csv(url: str, cache_file: Path, force: bool = False) -> pd.DataFrame:
    if not force and _is_cache_fresh(cache_file):
        return pd.read_csv(cache_file, encoding="utf-8")
    print(f"  Fetching {url}")
    resp = _fetch_url(url)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text, encoding="utf-8")
    return pd.read_csv(cache_file, encoding="utf-8")


def fetch_fpl_api(endpoint: str, force: bool = False) -> dict:
    """Fetch JSON from the FPL API, caching locally (30min TTL)."""
    cache_file = _cache_path(f"fpl_api_{endpoint}.json")
    if not force and _is_cache_fresh(cache_file, max_age=API_CACHE_MAX_AGE_SECONDS):
        return json.loads(cache_file.read_text(encoding="utf-8"))
    url = FPL_API_ENDPOINTS[endpoint]
    print(f"  Fetching {url}")
    resp = _fetch_url(url)
    data = resp.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def fetch_manager_entry(manager_id: int) -> dict:
    """Fetch manager overview (name, bank, value, current_event)."""
    url = f"{FPL_API_BASE}/entry/{manager_id}/"
    resp = _fetch_url(url)
    return resp.json()


def fetch_manager_picks(manager_id: int, event: int) -> dict:
    """Fetch manager's 15 picks for a gameweek."""
    url = f"{FPL_API_BASE}/entry/{manager_id}/event/{event}/picks/"
    resp = _fetch_url(url)
    return resp.json()


def fetch_manager_history(manager_id: int) -> dict:
    """Fetch per-GW history (transfers, chips) for FT calculation."""
    url = f"{FPL_API_BASE}/entry/{manager_id}/history/"
    resp = _fetch_url(url)
    return resp.json()


def _detect_max_gw_2526(force: bool = False) -> int:
    """Detect the latest available gameweek for 2025-2026 by checking playerstats.

    Always fetches fresh to avoid stale cache capping the GW loop and
    missing newly published gameweek data.
    """
    cache_file = _cache_path("2025-2026_playerstats.csv")
    url = f"{GITHUB_BASE}/2025-2026/playerstats.csv"
    df = _fetch_csv(url, cache_file, force=True)
    return int(df["gw"].max())


def fetch_2425_data(force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch 2024-2025 season data."""
    critical_keys = {"players", "playerstats", "playermatchstats", "matches"}
    data = {}
    for key, path in SEASON_2425_FILES.items():
        url = f"{GITHUB_BASE}/2024-2025/{path}"
        cache_file = _cache_path(f"2024-2025_{key}.csv")
        try:
            data[key] = _fetch_csv(url, cache_file, force=force)
        except requests.HTTPError as e:
            if key in critical_keys:
                print(f"  ERROR: failed to fetch critical file 2024-2025/{key}: {e}")
                raise
            print(f"  Warning: could not fetch 2024-2025/{key}: {e}")
    return data


def fetch_2526_data(force: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch 2025-2026 season data (root files + per-GW files combined)."""
    data = {}

    # Root-level files
    critical_keys = {"players", "playerstats"}
    for key, path in SEASON_2526_ROOT_FILES.items():
        url = f"{GITHUB_BASE}/2025-2026/{path}"
        cache_file = _cache_path(f"2025-2026_{key}.csv")
        try:
            data[key] = _fetch_csv(url, cache_file, force=force)
        except requests.HTTPError as e:
            if key in critical_keys:
                print(f"  ERROR: failed to fetch critical file 2025-2026/{key}: {e}")
                raise
            print(f"  Warning: could not fetch 2025-2026/{key}: {e}")

    # Per-GW files — combine across all GWs
    max_gw = _detect_max_gw_2526(force=force)
    print(f"  2025-2026: detected {max_gw} gameweeks")

    for filename in SEASON_2526_GW_FILES:
        key = filename.replace(".csv", "")
        frames = []
        for gw in range(1, max_gw + 1):
            url = f"{GITHUB_BASE}/2025-2026/By Gameweek/GW{gw}/{filename}"
            cache_file = _cache_path(f"2025-2026_gw{gw}_{filename}")
            try:
                df = _fetch_csv(url, cache_file, force=force)
                if "gameweek" not in df.columns:
                    df["gameweek"] = gw
                frames.append(df)
            except requests.HTTPError:
                pass  # GW not available yet
        if frames:
            data[key] = pd.concat(frames, ignore_index=True)

    return data


def load_all_data(force: bool = False) -> dict:
    """Main entry point: fetch everything.

    Returns dict with keys:
        '2425' -> dict of DataFrames for 2024-2025
        '2526' -> dict of DataFrames for 2025-2026
        'api'  -> dict with 'bootstrap' and 'fixtures'
    """
    print("Fetching 2024-2025 data...")
    data_2425 = fetch_2425_data(force=force)
    print("Fetching 2025-2026 data...")
    data_2526 = fetch_2526_data(force=force)
    print("Fetching FPL API data...")
    api_data = {ep: fetch_fpl_api(ep, force=force) for ep in FPL_API_ENDPOINTS}
    print("Data loading complete.")
    return {"2425": data_2425, "2526": data_2526, "api": api_data}
