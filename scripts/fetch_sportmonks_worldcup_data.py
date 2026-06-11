import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import canonical_team_name


BASE_URL = "https://api.sportmonks.com/v3/football"
WORLD_CUP_2026_LEAGUE_ID = 732
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "sportmonks" / "raw"
MATCH_STATS_PATH = DATA_DIR / "match_advanced_stats.csv"
LINEUPS_PATH = DATA_DIR / "match_lineups.csv"
EVENTS_PATH = DATA_DIR / "match_events.csv"

STAT_ALIASES = {
    "shots_total": [
        "shots total",
        "total shots",
        "shots",
    ],
    "shots_on_target": [
        "shots on target",
        "shots ongoal",
        "shots on goal",
    ],
    "possession": [
        "ball possession %",
        "ball possession",
        "possession",
    ],
    "corners": [
        "corners",
        "corner kicks",
    ],
    "passes": [
        "passes",
        "total passes",
    ],
    "xg": [
        "expected goals",
        "xg",
    ],
}


def clean_label(value):
    if value is None:
        return ""

    return str(value).strip().lower().replace("_", " ")


def numeric_value(value):
    if isinstance(value, dict):
        for key in ("value", "total", "count", "percentage"):
            if key in value:
                return numeric_value(value[key])
        return None

    if value is None:
        return None

    text = str(value).replace("%", "").strip()

    if not text:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    return int(number) if number.is_integer() else number


def request_sportmonks(path, token, params):
    request_params = {"api_token": token, **params}
    url = f"{BASE_URL}{path}"
    print(f"Requesting {url}")
    response = requests.get(
        url,
        params=request_params,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def next_page_url(payload):
    pagination = payload.get("pagination") or {}

    if pagination.get("has_more"):
        next_page = pagination.get("current_page", 1) + 1
        return next_page

    links = payload.get("links") or {}

    if links.get("next"):
        return links["next"]

    meta = payload.get("meta") or {}
    if meta.get("pagination", {}).get("links", {}).get("next"):
        return meta["pagination"]["links"]["next"]

    return None


def merge_payloads(payloads):
    if not payloads:
        return {"data": []}

    merged = dict(payloads[0])
    data = []

    for payload in payloads:
        current = payload.get("data", [])

        if isinstance(current, dict):
            data.append(current)
        else:
            data.extend(current)

    merged["data"] = data
    return merged


def get_fixture_payload(
    token,
    mode,
    start_date=None,
    end_date=None,
    league_id=WORLD_CUP_2026_LEAGUE_ID,
    use_league_filter=True,
    max_pages=1,
):
    include = (
        "participants;"
        "scores;"
        "statistics.type;"
        "events.type;"
        "lineups.details.type;"
        "lineups.player;"
        "lineups.type"
    )
    params = {
        "include": include,
        "per_page": 50,
    }

    if use_league_filter and league_id:
        params["filters"] = f"fixtureLeagues:{league_id}"

    if mode == "live":
        return request_sportmonks("/livescores/inplay", token, params)

    if mode == "latest":
        return request_sportmonks("/livescores/latest", token, params)

    if mode == "between":
        if not start_date or not end_date:
            raise ValueError("--start-date and --end-date are required for between")

        path = f"/fixtures/between/{start_date}/{end_date}"
    else:
        path = "/fixtures"

    payloads = []
    page = 1

    while page and len(payloads) < max_pages:
        page_params = dict(params)
        page_params["page"] = page
        payload = request_sportmonks(path, token, page_params)
        payloads.append(payload)
        next_page = next_page_url(payload)

        if isinstance(next_page, int):
            page = next_page
        else:
            page = None

    return merge_payloads(payloads)


def unwrap_data(payload):
    data = payload.get("data", [])

    if isinstance(data, dict):
        return [data]

    return data


def get_participants(fixture):
    participants = fixture.get("participants") or []
    home = None
    away = None

    for participant in participants:
        meta = participant.get("meta") or {}
        location = clean_label(meta.get("location"))
        name = participant.get("name")
        participant_id = participant.get("id")

        if location == "home":
            home = {
                "id": participant_id,
                "team": canonical_team_name(name),
                "raw_team": name,
            }
        elif location == "away":
            away = {
                "id": participant_id,
                "team": canonical_team_name(name),
                "raw_team": name,
            }

    if (home is None or away is None) and len(participants) >= 2:
        home = home or {
            "id": participants[0].get("id"),
            "team": canonical_team_name(participants[0].get("name")),
            "raw_team": participants[0].get("name"),
        }
        away = away or {
            "id": participants[1].get("id"),
            "team": canonical_team_name(participants[1].get("name")),
            "raw_team": participants[1].get("name"),
        }

    return home, away


def stat_name(stat):
    stat_type = stat.get("type") or {}

    for key in ("name", "code", "developer_name"):
        if key in stat_type:
            return clean_label(stat_type[key])

    return clean_label(stat.get("type_name") or stat.get("name"))


def stat_participant_id(stat):
    return (
        stat.get("participant_id")
        or stat.get("team_id")
        or (stat.get("participant") or {}).get("id")
    )


def stat_value(stat):
    for key in ("value", "data"):
        if key in stat:
            value = numeric_value(stat[key])

            if value is not None:
                return value

    return None


def classify_stat(name):
    for stat_key, aliases in STAT_ALIASES.items():
        if name in aliases:
            return stat_key

    return None


def extract_match_stats(fixtures):
    rows = []

    for fixture in fixtures:
        home, away = get_participants(fixture)

        if home is None or away is None:
            continue

        row = {
            "fixture_id": fixture.get("id"),
            "date": fixture.get("starting_at"),
            "home_team": home["team"],
            "away_team": away["team"],
            "home_raw_team": home["raw_team"],
            "away_raw_team": away["raw_team"],
            "source": "sportmonks",
        }

        for stat_key in STAT_ALIASES:
            row[f"home_{stat_key}"] = None
            row[f"away_{stat_key}"] = None

        for stat in fixture.get("statistics") or []:
            name = stat_name(stat)
            stat_key = classify_stat(name)

            if stat_key is None:
                continue

            participant_id = stat_participant_id(stat)
            value = stat_value(stat)

            if participant_id == home["id"]:
                row[f"home_{stat_key}"] = value
            elif participant_id == away["id"]:
                row[f"away_{stat_key}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def lineup_player_name(lineup):
    player = lineup.get("player") or {}
    return player.get("display_name") or player.get("name") or lineup.get("player_name")


def lineup_type(lineup):
    lineup_type_data = lineup.get("type") or {}
    label = (
        lineup_type_data.get("name")
        or lineup_type_data.get("code")
        or lineup_type_data.get("developer_name")
        or lineup.get("type_name")
    )
    return str(label) if label is not None else None


def extract_lineups(fixtures):
    rows = []

    for fixture in fixtures:
        home, away = get_participants(fixture)
        team_by_id = {}

        if home:
            team_by_id[home["id"]] = home["team"]

        if away:
            team_by_id[away["id"]] = away["team"]

        for lineup in fixture.get("lineups") or []:
            participant_id = (
                lineup.get("participant_id")
                or lineup.get("team_id")
                or (lineup.get("participant") or {}).get("id")
            )
            rows.append({
                "fixture_id": fixture.get("id"),
                "date": fixture.get("starting_at"),
                "team": team_by_id.get(participant_id),
                "player_id": lineup.get("player_id") or (lineup.get("player") or {}).get("id"),
                "player": lineup_player_name(lineup),
                "position_id": lineup.get("position_id"),
                "formation_position": lineup.get("formation_position"),
                "shirt_number": lineup.get("jersey_number") or lineup.get("number"),
                "lineup_type": lineup_type(lineup),
                "starter": clean_label(lineup_type(lineup)) in ("starting xi", "starter", "lineup"),
            })

    return pd.DataFrame(rows)


def event_type(event):
    event_type_data = event.get("type") or {}
    label = (
        event_type_data.get("name")
        or event_type_data.get("code")
        or event_type_data.get("developer_name")
        or event.get("type_name")
    )
    return str(label) if label is not None else None


def event_player_name(event):
    player = event.get("player") or {}
    return player.get("display_name") or player.get("name") or event.get("player_name")


def extract_events(fixtures):
    rows = []

    for fixture in fixtures:
        home, away = get_participants(fixture)
        team_by_id = {}

        if home:
            team_by_id[home["id"]] = home["team"]

        if away:
            team_by_id[away["id"]] = away["team"]

        for event in fixture.get("events") or []:
            participant_id = (
                event.get("participant_id")
                or event.get("team_id")
                or (event.get("participant") or {}).get("id")
            )
            rows.append({
                "fixture_id": fixture.get("id"),
                "date": fixture.get("starting_at"),
                "minute": event.get("minute"),
                "extra_minute": event.get("extra_minute"),
                "team": team_by_id.get(participant_id),
                "player_id": event.get("player_id") or (event.get("player") or {}).get("id"),
                "player": event_player_name(event),
                "event_type": event_type(event),
                "result": event.get("result"),
                "info": event.get("info"),
            })

    return pd.DataFrame(rows)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path.relative_to(ROOT)} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fixtures", "between", "live", "latest"],
        default="fixtures",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--league-id",
        type=int,
        default=WORLD_CUP_2026_LEAGUE_ID,
        help="Sportmonks league id. Defaults to World Cup 2026 league 732.",
    )
    parser.add_argument(
        "--no-league-filter",
        action="store_true",
        help="Do not filter by league. Useful for historical exploration.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum pages to fetch for paginated fixture endpoints.",
    )
    parser.add_argument(
        "--token-env",
        default="SPORTMONKS_API_TOKEN",
        help="Environment variable containing the Sportmonks API token.",
    )
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Write empty CSV files when the API returns no fixtures.",
    )
    args = parser.parse_args()

    token = os.getenv(args.token_env)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Token env: {args.token_env}")
    print(f"Token configured: {'yes' if token else 'no'}")
    print(
        "League filter: "
        f"{'off' if args.no_league_filter else args.league_id}"
    )
    print(f"Max pages: {args.max_pages}")

    if not token:
        raise RuntimeError(
            f"Set {args.token_env} with your Sportmonks API token before running."
        )

    try:
        payload = get_fixture_payload(
            token,
            args.mode,
            start_date=args.start_date,
            end_date=args.end_date,
            league_id=args.league_id,
            use_league_filter=not args.no_league_filter,
            max_pages=args.max_pages,
        )
    except requests.HTTPError as error:
        output_key = f"league_{args.league_id}" if not args.no_league_filter else "all_leagues"
        error_path = RAW_DIR / f"sportmonks_{output_key}_{args.mode}_error.txt"
        response = error.response
        body = response.text if response is not None else str(error)
        error_path.write_text(body, encoding="utf-8")
        print(f"Sportmonks request failed. Wrote {error_path.relative_to(ROOT)}")
        raise

    fixtures = unwrap_data(payload)

    output_key = f"league_{args.league_id}" if not args.no_league_filter else "all_leagues"
    raw_path = RAW_DIR / f"sportmonks_{output_key}_{args.mode}.json"
    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Fixtures received: {len(fixtures)}")

    if not fixtures and not args.write_empty:
        print("No fixtures received. CSV files were not overwritten.")
        print("Use --write-empty if you intentionally want empty CSV outputs.")
        return

    save_csv(extract_match_stats(fixtures), MATCH_STATS_PATH)
    save_csv(extract_lineups(fixtures), LINEUPS_PATH)
    save_csv(extract_events(fixtures), EVENTS_PATH)


if __name__ == "__main__":
    main()
