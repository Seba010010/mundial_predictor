from functools import lru_cache
from pathlib import Path
import unicodedata

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def normalize_name_key(name):
    if not isinstance(name, str):
        return name

    normalized = unicodedata.normalize("NFKD", name)
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )

    return " ".join(without_accents.lower().split())


@lru_cache(maxsize=1)
def load_preferred_team_names():
    preferred = {}

    # World Cup/team files define the spelling used by the simulator.
    for filename in ("groups.csv", "teams.csv", "team_elo_estimates.csv"):
        path = DATA_DIR / filename

        if not path.exists():
            continue

        teams = pd.read_csv(path)

        if "team" not in teams.columns:
            continue

        for team in teams["team"].dropna():
            key = normalize_name_key(team)
            preferred.setdefault(key, team)

    return preferred


@lru_cache(maxsize=1)
def load_former_name_rules():
    path = DATA_DIR / "former_names.csv"

    if not path.exists():
        return []

    former_names = pd.read_csv(path)
    former_names["start_date"] = pd.to_datetime(
        former_names["start_date"],
        errors="coerce"
    )
    former_names["end_date"] = pd.to_datetime(
        former_names["end_date"],
        errors="coerce"
    )

    rules = []
    preferred = load_preferred_team_names()

    for _, row in former_names.iterrows():
        current = row["current"]
        current_key = normalize_name_key(current)
        rules.append({
            "former_key": normalize_name_key(row["former"]),
            "current": preferred.get(current_key, current),
            "start_date": row["start_date"],
            "end_date": row["end_date"],
        })

    return rules


def canonical_team_name(team, match_date=None):
    if not isinstance(team, str):
        return team

    key = normalize_name_key(team)
    preferred = load_preferred_team_names()

    if key in preferred:
        return preferred[key]

    if match_date is not None:
        match_date = pd.Timestamp(match_date)

    for rule in load_former_name_rules():
        if key != rule["former_key"]:
            continue

        if match_date is None:
            return rule["current"]

        starts_before = (
            pd.isna(rule["start_date"]) or match_date >= rule["start_date"]
        )
        ends_after = (
            pd.isna(rule["end_date"]) or match_date <= rule["end_date"]
        )

        if starts_before and ends_after:
            return rule["current"]

    return team


def normalize_team_columns(df, columns, date_column=None):
    normalized = df.copy()

    for column in columns:
        if column not in normalized.columns:
            continue

        if date_column and date_column in normalized.columns:
            normalized[column] = [
                canonical_team_name(team, date)
                for team, date in zip(normalized[column], normalized[date_column])
            ]
        else:
            normalized[column] = normalized[column].map(canonical_team_name)

    return normalized


@lru_cache(maxsize=1)
def load_shootouts():
    path = DATA_DIR / "shootouts.csv"

    if not path.exists():
        return pd.DataFrame()

    shootouts = pd.read_csv(path)
    shootouts["date"] = pd.to_datetime(shootouts["date"])
    shootouts = normalize_team_columns(
        shootouts,
        ["home_team", "away_team", "winner", "first_shooter"],
        date_column="date"
    )

    return shootouts


@lru_cache(maxsize=1)
def load_goalscorers():
    path = DATA_DIR / "goalscorers.csv"

    if not path.exists():
        return pd.DataFrame()

    goalscorers = pd.read_csv(path)
    goalscorers["date"] = pd.to_datetime(goalscorers["date"])
    goalscorers = normalize_team_columns(
        goalscorers,
        ["home_team", "away_team", "team"],
        date_column="date"
    )

    return goalscorers
