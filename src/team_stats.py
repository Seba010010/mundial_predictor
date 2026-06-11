import pandas as pd
from datetime import datetime
from functools import lru_cache

from src.data_utils import canonical_team_name, normalize_team_columns


TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 2.0,
    "FIFA World Cup qualification": 1.6,
    "UEFA Euro": 1.8,
    "Copa América": 1.8,
    "Copa America": 1.8,
    "AFC Asian Cup": 1.5,
    "African Cup of Nations": 1.5,
    "CONCACAF Gold Cup": 1.4,
    "UEFA Nations League": 1.3,
    "Friendly": 0.6,
}


def load_matches():
    matches = pd.read_csv("data/matches_history.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    matches = normalize_team_columns(matches, ["home", "away"], date_column="date")
    return matches.sort_values("date")


def get_tournament_weight(tournament):
    return TOURNAMENT_WEIGHTS.get(tournament, 1.0)


def get_recency_weight(match_date):
    today = pd.Timestamp(datetime.today().date())
    days_diff = (today - match_date).days
    years_diff = days_diff / 365

    # Mientras más reciente, mayor peso.
    # Hoy ≈ 1.0
    # 2 años ≈ 0.5
    # 4 años ≈ 0.33
    return 1 / (1 + years_diff)

@lru_cache(maxsize=None)
def load_team_elos():
    teams = pd.read_csv("data/teams.csv")
    return dict(zip(teams["team"], teams["elo"]))


@lru_cache(maxsize=None)
def load_estimated_team_elos():
    try:
        estimates = pd.read_csv("data/team_elo_estimates.csv")
    except FileNotFoundError:
        return {}

    return dict(zip(estimates["team"], estimates["estimated_elo"]))


@lru_cache(maxsize=None)
def get_team_elo(team_name):
    team_name = canonical_team_name(team_name)
    team_elos = load_team_elos()

    if team_name in team_elos:
        return int(team_elos[team_name])

    estimated_team_elos = load_estimated_team_elos()

    if team_name in estimated_team_elos:
        return int(estimated_team_elos[team_name])

    return 1400

@lru_cache(maxsize=1)
def get_global_goal_average():
    matches = load_matches()

    total_goals = (
        matches["home_goals"].sum() +
        matches["away_goals"].sum()
    )

    total_matches = len(matches)

    return total_goals / (total_matches * 2)

@lru_cache(maxsize=None)
def calculate_team_stats(team_name, last_n=100):
    team_name = canonical_team_name(team_name)
    matches = load_matches()

    team_matches = matches[
        (matches["home"] == team_name) |
        (matches["away"] == team_name)
    ].tail(last_n)

    if team_matches.empty:
        return {
            "attack": 1.0,
            "defense": 1.0,
            "matches_played": 0,
            "goals_for": 0,
            "goals_against": 0,
            "weighted_goals_for": 0,
            "weighted_goals_against": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }

    weighted_goals_for = 0
    weighted_goals_against = 0
    total_weight = 0

    raw_goals_for = 0
    raw_goals_against = 0

    wins = 0
    draws = 0
    losses = 0

    for _, match in team_matches.iterrows():
        if match["home"] == team_name:
            goals_for = match["home_goals"]
            goals_against = match["away_goals"]
        else:
            goals_for = match["away_goals"]
            goals_against = match["home_goals"]

        if goals_for > goals_against:
            wins += 1
        elif goals_for == goals_against:
            draws += 1
        else:
            losses += 1

        tournament_weight = get_tournament_weight(match["tournament"])
        recency_weight = get_recency_weight(match["date"])
        final_weight = tournament_weight * recency_weight

        if match["home"] == team_name:
            opponent = match["away"]
        else:
            opponent = match["home"]

        opponent_elo = get_team_elo(opponent)

        opponent_strength = opponent_elo / 1700
        opponent_strength = max(0.80, min(opponent_strength, 1.20))

        weighted_goals_for += goals_for * final_weight * opponent_strength
        weighted_goals_against += goals_against * final_weight / opponent_strength
        total_weight += final_weight

        raw_goals_for += goals_for
        raw_goals_against += goals_against

    raw_attack = weighted_goals_for / total_weight
    raw_defense = weighted_goals_against / total_weight

    global_avg = get_global_goal_average()

    attack = (raw_attack / global_avg) ** 0.75
    defense = (raw_defense / global_avg) ** 0.75

    return {
        "attack": round(attack, 2),
        "defense": round(defense, 2),
        "matches_played": len(team_matches),
        "goals_for": int(raw_goals_for),
        "goals_against": int(raw_goals_against),
        "weighted_goals_for": round(weighted_goals_for, 2),
        "weighted_goals_against": round(weighted_goals_against, 2),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "raw_attack": round(raw_attack, 2),
        "raw_defense": round(raw_defense, 2),
    }


@lru_cache(maxsize=None)
def get_recent_form(team_name, last_n=10):
    team_name = canonical_team_name(team_name)
    matches = load_matches()

    team_matches = matches[
        (matches["home"] == team_name) |
        (matches["away"] == team_name)
    ].tail(last_n)

    form = []

    for _, match in team_matches.iterrows():
        if match["home"] == team_name:
            goals_for = match["home_goals"]
            goals_against = match["away_goals"]
        else:
            goals_for = match["away_goals"]
            goals_against = match["home_goals"]

        if goals_for > goals_against:
            form.append("🟢")
        elif goals_for == goals_against:
            form.append("🟡")
        else:
            form.append("🔴")

    return form
