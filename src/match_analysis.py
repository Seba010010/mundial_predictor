from functools import lru_cache
import math

import pandas as pd

from src.data_utils import load_goalscorers, normalize_name_key


@lru_cache(maxsize=1)
def load_worldcup_squads():
    try:
        squads = pd.read_csv("data/worldcup_squads.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    return squads.fillna("")


def squad_candidate_names(row):
    return [
        row.get("player_name", ""),
        f"{row.get('first_names', '')} {row.get('last_names', '')}",
        f"{row.get('first_names', '')} {row.get('shirt_name', '')}",
        row.get("shirt_name", ""),
        row.get("last_names", ""),
    ]


@lru_cache(maxsize=None)
def squad_player_keys(team_name):
    squads = load_worldcup_squads()

    if squads.empty:
        return None

    team_squad = squads[squads["team"] == team_name]

    if team_squad.empty:
        return None

    keys = []

    for _, row in team_squad.iterrows():
        for candidate in squad_candidate_names(row):
            key = normalize_name_key(candidate)

            if key:
                keys.append(key)

    return keys


def scorer_is_in_squad(team_name, scorer):
    keys = squad_player_keys(team_name)

    if keys is None:
        return True

    scorer_key = normalize_name_key(scorer)
    scorer_words = set(scorer_key.split())

    for key in keys:
        if scorer_key == key:
            return True

        key_words = set(key.split())

        if scorer_words and scorer_words.issubset(key_words):
            return True

    return False


def goal_distribution(score_probabilities, side):
    goal_key = f"{side}_goals"
    rows = []

    for goals in range(4):
        probability = sum(
            item["probability"]
            for item in score_probabilities
            if item[goal_key] == goals
        )
        rows.append({
            "goals": str(goals),
            "probability": round(probability * 100, 2),
        })

    probability_4_plus = sum(
        item["probability"]
        for item in score_probabilities
        if item[goal_key] >= 4
    )
    rows.append({
        "goals": "4+",
        "probability": round(probability_4_plus * 100, 2),
    })

    return rows


def total_goals_probability(score_probabilities, threshold, over=True):
    if over:
        probability = sum(
            item["probability"]
            for item in score_probabilities
            if item["home_goals"] + item["away_goals"] > threshold
        )
    else:
        probability = sum(
            item["probability"]
            for item in score_probabilities
            if item["home_goals"] + item["away_goals"] < threshold
        )

    return round(probability * 100, 2)


def clean_sheet_probability(score_probabilities, side):
    opponent_key = "away_goals" if side == "home" else "home_goals"
    probability = sum(
        item["probability"]
        for item in score_probabilities
        if item[opponent_key] == 0
    )

    return round(probability * 100, 2)


def team_scores_probability(score_probabilities, side):
    goal_key = f"{side}_goals"
    probability = sum(
        item["probability"]
        for item in score_probabilities
        if item[goal_key] > 0
    )

    return round(probability * 100, 2)


@lru_cache(maxsize=None)
def get_recent_goal_scorer_rates(team_name, cutoff_date="2018-01-01", top_n=12):
    goalscorers = load_goalscorers()

    if goalscorers.empty:
        return []

    goalscorers = goalscorers[
        (goalscorers["date"] >= pd.Timestamp(cutoff_date)) &
        (goalscorers["team"] == team_name) &
        (~goalscorers["own_goal"].astype(str).str.upper().eq("TRUE"))
    ].copy()

    if goalscorers.empty:
        return []

    goalscorers["penalty"] = goalscorers["penalty"].astype(str).str.upper().eq("TRUE")
    scorer_totals = goalscorers.groupby("scorer", as_index=False).agg(
        goals=("scorer", "size"),
        penalty_goals=("penalty", "sum"),
    )
    total_goals = scorer_totals["goals"].sum()
    scorer_totals["team_goal_share"] = scorer_totals["goals"] / total_goals
    scorer_totals["in_squad"] = scorer_totals["scorer"].map(
        lambda scorer: scorer_is_in_squad(team_name, scorer)
    )
    scorer_totals = scorer_totals[scorer_totals["in_squad"]]
    scorer_totals = scorer_totals.sort_values(
        ["goals", "team_goal_share"],
        ascending=[False, False]
    )

    return scorer_totals.head(top_n).to_dict("records")


def player_goal_probabilities(team_name, team_lambda, top_n=8):
    rows = []

    for scorer in get_recent_goal_scorer_rates(team_name, top_n=top_n):
        player_lambda = team_lambda * scorer["team_goal_share"]
        probability = 1 - math.exp(-player_lambda)
        rows.append({
            "player": scorer["scorer"],
            "historical_goals": int(scorer["goals"]),
            "penalty_goals": int(scorer["penalty_goals"]),
            "team_goal_share_pct": round(scorer["team_goal_share"] * 100, 2),
            "anytime_goal_pct": round(probability * 100, 2),
            "in_squad": bool(scorer["in_squad"]),
        })

    return rows


def build_advanced_match_analysis(result):
    score_probabilities = result["score_probabilities"]

    home_lambda = result["home_lambda"]
    away_lambda = result["away_lambda"]

    return {
        "home_goal_distribution": goal_distribution(score_probabilities, "home"),
        "away_goal_distribution": goal_distribution(score_probabilities, "away"),
        "markets": [
            {
                "market": "Over 1.5 goles",
                "probability": total_goals_probability(score_probabilities, 1.5),
            },
            {
                "market": "Under 1.5 goles",
                "probability": total_goals_probability(
                    score_probabilities,
                    1.5,
                    over=False
                ),
            },
            {
                "market": "Over 2.5 goles",
                "probability": total_goals_probability(score_probabilities, 2.5),
            },
            {
                "market": "Under 2.5 goles",
                "probability": total_goals_probability(
                    score_probabilities,
                    2.5,
                    over=False
                ),
            },
            {
                "market": "Over 3.5 goles",
                "probability": total_goals_probability(score_probabilities, 3.5),
            },
            {
                "market": "Ambos anotan",
                "probability": result["both_score"],
            },
            {
                "market": f"Marca {result['home_team']}",
                "probability": team_scores_probability(score_probabilities, "home"),
            },
            {
                "market": f"Marca {result['away_team']}",
                "probability": team_scores_probability(score_probabilities, "away"),
            },
            {
                "market": f"Arco en cero {result['home_team']}",
                "probability": clean_sheet_probability(score_probabilities, "home"),
            },
            {
                "market": f"Arco en cero {result['away_team']}",
                "probability": clean_sheet_probability(score_probabilities, "away"),
            },
        ],
        "home_scorers": player_goal_probabilities(
            result["home_team"],
            home_lambda,
        ),
        "away_scorers": player_goal_probabilities(
            result["away_team"],
            away_lambda,
        ),
    }
