import pandas as pd
import math
from functools import lru_cache
from src.poisson import score_matrix
from src.simulator import simulate_match
from src.head_to_head import get_head_to_head
from src.team_stats import calculate_team_stats, get_recent_form, get_global_goal_average

@lru_cache(maxsize=1)
def load_teams():
    return pd.read_csv("data/teams.csv")


@lru_cache(maxsize=1)
def load_market_values():
    try:
        market_values = pd.read_csv("data/team_market_values.csv")
    except FileNotFoundError:
        return {}, 1

    values = dict(
        zip(
            market_values["team"],
            market_values["market_value_millions_eur"]
        )
    )
    average_value = market_values["market_value_millions_eur"].mean()

    return values, average_value


def get_market_factor(team_name):
    market_values, average_value = load_market_values()

    if not average_value or team_name not in market_values:
        return 1.0

    market_factor = market_values[team_name] / average_value

    return max(0.90, min(market_factor, 1.10))


@lru_cache(maxsize=1)
def load_fifa_rankings():
    try:
        fifa_rankings = pd.read_csv("data/team_fifa_rankings.csv")
    except FileNotFoundError:
        return {}, 1, 1

    rankings = dict(zip(fifa_rankings["team"], fifa_rankings["fifa_rank"]))
    best_rank = fifa_rankings["fifa_rank"].min()
    worst_rank = fifa_rankings["fifa_rank"].max()

    return rankings, best_rank, worst_rank


def get_fifa_factor(team_name):
    rankings, best_rank, worst_rank = load_fifa_rankings()

    if team_name not in rankings or best_rank == worst_rank:
        return 1.0

    fifa_rank = rankings[team_name]
    rank_position = (fifa_rank - best_rank) / (worst_rank - best_rank)
    fifa_factor = 1.05 - (rank_position * 0.10)

    return max(0.95, min(fifa_factor, 1.05))

# def elo_adjustment(home_elo, away_elo):

#     diff = home_elo - away_elo

#     home_factor = 1 + (diff / 2500)
#     away_factor = 1 - (diff / 2500)

#     home_factor = max(0.80, min(home_factor, 1.20))
#     away_factor = max(0.80, min(away_factor, 1.20))

#     return home_factor, away_factor

def elo_adjustment(home_elo, away_elo):

    diff = home_elo - away_elo

    home_factor = math.exp(diff / 1600)
    away_factor = math.exp(-diff / 1600)

    home_factor = max(0.80, min(home_factor, 1.20))
    away_factor = max(0.80, min(away_factor, 1.20))

    return home_factor, away_factor


def predict_match(home_team, away_team, include_details=True, neutral=True):    
    teams = load_teams()

    home_row = teams[teams["team"] == home_team]
    away_row = teams[teams["team"] == away_team]

    if home_row.empty:
        raise ValueError(f"El equipo local no existe en teams.csv: {home_team}")

    if away_row.empty:
        raise ValueError(f"El equipo visitante no existe en teams.csv: {away_team}")

    home = home_row.iloc[0]
    away = away_row.iloc[0]


    home_stats = calculate_team_stats(home_team)
    away_stats = calculate_team_stats(away_team)
    home_form = None
    away_form = None
    head_to_head = None

    if include_details:
        home_form = get_recent_form(home_team)
        away_form = get_recent_form(away_team)
        head_to_head = get_head_to_head(home_team, away_team)

    home_factor, away_factor = elo_adjustment(home["elo"], away["elo"])
    home_market_factor = get_market_factor(home_team)
    away_market_factor = get_market_factor(away_team)
    home_fifa_factor = get_fifa_factor(home_team)
    away_fifa_factor = get_fifa_factor(away_team)

    home_advantage = 1.0

    if not neutral:
        home_advantage = 1.1

    global_avg = get_global_goal_average()

    home_lambda = (
        home_stats["attack"]
        * away_stats["defense"]
        * global_avg
        * home_factor
        * home_market_factor
        * home_fifa_factor
        * home_advantage
    )

    away_lambda = (
        away_stats["attack"]
        * home_stats["defense"]
        * global_avg
        * away_factor
        * away_market_factor
        * away_fifa_factor
    )

    matrix = score_matrix(home_lambda, away_lambda)

    simulation = None

    if include_details:
        simulation = simulate_match(matrix, simulations=10000)

    home_win = sum(prob for (h, a), prob in matrix.items() if h > a)
    draw = sum(prob for (h, a), prob in matrix.items() if h == a)
    away_win = sum(prob for (h, a), prob in matrix.items() if h < a)

    over_15 = sum(prob for (h, a), prob in matrix.items() if h + a > 1.5)
    over_25 = sum(prob for (h, a), prob in matrix.items() if h + a > 2.5)
    both_score = sum(prob for (h, a), prob in matrix.items() if h > 0 and a > 0)

    most_likely_scores = sorted(
        matrix.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "home_team": home_team,
        "away_team": away_team,

        "home_lambda": round(home_lambda, 2),
        "away_lambda": round(away_lambda, 2),

        "home_elo": int(home["elo"]),
        "away_elo": int(away["elo"]),

        "home_market_factor": round(home_market_factor, 3),
        "away_market_factor": round(away_market_factor, 3),
        "home_fifa_factor": round(home_fifa_factor, 3),
        "away_fifa_factor": round(away_fifa_factor, 3),

        "home_attack": home_stats["attack"],
        "home_defense": home_stats["defense"],
        "away_attack": away_stats["attack"],
        "away_defense": away_stats["defense"],

        "home_matches": home_stats["matches_played"],
        "away_matches": away_stats["matches_played"],

        "home_win": round(home_win * 100, 2),
        "draw": round(draw * 100, 2),
        "away_win": round(away_win * 100, 2),
        "over_15": round(over_15 * 100, 2),
        "over_25": round(over_25 * 100, 2),
        "both_score": round(both_score * 100, 2),
        "simulation": simulation,

        "home_stats": home_stats,
        "away_stats": away_stats,

        "home_form": home_form,
        "away_form": away_form,

        "head_to_head": head_to_head,

        "most_likely_scores": [
            {
                "score": f"{h}-{a}",
                "probability": round(float(prob) * 100, 2)
            }
            for (h, a), prob in most_likely_scores
        ],

        "score_probabilities": [
            {
                "home_goals": h,
                "away_goals": a,
                "probability": float(prob)
            }
            for (h, a), prob in matrix.items()
        ],
    }
