import random


def simulate_match(matrix, simulations=10000):
    scores = list(matrix.keys())
    probabilities = list(matrix.values())

    simulated_results = random.choices(
        scores,
        weights=probabilities,
        k=simulations
    )

    home_wins = 0
    draws = 0
    away_wins = 0
    score_count = {}

    for home_goals, away_goals in simulated_results:
        if home_goals > away_goals:
            home_wins += 1
        elif home_goals == away_goals:
            draws += 1
        else:
            away_wins += 1

        score = f"{home_goals}-{away_goals}"
        score_count[score] = score_count.get(score, 0) + 1

    most_common_scores = sorted(
        score_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "simulations": simulations,
        "home_wins": home_wins,
        "draws": draws,
        "away_wins": away_wins,
        "home_win_pct": round((home_wins / simulations) * 100, 2),
        "draw_pct": round((draws / simulations) * 100, 2),
        "away_win_pct": round((away_wins / simulations) * 100, 2),
        "most_common_scores": [
            {
                "score": score,
                "times": times,
                "percentage": round((times / simulations) * 100, 2)
            }
            for score, times in most_common_scores
        ]
    }