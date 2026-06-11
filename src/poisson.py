# src/poisson.py

import math

def poisson_probability(lmbda, goals):
    return (lmbda ** goals) * math.exp(-lmbda) / math.factorial(goals)


def score_matrix(home_lambda, away_lambda, max_goals=10):
    results = {}

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob_home = poisson_probability(home_lambda, home_goals)
            prob_away = poisson_probability(away_lambda, away_goals)
            results[(home_goals, away_goals)] = prob_home * prob_away

    total_probability = sum(results.values())

    if total_probability:
        results = {
            score: probability / total_probability
            for score, probability in results.items()
        }

    return results
