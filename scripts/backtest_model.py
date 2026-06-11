import argparse
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = ROOT / "data" / "matches_history.csv"
TEAMS_PATH = ROOT / "data" / "teams.csv"

sys.path.insert(0, str(ROOT))

from src.poisson import score_matrix
from src.predictor import (
    elo_adjustment,
    get_fifa_factor,
    get_market_factor,
)
from src.data_utils import load_shootouts, normalize_team_columns
from src.team_stats import (
    calculate_team_stats,
    get_global_goal_average,
    get_team_elo,
)

BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
TOURNAMENTS_TO_REPORT = [
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "Friendly",
    "Copa América",
    "Copa America",
    "UEFA Euro",
    "African Cup of Nations",
    "AFC Asian Cup",
]


def actual_outcome(row):
    if row["home_goals"] > row["away_goals"]:
        return "home"
    if row["home_goals"] < row["away_goals"]:
        return "away"
    return "draw"


def shootout_outcome(row):
    if not isinstance(row.get("shootout_winner"), str):
        return None

    if row["shootout_winner"] == row["home"]:
        return "home"

    if row["shootout_winner"] == row["away"]:
        return "away"

    return None


def predicted_probabilities(prediction):
    return {
        "home": prediction["home_win"] / 100,
        "draw": prediction["draw"] / 100,
        "away": prediction["away_win"] / 100,
    }


def predict_backtest_match(home_team, away_team):
    home_stats = calculate_team_stats(home_team)
    away_stats = calculate_team_stats(away_team)
    home_elo = get_team_elo(home_team)
    away_elo = get_team_elo(away_team)
    home_factor, away_factor = elo_adjustment(home_elo, away_elo)
    global_avg = get_global_goal_average()

    home_lambda = (
        home_stats["attack"]
        * away_stats["defense"]
        * global_avg
        * home_factor
        * get_market_factor(home_team)
        * get_fifa_factor(home_team)
    )
    away_lambda = (
        away_stats["attack"]
        * home_stats["defense"]
        * global_avg
        * away_factor
        * get_market_factor(away_team)
        * get_fifa_factor(away_team)
    )

    matrix = score_matrix(home_lambda, away_lambda)

    home_win = sum(prob for (h, a), prob in matrix.items() if h > a)
    draw = sum(prob for (h, a), prob in matrix.items() if h == a)
    away_win = sum(prob for (h, a), prob in matrix.items() if h < a)

    return {
        "home_win": home_win * 100,
        "draw": draw * 100,
        "away_win": away_win * 100,
    }


def probability_bin(probability):
    percentage = probability * 100

    if percentage < 20:
        return "0-20%"
    if percentage < 40:
        return "20-40%"
    if percentage < 60:
        return "40-60%"
    if percentage < 80:
        return "60-80%"
    return "80-100%"


def brier_score(probabilities, outcome):
    score = 0

    for label in ("home", "draw", "away"):
        expected = 1 if label == outcome else 0
        score += (probabilities[label] - expected) ** 2

    return score


def summarize(rows):
    if not rows:
        return None

    total = len(rows)
    correct = sum(row["predicted_outcome"] == row["actual_outcome"] for row in rows)
    favorite_rows = [row for row in rows if row["favorite_probability"] is not None]
    favorite_correct = sum(
        row["favorite_outcome"] == row["actual_outcome"]
        for row in favorite_rows
    )
    log_loss = sum(row["log_loss"] for row in rows) / total
    brier = sum(row["brier_score"] for row in rows) / total
    shootout_rows = [
        row for row in rows
        if row["actual_outcome"] == "draw" and row["shootout_outcome"] is not None
    ]
    shootout_correct = sum(
        row["non_draw_favorite_outcome"] == row["shootout_outcome"]
        for row in shootout_rows
    )

    return {
        "matches": total,
        "accuracy_1x2": round(correct / total * 100, 2),
        "log_loss": round(log_loss, 4),
        "brier_score": round(brier, 4),
        "favorite_accuracy": round(favorite_correct / len(favorite_rows) * 100, 2)
        if favorite_rows
        else None,
        "shootout_matches": len(shootout_rows),
        "shootout_winner_accuracy": round(
            shootout_correct / len(shootout_rows) * 100,
            2
        )
        if shootout_rows
        else None,
    }


def build_bin_summary(rows):
    bin_rows = []

    for label in BIN_LABELS:
        current = [row for row in rows if row["favorite_bin"] == label]

        if current:
            correct = sum(
                row["favorite_outcome"] == row["actual_outcome"]
                for row in current
            )
            accuracy = round(correct / len(current) * 100, 2)
        else:
            accuracy = None

        bin_rows.append({
            "bin": label,
            "matches": len(current),
            "favorite_accuracy": accuracy,
        })

    return pd.DataFrame(bin_rows)


def run_backtest(cutoff_date):
    matches = pd.read_csv(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = normalize_team_columns(matches, ["home", "away"], date_column="date")
    matches = matches[matches["date"] >= pd.Timestamp(cutoff_date)].copy()

    shootouts = load_shootouts()

    if not shootouts.empty:
        shootouts = shootouts.rename(
            columns={
                "home_team": "home",
                "away_team": "away",
                "winner": "shootout_winner",
            }
        )
        matches = matches.merge(
            shootouts[["date", "home", "away", "shootout_winner"]],
            on=["date", "home", "away"],
            how="left"
        )
    else:
        matches["shootout_winner"] = None

    teams = pd.read_csv(TEAMS_PATH)
    supported_teams = set(teams["team"])

    estimates_path = ROOT / "data" / "team_elo_estimates.csv"

    if estimates_path.exists():
        estimates = pd.read_csv(estimates_path)
        supported_teams.update(estimates["team"])

    supported_mask = (
        matches["home"].isin(supported_teams) &
        matches["away"].isin(supported_teams)
    )

    skipped = len(matches) - int(supported_mask.sum())
    matches = matches[supported_mask].sort_values("date")

    rows = []

    for _, match in matches.iterrows():
        prediction = predict_backtest_match(
            match["home"],
            match["away"],
        )
        probabilities = predicted_probabilities(prediction)
        outcome = actual_outcome(match)
        penalties_outcome = shootout_outcome(match)
        predicted_outcome = max(probabilities, key=probabilities.get)
        favorite_outcome = predicted_outcome
        favorite_probability = probabilities[favorite_outcome]
        outcome_probability = max(probabilities[outcome], 1e-15)
        non_draw_favorite_outcome = (
            "home" if probabilities["home"] >= probabilities["away"] else "away"
        )

        rows.append({
            "date": match["date"],
            "tournament": match["tournament"],
            "home": match["home"],
            "away": match["away"],
            "actual_outcome": outcome,
            "shootout_outcome": penalties_outcome,
            "predicted_outcome": predicted_outcome,
            "favorite_outcome": favorite_outcome,
            "non_draw_favorite_outcome": non_draw_favorite_outcome,
            "favorite_probability": favorite_probability,
            "favorite_bin": probability_bin(favorite_probability),
            "log_loss": -math.log(outcome_probability),
            "brier_score": brier_score(probabilities, outcome),
        })

    return rows, skipped, len(matches) + skipped


def print_summary(title, summary):
    print(title)

    if summary is None:
        print("  No matches")
        return

    print(f"  Matches: {summary['matches']}")
    print(f"  Accuracy 1X2: {summary['accuracy_1x2']}%")
    print(f"  Log loss: {summary['log_loss']}")
    print(f"  Brier score: {summary['brier_score']}")
    print(f"  Favorite accuracy: {summary['favorite_accuracy']}%")
    print(f"  Shootout matches: {summary['shootout_matches']}")

    if summary["shootout_winner_accuracy"] is not None:
        print(
            "  Shootout winner accuracy "
            f"(home/away favorite): {summary['shootout_winner_accuracy']}%"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-date", default="2022-01-01")
    args = parser.parse_args()

    rows, skipped, total_after_cutoff = run_backtest(args.cutoff_date)
    summary = summarize(rows)

    print(f"Cutoff date: {args.cutoff_date}")
    print(f"Historical matches after cutoff: {total_after_cutoff}")
    print(f"Backtested matches: {len(rows)}")
    print(f"Skipped unsupported matches: {skipped}")
    print()

    print_summary("Overall", summary)
    print()

    print("Favorite Probability Bins")
    print(build_bin_summary(rows).to_string(index=False))
    print()

    print("By Tournament")
    rows_by_tournament = {}

    for row in rows:
        rows_by_tournament.setdefault(row["tournament"], []).append(row)

    for tournament in TOURNAMENTS_TO_REPORT:
        tournament_rows = rows_by_tournament.get(tournament, [])
        print_summary(tournament, summarize(tournament_rows))


if __name__ == "__main__":
    main()
