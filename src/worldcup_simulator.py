import pandas as pd
import random
from src.predictor import predict_match
from bisect import bisect_left
from collections import Counter
from functools import lru_cache
import time
from typing import Any


@lru_cache(maxsize=1)
def load_worldcup_groups():
    groups_df = pd.read_csv("data/groups.csv")

    return tuple(
        (group_name, tuple(group_data["team"].tolist()))
        for group_name, group_data in groups_df.groupby("group")
    )


def build_score_distribution(score_probabilities):
    scores = []
    cumulative_probabilities = []
    cumulative_probability = 0.0

    for score in score_probabilities:
        cumulative_probability += score["probability"]
        scores.append((score["home_goals"], score["away_goals"]))
        cumulative_probabilities.append(cumulative_probability)

    if cumulative_probabilities:
        cumulative_probabilities[-1] = 1.0

    return scores, cumulative_probabilities


def sample_from_distribution(distribution):
    scores, cumulative_probabilities = distribution
    selected_index = bisect_left(cumulative_probabilities, random.random())
    return scores[selected_index]


@lru_cache(maxsize=None)
def get_score_distribution(home, away):
    result = predict_match(home, away, include_details=False)
    return build_score_distribution(result["score_probabilities"])


def simulate_score(result):
    distribution = build_score_distribution(result["score_probabilities"])
    return sample_from_distribution(distribution)


def get_head_to_head_tiebreak(team, tied_teams, matches):
    points = 0
    goals_for = 0
    goals_against = 0

    for match in matches:
        home = match["home"]
        away = match["away"]

        if team not in (home, away):
            continue

        opponent = away if home == team else home

        if opponent not in tied_teams:
            continue

        if home == team:
            team_goals = match["home_goals"]
            opponent_goals = match["away_goals"]
        else:
            team_goals = match["away_goals"]
            opponent_goals = match["home_goals"]

        goals_for += team_goals
        goals_against += opponent_goals

        if team_goals > opponent_goals:
            points += 3
        elif team_goals == opponent_goals:
            points += 1

    return points, goals_for - goals_against, goals_for


def sort_group_standings(table, matches):
    teams = list(table.values())

    for team in teams:
        tied_teams = [
            candidate["team"]
            for candidate in teams
            if (
                candidate["points"],
                candidate["gd"],
                candidate["gf"],
            ) == (
                team["points"],
                team["gd"],
                team["gf"],
            )
        ]

        if len(tied_teams) > 1:
            h2h_points, h2h_gd, h2h_gf = get_head_to_head_tiebreak(
                team["team"],
                tied_teams,
                matches
            )
        else:
            h2h_points, h2h_gd, h2h_gf = 0, 0, 0

        team["_h2h_points"] = h2h_points
        team["_h2h_gd"] = h2h_gd
        team["_h2h_gf"] = h2h_gf
        team["_drawing_lots"] = random.random()

    standings = sorted(
        teams,
        key=lambda x: (
            x["points"],
            x["gd"],
            x["gf"],
            x["_h2h_points"],
            x["_h2h_gd"],
            x["_h2h_gf"],
            x["_drawing_lots"],
        ),
        reverse=True
    )

    for team in standings:
        team.pop("_h2h_points", None)
        team.pop("_h2h_gd", None)
        team.pop("_h2h_gf", None)
        team.pop("_drawing_lots", None)

    return standings


def simulate_group(group_name, teams):
    table = {
        team: {
            "team": team,
            "points": 0,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
        }
        for team in teams
    }

    matches = []

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home = teams[i]
            away = teams[j]

            result = predict_match(home, away, include_details=False)
            home_goals, away_goals = simulate_score(result)

            table[home]["played"] += 1
            table[away]["played"] += 1

            table[home]["gf"] += home_goals
            table[home]["ga"] += away_goals

            table[away]["gf"] += away_goals
            table[away]["ga"] += home_goals

            if home_goals > away_goals:
                table[home]["points"] += 3
                table[home]["wins"] += 1
                table[away]["losses"] += 1
            elif home_goals < away_goals:
                table[away]["points"] += 3
                table[away]["wins"] += 1
                table[home]["losses"] += 1
            else:
                table[home]["points"] += 1
                table[away]["points"] += 1
                table[home]["draws"] += 1
                table[away]["draws"] += 1

            matches.append({
                "group": group_name,
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "score": f"{home_goals}-{away_goals}",
            })

    for team in table:
        table[team]["gd"] = table[team]["gf"] - table[team]["ga"]

    standings = sort_group_standings(table, matches)

    return standings, matches


def simulate_groups():
    results = {}

    for group_name, teams in load_worldcup_groups():
        standings, matches = simulate_group(group_name, teams)

        results[group_name] = {
            "standings": standings,
            "matches": matches,
        }

    return results


def get_qualified_teams(group_results):
    qualified = []

    for group_name, data in group_results.items():
        standings = data["standings"]

        qualified.append({
            "group": group_name,
            "position": 1,
            "team": standings[0]["team"]
        })

        qualified.append({
            "group": group_name,
            "position": 2,
            "team": standings[1]["team"]
        })

    return qualified


def create_round_of_32(group_results):
    group_winners = {}
    group_runners = {}
    third_places = {}

    for group_name, data in group_results.items():
        standings = data["standings"]

        group_winners[group_name] = standings[0]["team"]
        group_runners[group_name] = standings[1]["team"]

        third_places[group_name] = {
            "group": group_name,
            "team": standings[2]["team"],
            "points": standings[2]["points"],
            "gd": standings[2]["gd"],
            "gf": standings[2]["gf"],
        }

    best_thirds = sorted(
        third_places.values(),
        key=lambda x: (
            x["points"],
            x["gd"],
            x["gf"]
        ),
        reverse=True
    )[:8]

    best_thirds_by_group = {
        item["group"]: item["team"]
        for item in best_thirds
    }

    def get_third(allowed_groups):
        for group in allowed_groups:
            if group in best_thirds_by_group:
                return best_thirds_by_group.pop(group)

        # Fallback: si no queda ninguno de los grupos permitidos,
        # usa el mejor tercero disponible restante.
        if best_thirds_by_group:
            group, team = next(iter(best_thirds_by_group.items()))
            best_thirds_by_group.pop(group)
            return team

        raise ValueError(
            f"No quedan terceros clasificados disponibles para: {allowed_groups}"
        )

    matches = [
        # Match 73
        (group_runners["A"], group_runners["B"]),

        # Match 74
        (group_winners["E"], get_third(["A", "B", "C", "D", "F"])),

        # Match 75
        (group_winners["F"], group_runners["C"]),

        # Match 76
        (group_winners["C"], group_runners["F"]),

        # Match 77
        (group_winners["I"], get_third(["C", "D", "F", "G", "H"])),

        # Match 78
        (group_runners["E"], group_runners["I"]),

        # Match 79
        (group_winners["A"], get_third(["C", "E", "F", "H", "I"])),

        # Match 80
        (group_winners["L"], get_third(["E", "H", "I", "J", "K"])),

        # Match 81
        (group_winners["D"], get_third(["B", "E", "F", "I", "J"])),

        # Match 82
        (group_winners["G"], get_third(["A", "E", "H", "I", "J"])),

        # Match 83
        (group_runners["K"], group_runners["L"]),

        # Match 84
        (group_winners["H"], group_runners["J"]),

        # Match 85
        (group_winners["B"], get_third(["E", "F", "G", "I", "J"])),

        # Match 86
        (group_winners["J"], group_runners["H"]),

        # Match 87
        (group_winners["K"], get_third(["D", "E", "I", "J", "L"])),

        # Match 88
        (group_runners["D"], group_runners["G"]),
    ]

    return matches
def simulate_knockout_match(team_a, team_b):

    result = predict_match(team_a, team_b, include_details=False)

    scores = result["score_probabilities"]

    selected_score = random.choices(
        scores,
        weights=[s["probability"] for s in scores],
        k=1
    )[0]

    goals_a = selected_score["home_goals"]
    goals_b = selected_score["away_goals"]

    if goals_a > goals_b:
        winner = team_a
        method = "Tiempo regular"

    elif goals_b > goals_a:
        winner = team_b
        method = "Tiempo regular"

    else:
        winner = random.choice([team_a, team_b])
        method = "Penales"

    return {
        "team_a": team_a,
        "score": f"{goals_a}-{goals_b}",
        "team_b": team_b,
        "winner": winner,
        "method": method
    }

def simulate_round_of_32(group_results):
    matches = create_round_of_32(group_results)

    results = []

    for team_a, team_b in matches:
        results.append(
            simulate_knockout_match(team_a, team_b)
        )

    return results

def simulate_knockout_round(teams):
    results = []

    for i in range(0, len(teams), 2):
        team_a = teams[i]
        team_b = teams[i + 1]

        match = simulate_knockout_match(team_a, team_b)
        results.append(match)

    return results


def build_cached_score_sampler():
    def sample_score(home, away):
        return sample_from_distribution(get_score_distribution(home, away))

    return sample_score


def simulate_group_with_score_sampler(group_name, teams, sample_score):
    table = {
        team: {
            "team": team,
            "points": 0,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
        }
        for team in teams
    }
    matches = []

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home = teams[i]
            away = teams[j]
            home_goals, away_goals = sample_score(home, away)

            table[home]["played"] += 1
            table[away]["played"] += 1
            table[home]["gf"] += home_goals
            table[home]["ga"] += away_goals
            table[away]["gf"] += away_goals
            table[away]["ga"] += home_goals

            if home_goals > away_goals:
                table[home]["points"] += 3
                table[home]["wins"] += 1
                table[away]["losses"] += 1
            elif home_goals < away_goals:
                table[away]["points"] += 3
                table[away]["wins"] += 1
                table[home]["losses"] += 1
            else:
                table[home]["points"] += 1
                table[away]["points"] += 1
                table[home]["draws"] += 1
                table[away]["draws"] += 1

            matches.append({
                "group": group_name,
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "score": f"{home_goals}-{away_goals}",
            })

    for team in table:
        table[team]["gd"] = table[team]["gf"] - table[team]["ga"]

    return sort_group_standings(table, matches), matches


def simulate_groups_with_score_sampler(sample_score):
    results = {}

    for group_name, teams in load_worldcup_groups():
        standings, matches = simulate_group_with_score_sampler(
            group_name,
            teams,
            sample_score
        )
        results[group_name] = {
            "standings": standings,
            "matches": matches,
        }

    return results


def simulate_knockout_match_with_score_sampler(team_a, team_b, sample_score):
    goals_a, goals_b = sample_score(team_a, team_b)

    if goals_a > goals_b:
        winner = team_a
        method = "Tiempo regular"
    elif goals_b > goals_a:
        winner = team_b
        method = "Tiempo regular"
    else:
        winner = random.choice([team_a, team_b])
        method = "Penales"

    return {
        "team_a": team_a,
        "score": f"{goals_a}-{goals_b}",
        "team_b": team_b,
        "winner": winner,
        "method": method
    }


def simulate_round_of_32_with_score_sampler(group_results, sample_score):
    matches = create_round_of_32(group_results)

    return [
        simulate_knockout_match_with_score_sampler(team_a, team_b, sample_score)
        for team_a, team_b in matches
    ]


def simulate_knockout_round_with_score_sampler(teams, sample_score):
    return [
        simulate_knockout_match_with_score_sampler(
            teams[i],
            teams[i + 1],
            sample_score
        )
        for i in range(0, len(teams), 2)
    ]

def simulate_many_world_cups(n=1000, verbose=False, progress_callback=None):

    start = time.time()
    sample_score = build_cached_score_sampler()

    champions = []

    for i in range(n):

        if verbose and i % 10 == 0:
            print(f"Simulación {i}/{n}")

        if progress_callback and i % 50 == 0:
            progress_callback(i, n)

        group_results = simulate_groups_with_score_sampler(sample_score)

        if verbose and i < 5:  # solo las primeras 5 simulaciones
            print(f"\n===== MUNDIAL {i+1} =====")

            matches = create_round_of_32(group_results)

            for home, away in matches:
                print(f"{home} vs {away}")

        round32 = simulate_round_of_32_with_score_sampler(
            group_results,
            sample_score
        )

        round32_winners = [
            match["winner"]
            for match in round32
        ]

        round16 = simulate_knockout_round_with_score_sampler(
            round32_winners,
            sample_score
        )

        round16_winners = [
            match["winner"]
            for match in round16
        ]

        quarterfinals = simulate_knockout_round_with_score_sampler(
            round16_winners,
            sample_score
        )

        quarter_winners = [
            match["winner"]
            for match in quarterfinals
        ]

        semifinals = simulate_knockout_round_with_score_sampler(
            quarter_winners,
            sample_score
        )

        semifinal_winners = [
            match["winner"]
            for match in semifinals
        ]

        final = simulate_knockout_round_with_score_sampler(
            semifinal_winners,
            sample_score
        )

        champion = final[0]["winner"]

        champions.append(champion)

    end = time.time()

    if verbose:
        print(f"Tiempo total: {end - start:.2f} segundos")

    if progress_callback:
        progress_callback(n, n)

    counts = Counter(champions)

    results = []

    for team, wins in counts.most_common():

        results.append({
            "team": team,
            "titles": wins,
            "probability": round((wins / n) * 100, 2)
        })

    return results

def simulate_many_world_cups_with_stages(n=1000, verbose=False, progress_callback=None):

    stage_counts = {}
    sample_score = build_cached_score_sampler()

    def register_team(team):
        if team not in stage_counts:
            stage_counts[team] = {
                "team": team,
                "round32": 0,
                "round16": 0,
                "quarterfinals": 0,
                "semifinals": 0,
                "final": 0,
                "champion": 0,
            }

    for i in range(n):

        if verbose and i % 10 == 0:
            print(f"Simulación por fases {i}/{n}")

        if progress_callback and i % 50 == 0:
            progress_callback(i, n)

        group_results = simulate_groups_with_score_sampler(sample_score)

        round32 = simulate_round_of_32_with_score_sampler(
            group_results,
            sample_score
        )

        round32_teams = []

        for match in round32:
            register_team(match["team_a"])
            register_team(match["team_b"])

            stage_counts[match["team_a"]]["round32"] += 1
            stage_counts[match["team_b"]]["round32"] += 1

            round32_teams.append(match["winner"])

        round16 = simulate_knockout_round_with_score_sampler(
            round32_teams,
            sample_score
        )

        round16_teams = []

        for match in round16:
            register_team(match["team_a"])
            register_team(match["team_b"])

            stage_counts[match["team_a"]]["round16"] += 1
            stage_counts[match["team_b"]]["round16"] += 1

            round16_teams.append(match["winner"])

        quarterfinals = simulate_knockout_round_with_score_sampler(
            round16_teams,
            sample_score
        )

        quarterfinal_teams = []

        for match in quarterfinals:
            register_team(match["team_a"])
            register_team(match["team_b"])

            stage_counts[match["team_a"]]["quarterfinals"] += 1
            stage_counts[match["team_b"]]["quarterfinals"] += 1

            quarterfinal_teams.append(match["winner"])

        semifinals = simulate_knockout_round_with_score_sampler(
            quarterfinal_teams,
            sample_score
        )

        semifinal_teams = []

        for match in semifinals:
            register_team(match["team_a"])
            register_team(match["team_b"])

            stage_counts[match["team_a"]]["semifinals"] += 1
            stage_counts[match["team_b"]]["semifinals"] += 1

            semifinal_teams.append(match["winner"])

        final = simulate_knockout_round_with_score_sampler(
            semifinal_teams,
            sample_score
        )

        final_teams = []

        for match in final:
            register_team(match["team_a"])
            register_team(match["team_b"])

            stage_counts[match["team_a"]]["final"] += 1
            stage_counts[match["team_b"]]["final"] += 1

            final_teams.append(match["winner"])

        champion = final[0]["winner"]
        register_team(champion)
        stage_counts[champion]["champion"] += 1

    results = []

    for team, counts in stage_counts.items():
        results.append({
            "team": team,
            "round32_pct": round((counts["round32"] / n) * 100, 2),
            "round16_pct": round((counts["round16"] / n) * 100, 2),
            "quarterfinals_pct": round((counts["quarterfinals"] / n) * 100, 2),
            "semifinals_pct": round((counts["semifinals"] / n) * 100, 2),
            "final_pct": round((counts["final"] / n) * 100, 2),
            "champion_pct": round((counts["champion"] / n) * 100, 2),
        })

    results = sorted(
        results,
        key=lambda x: x["champion_pct"],
        reverse=True
    )

    if progress_callback:
        progress_callback(n, n)

    return results


def simulate_team_world_cup_path(team_name, n=10000):
    sample_score_cached = build_cached_score_sampler()

    def simulate_group_cached(group_name, teams):
        table = {
            team: {
                "team": team,
                "points": 0,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
            }
            for team in teams
        }
        matches = []

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home = teams[i]
                away = teams[j]
                home_goals, away_goals = sample_score_cached(home, away)

                table[home]["played"] += 1
                table[away]["played"] += 1
                table[home]["gf"] += home_goals
                table[home]["ga"] += away_goals
                table[away]["gf"] += away_goals
                table[away]["ga"] += home_goals

                if home_goals > away_goals:
                    table[home]["points"] += 3
                    table[home]["wins"] += 1
                    table[away]["losses"] += 1
                elif home_goals < away_goals:
                    table[away]["points"] += 3
                    table[away]["wins"] += 1
                    table[home]["losses"] += 1
                else:
                    table[home]["points"] += 1
                    table[away]["points"] += 1
                    table[home]["draws"] += 1
                    table[away]["draws"] += 1

                matches.append({
                    "group": group_name,
                    "home": home,
                    "away": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "score": f"{home_goals}-{away_goals}",
                })

        for team in table:
            table[team]["gd"] = table[team]["gf"] - table[team]["ga"]

        return sort_group_standings(table, matches), matches

    def simulate_groups_cached():
        results = {}

        for group_name, teams in load_worldcup_groups():
            standings, matches = simulate_group_cached(group_name, teams)
            results[group_name] = {
                "standings": standings,
                "matches": matches,
            }

        return results

    def simulate_knockout_match_cached(team_a, team_b):
        goals_a, goals_b = sample_score_cached(team_a, team_b)

        if goals_a > goals_b:
            winner = team_a
            method = "Tiempo regular"
        elif goals_b > goals_a:
            winner = team_b
            method = "Tiempo regular"
        else:
            winner = random.choice([team_a, team_b])
            method = "Penales"

        return {
            "team_a": team_a,
            "score": f"{goals_a}-{goals_b}",
            "team_b": team_b,
            "winner": winner,
            "method": method,
        }

    def simulate_knockout_round_cached(teams):
        return [
            simulate_knockout_match_cached(teams[i], teams[i + 1])
            for i in range(0, len(teams), 2)
        ]

    def simulate_round_of_32_cached(group_results):
        return [
            simulate_knockout_match_cached(team_a, team_b)
            for team_a, team_b in create_round_of_32(group_results)
        ]

    stage_counts = {
        "round32": 0,
        "round16": 0,
        "quarterfinals": 0,
        "semifinals": 0,
        "final": 0,
        "champion": 0,
    }
    elimination_counts = {
        "Fase de grupos": 0,
        "Dieciseisavos": 0,
        "Octavos": 0,
        "Cuartos": 0,
        "Semifinales": 0,
        "Subcampeón": 0,
        "Campeón": 0,
    }
    group_positions = []
    group_points = []
    group_goals_for = []
    group_goals_against = []
    tournament_stat_totals = {
        "matches": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
        "points": 0,
        "penalty_shootouts": 0,
        "penalty_shootout_wins": 0,
        "penalty_shootout_losses": 0,
    }
    group_position_counts = Counter()
    group_rival_above_counts = Counter()
    best_third_count = 0
    round_opponent_counts = {
        "Dieciseisavos": Counter(),
        "Octavos": Counter(),
        "Cuartos": Counter(),
        "Semifinales": Counter(),
        "Final": Counter(),
    }
    match_result_totals = {}
    eliminator_counts = Counter()

    def get_opponent(match):
        if match["team_a"] == team_name:
            return match["team_b"]

        if match["team_b"] == team_name:
            return match["team_a"]

        return None

    def record_round_opponent(round_name, matches):
        for match in matches:
            opponent = get_opponent(match)

            if opponent:
                round_opponent_counts[round_name][opponent] += 1
                return opponent

        return None

    def build_counter_rows(counter, total, label_key="team", top_n=None):
        rows = []
        items = counter.most_common(top_n)

        for label, count in items:
            rows.append({
                label_key: label,
                "count": count,
                "percentage": round((count / total) * 100, 2)
            })

        return rows

    def add_simulation_stats(simulation_stats):
        simulation_stats["goal_difference"] = (
            simulation_stats["goals_for"] -
            simulation_stats["goals_against"]
        )

        for key in tournament_stat_totals:
            tournament_stat_totals[key] += simulation_stats[key]

    def parse_score(score):
        goals_a, goals_b = score.split("-")
        return int(goals_a), int(goals_b)

    def add_knockout_match_stats(simulation_stats, match):
        if team_name not in (match["team_a"], match["team_b"]):
            return

        goals_a, goals_b = parse_score(match["score"])

        if match["team_a"] == team_name:
            goals_for = goals_a
            goals_against = goals_b
        else:
            goals_for = goals_b
            goals_against = goals_a

        simulation_stats["matches"] += 1
        simulation_stats["goals_for"] += goals_for
        simulation_stats["goals_against"] += goals_against

        if goals_for > goals_against:
            simulation_stats["wins"] += 1
            simulation_stats["points"] += 3
        elif goals_for < goals_against:
            simulation_stats["losses"] += 1
        else:
            simulation_stats["draws"] += 1
            simulation_stats["points"] += 1

            if match["method"] == "Penales":
                simulation_stats["penalty_shootouts"] += 1

                if match["winner"] == team_name:
                    simulation_stats["penalty_shootout_wins"] += 1
                else:
                    simulation_stats["penalty_shootout_losses"] += 1

    def record_match_result(stage, opponent, goals_for, goals_against):
        key = (stage, opponent)

        if key not in match_result_totals:
            match_result_totals[key] = {
                "stage": stage,
                "opponent": opponent,
                "matches": 0,
                "goals_for": 0,
                "goals_against": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
            }

        row = match_result_totals[key]
        row["matches"] += 1
        row["goals_for"] += goals_for
        row["goals_against"] += goals_against

        if goals_for > goals_against:
            row["wins"] += 1
        elif goals_for < goals_against:
            row["losses"] += 1
        else:
            row["draws"] += 1

    def record_group_match_results(group_data):
        for match in group_data["matches"]:
            if team_name not in (match["home"], match["away"]):
                continue

            if match["home"] == team_name:
                opponent = match["away"]
                goals_for = match["home_goals"]
                goals_against = match["away_goals"]
            else:
                opponent = match["home"]
                goals_for = match["away_goals"]
                goals_against = match["home_goals"]

            record_match_result(
                "Fase de grupos",
                opponent,
                goals_for,
                goals_against,
            )

    def record_knockout_match_result(stage, match):
        if team_name not in (match["team_a"], match["team_b"]):
            return

        goals_a, goals_b = parse_score(match["score"])

        if match["team_a"] == team_name:
            opponent = match["team_b"]
            goals_for = goals_a
            goals_against = goals_b
        else:
            opponent = match["team_a"]
            goals_for = goals_b
            goals_against = goals_a

        record_match_result(stage, opponent, goals_for, goals_against)

    for _ in range(n):
        group_results = simulate_groups_cached()
        team_group_data = None
        team_standing = None
        simulation_stats = {
            "matches": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
            "penalty_shootouts": 0,
            "penalty_shootout_wins": 0,
            "penalty_shootout_losses": 0,
        }

        for group_name, data in group_results.items():
            team_group_position = None
            positions_by_team = {
                standing["team"]: index
                for index, standing in enumerate(data["standings"], start=1)
            }

            for index, standing in enumerate(data["standings"], start=1):
                if standing["team"] == team_name:
                    team_group_data = data
                    team_standing = standing
                    team_group_position = index
                    group_positions.append(index)
                    group_position_counts[index] += 1
                    group_points.append(standing["points"])
                    group_goals_for.append(standing["gf"])
                    group_goals_against.append(standing["ga"])
                    simulation_stats["matches"] += standing["played"]
                    simulation_stats["wins"] += standing["wins"]
                    simulation_stats["draws"] += standing["draws"]
                    simulation_stats["losses"] += standing["losses"]
                    simulation_stats["goals_for"] += standing["gf"]
                    simulation_stats["goals_against"] += standing["ga"]
                    simulation_stats["points"] += standing["points"]
                    break

            if team_group_data and team_group_position is not None:
                for opponent, opponent_position in positions_by_team.items():
                    if opponent != team_name and team_group_position < opponent_position:
                        group_rival_above_counts[opponent] += 1

                record_group_match_results(team_group_data)
                break

        round32 = simulate_round_of_32_cached(group_results)
        round32_teams = []

        if any(
            match["team_a"] == team_name or match["team_b"] == team_name
            for match in round32
        ):
            stage_counts["round32"] += 1

            if group_positions[-1] == 3:
                best_third_count += 1
        else:
            elimination_counts["Fase de grupos"] += 1
            add_simulation_stats(simulation_stats)
            continue

        record_round_opponent("Dieciseisavos", round32)

        eliminated = False

        for match in round32:
            round32_teams.append(match["winner"])
            add_knockout_match_stats(simulation_stats, match)
            record_knockout_match_result("Dieciseisavos", match)

            if (
                match["team_a"] == team_name or match["team_b"] == team_name
            ) and match["winner"] != team_name:
                elimination_counts["Dieciseisavos"] += 1
                eliminator_counts[get_opponent(match)] += 1
                eliminated = True

        if eliminated:
            add_simulation_stats(simulation_stats)
            continue

        round16 = simulate_knockout_round_cached(round32_teams)
        round16_teams = []

        if any(
            match["team_a"] == team_name or match["team_b"] == team_name
            for match in round16
        ):
            stage_counts["round16"] += 1
            record_round_opponent("Octavos", round16)

        for match in round16:
            round16_teams.append(match["winner"])
            add_knockout_match_stats(simulation_stats, match)
            record_knockout_match_result("Octavos", match)

            if (
                match["team_a"] == team_name or match["team_b"] == team_name
            ) and match["winner"] != team_name:
                elimination_counts["Octavos"] += 1
                eliminator_counts[get_opponent(match)] += 1
                eliminated = True

        if eliminated:
            add_simulation_stats(simulation_stats)
            continue

        quarterfinals = simulate_knockout_round_cached(round16_teams)
        quarterfinal_teams = []

        if any(
            match["team_a"] == team_name or match["team_b"] == team_name
            for match in quarterfinals
        ):
            stage_counts["quarterfinals"] += 1
            record_round_opponent("Cuartos", quarterfinals)

        for match in quarterfinals:
            quarterfinal_teams.append(match["winner"])
            add_knockout_match_stats(simulation_stats, match)
            record_knockout_match_result("Cuartos", match)

            if (
                match["team_a"] == team_name or match["team_b"] == team_name
            ) and match["winner"] != team_name:
                elimination_counts["Cuartos"] += 1
                eliminator_counts[get_opponent(match)] += 1
                eliminated = True

        if eliminated:
            add_simulation_stats(simulation_stats)
            continue

        semifinals = simulate_knockout_round_cached(quarterfinal_teams)
        semifinal_teams = []

        if any(
            match["team_a"] == team_name or match["team_b"] == team_name
            for match in semifinals
        ):
            stage_counts["semifinals"] += 1
            record_round_opponent("Semifinales", semifinals)

        for match in semifinals:
            semifinal_teams.append(match["winner"])
            add_knockout_match_stats(simulation_stats, match)
            record_knockout_match_result("Semifinales", match)

            if (
                match["team_a"] == team_name or match["team_b"] == team_name
            ) and match["winner"] != team_name:
                elimination_counts["Semifinales"] += 1
                eliminator_counts[get_opponent(match)] += 1
                eliminated = True

        if eliminated:
            add_simulation_stats(simulation_stats)
            continue

        final = simulate_knockout_round_cached(semifinal_teams)

        if any(
            match["team_a"] == team_name or match["team_b"] == team_name
            for match in final
        ):
            stage_counts["final"] += 1
            record_round_opponent("Final", final)

        champion = final[0]["winner"]
        add_knockout_match_stats(simulation_stats, final[0])
        record_knockout_match_result("Final", final[0])

        if champion == team_name:
            stage_counts["champion"] += 1
            elimination_counts["Campeón"] += 1
        else:
            elimination_counts["Subcampeón"] += 1
            eliminator_counts[get_opponent(final[0])] += 1

        add_simulation_stats(simulation_stats)

    def pct(value):
        return round((value / n) * 100, 2)

    average_tournament_stats = {
        key: round(value / n, 2)
        for key, value in tournament_stat_totals.items()
    }

    most_likely_elimination = max(
        elimination_counts.items(),
        key=lambda item: item[1]
    )[0]

    group_position_distribution = [
        {
            "position": position,
            "percentage": pct(group_position_counts[position])
        }
        for position in [1, 2, 3, 4]
    ]
    round_opponents = {
        round_name: build_counter_rows(counter, n, label_key="opponent", top_n=10)
        for round_name, counter in round_opponent_counts.items()
    }
    group_rivals = build_counter_rows(
        group_rival_above_counts,
        n,
        label_key="opponent"
    )
    eliminators = build_counter_rows(
        eliminator_counts,
        max(sum(eliminator_counts.values()), 1),
        label_key="opponent"
    )
    stage_order = {
        "Fase de grupos": 0,
        "Dieciseisavos": 1,
        "Octavos": 2,
        "Cuartos": 3,
        "Semifinales": 4,
        "Final": 5,
    }
    average_match_results = []

    for row in match_result_totals.values():
        matches = row["matches"]
        average_match_results.append({
            "stage": row["stage"],
            "opponent": row["opponent"],
            "matches": matches,
            "appearance_pct": round((matches / n) * 100, 2),
            "avg_goals_for": round(row["goals_for"] / matches, 2),
            "avg_goals_against": round(row["goals_against"] / matches, 2),
            "avg_goal_difference": round(
                (row["goals_for"] - row["goals_against"]) / matches,
                2
            ),
            "win_pct": round((row["wins"] / matches) * 100, 2),
            "draw_pct": round((row["draws"] / matches) * 100, 2),
            "loss_pct": round((row["losses"] / matches) * 100, 2),
        })

    average_match_results = sorted(
        average_match_results,
        key=lambda item: (
            stage_order.get(item["stage"], 99),
            -item["matches"],
            item["opponent"],
        )
    )

    most_likely_path: dict[str, Any] = {
        "Grupo": group_rivals
    }

    for round_name, counter in round_opponent_counts.items():
        if counter:
            opponent, count = counter.most_common(1)[0]
            most_likely_path[round_name] = {
                "opponent": opponent,
                "percentage": round((count / n) * 100, 2)
            }
        else:
            most_likely_path[round_name] = {
                "opponent": None,
                "percentage": 0
            }

    top_eliminators = [
        f"{item['opponent']} ({item['percentage']}%)"
        for item in eliminators[:2]
    ]
    eliminator_text = (
        " y ".join(top_eliminators)
        if top_eliminators
        else "ningún rival en particular"
    )
    quarter_opponent = most_likely_path["Cuartos"]["opponent"]
    semifinal_opponent = most_likely_path["Semifinales"]["opponent"]
    path_bits = []

    if quarter_opponent:
        path_bits.append(f"{quarter_opponent} en cuartos")

    if semifinal_opponent:
        path_bits.append(f"{semifinal_opponent} en semifinales")

    common_path_text = (
        " y ".join(path_bits)
        if path_bits
        else "rondas variables según la simulación"
    )
    group_match_results = [
        row for row in average_match_results
        if row["stage"] == "Fase de grupos"
    ]
    hardest_group_match = (
        min(group_match_results, key=lambda row: row["avg_goal_difference"])
        if group_match_results
        else None
    )
    best_group_match = (
        max(group_match_results, key=lambda row: row["avg_goal_difference"])
        if group_match_results
        else None
    )
    round32_path = most_likely_path["Dieciseisavos"]
    round32_text = (
        f"{round32_path['opponent']} ({round32_path['percentage']}%)"
        if round32_path["opponent"]
        else "sin cruce dominante"
    )
    hardest_group_text = (
        f"{hardest_group_match['opponent']} "
        f"(DG {hardest_group_match['avg_goal_difference']})"
        if hardest_group_match
        else "sin rival claro"
    )
    best_group_text = (
        f"{best_group_match['opponent']} "
        f"(DG {best_group_match['avg_goal_difference']})"
        if best_group_match
        else "sin rival claro"
    )
    summary = (
        f"{team_name} promedia {average_tournament_stats['matches']} partidos "
        f"por Mundial simulado, con {average_tournament_stats['goals_for']} "
        f"goles a favor y {average_tournament_stats['goals_against']} en contra. "
        f"Gana su grupo el {pct(group_position_counts[1])}% de las veces y "
        f"alcanza semifinales el {pct(stage_counts['semifinals'])}%. "
        f"En fase de grupos, su cruce más favorable es {best_group_text} "
        f"y el más exigente es {hardest_group_text}. "
        f"Su rival más frecuente en dieciseisavos es {round32_text}; "
        f"cuando queda eliminado, los rivales que más lo sacan son "
        f"{eliminator_text}. Su camino más común hacia rondas finales pasa "
        f"por {common_path_text}."
    )

    return {
        "team": team_name,
        "simulations": n,
        "round32_pct": pct(stage_counts["round32"]),
        "round16_pct": pct(stage_counts["round16"]),
        "quarterfinals_pct": pct(stage_counts["quarterfinals"]),
        "semifinals_pct": pct(stage_counts["semifinals"]),
        "final_pct": pct(stage_counts["final"]),
        "champion_pct": pct(stage_counts["champion"]),
        "group_stage_elimination_pct": pct(elimination_counts["Fase de grupos"]),
        "round32_elimination_pct": pct(elimination_counts["Dieciseisavos"]),
        "round16_elimination_pct": pct(elimination_counts["Octavos"]),
        "quarterfinals_elimination_pct": pct(elimination_counts["Cuartos"]),
        "semifinals_elimination_pct": pct(elimination_counts["Semifinales"]),
        "runner_up_pct": pct(elimination_counts["Subcampeón"]),
        "most_likely_elimination": most_likely_elimination,
        "average_group_position": round(sum(group_positions) / n, 2),
        "average_group_points": round(sum(group_points) / n, 2),
        "average_group_goals_for": round(sum(group_goals_for) / n, 2),
        "average_group_goals_against": round(sum(group_goals_against) / n, 2),
        "average_tournament_stats": average_tournament_stats,
        "average_match_results": average_match_results,
        "group_position_distribution": group_position_distribution,
        "group_rivals": group_rivals,
        "group_winner_pct": pct(group_position_counts[1]),
        "group_runner_up_pct": pct(group_position_counts[2]),
        "best_third_pct": pct(best_third_count),
        "round_opponents": round_opponents,
        "eliminators": eliminators,
        "most_likely_path": most_likely_path,
        "summary": summary,
        "stage_counts": stage_counts,
        "elimination_counts": elimination_counts,
    }
