import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_goalscorers, normalize_team_columns


MATCHES_PATH = ROOT / "data" / "matches_history.csv"
GROUPS_PATH = ROOT / "data" / "groups.csv"

DEFAULT_FOCUS_TEAMS = [
    "Japan",
    "Morocco",
    "Spain",
    "Argentina",
    "France",
    "England",
    "Brazil",
    "Portugal",
    "Curacao",
]


def bool_series(series):
    if series.dtype == bool:
        return series

    return series.astype(str).str.upper().eq("TRUE")


def load_matches(cutoff_date):
    matches = pd.read_csv(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = normalize_team_columns(matches, ["home", "away"], date_column="date")

    if cutoff_date:
        matches = matches[matches["date"] >= pd.Timestamp(cutoff_date)]

    return matches.copy()


def add_match_key(df, home_col="home", away_col="away"):
    result = df.copy()
    result["match_key"] = (
        result["date"].dt.strftime("%Y-%m-%d")
        + "|"
        + result[home_col].astype(str)
        + "|"
        + result[away_col].astype(str)
    )
    return result


def build_team_match_rows(matches):
    home_rows = matches[[
        "date",
        "match_key",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "tournament",
    ]].rename(
        columns={
            "home": "team",
            "away": "opponent",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
        }
    )
    away_rows = matches[[
        "date",
        "match_key",
        "away",
        "home",
        "away_goals",
        "home_goals",
        "tournament",
    ]].rename(
        columns={
            "away": "team",
            "home": "opponent",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
        }
    )

    return pd.concat([home_rows, away_rows], ignore_index=True)


def build_goal_event_summary(matches):
    goalscorers = load_goalscorers()

    if goalscorers.empty:
        return pd.DataFrame(), pd.DataFrame()

    goalscorers = goalscorers.copy()
    goalscorers["own_goal"] = bool_series(goalscorers["own_goal"])
    goalscorers["penalty"] = bool_series(goalscorers["penalty"])

    if not matches.empty:
        min_date = matches["date"].min()
        max_date = matches["date"].max()
        goalscorers = goalscorers[
            (goalscorers["date"] >= min_date) &
            (goalscorers["date"] <= max_date)
        ].copy()

    goalscorers = add_match_key(
        goalscorers,
        home_col="home_team",
        away_col="away_team"
    )
    valid_keys = set(matches["match_key"])
    goalscorers = goalscorers[goalscorers["match_key"].isin(valid_keys)].copy()

    goal_events = goalscorers.groupby(
        ["match_key", "team"],
        as_index=False
    ).agg(
        classified_goals=("team", "size"),
        penalty_goals=("penalty", "sum"),
        own_goals_for=("own_goal", "sum"),
    )
    goal_events["normal_goals"] = (
        goal_events["classified_goals"]
        - goal_events["penalty_goals"]
        - goal_events["own_goals_for"]
    )

    return goalscorers, goal_events


def summarize_goal_sources(matches):
    matches = add_match_key(matches)
    team_matches = build_team_match_rows(matches)
    goalscorers, goal_events = build_goal_event_summary(matches)

    for_events = goal_events.rename(
        columns={
            "team": "team",
            "classified_goals": "classified_goals_for",
            "penalty_goals": "penalty_goals_for",
            "own_goals_for": "own_goals_for",
            "normal_goals": "normal_goals_for",
        }
    )
    against_events = goal_events.rename(
        columns={
            "team": "opponent",
            "classified_goals": "classified_goals_against",
            "penalty_goals": "penalty_goals_against",
            "own_goals_for": "own_goals_against",
            "normal_goals": "normal_goals_against",
        }
    )

    team_matches = team_matches.merge(
        for_events,
        on=["match_key", "team"],
        how="left"
    )
    team_matches = team_matches.merge(
        against_events[[
            "match_key",
            "opponent",
            "classified_goals_against",
            "penalty_goals_against",
            "own_goals_against",
            "normal_goals_against",
        ]],
        on=["match_key", "opponent"],
        how="left"
    )

    event_columns = [
        "classified_goals_for",
        "penalty_goals_for",
        "own_goals_for",
        "normal_goals_for",
        "classified_goals_against",
        "penalty_goals_against",
        "own_goals_against",
        "normal_goals_against",
    ]
    team_matches[event_columns] = team_matches[event_columns].fillna(0)
    team_matches["goals_for"] = team_matches["goals_for"].astype(float)
    team_matches["goals_against"] = team_matches["goals_against"].astype(float)
    team_matches["full_event_coverage"] = (
        team_matches["classified_goals_for"].eq(team_matches["goals_for"]) &
        team_matches["classified_goals_against"].eq(team_matches["goals_against"])
    )

    summary = team_matches.groupby("team", as_index=False).agg(
        matches=("match_key", "nunique"),
        covered_matches=("full_event_coverage", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        classified_goals_for=("classified_goals_for", "sum"),
        normal_goals_for=("normal_goals_for", "sum"),
        penalty_goals_for=("penalty_goals_for", "sum"),
        own_goals_for=("own_goals_for", "sum"),
        classified_goals_against=("classified_goals_against", "sum"),
        normal_goals_against=("normal_goals_against", "sum"),
        penalty_goals_against=("penalty_goals_against", "sum"),
        own_goals_against=("own_goals_against", "sum"),
    )

    def safe_pct(numerator, denominator):
        denominator = denominator.astype(float)
        values = numerator.astype(float).div(denominator.where(denominator > 0))
        return (values * 100).round(2).fillna(0)

    summary["coverage_pct"] = safe_pct(
        summary["covered_matches"],
        summary["matches"]
    )
    summary["penalty_share_for_pct"] = safe_pct(
        summary["penalty_goals_for"],
        summary["classified_goals_for"]
    )
    summary["own_goal_share_for_pct"] = safe_pct(
        summary["own_goals_for"],
        summary["classified_goals_for"]
    )
    summary["penalty_share_against_pct"] = safe_pct(
        summary["penalty_goals_against"],
        summary["classified_goals_against"]
    )
    summary["own_goal_share_against_pct"] = safe_pct(
        summary["own_goals_against"],
        summary["classified_goals_against"]
    )

    return summary, team_matches, goalscorers


def print_table(title, df, columns=None, limit=None):
    print(title)

    if limit:
        df = df.head(limit)

    if columns:
        df = df[columns]

    if df.empty:
        print("  No data")
    else:
        print(df.to_string(index=False))

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-date", default="2018-01-01")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--teams", nargs="*", default=DEFAULT_FOCUS_TEAMS)
    args = parser.parse_args()

    matches = load_matches(args.cutoff_date)
    summary, team_matches, goalscorers = summarize_goal_sources(matches)

    summary = summary.sort_values("goals_for", ascending=False)
    focus = summary[summary["team"].isin(args.teams)].sort_values("team")
    worldcup_teams = set()

    if GROUPS_PATH.exists():
        groups = pd.read_csv(GROUPS_PATH)
        worldcup_teams = set(groups["team"])

    worldcup_summary = summary[summary["team"].isin(worldcup_teams)].copy()

    columns = [
        "team",
        "matches",
        "coverage_pct",
        "goals_for",
        "classified_goals_for",
        "normal_goals_for",
        "penalty_goals_for",
        "own_goals_for",
        "penalty_share_for_pct",
        "own_goal_share_for_pct",
        "goals_against",
        "penalty_goals_against",
        "own_goals_against",
        "penalty_share_against_pct",
    ]

    print(f"Cutoff date: {args.cutoff_date}")
    print(f"Matches analyzed: {len(matches)}")
    print(f"Goal events matched: {len(goalscorers)}")
    print()

    print_table("Focus Teams", focus, columns)

    top_penalties = worldcup_summary[
        worldcup_summary["classified_goals_for"] >= 10
    ].sort_values("penalty_share_for_pct", ascending=False)
    print_table(
        "World Cup Teams - Highest Penalty Share",
        top_penalties,
        [
            "team",
            "classified_goals_for",
            "penalty_goals_for",
            "penalty_share_for_pct",
            "coverage_pct",
        ],
        args.top
    )

    top_own_goals = worldcup_summary[
        worldcup_summary["classified_goals_for"] >= 10
    ].sort_values("own_goal_share_for_pct", ascending=False)
    print_table(
        "World Cup Teams - Highest Own Goal Benefit Share",
        top_own_goals,
        [
            "team",
            "classified_goals_for",
            "own_goals_for",
            "own_goal_share_for_pct",
            "coverage_pct",
        ],
        args.top
    )

    defensive_penalties = worldcup_summary[
        worldcup_summary["classified_goals_against"] >= 10
    ].sort_values("penalty_share_against_pct", ascending=False)
    print_table(
        "World Cup Teams - Highest Penalty Goals Against Share",
        defensive_penalties,
        [
            "team",
            "classified_goals_against",
            "penalty_goals_against",
            "penalty_share_against_pct",
            "coverage_pct",
        ],
        args.top
    )

    low_coverage = worldcup_summary.sort_values("coverage_pct")
    print_table(
        "World Cup Teams - Lowest Event Coverage",
        low_coverage,
        ["team", "matches", "covered_matches", "coverage_pct"],
        args.top
    )


if __name__ == "__main__":
    main()
