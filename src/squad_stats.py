from functools import lru_cache
from datetime import date
import re

import pandas as pd


@lru_cache(maxsize=1)
def load_worldcup_squads():
    try:
        squads = pd.read_csv("data/worldcup_squads.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    squads["birth_date"] = pd.to_datetime(squads["birth_date"], errors="coerce")
    squads["height_cm"] = pd.to_numeric(squads["height_cm"], errors="coerce")

    return squads.fillna("")


def calculate_age(birth_date, reference_date=None):
    if pd.isna(birth_date) or birth_date == "":
        return None

    reference_date = reference_date or date.today()

    if isinstance(reference_date, str):
        reference_date = pd.Timestamp(reference_date).date()

    birth_date = pd.Timestamp(birth_date).date()
    age = reference_date.year - birth_date.year

    if (reference_date.month, reference_date.day) < (
        birth_date.month,
        birth_date.day
    ):
        age -= 1

    return age


def extract_club_country(club):
    match = re.search(r"\(([A-Z]{3})\)\s*$", str(club))

    if match:
        return match.group(1)

    return "UNK"


def get_team_squad(team_name, reference_date="2026-06-11"):
    squads = load_worldcup_squads()

    if squads.empty:
        return pd.DataFrame()

    team_squad = squads[squads["team"] == team_name].copy()

    if team_squad.empty:
        return pd.DataFrame()

    team_squad["age"] = team_squad["birth_date"].map(
        lambda value: calculate_age(value, reference_date)
    )
    team_squad["club_country"] = team_squad["club"].map(extract_club_country)

    return team_squad


def count_rows(df, column, value_name):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, value_name]
    return counts.to_dict("records")


def get_squad_profile(team_name):
    squad = get_team_squad(team_name)

    if squad.empty:
        return None

    youngest = squad.sort_values("age").head(5)
    oldest = squad.sort_values("age", ascending=False).head(5)
    tallest = squad.sort_values("height_cm", ascending=False).head(5)
    position_profile = (
        squad.groupby("position", as_index=False)
        .agg(
            players=("player_name", "count"),
            avg_age=("age", "mean"),
            avg_height_cm=("height_cm", "mean"),
        )
        .sort_values("position")
    )
    position_profile["avg_age"] = position_profile["avg_age"].round(1)
    position_profile["avg_height_cm"] = position_profile["avg_height_cm"].round(1)

    coach = squad.iloc[0]

    return {
        "team": team_name,
        "players": len(squad),
        "average_age": round(float(squad["age"].mean()), 1),
        "average_height_cm": round(float(squad["height_cm"].mean()), 1),
        "youngest_age": int(squad["age"].min()),
        "oldest_age": int(squad["age"].max()),
        "coach_name": coach.get("coach_name", ""),
        "coach_nationality": coach.get("coach_nationality", ""),
        "position_counts": count_rows(squad, "position", "players"),
        "club_country_counts": count_rows(squad, "club_country", "players"),
        "position_profile": position_profile.to_dict("records"),
        "youngest_players": youngest[[
            "player_name",
            "position",
            "age",
            "club",
        ]].to_dict("records"),
        "oldest_players": oldest[[
            "player_name",
            "position",
            "age",
            "club",
        ]].to_dict("records"),
        "tallest_players": tallest[[
            "player_name",
            "position",
            "height_cm",
            "club",
        ]].to_dict("records"),
        "squad": squad[[
            "number",
            "position",
            "player_name",
            "age",
            "club",
            "club_country",
            "height_cm",
        ]].sort_values("number").to_dict("records"),
    }
