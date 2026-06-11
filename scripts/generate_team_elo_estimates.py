from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import normalize_team_columns

MATCHES_PATH = ROOT / "data" / "matches_history.csv"
TEAMS_PATH = ROOT / "data" / "teams.csv"
CONFEDERATIONS_PATH = ROOT / "data" / "team_confederations.csv"
ESTIMATES_PATH = ROOT / "data" / "team_elo_estimates.csv"

FALLBACK_CONFEDERATION = "UNKNOWN"

CONFEDERATION_BASE_ELO = {
    "UEFA": 1580,
    "CONMEBOL": 1620,
    "CAF": 1450,
    "AFC": 1430,
    "CONCACAF": 1420,
    "OFC": 1250,
    "OTHER": 1200,
    FALLBACK_CONFEDERATION: 1400,
}

CONFEDERATION_TEAMS = {
    "UEFA": {
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
        "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
        "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo",
        "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Montenegro", "Netherlands", "North Macedonia",
        "Northern Ireland", "Norway", "Poland", "Portugal",
        "Republic of Ireland", "Romania", "Russia", "San Marino", "Scotland",
        "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "Turkey", "Ukraine", "Wales",
    },
    "CONMEBOL": {
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    },
    "CAF": {
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad",
        "Comoros", "Congo", "DR Congo", "Djibouti", "Egypt",
        "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
        "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast",
        "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi",
        "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique",
        "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe",
        "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
        "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
        "Zambia", "Zimbabwe",
    },
    "AFC": {
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan",
        "Brunei", "Cambodia", "China", "Chinese Taipei", "Guam", "Hong Kong",
        "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait",
        "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives",
        "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan",
        "Palestine", "Philippines", "Qatar", "Saudi Arabia", "Singapore",
        "South Korea", "Sri Lanka", "Syria", "Tajikistan", "Thailand",
        "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam",
        "Yemen",
    },
    "CONCACAF": {
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
        "Belize", "Bermuda", "British Virgin Islands", "Canada",
        "Cayman Islands", "Costa Rica", "Cuba", "Curacao", "Dominica",
        "Dominican Republic", "El Salvador", "French Guiana", "Grenada",
        "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica",
        "Martinique", "Mexico", "Montserrat", "Nicaragua", "Panama",
        "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Martin", "Saint Vincent and the Grenadines", "Sint Maarten",
        "Suriname", "Trinidad and Tobago", "Turks and Caicos Islands",
        "United States", "US Virgin Islands",
    },
    "OFC": {
        "American Samoa", "Cook Islands", "Fiji", "Kiribati", "New Caledonia",
        "New Zealand", "Papua New Guinea", "Samoa", "Solomon Islands",
        "Tahiti", "Tonga", "Tuvalu", "Vanuatu",
    },
    "OTHER": {
        "Abkhazia", "Artsakh", "Bonaire", "Catalonia", "Chagos Islands",
        "Corsica", "County of Nice", "Elba Island", "Ellan Vannin", "Frøya",
        "Greenland", "Isle of Man", "Jersey", "Kernow", "Matabeleland",
        "Northern Cyprus", "Parishes of Jersey", "Sardinia", "Somaliland",
        "South Ossetia", "Surrey", "Ticino", "Two Sicilies",
        "Western Armenia", "West Papua", "Ynys Môn", "Yorkshire",
    },
}


def infer_confederation(team):
    for confederation, teams in CONFEDERATION_TEAMS.items():
        if team in teams:
            return confederation
    return FALLBACK_CONFEDERATION


def build_confederations(all_teams):
    return pd.DataFrame(
        {
            "team": sorted(all_teams),
            "confederation": [
                infer_confederation(team)
                for team in sorted(all_teams)
            ],
        }
    )


def score_match(team, match):
    is_home = match["home"] == team
    goals_for = match["home_goals"] if is_home else match["away_goals"]
    goals_against = match["away_goals"] if is_home else match["home_goals"]
    opponent = match["away"] if is_home else match["home"]

    if goals_for > goals_against:
        points = 3
    elif goals_for == goals_against:
        points = 1
    else:
        points = 0

    return opponent, goals_for, goals_against, points


def recency_weight(match_date, max_date):
    years = max((max_date - match_date).days, 0) / 365
    return 1 / (1 + years)


def estimate_missing_team_elo(team, team_matches, known_elos, confederation, max_date):
    matches = len(team_matches)
    points = 0
    goals_for = 0
    goals_against = 0
    recency_total = 0
    recency_weight_total = 0
    known_opponent_matches = 0
    known_opponent_elo_total = 0
    known_points_total = 0

    for _, match in team_matches.iterrows():
        opponent, gf, ga, match_points = score_match(team, match)
        weight = recency_weight(match["date"], max_date)

        points += match_points
        goals_for += gf
        goals_against += ga
        recency_total += match_points * weight
        recency_weight_total += weight

        if opponent in known_elos:
            known_opponent_matches += 1
            known_opponent_elo_total += known_elos[opponent]
            known_points_total += match_points

    ppg = points / matches
    gd_per_match = (goals_for - goals_against) / matches
    gf_per_match = goals_for / matches
    ga_per_match = goals_against / matches
    recent_ppg = recency_total / recency_weight_total if recency_weight_total else ppg
    known_ppg = (
        known_points_total / known_opponent_matches
        if known_opponent_matches
        else ppg
    )
    known_avg_elo = (
        known_opponent_elo_total / known_opponent_matches
        if known_opponent_matches
        else CONFEDERATION_BASE_ELO.get(confederation, 1400)
    )

    base = CONFEDERATION_BASE_ELO.get(confederation, CONFEDERATION_BASE_ELO[FALLBACK_CONFEDERATION])

    elo = base
    elo += (ppg - 1.25) * 135
    elo += gd_per_match * 90
    elo += (gf_per_match - ga_per_match) * 25
    elo += (recent_ppg - ppg) * 55

    if known_opponent_matches:
        known_share = min(known_opponent_matches / 30, 1)
        known_signal = (known_avg_elo - 1500) * 0.35
        known_result = (known_ppg - 1.10) * 95
        elo += (known_signal + known_result) * known_share

    if matches < 10:
        confidence = "low"
        shrink = 0.45
    elif matches < 30:
        confidence = "medium"
        shrink = 0.70
    elif known_opponent_matches < 10:
        confidence = "medium"
        shrink = 0.80
    else:
        confidence = "high"
        shrink = 0.90

    elo = base + (elo - base) * shrink

    lower_bound = {
        "UEFA": 1150,
        "CONMEBOL": 1250,
        "CAF": 1100,
        "AFC": 1050,
        "CONCACAF": 1050,
        "OFC": 950,
        "OTHER": 900,
        FALLBACK_CONFEDERATION: 1000,
    }.get(confederation, 1000)

    upper_bound = {
        "UEFA": 1950,
        "CONMEBOL": 1900,
        "CAF": 1850,
        "AFC": 1800,
        "CONCACAF": 1750,
        "OFC": 1600,
        "OTHER": 1350,
        FALLBACK_CONFEDERATION: 1450,
    }.get(confederation, 1650)

    estimated_elo = int(round(max(lower_bound, min(elo, upper_bound))))

    return {
        "team": team,
        "estimated_elo": estimated_elo,
        "matches": matches,
        "ppg": round(ppg, 3),
        "gd_per_match": round(gd_per_match, 3),
        "known_opponent_matches": known_opponent_matches,
        "confidence": confidence,
    }


def main():
    matches = pd.read_csv(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = normalize_team_columns(matches, ["home", "away"], date_column="date")

    teams = pd.read_csv(TEAMS_PATH)
    known_elos = dict(zip(teams["team"], teams["elo"]))

    all_teams = set(matches["home"]).union(matches["away"]).union(teams["team"])
    confederations = build_confederations(all_teams)
    CONFEDERATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    confederations.to_csv(CONFEDERATIONS_PATH, index=False)

    confederation_by_team = dict(
        zip(confederations["team"], confederations["confederation"])
    )

    missing_teams = sorted(all_teams - set(known_elos))
    max_date = matches["date"].max()

    estimates = []
    for team in missing_teams:
        team_matches = matches[
            (matches["home"] == team) |
            (matches["away"] == team)
        ].sort_values("date")

        if team_matches.empty:
            continue

        confederation = confederation_by_team.get(team, FALLBACK_CONFEDERATION)
        estimates.append(
            estimate_missing_team_elo(
                team,
                team_matches,
                known_elos,
                confederation,
                max_date,
            )
        )

    estimates_df = pd.DataFrame(estimates).sort_values(
        ["estimated_elo", "matches"],
        ascending=[False, False],
    )
    estimates_df.to_csv(ESTIMATES_PATH, index=False)

    print(f"Wrote {CONFEDERATIONS_PATH.relative_to(ROOT)}")
    print(f"Wrote {ESTIMATES_PATH.relative_to(ROOT)}")
    print(f"Estimated teams: {len(estimates_df)}")


if __name__ == "__main__":
    main()
