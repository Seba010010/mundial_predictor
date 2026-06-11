import pandas as pd


def load_matches():
    matches = pd.read_csv("data/matches_history.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    return matches.sort_values("date")


def get_head_to_head(team_a, team_b, last_n=5):
    matches = load_matches()

    h2h_matches = matches[
        (
            (matches["home"] == team_a) &
            (matches["away"] == team_b)
        ) |
        (
            (matches["home"] == team_b) &
            (matches["away"] == team_a)
        )
    ].sort_values("date")

    team_a_wins = 0
    team_b_wins = 0
    draws = 0

    rows = []

    for _, match in h2h_matches.iterrows():
        home = match["home"]
        away = match["away"]
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])

        if home_goals == away_goals:
            draws += 1
            winner = "Empate"
        elif home_goals > away_goals:
            winner = home
        else:
            winner = away

        if winner == team_a:
            team_a_wins += 1
        elif winner == team_b:
            team_b_wins += 1

        rows.append({
            "Fecha": match["date"].strftime("%Y-%m-%d"),
            "Local": home,
            "Visitante": away,
            "Marcador": f"{home_goals}-{away_goals}",
            "Torneo": match["tournament"],
            "Ganador": winner,
        })

    recent_matches = rows[-last_n:][::-1]

    return {
        "total_matches": len(h2h_matches),
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "draws": draws,
        "recent_matches": recent_matches,
    }