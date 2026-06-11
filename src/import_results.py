import pandas as pd

results = pd.read_csv("data/results.csv")

# Excluir partidos futuros o sin marcador
results = results.dropna(
    subset=["home_score", "away_score"]
)

matches = results.rename(
    columns={
        "home_team": "home",
        "away_team": "away",
        "home_score": "home_goals",
        "away_score": "away_goals"
    }
)

matches = matches[
    [
        "date",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "tournament"
    ]
]

matches["date"] = pd.to_datetime(matches["date"])
matches = matches.sort_values("date")

# Para el modelo actual conviene usar partidos relativamente recientes
matches = matches[matches["date"] >= "2018-01-01"]

matches.to_csv(
    "data/matches_history.csv",
    index=False
)

print("Archivo data/matches_history.csv generado correctamente.")
print(f"Partidos usados: {len(matches)}")
print(matches.tail(10))