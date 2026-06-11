# src/ollama_report.py

import ollama

def generate_report(result):
    prompt = f"""
    Eres un analista deportivo especializado en modelos probabilísticos de fútbol.

    Debes interpretar los datos calculados por un modelo matemático.
    No prometas resultados seguros.
    No digas que esto es una recomendación de apuesta.
    No repitas todos los números en forma mecánica.

    Partido:
    {result['home_team']} vs {result['away_team']}

    Probabilidades principales:
    - Gana {result['home_team']}: {result['home_win']}%
    - Empate: {result['draw']}%
    - Gana {result['away_team']}: {result['away_win']}%

    Goles:
    - Más de 1.5 goles: {result['over_15']}%
    - Más de 2.5 goles: {result['over_25']}%
    - Ambos anotan: {result['both_score']}%

    Datos técnicos:
    - Elo {result['home_team']}: {result['home_elo']}
    - Elo {result['away_team']}: {result['away_elo']}
    - Goles esperados {result['home_team']}: {result['home_lambda']}
    - Goles esperados {result['away_team']}: {result['away_lambda']}
    - Ataque {result['home_team']}: {result['home_attack']}
    - Defensa {result['home_team']}: {result['home_defense']}
    - Ataque {result['away_team']}: {result['away_attack']}
    - Defensa {result['away_team']}: {result['away_defense']}

    Marcadores más probables:
    {result['most_likely_scores']}

    Genera un informe en español con esta estructura:

    ### Lectura general
    Explica quién llega con ventaja y qué tan cerrado parece el partido.

    ### Escenario de goles
    Explica si el modelo espera partido abierto, cerrado o intermedio.

    ### Marcadores probables
    Interpreta los marcadores más probables sin repetir toda la tabla.

    ### Riesgos del modelo
    Menciona limitaciones: muestra histórica pequeña, datos incompletos, lesiones, contexto del partido y alineaciones.

    ### Conclusión
    Cierra con una lectura prudente en 2 o 3 líneas.
    """
    response = ollama.chat(
        model="qwen3-coder:30b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]