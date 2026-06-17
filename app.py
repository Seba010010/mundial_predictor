# app.py

import streamlit as st
import pandas as pd
from src.predictor import predict_match
from src.match_analysis import build_advanced_match_analysis
from src.squad_stats import get_squad_profile
from src.ollama_report import generate_report
from src.worldcup_simulator import (
    simulate_groups,
    simulate_round_of_32,
    simulate_knockout_round,
    simulate_many_world_cups,
    simulate_many_world_cups_with_stages,
    simulate_team_world_cup_path
)
import random

st.set_page_config(
    page_title="Predictor Mundial 2026",
    layout="wide"
)

st.title("Predictor Mundial 2026")

st.subheader("Predicción de partido")

teams = pd.read_csv("data/teams.csv")["team"].tolist()

col_select_1, col_select_2 = st.columns(2)

with col_select_1:
    home_team = st.selectbox("Equipo local", teams)

with col_select_2:
    away_team = st.selectbox("Equipo visitante", teams)

if st.button("Calcular probabilidades"):
    if home_team == away_team:
        st.error("Selecciona equipos distintos.")
    else:
        st.session_state["result"] = predict_match(home_team, away_team)

if "result" in st.session_state:
    result = st.session_state["result"]
    simulation = result["simulation"]

    st.divider()

    st.header(f"{result['home_team']} vs {result['away_team']}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        f"Gana {result['home_team']}",
        f"{result['home_win']}%"
    )

    col2.metric(
        "Empate",
        f"{result['draw']}%"
    )

    col3.metric(
        f"Gana {result['away_team']}",
        f"{result['away_win']}%"
    )

    st.subheader("Comparativa de equipos")
    st.caption(
        "Estadísticas calculadas utilizando los últimos 20 partidos de cada selección."
)

    comparison_df = pd.DataFrame({
        
        "Métrica": [
            "Elo",
            "Goles esperados",
            "Índice ataque",
            "Índice defensa",
            "Forma últimos 10",
            "Victorias últimos 20",
            "Empates últimos 20",
            "Derrotas últimos 20",
            "Goles anotados últimos 20",
            "Goles recibidos últimos 20"
        ],
        result["home_team"]: [
            result["home_elo"],
            result["home_lambda"],
            result["home_attack"],
            result["home_defense"],
            " ".join(result["home_form"]),
            result["home_stats"]["wins"],
            result["home_stats"]["draws"],
            result["home_stats"]["losses"],
            result["home_stats"]["goals_for"],
            result["home_stats"]["goals_against"]
        ],
        result["away_team"]: [
            result["away_elo"],
            result["away_lambda"],
            result["away_attack"],
            result["away_defense"],
            " ".join(result["away_form"]),
            result["away_stats"]["wins"],
            result["away_stats"]["draws"],
            result["away_stats"]["losses"],
            result["away_stats"]["goals_for"],
            result["away_stats"]["goals_against"]
        ]
    })

    st.dataframe(
        comparison_df,
            width="stretch",
        hide_index=True
    )


    st.subheader("Historial entre ambos equipos")

    h2h = result["head_to_head"]

    if h2h["total_matches"] == 0:
        st.info("No hay enfrentamientos directos registrados entre estos equipos.")
    else:
        h2h_col1, h2h_col2, h2h_col3 = st.columns(3)

        h2h_col1.metric(
            f"Victorias {result['home_team']}",
            h2h["team_a_wins"]
        )

        h2h_col2.metric(
            "Empates",
            h2h["draws"]
        )

        h2h_col3.metric(
            f"Victorias {result['away_team']}",
            h2h["team_b_wins"]
        )

        st.write("Últimos enfrentamientos directos:")

        h2h_df = pd.DataFrame(h2h["recent_matches"])

        st.dataframe(
            h2h_df,
            width="stretch",
            hide_index=True
        )
    st.caption(f"Enfrentamientos encontrados en la base de datos: {h2h['total_matches']}")

    st.subheader("Probabilidades de goles")

    goal_col1, goal_col2, goal_col3 = st.columns(3)

    goal_col1.metric(
        "Más de 1.5 goles",
        f"{result['over_15']}%"
    )

    goal_col2.metric(
        "Más de 2.5 goles",
        f"{result['over_25']}%"
    )

    goal_col3.metric(
        "Ambos anotan",
        f"{result['both_score']}%"
    )

    st.subheader("Análisis avanzado de partido")

    advanced = build_advanced_match_analysis(result)

    distribution_col1, distribution_col2 = st.columns(2)

    with distribution_col1:
        st.write(f"Distribución de goles - {result['home_team']}")
        home_goals_df = pd.DataFrame(advanced["home_goal_distribution"])
        st.bar_chart(home_goals_df.set_index("goals"))

    with distribution_col2:
        st.write(f"Distribución de goles - {result['away_team']}")
        away_goals_df = pd.DataFrame(advanced["away_goal_distribution"])
        st.bar_chart(away_goals_df.set_index("goals"))

    markets_df = pd.DataFrame(advanced["markets"])
    markets_df["probability"] = markets_df["probability"].map(
        lambda value: f"{value:.2f}%"
    )
    st.write("Probabilidades derivadas")
    st.dataframe(markets_df, width="stretch", hide_index=True)

    scorer_col1, scorer_col2 = st.columns(2)

    with scorer_col1:
        st.write(f"Goleadores probables - {result['home_team']}")
        home_scorers_df = pd.DataFrame(advanced["home_scorers"])

        if home_scorers_df.empty:
            st.info("No hay datos suficientes de goleadores para esta selección.")
        else:
            home_scorers_df = home_scorers_df.rename(
                columns={
                    "player": "Jugador",
                    "historical_goals": "Goles históricos",
                    "penalty_goals": "Penales",
                    "team_goal_share_pct": "% goles del equipo",
                    "anytime_goal_pct": "% anota",
                    "in_squad": "Convocado",
                }
            )
            st.dataframe(home_scorers_df, width="stretch", hide_index=True)

    with scorer_col2:
        st.write(f"Goleadores probables - {result['away_team']}")
        away_scorers_df = pd.DataFrame(advanced["away_scorers"])

        if away_scorers_df.empty:
            st.info("No hay datos suficientes de goleadores para esta selección.")
        else:
            away_scorers_df = away_scorers_df.rename(
                columns={
                    "player": "Jugador",
                    "historical_goals": "Goles históricos",
                    "penalty_goals": "Penales",
                    "team_goal_share_pct": "% goles del equipo",
                    "anytime_goal_pct": "% anota",
                    "in_squad": "Convocado",
                }
            )
            st.dataframe(away_scorers_df, width="stretch", hide_index=True)

    st.caption(
        "Las probabilidades de goleador son una aproximación basada en la "
        "cuota histórica de goles desde 2018. Cuando existe plantilla cargada, "
        "solo se muestran jugadores convocados; todavía no considera "
        "titularidad ni minutos esperados."
    )

    st.subheader("Resultados exactos más probables")

    scores_df = pd.DataFrame(result["most_likely_scores"])
    scores_df["probability"] = scores_df["probability"].map(
        lambda x: f"{x:.2f}%"
    )

    st.table(scores_df)

    st.subheader("Simulación de 10.000 partidos")

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    sim_col1.metric(
        f"Victorias {result['home_team']}",
        simulation["home_wins"]
    )

    sim_col2.metric(
        "Empates",
        simulation["draws"]
    )

    sim_col3.metric(
        f"Victorias {result['away_team']}",
        simulation["away_wins"]
    )

    simulation_chart = pd.DataFrame({
        "Resultado": [
            result["home_team"],
            "Empate",
            result["away_team"]
        ],
        "Probabilidad simulada": [
            simulation["home_win_pct"],
            simulation["draw_pct"],
            simulation["away_win_pct"]
        ]
    })

    st.bar_chart(simulation_chart.set_index("Resultado"))

    st.subheader("Marcadores más repetidos en simulación")

    sim_scores_df = pd.DataFrame(simulation["most_common_scores"])
    sim_scores_df["percentage"] = sim_scores_df["percentage"].map(
        lambda x: f"{x:.2f}%"
    )

    st.table(sim_scores_df)

st.divider()

st.header("Simulación Mundial 2026")

st.subheader("Análisis por selección")

worldcup_teams = pd.read_csv("data/groups.csv")["team"].tolist()

team_select_col, team_action_col = st.columns([2, 1])

with team_select_col:
    selected_team_path = st.selectbox(
        "Selección",
        worldcup_teams,
        key="team_path_select"
    )

with team_action_col:
    st.write("")
    st.write("")
    simulate_team_path = st.button(
        "Simular rendimiento de selección",
        type="primary",
        use_container_width=True
    )

if simulate_team_path:
    st.session_state["team_worldcup_path"] = simulate_team_world_cup_path(
        selected_team_path,
        n=10000
    )

squad_profile = get_squad_profile(selected_team_path)

if squad_profile:
    st.subheader("Perfil de plantilla")

    squad_col1, squad_col2, squad_col3, squad_col4 = st.columns(4)
    squad_col1.metric("Jugadores", squad_profile["players"])
    squad_col2.metric("Edad promedio", squad_profile["average_age"])
    squad_col3.metric("Altura promedio", f"{squad_profile['average_height_cm']} cm")
    squad_col4.metric(
        "Entrenador",
        squad_profile["coach_name"] or "Sin dato"
    )

    squad_detail_col1, squad_detail_col2 = st.columns(2)

    with squad_detail_col1:
        position_counts_df = pd.DataFrame(squad_profile["position_counts"])
        position_counts_df = position_counts_df.rename(columns={
            "position": "Posición",
            "players": "Jugadores"
        })
        st.write("Distribución por posición")
        st.bar_chart(position_counts_df.set_index("Posición")["Jugadores"])

    with squad_detail_col2:
        club_country_df = pd.DataFrame(
            squad_profile["club_country_counts"]
        ).rename(columns={
            "club_country": "País del club",
            "players": "Jugadores"
        })
        st.write("Países de clubes más representados")
        st.dataframe(
            club_country_df.head(10),
            width="stretch",
            hide_index=True
        )

    position_profile_df = pd.DataFrame(
        squad_profile["position_profile"]
    ).rename(columns={
        "position": "Posición",
        "players": "Jugadores",
        "avg_age": "Edad promedio",
        "avg_height_cm": "Altura promedio"
    })
    profile_tab1, profile_tab2, profile_tab3, profile_tab4, profile_tab5 = st.tabs([
        "Plantilla",
        "Perfil por línea",
        "Más jóvenes",
        "Más veteranos",
        "Más altos",
    ])

    with profile_tab1:
        squad_df = pd.DataFrame(squad_profile["squad"]).rename(columns={
            "number": "#",
            "position": "Posición",
            "player_name": "Jugador",
            "age": "Edad",
            "club": "Club",
            "club_country": "País club",
            "height_cm": "Altura cm",
        })
        st.dataframe(squad_df, width="stretch", hide_index=True)

    with profile_tab2:
        st.dataframe(position_profile_df, width="stretch", hide_index=True)

    with profile_tab3:
        youngest_df = pd.DataFrame(
            squad_profile["youngest_players"]
        ).rename(columns={
            "player_name": "Jugador",
            "position": "Posición",
            "age": "Edad",
            "club": "Club",
        })
        st.dataframe(youngest_df, width="stretch", hide_index=True)

    with profile_tab4:
        oldest_df = pd.DataFrame(
            squad_profile["oldest_players"]
        ).rename(columns={
            "player_name": "Jugador",
            "position": "Posición",
            "age": "Edad",
            "club": "Club",
        })
        st.dataframe(oldest_df, width="stretch", hide_index=True)

    with profile_tab5:
        tallest_df = pd.DataFrame(
            squad_profile["tallest_players"]
        ).rename(columns={
            "player_name": "Jugador",
            "position": "Posición",
            "height_cm": "Altura cm",
            "club": "Club",
        })
        st.dataframe(tallest_df, width="stretch", hide_index=True)
else:
    st.info("No hay plantilla cargada para esta selección.")

if "team_worldcup_path" in st.session_state:
    team_path = st.session_state["team_worldcup_path"]
    required_team_path_keys = {
        "group_winner_pct",
        "group_runner_up_pct",
        "best_third_pct",
        "group_position_distribution",
        "group_rivals",
        "round_opponents",
        "eliminators",
        "most_likely_path",
        "summary",
        "average_tournament_stats",
        "average_match_results",
    }

    if not required_team_path_keys.issubset(team_path):
        st.warning(
            "El análisis guardado fue generado con una versión anterior. "
            "Vuelve a presionar 'Simular rendimiento de selección' para "
            "actualizarlo."
        )
        st.stop()

    st.write(
        f"Resultado de {team_path['simulations']} Mundiales simulados "
        f"para {team_path['team']}."
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    metric_col1.metric("Campeón", f"{team_path['champion_pct']}%")
    metric_col2.metric("Final", f"{team_path['final_pct']}%")
    metric_col3.metric("Semifinal", f"{team_path['semifinals_pct']}%")
    metric_col4.metric(
        "Eliminación más probable",
        team_path["most_likely_elimination"]
    )

    group_col1, group_col2, group_col3, group_col4 = st.columns(4)

    group_col1.metric(
        "Posición media en grupo",
        team_path["average_group_position"]
    )
    group_col2.metric(
        "Puntos medios en grupo",
        team_path["average_group_points"]
    )
    group_col3.metric(
        "GF medio en grupo",
        team_path["average_group_goals_for"]
    )
    group_col4.metric(
        "GC medio en grupo",
        team_path["average_group_goals_against"]
    )

    average_results = team_path["average_tournament_stats"]
    st.write("Resultados promedio en el Mundial simulado")

    average_results_col1, average_results_col2, average_results_col3 = st.columns(3)
    average_results_col1.metric("Partidos jugados", average_results["matches"])
    average_results_col2.metric("Victorias", average_results["wins"])
    average_results_col3.metric("Empates", average_results["draws"])

    average_results_df = pd.DataFrame([
        {"Métrica": "Partidos jugados", "Promedio": average_results["matches"]},
        {"Métrica": "Victorias", "Promedio": average_results["wins"]},
        {"Métrica": "Empates", "Promedio": average_results["draws"]},
        {"Métrica": "Derrotas", "Promedio": average_results["losses"]},
        {"Métrica": "Puntos", "Promedio": average_results["points"]},
        {"Métrica": "Goles a favor", "Promedio": average_results["goals_for"]},
        {"Métrica": "Goles en contra", "Promedio": average_results["goals_against"]},
        {"Métrica": "Diferencia de gol", "Promedio": average_results["goal_difference"]},
        {"Métrica": "Tandas de penales", "Promedio": average_results["penalty_shootouts"]},
        {"Métrica": "Tandas ganadas", "Promedio": average_results["penalty_shootout_wins"]},
        {"Métrica": "Tandas perdidas", "Promedio": average_results["penalty_shootout_losses"]},
    ])
    st.dataframe(
        average_results_df,
        width="stretch",
        hide_index=True
    )
    st.caption(
        "Los empates en eliminatorias cuentan como empate si el marcador termina "
        "igualado; las tandas de penales se muestran aparte."
    )

    average_match_results_df = pd.DataFrame(
        team_path["average_match_results"]
    ).rename(columns={
        "stage": "Fase",
        "opponent": "Rival",
        "matches": "Partidos",
        "appearance_pct": "Frecuencia del cruce %",
        "avg_goals_for": "GF promedio",
        "avg_goals_against": "GC promedio",
        "avg_goal_difference": "DG promedio",
        "win_pct": "Victoria %",
        "draw_pct": "Empate %",
        "loss_pct": "Derrota %",
    })

    st.write("Promedio de goles por partido y rival")
    st.dataframe(
        average_match_results_df,
        width="stretch",
        hide_index=True
    )

    group_result_col1, group_result_col2, group_result_col3 = st.columns(3)

    group_result_col1.metric(
        "Gana el grupo",
        f"{team_path['group_winner_pct']}%"
    )
    group_result_col2.metric(
        "Clasifica como segundo",
        f"{team_path['group_runner_up_pct']}%"
    )
    group_result_col3.metric(
        "Clasifica como mejor tercero",
        f"{team_path['best_third_pct']}%"
    )

    group_position_df = pd.DataFrame(
        team_path["group_position_distribution"]
    ).rename(columns={
        "position": "Posición",
        "percentage": "Probabilidad %"
    })

    st.write("Rendimiento en fase de grupos")
    st.dataframe(
        group_position_df,
        width="stretch",
        hide_index=True
    )
    st.bar_chart(group_position_df.set_index("Posición")["Probabilidad %"])

    group_rivals_df = pd.DataFrame(team_path["group_rivals"])

    if not group_rivals_df.empty:
        group_rivals_df = group_rivals_df.rename(columns={
            "opponent": "Rival",
            "count": "Veces que termina arriba",
            "percentage": "Probabilidad %"
        })
        st.write("Probabilidad de terminar por encima de cada rival del grupo")
        st.dataframe(
            group_rivals_df,
            width="stretch",
            hide_index=True
        )

    phase_df = pd.DataFrame([
        {"Fase": "Dieciseisavos", "Probabilidad %": team_path["round32_pct"]},
        {"Fase": "Octavos", "Probabilidad %": team_path["round16_pct"]},
        {"Fase": "Cuartos", "Probabilidad %": team_path["quarterfinals_pct"]},
        {"Fase": "Semifinales", "Probabilidad %": team_path["semifinals_pct"]},
        {"Fase": "Final", "Probabilidad %": team_path["final_pct"]},
        {"Fase": "Campeón", "Probabilidad %": team_path["champion_pct"]},
    ])

    st.write("Probabilidades por fase")
    st.dataframe(
        phase_df,
        width="stretch",
        hide_index=True
    )

    elimination_df = pd.DataFrame([
        {
            "Fase": "Fase de grupos",
            "Probabilidad %": team_path["group_stage_elimination_pct"]
        },
        {
            "Fase": "Dieciseisavos",
            "Probabilidad %": team_path["round32_elimination_pct"]
        },
        {
            "Fase": "Octavos",
            "Probabilidad %": team_path["round16_elimination_pct"]
        },
        {
            "Fase": "Cuartos",
            "Probabilidad %": team_path["quarterfinals_elimination_pct"]
        },
        {
            "Fase": "Semifinales",
            "Probabilidad %": team_path["semifinals_elimination_pct"]
        },
        {
            "Fase": "Subcampeón",
            "Probabilidad %": team_path["runner_up_pct"]
        },
        {
            "Fase": "Campeón",
            "Probabilidad %": team_path["champion_pct"]
        },
    ])

    st.write("Distribución de eliminación")
    st.dataframe(
        elimination_df,
        width="stretch",
        hide_index=True
    )
    st.bar_chart(elimination_df.set_index("Fase")["Probabilidad %"])

    st.write("Rivales más frecuentes por ronda")

    round_tabs = st.tabs([
        "Dieciseisavos",
        "Octavos",
        "Cuartos",
        "Semifinales",
        "Final"
    ])

    for tab, round_name in zip(round_tabs, team_path["round_opponents"]):
        with tab:
            opponents_df = pd.DataFrame(
                team_path["round_opponents"][round_name]
            )

            if opponents_df.empty:
                st.info("No hay rivales registrados para esta ronda.")
            else:
                opponents_df = opponents_df.rename(columns={
                    "opponent": "Rival",
                    "count": "Frecuencia",
                    "percentage": "Porcentaje %"
                })
                st.dataframe(
                    opponents_df,
                    width="stretch",
                    hide_index=True
                )

    eliminators_df = pd.DataFrame(team_path["eliminators"])

    st.write("Rivales que más eliminan a la selección")

    if eliminators_df.empty:
        st.info("No hay eliminaciones registradas.")
    else:
        eliminators_df = eliminators_df.rename(columns={
            "opponent": "Rival",
            "count": "Veces que elimina",
            "percentage": "Porcentaje de eliminaciones %"
        })
        st.dataframe(
            eliminators_df,
            width="stretch",
            hide_index=True
        )

    st.write("Camino más probable")

    path_rows = [
        {
            "Etapa": "Grupo",
            "Escenario más probable": (
                max(
                    team_path["most_likely_path"]["Grupo"],
                    key=lambda item: item["percentage"]
                )["opponent"]
                if team_path["most_likely_path"]["Grupo"]
                else "Sin rival"
            ),
            "Probabilidad %": (
                max(
                    team_path["most_likely_path"]["Grupo"],
                    key=lambda item: item["percentage"]
                )["percentage"]
                if team_path["most_likely_path"]["Grupo"]
                else 0
            )
        }
    ]

    for round_name in [
        "Dieciseisavos",
        "Octavos",
        "Cuartos",
        "Semifinales",
        "Final"
    ]:
        path_item = team_path["most_likely_path"][round_name]
        path_rows.append({
            "Etapa": round_name,
            "Escenario más probable": path_item["opponent"] or "Sin rival frecuente",
            "Probabilidad %": path_item["percentage"]
        })

    st.dataframe(
        pd.DataFrame(path_rows),
        width="stretch",
        hide_index=True
    )

    st.info(team_path["summary"])

st.divider()

st.subheader("Análisis global por Monte Carlo")

worldcup_simulations = st.selectbox(
    "Cantidad de simulaciones",
    [1000, 3000, 10000],
    index=1,
    key="worldcup_simulations_count"
)

if st.button(f"Simular {worldcup_simulations:,} Mundiales"):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_worldcup_progress(current, total):
        percentage = current / total
        progress_bar.progress(percentage)
        progress_text.write(
            f"Simulando Mundiales: {current:,}/{total:,}"
        )

    st.session_state["many_worldcups"] = simulate_many_world_cups(
        worldcup_simulations,
        progress_callback=update_worldcup_progress
    )
    progress_text.write("Simulación completada.")

if "many_worldcups" in st.session_state:

    results_df = pd.DataFrame(st.session_state["many_worldcups"])

    st.dataframe(
        results_df,
        width="stretch",
        hide_index=True
    )

    chart_df = results_df.set_index("team")
    st.bar_chart(chart_df["probability"])

if st.button(f"Simular fases de {worldcup_simulations:,} Mundiales"):
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_stage_progress(current, total):
        percentage = current / total
        progress_bar.progress(percentage)
        progress_text.write(
            f"Simulando fases del Mundial: {current:,}/{total:,}"
        )

    st.session_state["stage_probabilities"] = (
        simulate_many_world_cups_with_stages(
            worldcup_simulations,
            progress_callback=update_stage_progress
        )
    )
    progress_text.write("Simulación de fases completada.")

if "stage_probabilities" in st.session_state:

    stages_df = pd.DataFrame(
        st.session_state["stage_probabilities"]
    )

    stages_df = stages_df.rename(columns={
        "team": "Equipo",
        "round32_pct": "Dieciseisavos %",
        "round16_pct": "Octavos %",
        "quarterfinals_pct": "Cuartos %",
        "semifinals_pct": "Semifinal %",
        "final_pct": "Final %",
        "champion_pct": "Campeón %"
    })

    st.dataframe(
        stages_df,
        width="stretch",
        hide_index=True
    )

st.divider()

st.subheader("Mundial de ejemplo")
st.caption(
    "Esta sección genera una sola edición simulada. Es útil para ver una "
    "llave posible, no para inferir probabilidades."
)

if st.button("Simular un Mundial de ejemplo"):
    st.session_state["worldcup_results"] = simulate_groups()
    st.session_state.pop("single_worldcup_knockout", None)

if "worldcup_results" in st.session_state:

    group_results = st.session_state["worldcup_results"]

    for group_name, data in group_results.items():

        st.subheader(f"Grupo {group_name}")

        st.write("Partidos simulados")
        st.dataframe(
            pd.DataFrame(data["matches"]),
            width="stretch",
            hide_index=True
        )

        st.write("Tabla del grupo")
        st.dataframe(
            pd.DataFrame(data["standings"]),
            width="stretch",
            hide_index=True
        )

    st.header("Llave eliminatoria")
    st.divider()

    if "single_worldcup_knockout" not in st.session_state:

        round32 = simulate_round_of_32(group_results)


        round32_winners = [
            match["winner"]
            for match in round32
        ]

        round16 = simulate_knockout_round(round32_winners)

        round16_winners = [
            match["winner"]
            for match in round16
        ]

        quarterfinals = simulate_knockout_round(round16_winners)

        quarter_winners = [
            match["winner"]
            for match in quarterfinals
        ]

        semifinals = simulate_knockout_round(quarter_winners)

        semifinal_winners = [
            match["winner"]
            for match in semifinals
        ]

        final = simulate_knockout_round(semifinal_winners)

        champion = final[0]["winner"]

        st.session_state["single_worldcup_knockout"] = {
            "round32": round32,
            "round16": round16,
            "quarterfinals": quarterfinals,
            "semifinals": semifinals,
            "final": final,
            "champion": champion
        }

    knockout = st.session_state["single_worldcup_knockout"]

    round32 = knockout["round32"]
    round16 = knockout["round16"]
    quarterfinals = knockout["quarterfinals"]
    semifinals = knockout["semifinals"]
    final = knockout["final"]
    champion = knockout["champion"]


    st.header("Dieciseisavos de final")

    round32_df = pd.DataFrame(round32)

    round32_df = round32_df[
        ["team_a", "team_b", "score", "winner", "method"]
    ].rename(columns={
        "team_a": "Equipo 1",
        "team_b": "Equipo 2",
        "score": "Marcador",
        "winner": "Ganador",
        "method": "Definición"
    })

    st.dataframe(
        round32_df,
        width="stretch",
        hide_index=True
    )

    st.header("Octavos de final")

    round16_df = pd.DataFrame(round16)
    round16_df = round16_df[
        ["team_a", "team_b", "score", "winner", "method"]
    ].rename(columns={
        "team_a": "Equipo 1",
        "team_b": "Equipo 2",
        "score": "Marcador",
        "winner": "Ganador",
        "method": "Definición"
    })

    st.dataframe(
        round16_df,
        width="stretch",
        hide_index=True
    )

    st.header("Cuartos de final")

    quarterfinals_df = pd.DataFrame(quarterfinals)
    quarterfinals_df = quarterfinals_df[
        ["team_a", "team_b", "score", "winner", "method"]
    ].rename(columns={
        "team_a": "Equipo 1",
        "team_b": "Equipo 2",
        "score": "Marcador",
        "winner": "Ganador",
        "method": "Definición"
    })

    st.dataframe(
        quarterfinals_df,
        width="stretch",
        hide_index=True
    )

    st.header("Semifinales")

    semifinals_df = pd.DataFrame(semifinals)
    semifinals_df = semifinals_df[
        ["team_a", "team_b", "score", "winner", "method"]
    ].rename(columns={
        "team_a": "Equipo 1",
        "team_b": "Equipo 2",
        "score": "Marcador",
        "winner": "Ganador",
        "method": "Definición"
    })

    st.dataframe(
        semifinals_df,
        width="stretch",
        hide_index=True
    )

    st.header("Final")

    final_df = pd.DataFrame(final)
    final_df = final_df[
        ["team_a", "team_b", "score", "winner", "method"]
    ].rename(columns={
        "team_a": "Equipo 1",
        "team_b": "Equipo 2",
        "score": "Marcador",
        "winner": "Ganador",
        "method": "Definición"
    })

    st.dataframe(
        final_df,
        width="stretch",
        hide_index=True
    )

    st.header("🏆 Campeón simulado")
    st.success(champion)
