# Predictor Mundial FIFA 2026

Dashboard estadistico en Streamlit para simular partidos y escenarios del Mundial FIFA 2026 usando Python, Pandas, Monte Carlo y distribucion de Poisson.

El proyecto combina resultados historicos, ELO, forma reciente, valor de mercado, ranking FIFA, plantillas convocadas y simulaciones de torneo completo para estimar probabilidades por partido, fase y campeon.

## Funcionalidades

- Prediccion de partidos individuales.
- Probabilidades 1X2: gana equipo A, empate, gana equipo B.
- Lambdas y goles esperados por equipo.
- Marcadores mas probables.
- Distribucion de goles por equipo.
- Over/under, ambos anotan y arco en cero.
- Probabilidad aproximada de goleadores convocados.
- Simulacion completa del Mundial 2026.
- Simulaciones Monte Carlo por fase.
- Probabilidad de campeon.
- Analisis por seleccion:
  - probabilidad de avanzar por fase
  - eliminacion mas probable
  - rendimiento promedio en grupo
  - promedio de partidos, goles, victorias, empates y derrotas
  - rivales mas frecuentes por ronda
  - rivales que mas eliminan a una seleccion
  - camino mas probable
- Perfil de plantilla:
  - edad promedio
  - altura promedio
  - distribucion por posicion
  - clubes/paises mas representados
  - jugadores mas jovenes, veteranos y altos

## Stack

- Python 3.11
- Streamlit
- Pandas
- NumPy / SciPy
- Monte Carlo
- Distribucion de Poisson

## Estructura principal

```text
app.py
data/
  groups.csv
  teams.csv
  matches_history.csv
  results.csv
  goalscorers.csv
  shootouts.csv
  former_names.csv
  team_elo_estimates.csv
  team_confederations.csv
  team_market_values.csv
  team_fifa_rankings.csv
  worldcup_squads.csv
src/
  predictor.py
  team_stats.py
  worldcup_simulator.py
  match_analysis.py
  squad_stats.py
  poisson.py
  simulator.py
scripts/
  backtest_model.py
  generate_team_elo_estimates.py
  analyze_goal_sources.py
  extract_worldcup_squads.py
```

## Modelo

El modelo base estima goles esperados para cada partido con:

- ataque y defensa historica por seleccion
- peso por torneo
- peso por recencia
- ajuste por fuerza del rival usando ELO
- ajuste ELO directo entre equipos
- valor de mercado de plantilla con impacto moderado
- ranking FIFA como senal secundaria

La probabilidad de marcadores se calcula con una distribucion de Poisson. Para torneos completos, el sistema ejecuta simulaciones Monte Carlo de fase de grupos y eliminatorias.

## Datos

El proyecto usa archivos CSV locales en `data/`.

Entre las fuentes/datasets procesados hay:

- resultados historicos internacionales
- goleadores historicos
- definiciones por penales
- nombres historicos de selecciones
- grupos del Mundial 2026
- ELO de selecciones
- ELO estimado para equipos sin ELO directo
- valores de mercado por seleccion
- ranking FIFA
- plantillas convocadas procesadas desde PDF

Los archivos crudos de APIs externas y PDFs fuente no son necesarios para ejecutar la app si los CSV procesados ya estan presentes.

## Instalacion local

Crear y activar un entorno virtual:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Ejecutar la app:

```powershell
streamlit run app.py
```

## Scripts utiles

Backtesting del modelo:

```powershell
.\venv\Scripts\python.exe scripts\backtest_model.py
```

Regenerar ELO estimado:

```powershell
.\venv\Scripts\python.exe scripts\generate_team_elo_estimates.py
```

Analizar fuentes de goles, penales y autogoles:

```powershell
.\venv\Scripts\python.exe scripts\analyze_goal_sources.py
```

Extraer plantillas desde PDF:

```powershell
.\venv\Scripts\python.exe scripts\extract_worldcup_squads.py
```

## Despliegue en Streamlit Cloud

Configuracion recomendada:

- Branch: `main`
- Main file path: `app.py`
- Python: `3.11`

No usar Python 3.14 por ahora, ya que algunas dependencias pueden no tener soporte estable.

## Limitaciones

Este proyecto es un modelo estadistico exploratorio. No garantiza resultados reales.

Limitaciones principales:

- No incorpora lesiones ni decisiones tacticas de ultimo minuto.
- Las probabilidades de goleador son aproximadas y dependen del historial disponible.
- No hay datos completos de tiros, tiros al arco, xG, posesion o corners.
- La matriz oficial FIFA de combinaciones de mejores terceros no esta implementada; se usa una version simplificada del Round of 32.
- Algunos datos historicos pueden tener diferencias de cobertura entre selecciones.

## Estado del proyecto

El modelo ya permite simular:

- partidos individuales
- fase de grupos
- rondas eliminatorias
- probabilidades por fase
- campeon del Mundial
- analisis especifico por seleccion

Futuras mejoras posibles:

- calibracion fina via backtesting
- regularizacion de defensas extremas
- integracion de datos avanzados de partido si se consigue una fuente confiable
- integracion con una API de IA para resumenes automaticos
- comparativas entre perfiles de plantilla y rendimiento simulado

## Aviso

Este proyecto es para analisis deportivo y simulacion estadistica. No constituye recomendacion de apuestas.
