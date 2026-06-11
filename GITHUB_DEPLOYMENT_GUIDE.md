# Despliegue en Streamlit Cloud

Esta app es un dashboard Streamlit con backend Python. No es adecuada para
GitHub Pages ni para Vercel en su forma actual.

## Archivos principales

- `app.py`: entrada principal de Streamlit.
- `requirements.txt`: dependencias para instalar en Streamlit Cloud.
- `data/`: CSV necesarios para el modelo y la app.
- `src/`: lógica del predictor, simulador y análisis.

## Antes de subir a GitHub

Confirma que no se suban:

- `venv/`
- `__pycache__/`
- `.env`
- `.streamlit/secrets.toml`
- `data/sportmonks/raw/`
- `data/squads_pdf/*.pdf`

El archivo `.gitignore` ya excluye esos elementos.

## Primer push a GitHub

Desde la carpeta del proyecto:

```powershell
git init
git add .
git commit -m "Initial commit: World Cup predictor"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/NOMBRE_REPO.git
git push -u origin main
```

Si el repo ya fue inicializado localmente, omite `git init`.

## Desplegar en Streamlit Cloud

1. Entra a Streamlit Cloud.
2. Crea una nueva app.
3. Conecta tu cuenta de GitHub.
4. Selecciona el repositorio.
5. Usa:
   - Branch: `main`
   - Main file path: `app.py`
6. Presiona Deploy.

## Notas

- Las simulaciones de 10.000 Mundiales pueden tardar. Para uso público conviene
  dejar `3000` como opción por defecto o advertir al usuario.
- El token de Sportmonks no debe subirse al repositorio. Si se usa en el futuro,
  debe configurarse como secreto en Streamlit Cloud.
- La integración local con Ollama no funcionará en Streamlit Cloud salvo que se
  cambie por una API externa disponible desde internet.
