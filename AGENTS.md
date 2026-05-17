# AGENTS.md

## Project overview
- Python AI Ecommerce customer intelligence platform.
- Source package root is `src`, and most runtime imports use `from src...`.
- FastAPI backend is defined in `api/app.py`.
- Streamlit user interface is defined in `streamlit_app/home.py` with pages under `streamlit_app/pages/`.
- Model training and inference pipelines live in `pipelines/`.

## Important conventions
- Use `python.analysis.extraPaths: ["./src"]` so imports from `src` resolve correctly in editors and language servers.
- The repository currently has no populated dependency manifest in `requirements.txt`; verify runtime dependencies before assuming available packages.
- `main.py` is not the main API entrypoint; the FastAPI app is served from `api/app.py`.

## Typical run commands
- API: `python -m uvicorn api.app:app --reload`
- Streamlit app: `streamlit run streamlit_app/home.py`

## Key directories
- `src/`: main Python package
- `src/config/`: configuration and constants
- `src/ml/`: classic ML tasks like churn, CLV, delay prediction, recommendations
- `src/dl/`: deep learning models such as LightGCN and LSTM forecasting
- `src/nlp/`: NLP pipelines and prediction components
- `src/forecasting/`: time-series forecasting code
- `src/utils/`: utility helpers
- `artifacts/`, `data/`, `notebooks/`: persisted outputs, datasets, and analysis notebooks

## Agent guidance
- Preserve the existing `src` package structure and import style.
- Do not add or assume dependencies unless they are clearly required by existing code.
- Keep changes minimal and aligned with the existing FastAPI/Streamlit service organization.
