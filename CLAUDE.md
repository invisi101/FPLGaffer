# FPL Points Predictor — Claude Code Notes

## Environment

- **Python**: Use `.venv/bin/python`, NOT system `python3` (system Python lacks project dependencies)
- **Run server**: `.venv/bin/python -m src.app` (serves on `http://127.0.0.1:9876`)
- **Port 9876**: Often has leftover processes from previous sessions. Kill with `lsof -ti:9876 | xargs kill -9` before starting
- **No build step**: Frontend is a single file at `src/templates/index.html` (inline CSS + JS). Just edit and refresh.

## Project Structure

- `src/app.py` — Flask app, API endpoints, MILP solvers
- `src/templates/index.html` — Entire frontend (single file)
- `src/data_fetcher.py` — Data fetching + caching
- `src/feature_engineering.py` — 100+ features per player per GW
- `src/model.py` — XGBoost training (mean, quantile, sub-models)
- `src/predict.py` — Prediction pipeline
- `src/backtest.py` — Walk-forward backtesting
- `models/` — Saved .joblib model files
- `output/` — predictions.csv, charts, locked_teams.json
- `cache/` — Cached data (6h for GitHub CSVs, 30m for FPL API)

## Testing the App

1. Kill any existing process on port 9876
2. Start with `.venv/bin/python -m src.app`
3. Only one background task runs at a time (train, backtest, etc.)
4. Test API with curl, e.g.: `curl -s http://127.0.0.1:9876/api/my-team?manager_id=12904702`

## My Manager ID

12904702

## Full Project Context

See `CLAUDE_PROMPT.md` for comprehensive architecture, model details, and roadmap.
