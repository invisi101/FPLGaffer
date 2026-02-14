# FPL Points Predictor — Full Project Context for Claude

You are helping me build a Fantasy Premier League (FPL) prediction and team management system. This document gives you full context on the project's architecture, current capabilities, and future roadmap. The codebase is at `/Users/neil/dev/fpl-predictor`.

---

## 1. Project Purpose

An XGBoost-based FPL prediction system with a Flask web dashboard. It forecasts individual player points, recommends optimal squads and transfers, and aims to eventually manage a full FPL season autonomously from GW1 to GW38.

---

## 2. Tech Stack

- **Python 3.14** with `.venv` virtual environment
- **XGBoost** for all prediction models
- **scipy.optimize.milp** for squad/transfer optimisation (mixed-integer linear programming)
- **Flask** web dashboard (single-page app, no framework — vanilla JS)
- **pandas/numpy** for data processing
- **FPL API** (public, no auth) + **GitHub CSV data** (FPL-Core-Insights) for historical stats
- Server runs on `http://127.0.0.1:9876`

---

## 3. Directory Structure

```
fpl-predictor/
├── src/
│   ├── app.py                 # Flask web app, API endpoints, MILP solvers
│   ├── data_fetcher.py        # Data fetching from GitHub + FPL API with caching
│   ├── feature_engineering.py # 100+ features per player per GW
│   ├── feature_selection.py   # Phase 1: correlation, RF, Lasso, RFE analysis
│   ├── model.py               # XGBoost model training (mean, quantile, sub-models)
│   ├── predict.py             # Prediction pipeline orchestration
│   ├── backtest.py            # Walk-forward backtesting framework
│   ├── benchmark.py           # Model comparison (XGBoost vs LightGBM vs Ridge)
│   └── templates/
│       └── index.html         # Single-file frontend (HTML + CSS + JS)
├── models/                    # Saved .joblib model files
├── output/                    # predictions.csv, charts, reports, locked_teams.json
├── cache/                     # Cached CSV/JSON data files
├── requirements.txt
└── CLAUDE_PROMPT.md           # This file
```

---

## 4. Data Pipeline

### Sources
1. **GitHub (FPL-Core-Insights)**: Historical match stats, player stats, player match stats for 2024-2025 and 2025-2026 seasons. Cached for 6 hours.
2. **FPL Official API** (`fantasy.premierleague.com/api/`): Current player data (prices, form, injuries, ownership), fixtures, manager picks. Cached for 30 minutes.

### Key data tables
- `player_match_stats`: Per-player per-match stats (xG, xA, shots, tackles, etc.)
- `player_stats`: Season aggregates per player
- `matches`: Match results with team stats
- `bootstrap-static`: Current player metadata (cost, form, status, team)
- `fixtures`: Upcoming fixture schedule

### Feature Engineering (`feature_engineering.py`, 1400+ lines)
Builds a feature matrix with one row per player per gameweek. 100+ features including:

- **Player rolling stats** (3GW, 5GW windows): xG, xA, shots, touches, dribbles, crosses, tackles, clearances, goals, assists — all shifted by 1 GW to prevent leakage
- **EWM features** (span=5): Exponentially weighted xG, xA, xGOT, chances, shots on target
- **Upside/volatility**: xG volatility, form acceleration (trend detection), big chance frequency
- **Home/away form splits**: Separate rolling xG by venue
- **Opponent history**: Player's historical performance vs specific opponents (expanding mean)
- **Team rolling stats**: Team goals scored, xG, clean sheets, big chances (3GW rolling)
- **Opponent defensive stats**: Opponent xG conceded, shots conceded (rolling)
- **Rest/congestion**: Days rest between matches, fixture congestion rate
- **Fixture context**: FDR, is_home, opponent_elo, multi-GW lookahead (avg FDR next 3, home % next 3)
- **ICT/BPS**: Influence, creativity, threat, bonus point system stats
- **Market data**: Ownership, transfers in/out, net transfer momentum
- **Availability**: Chance of playing, availability rate over last 5 GWs
- **Interaction features**: xG × opponent goals conceded, CS opportunity (1/opp_xG)
- **Position encoding**: One-hot (pos_GKP, pos_DEF, pos_MID, pos_FWD)

### DGW (Double Gameweek) Handling
Players with 2+ fixtures in one GW get multiple rows (one per fixture) with different opponent data. Targets are divided by fixture count during training. Predictions are summed back at inference time.

---

## 5. Model Architecture

### Constants
```python
TARGETS = ["next_gw_points", "next_3gw_points"]
POSITION_GROUPS = ["GKP", "DEF", "MID", "FWD"]
```

### Tier 1: Mean Regression (Primary)
- 4 models (one per position) predicting `next_gw_points`
- XGBoost `reg:squarederror`
- Walk-forward CV with temporal ordering, 20 splits
- Sample weighting: current season 1.0, previous season 0.5
- Hyperparameter grid: n_estimators [100,200], max_depth [4,6], learning_rate [0.05,0.1]

### Tier 2: Quantile Models (Captain Picks)
- 2 models (MID, FWD only) predicting 80th percentile of `next_gw_points`
- XGBoost `reg:quantileerror`, alpha=0.80
- Used for captain scoring: `captain_score = 0.4 × mean + 0.6 × Q80`

### Tier 3: Decomposed Sub-Models
- ~20 models predicting individual point components: goals, assists, clean sheets, bonus, goals conceded, saves
- Position-specific objectives (Poisson for counts, logistic for binary CS)
- Combined via FPL scoring rules with playing probability weighting

### 3GW Predictions
Sum of three separate 1-GW predictions, each with correct future opponent data swapped in via `_build_offset_snapshot()`.

### Prediction Intervals
Residual-based percentiles (Q10, Q90) with conditional binning for heteroscedasticity.

---

## 6. MILP Squad Solver

### `_solve_milp_team()` — Optimal squad from scratch
Two-tier MILP: decision variables x_i (in squad) and s_i (starter).
- **Objective**: Maximise starting XI points + 0.1 × bench points
- **Constraints**: Budget, position counts (2 GKP, 5 DEF, 5 MID, 3 FWD), max 3 from same team, exactly 11 starters, formation rules (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), s_i ≤ x_i

### `_solve_transfer_milp()` — Optimal transfers
Same as above plus one extra constraint: `sum(x_i × is_current_i) >= 15 - max_transfers`. This forces the solver to keep at least (15 - N) current squad players, naturally producing the best squad reachable via N transfers.

**Budget calculation**: `bank + sum(now_cost of current squad)`. This ensures transfer affordability is realistic (can't spend more than you'd get from selling + bank). Slightly generous since real selling prices can be lower than now_cost for players who've risen.

---

## 7. Web Dashboard (app.py + index.html)

### Tabs
1. **Predictions**: Sortable player table with position filters, search, all prediction columns
2. **Best Team**: MILP solver for optimal 15-player squad within budget, pitch visualisation
3. **Backtest**: Walk-forward historical accuracy testing, per-position breakdown, bootstrap CIs
4. **GW Compare**: Lock in a recommended team, then compare actual vs best-possible after the GW
5. **My Team**: Import FPL squad by manager ID. Dual-pitch view (left: actual GW points with captain multiplier, right: predicted next GW). Shows squad value, sell value, bank, free transfers.
6. **Transfer Recommendations**: Below My Team. MILP-based transfer solver with controls for max transfers (1-5), target (1GW/3GW), wildcard mode. Shows transfer cards (OUT → IN with point/cost deltas), summary stats (points gained, hit, net gain), and new squad pitch with green-highlighted new signings.
7. **Feature Insights**: XGBoost feature importance charts, model accuracy cards

### API Endpoints
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/predictions` | Player predictions with filters/sorting |
| POST | `/api/refresh-data` | Re-fetch all data from sources |
| POST | `/api/train` | Train all model tiers |
| POST | `/api/feature-selection` | Run Phase 1 feature selection |
| POST | `/api/best-team` | MILP optimal squad |
| POST | `/api/lock-in-team` | Save team for later comparison |
| GET | `/api/locked-teams` | List saved teams |
| POST | `/api/backtest` | Run walk-forward backtest |
| GET | `/api/backtest-results` | Fetch backtest results |
| POST | `/api/gw-compare` | Compare locked team vs actual best |
| GET | `/api/gw-compare-results` | Fetch comparison results |
| GET | `/api/my-team?manager_id=X` | Import manager's FPL squad |
| POST | `/api/transfer-recommendations` | MILP transfer solver |
| GET | `/api/model-info` | Model metadata, cache age, next GW |
| GET | `/api/feature-report` | Feature selection report |
| POST | `/api/xgb-importance` | Generate XGBoost importance charts |
| GET | `/api/xgb-importance-report` | Importance report + chart filenames |
| GET | `/api/status` | SSE stream for live progress |

### Frontend Architecture
- Single HTML file with inline CSS and JS (no build tools)
- Dark theme with CSS variables
- Pitch visualisation using absolute-positioned rows on a gradient green background
- SSE connection for real-time task progress
- LocalStorage for persisting manager ID

---

## 8. Current State & Recent Work

- All core prediction and optimisation features are working
- Transfer recommendations use MILP with budget-aware constraint
- My Team shows dual pitches (actual points with captain multiplier / predicted)
- Squad value shows both now_cost sum (matches FPL app) and sell value (from API)
- Free transfer calculation walks through manager history accounting for wildcards/free hits
- Transfer cards show next 3 fixtures for both 1GW and 3GW targets
- Header styled with Outfit font, gradient text, and soccer ball SVG icon

---

## 9. Future Roadmap — Full Season Management

The end goal is for this app to manage an FPL team over a full season, starting from GW1 next year. Here's what needs to be built, roughly in priority order:

### Phase 1: Pre-Season (build before GW1)

**GW1 Cold Start**
- The model currently trains on in-season data. At GW1 there's none.
- Need pre-season predictions using last season's data + summer transfer info
- Everyone starts at 100.0m budget
- Strategy for early GWs when predictions are least reliable (lean more on priors/consensus)

**Season Dashboard & Weekly Tracking**
- Record each week: what the model recommended, what you actually did, actual results
- Track cumulative points, rank trajectory, budget evolution, model accuracy over time
- This turns it from a weekly tool into a season-long system

**Blank & Double Gameweek Detection**
- Some teams play 0 or 2 games in a GW — massive impact
- Detect from fixture calendar, factor into squad planning and chip timing
- Bench Boost during a DGW can be worth 20+ extra points

### Phase 2: Early Season

**Multi-Week Transfer Planning**
- Currently the solver is greedy (optimises this week only)
- Sometimes banking a transfer (0 this week → 2 free next week) is better
- Need a rolling horizon solver: "what's the best 2-3 week transfer plan"
- Consider: if I save a transfer, what could I do with 2 FTs next week vs 1 FT now?

**Price Change Integration**
- Player prices move daily based on transfer volume
- Buying before a rise saves 0.1m; selling before a drop costs 0.1m
- Over a season this compounds to 2-3m of extra budget
- Pull price change predictions or at least track transfer momentum

**Captain Optimisation**
- Captain doubles one player's points — the single biggest weekly decision
- The model has `captain_score` but the transfer solver doesn't optimise for "does my squad contain a strong captaincy option"
- Solver should weight having a premium captain candidate

### Phase 3: Mid-Season

**Chip Strategy Engine**
- When to play: Wildcard 1 (GW1-19), Wildcard 2 (GW20-38), Free Hit, Bench Boost, Triple Captain
- Logic: "fixture swing in GW14 — wildcard value is X points vs waiting"
- Bench Boost: best during DGW when bench players also have double fixtures
- Triple Captain: best on premium player during DGW with easy fixtures
- Free Hit: best during BGW when many teams don't play

**Fixture Difficulty Rotation**
- Plan 3-5 weeks ahead: "Arsenal have easy fixtures GW5-9, load up in GW4"
- The 3GW prediction handles some of this, but explicit fixture-swing transfer planning would be stronger

### Phase 4: When Trust Is Established

**Authenticated API Access**
- Currently read-only (public FPL API)
- With FPL login credentials, could:
  - Make transfers via the API automatically
  - Get exact selling prices (instead of now_cost approximation)
  - Set captain/vice-captain
  - Activate chips
- This is the "autopilot" endgame

### Summary Build Order
| Phase | What | When |
|-------|------|------|
| 1 | GW1 cold-start model (last season data) | Pre-season |
| 2 | Season dashboard & weekly tracking | Pre-season |
| 3 | Blank/double GW detection | Pre-season |
| 4 | Multi-week transfer planner (2-3 GW horizon) | Early season |
| 5 | Price change integration | Early season |
| 6 | Captain optimisation in solver | Early season |
| 7 | Chip strategy engine | Before first chip decision |
| 8 | Fixture rotation planning | Mid-season |
| 9 | Auth + auto-execution | When trusted |

---

## 10. Key Design Decisions Already Made

- **MILP over greedy**: Squad and transfer selection uses proper integer programming, not heuristic swaps. This finds globally optimal solutions.
- **Separate models per position**: GKP/DEF/MID/FWD have different feature sets and models because different stats matter for each.
- **Walk-forward validation**: All backtesting and CV uses strictly temporal ordering — no future data leakage.
- **Sub-model decomposition**: Predicting goals, assists, CS separately then combining via scoring rules, rather than just predicting total points. This helps with interpretability and captain picks.
- **Quantile regression for captaincy**: Q80 model captures explosive upside potential rather than just expected value.
- **now_cost budget**: Transfer solver uses `bank + sum(now_cost)` rather than API sell_value, because it better reflects real affordability of swaps.
- **Single-file frontend**: Everything in one index.html for simplicity. No React/Vue/build tools.
- **Background tasks with SSE**: Long operations (training, backtesting) run in threads with real-time progress via Server-Sent Events.

---

## 11. How to Run

```bash
# Activate virtualenv
source .venv/bin/activate

# Start web dashboard
python -m src.app
# → http://127.0.0.1:9876

# Or run predictions from CLI
python -m src.predict --train --tune
```

From the web UI: "Get Latest Data" → "Train Models" → predictions appear automatically. Then use Best Team, My Team, Transfer Recommendations etc.

---

## 12. My Manager ID

904686 (use this for testing My Team / Transfer features)
