# FPL Gaffer Brain — Claude Code Notes

## Project Goal

Build a fully autonomous FPL manager. This is NOT just a prediction tool — it should think and plan like a real FPL manager across the entire season:

- **Transfer planning**: Rolling 5-GW horizon with FT banking, price awareness, and fixture swings
- **Squad building**: Shape the squad toward upcoming fixture runs, not just next GW
- **Captain planning**: Joint captain optimization in the MILP solver + pre-planned captaincy calendar
- **Chip strategy**: Evaluate all 4 chips across every remaining GW using DGW/BGW awareness, squad-specific predictions, and chip synergies (WC→BB, FH+WC)
- **Price awareness**: Ownership-based price change predictions with probability scores
- **Reactive adjustments**: Auto-detect injuries, fixture changes, and prediction shifts that invalidate the plan — with SSE-driven alerts and one-click replan
- **Outcome tracking**: Record what was recommended vs what happened, track model accuracy over time

Every decision (transfer, captain, bench order, chip) is made in context of the bigger picture. The app produces a rolling multi-week plan that constantly recalculates as new information comes in.

## Repos

- **This project**: `https://github.com/invisi101/fplmanager` (active development)
- **Original predictor**: `https://github.com/invisi101/fplxti` (previous project this was forked from)

## Environment

- **Python**: Use `.venv/bin/python`, NOT system `python3` (system Python lacks project dependencies)
- **Run server**: `.venv/bin/python -m src.app` (serves on `http://127.0.0.1:9875`)
- **Port 9875**: Often has leftover processes from previous sessions. Kill with `lsof -ti:9875 | xargs kill -9` before starting
- **No build step**: Frontend is a single file at `src/templates/index.html` (inline CSS + JS). Just edit and refresh.

## My Manager ID

12904702

---

## Architecture Overview

Three-layer system: **Data → Features/Models → Strategy/Solver**, backed by SQLite and served via Flask.

### Project Structure

```
src/
├── app.py                  # Flask app, 40+ API endpoints, background task runner, SSE
├── templates/
│   └── index.html          # Entire frontend (single file, inline CSS + JS, dark theme)
├── data_fetcher.py         # GitHub CSV + FPL API data fetching with caching
├── feature_engineering.py  # ~1400 lines, 100+ features per player per GW
├── model.py                # XGBoost training: mean, quantile (Q80), decomposed sub-models
├── predict.py              # Prediction pipeline: 1-GW, 3-GW, 8-GW horizon, captain scores
├── backtest.py             # Walk-forward backtesting framework
├── solver.py               # MILP solvers: squad selection + transfer optimization + captain
├── strategy.py             # Strategic brain: ChipEvaluator, MultiWeekPlanner, CaptainPlanner, PlanSynthesizer
├── season_manager.py       # Season orchestration: recommendations, outcomes, prices, plan health
├── season_db.py            # SQLite with 8 tables (season, snapshots, recommendations, outcomes, prices, fixtures, plans, changelog)
└── feature_selection.py    # Feature importance analysis (correlation, RF, Lasso, RFE)

models/     # Saved .joblib model files (gitignored)
output/     # predictions.csv, charts (gitignored)
cache/      # Cached data: 6h for GitHub CSVs, 30m for FPL API (gitignored)
```

---

## Data Pipeline

### Sources
1. **GitHub (FPL-Core-Insights)**: Historical match stats, player stats, player match stats for 2024-2025 and 2025-2026 seasons. Cached 6 hours.
2. **FPL API** (public, no auth): Current player data (prices, form, injuries, ownership), fixtures, manager picks/history/transfers. Cached 30 minutes.

### Feature Engineering (`feature_engineering.py`)
100+ features per player per GW including:
- Player rolling stats (3/5/10 GW windows): xG, xA, shots, touches, dribbles, crosses, tackles, goals, assists
- EWM features (span=5): Exponentially weighted xG, xA, xGOT
- Upside/volatility: xG volatility, form acceleration, big chance frequency
- Home/away form splits
- Opponent history: Player's historical performance vs specific opponents
- Team rolling stats: Goals scored, xG, clean sheets, big chances
- Opponent defensive stats: xG conceded, shots conceded
- Rest/congestion: Days rest, fixture congestion rate
- Fixture context: FDR, is_home, opponent_elo, multi-GW lookahead
- ICT/BPS: Influence, creativity, threat, bonus points
- Market data: Ownership, transfer momentum
- Availability: Chance of playing, availability rate
- Interaction features: xG × opponent goals conceded, CS opportunity

All features shifted by 1 GW to prevent leakage. DGW handling: multiple rows per fixture, targets divided by fixture count, predictions summed.

---

## Model Architecture

### Tier 1: Mean Regression (Primary)
- 4 models per position × 2 targets (next_gw_points, next_3gw_points)
- XGBoost `reg:squarederror`, walk-forward CV (20 splits)
- Sample weighting: current season 1.0, previous 0.5

### Tier 2: Quantile Models (Captain Picks)
- MID + FWD only, 80th percentile of next_gw_points
- `captain_score = 0.4 × mean + 0.6 × Q80` — captures explosive upside

### Tier 3: Decomposed Sub-Models
- ~20 models predicting individual components: goals, assists, clean sheets, bonus, saves
- Position-specific objectives (Poisson for counts, logistic for binary CS)
- Combined via FPL scoring rules with playing probability weighting

### Multi-GW Predictions
- 3-GW: Sum of three 1-GW predictions with correct opponent data per offset
- 8-GW horizon: Model predictions for near-term, fixture heuristics for distant GWs
- Confidence decays with distance (0.95 → 0.77 at GW+5)

---

## MILP Solver (`solver.py`)

### `solve_milp_team()` — Optimal squad from scratch
Two-tier MILP with optional captain optimization:
- **Variables**: `x_i` (in squad), `s_i` (starter), `c_i` (captain, when `captain_col` provided)
- **Objective**: max(0.9 × starting XI pts + 0.1 × bench pts + captain bonus)
- **Constraints**: Budget, positions (2/5/5/3), max 3 per team, 11 starters, formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD), exactly 1 captain who is a starter
- **Returns**: starters, bench, total_cost, starting_points, captain_id

### `solve_transfer_milp()` — Optimal transfers
Same as above plus: `sum(x_i × is_current_i) >= 15 - max_transfers` — keeps at least (15 - N) current players.
- Budget = bank + sum(now_cost of current squad)
- Also supports joint captain optimization

---

## Strategic Planning Brain (`strategy.py`)

### ChipEvaluator
Evaluates all 4 chips (BB, TC, FH, WC) across every remaining GW:
- Near-term: Uses model predictions (solve MILP for FH/WC, bench sums for BB, best player for TC)
- Far-term: Fixture heuristics (DGW count, BGW count, FDR)
- Synergies: WC→BB (build BB-optimized squad), FH+WC (complementary strategy)

### MultiWeekPlanner
Rolling **5-GW** transfer planner:
- Forward simulation: tries each FT allocation for GW+1, picks path maximizing total points
- Considers: FT banking (save → use 2 next week), fixture swings, price change probability
- Reduces pool to top 200 players for efficiency
- Passes `captain_col` to MILP solver for captain-aware squad building

### CaptainPlanner
Pre-plans captaincy across the prediction horizon:
- Uses transfer plan squads to pick captain from the planned squad (not just current)
- Flags weak captain GWs (predicted < 4 pts)

### PlanSynthesizer
Combines all plans into a coherent timeline:
- Chip schedule (synergy-aware: uses WC→BB combos when valuable)
- Natural-language rationale explaining the overall strategy
- Comparison with previous plan → changelog

### Reactive Re-planning
- `detect_plan_invalidation()`: Checks injuries (critical), fixture changes (BB without DGW), prediction shifts (>50% captain drop), doubtful players
- `apply_availability_adjustments()`: Zeros predictions for injured/suspended players
- `check_plan_health()`: Lightweight check using bootstrap data (no prediction regeneration)
- Auto-triggers on data refresh via SSE `plan_invalidated` events

---

## Season Manager (`season_manager.py`)

Orchestrates everything for a full season:

### Weekly Workflow
1. **Refresh Data** → updates cache, detects availability issues, checks plan health
2. **Generate Recommendation** → multi-GW predictions, chip heatmap, transfer plan, captain plan, strategic plan synthesis, stores everything in DB
3. **Review Action Plan** → clear steps (transfer X out / Y in, set captain to Z, activate chip)
4. **Make Moves** → user executes on FPL website
5. **Record Results** → imports actual picks, compares to recommendation, tracks accuracy

### Price Tracking
- `track_prices()`: Snapshots prices for squad + top 30 transferred-in players
- `get_price_alerts()`: Raw net-transfer threshold alerts
- `predict_price_changes()`: Ownership-based algorithm approximation
  - `transfer_ratio = net_transfers / (ownership_pct × 100,000)`
  - Rise if ratio > 0.005, fall if < -0.005
  - Probability = min(1.0, abs(ratio) / 0.01)
- `get_price_history()`: Historical snapshots with date/price/net_transfers

### Pre-Season
- `generate_preseason_plan()`: MILP for initial squad (100.0m budget) + full-season chip schedule
- Falls back to price-based heuristic if no model predictions available

---

## Database Schema (`season_db.py`)

8 SQLite tables:
| Table | Purpose |
|-------|---------|
| `season` | Manager seasons (id, manager_id, name, start_gw, current_gw) |
| `gw_snapshot` | Per-GW state (squad_json, bank, team_value, points, rank, captain, transfers_in/out) |
| `recommendation` | Pre-GW advice (transfers_json, captain, chip, predicted_points) |
| `recommendation_outcome` | Post-GW tracking (followed_transfers, actual_points, point_delta) |
| `price_tracker` | Player price snapshots (price, transfers_in/out, snapshot_date) |
| `fixture_calendar` | GW × team fixture grid (fixture_count, fdr_avg, is_dgw, is_bgw) |
| `strategic_plan` | Full plan JSON + chip heatmap JSON (per season per GW) |
| `plan_changelog` | Plan change history (chip reschedule, captain change, reason) |

---

## API Endpoints

### Core
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/predictions` | Player predictions with filters/sorting |
| POST | `/api/refresh-data` | Re-fetch data, rebuild predictions, check plan health |
| POST | `/api/train` | Train all model tiers + generate predictions |
| POST | `/api/best-team` | MILP optimal squad |
| GET | `/api/my-team?manager_id=ID` | Import manager's FPL squad |
| POST | `/api/transfer-recommendations` | MILP transfer solver (with captain optimization) |
| GET | `/api/status` | SSE stream for live progress |

### Season Management
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/season/init` | Backfill season history |
| GET | `/api/season/dashboard` | Full dashboard (rank, budget, accuracy_history) |
| POST | `/api/season/generate-recommendation` | Generate strategic plan + recommendation |
| POST | `/api/season/record-results` | Import actual results, compare to advice |
| GET | `/api/season/action-plan` | Clear action items for next GW |
| GET | `/api/season/outcomes` | All recorded outcomes |

### Strategic Planning
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET/POST | `/api/season/strategic-plan` | Fetch/generate full strategic plan |
| GET | `/api/season/chip-heatmap` | Chip values across remaining GWs |
| GET | `/api/season/plan-health` | Check plan validity (injuries/fixtures) |
| GET | `/api/season/plan-changelog` | Plan change history |

### Prices
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/season/prices` | Latest prices + raw alerts |
| GET | `/api/season/price-predictions` | Ownership-based price predictions |
| GET | `/api/season/price-history` | Price movement history (date/price/transfers) |

### Other
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/backtest` | Walk-forward backtesting |
| POST | `/api/gw-compare` | Compare manager's actual team vs hindsight-best for any past GW |
| POST | `/api/preseason/generate` | Pre-season initial squad + chip plan |
| GET | `/api/season/fixtures` | Fixture calendar |

---

## UI Structure (`src/templates/index.html`)

Single-file frontend with dark theme, CSS variables, SSE progress, localStorage persistence.

### Main Tabs
1. **Predictions** — Sortable player table, position filters, search
2. **Best Team** — MILP squad selector with pitch visualization
3. **GW Compare** — Compare your actual FPL team vs hindsight-best for any past GW (dual pitch with overlap highlighting)
4. **My Team** — Import FPL squad, dual-pitch (actual pts / predicted), transfer recommendations
5. **Season** — Full season management dashboard

### Season Sub-tabs
- **Overview**: Rank chart, points bar chart, budget chart, prediction accuracy dual-line chart
- **Workflow**: Step indicators (Refresh → Recommend → Review → Execute → Record), action plan, outcomes
- **Fixtures**: GW × team fixture grid
- **Transfers**: Transfer history table
- **Chips**: Status (used/available) + values
- **Prices**: Alerts, ownership-based predictions (risers/fallers with probability bars), price history multi-line chart, squad prices table
- **Strategy**: Plan health banner (auto-detect + replan button), rationale, 5-GW transfer timeline cards, captain plan badges, chip schedule + synergy annotations, chip heatmap table (color-coded), plan changelog

### Charts
- `drawLineChart()` — Single-line (rank, budget)
- `drawBarChart()` — Bar chart (points per GW)
- `drawDualLineChart()` — Two-line with legend (predicted vs actual accuracy)
- Price history chart — Multi-line canvas for top 5 movers

---

## Testing the App

1. Kill any existing process: `lsof -ti:9875 | xargs kill -9`
2. Start: `.venv/bin/python -m src.app`
3. Only one background task runs at a time (train, backtest, etc.)
4. Test API: `curl -s http://127.0.0.1:9875/api/my-team?manager_id=12904702`

### Key test commands
```bash
# Strategic plan (5-GW timeline)
curl -s http://127.0.0.1:9875/api/season/strategic-plan?manager_id=12904702 | python3 -c "import sys,json; t=json.load(sys.stdin)['plan']['timeline']; print(f'{len(t)} GW entries')"

# Plan health
curl -s http://127.0.0.1:9875/api/season/plan-health?manager_id=12904702

# Price predictions
curl -s http://127.0.0.1:9875/api/season/price-predictions?manager_id=12904702

# Dashboard with accuracy history
curl -s http://127.0.0.1:9875/api/season/dashboard?manager_id=12904702 | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'accuracy_history: {len(d.get(\"accuracy_history\",[]))} entries')"

# GW Compare (manager's actual team vs hindsight-best)
curl -s http://127.0.0.1:9875/api/gw-compare -X POST -H 'Content-Type: application/json' -d '{"manager_id":12904702,"gameweek":20}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'My: {d[\"my_team\"][\"starting_actual\"]} pts, Best: {d[\"best_team\"][\"starting_actual\"]} pts, Capture: {d[\"capture_pct\"]}%')"
```

---

## What's Built (Complete)

All phases from the original roadmap are done:

- **Pre-season**: GW1 cold-start predictions, initial squad selection, full-season chip plan
- **Season dashboard**: GW-by-GW tracking of rank, points, budget, model accuracy
- **DGW/BGW detection**: Fixture calendar with DGW/BGW flags, factored into chip timing
- **Multi-week transfer planner**: 5-GW rolling horizon with FT banking simulation
- **Price change integration**: Ownership-based predictions with probability, price history charts
- **Captain optimization**: Joint MILP solver with captain decision variables (3n vars)
- **Chip strategy engine**: Full heatmap across all remaining GWs + synergy detection
- **Fixture rotation**: Fixture swing bonuses in transfer planning
- **Reactive re-planning**: Injury/fixture change detection with SSE alerts + one-click replan
- **Outcome tracking**: Recommendation vs actual comparison, accuracy history charts
- **GW Compare**: Compare your actual FPL team vs hindsight-best for any past GW
- **Transfer history**: Per-GW transfer details (player names, costs) stored in snapshots

---

## PRIORITY FIX: Confirmed Bugs (Audit Feb 2025)

A comprehensive audit found the following confirmed bugs. They are ordered by severity. Each should be confirmed by reading the cited lines before fixing. **Do not attempt to fix all of these at once** — work through them in priority order, testing each fix before moving on.

### CRITICAL — Silently producing wrong results now

**Bug 1: Outcome tracking always reports 100% transfer compliance**
- **File**: `season_manager.py:844`
- **Problem**: `transfers_json` is stored as `[{"out": {...}, "in": {...}}, ...]` (paired format), but the parsing iterates the top-level list and checks `t.get("direction")` which doesn't exist at that level. `rec_in_ids` is always empty → `followed_transfers` is always 1.
- **Fix**: Extract from nested structure: `rec_in_ids = {t["in"]["player_id"] for t in rec_transfers if t.get("in", {}).get("player_id")}`

**Bug 2: Chip follow-through is hardcoded to 0**
- **File**: `season_manager.py:853`
- **Problem**: `followed_chip=0` is always passed. The actual chip used (`chip_map.get(current_event)`) and recommended chip (`rec.get("chip_suggestion")`) are both available but never compared.
- **Fix**: Compare them: `followed_chip = 1 if chip_map.get(current_event) == rec.get("chip_suggestion") else 0`

**Bug 3: Re-initializing a season CASCADE deletes all history**
- **File**: `season_db.py:156`
- **Problem**: `create_season()` uses `INSERT OR REPLACE` which in SQLite is DELETE + INSERT. Since `id` is AUTOINCREMENT, the replaced row gets a new ID, cascading deletes across all 7 child tables (snapshots, recommendations, outcomes, prices, plans, changelog).
- **Fix**: Use `INSERT ... ON CONFLICT` to preserve the existing `id`, or check-then-insert:
- **Current code** (`season_db.py:155-163`):
  ```python
  cur = conn.execute(
      """INSERT OR REPLACE INTO season
         (manager_id, manager_name, team_name, season_name, start_gw, current_gw)
         VALUES (?, ?, ?, ?, ?, ?)""",
      (manager_id, manager_name, team_name, season_name, start_gw, start_gw),
  )
  season_id = cur.lastrowid
  ```
- **Fixed code**:
  ```python
  cur = conn.execute(
      """INSERT INTO season
         (manager_id, manager_name, team_name, season_name, start_gw, current_gw)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT(manager_id, season_name) DO UPDATE SET
           manager_name=excluded.manager_name,
           team_name=excluded.team_name""",
      (manager_id, manager_name, team_name, season_name, start_gw, start_gw),
  )
  season_id = cur.lastrowid or conn.execute(
      "SELECT id FROM season WHERE manager_id=? AND season_name=?",
      (manager_id, season_name),
  ).fetchone()[0]
  ```
  Note: `lastrowid` is 0 on conflict/update, so the fallback SELECT is needed.

**Bug 4: Opponent history feature has data leakage**
- **File**: `feature_engineering.py:316-330`
- **Problem**: `_build_opponent_history_features` computes expanding means without `shift(1)`. Every other rolling feature in the file uses `shift(1)`. When predicting GW N+1 against opponent X, the feature can include GW N's actual match stats against X if GW N's opponent was also X (e.g., second meeting in same season).
- **Fix**: Add `.shift(1)` before `.expanding()` like every other rolling feature.

### HIGH — Strategy/solver logic errors affecting recommendations

**Bug 5: Captain bonus massively undervalued in MILP**
- **File**: `solver.py:46-48` (also `solver.py:209-212`)
- **Problem**: Captain bonus in objective = `captain_score - pred` (~1-2 pts extra). But in FPL, captaincy doubles points — the real bonus is `pred` itself (~5-8 pts). The solver treats captaincy as nearly irrelevant to squad building. The objective for a captained starter is `pred + (captain_score - pred) = captain_score`, which is only ~7 pts instead of `pred + pred = ~12 pts`.
- **Fix**: Use `captain_score` directly as the bonus coefficient. This must be changed in both `solve_milp_team` AND `solve_transfer_milp`.
- **Current code** (appears twice — `solver.py:47-48` and `solver.py:211-212`):
  ```python
  captain_bonus = captain_scores - pred  # Extra value from captaining
  captain_bonus = np.maximum(captain_bonus, 0)  # Only positive bonus
  ```
- **Fixed code** (replace in both locations):
  ```python
  captain_bonus = captain_scores  # Captain doubles points; bonus = full captain_score
  ```
  This makes the total objective for a captained starter: `0.9*pred + captain_score`, properly valuing the captain pick. The `np.maximum` clamp is no longer needed since captain_scores are always non-negative.

**Bug 6: Budget lost between multi-week planning steps**
- **File**: `strategy.py:545,647` and `season_manager.py:1493`
- **Problem**: After each MILP solve, `budget = result["total_cost"]` (squad cost). Unspent bank is discarded. If solver picks a 98.5m squad with 102m available, 3.5m disappears.
- **Fix**: Don't reassign budget — it should stay as the manager's total available funds (bank + squad value), which doesn't change just because the solver picked a cheaper squad.

**Bug 7: Free Hit evaluation uses unlimited budget**
- **File**: `strategy.py:205` and `season_manager.py:1429`
- **Problem**: FH evaluation uses `budget=1000` (100m) instead of the manager's actual squad value + bank. Inflates FH value for managers under 100m, understates for managers over 100m.
- **Fix**: Pass `total_budget` (already available as a parameter) instead of hardcoded 1000.

**Bug 8: FT count not reset after Free Hit**
- **File**: `strategy.py:549`
- **Problem**: After FH in simulation, FTs accumulate normally (`ft = min(ft + 1, 5)`). In FPL, after a Free Hit, FTs reset to 1.
- **Fix**: Set `ft = 1` after a Free Hit GW.

**Bug 9: Injured players can be recommended as captain**
- **File**: `strategy.py:1091`
- **Problem**: `apply_availability_adjustments()` zeros `predicted_points` but not `captain_score`. The MILP solver uses `captain_score` for captain selection, so an injured player can still be picked.
- **Fix**: Also zero `captain_score` (and `predicted_next_gw_points_q80` if present) for injured/unavailable players.

**Bug 10: Multi-week planner only truly optimizes GW1**
- **File**: `strategy.py:572-573`
- **Problem**: For GW2-5, the planner always uses `min(ft, 2)` transfers greedily. It never considers saving FTs in GW2 for a bigger move in GW3, or using 3+ accumulated FTs.
- **Scope**: This is an algorithmic limitation, not a simple bug. A proper fix would require tree search or dynamic programming across the full horizon. Consider whether the current greedy approach is "good enough" or worth the complexity of a proper search.

### HIGH — Feature engineering errors affecting model quality

**Bug 11: Cross-season decay is completely disabled**
- **File**: `feature_engineering.py:1268-1272`
- **Problem**: Decay distance uses raw gameweek numbers which reset each season. GW1 of new season minus GW38 of old season = -37, clipped to 0, so decay = 0.90^0 = 1.0 (no decay at all). Carried-over values enter the new season at full strength.
- **Fix**: Create a monotonically increasing row counter so distance is always positive across seasons.
- **Current code** (`feature_engineering.py:1268-1272`):
  ```python
  last_real_gw = group_df.groupby(group_col)["gameweek"].transform(
      lambda s: s.where(group_df.loc[s.index, col].notna()).ffill()
  )
  distance = group_df["gameweek"] - last_real_gw
  decay = CROSS_SEASON_DECAY ** distance.clip(lower=0)
  ```
- **Fixed code**: Add a `gw_index` column before the `_ffill_with_decay` call (around line 1248) that increases monotonically across seasons, then use it instead of `gameweek`:
  ```python
  # Add monotonic GW counter before calling _ffill_with_decay
  season_order = {s: i for i, s in enumerate(sorted(combined["season"].unique()))}
  combined["_gw_index"] = combined["season"].map(season_order) * 38 + combined["gameweek"]
  ```
  Then inside `_ffill_with_decay`, replace `"gameweek"` with `"_gw_index"` in the distance calc:
  ```python
  last_real_gw = group_df.groupby(group_col)["_gw_index"].transform(
      lambda s: s.where(group_df.loc[s.index, col].notna()).ffill()
  )
  distance = group_df["_gw_index"] - last_real_gw
  decay = CROSS_SEASON_DECAY ** distance.clip(lower=0)
  ```
  Distance from GW38 of season 0 to GW1 of season 1 becomes `(1*38+1) - (0*38+38) = 1`, giving decay = 0.90. By GW5, decay = 0.90^5 = 0.59.

**Bug 12: DGW rolling stats use mean instead of sum**
- **File**: `feature_engineering.py:44,77,110` (also 219, 244 for team features)
- **Problem**: DGW match stats are averaged before entering rolling windows. A player with 0.8 + 0.5 xG across a DGW gets 0.65 — identical to a single-match 0.65. Systematically undervalues productive DGWs in form signals.
- **Fix**: Use `.sum()` instead of `.mean()` for per-GW aggregation of DGW stats, matching how targets work. Alternatively, keep per-match rows in rolling calculations.

**Bug 13: rest_days feature is off by one GW**
- **File**: `feature_engineering.py:1134-1138`
- **Problem**: Feature row at GW N (predicting GW N+1) contains rest days before GW N, not before GW N+1. All other "next GW" features are correctly shifted.
- **Fix**: Apply the same `gameweek - 1` shift to rest_days before merging, or compute rest for the next GW directly.

### MEDIUM — Functional issues that degrade quality

**Bug 14: Dashboard chip status ignores half-season reset**
- **File**: `season_manager.py:1808-1824`
- **Problem**: Flat check of chip usage across entire season. WC used in GW5 shows as "used" even after GW20 reset. The recommendation engine (lines 427-434) has correct half-season logic but the dashboard doesn't.
- **Fix**: Apply the same half-season logic from `_evaluate_chips()`.

**Bug 15: Injury detection stops after first affected GW**
- **File**: `strategy.py:1043,1054`
- **Problem**: `break` statement means multi-week injuries only show as affecting one GW in plan health. A 5-week injury may only trigger replan for the immediate GW.
- **Fix**: Remove `break` statements; collect all affected GWs.

**Bug 16: DGW fixture map only shows one opponent**
- **File**: `season_manager.py:894`
- **Problem**: `code not in result` guard discards second DGW fixture. Action plan shows "MCI (H)" instead of "MCI (H) + LIV (A)".
- **Fix**: Accumulate into a list: `result.setdefault(code, []).append(...)`.

**Bug 17: Stale event_points in record_results**
- **File**: `season_manager.py:776`
- **Problem**: Per-player `event_points` comes from bootstrap cache which may not have been refreshed. The overall points comparison uses live data, but per-player scores in the stored squad snapshot can be wrong.
- **Fix**: Force bootstrap refresh at start of `record_actual_results()`, or use live history data.

**Bug 18: Historical backfill stores current prices**
- **File**: `season_manager.py:217`
- **Problem**: Every historical GW's squad snapshot uses today's `now_cost`, not the price at that GW. Unavoidable from public API (no historical price data), but should be noted.

**Bug 19: `_get_next_gw` returns GW39 at end of season**
- **File**: `app.py:144` and `season_manager.py:91`
- **Problem**: When GW38 is current, returns 39. Propagates to recommendations, fixtures, and planning for non-existent GW.
- **Fix**: Clamp to 38 or return `None` when season is over.

**Bug 20: BB evaluates current squad for all future GWs**
- **File**: `strategy.py:128-149`
- **Problem**: BB value is calculated using today's bench for every remaining GW. For a BB in GW35, evaluation is meaningless since the squad will have changed. The WC→BB synergy uses a crude 30% uplift heuristic instead of solving the actual WC-optimized squad's bench.

### Advice for fixing session

1. **Confirm each bug first** — read the exact lines cited, understand the surrounding code, and verify the issue is real before changing anything.
2. **Start with Bugs 1-3** (critical) — these are simple fixes with the highest impact. Bug 1 is a one-line JSON parsing fix. Bug 3 requires changing the SQL strategy. Bug 4 requires adding `shift(1)`.
3. **Bugs 5-9** (high severity strategy) are all independent — they can be fixed in any order. Bug 5 (captain bonus) is the most impactful for recommendation quality.
4. **Bugs 11-13** (feature engineering) require retraining after fixing. Fix all three, then retrain.
5. **Test after each fix** — start the server, run the test commands above, check that nothing breaks.
6. **Do NOT attempt to fix Bug 10** (multi-week planner optimization) unless specifically asked — it's a significant algorithmic redesign.

---

## Remaining TODO: Rethink Backtesting & Feature Visualization

The following 4 UI buttons have been **removed from the frontend** (but all backend code is preserved):

- **Run Feature Selection** (`/api/feature-selection`) — `src/feature_selection.py` still exists
- **Model Importance** (`/api/xgb-importance`) — endpoint still in `src/app.py`
- **Feature Insights** panel — showed feature charts, reports, and XGBoost importance
- **Backtest** panel — walk-forward backtesting UI with per-GW breakdown

**Why removed**: After the dynamic season handling rewrite (multi-season support, graduated weights, generic data fetching), these features need rethinking. The backtest framework may need updates for multi-season walk-forward, and the feature visualization approach should be reconsidered given the new 100+ feature set across N seasons.

**To restore**: The backend endpoints (`/api/feature-selection`, `/api/xgb-importance`, `/api/feature-report`, `/api/xgb-importance-report`, `/api/backtest`, `/api/backtest-results`) and Python files (`src/backtest.py`, `src/feature_selection.py`) are all intact. To bring back the UI, re-add the buttons to the action bar in `src/templates/index.html`, re-add the HTML panels, CSS styles, and JS functions. Check git history for the removed code (commit after the dynamic season handling commit).

**What to consider when rebuilding**:
- Backtest should work seamlessly across all dynamically detected seasons
- Feature importance visualization could show per-season breakdowns
- Consider integrating model accuracy metrics into the Season dashboard instead of a separate panel
- The current walk-forward CV in `src/backtest.py` may need updating for 3+ season training

---

## Remaining TODO: Full Autonomy

The one remaining feature is **authenticated FPL API access for autonomous execution**:

### What it needs
1. **FPL Authentication**: Login with FPL credentials (email/password) to get session cookies
   - FPL uses `https://users.premierleague.com/accounts/login/` for auth
   - Returns session cookies needed for write endpoints
   - Credentials should be stored securely (env vars or encrypted config, never in code)

2. **Write API Endpoints**: Use authenticated session to:
   - `POST /api/transfers` — Execute transfers (player_in, player_out)
   - `POST /api/my-team/captain` — Set captain and vice-captain
   - `POST /api/my-team/` — Set starting XI and bench order
   - `POST /api/chips/activate` — Activate chips (wildcard, freehit, bboost, 3xc)

3. **Exact Selling Prices**: Authenticated API provides real selling prices (which account for price rise profit sharing — you only get 50% of price rises). Currently we use `now_cost` which is slightly generous.

4. **Execution Flow**: After generating a recommendation:
   - Show the action plan in the UI (already done)
   - Add "Execute All" button that calls authenticated endpoints
   - Confirm before executing (show what will happen)
   - Log what was executed vs what was planned
   - Handle errors gracefully (insufficient funds, player unavailable, deadline passed)

5. **Safety Guardrails**:
   - Never execute without explicit user confirmation (unless configured for full autopilot)
   - Deadline awareness: warn if close to GW deadline, refuse if past
   - Rollback info: show how to reverse transfers manually if something goes wrong
   - Rate limiting: respect FPL API rate limits
   - Dry-run mode: show what would happen without actually doing it

6. **Scheduling** (optional, for true autopilot):
   - Cron-like scheduler: refresh data daily, generate recommendation 24h before deadline
   - Auto-execute transfers N hours before deadline if confidence is high
   - Alert (email/push) if plan health issues detected

### Implementation approach
- Add `src/fpl_auth.py` for authentication + authenticated API calls
- Add execution methods to `SeasonManager` (execute_transfers, set_captain, activate_chip)
- Add `/api/season/execute` endpoint with confirmation flow
- Add "Execute" button to the Action Plan UI section
- Store credentials via environment variables (`FPL_EMAIL`, `FPL_PASSWORD`)
