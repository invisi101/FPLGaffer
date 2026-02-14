# FPL Points Predictor

XGBoost-based Fantasy Premier League points predictor with a local web dashboard.

---

## Windows (Standalone .exe)

No Python required. Download and run.

1. Go to the [latest release](https://github.com/invisi101/fplxti/releases/latest)
2. Download **`FPL-Predictor-Windows.zip`**
3. Extract the zip to a folder (e.g. your Desktop)
4. Double-click **`FPL Predictor.exe`**
5. A console window will open and your browser will launch automatically

To stop the app, close the console window.

---

## macOS

```bash
mkdir -p ~/fpl
git clone https://github.com/invisi101/fplxti.git ~/fpl/fpl-predictor
cd ~/fpl/fpl-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.app
```

Open http://127.0.0.1:9876 in your browser.

---

## Linux

```bash
mkdir -p ~/fpl
git clone https://github.com/invisi101/fplxti.git ~/fpl/fpl-predictor
cd ~/fpl/fpl-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.app
```

Open http://127.0.0.1:9876 in your browser.

### Desktop Launcher (Linux)

```bash
cp fpl-predictor.desktop ~/.local/share/applications/
```

Then search "FPL" in your app launcher.

---

## Running for the First Time

1. **Get Latest Data** — click this first to download player stats, fixtures, and team data from the FPL API
2. **Train Models** — trains XGBoost prediction models for all positions (takes a few minutes). You need to do this before predictions will appear
3. You're set — the Predictions tab will now show predicted points for every player

Retrain models every few weeks to keep predictions fresh as the season progresses.

---

## Using the App

### Action Bar

The buttons across the top control data and model operations. A status indicator shows when a task is running.

- **Get Latest Data** — fetches the latest player stats, fixtures, and injury news. Do this weekly or before making transfer decisions
- **Train Models** — trains XGBoost models for all positions and targets, plus quantile models for captain picks and decomposed sub-models. Generates new predictions
- **Run Feature Selection** — analyses which stats best predict FPL points using four selection methods (Lasso, Random Forest, RFE, XGBoost). Optional — for curiosity or model tuning
- **Model Importance** — extracts and charts feature importances from the trained XGBoost models

### Predictions

Sortable, searchable table of every player's predicted points. Filter by position (All / GKP / DEF / MID / FWD). Columns include cost, form, last GW points, total points, predicted next GW, predicted next 3 GWs, FDR (colour-coded fixture difficulty), home/away, FPL's own ep_next, and the next 3 opponents.

### Best Team

Build the optimal 15-player squad for a given budget using a MILP solver. Set your budget (default 100.0m), choose whether to optimise for next GW or next 3 GWs, and click **Pick Team**. The result is shown on an interactive pitch with the best XI and bench. The captain is automatically set to the highest-predicted player. Click **Lock In This Team** to save it for later comparison in GW Compare.

### My Team

Import your actual FPL squad by entering your Manager ID and clicking **Import Squad**. Shows your team name, overall rank, and points. Displays your current squad on a pitch with actual and predicted points, plus injury/suspension indicators.

The transfer recommender finds optimal transfers: set the max number of transfers (auto-set to your free transfers), choose GW or 3GW optimisation, optionally enable Wildcard mode, and click **Find Transfers**. Shows each transfer (out/in) with predicted points gained, any points hit cost, and the net gain. The new squad is displayed with incoming players highlighted.

### Season

Track your entire FPL season from any gameweek. Enter your Manager ID and click **Start Season** to backfill your full history from the FPL API.

- **Overview** — Rank trajectory, points-per-GW, and budget evolution charts. Summary cards for rank, points, team value, bank, and free transfers. Current squad displayed with player badges.
- **Weekly Workflow** — Generate transfer/captain/chip recommendations using the MILP solver. Includes a 2-week lookahead comparing bank-vs-use strategies for free transfers. After the gameweek, record actual results to compare against recommendations.
- **Fixtures** — FDR-coloured grid showing upcoming opponents for all 20 teams. Double and blank gameweeks highlighted automatically.
- **Transfer History** — Complete log of all transfers with cost, hit points, and whether the recommendation was followed.
- **Chips** — Tracks which chips you've used and when. Estimates the point value of each remaining chip.
- **Prices** — Daily price snapshots for squad players. Alerts for players likely to rise or fall based on transfer volume.

### Feature Insights

View model training status (date, feature count per position), feature importance charts from feature selection, and XGBoost model importance charts. Includes detailed text reports for both.

### Backtest & GW Compare

Available from the action bar. Backtest tests model accuracy across historical gameweeks. GW Compare lets you compare a locked-in team against the best possible team for a past gameweek.

---

## CLI (Advanced)

```bash
python -m src.predict                          # predictions only
python -m src.predict --train --tune           # train models then predict
python -m src.predict --feature-selection      # run feature selection first
python -m src.predict --force-fetch            # force re-fetch all data
```
