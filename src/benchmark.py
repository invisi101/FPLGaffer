"""Head-to-head benchmark: xtifpl2 (current) vs xtifpl (enhanced ensemble).

Tests multiple cutoff points to compare model accuracy across different
game-week windows. For each cutoff, trains on GW1-N, predicts GW(N+1)-(N+3),
and compares to actual results.

Usage:
    python -m src.benchmark
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.data_fetcher import load_all_data
from src.feature_engineering import build_features
from src.model import CURRENT_SEASON, DEFAULT_FEATURES, POSITION_GROUPS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CUTOFF_GWS = [10, 13, 15, 17, 19, 22]  # multiple windows
TARGET = "next_3gw_points"
TOP_N = 20


# ---------------------------------------------------------------------------
# Extra feature engineering for xtifpl model
# ---------------------------------------------------------------------------

def add_xtifpl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all xtifpl-specific features to the full dataframe at once."""
    df = df.copy()
    df = df.sort_values(["player_id", "season", "gameweek"]).reset_index(drop=True)

    # 1. EWM features
    ewm_stats = [
        "player_xg_last3", "player_xa_last3", "player_xgot_last3",
        "player_chances_created_last3", "player_shots_on_target_last3",
    ]
    ewm_cols = {}
    for col in ewm_stats:
        if col not in df.columns:
            continue
        ewm_cols[f"ewm_{col}"] = (
            df.groupby("player_id")[col]
            .transform(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
        )
    if ewm_cols:
        df = pd.concat([df, pd.DataFrame(ewm_cols, index=df.index)], axis=1)

    # 2. Team form
    if "event_points" in df.columns and "team_code" in df.columns:
        team_pts = (
            df.groupby(["team_code", "season", "gameweek"])["event_points"]
            .sum()
            .reset_index()
            .sort_values(["team_code", "season", "gameweek"])
        )
        team_pts["team_form_5"] = (
            team_pts.groupby("team_code")["event_points"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        df = df.merge(
            team_pts[["team_code", "season", "gameweek", "team_form_5"]],
            on=["team_code", "season", "gameweek"], how="left",
        )

    # 3. Opponent quality
    if "opponent_elo" in df.columns and "fdr" in df.columns:
        df["opp_quality"] = (df["opponent_elo"].fillna(1500) - 1500) / 200 + (df["fdr"].fillna(3) - 3) / 2

    # 4. Fixture congestion
    if "team_code" in df.columns:
        team_gws = (
            df[["team_code", "season", "gameweek"]]
            .drop_duplicates()
            .sort_values(["team_code", "season", "gameweek"])
        )
        team_gws["fixture_congestion"] = (
            team_gws.groupby("team_code")["gameweek"]
            .transform(lambda s: s.diff().rolling(3, min_periods=1).mean())
        )
        team_gws["fixture_congestion"] = 1.0 / (team_gws["fixture_congestion"].fillna(1) + 0.1)
        df = df.merge(
            team_gws[["team_code", "season", "gameweek", "fixture_congestion"]],
            on=["team_code", "season", "gameweek"], how="left",
        )

    # 5. Injury return flag
    if "chance_of_playing" in df.columns:
        prev_cop = df.groupby("player_id")["chance_of_playing"].shift(1)
        df["injury_return"] = (
            (prev_cop.fillna(100) < 75) & (df["chance_of_playing"].fillna(100) >= 75)
        ).astype(int)

    # 6. Season weight (previous season decay)
    df["season_weight"] = np.where(df["season"] == CURRENT_SEASON, 1.0, 0.5)

    return df


def get_xtifpl_features(position: str) -> list[str]:
    """Extended feature set for xtifpl model."""
    base = list(DEFAULT_FEATURES.get(position, DEFAULT_FEATURES["MID"]))
    extra = [
        "ewm_player_xg_last3", "ewm_player_xa_last3", "ewm_player_xgot_last3",
        "ewm_player_chances_created_last3", "ewm_player_shots_on_target_last3",
        "team_form_5", "opp_quality", "fixture_congestion", "injury_return",
    ]
    return base + extra


# ---------------------------------------------------------------------------
# Model A: xtifpl2 (current) — per-position XGBoost
# ---------------------------------------------------------------------------

def train_and_predict_xtifpl2(train_df: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Train the current xtifpl2 model, predict on snapshot."""
    predictions = []

    for position in POSITION_GROUPS:
        feature_cols = DEFAULT_FEATURES.get(position, DEFAULT_FEATURES["MID"])
        available = [c for c in feature_cols if c in train_df.columns]

        pos_train = train_df[train_df["position_clean"] == position].copy()
        pos_train = pos_train.dropna(subset=[TARGET])
        pos_train = pos_train.dropna(subset=available, thresh=len(available) // 2)
        for c in available:
            pos_train[c] = pos_train[c].fillna(0)

        if len(pos_train) < 50:
            continue

        model = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        model.fit(pos_train[available].values, pos_train[TARGET].values)

        pos_snap = snap[snap["position_clean"] == position].copy()
        if pos_snap.empty:
            continue
        for c in available:
            pos_snap[c] = pos_snap[c].fillna(0)

        pos_snap["xtifpl2_pred"] = model.predict(pos_snap[available].values)
        predictions.append(pos_snap[["player_id", "xtifpl2_pred"]])

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


# ---------------------------------------------------------------------------
# Model B: xtifpl — per-position XGB+LightGBM+Ridge ensemble
# ---------------------------------------------------------------------------

def train_and_predict_xtifpl(train_df: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Train the xtifpl ensemble, predict on snapshot."""
    if lgb is None:
        raise ImportError("lightgbm is required for xtifpl benchmark. Install with: pip install lightgbm")
    predictions = []

    for position in POSITION_GROUPS:
        feature_cols = get_xtifpl_features(position)
        available = [c for c in feature_cols if c in train_df.columns]

        pos_train = train_df[train_df["position_clean"] == position].copy()
        pos_train = pos_train.dropna(subset=[TARGET])
        pos_train = pos_train.dropna(subset=available, thresh=len(available) // 2)
        for c in available:
            pos_train[c] = pos_train[c].fillna(0)

        if len(pos_train) < 50:
            continue

        X_train = pos_train[available].values
        y_train = pos_train[TARGET].values
        weights = pos_train["season_weight"].values if "season_weight" in pos_train.columns else None

        xgb_model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train, sample_weight=weights)

        lgb_model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42, verbosity=-1,
        )
        lgb_model.fit(X_train, y_train, sample_weight=weights)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_scaled, y_train)

        pos_snap = snap[snap["position_clean"] == position].copy()
        if pos_snap.empty:
            continue
        for c in available:
            pos_snap[c] = pos_snap[c].fillna(0)

        X_pred = pos_snap[available].values
        pos_snap["xtifpl_pred"] = (
            0.4 * xgb_model.predict(X_pred)
            + 0.4 * lgb_model.predict(X_pred)
            + 0.2 * ridge_model.predict(scaler.transform(X_pred))
        )
        predictions.append(pos_snap[["player_id", "xtifpl_pred"]])

    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


# ---------------------------------------------------------------------------
# Per-window evaluation (returns metrics dict)
# ---------------------------------------------------------------------------

def compute_metrics(merged: pd.DataFrame, actual_col: str = "actual_pts") -> dict:
    """Compute all comparison metrics for one cutoff window."""
    top_v2 = merged.nlargest(TOP_N, "xtifpl2_pred")
    top_v1 = merged.nlargest(TOP_N, "xtifpl_pred")
    top_actual = merged.nlargest(TOP_N, actual_col)
    actual_ids = set(top_actual["player_id"])

    actual_top_df = merged[merged["player_id"].isin(actual_ids)]

    valid = merged.dropna(subset=[actual_col, "xtifpl2_pred", "xtifpl_pred"])

    return {
        "mae_own_top20_v2": mean_absolute_error(top_v2[actual_col], top_v2["xtifpl2_pred"]),
        "mae_own_top20_v1": mean_absolute_error(top_v1[actual_col], top_v1["xtifpl_pred"]),
        "mae_actual_top20_v2": mean_absolute_error(actual_top_df[actual_col], actual_top_df["xtifpl2_pred"]),
        "mae_actual_top20_v1": mean_absolute_error(actual_top_df[actual_col], actual_top_df["xtifpl_pred"]),
        "mae_all_v2": mean_absolute_error(valid[actual_col], valid["xtifpl2_pred"]),
        "mae_all_v1": mean_absolute_error(valid[actual_col], valid["xtifpl_pred"]),
        "overlap_v2": len(actual_ids & set(top_v2["player_id"])),
        "overlap_v1": len(actual_ids & set(top_v1["player_id"])),
        "corr_v2": valid[["xtifpl2_pred", actual_col]].corr().iloc[0, 1],
        "corr_v1": valid[["xtifpl_pred", actual_col]].corr().iloc[0, 1],
        "n_players": len(valid),
        "top_v2": top_v2,
        "top_v1": top_v1,
        "top_actual": top_actual,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_window_detail(cutoff: int, merged: pd.DataFrame, metrics: dict, actual_col: str = "actual_pts"):
    """Print the side-by-side top-20 table for one window."""
    top_actual = metrics["top_actual"]
    top_v2 = metrics["top_v2"]
    top_v1 = metrics["top_v1"]

    print(f"\n{'─' * 110}")
    print(f"  {'ACTUAL TOP 20':^34s} │ {'XTIFPL2 TOP 20':^34s} │ {'XTIFPL TOP 20':^34s}")
    print(f"{'─' * 110}")
    print(f"  {'Player':<18s} {'Pts':>5s} {'Pos':>4s} │ {'Player':<18s} {'Pred':>5s} {'Pos':>4s} │ {'Player':<18s} {'Pred':>5s} {'Pos':>4s}")
    print(f"{'─' * 110}")

    for i in range(TOP_N):
        a = top_actual.iloc[i] if i < len(top_actual) else None
        b = top_v2.iloc[i] if i < len(top_v2) else None
        c = top_v1.iloc[i] if i < len(top_v1) else None

        a_s = f"  {a['web_name']:<18s} {a[actual_col]:>5.1f} {a['position_clean']:>4s}" if a is not None else f"  {'':18s} {'':>5s} {'':>4s}"
        b_s = f"{b['web_name']:<18s} {b['xtifpl2_pred']:>5.1f} {b['position_clean']:>4s}" if b is not None else f"{'':18s} {'':>5s} {'':>4s}"
        c_s = f"{c['web_name']:<18s} {c['xtifpl_pred']:>5.1f} {c['position_clean']:>4s}" if c is not None else f"{'':18s} {'':>5s} {'':>4s}"

        print(f"{a_s} │ {b_s} │ {c_s}")

    print(f"{'─' * 110}")


def print_window_metrics(cutoff: int, m: dict):
    """Print compact metrics for one window."""
    def winner(v1, v2, direction):
        if direction == "lower":
            return "xtifpl2" if v1 < v2 else ("xtifpl" if v2 < v1 else "TIE")
        return "xtifpl2" if v1 > v2 else ("xtifpl" if v2 > v1 else "TIE")

    rows = [
        ("MAE (own top 20)", m["mae_own_top20_v2"], m["mae_own_top20_v1"], "lower"),
        ("MAE (actual top 20)", m["mae_actual_top20_v2"], m["mae_actual_top20_v1"], "lower"),
        ("MAE (all players)", m["mae_all_v2"], m["mae_all_v1"], "lower"),
        (f"Overlap w/ actual top {TOP_N}", m["overlap_v2"], m["overlap_v1"], "higher"),
        ("Correlation", m["corr_v2"], m["corr_v1"], "higher"),
    ]

    print(f"\n  {'Metric':<30s} {'xtifpl2':>10s} {'xtifpl':>10s} {'Winner':>10s}")
    print(f"  {'─' * 60}")
    for name, v1, v2, d in rows:
        w = winner(v1, v2, d)
        if isinstance(v1, int):
            print(f"  {name:<30s} {v1:>10d} {v2:>10d} {w:>10s}")
        else:
            print(f"  {name:<30s} {v1:>10.4f} {v2:>10.4f} {w:>10s}")


def print_aggregate(all_metrics: dict):
    """Print aggregated results across all windows."""
    cutoffs = sorted(all_metrics.keys())
    n = len(cutoffs)

    print("\n" + "=" * 110)
    print(f"  AGGREGATE RESULTS ACROSS {n} WINDOWS (cutoffs: {', '.join(f'GW{c}' for c in cutoffs)})")
    print("=" * 110)

    # Per-window summary table
    print(f"\n  {'Window':<12s} │ {'MAE all (v2)':>12s} {'MAE all (v1)':>12s} │ {'Overlap v2':>10s} {'Overlap v1':>10s} │ {'Corr v2':>8s} {'Corr v1':>8s}")
    print(f"  {'─' * 90}")
    for c in cutoffs:
        m = all_metrics[c]
        print(f"  GW{c:<3d}→{c+1}-{c+3} │ {m['mae_all_v2']:>12.4f} {m['mae_all_v1']:>12.4f} │ {m['overlap_v2']:>10d} {m['overlap_v1']:>10d} │ {m['corr_v2']:>8.4f} {m['corr_v1']:>8.4f}")

    # Averages
    def avg(key):
        return np.mean([all_metrics[c][key] for c in cutoffs])

    def win_count(key1, key2, direction):
        v2_wins = sum(1 for c in cutoffs
                      if (all_metrics[c][key1] < all_metrics[c][key2]) == (direction == "lower"))
        v1_wins = n - v2_wins
        return v2_wins, v1_wins

    print(f"  {'─' * 90}")
    print(f"  {'AVERAGE':<12s} │ {avg('mae_all_v2'):>12.4f} {avg('mae_all_v1'):>12.4f} │ {avg('overlap_v2'):>10.1f} {avg('overlap_v1'):>10.1f} │ {avg('corr_v2'):>8.4f} {avg('corr_v1'):>8.4f}")

    # Win/loss tally
    print(f"\n  {'─' * 70}")
    print(f"  {'Metric':<35s} {'xtifpl2 wins':>12s} {'xtifpl wins':>12s} {'Avg diff':>10s}")
    print(f"  {'─' * 70}")

    tally_rows = [
        ("MAE (own top 20)", "mae_own_top20_v2", "mae_own_top20_v1", "lower"),
        ("MAE (actual top 20)", "mae_actual_top20_v2", "mae_actual_top20_v1", "lower"),
        ("MAE (all players)", "mae_all_v2", "mae_all_v1", "lower"),
        (f"Overlap w/ actual top {TOP_N}", "overlap_v2", "overlap_v1", "higher"),
        ("Correlation", "corr_v2", "corr_v1", "higher"),
    ]

    for name, k1, k2, direction in tally_rows:
        if direction == "lower":
            w2 = sum(1 for c in cutoffs if all_metrics[c][k1] < all_metrics[c][k2])
            w1 = sum(1 for c in cutoffs if all_metrics[c][k2] < all_metrics[c][k1])
        else:
            w2 = sum(1 for c in cutoffs if all_metrics[c][k1] > all_metrics[c][k2])
            w1 = sum(1 for c in cutoffs if all_metrics[c][k2] > all_metrics[c][k1])

        diff = avg(k1) - avg(k2)
        sign = "+" if diff > 0 else ""
        if isinstance(all_metrics[cutoffs[0]][k1], int):
            print(f"  {name:<35s} {w2:>12d} {w1:>12d} {sign}{diff:>9.1f}")
        else:
            print(f"  {name:<35s} {w2:>12d} {w1:>12d} {sign}{diff:>9.4f}")

    # Overall verdict
    total_v2 = sum(
        sum(1 for c in cutoffs if all_metrics[c][k1] < all_metrics[c][k2])
        if d == "lower" else
        sum(1 for c in cutoffs if all_metrics[c][k1] > all_metrics[c][k2])
        for _, k1, k2, d in tally_rows
    )
    total_v1 = sum(
        sum(1 for c in cutoffs if all_metrics[c][k2] < all_metrics[c][k1])
        if d == "lower" else
        sum(1 for c in cutoffs if all_metrics[c][k2] > all_metrics[c][k1])
        for _, k1, k2, d in tally_rows
    )

    print(f"\n  Total metric-window wins: xtifpl2 {total_v2}, xtifpl {total_v1}")
    if total_v2 > total_v1:
        print(f"  >>> xtifpl2 (current model) is the overall winner")
    elif total_v1 > total_v2:
        print(f"  >>> xtifpl (enhanced ensemble) is the overall winner")
    else:
        print(f"  >>> Models are tied overall")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("  FPL Model Benchmark: xtifpl2 vs xtifpl")
    print(f"  Cutoff windows: {', '.join(f'GW{c}' for c in CUTOFF_GWS)}")
    print(f"  Each window: train on GW1-N, predict GW(N+1)-(N+3)")
    print("=" * 70)

    # 1. Load and build features once
    print("\n[1/3] Loading data and building features...")
    data = load_all_data()
    df = build_features(data)
    print(f"  Base feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")

    print("  Adding xtifpl extra features...")
    df = add_xtifpl_features(df)
    print(f"  Augmented feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")

    # 2. Run each cutoff window
    print(f"\n[2/3] Running {len(CUTOFF_GWS)} benchmark windows...")
    all_metrics = {}

    for cutoff in CUTOFF_GWS:
        print(f"\n{'=' * 110}")
        print(f"  WINDOW: Train GW1-{cutoff}, Predict GW{cutoff+1}-{cutoff+3}")
        print(f"{'=' * 110}")

        # Split
        train_mask = (df["season"] != CURRENT_SEASON) | (df["gameweek"] < cutoff)
        train_df = df[train_mask].copy()

        snap = df[
            (df["season"] == CURRENT_SEASON) & (df["gameweek"] == cutoff)
        ].drop_duplicates(subset=["player_id"], keep="first").copy()

        n_actuals = snap[TARGET].notna().sum()
        print(f"  Training: {len(train_df)} rows | Snapshot: {len(snap)} players | With actuals: {n_actuals}")

        if n_actuals == 0:
            print(f"  SKIPPED: no actuals for GW{cutoff+1}-{cutoff+3}")
            continue

        # Train both models
        print(f"  Training xtifpl2...")
        preds_v2 = train_and_predict_xtifpl2(train_df, snap)
        print(f"  Training xtifpl...")
        preds_v1 = train_and_predict_xtifpl(train_df, snap)

        if preds_v2.empty or preds_v1.empty:
            print(f"  SKIPPED: missing predictions")
            continue

        # Merge
        merged = snap[["player_id", "web_name", "position_clean", TARGET]].copy()
        merged = merged.rename(columns={TARGET: "actual_pts"})
        merged = merged.merge(preds_v2[["player_id", "xtifpl2_pred"]], on="player_id", how="left")
        merged = merged.merge(preds_v1[["player_id", "xtifpl_pred"]], on="player_id", how="left")
        merged = merged.dropna(subset=["actual_pts", "xtifpl2_pred", "xtifpl_pred"])

        print(f"  Players with all data: {len(merged)}")

        metrics = compute_metrics(merged)
        all_metrics[cutoff] = metrics

        # Print detail for this window
        print_window_detail(cutoff, merged, metrics)
        print_window_metrics(cutoff, metrics)

    # 3. Aggregate
    print(f"\n[3/3] Aggregating results...")
    if all_metrics:
        print_aggregate(all_metrics)
    else:
        print("  No valid windows to aggregate.")


if __name__ == "__main__":
    main()
