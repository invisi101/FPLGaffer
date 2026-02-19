"""Entry point: fetch latest data, run model, output predictions."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_fetcher import load_all_data
from src.feature_engineering import build_features, get_feature_columns, get_fixture_context
from src.feature_selection import run_feature_selection
from src.model import (
    CURRENT_SEASON,
    POSITION_GROUPS,
    SUB_MODELS_FOR_POSITION,
    load_model,
    load_sub_model,
    predict_decomposed,
    predict_for_position,
    train_all_models,
    train_all_quantile_models,
    train_all_sub_models,
)

if getattr(sys, "frozen", False):
    _BASE = Path(sys.executable).parent
else:
    _BASE = Path(__file__).resolve().parent.parent

OUTPUT_DIR = _BASE / "output"


def get_latest_gw(df: pd.DataFrame, season: str = CURRENT_SEASON) -> int:
    """Find the latest gameweek in the dataset for the current season."""
    season_df = df[df["season"] == season]
    if season_df.empty:
        return int(df["gameweek"].max())
    return int(season_df["gameweek"].max())


def _build_offset_snapshot(
    current: pd.DataFrame,
    df: pd.DataFrame,
    target_gw: int,
    fixture_map: pd.DataFrame,
    fdr_map: pd.DataFrame,
    elo: pd.DataFrame,
    opp_rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Build a prediction snapshot with opponent data for a specific future GW.

    Takes the current GW snapshot (player features) and swaps in fixture/opponent
    columns for ``target_gw`` so the 1-GW model can predict that future GW.
    """
    # Columns that are fixture-specific and must be replaced
    fixture_cols = [
        "opponent_code", "is_home", "fdr", "opponent_elo",
        "next_gw_fixture_count",
        # Multi-GW lookahead features (stale for offset > 1, must be recomputed)
        "avg_fdr_next3", "home_pct_next3", "avg_opponent_elo_next3",
        # Interaction features (will be recomputed)
        "xg_x_opp_goals_conceded", "chances_x_opp_big_chances",
        "cs_opportunity", "venue_matched_form",
    ]
    opp_cols = [c for c in current.columns if c.startswith("opp_")]
    # Also drop vs_opponent_* — these are keyed by opponent and would be
    # misattributed to the wrong fixture rows for DGW players
    vs_opp_cols = [c for c in current.columns if c.startswith("vs_opponent_")]
    drop_cols = [c for c in fixture_cols + opp_cols + vs_opp_cols if c in current.columns]

    # Deduplicate to one row per player (drop DGW fixture splits)
    snapshot = current.drop(columns=drop_cols).drop_duplicates(
        subset=["player_id"], keep="first"
    ).copy()

    # Look up fixtures for the target GW
    gw_fixtures = fixture_map[fixture_map["gameweek"] == target_gw][
        ["team_code", "opponent_code", "is_home"]
    ].copy()

    if gw_fixtures.empty:
        return pd.DataFrame()

    # Count fixtures per team in this GW (DGW detection)
    fx_counts = gw_fixtures.groupby("team_code").size().reset_index(
        name="next_gw_fixture_count"
    )

    # Merge fixtures — DGW teams get multiple rows (one per fixture)
    snapshot = snapshot.merge(gw_fixtures, on="team_code", how="inner")
    snapshot = snapshot.merge(fx_counts, on="team_code", how="left")
    snapshot["next_gw_fixture_count"] = snapshot["next_gw_fixture_count"].fillna(1).astype(int)

    if snapshot.empty:
        return pd.DataFrame()

    # Add FDR per fixture
    if not fdr_map.empty and "team_code" in fdr_map.columns and "opponent_code" in fdr_map.columns:
        fdr_lookup = fdr_map[["team_code", "gameweek", "opponent_code", "fdr"]].copy()
        fdr_lookup = fdr_lookup.dropna(subset=["opponent_code"])
        fdr_lookup["opponent_code"] = fdr_lookup["opponent_code"].astype(int)
        fdr_gw = fdr_lookup[fdr_lookup["gameweek"] == target_gw].drop(columns=["gameweek"])
        fdr_gw = fdr_gw.drop_duplicates(subset=["team_code", "opponent_code"], keep="first")
        snapshot = snapshot.merge(fdr_gw, on=["team_code", "opponent_code"], how="left")
    if "fdr" not in snapshot.columns:
        snapshot["fdr"] = 3.0
    snapshot["fdr"] = snapshot["fdr"].fillna(3.0)

    # Add opponent Elo
    if not elo.empty and "team_code" in elo.columns:
        opp_elo = elo.rename(columns={"team_code": "opponent_code", "team_elo": "opponent_elo"})
        snapshot = snapshot.merge(opp_elo, on="opponent_code", how="left")
    if "opponent_elo" not in snapshot.columns:
        snapshot["opponent_elo"] = 1500.0
    snapshot["opponent_elo"] = snapshot["opponent_elo"].fillna(1500.0)

    # Add opponent rolling features — use latest available per opponent team
    if not opp_rolling.empty:
        opp_feats = opp_rolling.rename(columns={"team_code": "opponent_code"})
        # Get the latest available GW per opponent
        opp_latest = (
            opp_feats.sort_values("gameweek")
            .drop_duplicates(subset=["opponent_code"], keep="last")
            .drop(columns=["gameweek"])
        )
        snapshot = snapshot.merge(opp_latest, on="opponent_code", how="left")

    # Look up vs_opponent_* history from df for the new opponents
    if vs_opp_cols:
        # Extract the latest vs_opponent record per (player_id, opponent_code)
        vs_src_cols = ["player_id", "opponent_code"] + vs_opp_cols
        vs_src = df[[c for c in vs_src_cols if c in df.columns]].copy()
        vs_src = vs_src.dropna(subset=["opponent_code"])
        vs_src = vs_src.dropna(subset=[c for c in vs_opp_cols if c in vs_src.columns], how="all")
        if not vs_src.empty:
            vs_src["opponent_code"] = vs_src["opponent_code"].astype(int)
            snapshot["opponent_code"] = snapshot["opponent_code"].astype(int)
            vs_latest = vs_src.drop_duplicates(
                subset=["player_id", "opponent_code"], keep="last"
            )
            snapshot = snapshot.merge(
                vs_latest, on=["player_id", "opponent_code"], how="left"
            )

    # Recompute interaction features
    if "player_xg_last3" in snapshot.columns and "opp_goals_conceded_last3" in snapshot.columns:
        snapshot["xg_x_opp_goals_conceded"] = (
            snapshot["player_xg_last3"] * snapshot["opp_goals_conceded_last3"]
        )
    if "player_chances_created_last3" in snapshot.columns and "opp_big_chances_allowed_last3" in snapshot.columns:
        snapshot["chances_x_opp_big_chances"] = (
            snapshot["player_chances_created_last3"] * snapshot["opp_big_chances_allowed_last3"]
        )
    if "opp_opponent_xg_last3" in snapshot.columns:
        snapshot["cs_opportunity"] = 1.0 / (snapshot["opp_opponent_xg_last3"] + 0.1)
    if "home_xg_form" in snapshot.columns and "away_xg_form" in snapshot.columns:
        snapshot["venue_matched_form"] = np.where(
            snapshot["is_home"] == 1, snapshot["home_xg_form"], snapshot["away_xg_form"]
        )

    # Recompute multi-GW lookahead features relative to the target GW
    ahead_gws = fixture_map[
        (fixture_map["gameweek"] > target_gw) & (fixture_map["gameweek"] <= target_gw + 3)
    ]
    if not ahead_gws.empty:
        for tc in snapshot["team_code"].unique():
            team_ahead = ahead_gws[ahead_gws["team_code"] == tc]
            mask = snapshot["team_code"] == tc
            if not team_ahead.empty:
                snapshot.loc[mask, "avg_fdr_next3"] = team_ahead.merge(
                    fdr_map[fdr_map["gameweek"].isin(team_ahead["gameweek"].unique())][
                        ["team_code", "gameweek", "opponent_code", "fdr"]
                    ].dropna(subset=["fdr"]),
                    on=["team_code", "gameweek", "opponent_code"], how="left"
                )["fdr"].mean()
                snapshot.loc[mask, "home_pct_next3"] = team_ahead["is_home"].mean()
                if not elo.empty and "team_code" in elo.columns:
                    opp_elo_map = dict(zip(elo["team_code"], elo["team_elo"]))
                    snapshot.loc[mask, "avg_opponent_elo_next3"] = (
                        team_ahead["opponent_code"].map(opp_elo_map).mean()
                    )
            else:
                snapshot.loc[mask, "avg_fdr_next3"] = 3.0
                snapshot.loc[mask, "home_pct_next3"] = 0.5
                snapshot.loc[mask, "avg_opponent_elo_next3"] = 1500.0
    # Fill any remaining NaN lookahead features with neutral defaults
    snapshot["avg_fdr_next3"] = snapshot.get("avg_fdr_next3", pd.Series(3.0, index=snapshot.index)).fillna(3.0)
    snapshot["home_pct_next3"] = snapshot.get("home_pct_next3", pd.Series(0.5, index=snapshot.index)).fillna(0.5)
    snapshot["avg_opponent_elo_next3"] = snapshot.get("avg_opponent_elo_next3", pd.Series(1500.0, index=snapshot.index)).fillna(1500.0)

    return snapshot


ENSEMBLE_WEIGHT_DECOMPOSED = 0.5


def _ensemble_predict_position(snapshot: pd.DataFrame, position: str) -> pd.DataFrame:
    """Predict for a position using the 50/50 ensemble blend.

    Blends decomposed sub-models with mean regression when both available.
    Falls back to whichever exists alone.  Returns DataFrame with at least
    player_id and predicted_next_gw_points columns, or empty DataFrame.
    """
    pred_col = "predicted_next_gw_points"

    components = SUB_MODELS_FOR_POSITION.get(position, [])
    has_sub = all(
        load_sub_model(position, comp) is not None for comp in components
    ) if components else False

    decomp_preds = predict_decomposed(snapshot, position) if has_sub else pd.DataFrame()

    model_dict = load_model(position, "next_gw_points")
    mean_preds = (
        predict_for_position(snapshot, position, "next_gw_points", model_dict)
        if model_dict is not None else pd.DataFrame()
    )

    if not decomp_preds.empty and not mean_preds.empty:
        w_d = ENSEMBLE_WEIGHT_DECOMPOSED
        w_m = 1 - w_d
        merged = decomp_preds[["player_id", pred_col]].merge(
            mean_preds[["player_id", pred_col]],
            on="player_id", suffixes=("_decomp", "_mean"),
        )
        merged[pred_col] = (
            w_d * merged[f"{pred_col}_decomp"]
            + w_m * merged[f"{pred_col}_mean"]
        )
        meta_cols = [c for c in decomp_preds.columns if c != pred_col]
        return decomp_preds[meta_cols].merge(merged[["player_id", pred_col]], on="player_id")
    elif not decomp_preds.empty:
        return decomp_preds
    elif not mean_preds.empty:
        return mean_preds
    return pd.DataFrame()


def _predict_next_3gw_from_1gw(
    current: pd.DataFrame,
    df: pd.DataFrame,
    fixture_context: dict,
    latest_gw: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Predict next-3-GW points by summing three 1-GW predictions.

    For each of the next 3 GWs, builds a snapshot with that GW's opponent data
    and runs the 1-GW ensemble model.  Returns a tuple of:
      - DataFrame with player_id and predicted_next_3gw_points (the sum)
      - List of per-GW detail dicts (for predictions_detail.json export)
    """
    fixture_map = fixture_context["fixture_map"]
    fdr_map = fixture_context["fdr_map"]
    elo = fixture_context["elo"]
    opp_rolling = fixture_context["opp_rolling"]

    per_gw_preds = []
    per_gw_detail = []  # [{gw, player_id, predicted_pts, opponent_code, is_home, fdr}]

    for offset in range(1, 4):
        target_gw = latest_gw + offset
        snapshot = _build_offset_snapshot(
            current, df, target_gw, fixture_map, fdr_map, elo, opp_rolling,
        )
        if snapshot.empty:
            print(f"  GW{target_gw}: no fixtures found, skipping")
            continue

        gw_preds = []
        for position in POSITION_GROUPS:
            preds = _ensemble_predict_position(snapshot, position)
            if not preds.empty:
                gw_preds.append(preds[["player_id", "predicted_next_gw_points"]].copy())

        if gw_preds:
            gw_df = pd.concat(gw_preds, ignore_index=True)
            gw_df = gw_df.rename(columns={"predicted_next_gw_points": f"pred_gw{target_gw}"})
            per_gw_preds.append(gw_df)
            n_players = len(gw_df)
            avg_pts = gw_df[f"pred_gw{target_gw}"].mean()
            print(f"  GW{target_gw}: {n_players} players, avg {avg_pts:.2f} pts")

            # Collect fixture info for each player in this GW
            fixture_info = snapshot[["player_id", "team_code"]].drop_duplicates(
                subset=["player_id"], keep="first"
            )
            gw_fixtures = fixture_map[fixture_map["gameweek"] == target_gw][
                ["team_code", "opponent_code", "is_home"]
            ].drop_duplicates()
            # Get FDR
            fdr_gw = fdr_map[fdr_map["gameweek"] == target_gw][
                ["team_code", "opponent_code", "fdr"]
            ].drop_duplicates() if not fdr_map.empty else pd.DataFrame()
            if not fdr_gw.empty:
                gw_fixtures = gw_fixtures.merge(
                    fdr_gw, on=["team_code", "opponent_code"], how="left"
                )
            else:
                gw_fixtures["fdr"] = 3

            fixture_info = fixture_info.merge(gw_fixtures, on="team_code", how="left")
            # Merge predicted points
            fixture_info = fixture_info.merge(
                gw_df.rename(columns={f"pred_gw{target_gw}": "predicted_pts"}),
                on="player_id", how="left"
            )
            fixture_info["gw"] = target_gw
            per_gw_detail.append(fixture_info)

    if not per_gw_preds:
        return pd.DataFrame(columns=["player_id", "predicted_next_3gw_points"]), []

    # Merge all per-GW predictions and sum
    merged = per_gw_preds[0]
    for extra in per_gw_preds[1:]:
        merged = merged.merge(extra, on="player_id", how="outer")

    pred_cols = [c for c in merged.columns if c.startswith("pred_gw")]
    merged["predicted_next_3gw_points"] = merged[pred_cols].sum(axis=1)

    return merged[["player_id", "predicted_next_3gw_points"]], per_gw_detail


def predict_future_gw(
    current: pd.DataFrame,
    df: pd.DataFrame,
    fixture_context: dict,
    latest_gw: int,
    target_gw: int,
) -> pd.DataFrame:
    """Predict points for a single future GW using the 1-GW model.

    Returns DataFrame with player_id, predicted_points, confidence columns.
    Applies confidence decay of 0.95^offset to account for increasing uncertainty.
    """
    offset = target_gw - latest_gw
    if offset < 1:
        return pd.DataFrame(columns=["player_id", "predicted_points", "confidence"])

    fixture_map = fixture_context["fixture_map"]
    fdr_map = fixture_context["fdr_map"]
    elo = fixture_context["elo"]
    opp_rolling = fixture_context["opp_rolling"]

    snapshot = _build_offset_snapshot(
        current, df, target_gw, fixture_map, fdr_map, elo, opp_rolling,
    )
    if snapshot.empty:
        return pd.DataFrame(columns=["player_id", "predicted_points", "confidence"])

    gw_preds = []
    for position in POSITION_GROUPS:
        preds = _ensemble_predict_position(snapshot, position)
        if not preds.empty:
            gw_preds.append(preds[["player_id", "predicted_next_gw_points"]].copy())

    if not gw_preds:
        return pd.DataFrame(columns=["player_id", "predicted_points", "confidence"])

    result = pd.concat(gw_preds, ignore_index=True)

    # Apply confidence decay — offset 1 (immediate next GW) gets no discount,
    # consistent with the standard 1-GW prediction path.
    confidence = 0.95 ** (offset - 1)
    result["predicted_points"] = result["predicted_next_gw_points"] * confidence
    result["confidence"] = confidence
    result = result.drop(columns=["predicted_next_gw_points"])

    return result


def predict_future_range(
    current: pd.DataFrame,
    df: pd.DataFrame,
    fixture_context: dict,
    latest_gw: int,
    horizon: int = 8,
) -> dict[int, pd.DataFrame]:
    """Predict points for GW+1 through GW+horizon.

    Returns {gw: DataFrame} where each DataFrame has
    player_id, predicted_points, confidence columns.
    """
    predictions = {}
    for offset in range(1, horizon + 1):
        target_gw = latest_gw + offset
        if target_gw > 38:
            break
        gw_df = predict_future_gw(
            current, df, fixture_context, latest_gw, target_gw,
        )
        if not gw_df.empty:
            predictions[target_gw] = gw_df
            n = len(gw_df)
            avg = gw_df["predicted_points"].mean()
            conf = gw_df["confidence"].iloc[0]
            print(f"  GW{target_gw}: {n} players, avg {avg:.2f} pts (conf {conf:.2f})")

    return predictions


def build_preseason_predictions(
    df: pd.DataFrame, data: dict, bootstrap: dict,
) -> pd.DataFrame:
    """Generate predictions when no current-season data exists (pre-GW1).

    Uses last-season GW38 snapshot with cross-season feature decay,
    swaps in new-season GW1 fixtures via _build_offset_snapshot().
    Adds price-based heuristics for new players (not in last season).
    """
    from src.model import CURRENT_SEASON
    from src.data_fetcher import get_previous_season

    prev_season = get_previous_season(CURRENT_SEASON)
    prev_gw38 = df[(df["season"] == prev_season) & (df["gameweek"] == 38)]
    if prev_gw38.empty:
        # Fall back to whatever is the latest available data
        prev_gw38 = df[df["gameweek"] == df["gameweek"].max()]

    if prev_gw38.empty:
        print("  No historical data available for pre-season predictions.")
        return pd.DataFrame()

    fixture_context = get_fixture_context(data)

    # Build snapshot for GW1 using previous season's latest data
    latest_gw_in_data = int(prev_gw38["gameweek"].max())
    snapshot = _build_offset_snapshot(
        prev_gw38, df, target_gw=1,
        fixture_map=fixture_context["fixture_map"],
        fdr_map=fixture_context["fdr_map"],
        elo=fixture_context["elo"],
        opp_rolling=fixture_context["opp_rolling"],
    )

    if snapshot.empty:
        print("  Could not build pre-season snapshot (no GW1 fixtures?).")
        return pd.DataFrame()

    # Run model predictions on the snapshot (ensemble when both models available)
    all_preds = []
    for position in POSITION_GROUPS:
        pred_col = "predicted_next_gw_points"
        keep = ["player_id", "position_clean", pred_col]

        preds = _ensemble_predict_position(snapshot, position)
        if preds.empty:
            continue

        if "web_name" in preds.columns and "web_name" not in keep:
            keep.insert(1, "web_name")
        all_preds.append(preds[keep].copy())

    if not all_preds:
        print("  No pre-season predictions generated (no models?).")
        return pd.DataFrame()

    result = pd.concat(all_preds, ignore_index=True)

    # Apply cross-season decay (0.90 per GW, ~38 GWs gap)
    decay_factor = 0.90 ** 5  # Approximate 5-GW equivalent decay for season gap
    result["predicted_next_gw_points"] = result["predicted_next_gw_points"] * decay_factor

    # Handle new players from bootstrap (not in last season's data)
    known_ids = set(result["player_id"].unique())
    elements = bootstrap.get("elements", [])
    element_type_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    new_rows = []
    for el in elements:
        pid = el["id"]
        if pid not in known_ids:
            cost = el.get("now_cost", 0) / 10
            # Price-based heuristic: higher cost ~ more points historically
            heuristic_pts = (cost / 10) * 0.5
            new_rows.append({
                "player_id": pid,
                "web_name": el.get("web_name", "Unknown"),
                "position_clean": element_type_map.get(el.get("element_type"), "MID"),
                "predicted_next_gw_points": round(heuristic_pts, 2),
            })

    if new_rows:
        result = pd.concat([result, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"  Added {len(new_rows)} new players via price heuristic.")

    return result


def run_predictions(df: pd.DataFrame, data: dict | None = None) -> pd.DataFrame:
    """Generate predictions for all players for the upcoming gameweek(s)."""
    latest_gw = get_latest_gw(df)
    print(f"\nLatest gameweek in data: GW{latest_gw}")

    # Use the latest GW snapshot as the basis for prediction
    current = df[(df["season"] == CURRENT_SEASON) & (df["gameweek"] == latest_gw)].copy()
    if current.empty:
        current = df[df["gameweek"] == latest_gw].copy()

    # DGW players have multiple rows (one per fixture) — keep all rows so
    # predict_for_position can predict each match separately and sum them
    unique_players = current.drop_duplicates(subset=["player_id"], keep="first")
    dgw_count = (unique_players["next_gw_fixture_count"] > 1).sum() if "next_gw_fixture_count" in unique_players.columns else 0
    print(f"Players in current GW: {len(unique_players)} ({dgw_count} with double GW)")

    # --- 1-GW predictions (next_gw_points) ---
    # Ensemble: blend decomposed sub-models with mean regression model (50/50)
    # when both are available. Fall back to whichever exists alone.
    all_preds = []
    for position in POSITION_GROUPS:
        pred_col = "predicted_next_gw_points"
        keep = ["player_id", "position_clean", pred_col]

        preds = _ensemble_predict_position(current, position)
        if not preds.empty:
            if "web_name" in preds.columns and "web_name" not in keep:
                keep.insert(1, "web_name")
            all_preds.append(preds[keep].copy())
            print(f"  {position}: {len(preds)} players")
        else:
            print(f"  No trained model for {position}/next_gw_points")

    if not all_preds:
        print("No predictions generated.")
        return pd.DataFrame()

    result = pd.concat(all_preds, ignore_index=True)

    # --- Prediction intervals from walk-forward residuals ---
    # Uses conditional (heteroscedastic) intervals when available (Fix 5)
    intervals = []
    for position in POSITION_GROUPS:
        model_dict = load_model(position, "next_gw_points")
        if model_dict is None:
            continue
        q10 = model_dict.get("residual_q10", 0.0)
        q90 = model_dict.get("residual_q90", 0.0)
        if q10 == 0.0 and q90 == 0.0:
            continue
        pos_mask = result["position_clean"] == position
        if not pos_mask.any():
            continue
        pos_rows = result.loc[pos_mask].copy()

        bin_edges = model_dict.get("bin_edges")
        residual_bins = model_dict.get("residual_bins")
        if bin_edges and residual_bins:
            pred_vals = pos_rows["predicted_next_gw_points"].values
            bins = np.digitize(pred_vals, bin_edges)
            q10s = np.array([residual_bins.get(b, {"q10": q10})["q10"] for b in bins])
            q90s = np.array([residual_bins.get(b, {"q90": q90})["q90"] for b in bins])
            pos_rows["prediction_low"] = (pred_vals + q10s).clip(min=0)
            pos_rows["prediction_high"] = (pred_vals + q90s).clip(min=0)
        else:
            pos_rows["prediction_low"] = (pos_rows["predicted_next_gw_points"] + q10).clip(lower=0)
            pos_rows["prediction_high"] = (pos_rows["predicted_next_gw_points"] + q90).clip(lower=0)

        intervals.append(pos_rows[["player_id", "prediction_low", "prediction_high"]])
    if intervals:
        interval_df = pd.concat(intervals, ignore_index=True)
        result = result.merge(interval_df, on="player_id", how="left")

    # --- Decomposed component predictions (for detail export) ---
    component_details = {}
    for position in POSITION_GROUPS:
        components = SUB_MODELS_FOR_POSITION.get(position, [])
        has_sub = all(
            load_sub_model(position, comp) is not None for comp in components
        ) if components else False
        if has_sub:
            decomp = predict_decomposed(current, position)
            if not decomp.empty:
                sub_cols = [c for c in decomp.columns if c.startswith("sub_") or c.startswith("pts_")]
                keep_cols = ["player_id"] + sub_cols + ["p_plays", "p_60plus"]
                keep_cols = [c for c in keep_cols if c in decomp.columns]
                for _, row in decomp[keep_cols].iterrows():
                    pid = int(row["player_id"])
                    component_details[pid] = {
                        c: round(float(row[c]), 4) if pd.notna(row[c]) else 0
                        for c in keep_cols if c != "player_id"
                    }

    # --- 3-GW predictions: sum three 1-GW predictions with per-GW opponents ---
    per_gw_detail = []
    if data is not None:
        print("\n  Deriving 3-GW predictions from 1-GW model...")
        fixture_context = get_fixture_context(data)
        preds_3gw, per_gw_detail = _predict_next_3gw_from_1gw(
            current, df, fixture_context, latest_gw,
        )
        if not preds_3gw.empty:
            result = result.merge(preds_3gw, on="player_id", how="left")
    else:
        print("  Skipping 3-GW predictions (no raw data provided)")

    # --- Quantile predictions for captain scoring (MID/FWD only) ---
    q80_preds = []
    for position in ["MID", "FWD"]:
        q_model = load_model(position, "next_gw_points", suffix="_q80")
        if q_model is None:
            continue
        q_preds = predict_for_position(
            current, position, "next_gw_points", q_model, suffix="_q80"
        )
        if not q_preds.empty:
            q80_preds.append(
                q_preds[["player_id", "predicted_next_gw_points_q80"]].copy()
            )

    if q80_preds:
        q80_df = pd.concat(q80_preds, ignore_index=True)
        result = result.merge(q80_df, on="player_id", how="left")

    # Composite captain score: blend mean + quantile for upside
    if "predicted_next_gw_points_q80" in result.columns:
        result["captain_score"] = (
            0.4 * result["predicted_next_gw_points"]
            + 0.6 * result["predicted_next_gw_points_q80"].fillna(
                result["predicted_next_gw_points"]
            )
        )
    elif "predicted_next_gw_points" in result.columns:
        result["captain_score"] = result["predicted_next_gw_points"]

    # --- Availability adjustments: zero predictions for unavailable players ---
    if data is not None:
        bootstrap_elements = (
            data.get("api", {}).get("bootstrap", {}).get("elements", [])
        )
        if bootstrap_elements:
            unavailable_ids = set()
            for el in bootstrap_elements:
                status = el.get("status", "a")
                chance = el.get("chance_of_playing_next_round")
                pid = el["id"]
                # Injured, suspended, unavailable, or left the league
                if status in ("i", "s", "u", "n"):
                    unavailable_ids.add(pid)
                # Doubtful: <50% chance of playing next round
                elif chance is not None and chance < 50:
                    unavailable_ids.add(pid)

            if unavailable_ids:
                mask = result["player_id"].isin(unavailable_ids)
                n_zeroed = mask.sum()
                pred_cols = [
                    c for c in result.columns
                    if c.startswith("predicted_") or c in ("captain_score",)
                ]
                for col in pred_cols:
                    result.loc[mask, col] = 0.0
                if n_zeroed > 0:
                    print(f"  Zeroed predictions for {n_zeroed} unavailable/doubtful players")

    # Store prediction detail data for export
    _prediction_detail_store["component_details"] = component_details
    _prediction_detail_store["per_gw_detail"] = per_gw_detail
    _prediction_detail_store["latest_gw"] = latest_gw

    return result


# Module-level store for rich prediction details (populated by run_predictions)
_prediction_detail_store: dict = {}


def save_prediction_details(result: pd.DataFrame, data: dict | None = None) -> None:
    """Export predictions_detail.json with per-player component and per-GW data.

    Reads from _prediction_detail_store (populated by run_predictions) and
    the formatted result DataFrame. Overwrites the file each time.
    """
    import json

    component_details = _prediction_detail_store.get("component_details", {})
    per_gw_detail = _prediction_detail_store.get("per_gw_detail", [])
    latest_gw = _prediction_detail_store.get("latest_gw", 0)

    # Build per-GW lookup: {player_id: [{gw, predicted_pts, opponent_code, is_home, fdr}]}
    per_gw_by_player: dict[int, list] = {}
    # Build team_code->short_name map from bootstrap
    team_code_to_short = {}
    if data is not None:
        for t in data.get("api", {}).get("bootstrap", {}).get("teams", []):
            team_code_to_short[t["code"]] = t["short_name"]

    for gw_df in per_gw_detail:
        if gw_df.empty:
            continue
        for _, row in gw_df.iterrows():
            pid = int(row["player_id"])
            opp_code = int(row["opponent_code"]) if pd.notna(row.get("opponent_code")) else 0
            entry = {
                "gw": int(row["gw"]),
                "predicted_pts": round(float(row.get("predicted_pts", 0)), 2),
                "opponent_short": team_code_to_short.get(opp_code, "?"),
                "is_home": bool(row["is_home"]) if pd.notna(row.get("is_home")) else None,
                "fdr": int(row["fdr"]) if pd.notna(row.get("fdr")) else 3,
            }
            per_gw_by_player.setdefault(pid, []).append(entry)

    # Build final detail dict keyed by player_id
    detail = {}
    for _, row in result.iterrows():
        pid = int(row["player_id"])
        entry = {
            "predicted_next_gw_points": round(float(row.get("predicted_next_gw_points", 0)), 2),
            "captain_score": round(float(row.get("captain_score", 0)), 2),
            "prediction_low": round(float(row.get("prediction_low", 0)), 2) if pd.notna(row.get("prediction_low")) else None,
            "prediction_high": round(float(row.get("prediction_high", 0)), 2) if pd.notna(row.get("prediction_high")) else None,
            "predicted_next_3gw_points": round(float(row.get("predicted_next_3gw_points", 0)), 2) if pd.notna(row.get("predicted_next_3gw_points")) else None,
            "q80": round(float(row.get("predicted_next_gw_points_q80", 0)), 2) if pd.notna(row.get("predicted_next_gw_points_q80")) else None,
        }
        # Add component breakdown
        if pid in component_details:
            entry["components"] = component_details[pid]
        # Add per-GW predictions
        if pid in per_gw_by_player:
            entry["per_gw"] = sorted(per_gw_by_player[pid], key=lambda x: x["gw"])

        detail[str(pid)] = entry

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUTPUT_DIR / "predictions_detail.json"
    detail_path.write_text(json.dumps({
        "latest_gw": latest_gw,
        "players": detail,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"  Prediction details saved to {detail_path} ({len(detail)} players)")


def format_predictions(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add player metadata and format the output table."""
    latest_gw = get_latest_gw(df)
    current = df[(df["season"] == CURRENT_SEASON) & (df["gameweek"] == latest_gw)]

    # Get player info — deduplicate
    info_cols = ["player_id", "team_code", "cost", "player_form",
                 "is_home", "opponent_elo", "fdr", "ep_next",
                 "event_points", "total_points"]
    available_info = [c for c in info_cols if c in current.columns]
    player_info = current[available_info].drop_duplicates(subset="player_id", keep="first")

    result = preds.merge(player_info, on="player_id", how="left")

    # Add web_name from the feature matrix if not already present
    if "web_name" not in result.columns or result["web_name"].isna().all():
        if "web_name" in df.columns:
            name_source = df[df["season"] == CURRENT_SEASON][["player_id", "web_name"]].dropna(subset=["web_name"])
            name_source = name_source.drop_duplicates(subset="player_id", keep="last")
            if "web_name" in result.columns:
                result = result.drop(columns=["web_name"])
            result = result.merge(name_source, on="player_id", how="left")

        # Fallback: get names from bootstrap API data
        if "web_name" not in result.columns or result["web_name"].isna().all():
            import json
            bootstrap_path = _BASE / "cache" / "fpl_api_bootstrap.json"
            if bootstrap_path.exists():
                try:
                    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
                    name_map = {el["id"]: el.get("web_name", "Unknown") for el in bootstrap.get("elements", [])}
                    if "web_name" in result.columns:
                        result = result.drop(columns=["web_name"])
                    result["web_name"] = result["player_id"].map(name_map).fillna("Unknown")
                except Exception:
                    pass

        # Last resort: use player_id as name so predictions still work
        if "web_name" not in result.columns:
            result["web_name"] = result["player_id"].astype(str)

    # Add team names from API data (will be added in main if available)

    # Clean up position column
    if "position_clean" in result.columns:
        result = result.rename(columns={"position_clean": "position"})

    # Sort by predicted next GW points
    sort_col = "predicted_next_gw_points"
    if sort_col in result.columns:
        result = result.sort_values(sort_col, ascending=False)

    return result


def print_predictions(result: pd.DataFrame, top_n: int = 50):
    """Print a formatted prediction table."""
    display_cols = ["web_name", "position", "cost", "player_form",
                    "predicted_next_gw_points", "predicted_next_3gw_points",
                    "captain_score", "fdr", "is_home", "ep_next"]
    available = [c for c in display_cols if c in result.columns]

    pd.set_option("display.max_rows", top_n + 5)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    pd.set_option("display.width", 140)

    print(f"\n{'=' * 100}")
    print(f"  FPL POINTS PREDICTIONS — Top {top_n} Players")
    print(f"{'=' * 100}\n")

    top = result.head(top_n)
    print(top[available].to_string(index=False))

    if "predicted_next_gw_points" in result.columns:
        print(f"\n{'=' * 100}")
        print(f"  BY POSITION (Top 10 each)")
        print(f"{'=' * 100}")
        for pos in POSITION_GROUPS:
            pos_df = result[result["position"] == pos].head(10)
            if not pos_df.empty:
                print(f"\n  --- {pos} ---")
                print(pos_df[available].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="FPL Points Predictor")
    parser.add_argument("--train", action="store_true", help="Train models before predicting")
    parser.add_argument("--feature-selection", action="store_true", help="Run feature selection (Phase 1)")
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters during training")
    parser.add_argument("--force-fetch", action="store_true", help="Force re-fetch of all data")
    parser.add_argument("--top", type=int, default=50, help="Number of top predictions to show")
    args = parser.parse_args()

    # 1. Fetch data
    print("Step 1: Loading data...")
    data = load_all_data(force=args.force_fetch)

    # 2. Build features
    print("\nStep 2: Engineering features...")
    df = build_features(data)
    feature_cols = get_feature_columns(df)
    print(f"  Feature matrix: {df.shape[0]} rows, {len(feature_cols)} features")

    # 3. Feature selection (optional)
    if args.feature_selection:
        print("\nStep 3: Running feature selection...")
        report = run_feature_selection(df, feature_cols)
        print(report[:2000] + "\n... (full report saved to output/)")

    # 4. Train models (optional, or if no models exist)
    # Only check for 1-GW models — 3-GW predictions are derived from 1-GW
    mean_models_exist = all(
        load_model(pos, "next_gw_points") is not None
        for pos in POSITION_GROUPS
    )
    quantile_models_exist = all(
        load_model(pos, "next_gw_points", suffix="_q80") is not None
        for pos in ["MID", "FWD"]
    )
    sub_models_exist = all(
        load_sub_model(pos, comp) is not None
        for pos in POSITION_GROUPS
        for comp in SUB_MODELS_FOR_POSITION.get(pos, [])
    )
    models_exist = mean_models_exist and quantile_models_exist and sub_models_exist

    if args.train or not models_exist:
        print("\nStep 4: Training models...")
        results = train_all_models(df, tune=args.tune)
        print("\n  Training Summary (mean models):")
        for r in results:
            print(f"    {r['position']:3s} / {r['target']:18s}: MAE = {r['mae']:.3f}")

        print("\n  Training quantile models (for captain picks)...")
        q_results = train_all_quantile_models(df)
        if q_results:
            print("\n  Training Summary (quantile models):")
            for r in q_results:
                print(f"    {r['position']:3s} / q80: MAE = {r['mae']:.3f}, calibration = {r['calibration']:.1%}")

        print("\n  Training decomposed sub-models...")
        sub_results = train_all_sub_models(df, tune=args.tune)
        if sub_results:
            print("\n  Training Summary (sub-models):")
            for r in sub_results:
                print(f"    {r['position']:3s} / {r['component']:18s}: MAE = {r['mae']:.4f}")

    # 5. Generate predictions
    print("\nStep 5: Generating predictions...")
    preds = run_predictions(df, data=data)

    if preds.empty:
        print("No predictions generated. Try running with --train first.")
        sys.exit(1)

    # 6. Format and display
    result = format_predictions(preds, df)
    print_predictions(result, top_n=args.top)

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "predictions.csv"
    result.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nPredictions saved to {csv_path}")

    # Save prediction details JSON
    save_prediction_details(result, data=data)


if __name__ == "__main__":
    main()
