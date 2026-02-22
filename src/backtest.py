"""Backtest module: evaluate model accuracy against historical gameweeks."""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import ndcg_score
from xgboost import XGBRegressor


def _json_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

from src.model import (
    CURRENT_SEASON,
    DEFAULT_FEATURES,
    FPL_SCORING,
    POSITION_GROUPS,
    SUB_MODEL_COMPONENTS,
    SUB_MODEL_FEATURES,
    SUB_MODEL_OBJECTIVES,
    SUB_MODELS_FOR_POSITION,
    _prepare_position_data,
    _season_weight,
    predict_decomposed,
    predict_for_position,
)
from src.predict import ENSEMBLE_WEIGHT_DECOMPOSED


def _bootstrap_ci(values, n_boot=10000, ci=0.95):
    """Return (lower, upper) bootstrap confidence interval for the mean."""
    values = np.array(values, dtype=float)
    if len(values) < 2:
        m = float(np.mean(values))
        return (m, m)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return (round(lower, 4), round(upper, 4))


def _train_backtest_model(
    train_df: pd.DataFrame, position: str, target: str = "next_gw_points",
    quantile_alpha: float | None = None,
) -> dict | None:
    """Train a lightweight XGBoost model for backtesting (no tuning/grid search).

    Uses only the provided train_df — no future data leakage.
    When quantile_alpha is set, trains a quantile regression model instead
    of the default mean regression model.
    Returns dict with 'model' and 'features', or None if insufficient data.
    """
    feature_cols = DEFAULT_FEATURES.get(position, DEFAULT_FEATURES["MID"])
    pos_df, available_feats = _prepare_position_data(train_df, position, target, feature_cols)

    if len(pos_df) < 50:
        return None

    # Season-based sample weights
    pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))

    X = pos_df[available_feats].values
    y = pos_df[target].values
    w = pos_df["_sample_weight"].values

    if quantile_alpha is not None:
        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=quantile_alpha,
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
    model.fit(X, y, sample_weight=w)

    return {"model": model, "features": available_feats}


def _train_backtest_sub_models(
    train_df: pd.DataFrame, position: str,
) -> dict[str, dict]:
    """Train all decomposed sub-models for backtesting.

    Returns dict mapping component name -> model_dict, or empty dict on failure.
    """
    components = SUB_MODELS_FOR_POSITION.get(position, [])
    models = {}
    for comp in components:
        target = SUB_MODEL_COMPONENTS[comp]
        # Mirror model.py's train_sub_model() override for MID/FWD DefCon
        if comp == "defcon" and position in ("MID", "FWD"):
            target = "next_gw_cbirt"
        feature_cols = SUB_MODEL_FEATURES.get(comp, [])
        pos_df, available_feats = _prepare_position_data(
            train_df, position, target, feature_cols,
        )
        # Train only on rows where the player actually played
        if "next_gw_minutes" in pos_df.columns:
            pos_df = pos_df[pos_df["next_gw_minutes"] > 0].copy()
        if len(pos_df) < 30:
            continue
        pos_df["_sample_weight"] = pos_df["season"].apply(lambda s: _season_weight(s, CURRENT_SEASON))
        X = pos_df[available_feats].values
        y = pos_df[target].values
        w = pos_df["_sample_weight"].values

        objective = SUB_MODEL_OBJECTIVES.get(comp, "reg:squarederror")
        obj_params = {"objective": objective}
        if objective == "binary:logistic":
            obj_params["eval_metric"] = "logloss"

        model = XGBRegressor(
            **obj_params,
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=0,
        )
        model.fit(X, y, sample_weight=w)
        models[comp] = {"model": model, "features": available_feats}
    return models



def predict_single_gw(
    df: pd.DataFrame, predict_gw: int, season: str, print_fn=print,
) -> pd.DataFrame | None:
    """Run model prediction for a single gameweek (walk-forward, no leakage).

    Returns a DataFrame with columns: player_id, predicted_next_gw_points,
    actual, position, web_name, cost, team_code.  Returns None on failure.
    """
    season_df = df[df["season"] == season]
    available_gws = sorted(season_df["gameweek"].unique())
    season_year = int(season.split("-")[0])

    snapshot_gw = predict_gw - 1
    if snapshot_gw not in available_gws or predict_gw not in available_gws:
        return None

    snapshot = season_df[season_df["gameweek"] == snapshot_gw].copy()
    if snapshot.empty:
        return None

    # Actual points for predict_gw
    actuals_df = (
        season_df[season_df["gameweek"] == predict_gw][["player_id", "event_points"]]
        .drop_duplicates(subset="player_id", keep="first")
        .copy()
    )
    if actuals_df.empty:
        return None

    actuals_dict = dict(zip(actuals_df["player_id"], actuals_df["event_points"]))

    # Train fresh models using only data up to (but NOT including) snapshot_gw
    train_df = df[
        (df["season"].apply(lambda s: int(s.split("-")[0])) < season_year) |
        ((df["season"] == season) & (df["gameweek"] < snapshot_gw))
    ].copy()

    # Ensemble predictions: blend decomposed + mean (same as production)
    pred_col = "predicted_next_gw_points"
    all_preds = []
    for pos in POSITION_GROUPS:
        pos_sub_models = _train_backtest_sub_models(train_df, pos)
        components = SUB_MODELS_FOR_POSITION.get(pos, [])
        decomp_preds = (
            predict_decomposed(snapshot, pos, sub_models=pos_sub_models)
            if pos_sub_models and len(pos_sub_models) == len(components)
            else pd.DataFrame()
        )
        model_dict = _train_backtest_model(train_df, pos, "next_gw_points")
        mean_preds = (
            predict_for_position(snapshot, pos, "next_gw_points", model_dict)
            if model_dict is not None else pd.DataFrame()
        )

        # Blend (matches _ensemble_predict_position in predict.py)
        if not decomp_preds.empty and not mean_preds.empty:
            w_d = ENSEMBLE_WEIGHT_DECOMPOSED
            merged = decomp_preds[["player_id", pred_col]].merge(
                mean_preds[["player_id", pred_col]],
                on="player_id", suffixes=("_decomp", "_mean"),
            )
            merged[pred_col] = (
                w_d * merged[f"{pred_col}_decomp"]
                + (1 - w_d) * merged[f"{pred_col}_mean"]
            )
            preds = merged[["player_id", pred_col]]
        elif not mean_preds.empty:
            preds = mean_preds[["player_id", pred_col]]
        elif not decomp_preds.empty:
            preds = decomp_preds[["player_id", pred_col]]
        else:
            continue

        if not preds.empty:
            all_preds.append(preds.copy())

    if not all_preds:
        return None

    pred_df = pd.concat(all_preds)

    pred_df["actual"] = pred_df["player_id"].map(actuals_dict)
    pred_df = pred_df.dropna(subset=["actual"])
    if pred_df.empty:
        return None

    # Map metadata from snapshot
    deduped = snapshot.drop_duplicates("player_id")
    for col, default in [("position_clean", "MID"), ("web_name", "?")]:
        col_map = dict(zip(deduped["player_id"], deduped[col] if col in deduped.columns else pd.Series()))
        target_col = "position" if col == "position_clean" else col
        pred_df[target_col] = pred_df["player_id"].map(col_map).fillna(default)

    if "now_cost" in deduped.columns:
        cost_map = dict(zip(deduped["player_id"], deduped["now_cost"].fillna(0)))
        pred_df["cost"] = pred_df["player_id"].map(cost_map).fillna(0) / 10.0
    elif "cost" in deduped.columns:
        cost_map = dict(zip(deduped["player_id"], deduped["cost"].fillna(0)))
        pred_df["cost"] = pred_df["player_id"].map(cost_map).fillna(0)
    else:
        pred_df["cost"] = 0.0

    if "team_code" in deduped.columns:
        tc_map = dict(zip(deduped["player_id"], deduped["team_code"]))
        pred_df["team_code"] = pred_df["player_id"].map(tc_map)

    return pred_df


def _run_season_backtest(
    df: pd.DataFrame,
    start_gw: int,
    end_gw: int,
    season: str,
    print_fn=print,
) -> tuple[list[dict], pd.DataFrame]:
    """Run walk-forward backtest for a single season.

    For each gameweek in [start_gw, end_gw], trains a fresh model using
    only data up to the previous GW (no data leakage), then predicts the
    target GW.

    Returns (gameweek_results, pooled_predictions) where pooled_predictions
    is a concatenated DataFrame of per-GW prediction data for diagnostics.
    """
    season_df = df[df["season"] == season]
    available_gws = sorted(season_df["gameweek"].unique())
    season_year = int(season.split("-")[0])

    gameweek_results = []
    all_gw_predictions = []  # Collect per-GW pred_df for diagnostics

    for predict_gw in range(start_gw, end_gw + 1):
        snapshot_gw = predict_gw - 1
        if snapshot_gw not in available_gws or predict_gw not in available_gws:
            continue

        snapshot = season_df[season_df["gameweek"] == snapshot_gw].copy()
        if snapshot.empty:
            continue

        # Actual points for predict_gw
        # Deduplicate first — DGW players have multiple rows but
        # event_points already contains the total for the whole GW.
        actuals_df = (
            season_df[season_df["gameweek"] == predict_gw][["player_id", "event_points"]]
            .drop_duplicates(subset="player_id", keep="first")
            .copy()
        )
        if actuals_df.empty:
            continue

        actuals_dict = dict(zip(actuals_df["player_id"], actuals_df["event_points"]))

        # --- Train fresh models using only data up to (but NOT including)
        # snapshot_gw, because snapshot_gw's target (next_gw_points) IS the
        # predict_gw actual points — including it would leak the answer.
        # For multi-season: only include seasons <= current season (not future).
        train_df = df[
            (df["season"].apply(lambda s: int(s.split("-")[0])) < season_year) |
            ((df["season"] == season) & (df["gameweek"] < snapshot_gw))
        ].copy()

        # --- Ensemble predictions: blend decomposed + mean (same as production) ---
        pred_col = "predicted_next_gw_points"
        all_preds = []
        for pos in POSITION_GROUPS:
            pos_sub_models = _train_backtest_sub_models(train_df, pos)
            components = SUB_MODELS_FOR_POSITION.get(pos, [])
            decomp_preds = (
                predict_decomposed(snapshot, pos, sub_models=pos_sub_models)
                if pos_sub_models and len(pos_sub_models) == len(components)
                else pd.DataFrame()
            )
            model_dict = _train_backtest_model(train_df, pos, "next_gw_points")
            mean_preds = (
                predict_for_position(snapshot, pos, "next_gw_points", model_dict)
                if model_dict is not None else pd.DataFrame()
            )

            # Blend (matches _ensemble_predict_position in predict.py)
            if not decomp_preds.empty and not mean_preds.empty:
                w_d = ENSEMBLE_WEIGHT_DECOMPOSED
                merged = decomp_preds[["player_id", pred_col]].merge(
                    mean_preds[["player_id", pred_col]],
                    on="player_id", suffixes=("_decomp", "_mean"),
                )
                merged[pred_col] = (
                    w_d * merged[f"{pred_col}_decomp"]
                    + (1 - w_d) * merged[f"{pred_col}_mean"]
                )
                preds = merged[["player_id", pred_col]]
            elif not mean_preds.empty:
                preds = mean_preds[["player_id", pred_col]]
            elif not decomp_preds.empty:
                preds = decomp_preds[["player_id", pred_col]]
            else:
                continue

            if not preds.empty:
                all_preds.append(preds.copy())

        if not all_preds:
            continue

        pred_df = pd.concat(all_preds)

        # --- Quantile predictions for captain scoring (MID/FWD only) ---
        q80_preds = []
        for pos in ["MID", "FWD"]:
            q_model_dict = _train_backtest_model(
                train_df, pos, "next_gw_points", quantile_alpha=0.80
            )
            if q_model_dict is None:
                continue
            q_preds = predict_for_position(
                snapshot, pos, "next_gw_points", q_model_dict, suffix="_q80"
            )
            if not q_preds.empty:
                q80_preds.append(
                    q_preds[["player_id", "predicted_next_gw_points_q80"]].copy()
                )

        if q80_preds:
            q80_df = pd.concat(q80_preds)
            pred_df = pred_df.merge(q80_df, on="player_id", how="left")

        # Composite captain score: blend mean + quantile for upside
        if "predicted_next_gw_points_q80" in pred_df.columns:
            pred_df["captain_score"] = (
                0.4 * pred_df["predicted_next_gw_points"]
                + 0.6 * pred_df["predicted_next_gw_points_q80"].fillna(
                    pred_df["predicted_next_gw_points"]
                )
            )
        else:
            pred_df["captain_score"] = pred_df["predicted_next_gw_points"]
        pred_df["actual"] = pred_df["player_id"].map(actuals_dict)
        pred_df = pred_df.dropna(subset=["actual"])

        if pred_df.empty:
            continue

        # --- Baselines ---
        # ep_next from snapshot
        ep_map = dict(
            zip(
                snapshot.drop_duplicates("player_id")["player_id"],
                snapshot.drop_duplicates("player_id")["ep_next"].fillna(0),
            )
        )
        pred_df["ep_next"] = pred_df["player_id"].map(ep_map).fillna(0)

        # Naive form baseline (just use player_form as the prediction)
        form_map = dict(
            zip(
                snapshot.drop_duplicates("player_id")["player_id"],
                snapshot.drop_duplicates("player_id")["player_form"].fillna(0),
            )
        )
        pred_df["form_pred"] = pred_df["player_id"].map(form_map).fillna(0)

        # Position-average baseline from training data
        pos_avg_map = train_df.groupby("position_clean")["next_gw_points"].mean().to_dict()

        # Last 3 GW average baseline
        prior_gws = [g for g in available_gws if g <= snapshot_gw]
        last3_gws = prior_gws[-3:] if len(prior_gws) >= 3 else prior_gws
        last3_df = season_df[season_df["gameweek"].isin(last3_gws)]
        last3_avg = (
            last3_df
            .drop_duplicates(subset=["player_id", "gameweek"], keep="first")
            .groupby("player_id")["event_points"]
            .mean()
            .to_dict()
        )
        pred_df["last3_avg_pred"] = pred_df["player_id"].map(last3_avg).fillna(0)

        # Position info
        pos_map = dict(
            zip(
                snapshot.drop_duplicates("player_id")["player_id"],
                snapshot.drop_duplicates("player_id")["position_clean"],
            )
        )
        pred_df["position"] = pred_df["player_id"].map(pos_map)

        # Name info
        deduped_snap = snapshot.drop_duplicates("player_id")
        name_map = dict(
            zip(deduped_snap["player_id"], deduped_snap.get("web_name", pd.Series()))
        )
        pred_df["web_name"] = pred_df["player_id"].map(name_map).fillna("?")

        # Cost info (for diagnostics) — feature data uses 'cost' (in £m), raw API uses 'now_cost' (in 0.1m)
        if "cost" in deduped_snap.columns:
            cost_map = dict(zip(deduped_snap["player_id"], deduped_snap["cost"].fillna(0)))
            pred_df["cost"] = pred_df["player_id"].map(cost_map).fillna(0)
        elif "now_cost" in deduped_snap.columns:
            cost_map = dict(zip(deduped_snap["player_id"], deduped_snap["now_cost"].fillna(0)))
            pred_df["cost"] = pred_df["player_id"].map(cost_map).fillna(0) / 10.0
        else:
            pred_df["cost"] = 0.0

        # FDR info (for diagnostics) — average FDR across fixtures for DGW
        if "fdr" in snapshot.columns:
            fdr_avg = snapshot.groupby("player_id")["fdr"].mean()
            pred_df["fdr"] = pred_df["player_id"].map(fdr_avg).fillna(3.0)
        else:
            pred_df["fdr"] = 3.0

        # --- MAE ---
        model_mae = float(
            np.abs(pred_df["predicted_next_gw_points"] - pred_df["actual"]).mean()
        )
        ep_mae = float(np.abs(pred_df["ep_next"] - pred_df["actual"]).mean())
        form_mae = float(np.abs(pred_df["form_pred"] - pred_df["actual"]).mean())
        last3_mae = float(np.abs(pred_df["last3_avg_pred"] - pred_df["actual"]).mean())

        # Played-only MAE (excludes 0-point players who didn't feature)
        played_mask = pred_df["actual"] > 0
        model_mae_played = float(
            np.abs(
                pred_df.loc[played_mask, "predicted_next_gw_points"]
                - pred_df.loc[played_mask, "actual"]
            ).mean()
        ) if played_mask.any() else model_mae

        # Position-average MAE
        pred_df["pos_avg_pred"] = pred_df["position"].map(pos_avg_map).fillna(2.0)
        pos_avg_mae = float(np.abs(pred_df["pos_avg_pred"] - pred_df["actual"]).mean())

        # --- Ranking metrics ---
        _sp = spearmanr(pred_df["actual"], pred_df["predicted_next_gw_points"]).correlation
        spearman_rho = float(_sp) if not np.isnan(_sp) else 0.0
        # NDCG requires non-negative relevance — shift actuals so min is 0
        actual_shifted = pred_df["actual"].values - pred_df["actual"].values.min()
        ndcg_top20 = float(ndcg_score(
            np.array([actual_shifted]),
            np.array([pred_df["predicted_next_gw_points"].values]),
            k=20,
        ))
        _ep_sp = spearmanr(pred_df["actual"], pred_df["ep_next"]).correlation
        ep_spearman = float(_ep_sp) if not np.isnan(_ep_sp) else 0.0

        # --- Per-position MAE ---
        pos_maes = {}
        for pos in POSITION_GROUPS:
            pos_rows = pred_df[pred_df["position"] == pos]
            if not pos_rows.empty:
                pos_maes[pos] = {
                    "model": float(
                        np.abs(
                            pos_rows["predicted_next_gw_points"] - pos_rows["actual"]
                        ).mean()
                    ),
                    "ep": float(
                        np.abs(pos_rows["ep_next"] - pos_rows["actual"]).mean()
                    ),
                    "form": float(
                        np.abs(pos_rows["form_pred"] - pos_rows["actual"]).mean()
                    ),
                    "last3": float(
                        np.abs(pos_rows["last3_avg_pred"] - pos_rows["actual"]).mean()
                    ),
                    "n_players": len(pos_rows),
                }

        # --- Top 11 comparison ---
        model_top11 = pred_df.nlargest(11, "predicted_next_gw_points")
        ep_top11 = pred_df.nlargest(11, "ep_next")
        form_top11 = pred_df.nlargest(11, "form_pred")
        last3_top11 = pred_df.nlargest(11, "last3_avg_pred")
        actual_top11 = pred_df.nlargest(11, "actual")

        model_pts = float(model_top11["actual"].sum())
        ep_pts = float(ep_top11["actual"].sum())
        form_pts = float(form_top11["actual"].sum())
        last3_pts = float(last3_top11["actual"].sum())
        actual_best = float(actual_top11["actual"].sum())

        actual_ids = set(actual_top11["player_id"])
        model_overlap = int(len(set(model_top11["player_id"]) & actual_ids))
        ep_overlap = int(len(set(ep_top11["player_id"]) & actual_ids))

        # --- Captain pick accuracy ---
        # Captain = highest captain_score (composite of mean + quantile)
        captain = pred_df.nlargest(1, "captain_score").iloc[0]
        actual_top3_ids = set(pred_df.nlargest(3, "actual")["player_id"])
        captain_in_top3 = bool(captain["player_id"] in actual_top3_ids)
        captain_actual_rank = int(
            (pred_df["actual"] > captain["actual"]).sum() + 1
        )

        # Winner
        if model_pts > ep_pts:
            winner = "MODEL"
        elif model_pts < ep_pts:
            winner = "ep_next"
        else:
            winner = "TIE"

        # Capture percentage for this GW
        gw_capture_pct = round((model_pts / actual_best) * 100, 1) if actual_best > 0 else 0

        gw_result = {
            "gw": predict_gw,
            "season": season,
            "n_players": len(pred_df),
            "model_mae": round(model_mae, 3),
            "model_mae_played": round(model_mae_played, 3),
            "ep_mae": round(ep_mae, 3),
            "form_mae": round(form_mae, 3),
            "last3_mae": round(last3_mae, 3),
            "pos_avg_mae": round(pos_avg_mae, 3),
            "spearman_rho": round(spearman_rho, 3),
            "ndcg_top20": round(ndcg_top20, 3),
            "ep_spearman": round(ep_spearman, 3),
            "model_top11_pts": round(model_pts, 1),
            "ep_top11_pts": round(ep_pts, 1),
            "form_top11_pts": round(form_pts, 1),
            "last3_top11_pts": round(last3_pts, 1),
            "actual_best_pts": round(actual_best, 1),
            "model_overlap": model_overlap,
            "ep_overlap": ep_overlap,
            "captain_name": str(captain.get("web_name", "?")),
            "captain_predicted": round(float(captain["predicted_next_gw_points"]), 2),
            "captain_actual": round(float(captain["actual"]), 1),
            "captain_in_top3": captain_in_top3,
            "captain_actual_rank": captain_actual_rank,
            "winner": winner,
            "capture_pct": gw_capture_pct,
            "pos_mae": pos_maes,
        }

        gameweek_results.append(gw_result)

        # Collect raw prediction data for diagnostics
        diag_df = pred_df[["player_id", "predicted_next_gw_points", "actual",
                           "position", "web_name", "ep_next", "captain_score"]].copy()
        diag_df["gw"] = predict_gw
        if "cost" in pred_df.columns:
            diag_df["cost"] = pred_df["cost"]
        if "fdr" in pred_df.columns:
            diag_df["fdr"] = pred_df["fdr"]
        all_gw_predictions.append(diag_df)

        print_fn(
            f"  [{season}] GW{predict_gw:2d}: MAE m={model_mae:.2f} pl={model_mae_played:.2f} ep={ep_mae:.2f} f={form_mae:.2f} l3={last3_mae:.2f}"
            f" rho={spearman_rho:.2f}"
            f" | Top11 m={model_pts:.0f} ep={ep_pts:.0f} best={actual_best:.0f}"
            f" | Cap: {captain.get('web_name','?')} ({captain['actual']:.0f}pts, rank {captain_actual_rank})"
            f" | {winner}"
        )

    # Combine all per-GW predictions for diagnostics
    pooled_predictions = pd.concat(all_gw_predictions, ignore_index=True) if all_gw_predictions else pd.DataFrame()

    return gameweek_results, pooled_predictions


def _compute_diagnostics(
    pooled: pd.DataFrame, gameweek_results: list[dict],
) -> dict:
    """Compute diagnostic breakdowns from pooled per-GW predictions vs actuals.

    Returns a dict with calibration, fixture difficulty, cost tier, haul/blank
    detection, captain analysis, and biggest misses.
    """
    if pooled.empty:
        return {}

    pred_col = "predicted_next_gw_points"
    diagnostics = {}

    # --- 2A: Calibration analysis ---
    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, float("inf"))]
    bin_labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    calibration = []
    for (lo, hi), label in zip(bins, bin_labels):
        mask = (pooled[pred_col] >= lo) & (pooled[pred_col] < hi)
        subset = pooled[mask]
        if len(subset) > 0:
            calibration.append({
                "bin": label,
                "predicted_avg": round(float(subset[pred_col].mean()), 2),
                "actual_avg": round(float(subset["actual"].mean()), 2),
                "count": int(len(subset)),
            })
    diagnostics["calibration"] = calibration

    # --- 2B: Fixture difficulty breakdown ---
    if "fdr" in pooled.columns:
        by_difficulty = {}
        for label, lo, hi in [("easy", 0, 2.5), ("medium", 2.5, 3.5), ("hard", 3.5, 6)]:
            mask = (pooled["fdr"] >= lo) & (pooled["fdr"] < hi)
            subset = pooled[mask]
            if len(subset) > 0:
                by_difficulty[label] = {
                    "mae": round(float(np.abs(subset[pred_col] - subset["actual"]).mean()), 3),
                    "avg_predicted": round(float(subset[pred_col].mean()), 2),
                    "avg_actual": round(float(subset["actual"].mean()), 2),
                    "count": int(len(subset)),
                }
        diagnostics["by_difficulty"] = by_difficulty

    # --- 2C: Player cost tier breakdown ---
    if "cost" in pooled.columns:
        by_cost_tier = {}
        for label, lo, hi in [("budget", 0, 5.0), ("mid_range", 5.0, 8.0), ("premium", 8.0, 100.0)]:
            mask = (pooled["cost"] >= lo) & (pooled["cost"] < hi)
            subset = pooled[mask]
            if len(subset) > 0:
                mae = float(np.abs(subset[pred_col] - subset["actual"]).mean())
                _sp = spearmanr(subset["actual"], subset[pred_col]).correlation
                by_cost_tier[label] = {
                    "mae": round(mae, 3),
                    "spearman": round(float(_sp) if not np.isnan(_sp) else 0.0, 3),
                    "count": int(len(subset)),
                }
        diagnostics["by_cost_tier"] = by_cost_tier

    # --- 2D: Haul and blank detection ---
    haul_threshold = 8
    blank_threshold = 1
    hauls = pooled[pooled["actual"] >= haul_threshold]
    blanks = pooled[pooled["actual"] <= blank_threshold]

    haul_detection = {"total_hauls": int(len(hauls))}

    # Per-GW top-20 analysis
    hauls_in_model_top20 = 0
    hauls_in_ep_top20 = 0
    model_false_positives = 0  # Model top-20 who blanked
    model_top20_total = 0

    for gw in pooled["gw"].unique():
        gw_df = pooled[pooled["gw"] == gw]
        model_top20 = set(gw_df.nlargest(20, pred_col)["player_id"])
        gw_hauls = set(gw_df[gw_df["actual"] >= haul_threshold]["player_id"])
        gw_blanks = set(gw_df[gw_df["actual"] <= blank_threshold]["player_id"])

        hauls_in_model_top20 += len(model_top20 & gw_hauls)
        model_false_positives += len(model_top20 & gw_blanks)
        model_top20_total += len(model_top20)

        if "ep_next" in gw_df.columns:
            ep_top20 = set(gw_df.nlargest(20, "ep_next")["player_id"])
            hauls_in_ep_top20 += len(ep_top20 & gw_hauls)

    haul_detection["hauls_in_model_top20"] = hauls_in_model_top20
    haul_detection["haul_capture_rate"] = (
        round(hauls_in_model_top20 / len(hauls), 3)
        if len(hauls) > 0 else 0
    )
    haul_detection["hauls_in_ep_top20"] = hauls_in_ep_top20
    haul_detection["avg_predicted_for_hauls"] = (
        round(float(hauls[pred_col].mean()), 2)
        if len(hauls) > 0 else 0
    )
    haul_detection["avg_predicted_for_blanks"] = (
        round(float(blanks[pred_col].mean()), 2)
        if len(blanks) > 0 else 0
    )
    haul_detection["false_positive_rate"] = (
        round(model_false_positives / model_top20_total, 3)
        if model_top20_total > 0 else 0
    )
    diagnostics["haul_detection"] = haul_detection

    # --- 2E: Captain analysis (detailed) ---
    captain_analysis = {}
    captain_pts_list = []
    best_captain_pts_list = []
    ep_captain_pts_list = []
    captain_in_top3_count = 0
    captain_in_top10_count = 0
    worst_captain_gws = []

    for gw_result in gameweek_results:
        gw = gw_result["gw"]
        gw_df = pooled[pooled["gw"] == gw]
        if gw_df.empty:
            continue

        # Model captain pick (highest captain_score)
        captain_row = gw_df.nlargest(1, "captain_score").iloc[0]
        captain_actual = float(captain_row["actual"])
        captain_pts_list.append(captain_actual)

        # Best possible captain
        best_row = gw_df.nlargest(1, "actual").iloc[0]
        best_captain_pts_list.append(float(best_row["actual"]))

        # ep_next captain
        if "ep_next" in gw_df.columns:
            ep_captain_row = gw_df.nlargest(1, "ep_next").iloc[0]
            ep_captain_pts_list.append(float(ep_captain_row["actual"]))

        # Top-N checks
        actual_sorted = gw_df.nlargest(10, "actual")
        top3_ids = set(actual_sorted.head(3)["player_id"])
        top10_ids = set(actual_sorted["player_id"])
        if captain_row["player_id"] in top3_ids:
            captain_in_top3_count += 1
        if captain_row["player_id"] in top10_ids:
            captain_in_top10_count += 1

        # Track worst captain GWs
        worst_captain_gws.append({
            "gw": gw,
            "captain": str(captain_row.get("web_name", "?")),
            "pts": round(captain_actual, 1),
            "best": str(best_row.get("web_name", "?")),
            "best_pts": round(float(best_row["actual"]), 1),
        })

    n_gws = len(captain_pts_list)
    if n_gws > 0:
        captain_analysis["avg_captain_pts"] = round(float(np.mean(captain_pts_list)), 1)
        captain_analysis["avg_best_captain_pts"] = round(float(np.mean(best_captain_pts_list)), 1)
        captain_analysis["captain_pts_lost"] = round(
            float(np.mean(best_captain_pts_list)) - float(np.mean(captain_pts_list)), 1
        )
        if ep_captain_pts_list:
            captain_analysis["ep_avg_captain_pts"] = round(float(np.mean(ep_captain_pts_list)), 1)
        captain_analysis["captain_in_top3_pct"] = round(captain_in_top3_count / n_gws * 100, 0)
        captain_analysis["captain_in_top10_pct"] = round(captain_in_top10_count / n_gws * 100, 0)

        # Worst 5 captain GWs by points lost
        worst_captain_gws.sort(key=lambda x: x["best_pts"] - x["pts"], reverse=True)
        captain_analysis["worst_captain_gws"] = worst_captain_gws[:5]

    diagnostics["captain_analysis"] = captain_analysis

    # --- 2F: Biggest misses ---
    # Work on a copy to avoid mutating the caller's DataFrame
    errors_df = pooled[["web_name", "gw", pred_col, "actual"]].copy()
    errors_df["error"] = errors_df[pred_col] - errors_df["actual"]

    # Overpredicted: high prediction, low actual (positive error)
    overpredicted = (
        errors_df[errors_df["error"] > 0]
        .nlargest(10, "error")
        .rename(columns={pred_col: "predicted"})
    )
    overpredicted["error"] = overpredicted["error"].round(1)
    overpredicted["predicted"] = overpredicted["predicted"].round(1)

    # Underpredicted: low prediction, high actual (negative error)
    underpredicted = (
        errors_df[errors_df["error"] < 0]
        .nsmallest(10, "error")
        .rename(columns={pred_col: "predicted"})
    )
    underpredicted["error"] = underpredicted["error"].abs().round(1)
    underpredicted["predicted"] = underpredicted["predicted"].round(1)

    diagnostics["biggest_misses"] = {
        "overpredicted": overpredicted[["web_name", "gw", "predicted", "actual", "error"]].to_dict("records"),
        "underpredicted": underpredicted[["web_name", "gw", "predicted", "actual", "error"]].to_dict("records"),
    }

    return diagnostics


def _compute_3gw_backtest(
    pooled: pd.DataFrame, print_fn=print,
) -> dict:
    """Compute 3-GW backtest metrics from pooled per-GW predictions.

    For each GW where we have predictions for GW, GW+1, GW+2:
    - Sum predicted 1-GW points across the 3-GW window (with 0.95 decay)
    - Sum actual points across the 3-GW window
    - Compare predicted vs actual 3-GW totals

    Returns dict with MAE_3gw, Spearman_3gw, and per-window results.
    """
    if pooled.empty:
        return {}

    pred_col = "predicted_next_gw_points"
    gws = sorted(pooled["gw"].unique())

    if len(gws) < 3:
        return {}

    window_results = []

    for i, start_gw in enumerate(gws):
        window_gws = [start_gw, start_gw + 1, start_gw + 2]
        # Check all 3 GWs have predictions
        if not all(g in gws for g in window_gws):
            continue

        # Collect per-GW data for this window
        # Note: each GW's prediction was made independently (fresh model trained
        # up to that GW), so no confidence decay is applied here.  Production's
        # 3-GW sum uses decay because it projects from a single snapshot, but
        # the backtest already has the best possible 1-GW prediction for each GW.
        gw_dfs = []
        for offset, gw in enumerate(window_gws):
            gw_df = pooled[pooled["gw"] == gw][["player_id", pred_col, "actual"]].copy()
            gw_df = gw_df.rename(columns={
                pred_col: f"pred_{gw}",
                "actual": f"actual_{gw}",
            })
            gw_dfs.append(gw_df)

        # Merge — only keep players with data in all 3 GWs
        merged = gw_dfs[0]
        for extra in gw_dfs[1:]:
            merged = merged.merge(extra, on="player_id", how="inner")

        if len(merged) < 20:
            continue

        pred_3gw_cols = [c for c in merged.columns if c.startswith("pred_")]
        actual_3gw_cols = [c for c in merged.columns if c.startswith("actual_")]

        merged["pred_3gw"] = merged[pred_3gw_cols].sum(axis=1)
        merged["actual_3gw"] = merged[actual_3gw_cols].sum(axis=1)

        mae_3gw = float(np.abs(merged["pred_3gw"] - merged["actual_3gw"]).mean())
        _sp = spearmanr(merged["actual_3gw"], merged["pred_3gw"]).correlation
        spearman_3gw = float(_sp) if not np.isnan(_sp) else 0.0

        # Top-11 transfer target accuracy
        model_top11 = merged.nlargest(11, "pred_3gw")
        actual_top11 = merged.nlargest(11, "actual_3gw")
        model_top11_pts = float(model_top11["actual_3gw"].sum())
        actual_best_pts = float(actual_top11["actual_3gw"].sum())
        overlap = int(len(set(model_top11["player_id"]) & set(actual_top11["player_id"])))

        window_results.append({
            "start_gw": int(start_gw),
            "end_gw": int(start_gw + 2),
            "n_players": int(len(merged)),
            "mae_3gw": round(mae_3gw, 3),
            "spearman_3gw": round(spearman_3gw, 3),
            "model_top11_pts": round(model_top11_pts, 1),
            "actual_best_pts": round(actual_best_pts, 1),
            "top11_overlap": overlap,
        })

    if not window_results:
        return {}

    avg_mae = round(np.mean([r["mae_3gw"] for r in window_results]), 3)
    avg_spearman = round(np.mean([r["spearman_3gw"] for r in window_results]), 3)
    avg_top11_pts = round(np.mean([r["model_top11_pts"] for r in window_results]), 1)
    avg_actual_best = round(np.mean([r["actual_best_pts"] for r in window_results]), 1)
    capture_pct = round((avg_top11_pts / avg_actual_best) * 100, 1) if avg_actual_best > 0 else 0

    print_fn(f"\n  === 3-GW Backtest ({len(window_results)} windows) ===")
    print_fn(f"  Avg MAE (3-GW): {avg_mae}")
    print_fn(f"  Avg Spearman (3-GW): {avg_spearman}")
    print_fn(f"  Avg Top-11 pts: {avg_top11_pts} / {avg_actual_best} ({capture_pct}%)")

    return {
        "n_windows": len(window_results),
        "avg_mae_3gw": avg_mae,
        "avg_spearman_3gw": avg_spearman,
        "avg_model_top11_pts": avg_top11_pts,
        "avg_actual_best_pts": avg_actual_best,
        "capture_pct_3gw": capture_pct,
        "windows": window_results,
    }


def run_backtest(
    df: pd.DataFrame,
    start_gw: int = 5,
    end_gw: int = 25,
    season: str = "",
    seasons: list[str] | None = None,
    print_fn=print,
) -> dict:
    """Run walk-forward backtest with per-GW model retraining.

    For each gameweek in [start_gw, end_gw], trains a fresh model using
    only data up to the previous GW (no data leakage), then predicts the
    target GW. Compares model predictions against FPL's ep_next, a
    naive form baseline, and a last-3-GW average baseline.

    When `seasons` is provided, runs backtest across multiple seasons and
    aggregates results. Default None preserves single-season behavior.

    Returns a dict with summary stats, per-GW results, and per-position breakdown.
    """
    # Determine which seasons to backtest
    if seasons:
        season_list = seasons
    else:
        if not season:
            from src.data_fetcher import detect_current_season
            season = detect_current_season()
        season_list = [season]

    # Collect gameweek results across all seasons
    gameweek_results = []
    all_pooled = []
    for s in season_list:
        print_fn(f"\n  --- Season {s} ---")
        gw_results, pooled = _run_season_backtest(df, start_gw, end_gw, s, print_fn)
        gameweek_results.extend(gw_results)
        if not pooled.empty:
            all_pooled.append(pooled)

    if not gameweek_results:
        return {"error": "No gameweeks available for backtest."}

    pooled_predictions = pd.concat(all_pooled, ignore_index=True) if all_pooled else pd.DataFrame()

    # --- Aggregate summary ---
    n_gws = len(gameweek_results)
    model_wins = sum(1 for r in gameweek_results if r["winner"] == "MODEL")
    ep_wins = sum(1 for r in gameweek_results if r["winner"] == "ep_next")
    ties = sum(1 for r in gameweek_results if r["winner"] == "TIE")

    avg_model_mae = round(np.mean([r["model_mae"] for r in gameweek_results]), 3)
    avg_model_mae_played = round(np.mean([r["model_mae_played"] for r in gameweek_results]), 3)
    avg_ep_mae = round(np.mean([r["ep_mae"] for r in gameweek_results]), 3)
    avg_form_mae = round(np.mean([r["form_mae"] for r in gameweek_results]), 3)
    avg_last3_mae = round(np.mean([r["last3_mae"] for r in gameweek_results]), 3)
    avg_pos_avg_mae = round(np.mean([r["pos_avg_mae"] for r in gameweek_results]), 3)

    avg_spearman = round(np.mean([r["spearman_rho"] for r in gameweek_results]), 3)
    avg_ndcg_top20 = round(np.mean([r["ndcg_top20"] for r in gameweek_results]), 3)
    avg_ep_spearman = round(np.mean([r["ep_spearman"] for r in gameweek_results]), 3)

    avg_model_pts = round(np.mean([r["model_top11_pts"] for r in gameweek_results]), 1)
    avg_ep_pts = round(np.mean([r["ep_top11_pts"] for r in gameweek_results]), 1)
    avg_form_pts = round(np.mean([r["form_top11_pts"] for r in gameweek_results]), 1)
    avg_last3_pts = round(np.mean([r["last3_top11_pts"] for r in gameweek_results]), 1)
    avg_actual_pts = round(
        np.mean([r["actual_best_pts"] for r in gameweek_results]), 1
    )

    avg_model_overlap = round(
        np.mean([r["model_overlap"] for r in gameweek_results]), 1
    )
    avg_ep_overlap = round(
        np.mean([r["ep_overlap"] for r in gameweek_results]), 1
    )

    captain_hit_rate = round(
        sum(1 for r in gameweek_results if r["captain_in_top3"]) / n_gws, 2
    )

    # Paired Wilcoxon signed-rank tests: model vs ep_next
    model_maes = [r["model_mae"] for r in gameweek_results]
    ep_maes = [r["ep_mae"] for r in gameweek_results]
    model_spears = [r["spearman_rho"] for r in gameweek_results]
    ep_spears = [r["ep_spearman"] for r in gameweek_results]

    mae_diffs = [m - e for m, e in zip(model_maes, ep_maes)]
    if len(mae_diffs) >= 6 and any(d != 0 for d in mae_diffs):
        _, mae_pvalue = wilcoxon(mae_diffs, alternative='less')
    else:
        mae_pvalue = float('nan')

    spear_diffs = [m - e for m, e in zip(model_spears, ep_spears)]
    if len(spear_diffs) >= 6 and any(d != 0 for d in spear_diffs):
        _, spear_pvalue = wilcoxon(spear_diffs, alternative='greater')
    else:
        spear_pvalue = float('nan')

    # --- Bootstrap confidence intervals ---
    per_gw_capture_pcts = [r["capture_pct"] for r in gameweek_results]
    per_gw_win_indicators = [1.0 if r["winner"] == "MODEL" else 0.0 for r in gameweek_results]

    model_mae_ci = _bootstrap_ci(model_maes)
    ep_mae_ci = _bootstrap_ci(ep_maes)
    model_top11_pts_ci = _bootstrap_ci([r["model_top11_pts"] for r in gameweek_results])
    capture_pct_ci = _bootstrap_ci(per_gw_capture_pcts)
    win_rate_ci = _bootstrap_ci(per_gw_win_indicators)

    # Per-position aggregate
    pos_summary = {}
    for pos in POSITION_GROUPS:
        pos_maes_list = [
            r["pos_mae"][pos] for r in gameweek_results if pos in r["pos_mae"]
        ]
        if pos_maes_list:
            pos_model_maes = [p["model"] for p in pos_maes_list]
            pos_ep_maes = [p["ep"] for p in pos_maes_list]

            # Per-position Wilcoxon p-value
            pos_mae_diffs = [m - e for m, e in zip(pos_model_maes, pos_ep_maes)]
            if len(pos_mae_diffs) >= 6 and any(d != 0 for d in pos_mae_diffs):
                _, pos_p = wilcoxon(pos_mae_diffs, alternative='less')
                pos_pvalue = round(float(pos_p), 4)
            else:
                pos_pvalue = None

            pos_summary[pos] = {
                "model_mae": round(np.mean(pos_model_maes), 3),
                "ep_mae": round(np.mean(pos_ep_maes), 3),
                "form_mae": round(
                    np.mean([p["form"] for p in pos_maes_list]), 3
                ),
                "last3_mae": round(
                    np.mean([p["last3"] for p in pos_maes_list]), 3
                ),
                "avg_players": round(
                    np.mean([p["n_players"] for p in pos_maes_list]), 0
                ),
                "model_mae_ci": _bootstrap_ci(pos_model_maes),
                "ep_mae_ci": _bootstrap_ci(pos_ep_maes),
                "mae_pvalue": pos_pvalue,
            }

    # Model % of theoretical maximum
    capture_pct = round((avg_model_pts / avg_actual_pts) * 100, 1) if avg_actual_pts > 0 else 0

    summary = {
        "start_gw": start_gw,
        "end_gw": end_gw,
        "seasons": season_list,
        "n_gameweeks": n_gws,
        "model_avg_mae": avg_model_mae,
        "model_avg_mae_played": avg_model_mae_played,
        "ep_avg_mae": avg_ep_mae,
        "form_avg_mae": avg_form_mae,
        "last3_avg_mae": avg_last3_mae,
        "pos_avg_mae": avg_pos_avg_mae,
        "avg_spearman": avg_spearman,
        "avg_ndcg_top20": avg_ndcg_top20,
        "avg_ep_spearman": avg_ep_spearman,
        "model_wins": model_wins,
        "ep_wins": ep_wins,
        "ties": ties,
        "model_avg_top11_pts": avg_model_pts,
        "ep_avg_top11_pts": avg_ep_pts,
        "form_avg_top11_pts": avg_form_pts,
        "last3_avg_top11_pts": avg_last3_pts,
        "actual_avg_top11_pts": avg_actual_pts,
        "model_capture_pct": capture_pct,
        "model_avg_overlap": avg_model_overlap,
        "ep_avg_overlap": avg_ep_overlap,
        "captain_hit_rate": captain_hit_rate,
        "mae_pvalue": round(mae_pvalue, 4) if not np.isnan(mae_pvalue) else None,
        "spearman_pvalue": round(spear_pvalue, 4) if not np.isnan(spear_pvalue) else None,
        "model_mae_ci": list(model_mae_ci),
        "ep_mae_ci": list(ep_mae_ci),
        "model_top11_pts_ci": list(model_top11_pts_ci),
        "capture_pct_ci": list(capture_pct_ci),
        "win_rate_ci": list(win_rate_ci),
    }

    season_label = ", ".join(season_list) if len(season_list) > 1 else season_list[0]
    print_fn(f"\n  === Backtest Summary ({season_label}, GW{start_gw}-{end_gw}) ===")
    print_fn(f"  Model wins: {model_wins}/{n_gws}, ep_next: {ep_wins}/{n_gws}, Ties: {ties}/{n_gws}")
    print_fn(f"  Avg MAE — Model: {avg_model_mae} (played: {avg_model_mae_played}), ep_next: {avg_ep_mae}, Form: {avg_form_mae}, Last3: {avg_last3_mae}, PosAvg: {avg_pos_avg_mae}")
    print_fn(f"  Avg Spearman — Model: {avg_spearman}, ep_next: {avg_ep_spearman} | NDCG@20: {avg_ndcg_top20}")
    print_fn(f"  Avg Top 11 — Model: {avg_model_pts}, ep_next: {avg_ep_pts}, Last3: {avg_last3_pts}, Actual: {avg_actual_pts}")
    print_fn(f"  Model captures {capture_pct}% of theoretical maximum")
    print_fn(f"  Captain in actual top 3: {int(captain_hit_rate * 100)}% of GWs")
    print_fn(f"  95% CIs — MAE: [{model_mae_ci[0]}, {model_mae_ci[1]}], Top11: [{model_top11_pts_ci[0]}, {model_top11_pts_ci[1]}]")
    mae_p_str = f"{mae_pvalue:.4f}" if not np.isnan(mae_pvalue) else "n/a"
    spear_p_str = f"{spear_pvalue:.4f}" if not np.isnan(spear_pvalue) else "n/a"
    print_fn(f"  Significance — MAE p={mae_p_str}, Spearman p={spear_p_str} (Wilcoxon signed-rank, one-sided)")

    # --- 3-GW Backtest ---
    backtest_3gw = _compute_3gw_backtest(pooled_predictions, print_fn)

    # --- Diagnostics ---
    diagnostics = _compute_diagnostics(pooled_predictions, gameweek_results)
    if diagnostics:
        # Print diagnostic highlights
        if "calibration" in diagnostics:
            print_fn(f"\n  === Diagnostics ===")
            cal = diagnostics["calibration"]
            print_fn(f"  Calibration ({len(cal)} bins):")
            for b in cal:
                delta = b["actual_avg"] - b["predicted_avg"]
                direction = "under" if delta > 0.1 else "over" if delta < -0.1 else "ok"
                print_fn(f"    {b['bin']:5s}: pred={b['predicted_avg']:.2f} actual={b['actual_avg']:.2f} ({direction}, n={b['count']})")
        if "haul_detection" in diagnostics:
            hd = diagnostics["haul_detection"]
            print_fn(f"  Haul detection: {hd['hauls_in_model_top20']}/{hd['total_hauls']} hauls in model top-20 ({hd['haul_capture_rate']:.1%})")
        if "captain_analysis" in diagnostics and diagnostics["captain_analysis"]:
            ca = diagnostics["captain_analysis"]
            print_fn(f"  Captain: avg {ca.get('avg_captain_pts', 0):.1f} pts (best possible: {ca.get('avg_best_captain_pts', 0):.1f}, lost: {ca.get('captain_pts_lost', 0):.1f})")

    result = {
        "summary": summary,
        "by_position": pos_summary,
        "gameweeks": gameweek_results,
        "diagnostics": _json_safe(diagnostics),
    }
    if backtest_3gw:
        result["backtest_3gw"] = _json_safe(backtest_3gw)

    return result
