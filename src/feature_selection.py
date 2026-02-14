"""Phase 1: Discover which stats best predict FPL points.

Runs four feature selection methods with walk-forward validation,
separately for each position group and each target.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

if getattr(sys, "frozen", False):
    _BASE = Path(sys.executable).parent
else:
    _BASE = Path(__file__).resolve().parent.parent

OUTPUT_DIR = _BASE / "output"
POSITION_GROUPS = ["GKP", "DEF", "MID", "FWD"]
TARGETS = ["next_gw_points"]

# Walk-forward: train on GW 1..N, test on N+1
MIN_TRAIN_GWS = 10  # minimum GWs for training before we start testing


def _prepare_data(df: pd.DataFrame, position: str, target: str, feature_cols: list[str]):
    """Filter to position, drop rows missing target or all features."""
    pos_df = df[df["position_clean"] == position].copy()
    pos_df = pos_df.dropna(subset=[target])

    # Only keep rows that have at least some rolling features
    available_feats = [c for c in feature_cols if c in pos_df.columns]
    pos_df = pos_df.dropna(subset=available_feats, thresh=(len(available_feats) + 1) // 2)

    # Fill remaining NaN with 0
    for c in available_feats:
        pos_df[c] = pos_df[c].fillna(0)

    return pos_df, available_feats


def _walk_forward_splits(df: pd.DataFrame, min_train_gws: int = MIN_TRAIN_GWS):
    """Generate walk-forward train/test splits.

    Yields (train_mask, test_mask) where train uses GWs 1..N
    and test uses GW N+1. Handles multi-season data by creating a
    sequential ordering across seasons.
    """
    # Create a sequential GW index across seasons
    df = df.copy()
    season_order = sorted(df["season"].unique())
    season_map = {s: i for i, s in enumerate(season_order)}
    df["_seq_gw"] = df["season"].map(season_map) * 100 + df["gameweek"]

    seq_gws = sorted(df["_seq_gw"].unique())
    if len(seq_gws) < min_train_gws + 1:
        return

    for i in range(min_train_gws, len(seq_gws)):
        train_gws = set(seq_gws[:i])
        test_gw = seq_gws[i]
        train_mask = df["_seq_gw"].isin(train_gws)
        test_mask = df["_seq_gw"] == test_gw
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            yield train_mask, test_mask


def correlation_analysis(df: pd.DataFrame, feature_cols: list[str], target: str) -> pd.Series:
    """Pearson correlation of each feature vs target."""
    correlations = {}
    for col in feature_cols:
        if col in df.columns:
            valid = df[[col, target]].dropna()
            # Skip constant features (zero variance → division by zero in corr)
            if len(valid) > 10 and valid[col].std() > 0:
                correlations[col] = valid[col].corr(valid[target])
    return pd.Series(correlations).sort_values(ascending=False)


def rf_feature_importance(
    df: pd.DataFrame, feature_cols: list[str], target: str
) -> tuple[pd.Series, float]:
    """Random Forest feature importance with walk-forward validation.

    Returns (importance_series, avg_mae).
    """
    importances = np.zeros(len(feature_cols))
    maes = []
    n_splits = 0

    for train_mask, test_mask in _walk_forward_splits(df):
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target].values

        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances += rf.feature_importances_
        preds = rf.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        n_splits += 1

        # Limit number of walk-forward steps for speed
        if n_splits >= 20:
            break

    if n_splits == 0:
        return pd.Series(dtype=float), float("nan")

    importances /= n_splits
    avg_mae = np.mean(maes)
    return pd.Series(importances, index=feature_cols).sort_values(ascending=False), avg_mae


def lasso_feature_selection(
    df: pd.DataFrame, feature_cols: list[str], target: str
) -> tuple[pd.Series, float]:
    """LASSO regression feature selection.

    Returns (coefficient_series, avg_mae).
    """
    maes = []
    coefs = np.zeros(len(feature_cols))
    n_splits = 0

    scaler = StandardScaler()

    for train_mask, test_mask in _walk_forward_splits(df):
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target].values

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=3, random_state=42, max_iter=5000)
            lasso.fit(X_train_scaled, y_train)

        coefs += np.abs(lasso.coef_)
        preds = lasso.predict(X_test_scaled)
        maes.append(mean_absolute_error(y_test, preds))
        n_splits += 1

        if n_splits >= 20:
            break

    if n_splits == 0:
        return pd.Series(dtype=float), float("nan")

    coefs /= n_splits
    avg_mae = np.mean(maes)
    return pd.Series(coefs, index=feature_cols).sort_values(ascending=False), avg_mae


def rfe_feature_selection(
    df: pd.DataFrame, feature_cols: list[str], target: str, n_features: int = 15
) -> tuple[list[str], float]:
    """Recursive Feature Elimination using RF as estimator.

    Returns (selected_feature_names, avg_mae_with_selected).
    """
    # Use all available data for RFE ranking, then validate
    X = df[feature_cols].values
    y = df[target].values

    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    selector = RFE(rf, n_features_to_select=min(n_features, len(feature_cols)), step=1)
    selector.fit(X, y)

    selected = [f for f, s in zip(feature_cols, selector.support_) if s]

    # Validate selected features with walk-forward
    maes = []
    n_splits = 0
    for train_mask, test_mask in _walk_forward_splits(df):
        X_train = df.loc[train_mask, selected].values
        y_train = df.loc[train_mask, target].values
        X_test = df.loc[test_mask, selected].values
        y_test = df.loc[test_mask, target].values

        rf2 = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf2.fit(X_train, y_train)
        preds = rf2.predict(X_test)
        maes.append(mean_absolute_error(y_test, preds))
        n_splits += 1
        if n_splits >= 20:
            break

    avg_mae = np.mean(maes) if maes else float("nan")
    return selected, avg_mae


def _plot_importance(
    importance: pd.Series, title: str, filepath: Path, top_n: int = 20
):
    """Bar chart of feature importance."""
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, orient="h")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)


def run_feature_selection(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Run all feature selection methods for all positions and targets.

    Returns the report text. Also saves charts to output/.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = ["=" * 70, "FPL FEATURE IMPORTANCE REPORT", "=" * 70, ""]
    all_consensus = {}  # position -> target -> list of consensus features

    for position in POSITION_GROUPS:
        for target in TARGETS:
            print(f"  Analysing {position} — {target}...")
            pos_df, avail_feats = _prepare_data(df, position, target, feature_cols)

            if len(pos_df) < 100:
                report_lines.append(f"\n--- {position} / {target}: insufficient data ({len(pos_df)} rows) ---\n")
                continue

            report_lines.append(f"\n{'=' * 60}")
            report_lines.append(f"  {position} — {target}  ({len(pos_df)} samples, {len(avail_feats)} features)")
            report_lines.append(f"{'=' * 60}\n")

            # 1. Correlation
            corr = correlation_analysis(pos_df, avail_feats, target)
            report_lines.append("1. CORRELATION ANALYSIS (top 15):")
            for feat, val in corr.head(15).items():
                report_lines.append(f"   {feat:50s}  r={val:+.4f}")
            report_lines.append("")

            _plot_importance(
                corr.abs().sort_values(ascending=False),
                f"Correlation — {position} — {target}",
                OUTPUT_DIR / f"corr_{position}_{target}.png",
            )

            # 2. Random Forest importance
            rf_imp, rf_mae = rf_feature_importance(pos_df, avail_feats, target)
            report_lines.append(f"2. RANDOM FOREST IMPORTANCE (walk-forward MAE: {rf_mae:.3f}):")
            for feat, val in rf_imp.head(15).items():
                report_lines.append(f"   {feat:50s}  imp={val:.4f}")
            report_lines.append("")

            if not rf_imp.empty:
                _plot_importance(
                    rf_imp,
                    f"RF Importance — {position} — {target} (MAE={rf_mae:.3f})",
                    OUTPUT_DIR / f"rf_{position}_{target}.png",
                )

            # 3. LASSO
            lasso_coefs, lasso_mae = lasso_feature_selection(pos_df, avail_feats, target)
            report_lines.append(f"3. LASSO COEFFICIENTS (walk-forward MAE: {lasso_mae:.3f}):")
            nonzero = lasso_coefs[lasso_coefs > 0.001]
            for feat, val in nonzero.head(15).items():
                report_lines.append(f"   {feat:50s}  coef={val:.4f}")
            report_lines.append(f"   (features zeroed out: {(lasso_coefs < 0.001).sum()})")
            report_lines.append("")

            if not nonzero.empty:
                _plot_importance(
                    nonzero,
                    f"LASSO — {position} — {target} (MAE={lasso_mae:.3f})",
                    OUTPUT_DIR / f"lasso_{position}_{target}.png",
                )

            # 4. RFE
            rfe_selected, rfe_mae = rfe_feature_selection(pos_df, avail_feats, target)
            report_lines.append(f"4. RFE SELECTED FEATURES (walk-forward MAE: {rfe_mae:.3f}):")
            for feat in rfe_selected:
                report_lines.append(f"   {feat}")
            report_lines.append("")

            # Summary: consensus features (appear in top 15 of at least 2 methods)
            top_corr = set(corr.abs().sort_values(ascending=False).head(15).index)
            top_rf = set(rf_imp.head(15).index) if not rf_imp.empty else set()
            top_lasso = set(nonzero.head(15).index) if not nonzero.empty else set()
            top_rfe = set(rfe_selected)

            consensus = {}
            for feat in avail_feats:
                count = sum(feat in s for s in [top_corr, top_rf, top_lasso, top_rfe])
                if count >= 2:
                    consensus[feat] = count

            consensus_sorted = sorted(consensus.items(), key=lambda x: -x[1])
            report_lines.append("CONSENSUS (in top 15 of 2+ methods):")
            for feat, count in consensus_sorted:
                report_lines.append(f"   {feat:50s}  methods={count}/4")
            report_lines.append("")

            # Save consensus features for each target
            if consensus_sorted:
                if position not in all_consensus:
                    all_consensus[position] = {}
                all_consensus[position][target] = [feat for feat, _ in consensus_sorted]

    report = "\n".join(report_lines)
    report_path = OUTPUT_DIR / "feature_importance_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {report_path}")

    # Save consensus features as JSON for model.get_features_for_position()
    if all_consensus:
        json_path = OUTPUT_DIR / "selected_features.json"
        with open(json_path, "w") as f:
            json.dump(all_consensus, f, indent=2)
        print(f"  Selected features saved to {json_path}")

    return report


def generate_xgb_importance_charts() -> str:
    """Extract and plot feature importances from trained XGBoost models.

    Loads each trained model (main + sub-models), extracts feature_importances_,
    generates horizontal bar charts, and writes a text report.

    Returns the report text. Also saves charts to output/.
    """
    from src.model import load_model, SUB_MODELS_FOR_POSITION

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = ["=" * 70, "XGBOOST MODEL FEATURE IMPORTANCE", "=" * 70, ""]
    charts = []

    # Main models: 4 positions x targets (both targets, since both are trained)
    xgb_targets = ["next_gw_points", "next_3gw_points"]
    for position in POSITION_GROUPS:
        for target in xgb_targets:
            model_dict = load_model(position, target)
            if model_dict is None:
                report_lines.append(f"  {position} / {target}: model not trained")
                report_lines.append("")
                continue

            model = model_dict["model"]
            features = model_dict["features"]
            importances = model.feature_importances_

            imp_series = pd.Series(importances, index=features).sort_values(ascending=False)

            target_label = "Next GW" if target == "next_gw_points" else "Next 3GW"
            report_lines.append(f"{'=' * 60}")
            report_lines.append(f"  {position} — {target_label}  ({len(features)} features)")
            report_lines.append(f"{'=' * 60}")
            report_lines.append("")
            for feat, val in imp_series.head(20).items():
                report_lines.append(f"   {feat:50s}  imp={val:.4f}")
            report_lines.append("")

            chart_name = f"xgb_{position}_{target}.png"
            _plot_importance(
                imp_series,
                f"XGBoost — {position} — {target_label}",
                OUTPUT_DIR / chart_name,
            )
            charts.append(chart_name)

    # Sub-models
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("SUB-MODEL FEATURE IMPORTANCE")
    report_lines.append("=" * 70)
    report_lines.append("")

    for position in POSITION_GROUPS:
        components = SUB_MODELS_FOR_POSITION.get(position, [])
        for component in components:
            model_dict = load_model(position, f"sub_{component}")
            if model_dict is None:
                continue

            model = model_dict["model"]
            features = model_dict["features"]
            importances = model.feature_importances_

            imp_series = pd.Series(importances, index=features).sort_values(ascending=False)

            report_lines.append(f"  {position} — {component}  ({len(features)} features)")
            for feat, val in imp_series.head(10).items():
                report_lines.append(f"   {feat:50s}  imp={val:.4f}")
            report_lines.append("")

            chart_name = f"xgb_{position}_sub_{component}.png"
            _plot_importance(
                imp_series,
                f"XGBoost — {position} — {component}",
                OUTPUT_DIR / chart_name,
            )
            charts.append(chart_name)

    report = "\n".join(report_lines)
    report_path = OUTPUT_DIR / "xgb_importance_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  XGBoost importance report saved to {report_path}")
    print(f"  Generated {len(charts)} charts")

    return report


if __name__ == "__main__":
    from src.data_fetcher import load_all_data
    from src.feature_engineering import build_features, get_feature_columns

    data = load_all_data()
    df = build_features(data)
    feature_cols = get_feature_columns(df)
    report = run_feature_selection(df, feature_cols)
    print(report)
