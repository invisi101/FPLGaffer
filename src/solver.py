"""MILP solvers for FPL squad selection and transfer optimization."""

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp


def scrub_nan(records: list[dict]) -> list[dict]:
    """Replace NaN/inf with None in a list of dicts for valid JSON."""
    for row in records:
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None
    return records


def solve_milp_team(
    player_df: pd.DataFrame, target_col: str,
    budget: float = 1000.0, team_cap: int = 3,
) -> dict | None:
    """Solve two-tier MILP for optimal squad selection.

    Returns {"starters": [...], "bench": [...], "total_cost": float,
    "starting_points": float} or None on failure.
    """
    from scipy.optimize import Bounds as ScipyBounds

    required = ["position", "cost", target_col]
    if not all(c in player_df.columns for c in required):
        return None

    df = player_df.dropna(subset=required).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None

    pred = df[target_col].values.astype(float)

    SUB_WEIGHT = 0.1
    c = np.concatenate([
        -SUB_WEIGHT * pred,
        -(1 - SUB_WEIGHT) * pred,
    ])

    integrality = np.ones(2 * n)

    A_rows = []
    lbs = []
    ubs = []

    def add_constraint(row_x, row_s, lb, ub):
        A_rows.append(np.concatenate([row_x, row_s]))
        lbs.append(lb)
        ubs.append(ub)

    zeros = np.zeros(n)
    costs = df["cost"].values.astype(float)

    add_constraint(costs, zeros, 0, budget)

    squad_req = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, count in squad_req.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(pos_mask, zeros, count, count)

    if "team_code" in df.columns:
        for tc in df["team_code"].unique():
            team_mask = (df["team_code"] == tc).astype(float).values
            add_constraint(team_mask, zeros, 0, team_cap)

    add_constraint(zeros, np.ones(n), 11, 11)

    start_min = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    for pos, (lo, hi) in start_min.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(zeros, pos_mask, lo, hi)

    for i in range(n):
        row_x = np.zeros(n)
        row_s = np.zeros(n)
        row_x[i] = -1.0
        row_s[i] = 1.0
        add_constraint(row_x, row_s, -np.inf, 0)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, lbs, ubs)
    variable_bounds = ScipyBounds(lb=0, ub=1)

    result = milp(c, integrality=integrality, bounds=variable_bounds, constraints=constraints)

    if not result.success:
        return None

    x_vals = result.x[:n]
    s_vals = result.x[n:]
    squad_mask = x_vals > 0.5
    starter_mask = s_vals > 0.5

    team_df = df[squad_mask].copy()
    team_df["starter"] = starter_mask[squad_mask]

    float_cols = team_df.select_dtypes(include="float").columns
    team_df[float_cols] = team_df[float_cols].round(2)

    starters = team_df[team_df["starter"]]
    bench = team_df[~team_df["starter"]]

    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    team_df["_pos_order"] = team_df["position"].map(pos_order)
    team_df = team_df.sort_values(
        ["starter", "_pos_order", target_col], ascending=[False, True, False],
    )
    team_df = team_df.drop(columns=["_pos_order"])

    return {
        "starters": scrub_nan(starters.to_dict(orient="records")),
        "bench": scrub_nan(bench.to_dict(orient="records")),
        "total_cost": round(team_df["cost"].sum(), 1),
        "starting_points": round(starters[target_col].sum(), 2),
        "players": scrub_nan(team_df.to_dict(orient="records")),
    }


def solve_transfer_milp(
    player_df: pd.DataFrame,
    current_player_ids: set[int],
    target_col: str,
    budget: float = 1000.0,
    max_transfers: int = 2,
    team_cap: int = 3,
) -> dict | None:
    """Solve MILP for optimal squad reachable via at most max_transfers changes.

    Identical to solve_milp_team() but adds one extra constraint:
    at least (15 - max_transfers) players must come from the current squad.
    """
    from scipy.optimize import Bounds as ScipyBounds

    required = ["position", "cost", target_col]
    if not all(c in player_df.columns for c in required):
        return None

    df = player_df.dropna(subset=required).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None

    pred = df[target_col].values.astype(float)

    # Mark which players are in the current squad
    is_current = df["player_id"].isin(current_player_ids).astype(float).values

    SUB_WEIGHT = 0.1
    c = np.concatenate([
        -SUB_WEIGHT * pred,
        -(1 - SUB_WEIGHT) * pred,
    ])

    integrality = np.ones(2 * n)

    A_rows = []
    lbs = []
    ubs = []

    def add_constraint(row_x, row_s, lb, ub):
        A_rows.append(np.concatenate([row_x, row_s]))
        lbs.append(lb)
        ubs.append(ub)

    zeros = np.zeros(n)
    costs = df["cost"].values.astype(float)

    # Budget
    add_constraint(costs, zeros, 0, budget)

    # Position counts (squad)
    squad_req = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for pos, count in squad_req.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(pos_mask, zeros, count, count)

    # Max 3 from same team
    if "team_code" in df.columns:
        for tc in df["team_code"].unique():
            team_mask = (df["team_code"] == tc).astype(float).values
            add_constraint(team_mask, zeros, 0, team_cap)

    # Exactly 11 starters
    add_constraint(zeros, np.ones(n), 11, 11)

    # Formation constraints
    start_min = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
    for pos, (lo, hi) in start_min.items():
        pos_mask = (df["position"] == pos).astype(float).values
        add_constraint(zeros, pos_mask, lo, hi)

    # s_i <= x_i (can only start if in squad)
    for i in range(n):
        row_x = np.zeros(n)
        row_s = np.zeros(n)
        row_x[i] = -1.0
        row_s[i] = 1.0
        add_constraint(row_x, row_s, -np.inf, 0)

    # TRANSFER CONSTRAINT: keep at least (15 - max_transfers) current players
    keep_min = max(0, 15 - max_transfers)
    add_constraint(is_current, zeros, keep_min, 15)

    A = np.array(A_rows)
    constraints = LinearConstraint(A, lbs, ubs)
    variable_bounds = ScipyBounds(lb=0, ub=1)

    result = milp(c, integrality=integrality, bounds=variable_bounds, constraints=constraints)

    if not result.success:
        return None

    x_vals = result.x[:n]
    s_vals = result.x[n:]
    squad_mask = x_vals > 0.5
    starter_mask = s_vals > 0.5

    team_df = df[squad_mask].copy()
    team_df["starter"] = starter_mask[squad_mask]

    float_cols = team_df.select_dtypes(include="float").columns
    team_df[float_cols] = team_df[float_cols].round(2)

    # Sort by starter first, then position, then predicted points
    pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    team_df["_pos_order"] = team_df["position"].map(pos_order)
    team_df = team_df.sort_values(
        ["starter", "_pos_order", target_col], ascending=[False, True, False],
    )
    team_df = team_df.drop(columns=["_pos_order"])

    new_squad_ids = set(team_df["player_id"].tolist())
    transfers_out_ids = current_player_ids - new_squad_ids
    transfers_in_ids = new_squad_ids - current_player_ids

    starters = team_df[team_df["starter"]]
    bench = team_df[~team_df["starter"]]

    return {
        "starters": scrub_nan(starters.to_dict(orient="records")),
        "bench": scrub_nan(bench.to_dict(orient="records")),
        "players": scrub_nan(team_df.to_dict(orient="records")),
        "total_cost": round(team_df["cost"].sum(), 1),
        "starting_points": round(starters[target_col].sum(), 2),
        "transfers_in_ids": transfers_in_ids,
        "transfers_out_ids": transfers_out_ids,
    }
