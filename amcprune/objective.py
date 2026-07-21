import numpy as np


def _min_max_normalize(rows, key):
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def apply_outlier_aware_objective(
    *,
    score_rows,
    outlier_rows,
    outlier_metric="outlier_ratio",
    outlier_weight=0.25,
):
    """Combine sensitivity score with OWL-style outlier risk for selection.

    Lower objective_score means safer pruning. The original score is preserved
    as sensitivity_score, while score is replaced by objective_score so the
    existing ranking and pruning-plan code can be reused.
    """
    outlier_by_block = {
        int(row["block"]): float(row.get(outlier_metric, 0.0) or 0.0)
        for row in outlier_rows
    }
    rows = [dict(row) for row in score_rows]
    sensitivity_norm = _min_max_normalize(rows, "score")
    outlier_values = [
        {"value": outlier_by_block.get(int(row["block"]), 0.0)}
        for row in rows
    ]
    outlier_norm = _min_max_normalize(outlier_values, "value")

    for row, s_norm, o_norm in zip(rows, sensitivity_norm, outlier_norm):
        sensitivity_score = float(row.get("score", 0.0) or 0.0)
        outlier_risk = outlier_by_block.get(int(row["block"]), 0.0)
        objective_score = s_norm + float(outlier_weight) * o_norm
        row["sensitivity_score"] = sensitivity_score
        row["sensitivity_score_normalized"] = s_norm
        row["outlier_metric"] = outlier_metric
        row["outlier_risk"] = outlier_risk
        row["outlier_risk_normalized"] = o_norm
        row["outlier_weight"] = float(outlier_weight)
        row["objective_score"] = objective_score
        row["score"] = objective_score
    return rows


def lagrangian_unit_allocation(unit_scores, unit_costs, keep_budget):
    """MCPrune-style Lagrangian allocation.

    Returns a boolean keep mask. Higher unit_scores are preferred for keeping,
    and keep_budget is the total cost allowed to survive.
    """
    unit_scores = np.asarray(unit_scores, dtype=np.float64)
    unit_costs = np.asarray(unit_costs, dtype=np.float64)
    if len(unit_scores) == 0:
        return np.array([], dtype=bool)
    if unit_scores.max() > unit_scores.min():
        scores = (unit_scores - unit_scores.min()) / (
            unit_scores.max() - unit_scores.min() + 1e-12
        )
    else:
        scores = np.ones_like(unit_scores)

    efficiencies = scores / (unit_costs + 1e-12)
    low = 0.0
    high = float(np.max(efficiencies)) * 2.0 if len(efficiencies) else 1.0
    best_keep = np.ones_like(unit_scores, dtype=bool)
    for _ in range(50):
        lambda_value = (low + high) / 2.0
        keep = (scores - lambda_value * unit_costs) > 0
        current_cost = np.sum(unit_costs[keep])
        if current_cost <= keep_budget:
            best_keep = keep
            high = lambda_value
        else:
            low = lambda_value

    if not np.any(best_keep):
        best_keep[int(np.argmax(scores))] = True

    # Repair if the binary search produced a keep set that is too small.
    while np.sum(unit_costs[best_keep]) < keep_budget:
        candidates = np.where(~best_keep)[0]
        if len(candidates) == 0:
            break
        best_candidate = candidates[np.argmax(efficiencies[candidates])]
        new_cost = np.sum(unit_costs[best_keep]) + unit_costs[best_candidate]
        if new_cost > keep_budget and np.any(best_keep):
            break
        best_keep[best_candidate] = True
    return best_keep


def build_unit_objective_plan(
    *,
    unit_rows,
    pruning_ratio,
    outlier_weight=0.25,
    memory_weight=0.25,
):
    """Select low-risk structured units with Lagrangian resource allocation."""
    rows = [dict(row) for row in unit_rows]
    sensitivity_norm = _min_max_normalize(rows, "sensitivity_score")
    outlier_norm = _min_max_normalize(rows, "outlier_risk")
    memory_norm = _min_max_normalize(rows, "memory_cost")

    for row, s_norm, o_norm, m_norm in zip(rows, sensitivity_norm, outlier_norm, memory_norm):
        prune_risk = s_norm + float(outlier_weight) * o_norm
        keep_score = 1.0 - prune_risk
        objective = prune_risk - float(memory_weight) * m_norm
        row["sensitivity_score_normalized"] = s_norm
        row["outlier_risk_normalized"] = o_norm
        row["memory_cost_normalized"] = m_norm
        row["outlier_weight"] = float(outlier_weight)
        row["memory_weight"] = float(memory_weight)
        row["prune_risk_score"] = prune_risk
        row["keep_score"] = keep_score
        row["objective_score"] = objective
        row["score"] = objective

    total_cost = sum(float(row.get("memory_cost", 0.0) or 0.0) for row in rows)
    target_pruned_cost = total_cost * float(pruning_ratio)
    keep_budget = max(total_cost - target_pruned_cost, 0.0)
    keep_mask = lagrangian_unit_allocation(
        [row["keep_score"] for row in rows],
        [row.get("memory_cost", 0.0) or 0.0 for row in rows],
        keep_budget,
    )
    for row, keep in zip(rows, keep_mask):
        row["selected"] = not bool(keep)
        row["reason"] = (
            "selected by Lagrangian low-risk high-cost pruning"
            if row["selected"]
            else "kept by Lagrangian allocation"
        )
    selected = [row for row in rows if row["selected"]]
    selected_cost = sum(float(row.get("memory_cost", 0.0) or 0.0) for row in selected)
    return {
        "objective": "lagrangian_sensitivity_outlier_memory",
        "pruning_ratio_target": float(pruning_ratio),
        "total_units": len(rows),
        "selected_units": len(selected),
        "actual_unit_pruning_ratio": len(selected) / max(len(rows), 1),
        "total_memory_cost": total_cost,
        "target_pruned_memory_cost": target_pruned_cost,
        "selected_memory_cost": selected_cost,
        "actual_memory_pruning_ratio": selected_cost / max(total_cost, 1e-12),
        "keep_budget": keep_budget,
        "outlier_weight": float(outlier_weight),
        "memory_weight": float(memory_weight),
        "units": rows,
    }
