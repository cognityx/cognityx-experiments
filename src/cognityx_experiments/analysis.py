"""Cross-run treatment analysis owned by Experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from cognityx_experiments.canonical import plain

ANALYSIS_SCHEMA = "cognityx.experiment.analysis/v1"


def analyse_records(
    *,
    experiment_id: str,
    control_id: str,
    primary_metric: str,
    records: Sequence[Mapping[str, Any]],
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Compute descriptive effects without declaring a hypothesis proven."""
    normalized = [plain(record) for record in records]
    finalized = [
        record
        for record in normalized
        if record.get("status", "finalized") == "finalized"
        and _numeric_metric(record, primary_metric) is not None
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in finalized:
        grouped[
            (str(record["treatment_id"]), str(record.get("role") or "unspecified"))
        ].append(record)
    treatment_summary: dict[str, Any] = {}
    for treatment_id in sorted({key[0] for key in grouped}):
        selected = [
            record
            for (arm, _), grouped_records in grouped.items()
            if arm == treatment_id
            for record in grouped_records
        ]
        role_summary: dict[str, Any] = {}
        for (arm, role), role_records in sorted(grouped.items()):
            if arm != treatment_id:
                continue
            metrics = [
                _numeric_metric(record, primary_metric) for record in role_records
            ]
            role_summary[role] = {
                "count": len(metrics),
                "rate": mean(value for value in metrics if value is not None),
            }
        treatment_values = [
            _numeric_metric(record, primary_metric) for record in selected
        ]
        treatment_summary[treatment_id] = {
            "count": len(treatment_values),
            "rate": mean(value for value in treatment_values if value is not None),
            "by_role": role_summary,
        }
    if control_id not in treatment_summary:
        raise ValueError(f"No finalized primary outcomes for control {control_id}")
    control_rate = float(treatment_summary[control_id]["rate"])
    deltas = {
        treatment_id: float(summary["rate"]) - control_rate
        for treatment_id, summary in treatment_summary.items()
        if treatment_id != control_id
    }
    per_seed: dict[str, dict[str, float]] = {}
    for seed in sorted({int(record["seed"]) for record in finalized}):
        seed_records = [record for record in finalized if int(record["seed"]) == seed]
        control_values = [
            value
            for record in seed_records
            if str(record["treatment_id"]) == control_id
            if (value := _numeric_metric(record, primary_metric)) is not None
        ]
        if not control_values:
            continue
        arm_effects: dict[str, float] = {}
        for treatment_id in sorted(
            {str(record["treatment_id"]) for record in seed_records}
        ):
            if treatment_id == control_id:
                continue
            arm_values = [
                value
                for record in seed_records
                if str(record["treatment_id"]) == treatment_id
                if (value := _numeric_metric(record, primary_metric)) is not None
            ]
            if arm_values:
                arm_effects[treatment_id] = mean(arm_values) - mean(control_values)
        per_seed[str(seed)] = arm_effects
    cluster_bootstrap: dict[str, Any] = {}
    cluster_field = _cluster_field(finalized)
    if cluster_field:
        for treatment_id in deltas:
            cluster_bootstrap[treatment_id] = _bootstrap_delta(
                finalized,
                control_id=control_id,
                treatment_id=treatment_id,
                metric=primary_metric,
                cluster_field=cluster_field,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
    unresolved = sum(record.get("status") != "finalized" for record in normalized)
    comparable = sum(bool(record.get("comparable", True)) for record in finalized)
    exact_roles = {"exact_recall", "exact-recall"}
    paraphrase_roles = {"paraphrase", "generalization"}
    role_diagnostics = {
        "exact_recall": _role_rates(finalized, primary_metric, exact_roles),
        "paraphrase_generalization": _role_rates(
            finalized, primary_metric, paraphrase_roles
        ),
    }
    return {
        "schema": ANALYSIS_SCHEMA,
        "experiment_id": experiment_id,
        "primary_metric": primary_metric,
        "control_id": control_id,
        "treatments": treatment_summary,
        "deltas_from_control": deltas,
        "per_seed_effects": per_seed,
        "comparable_pair_count": comparable,
        "unresolved_count": unresolved,
        "role_diagnostics": role_diagnostics,
        "cluster_bootstrap": {
            "cluster_field": cluster_field,
            "samples": bootstrap_samples if cluster_field else 0,
            "effects": cluster_bootstrap,
        },
        "resources": _sum_mapping(normalized, "resources"),
        "semantic_judge_invocation_cost": sum(
            float(record.get("semantic_judge_invocation_cost") or 0)
            for record in normalized
        ),
        "interpretation_status": "human_review_required",
    }


def _numeric_metric(record: Mapping[str, Any], metric: str) -> float | None:
    value = (record.get("metrics") or {}).get(metric)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _role_rates(
    records: Sequence[Mapping[str, Any]], metric: str, roles: set[str]
) -> dict[str, Any]:
    selected = [record for record in records if str(record.get("role")) in roles]
    values = [
        value
        for record in selected
        if (value := _numeric_metric(record, metric)) is not None
    ]
    return {"count": len(values), "rate": mean(values) if values else None}


def _cluster_field(records: Sequence[Mapping[str, Any]]) -> str | None:
    for name in ("knowledge_unit_id", "fact_group_id", "document_id"):
        if records and all(record.get(name) for record in records):
            return name
    return None


def _bootstrap_delta(
    records: Sequence[Mapping[str, Any]],
    *,
    control_id: str,
    treatment_id: str,
    metric: str,
    cluster_field: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_cluster: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record["treatment_id"]) in {control_id, treatment_id}:
            by_cluster[str(record[cluster_field])].append(record)
    clusters = sorted(by_cluster)
    effects: list[float] = []
    generator = random.Random(seed)
    for _ in range(samples):
        sampled = [generator.choice(clusters) for _ in clusters]
        selected = [record for cluster in sampled for record in by_cluster[cluster]]
        control = [
            value
            for record in selected
            if str(record["treatment_id"]) == control_id
            if (value := _numeric_metric(record, metric)) is not None
        ]
        treatment = [
            value
            for record in selected
            if str(record["treatment_id"]) == treatment_id
            if (value := _numeric_metric(record, metric)) is not None
        ]
        if control and treatment:
            effects.append(mean(treatment) - mean(control))
    effects.sort()
    if not effects:
        return {"cluster_count": len(clusters), "estimate": None, "interval_95": None}
    lower = effects[int(0.025 * (len(effects) - 1))]
    upper = effects[int(0.975 * (len(effects) - 1))]
    return {
        "cluster_count": len(clusters),
        "estimate": mean(effects),
        "interval_95": [lower, upper],
    }


def _sum_mapping(records: Sequence[Mapping[str, Any]], name: str) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for record in records:
        value = record.get(name)
        if not isinstance(value, Mapping):
            continue
        for key, item in value.items():
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                result[str(key)] += float(item)
    return dict(sorted(result.items()))
