"""Role-aware, paired cross-treatment analysis owned by Experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from cognityx_experiments.canonical import plain
from cognityx_experiments.contracts import EVALUATION_RESEARCH_ROLES

ANALYSIS_SCHEMA = "cognityx.experiment.analysis/v1"


def analyse_records(
    *,
    experiment_id: str,
    control_id: str,
    primary_metric: str,
    records: Sequence[Mapping[str, Any]],
    primary_role: str | None = None,
    bootstrap_samples: int = 500,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Estimate treatment effects from paired saved-adapter endpoints only."""
    normalized = [_normalize_record(record) for record in records]
    selected_role = _resolve_primary_role(normalized, primary_role)
    role_records = [
        record for record in normalized if record.get("research_role") == selected_role
    ]
    finalized = [
        record
        for record in role_records
        if _primary_finalized(record)
        and _numeric_metric(record, primary_metric) is not None
    ]
    if not any(str(record.get("treatment_id")) == control_id for record in finalized):
        raise ValueError(
            f"No finalized {selected_role} primary outcomes for control {control_id}"
        )

    treatment_ids = sorted(
        {
            str(record["treatment_id"])
            for record in role_records
            if record.get("treatment_id") is not None
        }
    )
    treatment_summary = {
        treatment_id: _treatment_summary(
            finalized, treatment_id=treatment_id, metric=primary_metric
        )
        for treatment_id in treatment_ids
    }
    contrasts = {
        treatment_id: _paired_contrast(
            role_records,
            control_id=control_id,
            treatment_id=treatment_id,
            metric=primary_metric,
        )
        for treatment_id in treatment_ids
        if treatment_id != control_id
    }
    deltas = {
        treatment_id: float(contrast["paired_delta"])
        for treatment_id, contrast in contrasts.items()
        if contrast["paired_delta"] is not None
    }
    per_seed = _per_seed_effects(
        role_records,
        control_id=control_id,
        treatment_ids=tuple(contrasts),
        metric=primary_metric,
    )

    cluster_field = _cluster_field(finalized)
    bootstrap_effects: dict[str, Any] = {}
    if cluster_field:
        for treatment_id in contrasts:
            bootstrap_effects[treatment_id] = _bootstrap_delta(
                finalized,
                control_id=control_id,
                treatment_id=treatment_id,
                metric=primary_metric,
                cluster_field=cluster_field,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )

    diagnostic_records = [
        record
        for record in normalized
        if _primary_finalized(record)
        and _numeric_metric(record, primary_metric) is not None
    ]
    role_diagnostics = {
        role: _role_rates(diagnostic_records, primary_metric, {role})
        for role in sorted(EVALUATION_RESEARCH_ROLES)
    }
    for treatment_id, summary in treatment_summary.items():
        summary["by_role"] = {
            role: _treatment_summary(
                [
                    record
                    for record in diagnostic_records
                    if record.get("research_role") == role
                ],
                treatment_id=treatment_id,
                metric=primary_metric,
            )
            for role in sorted(EVALUATION_RESEARCH_ROLES)
        }
    role_diagnostics["generalization"] = _role_rates(
        diagnostic_records,
        primary_metric,
        {"paraphrase_evaluation", "heldout_knowledge_unit"},
    )
    primary_unresolved = sum(
        not _primary_finalized(record)
        or _numeric_metric(record, primary_metric) is None
        for record in role_records
    )
    full_unresolved = sum(
        not bool(record.get("full_evaluation_finalized")) for record in role_records
    )
    finalized_pair_count = sum(
        int(contrast["finalized_paired_count"]) for contrast in contrasts.values()
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "experiment_id": experiment_id,
        "primary_metric": primary_metric,
        "primary_role": selected_role,
        "control_id": control_id,
        "treatments": treatment_summary,
        "contrasts_from_control": contrasts,
        "deltas_from_control": deltas,
        "per_seed_effects": per_seed,
        "comparable_pair_count": finalized_pair_count,
        "unresolved_count": primary_unresolved,
        "primary_endpoint_unresolved_count": primary_unresolved,
        "full_evaluation_unresolved_count": full_unresolved,
        "role_diagnostics": role_diagnostics,
        "secondary_role_summaries": role_diagnostics,
        "cluster_bootstrap": {
            "cluster_field": cluster_field,
            "samples": bootstrap_samples if cluster_field else 0,
            "effects": bootstrap_effects,
        },
        "resources": _sum_mapping(normalized, "resources"),
        "semantic_judge_invocation_cost": sum(
            float(record.get("semantic_judge_invocation_cost") or 0)
            for record in normalized
        ),
        "interpretation_status": "human_review_required",
    }


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    selected = plain(record)
    role = selected.get("research_role") or selected.get("role")
    if role is not None and str(role) not in EVALUATION_RESEARCH_ROLES:
        raise ValueError(f"Unsupported evaluation research role: {role}")
    selected["research_role"] = str(role) if role is not None else None
    if "primary_endpoint_finalized" not in selected:
        selected["primary_endpoint_finalized"] = (
            selected.get("status", "finalized") == "finalized"
        )
    if "full_evaluation_finalized" not in selected:
        selected["full_evaluation_finalized"] = (
            selected.get("status", "finalized") == "finalized"
        )
    return selected


def _resolve_primary_role(
    records: Sequence[Mapping[str, Any]], requested: str | None
) -> str:
    if requested is not None:
        if requested not in EVALUATION_RESEARCH_ROLES:
            raise ValueError(f"Unsupported primary outcome role: {requested}")
        return requested
    roles = {
        str(record["research_role"])
        for record in records
        if record.get("research_role") is not None
    }
    if len(roles) == 1:
        return next(iter(roles))
    if len(roles) > 1:
        raise ValueError(
            "Primary outcome role is ambiguous across evaluation research roles: "
            + ", ".join(sorted(roles))
        )
    raise ValueError("No canonical evaluation research role is available")


def _primary_finalized(record: Mapping[str, Any]) -> bool:
    return bool(record.get("primary_endpoint_finalized"))


def _numeric_metric(record: Mapping[str, Any], metric: str) -> float | None:
    value = record.get(metric)
    if value is None:
        value = (record.get("metrics") or {}).get(metric)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _treatment_summary(
    records: Sequence[Mapping[str, Any]], *, treatment_id: str, metric: str
) -> dict[str, Any]:
    values = [
        value
        for record in records
        if str(record.get("treatment_id")) == treatment_id
        if (value := _numeric_metric(record, metric)) is not None
    ]
    return {"count": len(values), "rate": mean(values) if values else None}


def _pair_key(record: Mapping[str, Any]) -> tuple[int, str]:
    if record.get("seed") is None or not record.get("evaluation_record_id"):
        raise ValueError(
            "Paired analysis records require seed and evaluation_record_id"
        )
    return int(record["seed"]), str(record["evaluation_record_id"])


def _arm_records(
    records: Sequence[Mapping[str, Any]], treatment_id: str
) -> dict[tuple[int, str], Mapping[str, Any]]:
    selected: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        if str(record.get("treatment_id")) != treatment_id:
            continue
        key = _pair_key(record)
        if key in selected:
            raise ValueError(
                "Duplicate analysis endpoint for treatment, seed, and "
                f"evaluation_record_id: {treatment_id} {key}"
            )
        selected[key] = record
    return selected


def _paired_contrast(
    records: Sequence[Mapping[str, Any]],
    *,
    control_id: str,
    treatment_id: str,
    metric: str,
) -> dict[str, Any]:
    control = _arm_records(records, control_id)
    treatment = _arm_records(records, treatment_id)
    all_keys = set(control) | set(treatment)
    paired_values: list[tuple[float, float]] = []
    for key in sorted(set(control) & set(treatment)):
        control_record = control[key]
        treatment_record = treatment[key]
        control_value = _numeric_metric(control_record, metric)
        treatment_value = _numeric_metric(treatment_record, metric)
        if (
            _primary_finalized(control_record)
            and _primary_finalized(treatment_record)
            and control_value is not None
            and treatment_value is not None
        ):
            paired_values.append((control_value, treatment_value))
    finalized_keys = len(paired_values)
    control_rate = mean(value[0] for value in paired_values) if paired_values else None
    treatment_rate = (
        mean(value[1] for value in paired_values) if paired_values else None
    )
    return {
        "finalized_paired_count": finalized_keys,
        "unresolved_paired_count": len(all_keys) - finalized_keys,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "paired_delta": (
            treatment_rate - control_rate
            if treatment_rate is not None and control_rate is not None
            else None
        ),
    }


def _per_seed_effects(
    records: Sequence[Mapping[str, Any]],
    *,
    control_id: str,
    treatment_ids: tuple[str, ...],
    metric: str,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for seed in sorted({int(record["seed"]) for record in records}):
        selected = [record for record in records if int(record["seed"]) == seed]
        effects: dict[str, float] = {}
        for treatment_id in treatment_ids:
            contrast = _paired_contrast(
                selected,
                control_id=control_id,
                treatment_id=treatment_id,
                metric=metric,
            )
            if contrast["paired_delta"] is not None:
                effects[treatment_id] = float(contrast["paired_delta"])
        result[str(seed)] = effects
    return result


def _role_rates(
    records: Sequence[Mapping[str, Any]], metric: str, roles: set[str]
) -> dict[str, Any]:
    values = [
        value
        for record in records
        if str(record.get("research_role")) in roles
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
    cluster_effects: dict[str, float] = {}
    for cluster in clusters:
        contrast = _paired_contrast(
            by_cluster[cluster],
            control_id=control_id,
            treatment_id=treatment_id,
            metric=metric,
        )
        if contrast["paired_delta"] is not None:
            cluster_effects[cluster] = float(contrast["paired_delta"])
    resolved_clusters = sorted(cluster_effects)
    effects: list[float] = []
    generator = random.Random(seed)
    for _ in range(samples):
        if resolved_clusters:
            sampled = [generator.choice(resolved_clusters) for _ in resolved_clusters]
            effects.append(mean(float(cluster_effects[cluster]) for cluster in sampled))
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
