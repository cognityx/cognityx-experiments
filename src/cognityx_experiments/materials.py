"""Automatic table and figure-ready data generation."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from typing import Any

from cognityx_experiments.canonical import plain


def experiment_table(analysis: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Return machine-readable rows and CSV for paper preparation."""
    control = str(analysis["control_id"])
    deltas = dict(analysis.get("deltas_from_control") or {})
    rows: list[dict[str, Any]] = []
    for treatment, summary in sorted((analysis.get("treatments") or {}).items()):
        rows.append(
            {
                "experiment_id": analysis["experiment_id"],
                "treatment": treatment,
                "control": control,
                "primary_metric": analysis["primary_metric"],
                "rate": summary.get("rate"),
                "effect_vs_control": (
                    0 if treatment == control else deltas.get(treatment)
                ),
                "replicates": len(analysis.get("per_seed_effects") or {}),
                "comparable_pairs": analysis.get("comparable_pair_count", 0),
                "unresolved_pairs": analysis.get("unresolved_count", 0),
                "semantic_judge_cost": analysis.get(
                    "semantic_judge_invocation_cost", 0
                ),
            }
        )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
    return rows, output.getvalue()


def figure_data(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Return underlying data without inventing a visually authoritative plot."""
    treatments = analysis.get("treatments") or {}
    role_effects: list[dict[str, Any]] = []
    control = str(analysis["control_id"])
    control_roles = (treatments.get(control) or {}).get("by_role") or {}
    for treatment, summary in sorted(treatments.items()):
        for role, role_summary in sorted((summary.get("by_role") or {}).items()):
            control_rate = (control_roles.get(role) or {}).get("rate")
            rate = role_summary.get("rate")
            role_effects.append(
                {
                    "treatment": treatment,
                    "role": role,
                    "rate": rate,
                    "effect_vs_control": (
                        None
                        if control_rate is None or rate is None
                        else float(rate) - float(control_rate)
                    ),
                    "count": role_summary.get("count", 0),
                }
            )
    return {
        "schema": "cognityx.experiment.figure-data/v1",
        "experiment_id": analysis["experiment_id"],
        "grounded_correct_rate_by_treatment": [
            {
                "treatment": treatment,
                "rate": summary.get("rate"),
                "count": summary.get("count"),
            }
            for treatment, summary in sorted(treatments.items())
        ],
        "effect_by_seed": [
            {"seed": seed, "treatment": treatment, "effect": effect}
            for seed, values in sorted((analysis.get("per_seed_effects") or {}).items())
            for treatment, effect in sorted(values.items())
        ],
        "role_specific_effects": role_effects,
        "cost_to_quality": {
            "resources": plain(analysis.get("resources") or {}),
            "semantic_judge_cost": analysis.get("semantic_judge_invocation_cost", 0),
        },
        "exploratory": True,
        "plot_recommendation": (
            "insufficient_sample_size"
            if sum(int(value.get("count") or 0) for value in treatments.values()) < 4
            else "eligible_for_reviewed_plot"
        ),
    }
