from cognityx_experiments.analysis import analyse_records


def test_analysis_separates_roles_seeds_cost_and_cluster_bootstrap() -> None:
    records = []
    for seed in (11, 29):
        for treatment, value in (("control", 0.5), ("qualified", 0.75)):
            for role, cluster in (("exact_recall", "fact-1"), ("paraphrase", "fact-2")):
                records.append(
                    {
                        "treatment_id": treatment,
                        "seed": seed,
                        "role": role,
                        "knowledge_unit_id": cluster,
                        "status": "finalized",
                        "metrics": {"grounded_correct": value},
                        "resources": {"gpu_seconds": 2},
                        "semantic_judge_invocation_cost": 0.01,
                    }
                )
    records.append({"treatment_id": "qualified", "seed": 29, "status": "unresolved"})
    result = analyse_records(
        experiment_id="EXP-1",
        control_id="control",
        primary_metric="grounded_correct",
        records=records,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert result["deltas_from_control"] == {"qualified": 0.25}
    assert result["per_seed_effects"] == {
        "11": {"qualified": 0.25},
        "29": {"qualified": 0.25},
    }
    assert result["unresolved_count"] == 1
    assert result["role_diagnostics"]["exact_recall"]["count"] == 4
    assert result["role_diagnostics"]["paraphrase_generalization"]["count"] == 4
    assert result["cluster_bootstrap"]["cluster_field"] == "knowledge_unit_id"
    assert result["resources"]["gpu_seconds"] == 16
    assert result["semantic_judge_invocation_cost"] == 0.08
    assert result["interpretation_status"] == "human_review_required"
