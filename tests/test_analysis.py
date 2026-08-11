import pytest

from cognityx_experiments.analysis import analyse_records


def _record(
    treatment: str,
    role: str,
    value: bool,
    *,
    record_id: str = "R1",
    seed: int = 11,
    primary_finalized: bool = True,
    full_finalized: bool = True,
) -> dict:
    return {
        "evaluation_record_id": record_id,
        "treatment_id": treatment,
        "seed": seed,
        "research_role": role,
        "grounded_correct": value,
        "primary_endpoint_finalized": primary_finalized,
        "full_evaluation_finalized": full_finalized,
        "answer_correctness": value,
        "required_fact_completeness": value,
        "fatal_contradiction": False,
        "source_faithfulness": True if full_finalized else None,
        "generation_status": "completed",
        "evaluator_status": "finalized" if full_finalized else "needs_review",
        "knowledge_unit_id": f"unit-{record_id}",
        "fact_group_id": f"fact-{record_id}",
        "document_id": "doc-1",
        "resources": {"gpu_seconds": 2},
        "semantic_judge_invocation_cost": 0.01,
    }


def test_primary_role_prevents_exact_recall_from_contaminating_paraphrase() -> None:
    records = [
        _record("raw", "exact_recall", False),
        _record("qualified", "exact_recall", True),
        _record("raw", "paraphrase_evaluation", False),
        _record("qualified", "paraphrase_evaluation", False),
        _record("raw", "heldout_knowledge_unit", False),
        _record("qualified", "heldout_knowledge_unit", True),
    ]

    result = analyse_records(
        experiment_id="EXP-1",
        control_id="raw",
        primary_metric="grounded_correct",
        primary_role="paraphrase_evaluation",
        records=records,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert result["primary_role"] == "paraphrase_evaluation"
    assert result["deltas_from_control"] == {"qualified": 0.0}
    assert result["role_diagnostics"]["exact_recall"]["rate"] == 0.5
    assert result["role_diagnostics"]["paraphrase_evaluation"]["rate"] == 0.0
    assert result["role_diagnostics"]["heldout_knowledge_unit"]["rate"] == 0.5
    assert result["role_diagnostics"]["generalization"]["count"] == 4


def test_omitted_primary_role_resolves_one_role_and_rejects_many() -> None:
    one_role = [
        _record("raw", "paraphrase_evaluation", False),
        _record("qualified", "paraphrase_evaluation", True),
    ]
    result = analyse_records(
        experiment_id="EXP-1",
        control_id="raw",
        primary_metric="grounded_correct",
        records=one_role,
    )
    assert result["primary_role"] == "paraphrase_evaluation"

    with pytest.raises(ValueError, match="ambiguous"):
        analyse_records(
            experiment_id="EXP-1",
            control_id="raw",
            primary_metric="grounded_correct",
            records=[*one_role, _record("raw", "exact_recall", True)],
        )


def test_primary_finalization_is_independent_of_full_evaluation() -> None:
    records = [
        _record(
            "raw",
            "paraphrase_evaluation",
            False,
            full_finalized=False,
        ),
        _record(
            "qualified",
            "paraphrase_evaluation",
            True,
            full_finalized=False,
        ),
    ]
    result = analyse_records(
        experiment_id="EXP-1",
        control_id="raw",
        primary_metric="grounded_correct",
        primary_role="paraphrase_evaluation",
        records=records,
    )

    assert result["deltas_from_control"] == {"qualified": 1.0}
    assert result["primary_endpoint_unresolved_count"] == 0
    assert result["full_evaluation_unresolved_count"] == 2


def test_saved_adapter_endpoint_is_used_instead_of_pair_outcome_or_base() -> None:
    raw = _record("raw", "paraphrase_evaluation", False)
    raw.update({"pair_outcome": "adapter_win", "base": {"grounded_correct": True}})
    qualified = _record("qualified", "paraphrase_evaluation", True)
    qualified.update({"pair_outcome": "base_win", "base": {"grounded_correct": False}})

    result = analyse_records(
        experiment_id="EXP-1",
        control_id="raw",
        primary_metric="grounded_correct",
        primary_role="paraphrase_evaluation",
        records=[raw, qualified],
    )

    assert result["deltas_from_control"] == {"qualified": 1.0}
    contrast = result["contrasts_from_control"]["qualified"]
    assert contrast["control_rate"] == 0.0
    assert contrast["treatment_rate"] == 1.0


def test_missing_treatment_endpoint_is_unresolved_not_a_tie() -> None:
    records = [
        _record("raw", "paraphrase_evaluation", False, record_id="R1"),
        _record("qualified", "paraphrase_evaluation", True, record_id="R1"),
        _record("raw", "paraphrase_evaluation", False, record_id="R2"),
    ]
    result = analyse_records(
        experiment_id="EXP-1",
        control_id="raw",
        primary_metric="grounded_correct",
        primary_role="paraphrase_evaluation",
        records=records,
        bootstrap_samples=20,
    )

    contrast = result["contrasts_from_control"]["qualified"]
    assert contrast["finalized_paired_count"] == 1
    assert contrast["unresolved_paired_count"] == 1
    assert contrast["paired_delta"] == 1.0
    assert result["cluster_bootstrap"]["cluster_field"] == "knowledge_unit_id"
    assert result["resources"] == {"gpu_seconds": 6.0}
    assert result["semantic_judge_invocation_cost"] == pytest.approx(0.03)


def test_all_unresolved_endpoints_produce_an_honest_terminal_analysis() -> None:
    records = [
        _record(
            "raw",
            "paraphrase_evaluation",
            False,
            primary_finalized=False,
            full_finalized=False,
        ),
        _record(
            "qualified",
            "paraphrase_evaluation",
            False,
            primary_finalized=False,
            full_finalized=False,
        ),
    ]
    for record in records:
        record["grounded_correct"] = None
        record["answer_correctness"] = "unresolved"

    result = analyse_records(
        experiment_id="EXP-1",
        control_id="raw",
        primary_metric="grounded_correct",
        primary_role="paraphrase_evaluation",
        records=records,
        bootstrap_samples=20,
    )

    assert result["treatments"]["raw"]["count"] == 0
    assert result["treatments"]["raw"]["rate"] is None
    assert result["treatments"]["qualified"]["count"] == 0
    assert result["treatments"]["qualified"]["rate"] is None
    contrast = result["contrasts_from_control"]["qualified"]
    assert contrast == {
        "finalized_paired_count": 0,
        "unresolved_paired_count": 1,
        "control_rate": None,
        "treatment_rate": None,
        "paired_delta": None,
    }
    assert result["deltas_from_control"] == {}
    assert result["comparable_pair_count"] == 0
    assert result["primary_endpoint_unresolved_count"] == 2
    assert result["full_evaluation_unresolved_count"] == 2
    assert result["cluster_bootstrap"] == {
        "cluster_field": None,
        "samples": 0,
        "effects": {},
    }
    assert result["interpretation_status"] == "human_review_required"


def test_missing_control_records_still_fail_closed() -> None:
    with pytest.raises(ValueError, match="primary outcome records for control raw"):
        analyse_records(
            experiment_id="EXP-1",
            control_id="raw",
            primary_metric="grounded_correct",
            primary_role="paraphrase_evaluation",
            records=[_record("qualified", "paraphrase_evaluation", True)],
        )
