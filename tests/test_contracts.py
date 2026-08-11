from copy import deepcopy

import pytest

from cognityx_experiments.contracts import ResearchSpec


def test_hierarchy_and_checksum_are_canonical(research_spec: ResearchSpec) -> None:
    value = research_spec.to_dict()
    reordered = {key: value[key] for key in reversed(value)}
    rebuilt = ResearchSpec.from_mapping(reordered)

    assert rebuilt.spec_checksum == research_spec.spec_checksum
    assert research_spec.experiments[0].addresses == ("POLICY-RQ1",)
    assert research_spec.research_questions[0].hypothesis_id == "POLICY-H1"


def test_unknown_question_and_duplicate_ids_fail(research_spec: ResearchSpec) -> None:
    value = deepcopy(research_spec.to_dict())
    value["experiments"][0]["addresses"] = ["UNKNOWN-RQ"]
    with pytest.raises(ValueError, match="unknown questions"):
        ResearchSpec.from_mapping(value)

    value = deepcopy(research_spec.to_dict())
    value["hypotheses"].append(value["hypotheses"][0])
    with pytest.raises(ValueError, match="Duplicate hypothesis"):
        ResearchSpec.from_mapping(value)


def test_experiment_can_address_multiple_questions(research_spec: ResearchSpec) -> None:
    value = deepcopy(research_spec.to_dict())
    value["research_questions"].append(
        {
            "id": "POLICY-RQ2",
            "hypothesis_id": "POLICY-H1",
            "question": "Does the observed effect vary by role?",
        }
    )
    value["experiments"][0]["addresses"].append("POLICY-RQ2")
    parsed = ResearchSpec.from_mapping(value)
    assert parsed.experiments[0].addresses == ("POLICY-RQ1", "POLICY-RQ2")


def test_optional_research_lineage_participates_in_checksum(
    research_spec: ResearchSpec,
) -> None:
    value = research_spec.to_dict()
    without_lineage = deepcopy(value)
    without_lineage.pop("lineage")

    assert research_spec.lineage is not None
    assert research_spec.lineage.research_role == "confirmatory"
    assert (
        ResearchSpec.from_mapping(without_lineage).spec_checksum
        != research_spec.spec_checksum
    )


def test_multiple_declared_roles_require_primary_role(
    research_spec: ResearchSpec,
) -> None:
    value = research_spec.to_dict()
    value["experiments"][0]["design"]["primary_outcome"].pop("role")

    with pytest.raises(ValueError, match="primary_outcome.role"):
        ResearchSpec.from_mapping(value)
