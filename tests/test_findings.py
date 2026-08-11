import json
from typing import Any

import pytest

from cognityx_experiments.findings import factual_note, generate_finding
from cognityx_experiments.synthesis import FindingSynthesizer


def _analysis() -> dict[str, Any]:
    return {
        "experiment_id": "POLICY-EXP-001",
        "primary_metric": "grounded_correct",
        "control_id": "control",
        "deltas_from_control": {"qualified": 0.25},
        "treatments": {
            "control": {"count": 4, "rate": 0.5},
            "qualified": {"count": 4, "rate": 0.75},
        },
        "per_seed_effects": {
            "11": {"qualified": 0.25},
            "29": {"qualified": 0.25},
        },
        "cluster_bootstrap": {"lower": 0.1, "upper": 0.4},
        "comparable_pair_count": 8,
        "unresolved_count": 1,
    }


def test_research_finding_is_deterministic_and_conservative(research_spec) -> None:
    experiment = research_spec.experiments[0]
    arguments = {
        "execution_id": "execution-fixed",
        "analysis": _analysis(),
        "evidence_references": [
            {"manifest_uri": "storage://analysis.json", "checksum": "abc"}
        ],
        "generated_at": "2026-08-11T00:00:00+00:00",
    }

    first = generate_finding(research_spec, experiment, **arguments)
    second = generate_finding(research_spec, experiment, **arguments)

    assert first.to_dict() == second.to_dict()
    assert first.hypothesis_relation == "inconclusive"
    assert first.novelty_candidate is False
    assert "0.25" in first.observed_claim
    assert "8 finalized observations" in first.observed_claim
    note = factual_note(first)
    assert first.observed_claim in note
    assert "significant" not in note.lower()
    assert "caused" not in note.lower()


class _InferenceClient:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def chat(self, **parameters: Any) -> dict[str, Any]:
        self.calls.append(parameters)
        return {"content": self.content}


def test_synthesizer_uses_only_cognityx_inference_typed_boundary() -> None:
    client = _InferenceClient(
        json.dumps(
            {
                "result_summary": "The frozen contrast was 0.25.",
                "limitations": ["One declared setup."],
                "exploratory_observations": [],
                "follow_up_questions": ["Does the effect replicate?"],
                "literature_checks": ["Find comparable units."],
            }
        )
    )
    synthesizer = FindingSynthesizer(
        client, model="example/research-model", model_revision="revision-1"
    )

    result = synthesizer.synthesize({"analysis": _analysis()})

    assert result["result_summary"].endswith("0.25.")
    assert client.calls[0]["thinking"] == "disabled"
    assert client.calls[0]["temperature"] == 0
    assert synthesizer.generator_identity["boundary"] == "cognityx-inference"


def test_synthesizer_rejects_fields_that_could_rewrite_evidence() -> None:
    client = _InferenceClient({"result_summary": "summary", "p_value": 0.01})
    synthesizer = FindingSynthesizer(client, model="example/research-model")

    with pytest.raises(ValueError, match="unsupported fields"):
        synthesizer.synthesize({"analysis": _analysis()})
