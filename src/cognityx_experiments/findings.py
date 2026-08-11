"""Evidence-grounded ResearchFinding generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cognityx_experiments.canonical import checksum, plain
from cognityx_experiments.contracts import Experiment, ResearchSpec

FINDING_SCHEMA = "cognityx.research.finding/v1"
FINDING_CLASSES = frozenset(
    {
        "confirmatory",
        "exploratory",
        "methodological",
        "operational",
        "negative_result",
        "anomaly",
    }
)
HYPOTHESIS_RELATIONS = frozenset(
    {"supports", "contradicts", "inconclusive", "not_applicable"}
)


@dataclass(frozen=True, slots=True)
class ResearchFinding:
    finding_id: str
    research_area_id: str
    hypothesis_id: str
    research_question_ids: tuple[str, ...]
    experiment_id: str
    execution_id: str
    finding_class: str
    observed_claim: str
    primary_outcome: Mapping[str, Any]
    effect_values: Mapping[str, Any]
    scope: Mapping[str, Any]
    evidence_references: tuple[Mapping[str, Any], ...]
    limitations: tuple[str, ...]
    hypothesis_relation: str
    novelty_candidate: bool
    literature_check_required: bool
    follow_up_questions: tuple[str, ...]
    proposed_next_experiments: tuple[str, ...]
    generator: Mapping[str, Any]
    generated_at: str
    human_review_status: str = "pending"
    exploratory_observations: tuple[str, ...] = ()
    literature_questions: tuple[str, ...] = ()
    human_notes: tuple[str, ...] = ()
    schema: str = FINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.finding_class not in FINDING_CLASSES:
            raise ValueError(f"Unsupported finding_class: {self.finding_class}")
        if self.hypothesis_relation not in HYPOTHESIS_RELATIONS:
            raise ValueError(
                f"Unsupported hypothesis_relation: {self.hypothesis_relation}"
            )
        if self.human_review_status not in {"pending", "reviewed", "rejected"}:
            raise ValueError("Unsupported human_review_status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "finding_id": self.finding_id,
            "research_area_id": self.research_area_id,
            "hypothesis_id": self.hypothesis_id,
            "research_question_ids": list(self.research_question_ids),
            "experiment_id": self.experiment_id,
            "execution_id": self.execution_id,
            "finding_class": self.finding_class,
            "observed": {
                "claim": self.observed_claim,
                "primary_outcome": plain(self.primary_outcome),
                "effect_values": plain(self.effect_values),
                "scope": plain(self.scope),
                "evidence_references": plain(self.evidence_references),
            },
            "confirmatory_interpretation": {
                "hypothesis_relation": self.hypothesis_relation,
                "human_review_required": True,
            },
            "exploratory_observations": list(self.exploratory_observations),
            "literature_questions": list(self.literature_questions),
            "follow_up": {
                "questions": list(self.follow_up_questions),
                "proposed_next_experiments": list(self.proposed_next_experiments),
            },
            "limitations": list(self.limitations),
            "novelty_candidate": self.novelty_candidate,
            "literature_check_required": self.literature_check_required,
            "generator": plain(self.generator),
            "generated_at": self.generated_at,
            "human_review": {
                "status": self.human_review_status,
                "notes": list(self.human_notes),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchFinding:
        observed = dict(value["observed"])
        confirmatory = dict(value["confirmatory_interpretation"])
        follow_up = dict(value.get("follow_up") or {})
        human = dict(value.get("human_review") or {})
        return cls(
            finding_id=str(value["finding_id"]),
            research_area_id=str(value["research_area_id"]),
            hypothesis_id=str(value["hypothesis_id"]),
            research_question_ids=tuple(value["research_question_ids"]),
            experiment_id=str(value["experiment_id"]),
            execution_id=str(value["execution_id"]),
            finding_class=str(value["finding_class"]),
            observed_claim=str(observed["claim"]),
            primary_outcome=dict(observed["primary_outcome"]),
            effect_values=dict(observed["effect_values"]),
            scope=dict(observed["scope"]),
            evidence_references=tuple(observed["evidence_references"]),
            limitations=tuple(value.get("limitations") or ()),
            hypothesis_relation=str(confirmatory["hypothesis_relation"]),
            novelty_candidate=bool(value.get("novelty_candidate", False)),
            literature_check_required=bool(
                value.get("literature_check_required", True)
            ),
            follow_up_questions=tuple(follow_up.get("questions") or ()),
            proposed_next_experiments=tuple(
                follow_up.get("proposed_next_experiments") or ()
            ),
            generator=dict(value["generator"]),
            generated_at=str(value["generated_at"]),
            human_review_status=str(human.get("status") or "pending"),
            exploratory_observations=tuple(value.get("exploratory_observations") or ()),
            literature_questions=tuple(value.get("literature_questions") or ()),
            human_notes=tuple(human.get("notes") or ()),
        )


def generate_finding(
    spec: ResearchSpec,
    experiment: Experiment,
    *,
    execution_id: str,
    analysis: Mapping[str, Any],
    evidence_references: Sequence[Mapping[str, Any]],
    limitations: Sequence[str] = (),
    generated_at: str | None = None,
) -> ResearchFinding:
    """Freeze computed evidence without making an automatic scholarly verdict."""
    questions = {value.research_question_id: value for value in spec.research_questions}
    hypothesis_ids = {
        questions[question_id].hypothesis_id for question_id in experiment.addresses
    }
    if len(hypothesis_ids) != 1:
        raise ValueError("One finding cannot silently combine multiple hypotheses")
    hypothesis_id = next(iter(hypothesis_ids))
    metric = str(analysis["primary_metric"])
    control = str(analysis["control_id"])
    deltas = dict(analysis.get("deltas_from_control") or {})
    comparison_parts = [
        f"{treatment} minus {control} was {float(effect):.6g}"
        for treatment, effect in sorted(deltas.items())
    ]
    count = sum(
        int(value.get("count") or 0)
        for value in (analysis.get("treatments") or {}).values()
    )
    comparison = "; ".join(comparison_parts) or "no treatment contrast was estimable"
    claim = (
        f"Under the frozen plan, {comparison} on {metric} across {count} "
        "finalized observations. This is computed evidence and requires human "
        "scholarly interpretation."
    )
    identity = {
        "research_area_id": spec.research_area.research_area_id,
        "hypothesis_id": hypothesis_id,
        "research_question_ids": list(experiment.addresses),
        "experiment_id": experiment.experiment_id,
        "execution_id": execution_id,
        "analysis_checksum": checksum(analysis),
        "evidence": plain(evidence_references),
    }
    return ResearchFinding(
        finding_id=f"finding-{checksum(identity)[:24]}",
        research_area_id=spec.research_area.research_area_id,
        hypothesis_id=hypothesis_id,
        research_question_ids=experiment.addresses,
        experiment_id=experiment.experiment_id,
        execution_id=execution_id,
        finding_class="confirmatory",
        observed_claim=claim,
        primary_outcome={
            "metric": metric,
            "control_id": control,
            "comparable_pair_count": analysis.get("comparable_pair_count", 0),
            "unresolved_count": analysis.get("unresolved_count", 0),
        },
        effect_values={
            "deltas_from_control": deltas,
            "per_seed_effects": plain(analysis.get("per_seed_effects") or {}),
            "cluster_bootstrap": plain(analysis.get("cluster_bootstrap") or {}),
        },
        scope={
            "research_profile": experiment.design.research_profile,
            "seeds": list(experiment.design.seeds),
            "experimental_unit": experiment.design.experimental_unit,
        },
        evidence_references=tuple(plain(evidence_references)),
        limitations=tuple(limitations)
        or (
            "Effect estimates apply only to the frozen treatments, model, data, "
            "and evaluation suites.",
            "Automatic generation does not establish causality, novelty, or "
            "statistical independence.",
        ),
        hypothesis_relation="inconclusive",
        novelty_candidate=False,
        literature_check_required=True,
        follow_up_questions=(),
        proposed_next_experiments=(),
        generator={
            "component": "cognityx-experiments",
            "mode": "deterministic",
            "analysis_checksum": checksum(analysis),
        },
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        literature_questions=(
            "Which prior studies used a comparable experimental unit and estimand?",
        ),
    )


def factual_note(
    finding: ResearchFinding,
    *,
    enriched: Mapping[str, Any] | None = None,
) -> str:
    """Render a minimum note that never invents metrics or significance."""
    selected = dict(enriched or {})
    exploratory = selected.get("exploratory_observations") or list(
        finding.exploratory_observations
    )
    questions = selected.get("follow_up_questions") or list(finding.follow_up_questions)
    literature = selected.get("literature_checks") or list(finding.literature_questions)
    limitations = selected.get("limitations") or list(finding.limitations)
    return "\n".join(
        [
            f"# Research finding {finding.finding_id}",
            "",
            "## Observed",
            "",
            finding.observed_claim,
            "",
            "## Confirmatory interpretation",
            "",
            "The preregistered relationship remains inconclusive until human review.",
            "",
            "## Limitations",
            "",
            *_bullets(limitations),
            "",
            "## Exploratory observations",
            "",
            *_bullets(exploratory or ["None automatically asserted."]),
            "",
            "## Questions raised",
            "",
            *_bullets(questions or ["No automatic follow-up question was asserted."]),
            "",
            "## Literature checks",
            "",
            *_bullets(literature),
            "",
            "## Suggested next experiments",
            "",
            *_bullets(
                finding.proposed_next_experiments
                or (
                    "Human review should decide whether another experiment is "
                    "warranted.",
                )
            ),
            "",
        ]
    )


def _bullets(values: Sequence[Any]) -> list[str]:
    return [f"- {value}" for value in values]
