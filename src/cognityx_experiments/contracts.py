"""Typed research, logical-plan, and execution-plan contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cognityx_experiments.canonical import checksum, plain

RESEARCH_SPEC_SCHEMA = "cognityx.research.spec/v1"
LOGICAL_PLAN_SCHEMA = "cognityx.experiment.plan/v1"
EXECUTION_PLAN_SCHEMA = "cognityx.experiment.execution-plan/v1"
TRAINING_RECIPE = "training_treatment_comparison"
EVALUATOR_RECIPE = "evaluator_method_comparison"
KNOWN_RECIPES = frozenset({TRAINING_RECIPE, EVALUATOR_RECIPE})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _required(value: Any, name: str) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValueError(f"{name} must be a non-empty string")
    return selected


def _identifier(value: Any, name: str) -> str:
    selected = _required(value, name)
    if _ID.fullmatch(selected) is None:
        raise ValueError(f"{name} contains unsupported characters: {selected}")
    return selected


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return plain(value)


def _items(value: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"Every {name} item must be a mapping")
    return list(value)


@dataclass(frozen=True, slots=True)
class ResearchArea:
    research_area_id: str
    title: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "id": self.research_area_id,
                "title": self.title,
                "description": self.description,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    null_hypothesis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "id": self.hypothesis_id,
                "statement": self.statement,
                "null_hypothesis": self.null_hypothesis,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    research_question_id: str
    question: str
    hypothesis_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.research_question_id,
            "question": self.question,
            "hypothesis_id": self.hypothesis_id,
        }


@dataclass(frozen=True, slots=True)
class Outcome:
    metric: str
    role: str | None = None
    direction: str | None = None
    aggregation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "metric": self.metric,
                "role": self.role,
                "direction": self.direction,
                "aggregation": self.aggregation,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class Treatment:
    treatment_id: str
    role: str
    inputs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.treatment_id,
            "role": self.role,
            "inputs": plain(self.inputs),
        }


@dataclass(frozen=True, slots=True)
class ExperimentalDesign:
    design_type: str
    experimental_unit: str
    treatments: tuple[Treatment, ...]
    control_id: str
    comparator_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    primary_outcome: Outcome
    secondary_outcomes: tuple[Outcome, ...] = ()
    blocking_variables: tuple[str, ...] = ()
    controlled_variables: Mapping[str, Any] = field(default_factory=dict)
    estimand: Mapping[str, Any] = field(default_factory=dict)
    stopping_rule: Mapping[str, Any] = field(default_factory=dict)
    analysis_plan: Mapping[str, Any] = field(default_factory=dict)
    resource_constraints: Mapping[str, Any] = field(default_factory=dict)
    evaluation_suites: tuple[Mapping[str, Any], ...] = ()
    declared_exclusions: tuple[Mapping[str, Any], ...] = ()
    research_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.design_type,
            "experimental_unit": self.experimental_unit,
            "treatments": [value.to_dict() for value in self.treatments],
            "control": self.control_id,
            "comparators": list(self.comparator_ids),
            "seeds": list(self.seeds),
            "primary_outcome": self.primary_outcome.to_dict(),
            "secondary_outcomes": [
                value.to_dict() for value in self.secondary_outcomes
            ],
            "blocking_variables": list(self.blocking_variables),
            "controlled_variables": plain(self.controlled_variables),
            "estimand": plain(self.estimand),
            "stopping_rule": plain(self.stopping_rule),
            "analysis_plan": plain(self.analysis_plan),
            "resource_constraints": plain(self.resource_constraints),
            "evaluation_suites": plain(self.evaluation_suites),
            "declared_exclusions": plain(self.declared_exclusions),
            **(
                {"research_profile": self.research_profile}
                if self.research_profile
                else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class Experiment:
    experiment_id: str
    addresses: tuple[str, ...]
    recipe: str
    design: ExperimentalDesign
    execution: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.experiment_id,
            "addresses": list(self.addresses),
            "recipe": self.recipe,
            "design": self.design.to_dict(),
            "execution": plain(self.execution),
        }


@dataclass(frozen=True, slots=True)
class ResearchSpec:
    research_area: ResearchArea
    hypotheses: tuple[Hypothesis, ...]
    research_questions: tuple[ResearchQuestion, ...]
    experiments: tuple[Experiment, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = RESEARCH_SPEC_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchSpec:
        schema = value.get("schema") or value.get("schema_version")
        if schema != RESEARCH_SPEC_SCHEMA:
            raise ValueError(f"Unsupported research schema: {schema}")
        area_value = _mapping(value.get("research_area"), "research_area")
        area = ResearchArea(
            research_area_id=_identifier(area_value.get("id"), "research_area.id"),
            title=str(area_value["title"]) if area_value.get("title") else None,
            description=(
                str(area_value["description"])
                if area_value.get("description")
                else None
            ),
        )
        hypotheses_value = value.get("hypotheses")
        if hypotheses_value is None and value.get("hypothesis") is not None:
            hypotheses_value = [value["hypothesis"]]
        hypotheses = tuple(
            _parse_hypothesis(item) for item in _items(hypotheses_value, "hypotheses")
        )
        questions = tuple(
            _parse_question(item)
            for item in _items(value.get("research_questions"), "research_questions")
        )
        experiments = tuple(
            _parse_experiment(item)
            for item in _items(value.get("experiments"), "experiments")
        )
        spec = cls(
            research_area=area,
            hypotheses=hypotheses,
            research_questions=questions,
            experiments=experiments,
            metadata=_mapping(value.get("metadata"), "metadata"),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.hypotheses or not self.research_questions or not self.experiments:
            raise ValueError(
                "ResearchSpec requires hypotheses, questions, and experiments"
            )
        hypothesis_ids = _unique(
            [item.hypothesis_id for item in self.hypotheses], "hypothesis"
        )
        question_ids = _unique(
            [item.research_question_id for item in self.research_questions],
            "research question",
        )
        _unique([item.experiment_id for item in self.experiments], "experiment")
        for question in self.research_questions:
            if question.hypothesis_id not in hypothesis_ids:
                raise ValueError(
                    f"Research question {question.research_question_id} links unknown "
                    f"hypothesis {question.hypothesis_id}"
                )
        for experiment in self.experiments:
            unknown = set(experiment.addresses) - question_ids
            if unknown:
                raise ValueError(
                    f"Experiment {experiment.experiment_id} addresses unknown "
                    "questions: "
                    f"{', '.join(sorted(unknown))}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "research_area": self.research_area.to_dict(),
            "hypotheses": [value.to_dict() for value in self.hypotheses],
            "research_questions": [
                value.to_dict() for value in self.research_questions
            ],
            "experiments": [value.to_dict() for value in self.experiments],
            "metadata": plain(self.metadata),
        }

    @property
    def spec_checksum(self) -> str:
        return checksum(self.to_dict())


@dataclass(frozen=True, slots=True)
class LogicalRun:
    run_key: str
    experiment_id: str
    treatment_id: str
    treatment_role: str
    seed: int
    inputs: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "experiment_id": self.experiment_id,
            "treatment_id": self.treatment_id,
            "treatment_role": self.treatment_role,
            "seed": self.seed,
            "inputs": plain(self.inputs),
        }


@dataclass(frozen=True, slots=True)
class LogicalExperimentPlan:
    spec_checksum: str
    research_area_id: str
    experiments: tuple[Mapping[str, Any], ...]
    runs: tuple[LogicalRun, ...]
    schema: str = LOGICAL_PLAN_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "spec_checksum": self.spec_checksum,
            "research_area_id": self.research_area_id,
            "experiments": plain(self.experiments),
            "runs": [run.to_dict() for run in self.runs],
        }

    @property
    def plan_checksum(self) -> str:
        return checksum(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "plan_checksum": self.plan_checksum}


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    step_id: str
    operation: str
    component: str
    experiment_id: str
    treatment_id: str | None
    seed: int | None
    dependencies: tuple[str, ...]
    input_references: Mapping[str, Any]
    output_contract: str
    idempotency_key: str
    resource_requirements: Mapping[str, Any]
    retry_policy: Mapping[str, Any]
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "component": self.component,
            "experiment_id": self.experiment_id,
            "treatment_id": self.treatment_id,
            "seed": self.seed,
            "dependencies": list(self.dependencies),
            "input_references": plain(self.input_references),
            "output_contract": self.output_contract,
            "idempotency_key": self.idempotency_key,
            "resource_requirements": plain(self.resource_requirements),
            "retry_policy": plain(self.retry_policy),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    execution_id: str
    plan_checksum: str
    spec_checksum: str
    inference_service: Mapping[str, Any]
    steps: tuple[ExecutionStep, ...]
    schema: str = EXECUTION_PLAN_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_id": self.execution_id,
            "plan_checksum": self.plan_checksum,
            "spec_checksum": self.spec_checksum,
            "inference_service": plain(self.inference_service),
            "steps": [step.to_dict() for step in self.steps],
        }

    @property
    def execution_plan_checksum(self) -> str:
        return checksum(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload(),
            "execution_plan_checksum": self.execution_plan_checksum,
        }


def _parse_hypothesis(value: Mapping[str, Any]) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=_identifier(value.get("id"), "hypothesis.id"),
        statement=_required(value.get("statement"), "hypothesis.statement"),
        null_hypothesis=(
            str(value["null_hypothesis"]) if value.get("null_hypothesis") else None
        ),
    )


def _parse_question(value: Mapping[str, Any]) -> ResearchQuestion:
    return ResearchQuestion(
        research_question_id=_identifier(value.get("id"), "research_question.id"),
        question=_required(value.get("question"), "research_question.question"),
        hypothesis_id=_identifier(
            value.get("hypothesis_id") or value.get("hypothesis"),
            "research_question.hypothesis_id",
        ),
    )


def _parse_outcome(value: Any, name: str) -> Outcome:
    selected = _mapping(value, name)
    return Outcome(
        metric=_identifier(selected.get("metric"), f"{name}.metric"),
        role=str(selected["role"]) if selected.get("role") else None,
        direction=str(selected["direction"]) if selected.get("direction") else None,
        aggregation=str(selected["aggregation"])
        if selected.get("aggregation")
        else None,
    )


def _parse_treatment(value: Mapping[str, Any]) -> Treatment:
    role = str(value.get("role") or "treatment")
    if role not in {"treatment", "control", "comparator"}:
        raise ValueError(f"Unsupported treatment role: {role}")
    return Treatment(
        treatment_id=_identifier(value.get("id"), "treatment.id"),
        role=role,
        inputs=_mapping(value.get("inputs"), "treatment.inputs"),
    )


def _parse_experiment(value: Mapping[str, Any]) -> Experiment:
    experiment_id = _identifier(value.get("id"), "experiment.id")
    recipe = _identifier(value.get("recipe"), "experiment.recipe")
    if recipe not in KNOWN_RECIPES:
        raise ValueError(f"Unsupported experiment recipe: {recipe}")
    addresses_value = value.get("addresses")
    if not isinstance(addresses_value, Sequence) or isinstance(addresses_value, str):
        raise ValueError(f"Experiment {experiment_id} addresses must be a list")
    addresses = tuple(
        _identifier(item, "experiment.addresses") for item in addresses_value
    )
    if not addresses:
        raise ValueError(
            f"Experiment {experiment_id} must address at least one question"
        )
    design_value = _mapping(value.get("design"), "experiment.design")
    treatments = tuple(
        _parse_treatment(item)
        for item in _items(design_value.get("treatments"), "treatments")
    )
    if len(treatments) < 2:
        raise ValueError(f"Experiment {experiment_id} requires at least two treatments")
    treatment_ids = _unique([item.treatment_id for item in treatments], "treatment")
    control_id = _identifier(
        design_value.get("control")
        or next(
            (item.treatment_id for item in treatments if item.role == "control"), None
        ),
        "design.control",
    )
    if control_id not in treatment_ids:
        raise ValueError(f"Experiment {experiment_id} control is not a treatment")
    seeds_value = design_value.get("seeds") or []
    if not isinstance(seeds_value, Sequence) or isinstance(seeds_value, str):
        raise ValueError("design.seeds must be a list")
    seeds = tuple(int(seed) for seed in seeds_value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Experiment {experiment_id} requires unique seeds")
    secondary = design_value.get("secondary_outcomes") or []
    design = ExperimentalDesign(
        design_type=_identifier(design_value.get("type"), "design.type"),
        experimental_unit=_required(
            design_value.get("experimental_unit"), "design.experimental_unit"
        ),
        treatments=treatments,
        control_id=control_id,
        comparator_ids=tuple(
            _identifier(item, "design.comparators")
            for item in (design_value.get("comparators") or [])
        ),
        seeds=seeds,
        primary_outcome=_parse_outcome(
            design_value.get("primary_outcome"), "design.primary_outcome"
        ),
        secondary_outcomes=tuple(
            _parse_outcome(item, "design.secondary_outcomes") for item in secondary
        ),
        blocking_variables=tuple(
            str(item) for item in (design_value.get("blocking_variables") or [])
        ),
        controlled_variables=_mapping(
            design_value.get("controlled_variables"), "design.controlled_variables"
        ),
        estimand=_mapping(design_value.get("estimand"), "design.estimand"),
        stopping_rule=_mapping(
            design_value.get("stopping_rule"), "design.stopping_rule"
        ),
        analysis_plan=_mapping(
            design_value.get("analysis_plan"), "design.analysis_plan"
        ),
        resource_constraints=_mapping(
            design_value.get("resource_constraints"), "design.resource_constraints"
        ),
        evaluation_suites=tuple(
            _mapping(item, "design.evaluation_suites")
            for item in (design_value.get("evaluation_suites") or [])
        ),
        declared_exclusions=tuple(
            _mapping(item, "design.declared_exclusions")
            for item in (design_value.get("declared_exclusions") or [])
        ),
        research_profile=(
            str(design_value["research_profile"])
            if design_value.get("research_profile")
            else None
        ),
    )
    return Experiment(
        experiment_id=experiment_id,
        addresses=addresses,
        recipe=recipe,
        design=design,
        execution=_mapping(value.get("execution"), "experiment.execution"),
    )


def _unique(values: Sequence[str], name: str) -> set[str]:
    selected = set(values)
    if len(selected) != len(values):
        raise ValueError(f"Duplicate {name} ID")
    return selected
