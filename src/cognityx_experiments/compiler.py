"""Deterministic ResearchSpec compilers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cognityx_experiments.canonical import checksum, plain
from cognityx_experiments.contracts import (
    EVALUATOR_RECIPE,
    TRAINING_RECIPE,
    ExecutionPlan,
    ExecutionStep,
    LogicalExperimentPlan,
    LogicalRun,
    ResearchSpec,
)

_FACTUAL_INFERENCE_DEFAULTS: dict[str, Any] = {
    "thinking": "disabled",
    "max_output_tokens": 512,
    "temperature": 0,
    "top_p": 1,
}
_RETRY = {"max_attempts": 1, "backoff_seconds": 0}


def compile_logical_plan(spec: ResearchSpec) -> LogicalExperimentPlan:
    """Expand treatments and seeds without making scheduling decisions."""
    runs: list[LogicalRun] = []
    experiments: list[Mapping[str, Any]] = []
    for experiment in spec.experiments:
        execution = plain(experiment.execution)
        if experiment.design.research_profile == "factual_knowledge_acquisition":
            inference = dict(execution.get("inference") or {})
            for name, value in _FACTUAL_INFERENCE_DEFAULTS.items():
                inference.setdefault(name, value)
            execution["inference"] = inference
        for treatment in experiment.design.treatments:
            for seed in experiment.design.seeds:
                runs.append(
                    LogicalRun(
                        run_key=f"{experiment.experiment_id}:{treatment.treatment_id}:{seed}",
                        experiment_id=experiment.experiment_id,
                        treatment_id=treatment.treatment_id,
                        treatment_role=treatment.role,
                        seed=seed,
                        inputs={
                            "treatment": treatment.to_dict(),
                            "model": plain(execution.get("model") or {}),
                            "dataforge": plain(execution.get("dataforge") or {}),
                            "training": plain(execution.get("training") or {}),
                            "inference": plain(execution.get("inference") or {}),
                            "evaluator": plain(execution.get("evaluator") or {}),
                            "evaluation_suites": plain(
                                experiment.design.evaluation_suites
                            ),
                        },
                    )
                )
        experiments.append(
            {
                "experiment_id": experiment.experiment_id,
                "addresses": list(experiment.addresses),
                "recipe": experiment.recipe,
                "design": experiment.design.to_dict(),
                "execution": execution,
            }
        )
    return LogicalExperimentPlan(
        spec_checksum=spec.spec_checksum,
        research_area_id=spec.research_area.research_area_id,
        experiments=tuple(experiments),
        runs=tuple(runs),
    )


def compile_execution_plan(
    logical: LogicalExperimentPlan,
    *,
    execution_id: str | None = None,
) -> ExecutionPlan:
    """Create a conservative topological schedule of known operations."""
    selected_execution_id = execution_id or f"execution-{logical.plan_checksum[:20]}"
    steps: list[ExecutionStep] = []
    service_modes: set[str] = set()
    for experiment in logical.experiments:
        experiment_id = str(experiment["experiment_id"])
        recipe = str(experiment["recipe"])
        design = dict(experiment["design"])
        execution = dict(experiment["execution"])
        inference = dict(execution.get("inference") or {})
        service = dict(inference.get("service") or {})
        mode = str(service.get("mode") or "external")
        if mode not in {"external", "local_managed"}:
            raise ValueError(f"Unsupported inference service mode: {mode}")
        service_modes.add(mode)
        experiment_runs = [
            run for run in logical.runs if run.experiment_id == experiment_id
        ]
        if recipe == TRAINING_RECIPE:
            steps.extend(
                _training_steps(
                    logical.plan_checksum,
                    selected_execution_id,
                    experiment_id,
                    experiment_runs,
                    design,
                    execution,
                )
            )
        elif recipe == EVALUATOR_RECIPE:
            steps.extend(
                _unsupported_evaluator_steps(
                    logical.plan_checksum,
                    selected_execution_id,
                    experiment_id,
                    experiment_runs,
                    design,
                    execution,
                )
            )
        else:  # pragma: no cover - ResearchSpec validation owns this boundary
            raise ValueError(f"Unsupported recipe: {recipe}")
    inference_service = {
        "modes": sorted(service_modes),
        "scheduling": "shared_compatible_runtime_window",
        "cold_load_policy": "start_or_reuse_once_per_experiment",
    }
    return ExecutionPlan(
        execution_id=selected_execution_id,
        plan_checksum=logical.plan_checksum,
        spec_checksum=logical.spec_checksum,
        inference_service=inference_service,
        steps=tuple(steps),
    )


def _training_steps(
    plan_checksum: str,
    execution_id: str,
    experiment_id: str,
    runs: list[LogicalRun],
    design: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> list[ExecutionStep]:
    steps: list[ExecutionStep] = []
    treatments = {run.treatment_id for run in runs}
    prepare_ids: dict[str, str] = {}
    for treatment_id in sorted(treatments):
        run = next(item for item in runs if item.treatment_id == treatment_id)
        step_id = f"{experiment_id}:prepare:{treatment_id}"
        prepare_ids[treatment_id] = step_id
        steps.append(
            _step(
                plan_checksum,
                execution_id,
                step_id=step_id,
                operation="prepare_data",
                component="dataforge",
                experiment_id=experiment_id,
                treatment_id=treatment_id,
                seed=None,
                dependencies=(),
                inputs=run.inputs,
                output_contract="cognityx.dataforge.research-package/v1",
                resources={"gpu": False},
            )
        )
    train_ids: list[str] = []
    for run in runs:
        step_id = f"{experiment_id}:train:{run.treatment_id}:{run.seed}"
        train_ids.append(step_id)
        steps.append(
            _step(
                plan_checksum,
                execution_id,
                step_id=step_id,
                operation="train",
                component="training",
                experiment_id=experiment_id,
                treatment_id=run.treatment_id,
                seed=run.seed,
                dependencies=(prepare_ids[run.treatment_id],),
                inputs=run.inputs,
                output_contract="cognityx.training.publication/v1",
                resources=dict(design.get("resource_constraints") or {}),
            )
        )
    start_id = f"{experiment_id}:inference:start-or-reuse"
    steps.append(
        _step(
            plan_checksum,
            execution_id,
            step_id=start_id,
            operation="start_or_reuse_inference",
            component="inference",
            experiment_id=experiment_id,
            treatment_id=None,
            seed=None,
            dependencies=tuple(train_ids),
            inputs={
                "model": plain(execution.get("model") or {}),
                "inference": plain(execution.get("inference") or {}),
            },
            output_contract="cognityx.inference.runtime-reference/v1",
            resources=dict(design.get("resource_constraints") or {}),
        )
    )
    inference_ids: list[str] = []
    for run in runs:
        step_id = f"{experiment_id}:infer:{run.treatment_id}:{run.seed}"
        inference_ids.append(step_id)
        train_id = f"{experiment_id}:train:{run.treatment_id}:{run.seed}"
        steps.append(
            _step(
                plan_checksum,
                execution_id,
                step_id=step_id,
                operation="execute_inference_pair",
                component="inference",
                experiment_id=experiment_id,
                treatment_id=run.treatment_id,
                seed=run.seed,
                dependencies=(start_id, train_id),
                inputs=run.inputs,
                output_contract="cognityx.inference.pair/v1",
                resources={"reuse_resident_base_model": True},
            )
        )
    stop_id = f"{experiment_id}:inference:stop"
    steps.append(
        _step(
            plan_checksum,
            execution_id,
            step_id=stop_id,
            operation="stop_inference",
            component="inference",
            experiment_id=experiment_id,
            treatment_id=None,
            seed=None,
            dependencies=tuple(inference_ids),
            inputs={
                "service": plain(
                    (execution.get("inference") or {}).get("service") or {}
                )
            },
            output_contract="cognityx.inference.runtime-stop/v1",
            resources={"gpu": False},
        )
    )
    evaluator_ids: list[str] = []
    for run, inference_id in zip(runs, inference_ids, strict=True):
        step_id = f"{experiment_id}:evaluate:{run.treatment_id}:{run.seed}"
        evaluator_ids.append(step_id)
        steps.append(
            _step(
                plan_checksum,
                execution_id,
                step_id=step_id,
                operation="evaluate_pair",
                component="evaluator",
                experiment_id=experiment_id,
                treatment_id=run.treatment_id,
                seed=run.seed,
                dependencies=(inference_id,),
                inputs={
                    "evaluator": plain(execution.get("evaluator") or {}),
                    "treatment_role": run.treatment_role,
                    "outcomes": {
                        "primary": plain(design.get("primary_outcome") or {}),
                        "secondary": plain(design.get("secondary_outcomes") or []),
                    },
                },
                output_contract="cognityx.evaluator.run/v1",
                resources={"gpu": False},
            )
        )
    steps.append(
        _step(
            plan_checksum,
            execution_id,
            step_id=f"{experiment_id}:analyse",
            operation="analyse_experiment",
            component="experiments",
            experiment_id=experiment_id,
            treatment_id=None,
            seed=None,
            dependencies=tuple([*evaluator_ids, stop_id]),
            inputs={
                "control": design.get("control"),
                "analysis_plan": plain(design.get("analysis_plan") or {}),
                "primary_outcome": plain(design.get("primary_outcome") or {}),
            },
            output_contract="cognityx.experiment.analysis/v1",
            resources={"gpu": False},
        )
    )
    return steps


def _unsupported_evaluator_steps(
    plan_checksum: str,
    execution_id: str,
    experiment_id: str,
    runs: list[LogicalRun],
    design: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> list[ExecutionStep]:
    steps: list[ExecutionStep] = []
    for run in runs:
        steps.append(
            _step(
                plan_checksum,
                execution_id,
                step_id=f"{experiment_id}:evaluate-method:{run.treatment_id}:{run.seed}",
                operation="evaluate_method_variant",
                component="evaluator",
                experiment_id=experiment_id,
                treatment_id=run.treatment_id,
                seed=run.seed,
                dependencies=(),
                inputs={
                    "frozen_evidence": plain(execution),
                    "variant": plain(run.inputs),
                },
                output_contract="unsupported-until-evaluator-variant-public-api",
                resources={"gpu": False},
                status="unsupported",
            )
        )
    steps.append(
        _step(
            plan_checksum,
            execution_id,
            step_id=f"{experiment_id}:analyse",
            operation="analyse_experiment",
            component="experiments",
            experiment_id=experiment_id,
            treatment_id=None,
            seed=None,
            dependencies=tuple(step.step_id for step in steps),
            inputs={"control": design.get("control")},
            output_contract="cognityx.experiment.analysis/v1",
            resources={"gpu": False},
            status="unsupported",
        )
    )
    return steps


def _step(
    plan_checksum: str,
    execution_id: str,
    *,
    step_id: str,
    operation: str,
    component: str,
    experiment_id: str,
    treatment_id: str | None,
    seed: int | None,
    dependencies: tuple[str, ...],
    inputs: Mapping[str, Any],
    output_contract: str,
    resources: Mapping[str, Any],
    status: str = "planned",
) -> ExecutionStep:
    identity = {
        "plan_checksum": plan_checksum,
        "execution_id": execution_id,
        "experiment_id": experiment_id,
        "treatment_id": treatment_id,
        "seed": seed,
        "operation": operation,
        "inputs": plain(inputs),
    }
    return ExecutionStep(
        step_id=step_id,
        operation=operation,
        component=component,
        experiment_id=experiment_id,
        treatment_id=treatment_id,
        seed=seed,
        dependencies=dependencies,
        input_references=plain(inputs),
        output_contract=output_contract,
        idempotency_key=checksum(identity),
        resource_requirements=plain(resources),
        retry_policy=_RETRY,
        status=status,
    )
