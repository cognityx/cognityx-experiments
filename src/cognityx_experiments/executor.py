"""Conservative typed experiment executor with resume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from cognityx_observability import (
    ArtifactReference,
    NoOpExporter,
    ObservationContext,
    ObservationExporter,
    ObservationSession,
)
from cognityx_resource import ExecutionContext

from cognityx_experiments.analysis import analyse_records
from cognityx_experiments.canonical import checksum, plain
from cognityx_experiments.contracts import (
    ExecutionPlan,
    ExecutionStep,
    LogicalExperimentPlan,
    ResearchSpec,
)
from cognityx_experiments.ledger import ExperimentLedger


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """Compact references returned by one owning component."""

    manifest_uri: str
    manifest_checksum: str
    run_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    analysis_records: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_uri": self.manifest_uri,
            "manifest_checksum": self.manifest_checksum,
            "run_id": self.run_id,
            "attributes": plain(self.attributes),
            "analysis_records": plain(self.analysis_records),
        }


class ComponentGateway(Protocol):
    """Known Cognityx operations; deliberately not a generic plugin hook."""

    def prepare_data(
        self, step: ExecutionStep, *, parent_run_id: str | None
    ) -> ComponentResult: ...

    def train(
        self,
        step: ExecutionStep,
        prepared: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult: ...

    def start_or_reuse_inference(
        self,
        step: ExecutionStep,
        training_results: Sequence[Mapping[str, Any]],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult: ...

    def execute_inference_pair(
        self,
        step: ExecutionStep,
        training_result: Mapping[str, Any],
        runtime_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult: ...

    def stop_inference(
        self,
        step: ExecutionStep,
        runtime_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult: ...

    def evaluate_pair(
        self,
        step: ExecutionStep,
        pair_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult: ...


class ResearchMaterialHook(Protocol):
    """Automatic Storage-first material generation around scientific execution."""

    def preregister(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        plan: ExecutionPlan,
    ) -> Mapping[str, Any]: ...

    def complete(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        plan: ExecutionPlan,
        results: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


class ExperimentExecutor:
    """Execute a frozen plan once, reusing immutable completed-step records."""

    def __init__(
        self,
        gateway: ComponentGateway,
        ledger: ExperimentLedger,
        *,
        exporter: ObservationExporter | None = None,
        observability_failure_policy: str = "warn",
        synthetic: bool = False,
        material_hook: ResearchMaterialHook | None = None,
    ) -> None:
        self.gateway = gateway
        self.ledger = ledger
        self.exporter = exporter or NoOpExporter()
        self.observability_failure_policy = observability_failure_policy
        self.synthetic = synthetic
        self.material_hook = material_hook
        if not synthetic and material_hook is None:
            raise ValueError(
                "Scientific execution requires an automatic research-material hook"
            )

    def run(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        plan: ExecutionPlan,
        execution_context: ExecutionContext,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        if execution_context.run_id != plan.execution_id:
            raise ValueError("Resource run_id must equal the experiment execution_id")
        self.ledger.initialize(
            spec,
            logical,
            plan,
            resume=resume,
            synthetic=self.synthetic,
        )
        preregistration: Mapping[str, Any] = {}
        if self.material_hook is not None:
            preregistration = self.material_hook.preregister(spec, logical, plan)
        context = ObservationContext.from_execution_context(
            execution_context,
            component="experiments",
            operation="execute_research_plan",
            idempotency_key=plan.execution_plan_checksum,
            attributes={
                "research_area_id": spec.research_area.research_area_id,
                "experiment_ids": [item.experiment_id for item in spec.experiments],
                "spec_checksum": spec.spec_checksum,
                "plan_checksum": logical.plan_checksum,
                "synthetic": self.synthetic,
            },
        )
        session = ObservationSession(
            context,
            self.exporter,
            failure_policy=self.observability_failure_policy,
        )
        started = session.start()
        parent_run_id = started.external_run_id
        results: dict[str, dict[str, Any]] = {}
        for step in plan.steps:
            if step.status == "unsupported":
                session.event(
                    "experiment.step.unsupported",
                    attributes={"step_id": step.step_id, "operation": step.operation},
                )
                continue
            if self.ledger.has_completed(step):
                results[step.step_id] = dict(self.ledger.completed(step)["result"])
                session.event(
                    "experiment.step.reused",
                    attributes={
                        "step_id": step.step_id,
                        "idempotency_key": step.idempotency_key,
                    },
                )
                continue
            missing = [name for name in step.dependencies if name not in results]
            if missing:
                error = RuntimeError(
                    f"Step {step.step_id} has incomplete dependencies: "
                    f"{', '.join(missing)}"
                )
                session.fail(error, attributes={"step_id": step.step_id})
                raise error
            attempt = self.ledger.next_attempt(step)
            self.ledger.record_started(step, attempt)
            session.event(
                "experiment.step.started",
                attributes={
                    "step_id": step.step_id,
                    "operation": step.operation,
                    "component": step.component,
                    "attempt": attempt,
                    "experiment_id": step.experiment_id,
                    "treatment_id": step.treatment_id,
                    "seed": step.seed,
                },
            )
            try:
                result = self._execute(step, results, parent_run_id).to_dict()
                completed = self.ledger.record_completed(step, attempt, result)
            except Exception as exc:
                self.ledger.record_failed(step, attempt, exc)
                session.event(
                    "experiment.step.failed",
                    attributes={
                        "step_id": step.step_id,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                )
                session.fail(exc, attributes={"step_id": step.step_id})
                raise
            results[step.step_id] = result
            session.artifact(
                ArtifactReference(
                    name=f"{step.step_id}.manifest",
                    uri=str(result["manifest_uri"]),
                    checksum=str(result["manifest_checksum"]),
                    schema=step.output_contract,
                    role="component_result",
                )
            )
            session.event(
                "experiment.step.completed",
                attributes={
                    "step_id": step.step_id,
                    "attempt": attempt,
                    "ledger_uri": completed["ledger_uri"],
                },
            )
        terminal = self.ledger.finalize(plan)
        materials: Mapping[str, Any] = {}
        material_error: dict[str, str] | None = None
        if self.material_hook is not None:
            try:
                materials = self.material_hook.complete(spec, logical, plan, results)
            except Exception as exc:  # noqa: BLE001 - scientific work is already durable
                material_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        for publication in materials.get("terminal_publications", []):
            receipt = publication.get("publication_receipt") or {}
            receipt_uri = publication.get("receipt_uri")
            if receipt_uri:
                session.artifact(
                    ArtifactReference(
                        name=f"{publication['experiment_id']}.publication_receipt",
                        uri=str(receipt_uri),
                        checksum=str(publication["snapshot_id"]),
                        schema="cognityx.research.git-publication-receipt/v1",
                        role="publication_receipt",
                    )
                )
            if receipt:
                session.event(
                    "experiment.git_publication.completed",
                    attributes={
                        "experiment_id": publication["experiment_id"],
                        "snapshot_id": publication["snapshot_id"],
                        "repository": receipt.get("repository"),
                        "commit_sha": receipt.get("commit_sha"),
                    },
                )
        session.metrics(
            {
                "experiment.step_count": terminal["step_count"],
                "experiment.completed_count": terminal["completed_count"],
            }
        )
        session.finish(
            "completed",
            attributes={
                "ledger_uri": terminal["ledger_uri"],
                "synthetic": self.synthetic,
                "research_material_status": (
                    "pending_retry" if material_error else "completed"
                ),
            },
        )
        return {
            **terminal,
            "scientific_execution_status": (
                "synthetic_completed" if self.synthetic else "completed"
            ),
            "observability_status": session.result.status,
            **dict(preregistration),
            **dict(materials),
            **({"research_material_error": material_error} if material_error else {}),
        }

    def _execute(
        self,
        step: ExecutionStep,
        results: Mapping[str, Mapping[str, Any]],
        parent_run_id: str | None,
    ) -> ComponentResult:
        dependencies = [results[name] for name in step.dependencies]
        if step.operation == "prepare_data":
            return self.gateway.prepare_data(step, parent_run_id=parent_run_id)
        if step.operation == "train":
            return self.gateway.train(
                step, dependencies[0], parent_run_id=parent_run_id
            )
        if step.operation == "start_or_reuse_inference":
            return self.gateway.start_or_reuse_inference(
                step, dependencies, parent_run_id=parent_run_id
            )
        if step.operation == "execute_inference_pair":
            runtime = next(
                value
                for name, value in zip(step.dependencies, dependencies, strict=True)
                if "start-or-reuse" in name
            )
            training = next(
                value
                for name, value in zip(step.dependencies, dependencies, strict=True)
                if ":train:" in name
            )
            return self.gateway.execute_inference_pair(
                step,
                training,
                runtime,
                parent_run_id=parent_run_id,
            )
        if step.operation == "stop_inference":
            runtime = next(
                result
                for name, result in results.items()
                if name == f"{step.experiment_id}:inference:start-or-reuse"
            )
            return self.gateway.stop_inference(
                step, runtime, parent_run_id=parent_run_id
            )
        if step.operation == "evaluate_pair":
            return self.gateway.evaluate_pair(
                step, dependencies[0], parent_run_id=parent_run_id
            )
        if step.operation == "analyse_experiment":
            return self._analyse(step, results)
        raise ValueError(f"Unsupported execution operation: {step.operation}")

    def _analyse(
        self, step: ExecutionStep, results: Mapping[str, Mapping[str, Any]]
    ) -> ComponentResult:
        records = [
            record
            for step_id, result in results.items()
            if step_id.startswith(f"{step.experiment_id}:evaluate:")
            for record in result.get("analysis_records") or []
        ]
        outcome = dict(step.input_references["primary_outcome"])
        analysis = analyse_records(
            experiment_id=step.experiment_id,
            control_id=str(step.input_references["control"]),
            primary_metric=str(outcome["metric"]),
            primary_role=(str(outcome["role"]) if outcome.get("role") else None),
            records=records,
            bootstrap_samples=int(
                (step.input_references.get("analysis_plan") or {}).get(
                    "bootstrap_samples", 500
                )
            ),
        )
        uri, digest = self.ledger.publish_analysis(step.experiment_id, analysis)
        return ComponentResult(
            manifest_uri=uri,
            manifest_checksum=digest,
            run_id=step.experiment_id,
            attributes={"analysis": analysis},
        )


class DryRunGateway:
    """Explicitly synthetic gateway for structural validation only."""

    def __init__(self, *, fail_once_at: str | None = None) -> None:
        self.fail_once_at = fail_once_at
        self.failed = False
        self.calls: list[str] = []
        self.inference_start_count = 0

    def _before(self, step: ExecutionStep) -> None:
        self.calls.append(step.step_id)
        if self.fail_once_at == step.step_id and not self.failed:
            self.failed = True
            raise RuntimeError(f"simulated failure at {step.step_id}")

    def _result(self, step: ExecutionStep, kind: str) -> ComponentResult:
        self._before(step)
        payload = {
            "synthetic": True,
            "step_id": step.step_id,
            "kind": kind,
            "idempotency_key": step.idempotency_key,
        }
        return ComponentResult(
            manifest_uri=f"storage://synthetic/experiments/{checksum(payload)[:24]}.json",
            manifest_checksum=checksum(payload),
            run_id=f"synthetic-{checksum(step.step_id)[:16]}",
            attributes=payload,
        )

    def prepare_data(
        self, step: ExecutionStep, *, parent_run_id: str | None
    ) -> ComponentResult:
        del parent_run_id
        return self._result(step, "dataforge")

    def train(
        self,
        step: ExecutionStep,
        prepared: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        del prepared, parent_run_id
        return self._result(step, "training")

    def start_or_reuse_inference(
        self,
        step: ExecutionStep,
        training_results: Sequence[Mapping[str, Any]],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        del training_results, parent_run_id
        self.inference_start_count += 1
        return self._result(step, "inference_runtime")

    def execute_inference_pair(
        self,
        step: ExecutionStep,
        training_result: Mapping[str, Any],
        runtime_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        del training_result, runtime_result, parent_run_id
        return self._result(step, "inference_pair")

    def stop_inference(
        self,
        step: ExecutionStep,
        runtime_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        del runtime_result, parent_run_id
        return self._result(step, "inference_stop")

    def evaluate_pair(
        self,
        step: ExecutionStep,
        pair_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        del pair_result, parent_run_id
        result = self._result(step, "evaluator")
        metric = str(step.input_references["outcomes"]["primary"]["metric"])
        treatment_role = str(step.input_references.get("treatment_role") or "treatment")
        value = {"control": 0.5, "treatment": 0.75, "comparator": 0.625}[treatment_role]
        role = str(step.input_references["evaluation_suite"]["research_role"])
        records = (
            {
                "evaluation_record_id": f"synthetic-{step.seed}-{role}",
                "treatment_id": step.treatment_id,
                "seed": step.seed,
                "research_role": role,
                "grounded_correct": value,
                "primary_endpoint_finalized": True,
                "full_evaluation_finalized": True,
                "answer_correctness": True,
                "required_fact_completeness": True,
                "fatal_contradiction": False,
                "source_faithfulness": True,
                "generation_status": "completed",
                "evaluator_status": "finalized",
                "knowledge_unit_id": f"synthetic-unit-{role}",
                "fact_group_id": f"synthetic-fact-{role}",
                "document_id": "synthetic-document",
                "metrics": {metric: value},
                "resources": {"synthetic_cost_units": 1},
                "semantic_judge_invocation_cost": 0,
            },
        )
        return ComponentResult(
            manifest_uri=result.manifest_uri,
            manifest_checksum=result.manifest_checksum,
            run_id=result.run_id,
            attributes=result.attributes,
            analysis_records=records,
        )
