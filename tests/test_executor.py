from pathlib import Path

import pytest
from cognityx_resource import ExecutionContext, ResourceContext
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_experiments.compiler import compile_execution_plan, compile_logical_plan
from cognityx_experiments.contracts import ResearchSpec
from cognityx_experiments.executor import DryRunGateway, ExperimentExecutor
from cognityx_experiments.ledger import ExperimentLedger


def _store(root: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=root))
    return runtime.for_role("artifact")


def _context(execution_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=execution_id,
        correlation_id="correlation-fixture",
        context=ResourceContext(project_id="research-fixture"),
    )


def test_synthetic_execution_creates_analysis_and_reuses_one_inference_session(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    logical = compile_logical_plan(research_spec)
    plan = compile_execution_plan(logical)
    ledger = ExperimentLedger(_store(tmp_path), plan.execution_id)
    gateway = DryRunGateway()
    executor = ExperimentExecutor(gateway, ledger, synthetic=True)

    result = executor.run(research_spec, logical, plan, _context(plan.execution_id))

    assert result["scientific_execution_status"] == "synthetic_completed"
    assert result["completed_count"] == 17
    assert gateway.inference_start_count == 1
    status = ledger.status(plan)
    assert status["pending_count"] == 0
    analysis_step = next(
        step for step in plan.steps if step.operation == "analyse_experiment"
    )
    analysis = ledger.completed(analysis_step)["result"]["attributes"]["analysis"]
    assert analysis["deltas_from_control"] == {"qualified": 0.25}
    assert analysis["cluster_bootstrap"]["cluster_field"] == "knowledge_unit_id"


def test_resume_skips_every_successful_expensive_step(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    logical = compile_logical_plan(research_spec)
    plan = compile_execution_plan(logical)
    failing_step = "POLICY-EXP-001:infer:control:11"
    ledger = ExperimentLedger(_store(tmp_path), plan.execution_id)
    gateway = DryRunGateway(fail_once_at=failing_step)
    executor = ExperimentExecutor(gateway, ledger, synthetic=True)

    with pytest.raises(RuntimeError, match="simulated failure"):
        executor.run(research_spec, logical, plan, _context(plan.execution_id))
    calls_before_resume = tuple(gateway.calls)
    assert gateway.inference_start_count == 1

    result = executor.run(
        research_spec,
        logical,
        plan,
        _context(plan.execution_id),
        resume=True,
    )
    resumed_calls = gateway.calls[len(calls_before_resume) :]
    assert result["scientific_execution_status"] == "synthetic_completed"
    assert not any(name.endswith(":prepare:control") for name in resumed_calls)
    assert not any(":train:" in name for name in resumed_calls)
    assert "POLICY-EXP-001:inference:start-or-reuse" not in resumed_calls
    assert gateway.inference_start_count == 1


def test_resume_rejects_a_changed_frozen_plan(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    logical = compile_logical_plan(research_spec)
    plan = compile_execution_plan(logical, execution_id="execution-fixed")
    ledger = ExperimentLedger(_store(tmp_path), plan.execution_id)
    executor = ExperimentExecutor(DryRunGateway(), ledger, synthetic=True)
    executor.run(research_spec, logical, plan, _context(plan.execution_id))

    changed = compile_execution_plan(logical, execution_id="execution-other")
    mismatched = ExperimentLedger(_store(tmp_path), plan.execution_id)
    with pytest.raises(ValueError, match="checksum mismatch"):
        mismatched.initialize(
            research_spec,
            logical,
            changed,
            resume=True,
            synthetic=True,
        )
