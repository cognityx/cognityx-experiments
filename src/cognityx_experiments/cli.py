"""Package command line for deterministic research protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cognityx_resource import ExecutionContext, ResourceContext
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_experiments.canonical import load_yaml
from cognityx_experiments.compiler import compile_execution_plan, compile_logical_plan
from cognityx_experiments.contracts import ExecutionPlan, ExecutionStep, ResearchSpec
from cognityx_experiments.executor import DryRunGateway, ExperimentExecutor
from cognityx_experiments.ledger import ExperimentLedger
from cognityx_experiments.mermaid import render_mermaid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cognityx-experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "show-plan"):
        command = commands.add_parser(name)
        command.add_argument("research_yaml", type=Path)
        command.add_argument("--execution-id")
    run = commands.add_parser("run")
    run.add_argument("research_yaml", type=Path)
    run.add_argument("--execution-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--storage-root", type=Path, default=Path("experiment-storage"))
    status = commands.add_parser("status")
    status.add_argument("execution_id")
    status.add_argument("--storage-root", type=Path, default=Path("experiment-storage"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        ledger = ExperimentLedger(_store(args.storage_root), args.execution_id)
        plan = _execution_plan(ledger.load_execution_plan())
        print(json.dumps(ledger.status(plan), indent=2, sort_keys=True))
        return 0
    spec = ResearchSpec.from_mapping(load_yaml(args.research_yaml))
    logical = compile_logical_plan(spec)
    plan = compile_execution_plan(logical, execution_id=args.execution_id)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema": spec.schema,
                    "spec_checksum": spec.spec_checksum,
                    "experiment_count": len(spec.experiments),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "research_spec": spec.to_dict(),
                    "logical_plan": logical.to_dict(),
                    "execution_plan": plan.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "show-plan":
        print(render_mermaid(plan), end="")
        return 0
    if not args.dry_run:
        parser.error(
            "run requires a production component composition; use --dry-run only "
            "for explicitly synthetic structural validation"
        )
    ledger = ExperimentLedger(_store(args.storage_root), plan.execution_id)
    executor = ExperimentExecutor(DryRunGateway(), ledger, synthetic=True)
    execution_context = ExecutionContext(
        run_id=plan.execution_id,
        correlation_id=f"correlation-{plan.execution_plan_checksum[:20]}",
        context=ResourceContext(),
    )
    result = executor.run(
        spec,
        logical,
        plan,
        execution_context,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _store(root: Path) -> Any:
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=root))
    return runtime.for_role("artifact")


def _execution_plan(value: dict[str, Any]) -> ExecutionPlan:
    steps = tuple(
        ExecutionStep(
            step_id=str(item["step_id"]),
            operation=str(item["operation"]),
            component=str(item["component"]),
            experiment_id=str(item["experiment_id"]),
            treatment_id=(
                str(item["treatment_id"]) if item.get("treatment_id") else None
            ),
            seed=int(item["seed"]) if item.get("seed") is not None else None,
            dependencies=tuple(str(name) for name in item["dependencies"]),
            input_references=dict(item["input_references"]),
            output_contract=str(item["output_contract"]),
            idempotency_key=str(item["idempotency_key"]),
            resource_requirements=dict(item["resource_requirements"]),
            retry_policy=dict(item["retry_policy"]),
            status=str(item["status"]),
        )
        for item in value["steps"]
    )
    return ExecutionPlan(
        execution_id=str(value["execution_id"]),
        plan_checksum=str(value["plan_checksum"]),
        spec_checksum=str(value["spec_checksum"]),
        inference_service=dict(value["inference_service"]),
        steps=steps,
    )
