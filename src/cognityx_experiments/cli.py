"""Package command line for deterministic research protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cognityx_resource import ExecutionContext, ResourceContext
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_experiments.aggregation import paper_material, research_summary
from cognityx_experiments.canonical import load_yaml
from cognityx_experiments.compiler import compile_execution_plan, compile_logical_plan
from cognityx_experiments.contracts import (
    ExecutionPlan,
    ExecutionStep,
    ResearchSpec,
    SoftwareIdentity,
)
from cognityx_experiments.executor import (
    ComponentGateway,
    DryRunGateway,
    ExperimentExecutor,
)
from cognityx_experiments.ledger import ExperimentLedger
from cognityx_experiments.mermaid import render_mermaid
from cognityx_experiments.pipeline import ResearchMaterialPipeline
from cognityx_experiments.preflight import (
    ProductionPreflight,
    resolve_installed_software_identities,
    synthetic_software_identities,
)
from cognityx_experiments.production import CognityxComponentGateway
from cognityx_experiments.publication import GitResearchPublisher


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
    run.add_argument("--storage-config", type=Path)
    run.add_argument("--results-repo", type=Path)
    run.add_argument("--push-results", action="store_true")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("research_yaml", type=Path)
    preflight.add_argument("--execution-id")
    preflight.add_argument(
        "--storage-root", type=Path, default=Path("experiment-storage")
    )
    preflight.add_argument("--storage-config", type=Path)
    preflight.add_argument("--results-repo", type=Path, required=True)
    preflight.add_argument("--push-results", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("execution_id")
    status.add_argument("--storage-root", type=Path, default=Path("experiment-storage"))
    for name in ("research-summary", "paper-material"):
        command = commands.add_parser(name)
        command.add_argument("target")
        command.add_argument("--results-repo", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "research-summary":
        print(research_summary(args.results_repo, args.target), end="")
        return 0
    if args.command == "paper-material":
        print(
            json.dumps(
                paper_material(args.results_repo, args.target),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "status":
        ledger = ExperimentLedger(
            _runtime(args.storage_root).for_role("artifact"), args.execution_id
        )
        plan = _execution_plan(ledger.load_execution_plan())
        print(json.dumps(ledger.status(plan), indent=2, sort_keys=True))
        return 0
    spec = ResearchSpec.from_mapping(load_yaml(args.research_yaml))
    logical = compile_logical_plan(spec)
    software_identities = (
        synthetic_software_identities()
        if args.command == "run" and args.dry_run
        else resolve_installed_software_identities()
    )
    plan = compile_execution_plan(
        logical,
        execution_id=args.execution_id,
        software_identities=software_identities,
    )
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
    runtime = _runtime(args.storage_root, args.storage_config)
    if args.command == "preflight":
        preflight_result = ProductionPreflight(
            runtime,
            results_repository=args.results_repo,
            actual_software=software_identities,
            push_enabled=args.push_results,
        ).run(spec, logical, plan)
        print(json.dumps(preflight_result.to_dict(), indent=2, sort_keys=True))
        return 0 if preflight_result.passed else 2
    if not args.dry_run and args.results_repo is None:
        parser.error("production run requires --results-repo")
    if args.dry_run:
        gateway: ComponentGateway = DryRunGateway()
        material_hook = None
        synthetic = True
    else:
        preflight = ProductionPreflight(
            runtime,
            results_repository=args.results_repo,
            actual_software=software_identities,
            push_enabled=args.push_results,
        ).run(spec, logical, plan)
        if not preflight.passed:
            print(json.dumps(preflight.to_dict(), indent=2, sort_keys=True))
            return 2
        publisher = GitResearchPublisher(
            args.results_repo,
            push=args.push_results,
        )
        gateway = CognityxComponentGateway(runtime)
        material_hook = ResearchMaterialPipeline(
            runtime.for_role("artifact"), publisher
        )
        synthetic = False
    ledger = ExperimentLedger(runtime.for_role("artifact"), plan.execution_id)
    executor = ExperimentExecutor(
        gateway,
        ledger,
        synthetic=synthetic,
        material_hook=material_hook,
    )
    execution_context = ExecutionContext(
        run_id=plan.execution_id,
        correlation_id=f"correlation-{plan.execution_plan_checksum[:20]}",
        context=ResourceContext(),
    )
    execution_result = executor.run(
        spec,
        logical,
        plan,
        execution_context,
        resume=args.resume,
    )
    print(json.dumps(execution_result, indent=2, sort_keys=True))
    return 0


def _runtime(root: Path, config: Path | None = None) -> StorageRuntime:
    if config is not None:
        return StorageRuntime.load(config_file=config)
    return StorageRuntime.from_config(StorageConfig.built_in(root=root))


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
        software_identities=tuple(
            SoftwareIdentity.from_mapping(item)
            for item in value.get("software_identities") or []
        ),
        steps=steps,
    )
