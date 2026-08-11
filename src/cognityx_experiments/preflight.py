"""Fail-closed production readiness checks that do not load a model."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from cognityx_experiments.contracts import (
    EVALUATION_RESEARCH_ROLES,
    RESULT_CHANGING_COMPONENTS,
    ExecutionPlan,
    LogicalExperimentPlan,
    ResearchSpec,
    SoftwareIdentity,
)
from cognityx_experiments.production import JsonHttpTransport, StorageEvidenceVerifier


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    category: str
    check: str
    status: str
    detail: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class PreflightResult:
    execution_id: str
    execution_plan_checksum: str
    passed: bool
    checks: tuple[PreflightCheck, ...]
    budget: Mapping[str, Any]
    schema: str = "cognityx.experiment.preflight/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_id": self.execution_id,
            "execution_plan_checksum": self.execution_plan_checksum,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "budget": dict(self.budget),
        }


class ProductionPreflight:
    """Inspect frozen inputs, runtime prerequisites, and the private journal."""

    def __init__(
        self,
        storage_runtime: Any,
        *,
        results_repository: str | Path,
        repository_visibility: Callable[[Path], str] | None = None,
        inference_probe: Callable[[str], Mapping[str, Any]] | None = None,
        gpu_inventory: Callable[[], Mapping[str, Any]] | None = None,
        tracking_probe: Callable[[Mapping[str, Any]], bool] | None = None,
        actual_software: Sequence[SoftwareIdentity] | None = None,
        push_enabled: bool = False,
    ) -> None:
        self.runtime = storage_runtime
        self.evidence = StorageEvidenceVerifier(storage_runtime)
        self.results_repository = Path(results_repository).resolve()
        self.repository_visibility = (
            repository_visibility or _github_repository_visibility
        )
        self.inference_probe = inference_probe or _inference_probe
        self.gpu_inventory = gpu_inventory or _gpu_inventory
        self.tracking_probe = tracking_probe or (lambda config: True)
        self.actual_software = tuple(
            actual_software
            if actual_software is not None
            else resolve_installed_software_identities()
        )
        self.push_enabled = push_enabled

    def run(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        plan: ExecutionPlan,
    ) -> PreflightResult:
        checks: list[PreflightCheck] = []
        checks.extend(self._research(spec, logical, plan))
        checks.extend(self._software(plan))
        checks.extend(self._storage_and_data(spec))
        checks.extend(self._inference_and_gpu(spec))
        checks.extend(self._observability(spec))
        checks.extend(self._git_journal())
        budget = self._budget(spec, plan)
        checks.extend(self._budget_checks(spec, budget))
        return PreflightResult(
            execution_id=plan.execution_id,
            execution_plan_checksum=plan.execution_plan_checksum,
            passed=all(check.status == "passed" for check in checks),
            checks=tuple(checks),
            budget=budget,
        )

    def _research(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        plan: ExecutionPlan,
    ) -> list[PreflightCheck]:
        try:
            spec.validate()
            if plan.spec_checksum != spec.spec_checksum:
                raise ValueError("execution plan does not freeze this ResearchSpec")
            if plan.plan_checksum != logical.plan_checksum:
                raise ValueError("execution plan does not freeze this logical plan")
            for experiment in spec.experiments:
                roles = {
                    str(item.get("research_role"))
                    for item in experiment.design.evaluation_suites
                }
                if roles - EVALUATION_RESEARCH_ROLES:
                    raise ValueError("evaluation suite has a non-canonical role")
                if experiment.design.primary_outcome.role is None and len(roles) != 1:
                    raise ValueError("primary outcome role is ambiguous")
            return [
                _passed(
                    "research",
                    "frozen_plan",
                    "ResearchSpec and both plan checksums are internally consistent.",
                    {"spec_checksum": spec.spec_checksum},
                )
            ]
        except Exception as exc:
            return [_failed("research", "frozen_plan", exc)]

    def _software(self, plan: ExecutionPlan) -> list[PreflightCheck]:
        frozen = {identity.component: identity for identity in plan.software_identities}
        actual = {identity.component: identity for identity in self.actual_software}
        failures: list[str] = []
        for component in sorted(RESULT_CHANGING_COMPONENTS):
            expected = frozen.get(component)
            observed = actual.get(component)
            if expected is None:
                failures.append(f"{component}: missing frozen identity")
                continue
            if expected.git_revision.lower() in {"unknown", "unresolved"}:
                failures.append(f"{component}: frozen revision is unknown")
                continue
            if observed is None:
                failures.append(f"{component}: installed identity unavailable")
                continue
            if (
                observed.git_revision != expected.git_revision
                or observed.package_version != expected.package_version
            ):
                failures.append(f"{component}: installed identity differs")
        if failures:
            return [
                _failed(
                    "software",
                    "frozen_revisions",
                    ValueError("; ".join(failures)),
                )
            ]
        return [
            _passed(
                "software",
                "frozen_revisions",
                "Result-changing installed revisions match the execution plan.",
                {"component_count": len(frozen)},
            )
        ]

    def _storage_and_data(self, spec: ResearchSpec) -> list[PreflightCheck]:
        checks: list[PreflightCheck] = []
        try:
            for role in ("artifact", "dataset", "model"):
                description = self.runtime.for_role(role).describe()
                capabilities = dict(description.get("capabilities") or {})
                if not capabilities.get("stream_read"):
                    raise ValueError(f"Storage role {role} is not readable")
                if not capabilities.get("stream_write"):
                    raise ValueError(f"Storage role {role} is not writable")
            checks.append(
                _passed(
                    "storage",
                    "profiles",
                    "Required Storage roles resolve through configured profiles.",
                    {},
                )
            )
        except Exception as exc:
            checks.append(_failed("storage", "profiles", exc))
        try:
            for experiment in spec.experiments:
                training = dict(experiment.execution.get("training") or {})
                training_config = training.get("config")
                if not training_config:
                    raise ValueError("Training execution requires a config path")
                if not Path(str(training_config)).is_file():
                    raise ValueError(
                        f"Training config does not exist: {training_config}"
                    )
                evaluator = dict(experiment.execution.get("evaluator") or {})
                judge_config = evaluator.get("judge_config")
                if judge_config and not Path(str(judge_config)).is_file():
                    raise ValueError(
                        f"Evaluator judge config does not exist: {judge_config}"
                    )
            checks.append(
                _passed(
                    "configuration",
                    "component_inputs",
                    "Declared component configuration files exist.",
                    {},
                )
            )
        except Exception as exc:
            checks.append(_failed("configuration", "component_inputs", exc))
        try:
            training_uris: set[str] = set()
            evaluation_uris: set[str] = set()
            record_count = 0
            for experiment in spec.experiments:
                for treatment in experiment.design.treatments:
                    inputs = dict(treatment.inputs)
                    uri = (
                        inputs.get("research_package_uri")
                        or inputs.get("data_package_uri")
                        or inputs.get("dataset_manifest_uri")
                    )
                    if not uri:
                        dataforge = dict(experiment.execution.get("dataforge") or {})
                        required = ("source", "recipe", "config")
                        if not all(dataforge.get(name) for name in required):
                            raise ValueError(
                                f"Treatment {treatment.treatment_id} has no "
                                "frozen package "
                                "and no complete DataForge build declaration"
                            )
                    else:
                        training_uris.add(str(uri))
                        self.evidence.verify_json(str(uri))
                for suite in experiment.design.evaluation_suites:
                    uri = suite.get("manifest_uri")
                    if not uri:
                        evaluation_role = suite.get("research_role")
                        raise ValueError(
                            f"Evaluation role {evaluation_role} has no manifest_uri"
                        )
                    evaluation_uris.add(str(uri))
                    manifest = self.evidence.verify_json(str(uri)).value
                    if manifest.get("research_role") != suite.get("research_role"):
                        raise ValueError("Evaluation manifest research role changed")
                    records_uri = manifest.get("records_uri")
                    if records_uri:
                        rows, _ = self.evidence.read_jsonl(
                            str(records_uri),
                            expected_checksum=(
                                str(manifest["records_checksum"])
                                if manifest.get("records_checksum")
                                else None
                            ),
                        )
                        if any(
                            row.get("training_eligible") is not False for row in rows
                        ):
                            raise ValueError(
                                "Frozen evaluation record is not marked as excluded "
                                "from training"
                            )
                        record_count += len(rows)
            overlap = training_uris & evaluation_uris
            if overlap:
                raise ValueError(
                    "An evaluation-set manifest is also declared as training input"
                )
            checks.append(
                _passed(
                    "data",
                    "frozen_inputs",
                    "Training and evaluation inputs are distinct and resolvable.",
                    {
                        "training_input_count": len(training_uris),
                        "evaluation_set_count": len(evaluation_uris),
                        "evaluation_record_count": record_count,
                    },
                )
            )
        except Exception as exc:
            checks.append(_failed("data", "frozen_inputs", exc))
        return checks

    def _inference_and_gpu(self, spec: ResearchSpec) -> list[PreflightCheck]:
        checks: list[PreflightCheck] = []
        local_required = False
        try:
            identities: list[Mapping[str, Any]] = []
            for experiment in spec.experiments:
                execution = dict(experiment.execution)
                model = dict(execution.get("model") or {})
                inference = dict(execution.get("inference") or {})
                service = dict(inference.get("service") or {})
                if not model.get("name") or not model.get("revision"):
                    raise ValueError("Inference model and revision must be frozen")
                expected = {
                    "thinking": "disabled",
                    "max_output_tokens": 512,
                    "temperature": 0,
                    "top_p": 1,
                }
                for name, value in expected.items():
                    if inference.get(name) != value:
                        raise ValueError(f"Inference {name} is not frozen to {value}")
                mode = str(service.get("mode") or "external")
                if mode == "external":
                    base_url = str(service.get("base_url") or "")
                    if not base_url:
                        raise ValueError("External Inference requires service.base_url")
                    identities.append(self.inference_probe(base_url))
                elif mode == "local_managed":
                    local_required = True
                    certified = service.get("certified_profile_uri")
                    if not certified:
                        raise ValueError(
                            "Local managed Inference requires certified_profile_uri"
                        )
                    self.evidence.verify_json(str(certified))
                else:
                    raise ValueError(f"Unsupported Inference service mode: {mode}")
            checks.append(
                _passed(
                    "inference",
                    "runtime_contract",
                    "Model, decoding, service, and certified runtime inputs resolve.",
                    {"external_identity_count": len(identities)},
                )
            )
        except Exception as exc:
            checks.append(_failed("inference", "runtime_contract", exc))
        if local_required:
            try:
                inventory = self.gpu_inventory()
                if not inventory.get("visible"):
                    raise ValueError("No supported local GPU is visible")
                checks.append(
                    _passed(
                        "gpu_runtime",
                        "inventory",
                        "Local GPU inventory is visible without loading a model.",
                        inventory,
                    )
                )
            except Exception as exc:
                checks.append(_failed("gpu_runtime", "inventory", exc))
        else:
            checks.append(
                _passed(
                    "gpu_runtime",
                    "inventory",
                    "No local managed GPU service is planned.",
                    {"required": False},
                )
            )
        return checks

    def _observability(self, spec: ResearchSpec) -> list[PreflightCheck]:
        try:
            configurations = [
                dict(experiment.execution.get("observability") or {})
                for experiment in spec.experiments
            ]
            for config in configurations:
                backend = str(config.get("backend") or "none")
                if backend == "mlflow" and not self.tracking_probe(config):
                    raise ValueError(
                        "Configured MLflow tracking backend is unavailable"
                    )
                if backend not in {"none", "structured_logging", "mlflow"}:
                    raise ValueError(f"Unsupported observability backend: {backend}")
            return [
                _passed(
                    "observability",
                    "backend",
                    "Configured observation backends are available.",
                    {},
                )
            ]
        except Exception as exc:
            return [_failed("observability", "backend", exc)]

    def _git_journal(self) -> list[PreflightCheck]:
        try:
            repository = self.results_repository
            if not (repository / ".git").exists():
                raise ValueError("Results path is not a Git repository")
            origin = _git(repository, "remote", "get-url", "origin")
            normalized = _normalize_repository(origin)
            if normalized != "cognityx/cognityx-experiment-results":
                raise ValueError(f"Unexpected results repository: {normalized}")
            if _git(repository, "status", "--porcelain"):
                raise ValueError("Results repository worktree is not clean")
            visibility = self.repository_visibility(repository).upper()
            if visibility != "PRIVATE":
                raise ValueError(
                    "Results repository visibility must be PRIVATE for real publication"
                )
            if self.push_enabled:
                subprocess.run(
                    ["gh", "auth", "status"],
                    check=True,
                    text=True,
                    capture_output=True,
                )
            return [
                _passed(
                    "git_journal",
                    "private_clean_repository",
                    "Expected private results repository is clean and authenticated.",
                    {"repository": normalized, "visibility": visibility},
                )
            ]
        except Exception as exc:
            return [_failed("git_journal", "private_clean_repository", exc)]

    @staticmethod
    def _budget(spec: ResearchSpec, plan: ExecutionPlan) -> dict[str, Any]:
        operations = [step.operation for step in plan.steps]
        ceilings = [
            dict(experiment.design.resource_constraints)
            for experiment in spec.experiments
        ]
        return {
            "training_runs": operations.count("train"),
            "inference_pairs": operations.count("execute_inference_pair"),
            "evaluator_runs": operations.count("evaluate_pair"),
            "expected_gpu_jobs": operations.count("train"),
            "declared_gpu_hour_ceiling": sum(
                float(value.get("gpu_hour_ceiling") or 0) for value in ceilings
            ),
            "declared_cost_ceiling": sum(
                float(value.get("cost_ceiling") or 0) for value in ceilings
            ),
        }

    @staticmethod
    def _budget_checks(
        spec: ResearchSpec, budget: Mapping[str, Any]
    ) -> list[PreflightCheck]:
        try:
            for experiment in spec.experiments:
                constraints = dict(experiment.design.resource_constraints)
                for name in ("gpu_hour_ceiling", "cost_ceiling"):
                    if name in constraints and float(constraints[name]) <= 0:
                        raise ValueError(f"{name} must be positive when declared")
            return [
                _passed(
                    "budget",
                    "declared_work",
                    "Expected operation counts and declared ceilings resolve.",
                    budget,
                )
            ]
        except Exception as exc:
            return [_failed("budget", "declared_work", exc)]


def resolve_installed_software_identities() -> tuple[SoftwareIdentity, ...]:
    """Resolve package metadata and PEP 610 VCS commits without using cwd Git."""
    packages = sorted(
        RESULT_CHANGING_COMPONENTS
        | {
            "cognityx-observability",
            "cognityx-resource",
            "cognityx-storage",
        }
    )
    identities: list[SoftwareIdentity] = []
    for package in packages:
        try:
            distribution = metadata.distribution(package)
            version = distribution.version
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            vcs = direct.get("vcs_info") or {}
            revision = str(vcs.get("commit_id") or "")
            direct_url = str(direct.get("url") or "")
            source = "vcs" if revision else direct_url or "installed"
            if not revision and direct_url.startswith("file:"):
                source_path = Path(
                    urllib.parse.unquote(urllib.parse.urlparse(direct_url).path)
                )
                try:
                    revision = _git(source_path, "rev-parse", "HEAD")
                    source = "editable-vcs"
                except (OSError, subprocess.CalledProcessError):
                    pass
        except metadata.PackageNotFoundError:
            version = "unknown"
            revision = ""
            source = "unavailable"
        environment_name = (
            "COGNITYX_" + package.removeprefix("cognityx-").replace("-", "_").upper()
        )
        revision = os.environ.get(f"{environment_name}_GIT_REVISION", revision)
        identities.append(
            SoftwareIdentity(
                component=package,
                package_name=package,
                package_version=version,
                git_revision=revision or "unknown",
                source=source,
            )
        )
    return tuple(identities)


def synthetic_software_identities() -> tuple[SoftwareIdentity, ...]:
    """Return explicit deterministic identities for non-scientific dry runs."""
    return tuple(
        SoftwareIdentity(
            component=component,
            package_name=component,
            package_version="0.0.0-synthetic",
            git_revision=f"synthetic-{component}",
            source="synthetic",
        )
        for component in sorted(
            RESULT_CHANGING_COMPONENTS
            | {
                "cognityx-observability",
                "cognityx-resource",
                "cognityx-storage",
            }
        )
    )


def _passed(
    category: str,
    check: str,
    detail: str,
    evidence: Mapping[str, Any],
) -> PreflightCheck:
    return PreflightCheck(category, check, "passed", detail, dict(evidence))


def _failed(category: str, check: str, error: Exception) -> PreflightCheck:
    return PreflightCheck(
        category,
        check,
        "failed",
        str(error),
        {"error_type": type(error).__name__},
    )


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _normalize_repository(value: str) -> str:
    selected = value.strip().removesuffix(".git").rstrip("/")
    if selected.startswith("git@github.com:"):
        return selected.split(":", 1)[1]
    if "github.com/" in selected:
        return selected.split("github.com/", 1)[1]
    return selected


def _github_repository_visibility(repository: Path) -> str:
    origin = _git(repository, "remote", "get-url", "origin")
    name = _normalize_repository(origin)
    return subprocess.run(
        ["gh", "repo", "view", name, "--json", "visibility", "--jq", ".visibility"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _inference_probe(base_url: str) -> Mapping[str, Any]:
    return JsonHttpTransport().get(
        f"{base_url.rstrip('/')}/v1/models", timeout_seconds=10
    )


def _gpu_inventory() -> Mapping[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"visible": bool(rows), "gpus": rows}
