from __future__ import annotations

import hashlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from cognityx_resource import ExecutionContext, ResourceContext
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_experiments.canonical import canonical_bytes, checksum
from cognityx_experiments.compiler import (
    compile_execution_plan,
    compile_logical_plan,
)
from cognityx_experiments.contracts import (
    RESULT_CHANGING_COMPONENTS,
    ResearchSpec,
    SoftwareIdentity,
)
from cognityx_experiments.executor import ExperimentExecutor
from cognityx_experiments.ledger import ExperimentLedger
from cognityx_experiments.pipeline import ResearchMaterialPipeline
from cognityx_experiments.preflight import ProductionPreflight
from cognityx_experiments.production import (
    TRAINING_CLI_RESULT_SCHEMA,
    CliDataForgeOperation,
    CliEvaluatorOperation,
    CliTrainingOperation,
    CognityxComponentGateway,
    ComponentMachineOutputError,
    HttpInferenceOperation,
    JsonCommandRunner,
    build_training_command,
    validate_training_cli_result,
)
from cognityx_experiments.publication import GitResearchPublisher
from cognityx_experiments.synthesis import FindingSynthesizer


def _runtime(root: Path) -> StorageRuntime:
    return StorageRuntime.from_config(StorageConfig.built_in(root=root))


def _put_manifest(
    runtime: StorageRuntime,
    key: str,
    value: dict[str, Any],
    *,
    embedded_checksum: bool = True,
) -> dict[str, str]:
    payload = dict(value)
    if embedded_checksum:
        payload["manifest_checksum"] = checksum(payload)
    stored = runtime.for_role("artifact").put_json_idempotent(key, payload)
    return {
        "manifest_uri": str(stored.uri),
        "manifest_checksum": str(payload.get("manifest_checksum") or ""),
    }


def _identities() -> tuple[SoftwareIdentity, ...]:
    return tuple(
        SoftwareIdentity(
            component=component,
            package_name=component,
            package_version="1.0.0-contract-test",
            git_revision=(component.encode().hex() + "0" * 40)[:40],
            source="contract-test",
        )
        for component in sorted(RESULT_CHANGING_COMPONENTS)
    )


def _frozen_spec(
    runtime: StorageRuntime,
    source: ResearchSpec,
    configuration_directory: Path,
) -> ResearchSpec:
    value = deepcopy(source.to_dict())
    experiment = value["experiments"][0]
    training_config = configuration_directory / "training.toml"
    training_config.write_text("[training]\n", encoding="utf-8")
    experiment["execution"]["training"]["config"] = str(training_config)
    for treatment in experiment["design"]["treatments"]:
        package = _put_manifest(
            runtime,
            f"fixtures/packages/{treatment['id']}.json",
            {
                "schema": "cognityx.dataforge.research-package/v1",
                "research_package_id": f"package-{treatment['id']}",
                "research_package_version": "v1",
            },
        )
        treatment["inputs"] = {"research_package_uri": package["manifest_uri"]}
    for suite in experiment["design"]["evaluation_suites"]:
        role = suite["research_role"]
        row = {
            "record_id": f"record-{role}",
            "research_role": role,
            "training_eligible": False,
            "knowledge_unit_id": f"unit-{role}",
            "fact_group_id": f"fact-{role}",
            "document_id": "policy-document",
        }
        raw = canonical_bytes(row) + b"\n"
        records = runtime.for_role("dataset").put_bytes(
            f"fixtures/evaluation/{role}.jsonl",
            raw,
            media_type="application/x-ndjson",
        )
        manifest = _put_manifest(
            runtime,
            f"fixtures/evaluation/{role}.json",
            {
                "schema": "cognityx.dataforge.evaluation-set/v1",
                "research_role": role,
                "records_uri": str(records.uri),
                "records_checksum": checksum(raw.decode()),
            },
        )
        suite["manifest_uri"] = manifest["manifest_uri"]
    experiment["design"]["resource_constraints"] = {
        "local_gpu_count": 1,
        "gpu_hour_ceiling": 4,
        "cost_ceiling": 25,
    }
    experiment["execution"]["inference"].update(
        {
            "thinking": "disabled",
            "max_output_tokens": 512,
            "temperature": 0,
            "top_p": 1,
            "service": {
                "mode": "external",
                "base_url": "http://inference.contract.test",
            },
        }
    )
    experiment["execution"]["observability"] = {"backend": "none"}
    return ResearchSpec.from_mapping(value)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _results_repository(path: Path, *, remote: Path | None = None) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "tests@cognityx.invalid")
    _git(path, "config", "user.name", "Cognityx Tests")
    (path / "README.md").write_text("# Contract-test results\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "Bootstrap")
    if remote is None:
        _git(
            path,
            "remote",
            "add",
            "origin",
            "https://github.com/cognityx/cognityx-experiment-results.git",
        )
    else:
        remote.mkdir()
        _git(remote, "init", "--bare")
        _git(path, "remote", "add", "origin", str(remote))
        _git(path, "push", "-u", "origin", "main")
    return path


class _DataForge:
    def __init__(self, runtime: StorageRuntime) -> None:
        self.runtime = runtime
        self.calls: list[str] = []

    def prepare(self, step, *, run_id, parent_run_id):
        del parent_run_id
        self.calls.append(step.step_id)
        return _put_manifest(
            self.runtime,
            f"component/dataforge/{run_id}.json",
            {
                "schema": "cognityx.dataforge.research-package/v1",
                "research_package_id": f"package-{step.treatment_id}",
                "research_package_version": "v1",
                "run_id": run_id,
            },
        )


class _Training:
    def __init__(self, runtime: StorageRuntime, *, fail_once: bool = False) -> None:
        self.runtime = runtime
        self.fail_once = fail_once
        self.failed = False
        self.calls: list[str] = []

    def train(self, step, prepared, *, run_id, parent_run_id):
        del prepared, parent_run_id
        self.calls.append(step.step_id)
        if self.fail_once and not self.failed:
            self.failed = True
            raise RuntimeError("simulated Training boundary failure")
        adapter = _put_manifest(
            self.runtime,
            f"component/training/{run_id}-adapter.json",
            {
                "schema_version": "cognityx.training.adapter/v1",
                "adapter_id": f"adapter-{step.treatment_id}-{step.seed}",
            },
            embedded_checksum=False,
        )
        publication = _put_manifest(
            self.runtime,
            f"component/training/{run_id}-publication.json",
            {
                "schema_version": "cognityx.training.publication/v1",
                "status": "completed",
                "training_run_id": run_id,
                "adapter_manifest_uri": adapter["manifest_uri"],
            },
            embedded_checksum=False,
        )
        return {
            "publication_manifest_uri": publication["manifest_uri"],
            "adapter_manifest_uri": adapter["manifest_uri"],
        }


class _Inference:
    def __init__(
        self, runtime: StorageRuntime, *, fail_start_once: bool = False
    ) -> None:
        self.runtime = runtime
        self.fail_start_once = fail_start_once
        self.start_failed = False
        self.start_calls = 0
        self.pair_calls: list[str] = []
        self.stop_calls = 0

    def start_or_reuse(self, step, *, parent_run_id):
        del step, parent_run_id
        self.start_calls += 1
        if self.fail_start_once and not self.start_failed:
            self.start_failed = True
            raise RuntimeError("simulated Inference process start failure")
        return {
            "mode": "external",
            "base_url": "http://inference.contract.test",
            "owned": False,
            "service_identity": {"models": ["example/base-model"]},
        }

    def execute_pair(self, step, training, runtime, *, pair_id, parent_run_id):
        del training, runtime, parent_run_id
        self.pair_calls.append(step.step_id)
        return _put_manifest(
            self.runtime,
            f"component/inference/{pair_id}.json",
            {
                "schema": "cognityx.inference.pair/v1",
                "status": "completed",
                "pair_validation": "passed",
                "inference_pair_id": pair_id,
                "base_run": {"run_id": f"base-{pair_id}"},
                "adapter_run": {"run_id": f"adapter-{pair_id}"},
            },
        )

    def stop(self, step, runtime, *, parent_run_id):
        del step, runtime, parent_run_id
        self.stop_calls += 1
        return {"owned_process_stopped": False}


class _Evaluator:
    def __init__(self, runtime: StorageRuntime, *, fail_once: bool = False) -> None:
        self.runtime = runtime
        self.fail_once = fail_once
        self.failed = False
        self.calls: list[str] = []

    def evaluate(self, step, pair, *, run_id, parent_run_id):
        del pair, parent_run_id
        self.calls.append(step.step_id)
        if self.fail_once and not self.failed:
            self.failed = True
            raise RuntimeError("simulated Evaluator boundary failure")
        role = step.input_references["evaluation_suite"]["research_role"]
        grounded = step.treatment_id == "qualified"
        row = {
            "schema": "cognityx.evaluator.score/v1",
            "evaluation_record_id": f"record-{step.seed}-{role}",
            "document_id": "policy-document",
            "source_record_id": f"source-{role}",
            "knowledge_unit_id": f"unit-{role}",
            "fact_group_id": f"fact-{role}",
            "research_role": role,
            "base": {"grounded_correct": not grounded},
            "adapter": {
                "grounded_correct": grounded,
                "generation_status": "complete",
                "evaluation_status": "scored",
                "answer_correctness": "passed" if grounded else "failed",
                "required_fact_completeness": "passed" if grounded else "failed",
                "fatal_contradiction": not grounded,
                "source_faithfulness": "passed",
                "semantic_judge_result": {"invocation_cost": 0.01},
            },
            "pair_outcome": "base_win" if grounded else "adapter_win",
        }
        raw = canonical_bytes(row) + b"\n"
        scores = self.runtime.for_role("artifact").put_bytes(
            f"component/evaluator/{run_id}.jsonl",
            raw,
            media_type="application/x-ndjson",
        )
        return _put_manifest(
            self.runtime,
            f"component/evaluator/{run_id}.json",
            {
                "schema": "cognityx.evaluator.run/v1",
                "status": "completed",
                "evaluator_run_id": run_id,
                "scores_uri": str(scores.uri),
                "scores_checksum": hashlib.sha256(raw).hexdigest(),
                "summary_uri": None,
                "resources": {"evaluator_runs": 1},
            },
        )


def _context(execution_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=execution_id,
        correlation_id="production-contract-test",
        context=ResourceContext(project_id="production-contract-test"),
    )


class _FailThenSucceedClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **parameters: Any) -> dict[str, Any]:
        del parameters
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated synthesis failure")
        return {
            "content": {
                "result_summary": "The frozen treatment contrast completed.",
                "interpretation": "Human review remains required.",
                "limitations": ["Contract-test component clients."],
                "exploratory_observations": [],
                "follow_up_questions": ["Does the result replicate?"],
                "literature_checks": ["Compare the experimental unit."],
            }
        }


class _FailSecondPush:
    def __init__(self) -> None:
        self.push_calls = 0

    def __call__(self, command, **parameters):
        if command[:2] == ["git", "push"]:
            self.push_calls += 1
            if self.push_calls == 2:
                raise subprocess.CalledProcessError(
                    1,
                    command,
                    stderr="simulated Git push failure",
                )
        return subprocess.run(command, **parameters)


class _TrainingContractRunner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments, timeout_seconds):
        del timeout_seconds
        selected = tuple(arguments)
        self.calls.append(selected)
        experiment_id = selected[selected.index("--experiment-id") + 1]
        if self.fail or experiment_id == "invalid":
            raise subprocess.CalledProcessError(2, selected)
        return {
            "schema": TRAINING_CLI_RESULT_SCHEMA,
            "mode": "dry_run",
            "experiment_id": experiment_id,
            "training_run_id": selected[selected.index("--run-id") + 1],
            "total_records": 3,
            "accepted_training_examples": 1,
            "evaluation_records": 2,
            "micro_batch_size": 1,
            "effective_batch_size": 1,
            "optimizer_steps": 1,
        }


def _composition(
    tmp_path: Path,
    source: ResearchSpec,
    *,
    training_failure: bool = False,
    evaluator_failure: bool = False,
    inference_start_failure: bool = False,
    synthesizer: FindingSynthesizer | None = None,
    git_push_failure: bool = False,
):
    runtime = _runtime(tmp_path / "storage")
    spec = _frozen_spec(runtime, source, tmp_path)
    logical = compile_logical_plan(spec)
    identities = _identities()
    plan = compile_execution_plan(logical, software_identities=identities)
    remote = tmp_path / "remote.git" if git_push_failure else None
    results = _results_repository(tmp_path / "results", remote=remote)
    preflight = ProductionPreflight(
        runtime,
        results_repository=results,
        repository_visibility=lambda repository: "PRIVATE",
        inference_probe=lambda base_url: {"base_url": base_url, "ready": True},
        training_contract_runner=_TrainingContractRunner(),
        actual_software=identities,
    ).run(spec, logical, plan)
    dataforge = _DataForge(runtime)
    training = _Training(runtime, fail_once=training_failure)
    inference = _Inference(runtime, fail_start_once=inference_start_failure)
    evaluator = _Evaluator(runtime, fail_once=evaluator_failure)
    gateway = CognityxComponentGateway(
        runtime,
        dataforge=dataforge,
        training=training,
        inference=inference,
        evaluator=evaluator,
    )
    push_runner = _FailSecondPush() if git_push_failure else None
    publisher = GitResearchPublisher(
        results,
        push=git_push_failure,
        expected_repository=None
        if git_push_failure
        else "cognityx/cognityx-experiment-results",
        runner=push_runner,
    )
    executor = ExperimentExecutor(
        gateway,
        ExperimentLedger(runtime.for_role("artifact"), plan.execution_id),
        material_hook=ResearchMaterialPipeline(
            runtime.for_role("artifact"),
            publisher,
            synthesizer=synthesizer,
        ),
    )
    return (
        preflight,
        executor,
        spec,
        logical,
        plan,
        dataforge,
        training,
        inference,
        evaluator,
        results,
        push_runner,
        runtime,
    )


@pytest.mark.parametrize(
    (
        "visibility",
        "visibility_policy",
        "classification",
        "expected_passed",
    ),
    [
        ("PRIVATE", "private_required", "unspecified", True),
        ("PUBLIC", "private_required", "public", False),
        ("PUBLIC", "public_summary", "public", True),
        ("PUBLIC", "public_summary", "internal", False),
        ("PUBLIC", "public_summary", "confidential", False),
        ("PUBLIC", "public_summary", "restricted", False),
        ("PUBLIC", "public_summary", "unspecified", False),
        ("PRIVATE", "public_summary", "internal", True),
    ],
)
def test_results_repository_visibility_obeys_frozen_publication_policy(
    tmp_path: Path,
    research_spec: ResearchSpec,
    visibility: str,
    visibility_policy: str,
    classification: str,
    expected_passed: bool,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    frozen = _frozen_spec(runtime, research_spec, tmp_path).to_dict()
    publication = frozen["experiments"][0]["execution"]["publication"]
    publication["repository_visibility_policy"] = visibility_policy
    publication["data_classification"] = classification
    spec = ResearchSpec.from_mapping(frozen)
    logical = compile_logical_plan(spec)
    identities = _identities()
    plan = compile_execution_plan(logical, software_identities=identities)
    result = ProductionPreflight(
        runtime,
        results_repository=_results_repository(tmp_path / "results"),
        repository_visibility=lambda repository: visibility,
        inference_probe=lambda base_url: {"base_url": base_url, "ready": True},
        training_contract_runner=_TrainingContractRunner(),
        actual_software=identities,
    ).run(spec, logical, plan)

    assert result.passed is expected_passed
    assert result.publication == {
        "declared_repository": "cognityx/cognityx-experiment-results",
        "declared_repository_visibility_policy": visibility_policy,
        "declared_data_classification": classification,
        "declared_content_policy": "sanitized",
        "effective_content_projection": (
            "public_summary" if visibility_policy == "public_summary" else "sanitized"
        ),
        "observed_repository": "cognityx/cognityx-experiment-results",
        "observed_repository_visibility": visibility,
    }
    git_check = next(
        check for check in result.checks if check.category == "git_journal"
    )
    assert git_check.status == ("passed" if expected_passed else "failed")


def test_production_gateway_completes_real_storage_and_git_contracts(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    composed = _composition(tmp_path, research_spec)
    preflight, executor, spec, logical, plan = composed[:5]
    dataforge, training, inference, evaluator, results = composed[5:10]

    assert preflight.passed is True
    execution = executor.run(spec, logical, plan, _context(plan.execution_id))

    assert execution["scientific_execution_status"] == "completed"
    assert execution["completed_count"] == 33
    assert len(dataforge.calls) == 2
    assert len(training.calls) == 4
    assert inference.start_calls == 1
    assert len(inference.pair_calls) == 12
    assert len(evaluator.calls) == 12
    assert inference.stop_calls == 1
    assert execution["terminal_publications"][0]["git_publication_status"] == (
        "completed"
    )
    assert list(results.glob("experiments/POLICY-EXP-001/*/finding.json"))


@pytest.mark.parametrize("failure", ["training", "evaluator"])
def test_production_resume_reuses_completed_expensive_steps(
    tmp_path: Path, research_spec: ResearchSpec, failure: str
) -> None:
    composed = _composition(
        tmp_path,
        research_spec,
        training_failure=failure == "training",
        evaluator_failure=failure == "evaluator",
    )
    _, executor, spec, logical, plan = composed[:5]
    dataforge, training, inference, evaluator = composed[5:9]

    with pytest.raises(RuntimeError, match=f"simulated {failure.title()}"):
        executor.run(spec, logical, plan, _context(plan.execution_id))
    dataforge_before = tuple(dataforge.calls)
    training_before = tuple(training.calls)
    inference_before = tuple(inference.pair_calls)
    evaluator_before = tuple(evaluator.calls)

    execution = executor.run(
        spec,
        logical,
        plan,
        _context(plan.execution_id),
        resume=True,
    )

    assert execution["scientific_execution_status"] == "completed"
    assert tuple(dataforge.calls[: len(dataforge_before)]) == dataforge_before
    assert len(dataforge.calls) == 2
    if failure == "training":
        assert len(training.calls) == 5
    else:
        assert tuple(training.calls) == training_before
        assert len(inference.pair_calls) == 12
        assert len(evaluator.calls) == 13
        assert tuple(inference.pair_calls[: len(inference_before)]) == inference_before
        assert tuple(evaluator.calls[: len(evaluator_before)]) == evaluator_before


def test_production_resume_after_process_restart_reuses_training(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    composed = _composition(
        tmp_path,
        research_spec,
        inference_start_failure=True,
    )
    _, first_executor, spec, logical, plan = composed[:5]
    dataforge, training, _, evaluator, results = composed[5:10]
    runtime = composed[11]

    with pytest.raises(RuntimeError, match="process start failure"):
        first_executor.run(spec, logical, plan, _context(plan.execution_id))
    assert len(training.calls) == 4
    training_calls = tuple(training.calls)

    restarted_inference = _Inference(runtime)
    restarted_gateway = CognityxComponentGateway(
        runtime,
        dataforge=dataforge,
        training=training,
        inference=restarted_inference,
        evaluator=evaluator,
    )
    restarted_executor = ExperimentExecutor(
        restarted_gateway,
        ExperimentLedger(runtime.for_role("artifact"), plan.execution_id),
        material_hook=ResearchMaterialPipeline(
            runtime.for_role("artifact"), GitResearchPublisher(results)
        ),
    )

    execution = restarted_executor.run(
        spec,
        logical,
        plan,
        _context(plan.execution_id),
        resume=True,
    )

    assert execution["scientific_execution_status"] == "completed"
    assert tuple(training.calls) == training_calls
    assert restarted_inference.start_calls == 1
    assert len(restarted_inference.pair_calls) == 12


def test_production_resume_retries_synthesis_without_rerunning_science(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    client = _FailThenSucceedClient()
    composed = _composition(
        tmp_path,
        research_spec,
        synthesizer=FindingSynthesizer(client, model="contract-test/model"),
    )
    _, executor, spec, logical, plan = composed[:5]
    dataforge, training, inference, evaluator = composed[5:9]

    first = executor.run(spec, logical, plan, _context(plan.execution_id))
    science_calls = (
        tuple(dataforge.calls),
        tuple(training.calls),
        tuple(inference.pair_calls),
        tuple(evaluator.calls),
    )
    second = executor.run(
        spec,
        logical,
        plan,
        _context(plan.execution_id),
        resume=True,
    )

    assert first["terminal_publications"][0]["research_material_status"] == (
        "pending_enrichment"
    )
    assert second["terminal_publications"][0]["research_material_status"] == (
        "completed"
    )
    assert science_calls == (
        tuple(dataforge.calls),
        tuple(training.calls),
        tuple(inference.pair_calls),
        tuple(evaluator.calls),
    )


def test_production_resume_retries_git_push_without_rerunning_science(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    composed = _composition(tmp_path, research_spec, git_push_failure=True)
    _, executor, spec, logical, plan = composed[:5]
    dataforge, training, inference, evaluator = composed[5:9]

    first = executor.run(spec, logical, plan, _context(plan.execution_id))
    science_calls = (
        tuple(dataforge.calls),
        tuple(training.calls),
        tuple(inference.pair_calls),
        tuple(evaluator.calls),
    )
    second = executor.run(
        spec,
        logical,
        plan,
        _context(plan.execution_id),
        resume=True,
    )

    assert first["terminal_publications"][0]["git_publication_status"] == (
        "pending_retry"
    )
    assert second["terminal_publications"][0]["git_publication_status"] == ("completed")
    assert science_calls == (
        tuple(dataforge.calls),
        tuple(training.calls),
        tuple(inference.pair_calls),
        tuple(evaluator.calls),
    )


class _CommandRunner:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, ...]] = []

    def run(self, arguments, *, timeout_seconds=None):
        del timeout_seconds
        self.calls.append(tuple(arguments))
        return self.response


class _Transport:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any], float]] = []

    def get(self, url, *, timeout_seconds):
        del timeout_seconds
        return {"object": "list", "data": [{"id": "example/base-model"}]}

    def post(self, url, payload, *, timeout_seconds):
        self.posts.append((url, dict(payload), timeout_seconds))
        return {"manifest_uri": "storage://pair/manifest.json"}


class _Process:
    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        del timeout
        return 0

    def kill(self):
        self.terminated = True


class _ProcessRunner:
    def __init__(self) -> None:
        self.process = _Process()
        self.calls: list[tuple[str, ...]] = []

    def start(self, arguments):
        self.calls.append(tuple(arguments))
        return self.process


def test_public_cli_adapters_emit_machine_readable_component_contracts(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    runtime = _runtime(tmp_path / "storage")
    spec = _frozen_spec(runtime, research_spec, tmp_path)
    plan = compile_execution_plan(
        compile_logical_plan(spec), software_identities=_identities()
    )
    prepare = next(step for step in plan.steps if step.operation == "prepare_data")
    train = next(step for step in plan.steps if step.operation == "train")
    evaluate = next(step for step in plan.steps if step.operation == "evaluate_pair")

    dataforge_runner = _CommandRunner(
        {"schema": "cognityx.dataforge.research-package/v1"}
    )
    prepared = CliDataForgeOperation(dataforge_runner).prepare(
        prepare, run_id="df-run", parent_run_id="parent"
    )
    declared_uri = prepare.input_references["treatment"]["inputs"][
        "research_package_uri"
    ]
    assert prepared["manifest_uri"] == declared_uri
    assert dataforge_runner.calls[0][1:4] == (
        "research-package",
        "show",
        declared_uri,
    )

    training_runner = _CommandRunner(
        {
            "schema": TRAINING_CLI_RESULT_SCHEMA,
            "mode": "completed",
            "experiment_id": "exp-contract",
            "training_variant_id": "tvar-contract",
            "training_run_id": "trun-stable",
            "adapter_id": "adp-contract",
            "adapter_manifest_uri": "storage://adapter",
            "training_report_uri": "storage://report",
            "publication_manifest_uri": "storage://pub",
            "artifact_uri": "storage://artifact",
        }
    )
    CliTrainingOperation(training_runner).train(
        train,
        {"manifest_uri": declared_uri},
        run_id="trun-stable",
        parent_run_id="observation-parent",
    )
    training_arguments = training_runner.calls[0]
    assert training_arguments == build_training_command(
        train,
        manifest_uri=declared_uri,
        run_id="trun-stable",
        parent_run_id="observation-parent",
    )
    assert ("--dataset-uri", declared_uri) == training_arguments[
        training_arguments.index("--dataset-uri") : training_arguments.index(
            "--dataset-uri"
        )
        + 2
    ]
    assert "--seed" in training_arguments
    assert training_arguments[training_arguments.index("--output-format") :][:2] == (
        "--output-format",
        "json",
    )
    assert training_arguments[-2:] == ("--parent-run-id", "observation-parent")

    evaluator_runner = _CommandRunner({"manifest_uri": "storage://eval"})
    CliEvaluatorOperation(evaluator_runner).evaluate(
        evaluate,
        {"manifest_uri": "storage://pair"},
        run_id="eval-stable",
        parent_run_id="observation-parent",
    )
    assert evaluator_runner.calls[0][:3] == (
        "cognityx-evaluator",
        "evaluate",
        "pair",
    )
    assert evaluator_runner.calls[0][-2:] == (
        "--parent-run-id",
        "observation-parent",
    )


def test_preflight_uses_exact_training_dry_run_contract(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    runtime = _runtime(tmp_path / "storage")
    spec = _frozen_spec(runtime, research_spec, tmp_path)
    training_config = Path(spec.experiments[0].execution["training"]["config"])
    training_config.write_text(
        '[experiment]\nid = "EXP-SYS-E2E-001"\n', encoding="utf-8"
    )
    logical = compile_logical_plan(spec)
    identities = _identities()
    plan = compile_execution_plan(logical, software_identities=identities)
    runner = _TrainingContractRunner()

    result = ProductionPreflight(
        runtime,
        results_repository=_results_repository(tmp_path / "results"),
        repository_visibility=lambda repository: "PRIVATE",
        inference_probe=lambda base_url: {"base_url": base_url, "ready": True},
        training_contract_runner=runner,
        actual_software=identities,
    ).run(spec, logical, plan)

    assert result.passed is True
    train_steps = tuple(step for step in plan.steps if step.operation == "train")
    assert len(runner.calls) == len(train_steps)
    for step, arguments in zip(train_steps, runner.calls, strict=True):
        manifest_uri = step.input_references["treatment"]["inputs"][
            "research_package_uri"
        ]
        assert arguments == build_training_command(
            step,
            manifest_uri=manifest_uri,
            run_id=f"trun-{step.idempotency_key[:24]}",
            parent_run_id=None,
            dry_run=True,
        )
    checks = [
        check
        for check in result.checks
        if check.category == "component_contract" and check.check == "training_dry_run"
    ]
    assert len(checks) == len(train_steps)
    assert all(check.status == "passed" for check in checks)
    assert all(check.evidence["accepted_training_examples"] == 1 for check in checks)


def test_json_command_runner_rejects_mixed_stdout_without_echoing_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private-training-status"
    completed = subprocess.CompletedProcess(
        ["/opt/bin/cognityx-training"],
        0,
        stdout=f'{secret}\n{{"schema": "example"}}\n',
        stderr="warning text",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(ComponentMachineOutputError) as caught:
        JsonCommandRunner().run(["/opt/bin/cognityx-training"])

    message = str(caught.value)
    assert secret not in message
    assert "warning text" not in message
    assert "cognityx-training" in message
    assert "stdout_bytes=" in message
    assert "stderr_bytes=" in message


@pytest.mark.parametrize(
    "stdout",
    ["[]", '"string"', "null"],
)
def test_json_command_runner_requires_one_json_object(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
) -> None:
    completed = subprocess.CompletedProcess(["component"], 0, stdout, "")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(ComponentMachineOutputError, match="not an object"):
        JsonCommandRunner().run(["component"])


def test_json_command_runner_redacts_nonzero_component_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        ["component"], 2, "unsafe stdout", "unsafe stderr"
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(ComponentMachineOutputError) as caught:
        JsonCommandRunner().run(["component"])

    assert "unsafe" not in str(caught.value)
    assert caught.value.exit_status == 2


def test_training_machine_envelope_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported schema"):
        validate_training_cli_result(
            {"schema": "unexpected", "mode": "dry_run"},
            expected_mode="dry_run",
        )
    with pytest.raises(ValueError, match="mode"):
        validate_training_cli_result(
            {"schema": TRAINING_CLI_RESULT_SCHEMA, "mode": "completed"},
            expected_mode="dry_run",
        )
    with pytest.raises(ValueError, match="adapter_manifest_uri"):
        validate_training_cli_result(
            {
                "schema": TRAINING_CLI_RESULT_SCHEMA,
                "mode": "completed",
                "experiment_id": "exp-example",
                "training_run_id": "trun-example",
                "training_variant_id": "tvar-example",
                "adapter_id": "adp-example",
            },
            expected_mode="completed",
        )


def test_preflight_fails_when_training_rejects_final_experiment_override(
    tmp_path: Path,
    research_spec: ResearchSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    spec = _frozen_spec(runtime, research_spec, tmp_path)
    logical = compile_logical_plan(spec)
    identities = _identities()
    plan = compile_execution_plan(logical, software_identities=identities)

    def invalid_training_command(*args, **kwargs):
        arguments = list(build_training_command(*args, **kwargs))
        arguments[arguments.index("--experiment-id") + 1] = "invalid"
        return tuple(arguments)

    monkeypatch.setattr(
        "cognityx_experiments.preflight.build_training_command",
        invalid_training_command,
    )
    runner = _TrainingContractRunner()

    result = ProductionPreflight(
        runtime,
        results_repository=_results_repository(tmp_path / "results"),
        repository_visibility=lambda repository: "PRIVATE",
        inference_probe=lambda base_url: {"base_url": base_url, "ready": True},
        training_contract_runner=runner,
        actual_software=identities,
    ).run(spec, logical, plan)

    assert result.passed is False
    failed = [
        check
        for check in result.checks
        if check.category == "component_contract" and check.check == "training_dry_run"
    ]
    assert len(failed) == sum(step.operation == "train" for step in plan.steps)
    assert all(check.status == "failed" for check in failed)
    assert all(check.evidence["exit_code"] == 2 for check in failed)


@pytest.mark.parametrize(
    ("probe_source", "expected_status"),
    [
        (
            "raise ModuleNotFoundError(\"No module named 'cognityx_observability'\")",
            "failed",
        ),
        ("pass", "passed"),
    ],
)
def test_preflight_probes_exact_local_inference_launch_environment(
    tmp_path: Path,
    research_spec: ResearchSpec,
    probe_source: str,
    expected_status: str,
) -> None:
    runtime = _runtime(tmp_path / "storage")
    value = _frozen_spec(runtime, research_spec, tmp_path).to_dict()
    profile = _put_manifest(
        runtime,
        "fixtures/inference/certified-profile.json",
        {"schema": "cognityx.inference.certified-profile/v1"},
    )
    service = value["experiments"][0]["execution"]["inference"]["service"]
    service.update(
        {
            "mode": "local_managed",
            "command": [sys.executable, "-c", "pass"],
            "probe_command": [sys.executable, "-c", probe_source],
            "certified_profile_uri": profile["manifest_uri"],
        }
    )
    spec = ResearchSpec.from_mapping(value)
    logical = compile_logical_plan(spec)
    identities = _identities()
    plan = compile_execution_plan(logical, software_identities=identities)

    result = ProductionPreflight(
        runtime,
        results_repository=_results_repository(tmp_path / "results"),
        repository_visibility=lambda repository: "PRIVATE",
        inference_probe=lambda base_url: {"base_url": base_url, "ready": True},
        gpu_inventory=lambda: {"visible": True, "gpus": ["test-gpu"]},
        training_contract_runner=_TrainingContractRunner(),
        actual_software=identities,
    ).run(spec, logical, plan)

    launch = next(
        check for check in result.checks if check.check == "local_inference_launch"
    )
    assert launch.status == expected_status
    assert launch.evidence["launch_executable"] == Path(sys.executable).resolve().name
    if expected_status == "passed":
        assert launch.evidence["exit_code"] == 0
        assert launch.evidence["success"] is True
    else:
        assert result.passed is False
        assert "exit_status=1" in launch.detail
        assert "cognityx_observability" not in launch.detail
        assert launch.evidence["success"] is False


def test_inference_http_adapter_uses_supported_context_and_owns_only_local_process(
    tmp_path: Path, research_spec: ResearchSpec
) -> None:
    runtime = _runtime(tmp_path / "storage")
    value = _frozen_spec(runtime, research_spec, tmp_path).to_dict()
    execution = value["experiments"][0]["execution"]
    execution["inference"]["service"] = {
        "mode": "local_managed",
        "base_url": "http://127.0.0.1:8124",
        "command": ["cognityx-inference", "serve", "--port", "8124"],
    }
    spec = ResearchSpec.from_mapping(value)
    plan = compile_execution_plan(
        compile_logical_plan(spec), software_identities=_identities()
    )
    start = next(
        step for step in plan.steps if step.operation == "start_or_reuse_inference"
    )
    pair = next(
        step for step in plan.steps if step.operation == "execute_inference_pair"
    )
    stop = next(step for step in plan.steps if step.operation == "stop_inference")
    transport = _Transport()
    processes = _ProcessRunner()
    operation = HttpInferenceOperation(transport, processes, sleep=lambda seconds: None)

    resident = operation.start_or_reuse(start, parent_run_id="parent")
    operation.execute_pair(
        pair,
        {
            "adapter_manifest_uri": "storage://adapter/manifest.json",
            "run_id": "training-run",
        },
        resident,
        pair_id="pair-stable",
        parent_run_id="parent",
    )
    stopped = operation.stop(stop, resident, parent_run_id="parent")

    assert len(processes.calls) == 1
    payload = transport.posts[0][1]
    assert payload["research_context"]["arm_id"] == pair.treatment_id
    assert "treatment_id" not in payload["research_context"]
    assert payload["research_context"]["training_run_id"] == "training-run"
    assert payload["thinking"] == "disabled"
    assert stopped["owned_process_stopped"] is True
    assert processes.process.terminated is True
