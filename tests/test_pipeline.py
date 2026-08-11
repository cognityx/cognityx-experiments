from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cognityx_observability import ObservationResult
from cognityx_resource import ExecutionContext, ResourceContext
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_experiments.aggregation import paper_material, research_summary
from cognityx_experiments.compiler import (
    compile_execution_plan,
    compile_logical_plan,
)
from cognityx_experiments.executor import DryRunGateway, ExperimentExecutor
from cognityx_experiments.ledger import ExperimentLedger
from cognityx_experiments.pipeline import ResearchMaterialPipeline
from cognityx_experiments.publication import GitResearchPublisher
from cognityx_experiments.synthesis import FindingSynthesizer


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, *, remote: Path | None = None) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "tests@cognityx.invalid")
    _git(path, "config", "user.name", "Cognityx Tests")
    (path / "README.md").write_text("# Results\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "Bootstrap")
    if remote is not None:
        remote.mkdir()
        _git(remote, "init", "--bare")
        _git(path, "remote", "add", "origin", str(remote))
        _git(path, "push", "-u", "origin", "main")
    return path


def _store(root: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=root))
    return runtime.for_role("artifact")


def _context(execution_id: str) -> ExecutionContext:
    return ExecutionContext(
        run_id=execution_id,
        correlation_id="correlation-pipeline-test",
        context=ResourceContext(project_id="pipeline-test"),
    )


class _RecordingExporter:
    backend = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def public_identity(self) -> Mapping[str, object]:
        return {"backend": self.backend}

    def start(self, context):
        self.calls.append(("start", context))
        return ObservationResult("started", self.backend, "external-1")

    def event(self, context, event):
        self.calls.append(("event", event))

    def metric(self, context, metric):
        self.calls.append(("metric", metric))

    def artifact(self, context, artifact):
        self.calls.append(("artifact", artifact))

    def finish(self, context, status, attributes):
        self.calls.append(("finish", status, dict(attributes)))
        return ObservationResult("completed", self.backend, "external-1")

    def fail(self, context, error, attributes):
        self.calls.append(("fail", str(error), dict(attributes)))
        return ObservationResult("failed", self.backend, "external-1")


class _FailThenSucceedClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **parameters: Any) -> dict[str, Any]:
        del parameters
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("narrative runtime unavailable")
        return {
            "content": {
                "result_summary": "The frozen contrast was 0.25.",
                "interpretation": "Human review remains required.",
                "limitations": ["One frozen setup."],
                "exploratory_observations": [],
                "follow_up_questions": ["Does this replicate?"],
                "literature_checks": ["Compare experimental units."],
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


def _executor(
    research_spec,
    store,
    publisher,
    *,
    gateway=None,
    exporter=None,
    synthesizer=None,
):
    logical = compile_logical_plan(research_spec)
    plan = compile_execution_plan(logical)
    selected_gateway = gateway or DryRunGateway()
    pipeline = ResearchMaterialPipeline(
        store,
        publisher,
        synthesizer=synthesizer,
    )
    executor = ExperimentExecutor(
        selected_gateway,
        ExperimentLedger(store, plan.execution_id),
        exporter=exporter,
        synthetic=True,
        material_hook=pipeline,
    )
    return executor, selected_gateway, logical, plan


def test_successful_pipeline_generates_storage_git_journal_and_observability(
    tmp_path: Path, research_spec
) -> None:
    store = _store(tmp_path / "storage")
    repository = _repository(tmp_path / "results")
    exporter = _RecordingExporter()
    executor, _, logical, plan = _executor(
        research_spec,
        store,
        GitResearchPublisher(repository),
        exporter=exporter,
    )

    result = executor.run(
        research_spec,
        logical,
        plan,
        _context(plan.execution_id),
    )

    publication = result["terminal_publications"][0]
    assert result["scientific_execution_status"] == "synthetic_completed"
    assert publication["research_material_status"] == "completed"
    assert publication["git_publication_status"] == "completed"
    assert publication["receipt_uri"].startswith("storage://")
    assert publication["publication_receipt"]["commit_sha"]
    assert list(repository.glob("experiments/POLICY-EXP-001/*/finding.json"))
    rq = repository / "research/POLICY-KNOWLEDGE/POLICY-H1/POLICY-RQ1"
    assert (rq / "findings.jsonl").exists()
    assert (rq / "experiment-table.csv").exists()
    assert list((rq / "figure-data").glob("*.json"))
    assert "Inconclusive" in research_summary(repository, "POLICY-H1")
    ingredients = paper_material(repository, "POLICY-RQ1")
    assert ingredients["methods_ready_experiments"] == ["POLICY-EXP-001"]
    artifacts = [call[1] for call in exporter.calls if call[0] == "artifact"]
    events = [call[1] for call in exporter.calls if call[0] == "event"]
    assert any(item.role == "publication_receipt" for item in artifacts)
    publication_event = next(
        item for item in events if item.name == "experiment.git_publication.completed"
    )
    assert publication_event.attributes["commit_sha"]


def test_resume_retries_synthesis_without_rerunning_science(
    tmp_path: Path, research_spec
) -> None:
    store = _store(tmp_path / "storage")
    repository = _repository(tmp_path / "results")
    client = _FailThenSucceedClient()
    synthesizer = FindingSynthesizer(client, model="example/research-model")
    executor, gateway, logical, plan = _executor(
        research_spec,
        store,
        GitResearchPublisher(repository),
        synthesizer=synthesizer,
    )

    first = executor.run(research_spec, logical, plan, _context(plan.execution_id))
    calls_after_science = tuple(gateway.calls)
    second = executor.run(
        research_spec,
        logical,
        plan,
        _context(plan.execution_id),
        resume=True,
    )

    initial = first["terminal_publications"][0]
    enriched = second["terminal_publications"][0]
    assert initial["research_material_status"] == "pending_enrichment"
    assert enriched["research_material_status"] == "completed"
    assert tuple(gateway.calls) == calls_after_science
    assert enriched["snapshot_id"] != initial["snapshot_id"]
    manifest_path = repository / enriched["publication_receipt"]["snapshot_path"]
    manifest = (manifest_path / "snapshot-manifest.json").read_text()
    assert initial["snapshot_id"] in manifest


def test_resume_retries_git_push_without_rerunning_science(
    tmp_path: Path, research_spec
) -> None:
    store = _store(tmp_path / "storage")
    remote = tmp_path / "remote.git"
    repository = _repository(tmp_path / "results", remote=remote)
    runner = _FailSecondPush()
    executor, gateway, logical, plan = _executor(
        research_spec,
        store,
        GitResearchPublisher(repository, push=True, runner=runner),
    )

    first = executor.run(research_spec, logical, plan, _context(plan.execution_id))
    calls_after_science = tuple(gateway.calls)
    second = executor.run(
        research_spec,
        logical,
        plan,
        _context(plan.execution_id),
        resume=True,
    )

    assert first["terminal_publications"][0]["git_publication_status"] == (
        "pending_retry"
    )
    assert second["terminal_publications"][0]["git_publication_status"] == ("completed")
    assert tuple(gateway.calls) == calls_after_science
    local_head = _git(repository, "rev-parse", "HEAD")
    remote_head = _git(repository, "rev-parse", "origin/main")
    assert local_head == remote_head
