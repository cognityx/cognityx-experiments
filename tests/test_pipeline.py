from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from copy import deepcopy
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
from cognityx_experiments.contracts import ResearchSpec
from cognityx_experiments.executor import (
    ComponentResult,
    DryRunGateway,
    ExperimentExecutor,
)
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


class _AdversarialGateway(DryRunGateway):
    def evaluate_pair(self, step, pair_result, *, parent_run_id):
        result = super().evaluate_pair(
            step,
            pair_result,
            parent_run_id=parent_run_id,
        )
        records = []
        for record in result.analysis_records:
            records.append(
                {
                    **record,
                    "prompt": "LEAK-RAW-PROMPT",
                    "candidate_answer": "LEAK-CANDIDATE-ANSWER",
                    "reference_answer": "LEAK-REFERENCE-ANSWER",
                    "gold_reference": "LEAK-GOLD-REFERENCE",
                    "generated_answer": "LEAK-GENERATED-ANSWER",
                    "source_text": "LEAK-SOURCE-TEXT",
                    "source_evidence": "LEAK-SOURCE-EVIDENCE",
                    "raw_response": "LEAK-RAW-RESPONSE",
                    "api_token": "LEAK-API-TOKEN",
                    "password": "LEAK-PASSWORD",
                    "private_key": "LEAK-PRIVATE-KEY",
                    "local_path": "/home/researcher/private/data.json",
                    "windows_path": r"C:\Users\researcher\private\data.json",
                    "temporary_path": "/tmp/private/data.json",
                    "private_storage_uri": "storage://private/raw/evidence.json",
                    "resources": {"synthetic_cost_units": 1, "prompt_tokens": 17},
                }
            )
        return ComponentResult(
            manifest_uri=result.manifest_uri,
            manifest_checksum=result.manifest_checksum,
            run_id=result.run_id,
            attributes=result.attributes,
            analysis_records=tuple(records),
        )


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
        GitResearchPublisher(repository, expected_repository=None),
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
        GitResearchPublisher(repository, expected_repository=None),
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
    assert enriched["snapshot_id"] == initial["snapshot_id"]
    manifest_path = repository / enriched["publication_receipt"]["snapshot_path"]
    manifest = (manifest_path / "snapshot-manifest.json").read_text()
    assert '"effective_content_projection": "public_summary"' in manifest
    assert not list(manifest_path.glob("narrative.json"))


def test_public_summary_projection_blocks_adversarial_record_content(
    tmp_path: Path,
    research_spec: ResearchSpec,
) -> None:
    raw = deepcopy(research_spec.to_dict())
    experiment = raw["experiments"][0]
    experiment["design"]["treatments"][0]["inputs"].update(
        {
            "prompt": "LEAK-SPEC-PROMPT",
            "source_text": "LEAK-SPEC-SOURCE",
            "private_storage_uri": "storage://private/spec.json",
        }
    )
    experiment["execution"]["model"].update(
        {
            "prompt_template": "LEAK-MODEL-PROMPT",
            "tokenizer": "/home/researcher/private/tokenizer.json",
        }
    )
    spec = ResearchSpec.from_mapping(raw)
    store = _store(tmp_path / "storage")
    repository = _repository(tmp_path / "results")
    executor, _, logical, plan = _executor(
        spec,
        store,
        GitResearchPublisher(repository, expected_repository=None),
        gateway=_AdversarialGateway(),
    )

    result = executor.run(spec, logical, plan, _context(plan.execution_id))

    publication = result["terminal_publications"][0]
    assert publication["git_publication_status"] == "completed"
    published_paths = [
        path
        for root in (repository / "experiments", repository / "research")
        for path in root.rglob("*")
        if path.is_file()
    ]
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in published_paths)
    for forbidden in (
        "LEAK-RAW-PROMPT",
        "LEAK-CANDIDATE-ANSWER",
        "LEAK-REFERENCE-ANSWER",
        "LEAK-GOLD-REFERENCE",
        "LEAK-GENERATED-ANSWER",
        "LEAK-SOURCE-TEXT",
        "LEAK-SOURCE-EVIDENCE",
        "LEAK-RAW-RESPONSE",
        "LEAK-API-TOKEN",
        "LEAK-PASSWORD",
        "LEAK-PRIVATE-KEY",
        "LEAK-SPEC-PROMPT",
        "LEAK-SPEC-SOURCE",
        "LEAK-MODEL-PROMPT",
        "/home/researcher",
        r"C:\Users\researcher",
        "/tmp/private",
        "storage://private",
    ):
        assert forbidden not in rendered
    assert not list(repository.rglob("records.jsonl"))

    snapshot = repository / publication["publication_receipt"]["snapshot_path"]
    summary = json.loads((snapshot / "research-summary.json").read_text())
    preregistration = json.loads(
        next(
            repository.glob("experiments/POLICY-EXP-001/*/preregistration.json")
        ).read_text()
    )
    resources = json.loads((snapshot / "resources-summary.json").read_text())
    assert summary["research_questions"][0]["id"] == "POLICY-RQ1"
    assert summary["experiment"]["design"]["seeds"] == [11, 29]
    assert summary["experiment"]["model"] == {
        "name": "example/base-model",
        "revision": "0123456789abcdef",
        "tokenizer_checksum": (
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        ),
        "tokenizer_revision": "0123456789abcdef",
    }
    assert preregistration["checksums"]["research_spec"] == spec.spec_checksum
    assert resources["aggregate_resources"]["prompt_tokens"] == 204.0
    assert "Inconclusive" in research_summary(repository, "POLICY-H1")
    assert paper_material(repository, "POLICY-RQ1")["methods_ready_experiments"] == [
        "POLICY-EXP-001"
    ]


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
        GitResearchPublisher(
            repository,
            push=True,
            expected_repository=None,
            runner=runner,
        ),
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
