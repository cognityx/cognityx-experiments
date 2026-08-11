import json
import subprocess
from pathlib import Path

import pytest

from cognityx_experiments.publication import (
    GitResearchPublisher,
    JournalRecord,
    PublicationPolicy,
    build_snapshot,
)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "tests@cognityx.invalid")
    _git(path, "config", "user.name", "Cognityx Tests")
    (path / "README.md").write_text("# Results\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "Bootstrap")
    return path


class _RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command, **parameters):
        self.commands.append(list(command))
        return subprocess.run(command, **parameters)


def _preregistration_content() -> dict[str, object]:
    return {
        "research-spec.yaml": {
            "id": "EXP-1",
            "api_token": "not-publishable",
            "source_text": "not-publishable",
            "path": "/home/person/private",
            "environment": {"CUDA_VISIBLE_DEVICES": "0"},
        },
        "logical-plan.json": {"checksum": "logical"},
        "execution-plan.json": {"checksum": "execution"},
        "preregistration.json": {"seeds": [11, 29]},
    }


def _terminal_content() -> dict[str, object]:
    finding = {
        "finding_id": "finding-1",
        "experiment_id": "EXP-1",
        "research_question_ids": ["RQ-1"],
        "observed": {"claim": "Treatment minus control was 0.25."},
        "confirmatory_interpretation": {"hypothesis_relation": "inconclusive"},
        "limitations": ["One frozen configuration."],
        "literature_questions": ["Which units are comparable?"],
        "follow_up": {"questions": []},
    }
    return {
        "research-spec.yaml": {"id": "EXP-1"},
        "execution-plan.json": {"checksum": "execution"},
        "experiment.json": {"id": "EXP-1"},
        "lineage.json": [{"uri": "storage://analysis"}],
        "statistics.json": {"deltas_from_control": {"treatment": 0.25}},
        "finding.json": finding,
        "finding.md": "# Finding\n\nTreatment minus control was 0.25.\n",
        "tables/experiment-table.csv": "experiment,effect\nEXP-1,0.25\n",
        "figure-data/treatment-effects.json": {"effect": 0.25},
    }


def _public_terminal_content() -> dict[str, object]:
    finding = _terminal_content()["finding.json"]
    return {
        "research-summary.json": {"experiment_id": "EXP-1", "effect": 0.25},
        "finding.json": finding,
        "finding.md": "# Finding\n\nTreatment minus control was 0.25.\n",
        "statistics.json": {"deltas_from_control": {"treatment": 0.25}},
        "resources-summary.json": {"gpu_hours": 0.5},
        "tables/experiment-table.csv": "experiment,effect\nEXP-1,0.25\n",
        "figure-data/treatment-effects.json": {"effect": 0.25},
        "lineage-summary.json": {"analysis_checksum": "sha256:analysis"},
    }


def test_publication_policy_defaults_are_private_and_frozen_in_snapshot() -> None:
    default = PublicationPolicy.from_mapping(None)
    assert default.repository_visibility_policy == "private_required"
    assert default.data_classification == "unspecified"
    assert default.content_policy == "sanitized"

    policy = PublicationPolicy.from_mapping(
        {
            "repository_visibility_policy": "public_summary",
            "data_classification": "public",
            "content_policy": "full",
        }
    )
    snapshot = build_snapshot(
        moment="terminal",
        experiment_id="EXP-1",
        execution_id="execution-1",
        publication_policy=policy,
        content=_public_terminal_content(),
    )

    assert snapshot.manifest["publication_policy"] == policy.to_dict()
    assert snapshot.manifest["effective_content_projection"] == "public_summary"
    assert "records.jsonl" not in snapshot.files


def test_public_summary_rejects_non_whitelisted_or_unsafe_content() -> None:
    policy = PublicationPolicy(
        repository_visibility_policy="public_summary",
        data_classification="public",
    )
    with pytest.raises(ValueError, match="non-whitelisted"):
        build_snapshot(
            moment="terminal",
            experiment_id="EXP-1",
            execution_id="execution-1",
            publication_policy=policy,
            content={
                **_public_terminal_content(),
                "records.jsonl": [{"generated_answer": "secret"}],
            },
        )
    unsafe = _public_terminal_content()
    unsafe["research-summary.json"] = {
        "experiment_id": "EXP-1",
        "nested": {"prompt": "private prompt"},
    }
    with pytest.raises(ValueError, match="forbidden field"):
        build_snapshot(
            moment="terminal",
            experiment_id="EXP-1",
            execution_id="execution-1",
            publication_policy=policy,
            content=unsafe,
        )


def test_snapshot_whitelist_redaction_and_superseding_identity() -> None:
    snapshot = build_snapshot(
        moment="preregistration",
        experiment_id="EXP-1",
        execution_id="execution-1",
        content=_preregistration_content(),
    )
    frozen = snapshot.files["research-spec.yaml"].decode()

    assert "not-publishable" not in frozen
    assert "/home/person" not in frozen
    assert "<redacted-secret>" in frozen
    assert "<redacted-content>" in frozen
    assert "<redacted-environment>" in frozen
    with pytest.raises(ValueError, match="non-whitelisted"):
        build_snapshot(
            moment="preregistration",
            experiment_id="EXP-1",
            execution_id="execution-1",
            content={**_preregistration_content(), "weights.bin": b"no"},
        )
    corrected = build_snapshot(
        moment="preregistration",
        experiment_id="EXP-1",
        execution_id="execution-1",
        content=_preregistration_content(),
        supersedes_snapshot_id=snapshot.snapshot_id,
    )
    assert corrected.snapshot_id != snapshot.snapshot_id
    assert corrected.manifest["supersedes_snapshot_id"] == snapshot.snapshot_id


def test_structured_jsonl_is_sanitized_before_serialization() -> None:
    record = {
        "treatment_id": "qualified",
        "seed": 11,
        "research_question_id": "RQ-1",
        "model_revision": "model-commit",
        "manifest_checksum": "sha256:manifest",
        "tokenizer_revision": "tokenizer-commit",
        "tokenizer_checksum": "sha256:tokenizer",
        "prompt_tokens": 21,
        "completion_tokens": 8,
        "token_budget": 512,
        "nested": {
            "prompt": "private prompt",
            "candidate_answer": "private candidate",
            "answer": "private answer",
            "response": "private response",
            "source_text": "private source",
            "raw_text": "private raw text",
            "path": "/tmp/private-record.json",
            "access_token": "credential",
            "api_key": "credential",
            "password": "credential",
            "private_key": "credential",
            "Authorization": "Bearer credential",
        },
    }
    snapshot = build_snapshot(
        moment="terminal",
        experiment_id="EXP-1",
        execution_id="execution-1",
        content={**_terminal_content(), "records.jsonl": [record]},
    )

    rendered = snapshot.files["records.jsonl"].decode()
    parsed = json.loads(rendered)
    for secret_value in (
        "private prompt",
        "private candidate",
        "private answer",
        "private response",
        "private source",
        "private raw text",
        "/tmp/private-record.json",
        "Bearer credential",
    ):
        assert secret_value not in rendered
    assert parsed["tokenizer_revision"] == "tokenizer-commit"
    assert parsed["tokenizer_checksum"] == "sha256:tokenizer"
    assert parsed["prompt_tokens"] == 21
    assert parsed["completion_tokens"] == 8
    assert parsed["token_budget"] == 512
    assert parsed["model_revision"] == "model-commit"
    assert parsed["manifest_checksum"] == "sha256:manifest"
    assert parsed["treatment_id"] == "qualified"
    assert parsed["seed"] == 11
    assert parsed["research_question_id"] == "RQ-1"
    assert set(parsed["nested"].values()) == {
        "<redacted-content>",
        "<redacted-path>",
        "<redacted-secret>",
    }

    metadata_only = build_snapshot(
        moment="terminal",
        experiment_id="EXP-1",
        execution_id="execution-1",
        content={**_terminal_content(), "records.jsonl": [record]},
        content_policy="metadata_only",
    )
    assert "records.jsonl" not in metadata_only.files


def test_git_publisher_is_idempotent_and_detects_immutable_conflict(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "results")
    snapshot = build_snapshot(
        moment="terminal",
        experiment_id="EXP-1",
        execution_id="execution-1",
        content=_terminal_content(),
    )
    finding = json.loads(snapshot.files["finding.json"])
    journal = JournalRecord(
        research_area_id="AREA-1",
        hypothesis_id="H-1",
        research_question_ids=("RQ-1",),
        hypothesis={"id": "H-1", "statement": "Frozen statement."},
        questions={
            "RQ-1": {
                "id": "RQ-1",
                "hypothesis_id": "H-1",
                "question": "Frozen question?",
            }
        },
        finding=finding,
        table_csv="experiment,effect\nEXP-1,0.25\n",
        figure_data={"effect": 0.25},
    )
    runner = _RecordingRunner()
    publisher = GitResearchPublisher(
        repository,
        expected_repository=None,
        runner=runner,
    )

    receipt = publisher.publish(snapshot, journal=journal)
    repeated = publisher.publish(snapshot, journal=journal)

    assert receipt.commit_sha == repeated.commit_sha
    add_commands = [command for command in runner.commands if command[1] == "add"]
    assert add_commands
    assert all("research" not in command for command in add_commands)
    assert all(snapshot.relative_path not in command for command in add_commands)
    assert (repository / "research/AREA-1/H-1/evidence-ledger.jsonl").exists()
    assert (repository / "research/AREA-1/H-1/RQ-1/findings.jsonl").exists()
    assert (
        repository / "research/AREA-1/H-1/RQ-1/experiments/EXP-1/snapshots.jsonl"
    ).exists()
    snapshot_file = repository / snapshot.relative_path / "statistics.json"
    snapshot_file.write_text("{}\n", encoding="utf-8")
    _git(repository, "add", snapshot.relative_path)
    _git(repository, "commit", "-m", "Simulate conflicting immutable history")
    with pytest.raises(FileExistsError, match="Immutable snapshot file conflict"):
        publisher.publish(snapshot, journal=journal)


def test_git_publisher_rejects_wrong_repository_and_dirty_worktree(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "results")
    snapshot = build_snapshot(
        moment="terminal",
        experiment_id="EXP-1",
        execution_id="execution-1",
        content=_terminal_content(),
    )

    with pytest.raises(ValueError, match="Unexpected research-results repository"):
        GitResearchPublisher(repository).publish(snapshot)

    (repository / "unrelated.txt").write_text("do not stage\n", encoding="utf-8")
    publisher = GitResearchPublisher(repository, expected_repository=None)
    with pytest.raises(RuntimeError, match="worktree is not clean"):
        publisher.publish(snapshot)
    assert not (repository / snapshot.relative_path).exists()
