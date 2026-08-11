import json
import subprocess
from pathlib import Path

import pytest

from cognityx_experiments.publication import (
    GitResearchPublisher,
    JournalRecord,
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
    publisher = GitResearchPublisher(repository)

    receipt = publisher.publish(snapshot, journal=journal)
    repeated = publisher.publish(snapshot, journal=journal)

    assert receipt.commit_sha == repeated.commit_sha
    assert (repository / "research/AREA-1/H-1/evidence-ledger.jsonl").exists()
    assert (repository / "research/AREA-1/H-1/RQ-1/findings.jsonl").exists()
    assert (
        repository / "research/AREA-1/H-1/RQ-1/experiments/EXP-1/snapshots.jsonl"
    ).exists()
    snapshot_file = repository / snapshot.relative_path / "statistics.json"
    snapshot_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Immutable snapshot file conflict"):
        publisher.publish(snapshot, journal=journal)
