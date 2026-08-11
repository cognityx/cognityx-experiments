"""Whitelisted immutable Git snapshots with Storage-first receipts."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cognityx_experiments.canonical import canonical_bytes, checksum, dump_yaml, plain

SNAPSHOT_SCHEMA = "cognityx.research.snapshot/v1"
RECEIPT_SCHEMA = "cognityx.research.git-publication-receipt/v1"
CONTENT_POLICIES = frozenset({"full", "sanitized", "metadata_only"})
_PRE_FILES = frozenset(
    {
        "research-spec.yaml",
        "logical-plan.json",
        "execution-plan.json",
        "preregistration.json",
    }
)
_TERMINAL_FILES = frozenset(
    {
        "research-spec.yaml",
        "execution-plan.json",
        "experiment.json",
        "lineage.json",
        "comparison.json",
        "comparison.md",
        "records.jsonl",
        "statistics.json",
        "resources.json",
        "finding.json",
        "finding.md",
        "tables/experiment-table.csv",
        "tables/experiment-table.json",
        "figure-data/treatment-effects.json",
        "narrative.json",
    }
)
_SECRET_PARTS = frozenset(
    {"secret", "token", "password", "credential", "api_key", "private_key"}
)
_SANITIZED_TEXT_KEYS = frozenset(
    {"answer", "candidate_answer", "prompt", "response", "source_text", "raw_text"}
)
_PATH = re.compile(r"(?:/home/|/tmp/|[A-Za-z]:\\Users\\)")


@dataclass(frozen=True, slots=True)
class Snapshot:
    snapshot_id: str
    experiment_id: str
    execution_id: str
    moment: str
    content_policy: str
    files: Mapping[str, bytes]
    manifest: Mapping[str, Any]

    @property
    def relative_path(self) -> str:
        return f"experiments/{self.experiment_id}/{self.snapshot_id}"


@dataclass(frozen=True, slots=True)
class JournalRecord:
    research_area_id: str
    hypothesis_id: str
    research_question_ids: tuple[str, ...]
    hypothesis: Mapping[str, Any]
    questions: Mapping[str, Mapping[str, Any]]
    finding: Mapping[str, Any]
    table_csv: str
    figure_data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GitPublicationReceipt:
    repository: str
    commit_sha: str
    snapshot_path: str
    snapshot_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": RECEIPT_SCHEMA,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "snapshot_path": self.snapshot_path,
            "snapshot_id": self.snapshot_id,
        }


def build_snapshot(
    *,
    moment: str,
    experiment_id: str,
    execution_id: str,
    content: Mapping[str, Any],
    content_policy: str = "sanitized",
    supersedes_snapshot_id: str | None = None,
) -> Snapshot:
    """Build an immutable snapshot from an explicit filename whitelist."""
    if moment not in {"preregistration", "terminal"}:
        raise ValueError("Snapshot moment must be preregistration or terminal")
    if content_policy not in CONTENT_POLICIES:
        raise ValueError(f"Unsupported content policy: {content_policy}")
    allowed = _PRE_FILES if moment == "preregistration" else _TERMINAL_FILES
    unknown = set(content) - allowed
    if unknown:
        raise ValueError(
            f"Snapshot contains non-whitelisted files: {', '.join(sorted(unknown))}"
        )
    required = (
        {"research-spec.yaml", "logical-plan.json", "execution-plan.json"}
        if moment == "preregistration"
        else {
            "experiment.json",
            "lineage.json",
            "statistics.json",
            "finding.json",
            "finding.md",
            "tables/experiment-table.csv",
            "figure-data/treatment-effects.json",
        }
    )
    missing = required - set(content)
    if missing:
        raise ValueError(
            f"Snapshot is missing required files: {', '.join(sorted(missing))}"
        )
    selected: dict[str, bytes] = {}
    for name, value in content.items():
        if content_policy == "metadata_only" and name in {
            "records.jsonl",
            "comparison.md",
            "narrative.json",
        }:
            continue
        sanitized = _sanitize(value, content_policy=content_policy)
        selected[name] = _file_bytes(name, sanitized)
    file_manifest = [
        {"path": name, "sha256": _sha256_bytes(value), "size_bytes": len(value)}
        for name, value in sorted(selected.items())
    ]
    frozen_manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "moment": moment,
        "experiment_id": experiment_id,
        "execution_id": execution_id,
        "content_policy": content_policy,
        "supersedes_snapshot_id": supersedes_snapshot_id,
        "files": file_manifest,
    }
    snapshot_id = checksum(frozen_manifest)
    manifest = {**frozen_manifest, "snapshot_id": snapshot_id}
    selected["snapshot-manifest.json"] = _file_bytes("snapshot-manifest.json", manifest)
    return Snapshot(
        snapshot_id=snapshot_id,
        experiment_id=experiment_id,
        execution_id=execution_id,
        moment=moment,
        content_policy=content_policy,
        files=selected,
        manifest=manifest,
    )


class GitResearchPublisher:
    """Publish immutable snapshots and idempotent cumulative journal records."""

    def __init__(
        self,
        repository: str | Path,
        *,
        push: bool = False,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.push = push
        self.runner = runner or subprocess.run
        if not (self.repository / ".git").exists():
            raise ValueError(f"Not a Git repository: {self.repository}")

    def publish(
        self,
        snapshot: Snapshot,
        *,
        journal: JournalRecord | None = None,
    ) -> GitPublicationReceipt:
        destination = self.repository / snapshot.relative_path
        self._write_snapshot(destination, snapshot)
        if journal is not None:
            if snapshot.moment != "terminal":
                raise ValueError("Only terminal snapshots update the research journal")
            self._update_journal(snapshot, journal)
        add_paths = [snapshot.relative_path]
        if (self.repository / "research").exists():
            add_paths.append("research")
        self._git("add", "--", *add_paths)
        status = self._git("status", "--porcelain").stdout.strip()
        if status:
            self._git(
                "commit",
                "-m",
                f"Publish research snapshot {snapshot.snapshot_id}",
            )
        commit_sha = self._git("rev-parse", "HEAD").stdout.strip()
        if self.push:
            self._git("push", "origin", "HEAD")
        repository = self._repository_identity()
        return GitPublicationReceipt(
            repository=repository,
            commit_sha=commit_sha,
            snapshot_path=snapshot.relative_path,
            snapshot_id=snapshot.snapshot_id,
        )

    def _write_snapshot(self, destination: Path, snapshot: Snapshot) -> None:
        if destination.exists():
            existing = destination / "snapshot-manifest.json"
            if (
                not existing.exists()
                or existing.read_bytes() != snapshot.files["snapshot-manifest.json"]
            ):
                raise FileExistsError(
                    f"Immutable snapshot conflict: {snapshot.snapshot_id}"
                )
            for name, value in snapshot.files.items():
                target = destination / name
                if not target.exists() or target.read_bytes() != value:
                    raise FileExistsError(
                        "Immutable snapshot file conflict: "
                        f"{snapshot.snapshot_id}/{name}"
                    )
            return
        for name, value in snapshot.files.items():
            target = destination / PurePosixPath(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)

    def _update_journal(self, snapshot: Snapshot, journal: JournalRecord) -> None:
        base = (
            self.repository
            / "research"
            / journal.research_area_id
            / journal.hypothesis_id
        )
        base.mkdir(parents=True, exist_ok=True)
        hypothesis_path = base / "hypothesis.yaml"
        hypothesis_bytes = dump_yaml(journal.hypothesis).encode()
        if (
            hypothesis_path.exists()
            and hypothesis_path.read_bytes() != hypothesis_bytes
        ):
            raise FileExistsError("Frozen hypothesis journal identity changed")
        if not hypothesis_path.exists():
            hypothesis_path.write_bytes(hypothesis_bytes)
        finding_id = str(journal.finding["finding_id"])
        evidence = {
            "snapshot_id": snapshot.snapshot_id,
            "experiment_id": snapshot.experiment_id,
            "execution_id": snapshot.execution_id,
            "finding_id": finding_id,
            "snapshot_path": snapshot.relative_path,
            "hypothesis_relation": (
                journal.finding.get("confirmatory_interpretation") or {}
            ).get("hypothesis_relation"),
        }
        _append_jsonl(base / "evidence-ledger.jsonl", evidence, "snapshot_id")
        for rq_id in journal.research_question_ids:
            rq = base / rq_id
            rq.mkdir(parents=True, exist_ok=True)
            rq_value = journal.questions[rq_id]
            rq_path = rq / "rq.yaml"
            rq_bytes = dump_yaml(rq_value).encode()
            if rq_path.exists() and rq_path.read_bytes() != rq_bytes:
                raise FileExistsError(f"Frozen research question changed: {rq_id}")
            if not rq_path.exists():
                rq_path.write_bytes(rq_bytes)
            _append_jsonl(rq / "findings.jsonl", journal.finding, "finding_id")
            (rq / "findings.md").write_text(
                _findings_markdown(_read_jsonl(rq / "findings.jsonl")),
                encoding="utf-8",
            )
            _append_csv(rq / "experiment-table.csv", journal.table_csv)
            figures = rq / "figure-data"
            figures.mkdir(exist_ok=True)
            figure_path = figures / f"{snapshot.snapshot_id}.json"
            figure_bytes = _file_bytes("figure.json", journal.figure_data)
            if figure_path.exists() and figure_path.read_bytes() != figure_bytes:
                raise FileExistsError("Immutable figure-data conflict")
            figure_path.write_bytes(figure_bytes)
            experiment_path = (
                rq / "experiments" / snapshot.experiment_id / "snapshots.jsonl"
            )
            experiment_path.parent.mkdir(parents=True, exist_ok=True)
            _append_jsonl(
                experiment_path,
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_path": snapshot.relative_path,
                    "execution_id": snapshot.execution_id,
                    "finding_id": finding_id,
                },
                "snapshot_id",
            )
        (base / "evidence-summary.md").write_text(
            _evidence_summary(_read_jsonl(base / "evidence-ledger.jsonl")),
            encoding="utf-8",
        )

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return self.runner(
            ["git", *arguments],
            cwd=self.repository,
            check=True,
            text=True,
            capture_output=True,
        )

    def _repository_identity(self) -> str:
        try:
            return self._git("remote", "get-url", "origin").stdout.strip()
        except subprocess.CalledProcessError:
            return str(self.repository)


def write_publication_receipt(
    store: Any,
    *,
    execution_id: str,
    receipt: GitPublicationReceipt,
) -> str:
    """Write a separate receipt so the frozen snapshot checksum cannot cycle."""
    key = (
        f"experiments/executions/{execution_id}/publication-receipts/"
        f"{receipt.snapshot_id}.json"
    )
    stored = store.put_json_idempotent(key, receipt.to_dict())
    return str(stored.uri)


def _sanitize(value: Any, *, content_policy: str, key: str = "") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for name, item in value.items():
            selected = str(name)
            lowered = selected.lower()
            parts = set(re.split(r"[^a-z0-9_]+", lowered))
            if parts & _SECRET_PARTS or any(part in lowered for part in _SECRET_PARTS):
                result[selected] = "<redacted-secret>"
            elif content_policy != "full" and lowered in _SANITIZED_TEXT_KEYS:
                result[selected] = "<redacted-content>"
            elif lowered in {"environment", "environment_variables", "env"}:
                result[selected] = "<redacted-environment>"
            else:
                result[selected] = _sanitize(
                    item, content_policy=content_policy, key=selected
                )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _sanitize(item, content_policy=content_policy, key=key) for item in value
        ]
    if isinstance(value, str) and _PATH.search(value):
        return "<redacted-path>"
    return value


def _file_bytes(name: str, value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if name.endswith((".md", ".csv", ".jsonl")) and isinstance(value, str):
        return value.encode()
    if name.endswith((".yaml", ".yml")):
        return dump_yaml(value).encode()
    return (
        json.dumps(plain(value), indent=2, sort_keys=True, ensure_ascii=False).encode()
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _append_jsonl(path: Path, value: Mapping[str, Any], identity: str) -> None:
    existing = _read_jsonl(path)
    selected = plain(value)
    matches = [
        item for item in existing if item.get(identity) == selected.get(identity)
    ]
    if matches:
        if matches[0] != selected:
            selected_identity = selected.get(identity)
            raise FileExistsError(f"Immutable journal conflict for {selected_identity}")
        return
    with path.open("a", encoding="utf-8") as output:
        output.write(canonical_bytes(selected).decode() + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_csv(path: Path, csv_text: str) -> None:
    lines = [line for line in csv_text.splitlines() if line]
    if not lines:
        return
    if not path.exists():
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    existing = path.read_text(encoding="utf-8").splitlines()
    if existing[0] != lines[0]:
        raise ValueError("Experiment table header changed")
    additions = [line for line in lines[1:] if line not in existing[1:]]
    if additions:
        with path.open("a", encoding="utf-8") as output:
            output.write("\n".join(additions) + "\n")


def _findings_markdown(findings: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Immutable research findings", ""]
    for finding in findings:
        lines.extend(
            [
                f"## {finding['finding_id']}",
                "",
                str((finding.get("observed") or {}).get("claim") or ""),
                "",
            ]
        )
    return "\n".join(lines)


def _evidence_summary(records: Sequence[Mapping[str, Any]]) -> str:
    relations: dict[str, int] = {}
    for record in records:
        relation = str(record.get("hypothesis_relation") or "unclassified")
        relations[relation] = relations.get(relation, 0) + 1
    lines = ["# Evidence summary", "", f"Experiments recorded: {len(records)}", ""]
    lines.extend(f"- {name}: {count}" for name, count in sorted(relations.items()))
    return "\n".join(lines) + "\n"
