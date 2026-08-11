"""Research journal summaries and paper-material assembly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def research_summary(repository: str | Path, target: str) -> str:
    """Summarize immutable findings for one hypothesis or research question."""
    root = Path(repository) / "research"
    hypothesis = _find_target(root, target)
    findings = _findings(hypothesis, target)
    experiments = sorted({str(item["experiment_id"]) for item in findings})
    relations: dict[str, list[str]] = {
        "supports": [],
        "contradicts": [],
        "inconclusive": [],
        "not_applicable": [],
    }
    literature: set[str] = set()
    follow_up: set[str] = set()
    tested_rqs: set[str] = set()
    for finding in findings:
        relation = str(
            (finding.get("confirmatory_interpretation") or {}).get(
                "hypothesis_relation", "inconclusive"
            )
        )
        relations.setdefault(relation, []).append(str(finding["finding_id"]))
        literature.update(finding.get("literature_questions") or [])
        follow_up.update((finding.get("follow_up") or {}).get("questions") or [])
        tested_rqs.update(finding.get("research_question_ids") or [])
    declared_rqs = {path.name for path in hypothesis.iterdir() if path.is_dir()}
    unresolved = sorted(declared_rqs - tested_rqs)
    lines = [
        f"# Research summary: {target}",
        "",
        f"Experiments contributing evidence: {len(experiments)}",
        *(f"- {value}" for value in experiments),
        "",
    ]
    for relation in ("supports", "contradicts", "inconclusive", "not_applicable"):
        lines.extend(
            [
                f"## {relation.replace('_', ' ').title()}",
                "",
                *(_bullets(relations.get(relation) or ["None recorded."])),
                "",
            ]
        )
    lines.extend(
        [
            "## Unresolved research questions",
            "",
            *_bullets(unresolved or ["None identified from the journal structure."]),
            "",
            "## Literature checks",
            "",
            *_bullets(sorted(literature) or ["None recorded."]),
            "",
            "## Suggested next questions",
            "",
            *_bullets(sorted(follow_up) or ["None automatically proposed."]),
            "",
        ]
    )
    return "\n".join(lines)


def paper_material(repository: str | Path, target: str) -> dict[str, Any]:
    """Assemble existing ingredients without generating a final paper."""
    root = Path(repository) / "research"
    hypothesis = _find_target(root, target)
    findings = _findings(hypothesis, target)
    rq_paths = (
        [hypothesis / target]
        if (hypothesis / target).is_dir()
        else [path for path in hypothesis.iterdir() if path.is_dir()]
    )
    tables = [
        str(path / "experiment-table.csv")
        for path in rq_paths
        if (path / "experiment-table.csv").exists()
    ]
    figures = sorted(
        str(value)
        for path in rq_paths
        for value in (path / "figure-data").glob("*.json")
        if (path / "figure-data").exists()
    )
    return {
        "target": target,
        "methods_ready_experiments": sorted(
            {str(item["experiment_id"]) for item in findings}
        ),
        "results_ready_factual_paragraphs": [
            str((item.get("observed") or {}).get("claim") or "") for item in findings
        ],
        "negative_results": [
            item["finding_id"]
            for item in findings
            if item.get("finding_class") == "negative_result"
        ],
        "limitations": sorted(
            {value for item in findings for value in item.get("limitations") or []}
        ),
        "experiment_tables": tables,
        "figure_data": figures,
        "provenance_references": [
            value
            for item in findings
            for value in (item.get("observed") or {}).get("evidence_references", [])
        ],
    }


def _find_target(root: Path, target: str) -> Path:
    matches = list(root.glob(f"*/{target}")) + list(root.glob(f"*/*/{target}"))
    if not matches:
        raise FileNotFoundError(f"Research journal target not found: {target}")
    selected = matches[0]
    if selected.parent.parent != root and selected.name == target:
        return selected.parent
    return selected


def _findings(hypothesis: Path, target: str) -> list[dict[str, Any]]:
    paths = (
        [hypothesis / target / "findings.jsonl"]
        if (hypothesis / target).is_dir()
        else list(hypothesis.glob("*/findings.jsonl"))
    )
    results: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                results[str(value["finding_id"])] = value
    return list(results.values())


def _bullets(values: Sequence[Any]) -> list[str]:
    return [f"- {value}" for value in values]
