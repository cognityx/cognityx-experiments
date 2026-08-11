"""Automatic preregistration and terminal research-material pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cognityx_experiments.canonical import checksum, plain
from cognityx_experiments.contracts import (
    ExecutionPlan,
    Experiment,
    LogicalExperimentPlan,
    ResearchSpec,
)
from cognityx_experiments.findings import (
    ResearchFinding,
    factual_note,
    generate_finding,
)
from cognityx_experiments.materials import experiment_table, figure_data
from cognityx_experiments.publication import (
    GitResearchPublisher,
    JournalRecord,
    PublicationPolicy,
    Snapshot,
    build_snapshot,
    write_publication_receipt,
)
from cognityx_experiments.synthesis import FindingSynthesizer


class ResearchMaterialPipeline:
    """Keep scientific completion separate from optional narrative and Git state."""

    def __init__(
        self,
        store: Any,
        publisher: GitResearchPublisher,
        *,
        synthesizer: FindingSynthesizer | None = None,
    ) -> None:
        self.store = store
        self.publisher = publisher
        self.synthesizer = synthesizer

    def preregister(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        plan: ExecutionPlan,
    ) -> dict[str, Any]:
        publications: list[dict[str, Any]] = []
        for experiment in spec.experiments:
            policy = PublicationPolicy.from_experiment(experiment)
            content = (
                _public_preregistration_content(
                    spec,
                    experiment,
                    logical,
                    plan,
                    policy,
                )
                if policy.effective_content_projection == "public_summary"
                else {
                    "research-spec.yaml": spec.to_dict(),
                    "logical-plan.json": logical.to_dict(),
                    "execution-plan.json": plan.to_dict(),
                    "preregistration.json": {
                        "experiment_id": experiment.experiment_id,
                        "treatments": [
                            item.to_dict() for item in experiment.design.treatments
                        ],
                        "seeds": list(experiment.design.seeds),
                        "primary_outcome": (
                            experiment.design.primary_outcome.to_dict()
                        ),
                        "planned_analyses": plain(experiment.design.analysis_plan),
                        "stopping_rule": plain(experiment.design.stopping_rule),
                        "plan_checksum": logical.plan_checksum,
                    },
                }
            )
            snapshot = build_snapshot(
                moment="preregistration",
                experiment_id=experiment.experiment_id,
                execution_id=plan.execution_id,
                publication_policy=policy,
                content=content,
            )
            storage_uri = self._store_snapshot(snapshot)
            try:
                receipt = self.publisher.publish(snapshot)
                git_status = "completed"
                receipt_uri = write_publication_receipt(
                    self.store, execution_id=plan.execution_id, receipt=receipt
                )
            except Exception as exc:
                # Publication must not erase the Storage preregistration.
                git_status = "pending_retry"
                receipt_uri = None
                error = {"type": type(exc).__name__, "message": str(exc)}
            publications.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "storage_snapshot_uri": storage_uri,
                    "git_publication_status": git_status,
                    "receipt_uri": receipt_uri,
                    **({"git_error": error} if git_status != "completed" else {}),
                }
            )
        return {"preregistration": publications}

    def complete(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        plan: ExecutionPlan,
        results: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        publications: list[dict[str, Any]] = []
        for experiment in spec.experiments:
            policy = PublicationPolicy.from_experiment(experiment)
            analysis = self._analysis(experiment, results)
            evidence = self._evidence(experiment, results)
            finding = self._finding(spec, experiment, plan, analysis, evidence)
            enriched: dict[str, Any] | None = None
            synthesis_error: dict[str, str] | None = None
            if self.synthesizer is not None:
                try:
                    enriched = self.synthesizer.synthesize(
                        {
                            "hypothesis": next(
                                item.to_dict()
                                for item in spec.hypotheses
                                if item.hypothesis_id == finding.hypothesis_id
                            ),
                            "research_questions": [
                                item.to_dict()
                                for item in spec.research_questions
                                if item.research_question_id
                                in finding.research_question_ids
                            ],
                            "design": experiment.design.to_dict(),
                            "analysis": analysis,
                            "lineage": evidence,
                            "finding": finding.to_dict(),
                        }
                    )
                except Exception as exc:
                    synthesis_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
            note = factual_note(
                finding,
                enriched=(
                    None
                    if policy.effective_content_projection == "public_summary"
                    else enriched
                ),
            )
            table_rows, table_csv = experiment_table(analysis)
            figures = figure_data(analysis)
            if policy.effective_content_projection == "public_summary":
                public_finding = _public_finding(finding)
                content = _public_terminal_content(
                    spec,
                    experiment,
                    plan,
                    policy,
                    analysis,
                    evidence,
                    public_finding,
                    note,
                    table_rows,
                    table_csv,
                    figures,
                )
            else:
                public_finding = None
                records = self._records(experiment, results)
                content = {
                    "research-spec.yaml": spec.to_dict(),
                    "execution-plan.json": plan.to_dict(),
                    "experiment.json": experiment.to_dict(),
                    "lineage.json": evidence,
                    "comparison.json": analysis,
                    "comparison.md": finding.observed_claim + "\n",
                    "records.jsonl": records,
                    "statistics.json": analysis,
                    "resources.json": analysis.get("resources") or {},
                    "finding.json": finding.to_dict(),
                    "finding.md": note,
                    "tables/experiment-table.csv": table_csv,
                    "tables/experiment-table.json": table_rows,
                    "figure-data/treatment-effects.json": figures,
                }
            if enriched is not None and public_finding is None:
                content["narrative.json"] = {
                    "generator": self.synthesizer.generator_identity
                    if self.synthesizer
                    else None,
                    "content": enriched,
                }
            snapshot = build_snapshot(
                moment="terminal",
                experiment_id=experiment.experiment_id,
                execution_id=plan.execution_id,
                publication_policy=policy,
                content=content,
            )
            latest_snapshot_id = self._latest_terminal_snapshot_id(
                plan.execution_id,
                experiment.experiment_id,
            )
            if latest_snapshot_id and latest_snapshot_id != snapshot.snapshot_id:
                snapshot = build_snapshot(
                    moment="terminal",
                    experiment_id=experiment.experiment_id,
                    execution_id=plan.execution_id,
                    publication_policy=policy,
                    content=content,
                    supersedes_snapshot_id=latest_snapshot_id,
                )
            storage_uri = self._store_snapshot(snapshot)
            journal = self._journal(
                spec,
                finding,
                snapshot,
                table_csv,
                figures,
                finding_value=public_finding,
            )
            try:
                receipt = self.publisher.publish(snapshot, journal=journal)
                git_status = "completed"
                receipt_uri = write_publication_receipt(
                    self.store, execution_id=plan.execution_id, receipt=receipt
                )
            except Exception as exc:
                git_status = "pending_retry"
                receipt_uri = None
                git_error = {"type": type(exc).__name__, "message": str(exc)}
            material_status = "pending_enrichment" if synthesis_error else "completed"
            publications.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "finding_id": finding.finding_id,
                    "snapshot_id": snapshot.snapshot_id,
                    "storage_snapshot_uri": storage_uri,
                    "scientific_execution_status": "completed",
                    "analysis_status": "completed",
                    "research_material_status": material_status,
                    "git_publication_status": git_status,
                    "receipt_uri": receipt_uri,
                    **(
                        {"publication_receipt": receipt.to_dict()}
                        if git_status == "completed"
                        else {}
                    ),
                    **({"synthesis_error": synthesis_error} if synthesis_error else {}),
                    **({"git_error": git_error} if git_status != "completed" else {}),
                }
            )
        return {"terminal_publications": publications}

    def _finding(
        self,
        spec: ResearchSpec,
        experiment: Experiment,
        plan: ExecutionPlan,
        analysis: Mapping[str, Any],
        evidence: list[dict[str, Any]],
    ) -> ResearchFinding:
        key = (
            f"experiments/executions/{plan.execution_id}/research-material/"
            f"{experiment.experiment_id}/finding.json"
        )
        if self.store.exists(key):
            with self.store.open(key) as source:
                return ResearchFinding.from_dict(json.load(source))
        finding = generate_finding(
            spec,
            experiment,
            execution_id=plan.execution_id,
            analysis=analysis,
            evidence_references=evidence,
        )
        self.store.put_json_idempotent(key, finding.to_dict())
        return finding

    def _store_snapshot(self, snapshot: Snapshot) -> str:
        key = (
            f"experiments/executions/{snapshot.execution_id}/research-material/"
            f"{snapshot.moment}/{snapshot.snapshot_id}.json"
        )
        value = {
            "manifest": plain(snapshot.manifest),
            "files": {
                name: {
                    "sha256": next(
                        item["sha256"]
                        for item in snapshot.manifest["files"]
                        if item["path"] == name
                    )
                    if name != "snapshot-manifest.json"
                    else None,
                    "content": content.decode("utf-8"),
                }
                for name, content in snapshot.files.items()
            },
        }
        return str(self.store.put_json_idempotent(key, value).uri)

    def _latest_terminal_snapshot_id(
        self, execution_id: str, experiment_id: str
    ) -> str | None:
        prefix = f"experiments/executions/{execution_id}/research-material/terminal"
        if not self.store.exists(prefix):
            return None
        manifests: list[Mapping[str, Any]] = []
        for stored in self.store.list(prefix):
            namespace = str(getattr(self.store, "namespace", "")).strip("/")
            relative_key = str(stored.key)
            if namespace and relative_key.startswith(f"{namespace}/"):
                relative_key = relative_key[len(namespace) + 1 :]
            with self.store.open(relative_key) as source:
                value = json.load(source)
            manifest = value.get("manifest") or {}
            if manifest.get("experiment_id") == experiment_id:
                manifests.append(manifest)
        if not manifests:
            return None
        superseded = {
            str(value["supersedes_snapshot_id"])
            for value in manifests
            if value.get("supersedes_snapshot_id")
        }
        latest = [
            str(value["snapshot_id"])
            for value in manifests
            if str(value["snapshot_id"]) not in superseded
        ]
        if len(latest) != 1:
            raise ValueError("Terminal snapshot history has multiple active heads")
        return latest[0]

    @staticmethod
    def _analysis(
        experiment: Experiment, results: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        key = f"{experiment.experiment_id}:analyse"
        try:
            return dict(results[key]["attributes"]["analysis"])
        except KeyError as exc:
            raise ValueError(f"Missing authoritative analysis result: {key}") from exc

    @staticmethod
    def _evidence(
        experiment: Experiment, results: Mapping[str, Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "step_id": step_id,
                "manifest_uri": result.get("manifest_uri"),
                "manifest_checksum": result.get("manifest_checksum"),
                "run_id": result.get("run_id"),
            }
            for step_id, result in sorted(results.items())
            if step_id.startswith(f"{experiment.experiment_id}:")
        ]

    @staticmethod
    def _records(
        experiment: Experiment, results: Mapping[str, Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            plain(record)
            for step_id, result in sorted(results.items())
            if step_id.startswith(f"{experiment.experiment_id}:evaluate:")
            for record in result.get("analysis_records") or []
        ]

    @staticmethod
    def _journal(
        spec: ResearchSpec,
        finding: ResearchFinding,
        snapshot: Snapshot,
        table_csv: str,
        figures: Mapping[str, Any],
        *,
        finding_value: Mapping[str, Any] | None = None,
    ) -> JournalRecord:
        hypothesis = next(
            item.to_dict()
            for item in spec.hypotheses
            if item.hypothesis_id == finding.hypothesis_id
        )
        questions = {
            item.research_question_id: item.to_dict()
            for item in spec.research_questions
            if item.research_question_id in finding.research_question_ids
        }
        return JournalRecord(
            research_area_id=finding.research_area_id,
            hypothesis_id=finding.hypothesis_id,
            research_question_ids=finding.research_question_ids,
            hypothesis=hypothesis,
            questions=questions,
            finding=plain(finding_value or finding.to_dict()),
            table_csv=table_csv,
            figure_data=figures,
        )


def _public_preregistration_content(
    spec: ResearchSpec,
    experiment: Experiment,
    logical: LogicalExperimentPlan,
    plan: ExecutionPlan,
    policy: PublicationPolicy,
) -> dict[str, Any]:
    questions = [
        value.to_dict()
        for value in spec.research_questions
        if value.research_question_id in experiment.addresses
    ]
    hypothesis_ids = {str(value["hypothesis_id"]) for value in questions}
    hypotheses = [
        value.to_dict()
        for value in spec.hypotheses
        if value.hypothesis_id in hypothesis_ids
    ]
    model = dict(experiment.execution.get("model") or {})
    return {
        "preregistration.json": {
            "schema": "cognityx.research.public-preregistration/v1",
            "publication_policy": policy.to_dict(),
            "research_area": spec.research_area.to_dict(),
            "hypotheses": hypotheses,
            "research_questions": questions,
            "experiment": _public_experiment(experiment, model),
            "research_lineage": spec.lineage.to_dict() if spec.lineage else None,
            "software_identities": _public_software_identities(plan),
            "checksums": {
                "research_spec": spec.spec_checksum,
                "logical_plan": logical.plan_checksum,
                "execution_plan": plan.execution_plan_checksum,
            },
        }
    }


def _public_terminal_content(
    spec: ResearchSpec,
    experiment: Experiment,
    plan: ExecutionPlan,
    policy: PublicationPolicy,
    analysis: Mapping[str, Any],
    evidence: list[dict[str, Any]],
    finding: Mapping[str, Any],
    note: str,
    table_rows: list[dict[str, Any]],
    table_csv: str,
    figures: Mapping[str, Any],
) -> dict[str, Any]:
    questions = [
        value.to_dict()
        for value in spec.research_questions
        if value.research_question_id in experiment.addresses
    ]
    hypothesis_ids = {str(value["hypothesis_id"]) for value in questions}
    hypotheses = [
        value.to_dict()
        for value in spec.hypotheses
        if value.hypothesis_id in hypothesis_ids
    ]
    model = dict(experiment.execution.get("model") or {})
    statistics = _public_statistics(analysis)
    evidence_summary = _public_evidence(evidence)
    return {
        "research-summary.json": {
            "schema": "cognityx.research.public-summary/v1",
            "publication_policy": policy.to_dict(),
            "research_area": spec.research_area.to_dict(),
            "hypotheses": hypotheses,
            "research_questions": questions,
            "experiment": _public_experiment(experiment, model),
            "software_identities": _public_software_identities(plan),
            "statistics": statistics,
            "finding_id": finding["finding_id"],
        },
        "finding.json": plain(finding),
        "finding.md": note,
        "statistics.json": statistics,
        "resources-summary.json": _public_resources(analysis),
        "tables/experiment-table.csv": table_csv,
        "tables/experiment-table.json": table_rows,
        "figure-data/treatment-effects.json": plain(figures),
        "lineage-summary.json": {
            "schema": "cognityx.research.public-lineage/v1",
            "experiment_id": experiment.experiment_id,
            "execution_id": plan.execution_id,
            "finding_id": finding["finding_id"],
            "research_lineage": spec.lineage.to_dict() if spec.lineage else None,
            "checksums": {
                "research_spec": spec.spec_checksum,
                "execution_plan": plan.execution_plan_checksum,
                "analysis": checksum(analysis),
            },
            "component_evidence": evidence_summary,
        },
    }


def _public_experiment(
    experiment: Experiment,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    design = experiment.design
    return {
        "id": experiment.experiment_id,
        "addresses": list(experiment.addresses),
        "recipe": experiment.recipe,
        "design": {
            "type": design.design_type,
            "research_profile": design.research_profile,
            "experimental_unit": design.experimental_unit,
            "treatments": [
                {"id": value.treatment_id, "role": value.role}
                for value in design.treatments
            ],
            "control": design.control_id,
            "comparators": list(design.comparator_ids),
            "seeds": list(design.seeds),
            "primary_outcome": design.primary_outcome.to_dict(),
            "secondary_outcomes": [
                value.to_dict() for value in design.secondary_outcomes
            ],
            "estimand": plain(design.estimand),
            "stopping_rule": plain(design.stopping_rule),
            "evaluation_roles": [
                value.get("research_role") for value in design.evaluation_suites
            ],
        },
        "model": {
            key: model[key]
            for key in (
                "name",
                "revision",
                "tokenizer_revision",
                "tokenizer_checksum",
            )
            if model.get(key) is not None
        },
    }


def _public_software_identities(plan: ExecutionPlan) -> list[dict[str, Any]]:
    return [
        {
            "component": value.component,
            "package_name": value.package_name,
            "package_version": value.package_version,
            "git_revision": value.git_revision,
        }
        for value in plan.software_identities
    ]


def _public_statistics(analysis: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "schema",
        "experiment_id",
        "primary_metric",
        "primary_role",
        "control_id",
        "treatments",
        "contrasts_from_control",
        "deltas_from_control",
        "per_seed_effects",
        "comparable_pair_count",
        "unresolved_count",
        "primary_endpoint_unresolved_count",
        "full_evaluation_unresolved_count",
        "role_diagnostics",
        "secondary_role_summaries",
        "cluster_bootstrap",
        "semantic_judge_invocation_cost",
        "interpretation_status",
    )
    return {key: plain(analysis[key]) for key in allowed if key in analysis}


def _public_resources(analysis: Mapping[str, Any]) -> dict[str, Any]:
    resources = {
        str(key): float(value)
        for key, value in (analysis.get("resources") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {
        "schema": "cognityx.research.public-resource-summary/v1",
        "aggregate_resources": resources,
        "semantic_judge_invocation_cost": float(
            analysis.get("semantic_judge_invocation_cost") or 0
        ),
    }


def _public_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value[key]
            for key in ("step_id", "manifest_checksum", "run_id")
            if value.get(key) is not None
        }
        for value in evidence
    ]


def _public_finding(finding: ResearchFinding) -> dict[str, Any]:
    value = finding.to_dict()
    observed = dict(value["observed"])
    observed["evidence_references"] = _public_evidence(
        [dict(item) for item in observed.get("evidence_references") or []]
    )
    return {
        key: plain(item)
        for key, item in {
            "schema": value["schema"],
            "finding_id": value["finding_id"],
            "research_area_id": value["research_area_id"],
            "hypothesis_id": value["hypothesis_id"],
            "research_question_ids": value["research_question_ids"],
            "experiment_id": value["experiment_id"],
            "execution_id": value["execution_id"],
            "finding_class": value["finding_class"],
            "observed": observed,
            "confirmatory_interpretation": value["confirmatory_interpretation"],
            "exploratory_observations": value["exploratory_observations"],
            "literature_questions": value["literature_questions"],
            "follow_up": value["follow_up"],
            "limitations": value["limitations"],
            "novelty_candidate": value["novelty_candidate"],
            "literature_check_required": value["literature_check_required"],
            "generator": {
                key: item
                for key, item in (value.get("generator") or {}).items()
                if key in {"component", "mode", "analysis_checksum"}
            },
            "generated_at": value["generated_at"],
            "human_review": {"status": value["human_review"]["status"]},
        }.items()
    }
