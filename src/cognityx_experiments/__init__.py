"""Deterministic research planning and experiment execution."""

from cognityx_experiments.aggregation import paper_material, research_summary
from cognityx_experiments.analysis import analyse_records
from cognityx_experiments.compiler import compile_execution_plan, compile_logical_plan
from cognityx_experiments.contracts import (
    ExecutionPlan,
    ExecutionStep,
    Experiment,
    Hypothesis,
    LogicalExperimentPlan,
    ResearchArea,
    ResearchLineage,
    ResearchQuestion,
    ResearchSpec,
    SoftwareIdentity,
)
from cognityx_experiments.executor import (
    ComponentGateway,
    ComponentResult,
    DryRunGateway,
    ExperimentExecutor,
)
from cognityx_experiments.ledger import ExperimentLedger
from cognityx_experiments.mermaid import render_mermaid
from cognityx_experiments.pipeline import ResearchMaterialPipeline
from cognityx_experiments.preflight import (
    PreflightCheck,
    PreflightResult,
    ProductionPreflight,
)
from cognityx_experiments.production import CognityxComponentGateway
from cognityx_experiments.publication import (
    GitResearchPublisher,
    PublicationPolicy,
    Snapshot,
    build_snapshot,
)

__all__ = [
    "ComponentGateway",
    "ComponentResult",
    "CognityxComponentGateway",
    "DryRunGateway",
    "ExecutionPlan",
    "ExecutionStep",
    "Experiment",
    "ExperimentExecutor",
    "ExperimentLedger",
    "GitResearchPublisher",
    "Hypothesis",
    "LogicalExperimentPlan",
    "PublicationPolicy",
    "PreflightCheck",
    "PreflightResult",
    "ProductionPreflight",
    "ResearchArea",
    "ResearchLineage",
    "ResearchQuestion",
    "ResearchSpec",
    "SoftwareIdentity",
    "ResearchMaterialPipeline",
    "Snapshot",
    "analyse_records",
    "compile_execution_plan",
    "compile_logical_plan",
    "build_snapshot",
    "paper_material",
    "render_mermaid",
    "research_summary",
]
