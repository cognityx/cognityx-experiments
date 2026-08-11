"""Execution-plan Mermaid rendering."""

from __future__ import annotations

import re

from cognityx_experiments.contracts import ExecutionPlan


def render_mermaid(plan: ExecutionPlan) -> str:
    """Render a non-authoritative visualization of a frozen execution plan."""
    lines = ["flowchart TD"]
    names: dict[str, str] = {}
    for index, step in enumerate(plan.steps):
        node = f"S{index}_{re.sub(r'[^A-Za-z0-9_]', '_', step.step_id)}"
        names[step.step_id] = node
        label = f"{step.component}: {step.operation}"
        if step.treatment_id is not None:
            label += f"\\n{step.treatment_id}"
        if step.seed is not None:
            label += f" seed={step.seed}"
        if step.status == "unsupported":
            label += "\\nUNSUPPORTED"
        lines.append(f'  {node}["{label}"]')
    for step in plan.steps:
        for dependency in step.dependencies:
            lines.append(f"  {names[dependency]} --> {names[step.step_id]}")
    return "\n".join(lines) + "\n"
