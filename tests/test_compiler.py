from copy import deepcopy

from cognityx_experiments.compiler import compile_execution_plan, compile_logical_plan
from cognityx_experiments.contracts import ResearchSpec, SoftwareIdentity
from cognityx_experiments.mermaid import render_mermaid


def test_logical_plan_expands_treatments_seeds_and_freezes_defaults(
    research_spec: ResearchSpec,
) -> None:
    logical = compile_logical_plan(research_spec)
    assert len(logical.runs) == 4
    assert len({run.run_key for run in logical.runs}) == 4
    inference = logical.experiments[0]["execution"]["inference"]
    assert inference == {
        "max_output_tokens": 512,
        "runtime_revision": "b" * 64,
        "service": {"mode": "external"},
        "temperature": 0,
        "thinking": "disabled",
        "top_p": 1,
    }
    assert logical.plan_checksum == compile_logical_plan(research_spec).plan_checksum


def test_execution_plan_groups_training_before_one_inference_window(
    research_spec: ResearchSpec,
) -> None:
    logical = compile_logical_plan(research_spec)
    plan = compile_execution_plan(logical)
    operations = [step.operation for step in plan.steps]
    assert len(plan.steps) == 33
    assert operations.count("start_or_reuse_inference") == 1
    assert operations.count("execute_inference_pair") == 12
    assert operations.count("evaluate_pair") == 12
    start = next(
        step for step in plan.steps if step.operation == "start_or_reuse_inference"
    )
    assert len(start.dependencies) == 4
    assert len({step.idempotency_key for step in plan.steps}) == len(plan.steps)
    assert (
        plan.execution_plan_checksum
        == compile_execution_plan(logical).execution_plan_checksum
    )

    diagram = render_mermaid(plan)
    assert diagram.startswith("flowchart TD")
    assert "start_or_reuse_inference" in diagram
    assert "analyse_experiment" in diagram


def test_evaluator_method_recipe_compiles_as_explicitly_unsupported(
    research_spec: ResearchSpec,
) -> None:
    value = deepcopy(research_spec.to_dict())
    value["experiments"][0]["recipe"] = "evaluator_method_comparison"
    parsed = ResearchSpec.from_mapping(value)
    plan = compile_execution_plan(compile_logical_plan(parsed))
    assert plan.steps
    assert {step.status for step in plan.steps} == {"unsupported"}
    assert all("unsupported" in step.output_contract for step in plan.steps[:-1])


def test_software_revision_is_frozen_into_execution_plan_checksum(
    research_spec: ResearchSpec,
) -> None:
    logical = compile_logical_plan(research_spec)
    identity = SoftwareIdentity(
        component="cognityx-inference",
        package_name="cognityx-inference",
        package_version="0.1.0",
        git_revision="a" * 40,
        source="git",
    )
    changed = SoftwareIdentity(
        component=identity.component,
        package_name=identity.package_name,
        package_version=identity.package_version,
        git_revision="b" * 40,
        source=identity.source,
    )

    first = compile_execution_plan(logical, software_identities=(identity,))
    second = compile_execution_plan(logical, software_identities=(changed,))

    assert first.software_identities == (identity,)
    assert first.execution_plan_checksum != second.execution_plan_checksum
