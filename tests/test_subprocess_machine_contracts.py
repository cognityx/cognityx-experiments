"""Opt-in contract probe for installed Cognityx component executables."""

from __future__ import annotations

import json
import os

import pytest

from cognityx_experiments.production import JsonCommandRunner


def test_installed_component_commands_emit_one_json_object() -> None:
    """Run exact safe component commands supplied by an integration environment."""
    encoded = os.environ.get("COGNITYX_SUBPROCESS_CONTRACT_COMMANDS")
    if encoded is None:
        pytest.skip("installed component contract commands were not supplied")
    specifications = json.loads(encoded)
    assert isinstance(specifications, list)
    assert {item["component"] for item in specifications} == {
        "dataforge",
        "training",
        "evaluator",
    }
    runner = JsonCommandRunner()
    for specification in specifications:
        result = runner.run(
            tuple(map(str, specification["arguments"])),
            timeout_seconds=float(specification.get("timeout_seconds", 120)),
        )
        for field in specification["required_fields"]:
            assert result.get(field) not in (None, ""), (
                specification["component"],
                field,
            )
