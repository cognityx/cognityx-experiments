from pathlib import Path

import pytest

from cognityx_experiments import cli
from cognityx_experiments.human import render_human


def test_human_renderer_handles_empty_table_nested_and_overrides() -> None:
    assert render_human([]) == "No records."
    assert render_human([{}]) == "Record 1:\n  No fields."
    table = render_human([{"job_id": "job-full-identifier", "state": "ready"}])
    assert "Job id" in table
    assert "job-full-identifier" in table
    nested = render_human(
        {
            "master_config": {"sha256": "a" * 64},
            "overrides": [
                {
                    "key": "seed",
                    "source": "--seed",
                    "previous": 17,
                    "effective": 29,
                    "changed": True,
                }
            ],
        }
    )
    assert "a" * 64 in nested
    assert "seed: 17 -> 29 (--seed)" in nested
    assert "\x1b" not in nested


def test_validate_human_is_opt_in_and_uses_same_payload(capsys) -> None:
    fixture = Path(__file__).parents[1] / "examples" / "training-comparison.yaml"

    assert cli.main(["validate", str(fixture), "--human"]) == 0
    output = capsys.readouterr().out

    assert "Valid: true" in output
    assert "Spec checksum:" in output
    assert not output.lstrip().startswith("{")


def test_config_human_calls_resolver_once(monkeypatch, capsys) -> None:
    calls = 0
    payload = {
        "component": "experiments",
        "valid": True,
        "master_config": {"kind": "built-in", "sha256": None},
        "config_layers": [],
        "overrides": [],
        "warnings": [],
        "errors": [],
    }

    def report(root, config):
        nonlocal calls
        calls += 1
        return payload

    monkeypatch.setattr(cli, "_configuration_report", report)
    assert cli.main(["config", "show", "--human"]) == 0
    assert calls == 1
    assert "Component: experiments" in capsys.readouterr().out


def test_native_output_rejects_human_flag_before_loading() -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main(["show-plan", "missing.yaml", "--human"])

    assert captured.value.code == 2
