import json
from pathlib import Path

import pytest
from cognityx_storage import StorageConfig

from cognityx_experiments.cli import main


def test_validate_plan_and_show_plan(capsys, tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "examples" / "training-comparison.yaml"
    assert main(["validate", str(fixture)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert main(["show-plan", str(fixture)]) == 0
    assert capsys.readouterr().out.startswith("flowchart TD")

    assert (
        main(
            [
                "run",
                str(fixture),
                "--dry-run",
                "--storage-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["scientific_execution_status"]
        == "synthetic_completed"
    )


def test_cli_refuses_to_misrepresent_unconfigured_run_as_real(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "examples" / "training-comparison.yaml"
    with pytest.raises(SystemExit):
        main(["run", str(fixture), "--storage-root", str(tmp_path)])


def test_storage_selectors_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "config",
                "show",
                "--storage-root",
                "root",
                "--storage-config",
                "storage.toml",
            ]
        )


def test_config_discovery_and_explicit_selection_are_static_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    project = tmp_path / "project"
    config = project / ".cognityx" / "storage.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        '[storage]\ndefault_profile="local-main"\n'
        '[storage.profiles.local-main]\ntype="filesystem"\nroot="project-storage"\n'
        '[storage.roles.artifact]\nprofile="local-main"\nnamespace="artifacts"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-user"))

    assert main(["config", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["configuration_kind"] == "composed-dependencies"
    assert shown["dependencies"]["storage"]["master_config"]["path"] == str(
        config.resolve()
    )

    assert main(["config", "validate", "--storage-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert (
        validated["dependencies"]["storage"]["master_config"]["sha256"]
        == shown["dependencies"]["storage"]["master_config"]["sha256"]
    )


def test_no_real_storage_file_preserves_local_compatibility_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-user"))

    assert main(["config", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)

    assert shown["effective"]["storage_compatibility_fallback"] == (
        "built-in-compatibility-fallback"
    )
    root = shown["dependencies"]["storage"]["effective"]["profiles"][0]["options"][
        "root"
    ]
    assert root == "experiment-storage"

    explicit_root = tmp_path / "explicit-storage"
    assert main(["config", "show", "--storage-root", str(explicit_root)]) == 0
    overridden = json.loads(capsys.readouterr().out)
    change = overridden["dependencies"]["storage"]["overrides"][0]
    assert change == {
        "changed": True,
        "effective": str(explicit_root),
        "key": "storage.profiles.local-main.options.root",
        "previous": str(
            StorageConfig.built_in().profiles["local-main"].options["root"]
        ),
        "source": "--storage-root",
    }
    assert overridden["dependencies"]["storage"]["field_sources"][change["key"]] == (
        "--storage-root"
    )


def test_missing_explicit_storage_file_returns_nonzero_json(capsys) -> None:
    assert main(["config", "validate", "--storage-config", "missing.toml"]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
