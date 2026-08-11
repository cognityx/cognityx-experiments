import json
from pathlib import Path

import pytest

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
