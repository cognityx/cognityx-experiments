from pathlib import Path

import pytest

from cognityx_experiments.canonical import load_yaml
from cognityx_experiments.contracts import ResearchSpec


@pytest.fixture
def research_spec() -> ResearchSpec:
    path = Path(__file__).parents[1] / "examples" / "training-comparison.yaml"
    return ResearchSpec.from_mapping(load_yaml(path))
