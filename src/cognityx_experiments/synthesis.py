"""Optional narrative enrichment through the Cognityx Inference boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from cognityx_experiments.canonical import plain

SYNTHESIS_FIELDS = frozenset(
    {
        "result_summary",
        "interpretation",
        "limitations",
        "exploratory_observations",
        "follow_up_questions",
        "literature_checks",
    }
)


class CognityxInferenceClient(Protocol):
    """The only allowed LLM boundary for finding synthesis."""

    def chat(self, **parameters: Any) -> Mapping[str, Any]: ...


class FindingSynthesizer:
    def __init__(
        self,
        client: CognityxInferenceClient,
        *,
        model: str,
        model_revision: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.model_revision = model_revision

    @property
    def generator_identity(self) -> dict[str, Any]:
        return {
            "boundary": "cognityx-inference",
            "model": self.model,
            "model_revision": self.model_revision,
        }

    def synthesize(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Request typed prose while keeping all computed values immutable."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return one JSON object with only result_summary, "
                        "interpretation, limitations, exploratory_observations, "
                        "follow_up_questions, and literature_checks. Do not change "
                        "metrics, hypotheses, questions, treatments, or significance."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(plain(evidence), sort_keys=True),
                },
            ],
            temperature=0,
            thinking="disabled",
            response_format={"type": "json_object"},
        )
        value: Any = response.get("content")
        if value is None:
            value = ((response.get("choices") or [{}])[0].get("message") or {}).get(
                "content"
            )
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise ValueError("Cognityx Inference synthesis did not return an object")
        unknown = set(value) - SYNTHESIS_FIELDS
        if unknown:
            raise ValueError(
                f"Synthesis returned unsupported fields: {', '.join(sorted(unknown))}"
            )
        selected = plain(value)
        for name in (
            "limitations",
            "exploratory_observations",
            "follow_up_questions",
            "literature_checks",
        ):
            if name in selected and not isinstance(selected[name], list):
                raise ValueError(f"Synthesis field {name} must be a list")
        return selected
