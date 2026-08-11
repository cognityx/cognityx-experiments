"""Immutable Storage-backed experiment execution ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Protocol

from cognityx_experiments.canonical import checksum, plain
from cognityx_experiments.contracts import (
    ExecutionPlan,
    ExecutionStep,
    LogicalExperimentPlan,
    ResearchSpec,
)

LEDGER_SCHEMA = "cognityx.experiment.execution-ledger/v1"


class StoredObjectLike(Protocol):
    uri: str


class LedgerStore(Protocol):
    def put_json_idempotent(self, key: str, value: Any) -> StoredObjectLike: ...

    def exists(self, key: str) -> bool: ...

    def open(self, key: str) -> Any: ...

    def uri(self, key: str) -> str: ...


class ExperimentLedger:
    """Store immutable execution facts at deterministic keys."""

    def __init__(self, store: LedgerStore, execution_id: str) -> None:
        self.store = store
        self.execution_id = execution_id
        self.root = f"experiments/executions/{execution_id}"

    def initialize(
        self,
        spec: ResearchSpec,
        logical: LogicalExperimentPlan,
        execution: ExecutionPlan,
        *,
        resume: bool,
        synthetic: bool,
    ) -> dict[str, Any]:
        metadata_key = f"{self.root}/execution.json"
        if self.store.exists(metadata_key):
            existing = self._read(metadata_key)
            if not resume:
                raise FileExistsError(
                    f"Execution {self.execution_id} already exists; use --resume"
                )
            expected = {
                "spec_checksum": spec.spec_checksum,
                "plan_checksum": logical.plan_checksum,
                "execution_plan_checksum": execution.execution_plan_checksum,
            }
            for name, value in expected.items():
                if existing.get(name) != value:
                    raise ValueError(
                        f"Resume checksum mismatch for {name}: "
                        f"{existing.get(name)} != {value}"
                    )
            return existing
        if resume:
            raise FileNotFoundError(f"Execution {self.execution_id} does not exist")
        spec_object = self.store.put_json_idempotent(
            f"{self.root}/frozen/research-spec.json", spec.to_dict()
        )
        logical_object = self.store.put_json_idempotent(
            f"{self.root}/frozen/logical-plan.json", logical.to_dict()
        )
        execution_object = self.store.put_json_idempotent(
            f"{self.root}/frozen/execution-plan.json", execution.to_dict()
        )
        metadata = {
            "schema": LEDGER_SCHEMA,
            "execution_id": self.execution_id,
            "synthetic": synthetic,
            "spec_uri": spec_object.uri,
            "spec_checksum": spec.spec_checksum,
            "logical_plan_uri": logical_object.uri,
            "plan_checksum": logical.plan_checksum,
            "execution_plan_uri": execution_object.uri,
            "execution_plan_checksum": execution.execution_plan_checksum,
            "created_at": _now(),
        }
        self.store.put_json_idempotent(metadata_key, metadata)
        return metadata

    def has_completed(self, step: ExecutionStep) -> bool:
        return self.store.exists(self._completed_key(step))

    def completed(self, step: ExecutionStep) -> dict[str, Any]:
        return self._read(self._completed_key(step))

    def next_attempt(self, step: ExecutionStep) -> int:
        attempt = 1
        while self.store.exists(self._attempt_key(step, attempt, "started")):
            attempt += 1
        return attempt

    def record_started(self, step: ExecutionStep, attempt: int) -> None:
        self.store.put_json_idempotent(
            self._attempt_key(step, attempt, "started"),
            {
                "step_id": step.step_id,
                "idempotency_key": step.idempotency_key,
                "attempt": attempt,
                "status": "started",
                "started_at": _now(),
            },
        )

    def record_failed(
        self, step: ExecutionStep, attempt: int, error: BaseException
    ) -> None:
        self.store.put_json_idempotent(
            self._attempt_key(step, attempt, "failed"),
            {
                "step_id": step.step_id,
                "idempotency_key": step.idempotency_key,
                "attempt": attempt,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": _now(),
            },
        )

    def record_completed(
        self,
        step: ExecutionStep,
        attempt: int,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "step_id": step.step_id,
            "operation": step.operation,
            "component": step.component,
            "experiment_id": step.experiment_id,
            "treatment_id": step.treatment_id,
            "seed": step.seed,
            "idempotency_key": step.idempotency_key,
            "attempt": attempt,
            "status": "completed",
            "result": plain(result),
            "completed_at": _now(),
        }
        stored = self.store.put_json_idempotent(self._completed_key(step), payload)
        return {**payload, "ledger_uri": stored.uri}

    def publish_analysis(
        self, experiment_id: str, analysis: Mapping[str, Any]
    ) -> tuple[str, str]:
        selected = plain(analysis)
        digest = checksum(selected)
        stored = self.store.put_json_idempotent(
            f"{self.root}/analysis/{experiment_id}/{digest}.json", selected
        )
        return stored.uri, digest

    def finalize(self, plan: ExecutionPlan) -> dict[str, Any]:
        statuses = self.status(plan)
        if statuses["pending_count"]:
            raise RuntimeError("Cannot finalize an execution with pending steps")
        payload = {
            "schema": LEDGER_SCHEMA,
            "execution_id": self.execution_id,
            "status": "completed",
            "step_count": statuses["step_count"],
            "completed_count": statuses["completed_count"],
            "unsupported_count": statuses["unsupported_count"],
            "completed_at": _now(),
        }
        stored = self.store.put_json_idempotent(
            f"{self.root}/ledger-completed.json", payload
        )
        return {**payload, "ledger_uri": stored.uri}

    def status(self, plan: ExecutionPlan) -> dict[str, Any]:
        completed = [step for step in plan.steps if self.has_completed(step)]
        unsupported = [step for step in plan.steps if step.status == "unsupported"]
        pending = [
            step
            for step in plan.steps
            if step.status != "unsupported" and not self.has_completed(step)
        ]
        return {
            "execution_id": self.execution_id,
            "step_count": len(plan.steps),
            "completed_count": len(completed),
            "unsupported_count": len(unsupported),
            "pending_count": len(pending),
            "completed_steps": [step.step_id for step in completed],
            "pending_steps": [step.step_id for step in pending],
        }

    def load_execution_plan(self) -> dict[str, Any]:
        return self._read(f"{self.root}/frozen/execution-plan.json")

    def _completed_key(self, step: ExecutionStep) -> str:
        return f"{self.root}/steps/{_token(step.step_id)}/completed.json"

    def _attempt_key(self, step: ExecutionStep, attempt: int, status: str) -> str:
        root = f"{self.root}/steps/{_token(step.step_id)}/attempts"
        return f"{root}/{attempt:04d}-{status}.json"

    def _read(self, key: str) -> dict[str, Any]:
        with self.store.open(key) as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise TypeError(f"Stored ledger object is not a mapping: {key}")
        return value


def _token(step_id: str) -> str:
    return sha256(step_id.encode()).hexdigest()[:24]


def _now() -> str:
    return datetime.now(UTC).isoformat()
