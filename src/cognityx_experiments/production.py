"""Production composition over public Cognityx component boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from cognityx_experiments.canonical import canonical_bytes, checksum, plain
from cognityx_experiments.contracts import ExecutionStep
from cognityx_experiments.executor import ComponentResult


class ComponentCommandRunner(Protocol):
    """Run one public machine-readable component command."""

    def run(
        self, arguments: Sequence[str], *, timeout_seconds: float | None = None
    ) -> Mapping[str, Any]: ...


class ProcessHandle(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class ProcessRunner(Protocol):
    def start(self, arguments: Sequence[str]) -> ProcessHandle: ...


class HttpTransport(Protocol):
    def get(self, url: str, *, timeout_seconds: float) -> Mapping[str, Any]: ...

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class DataForgeOperation(Protocol):
    def prepare(
        self,
        step: ExecutionStep,
        *,
        run_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]: ...


class TrainingOperation(Protocol):
    def train(
        self,
        step: ExecutionStep,
        prepared: Mapping[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]: ...


class InferenceOperation(Protocol):
    def start_or_reuse(
        self,
        step: ExecutionStep,
        *,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]: ...

    def execute_pair(
        self,
        step: ExecutionStep,
        training: Mapping[str, Any],
        runtime: Mapping[str, Any],
        *,
        pair_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]: ...

    def stop(
        self,
        step: ExecutionStep,
        runtime: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]: ...


class EvaluatorOperation(Protocol):
    def evaluate(
        self,
        step: ExecutionStep,
        pair: Mapping[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]: ...


class JsonCommandRunner:
    """Subprocess adapter for Cognityx CLIs that print one JSON document."""

    def run(
        self, arguments: Sequence[str], *, timeout_seconds: float | None = None
    ) -> Mapping[str, Any]:
        completed = subprocess.run(
            list(arguments),
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        value = json.loads(completed.stdout)
        if not isinstance(value, Mapping):
            raise ValueError("Component command did not return a JSON object")
        return plain(value)


class SubprocessRunner:
    """Start an owned local Inference process without shell interpretation."""

    def start(self, arguments: Sequence[str]) -> ProcessHandle:
        return subprocess.Popen(  # noqa: S603 - explicit argument vector
            list(arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class JsonHttpTransport:
    """Small JSON HTTP adapter for the public Inference service boundary."""

    def get(self, url: str, *, timeout_seconds: float) -> Mapping[str, Any]:
        return self._request("GET", url, None, timeout_seconds)

    def post(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        return self._request("POST", url, payload, timeout_seconds)

    @staticmethod
    def _request(
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        body = canonical_bytes(payload) if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            value = json.load(response)
        if not isinstance(value, Mapping):
            raise ValueError("Inference HTTP endpoint did not return a JSON object")
        return plain(value)


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    uri: str
    checksum: str
    value: Mapping[str, Any]


class StorageEvidenceVerifier:
    """Verify authoritative component outputs through Storage read streams."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def verify_json(
        self,
        uri: str,
        *,
        expected_checksum: str | None = None,
        expected_schema: str | tuple[str, ...] | None = None,
    ) -> VerifiedManifest:
        with self.runtime.open_uri(uri) as source:
            raw = source.read()
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError(f"Storage object is not a JSON manifest: {uri}")
        selected = plain(value)
        embedded = selected.get("manifest_checksum")
        if embedded is not None:
            calculated = checksum(
                {
                    key: item
                    for key, item in selected.items()
                    if key != "manifest_checksum"
                }
            )
            if str(embedded) != calculated:
                raise ValueError(f"Manifest checksum mismatch: {uri}")
        content_checksum = hashlib.sha256(raw).hexdigest()
        effective = str(embedded or content_checksum)
        if expected_checksum and expected_checksum not in {effective, content_checksum}:
            raise ValueError(f"Expected Storage checksum does not match: {uri}")
        if expected_schema is not None:
            actual = selected.get("schema") or selected.get("schema_version")
            allowed = (
                (expected_schema,)
                if isinstance(expected_schema, str)
                else expected_schema
            )
            if actual not in allowed:
                raise ValueError(f"Unexpected manifest schema at {uri}: {actual}")
        return VerifiedManifest(uri=uri, checksum=effective, value=selected)

    def read_jsonl(
        self, uri: str, *, expected_checksum: str | None = None
    ) -> tuple[tuple[Mapping[str, Any], ...], str]:
        with self.runtime.open_uri(uri) as source:
            raw = source.read()
        digest = hashlib.sha256(raw).hexdigest()
        text_value = raw.decode()
        dataforge_digest = checksum(text_value)
        if expected_checksum and expected_checksum not in {
            digest,
            dataforge_digest,
        }:
            raise ValueError(f"JSONL checksum mismatch: {uri}")
        rows = tuple(
            plain(json.loads(line)) for line in text_value.splitlines() if line.strip()
        )
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError(f"JSONL contains a non-object row: {uri}")
        return rows, digest

    def publish_reference(
        self, step: ExecutionStep, value: Mapping[str, Any]
    ) -> VerifiedManifest:
        payload = {
            "schema": "cognityx.experiment.component-reference/v1",
            "step_id": step.step_id,
            "idempotency_key": step.idempotency_key,
            **plain(value),
        }
        store = self.runtime.for_role("artifact")
        key = f"experiments/component-references/{step.idempotency_key}.json"
        stored = store.put_json_idempotent(key, payload)
        return self.verify_json(str(stored.uri))


class CliDataForgeOperation:
    """Use DataForge's public build and research-package JSON CLI commands."""

    def __init__(self, runner: ComponentCommandRunner) -> None:
        self.runner = runner

    def prepare(
        self,
        step: ExecutionStep,
        *,
        run_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]:
        del parent_run_id
        treatment = dict(step.input_references["treatment"])
        inputs = dict(treatment.get("inputs") or {})
        config = dict(step.input_references.get("dataforge") or {})
        package_uri = inputs.get("research_package_uri")
        if package_uri:
            result = self.runner.run(
                [
                    str(config.get("executable") or "cognityx-dataforge"),
                    "research-package",
                    "show",
                    str(package_uri),
                    *_storage_arguments(config),
                ]
            )
            return {**plain(result), "manifest_uri": str(package_uri)}
        dataset_uri = inputs.get("dataset_manifest_uri") or inputs.get(
            "data_package_uri"
        )
        if not dataset_uri:
            result = self.runner.run(
                [
                    str(config.get("executable") or "cognityx-dataforge"),
                    "build",
                    str(config["recipe"]),
                    "--source",
                    str(config["source"]),
                    "--experiment-id",
                    step.experiment_id,
                    "--run-id",
                    run_id,
                    "--config",
                    str(config["config"]),
                    *_storage_arguments(config),
                ],
                timeout_seconds=float(config.get("timeout_seconds", 3600)),
            )
            dataset_uri = result["dataset_manifest_uri"]
        suites = tuple(step.input_references.get("evaluation_suites") or ())
        evaluation_uris = [str(value["manifest_uri"]) for value in suites]
        return self.runner.run(
            [
                str(config.get("executable") or "cognityx-dataforge"),
                "research-package",
                "create",
                "--name",
                f"{step.experiment_id}-{step.treatment_id}",
                "--dataset-manifest",
                str(dataset_uri),
                *[
                    argument
                    for uri in evaluation_uris
                    for argument in ("--evaluation-manifest", uri)
                ],
                *_storage_arguments(config),
            ]
        )


class CliTrainingOperation:
    """Use Training's configuration CLI with narrow per-run overrides."""

    def __init__(self, runner: ComponentCommandRunner) -> None:
        self.runner = runner

    def train(
        self,
        step: ExecutionStep,
        prepared: Mapping[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]:
        config = dict(step.input_references.get("training") or {})
        arguments = build_training_command(
            step,
            manifest_uri=str(prepared["manifest_uri"]),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        return self.runner.run(
            arguments,
            timeout_seconds=float(config.get("timeout_seconds", 21600)),
        )


class HttpInferenceOperation:
    """Use public Inference HTTP plus its supported local service command."""

    def __init__(
        self,
        transport: HttpTransport,
        process_runner: ProcessRunner,
        *,
        sleep: Any = time.sleep,
    ) -> None:
        self.transport = transport
        self.process_runner = process_runner
        self.sleep = sleep
        self._owned: dict[str, ProcessHandle] = {}

    def start_or_reuse(
        self,
        step: ExecutionStep,
        *,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]:
        del parent_run_id
        inference = dict(step.input_references.get("inference") or {})
        service = dict(inference.get("service") or {})
        mode = str(service.get("mode") or "external")
        base_url = str(service.get("base_url") or "http://127.0.0.1:8000").rstrip("/")
        owned = False
        if mode == "local_managed" and base_url not in self._owned:
            command = service.get("command") or [
                "cognityx-inference",
                "serve",
                "--host",
                str(service.get("host") or "127.0.0.1"),
                "--port",
                str(service.get("port") or 8000),
            ]
            self._owned[base_url] = self.process_runner.start(tuple(map(str, command)))
            owned = True
        timeout = float(service.get("cold_load_timeout_seconds", 900))
        deadline = time.monotonic() + timeout
        while True:
            try:
                models = self.transport.get(f"{base_url}/v1/models", timeout_seconds=10)
                break
            except (OSError, urllib.error.URLError) as exc:
                process = self._owned.get(base_url)
                if process is not None and process.poll() is not None:
                    raise RuntimeError(
                        "Owned Inference process exited before readiness"
                    ) from exc
                if time.monotonic() >= deadline:
                    self._cleanup(base_url)
                    raise TimeoutError(
                        "Inference service did not become ready"
                    ) from exc
                self.sleep(1)
        return {
            "mode": mode,
            "base_url": base_url,
            "owned": owned or base_url in self._owned,
            "service_identity": plain(models),
        }

    def execute_pair(
        self,
        step: ExecutionStep,
        training: Mapping[str, Any],
        runtime: Mapping[str, Any],
        *,
        pair_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]:
        model = dict(step.input_references.get("model") or {})
        inference = dict(step.input_references.get("inference") or {})
        suite = dict(step.input_references["evaluation_suite"])
        payload = {
            "evaluation_manifest_uri": suite["manifest_uri"],
            "adapter_manifest_uri": training["adapter_manifest_uri"],
            "model": model.get("name") or model.get("model"),
            "model_revision": model.get("revision"),
            "backend": inference.get("backend", "vllm"),
            "profile": inference.get("profile", "bf16"),
            "temperature": inference.get("temperature", 0),
            "top_p": inference.get("top_p", 1),
            "max_output_tokens": inference.get("max_output_tokens", 512),
            "thinking": inference.get("thinking", "disabled"),
            "required_context_length": inference.get("required_context_length"),
            "runtime": inference.get("runtime") or {},
            "inference_pair_id": pair_id,
            "research_context": {
                "experiment_id": step.experiment_id,
                "comparison_id": step.experiment_id,
                "arm_id": step.treatment_id,
                "seed": step.seed,
                "parent_run_id": parent_run_id,
                "training_run_id": training.get("run_id"),
            },
        }
        base_url = str(runtime["base_url"]).rstrip("/")
        return self.transport.post(
            f"{base_url}/v1/cognityx/research/pairs",
            payload,
            timeout_seconds=float(inference.get("pair_timeout_seconds", 3600)),
        )

    def stop(
        self,
        step: ExecutionStep,
        runtime: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]:
        del step, parent_run_id
        base_url = str(runtime["base_url"])
        stopped = self._cleanup(base_url)
        return {"base_url": base_url, "owned_process_stopped": stopped}

    def _cleanup(self, base_url: str) -> bool:
        process = self._owned.pop(base_url, None)
        if process is None:
            return False
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        return True


class CliEvaluatorOperation:
    """Use Evaluator's public pair command and machine-readable result."""

    def __init__(self, runner: ComponentCommandRunner) -> None:
        self.runner = runner

    def evaluate(
        self,
        step: ExecutionStep,
        pair: Mapping[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None,
    ) -> Mapping[str, Any]:
        config = dict(step.input_references.get("evaluator") or {})
        arguments = [
            str(config.get("executable") or "cognityx-evaluator"),
            "evaluate",
            "pair",
            "--pair-manifest",
            str(pair["manifest_uri"]),
            "--experiment-id",
            step.experiment_id,
            "--evaluator-run-id",
            run_id,
            *_storage_arguments(config),
        ]
        if config.get("deterministic_only"):
            arguments.append("--deterministic-only")
        if config.get("judge_config"):
            arguments.extend(("--judge-config", str(config["judge_config"])))
        if parent_run_id:
            arguments.extend(("--parent-run-id", parent_run_id))
        return self.runner.run(
            arguments,
            timeout_seconds=float(config.get("timeout_seconds", 3600)),
        )


class CognityxComponentGateway:
    """Compose public Cognityx seams and return verified compact references."""

    def __init__(
        self,
        storage_runtime: Any,
        *,
        dataforge: DataForgeOperation | None = None,
        training: TrainingOperation | None = None,
        inference: InferenceOperation | None = None,
        evaluator: EvaluatorOperation | None = None,
        command_runner: ComponentCommandRunner | None = None,
        http_transport: HttpTransport | None = None,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        commands = command_runner or JsonCommandRunner()
        self.evidence = StorageEvidenceVerifier(storage_runtime)
        self.dataforge = dataforge or CliDataForgeOperation(commands)
        self.training = training or CliTrainingOperation(commands)
        self.inference = inference or HttpInferenceOperation(
            http_transport or JsonHttpTransport(),
            process_runner or SubprocessRunner(),
        )
        self.evaluator = evaluator or CliEvaluatorOperation(commands)

    def prepare_data(
        self, step: ExecutionStep, *, parent_run_id: str | None
    ) -> ComponentResult:
        run_id = f"df-{step.idempotency_key[:24]}"
        result = self.dataforge.prepare(
            step, run_id=run_id, parent_run_id=parent_run_id
        )
        uri = str(
            result.get("manifest_uri") or result.get("research_package_manifest_uri")
        )
        verified = self.evidence.verify_json(
            uri,
            expected_checksum=_optional_string(result.get("manifest_checksum")),
            expected_schema="cognityx.dataforge.research-package/v1",
        )
        return ComponentResult(
            manifest_uri=uri,
            manifest_checksum=verified.checksum,
            run_id=str(result.get("run_id") or run_id),
            attributes={
                "research_package_id": verified.value.get("research_package_id"),
                "research_package_version": verified.value.get(
                    "research_package_version"
                ),
                "evaluation_sets": plain(verified.value.get("evaluation_sets") or []),
            },
        )

    def train(
        self,
        step: ExecutionStep,
        prepared: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        run_id = f"trun-{step.idempotency_key[:24]}"
        result = self.training.train(
            step,
            prepared,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        uri = str(result["publication_manifest_uri"])
        publication = self.evidence.verify_json(
            uri, expected_schema="cognityx.training.publication/v1"
        )
        if publication.value.get("status") != "completed":
            raise ValueError("Training publication is not completed")
        adapter_uri = str(
            publication.value.get("adapter_manifest_uri")
            or result.get("adapter_manifest_uri")
        )
        adapter = self.evidence.verify_json(
            adapter_uri, expected_schema="cognityx.training.adapter/v1"
        )
        return ComponentResult(
            manifest_uri=uri,
            manifest_checksum=publication.checksum,
            run_id=str(publication.value.get("training_run_id") or run_id),
            attributes={
                "adapter_manifest_uri": adapter.uri,
                "adapter_manifest_checksum": adapter.checksum,
                "adapter_id": adapter.value.get("adapter_id"),
            },
        )

    def start_or_reuse_inference(
        self,
        step: ExecutionStep,
        training_results: Sequence[Mapping[str, Any]],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        del training_results
        result = self.inference.start_or_reuse(step, parent_run_id=parent_run_id)
        reference = self.evidence.publish_reference(step, result)
        return ComponentResult(
            manifest_uri=reference.uri,
            manifest_checksum=reference.checksum,
            run_id=f"runtime-{step.idempotency_key[:24]}",
            attributes=plain(result),
        )

    def execute_inference_pair(
        self,
        step: ExecutionStep,
        training_result: Mapping[str, Any],
        runtime_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        pair_id = f"pair-{step.idempotency_key[:24]}"
        training = {
            **plain(training_result.get("attributes") or {}),
            "manifest_uri": training_result.get("manifest_uri"),
            "run_id": training_result.get("run_id"),
        }
        runtime = plain(runtime_result.get("attributes") or {})
        result = self.inference.execute_pair(
            step,
            training,
            runtime,
            pair_id=pair_id,
            parent_run_id=parent_run_id,
        )
        uri = str(result["manifest_uri"])
        verified = self.evidence.verify_json(
            uri,
            expected_checksum=_optional_string(result.get("manifest_checksum")),
            expected_schema="cognityx.inference.pair/v1",
        )
        if (
            verified.value.get("status") != "completed"
            or verified.value.get("pair_validation") != "passed"
        ):
            raise ValueError("Inference pair publication did not pass validation")
        return ComponentResult(
            manifest_uri=uri,
            manifest_checksum=verified.checksum,
            run_id=str(verified.value.get("inference_pair_id") or pair_id),
            attributes={
                "research_role": (
                    step.input_references["evaluation_suite"]["research_role"]
                ),
                "base_run": plain(verified.value.get("base_run") or {}),
                "adapter_run": plain(verified.value.get("adapter_run") or {}),
            },
        )

    def stop_inference(
        self,
        step: ExecutionStep,
        runtime_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        runtime = plain(runtime_result.get("attributes") or {})
        result = self.inference.stop(step, runtime, parent_run_id=parent_run_id)
        reference = self.evidence.publish_reference(step, result)
        return ComponentResult(
            manifest_uri=reference.uri,
            manifest_checksum=reference.checksum,
            run_id=f"runtime-stop-{step.idempotency_key[:20]}",
            attributes=plain(result),
        )

    def evaluate_pair(
        self,
        step: ExecutionStep,
        pair_result: Mapping[str, Any],
        *,
        parent_run_id: str | None,
    ) -> ComponentResult:
        run_id = f"eval-{step.idempotency_key[:24]}"
        pair = {
            "manifest_uri": pair_result["manifest_uri"],
            **plain(pair_result.get("attributes") or {}),
        }
        result = self.evaluator.evaluate(
            step,
            pair,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        uri = str(result["manifest_uri"])
        manifest = self.evidence.verify_json(
            uri,
            expected_checksum=_optional_string(result.get("manifest_checksum")),
            expected_schema="cognityx.evaluator.run/v1",
        )
        scores, _ = self.evidence.read_jsonl(
            str(manifest.value["scores_uri"]),
            expected_checksum=str(manifest.value["scores_checksum"]),
        )
        if step.seed is None:
            raise ValueError("Evaluator step is missing its frozen seed")
        records = tuple(
            _analysis_record(
                row,
                treatment_id=str(step.treatment_id),
                seed=step.seed,
                manifest=manifest.value,
            )
            for row in scores
        )
        return ComponentResult(
            manifest_uri=uri,
            manifest_checksum=manifest.checksum,
            run_id=str(manifest.value.get("evaluator_run_id") or run_id),
            attributes={
                "scores_uri": manifest.value["scores_uri"],
                "scores_checksum": manifest.value["scores_checksum"],
                "summary_uri": manifest.value.get("summary_uri"),
            },
            analysis_records=records,
        )


def _analysis_record(
    row: Mapping[str, Any],
    *,
    treatment_id: str,
    seed: int,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    adapter = dict(row.get("adapter") or {})
    grounded = adapter.get("grounded_correct")
    generation_status = adapter.get("generation_status")
    evaluator_status = adapter.get("evaluation_status")
    primary_finalized = generation_status == "complete" and isinstance(grounded, bool)
    full_finalized = (
        evaluator_status == "scored"
        and all(
            adapter.get(name) in {"passed", "failed"}
            for name in (
                "answer_correctness",
                "required_fact_completeness",
                "source_faithfulness",
            )
        )
        and isinstance(adapter.get("fatal_contradiction"), bool)
    )
    judge = adapter.get("semantic_judge_result") or {}
    return {
        "evaluation_record_id": row["evaluation_record_id"],
        "treatment_id": treatment_id,
        "seed": seed,
        "research_role": row["research_role"],
        "grounded_correct": grounded,
        "primary_endpoint_finalized": primary_finalized,
        "full_evaluation_finalized": full_finalized,
        "answer_correctness": adapter.get("answer_correctness"),
        "required_fact_completeness": adapter.get("required_fact_completeness"),
        "fatal_contradiction": adapter.get("fatal_contradiction"),
        "source_faithfulness": adapter.get("source_faithfulness"),
        "generation_status": generation_status,
        "evaluator_status": evaluator_status,
        "knowledge_unit_id": row.get("knowledge_unit_id"),
        "fact_group_id": row.get("fact_group_id"),
        "document_id": row.get("document_id"),
        "resources": plain(manifest.get("resources") or {}),
        "semantic_judge_invocation_cost": float(
            judge.get("invocation_cost") or judge.get("cost") or 0
        ),
        "base_diagnostic": plain(row.get("base") or {}),
        "pair_outcome_diagnostic": row.get("pair_outcome"),
    }


def _storage_arguments(config: Mapping[str, Any]) -> list[str]:
    if config.get("storage_config"):
        return ["--storage-config", str(config["storage_config"])]
    if config.get("storage_root"):
        return ["--storage-root", str(config["storage_root"])]
    return []


def build_training_command(
    step: ExecutionStep,
    *,
    manifest_uri: str,
    run_id: str,
    parent_run_id: str | None,
    dry_run: bool = False,
) -> tuple[str, ...]:
    """Build the one authoritative Training CLI argument vector."""
    if step.seed is None:
        raise ValueError("Training step is missing its frozen seed")
    config = dict(step.input_references.get("training") or {})
    arguments = [
        str(config.get("executable") or "cognityx-training"),
        "--config",
        str(config["config"]),
        "--dataset-uri",
        manifest_uri,
        "--dataset-input-mode",
        "dataforge_manifest",
        "--run-id",
        run_id,
        "--experiment-id",
        f"exp-{checksum(step.experiment_id)[:24]}",
        "--seed",
        str(step.seed),
        *_storage_arguments(config),
    ]
    if parent_run_id:
        arguments.extend(("--parent-run-id", parent_run_id))
    if dry_run:
        arguments.append("--dry-run")
    return tuple(arguments)


def _optional_string(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None
