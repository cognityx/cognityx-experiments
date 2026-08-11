# Cognityx Experiments

`cognityx-experiments` turns a written research protocol into a frozen,
repeatable execution plan. In ordinary language, it keeps the question being
asked separate from the machines used to answer it, records every completed
step, and can continue after an interruption without repeating expensive work.

```text
researcher-authored YAML
          ↓
frozen research intent and scientific plan
          ↓
hardware-aware execution plan
          ↓
DataForge / Training / Inference / Evaluator
          ↓
Storage evidence + Observability index
```

The package does not train models or score candidates itself. Each operation
is delegated to the Cognityx component that owns it. Cognityx Storage remains
the durable source of truth; observation backends are searchable secondary
indexes.

Research snapshots can also feed a public results journal. Public publication
must be declared in the frozen research YAML, is allowed only for data marked
public, and produces a strict summary rather than copying record-level
evidence. Raw sources, prompts, answers, credentials, local paths, and private
Storage addresses remain outside the public repository. See
[Research publication](docs/research-publication.md) for the exact policy.

## Quick start

```bash
uv sync --locked --all-extras --dev
uv run cognityx-experiments validate examples/training-comparison.yaml
uv run cognityx-experiments plan examples/training-comparison.yaml
uv run cognityx-experiments show-plan examples/training-comparison.yaml
uv run cognityx-experiments run examples/training-comparison.yaml --dry-run
```

`--dry-run` uses explicitly synthetic component results and never creates
scientific evidence. A real run first checks every frozen input and runtime
boundary:

```bash
uv run cognityx-experiments preflight research.yaml \
  --storage-config storage.toml \
  --results-repo ../cognityx-experiment-results
uv run cognityx-experiments run research.yaml \
  --storage-config storage.toml \
  --results-repo ../cognityx-experiment-results \
  --push-results
```

The results repository must be private and clean before a real run can begin.
This is a fail-closed safety rule: if the repository is public, the preflight
reports the problem and changes nothing. The example protocol is intentionally
structural; add frozen Storage addresses for all evaluation sets and an
Inference service address before using it for production.

See the [documentation](docs/index.md) for contracts and ownership boundaries.
