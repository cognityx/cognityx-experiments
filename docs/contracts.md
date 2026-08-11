# Contracts

## Research specification

`cognityx.research.spec/v1` is the human-authored control boundary. It records
a research area, hypotheses, linked research questions, and one or more
experiments. Each experiment may address several questions and declares its
experimental unit, treatments, control, seeds, outcomes, estimand, analysis,
stopping rule, exclusions, and resource limits.

Canonical JSON uses sorted keys, UTF-8, no insignificant whitespace, and the
original scalar values. Its SHA-256 digest is the immutable specification
checksum. YAML formatting and comments do not affect that checksum.

## Logical plan

`cognityx.experiment.plan/v1` expands every declared treatment and seed while
preserving the frozen model, data package, training, inference, evaluator,
outcome, and analysis values. Its checksum excludes only the checksum field
itself.

## Execution plan

`cognityx.experiment.execution-plan/v1` contains only known operations. A step
has an owner, dependencies, inputs, output contract, resource request, retry
policy, status, and deterministic idempotency key. The key hashes the logical
plan checksum, execution identity, experiment, treatment, seed, operation, and
immutable inputs.

## Ledger

The Storage ledger contains immutable metadata and one known object per step
state. A completed step stores only the owning component's run and manifest
references, checksums, compact observations, and timestamps. It never copies
model weights, predictions, or score blobs. Resume checks the deterministic
completed-step object before invoking a component.
