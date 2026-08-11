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

An optional `lineage` object says which earlier frozen specification,
execution, or findings motivated this new one. This trace is called research
lineage. It may label the work `confirmatory` (checking a prior claim) or
`exploratory` (looking for a new pattern). The lineage changes the checksum,
but it never starts another experiment or changes its parent.

The primary outcome may name one of three stored evaluation roles:
`exact_recall`, `paraphrase_evaluation`, or `heldout_knowledge_unit`. When more
than one role is declared, the primary role is required. Exact recall remains
a diagnostic and is never silently averaged into a paraphrase estimate.

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

The execution plan also freezes the package version and source-code revision
for software that can change the result. This record is called a software
identity (`SoftwareIdentity`). Changing one frozen revision changes the
execution-plan checksum. Machine paths and timestamps are deliberately absent.

## Analysis records

Experiments compares the saved adapter result for each treatment. The base
model result and Inference `pair_outcome` remain useful integrity diagnostics,
but they are not the raw-versus-qualified scientific endpoint.

Records are paired by seed and `evaluation_record_id`. A missing treatment
endpoint is counted as unresolved, not as a tie. When available, knowledge
unit, fact-group, or document identity keeps resampling grouped around the
underlying fact instead of pretending paraphrases are independent facts.

`primary_endpoint_finalized` answers the narrow question “is the declared
primary number usable?” `full_evaluation_finalized` answers the broader
question “are all evaluation dimensions complete?” For example, grounded
correctness can be usable while source faithfulness still needs human review.
The primary estimate keeps that usable endpoint and reports the broader
unresolved count separately.

## Ledger

The Storage ledger contains immutable metadata and one known object per step
state. A completed step stores only the owning component's run and manifest
references, checksums, compact observations, and timestamps. It never copies
model weights, predictions, or score blobs. Resume checks the deterministic
completed-step object before invoking a component.
