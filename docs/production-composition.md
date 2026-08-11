# Production composition

Cognityx Experiments is the conductor of a research run. It reads the frozen
plan, asks each specialist service to do its part, and keeps compact addresses
to the evidence. It does not copy the specialist's internal logic. This narrow
connection to an owning service is technically called an adapter.

```text
ResearchSpec and frozen software identities
                    |
                    v
          fail-closed preflight
                    |
                    v
 DataForge -> Training -> one Inference service
                         |              |
                         +-> base/adapter pairs -> Evaluator
                                                   |
                                                   v
             Storage evidence -> analysis -> policy-gated Git journal
```

## Public component seams

The production gateway uses the existing public boundaries:

- DataForge is called through its machine-readable build and research-package
  commands. It owns dataset preparation and package meaning.
- Training is called through its configuration command with the frozen dataset
  address, seed, run identity, and optional parent observation identity. It owns
  adapter creation and publication.
- Inference is called through its public JSON service. A run can reuse an
  external service or start one explicitly declared local process. Experiments
  stops only a process it started itself.
- Evaluator is called through its public pair command. It owns every score and
  comparison diagnostic.
- Storage is used to reopen and verify every returned manifest and score file.
  A component's standard output is only a compact handoff, never the evidence
  itself.

Before preregistration, preflight asks Training two separate questions through
its public command. First, `--check-runtime` verifies that the Training-owned
execution libraries are installed and, when four-bit training is configured,
that Torch can see CUDA. Experiments validates the result but does not copy or
redefine Training's package list. Second, the shared production command builder
adds `--dry-run`, so Training validates the effective configuration, Storage
package, tokenizer selection, sequence length and accepted examples. Neither
check loads model weights, creates an optimizer, trains, or writes research
evidence.

A locally managed Inference service has a second inexpensive contract check.
The frozen configuration supplies the production launcher and may supply a
`probe_command` that uses that same launcher. Preflight executes the probe and
requires a successful exit before any GPU work. The probe is responsible for
importing the service and its direct runtime packages without loading model
weights. This catches a broken virtual environment or incomplete Python search
path before Training starts. Preflight records only the launcher name, exit
status, argument count, and output byte counts; it does not copy diagnostic
output into research evidence.

Every component command has one simple machine boundary: a successful command
writes exactly one JSON object to standard output. Progress and warnings belong
on standard error. Experiments does not search for a JSON fragment inside human
text; malformed or non-object output fails with a safe contract error that
records only the executable name, exit status, and output byte counts.

The shared Training command always requests `--output-format json`. Its dry-run
response supplies the safe record and batch counts directly, so preflight does
not parse human sentences. A completed response is only a handoff: Experiments
still reads and verifies the immutable Training publication and adapter
manifests from Storage before it records the step as complete.

These are explicit adapters rather than a generic plugin framework because the
set of scientific operations is small and each owner has a different contract.
The separation prevents Experiments from interpreting private model or storage
details.

## Observation readiness

Research may select structured JSON logging or MLflow without changing any
component's scoring or generation meaning. When MLflow is selected, preflight
requires an explicit tracking URI and experiment name, constructs a read-only
MLflow client, and verifies the index can be queried. It creates no run during
preflight. Root and component contexts carry parent run identities; components
publish scalar metrics and Storage URI/checksum references. Storage remains the
authoritative evidence store, while MLflow is only a search and visualization
index.

## What is frozen

The execution-plan checksum includes exact software identities for every
component that can change a scientific result. A software identity contains the
installed package version and source revision. Production preflight rejects a
missing, unknown, or different identity. It does not infer a revision from an
unrelated current working directory.

The protocol must also name an immutable model revision, deterministic decoding
settings, distinct training packages and evaluation manifests, and an explicit
primary evaluation role. Evaluation records must say they are excluded from
training (`training_eligible: false`).

## Safe publication boundary

The Git repository is a readable research journal, while Storage remains the
durable evidence store. A real run requires a clean worktree whose Git identity
matches the repository frozen in the research YAML. It never changes repository
visibility.

The default policy (`private_required`) accepts only a private repository and
keeps the existing `full`, `sanitized`, and `metadata_only` content choices. A
deliberate `public_summary` policy may use a public repository only when the
same frozen configuration classifies the data as `public`. Internal,
confidential, restricted, and unspecified data fail closed against public Git.
A private repository may also use `public_summary`, but it still receives only
the stricter summary.

The public summary is constructed from approved research identities,
revisions, checksums, aggregate statistics, findings, tables, and figure data.
It omits individual records, prompts, answers, source content, private Storage
addresses, credentials, environment values, and local paths. A Storage receipt
records the exact Git commit and snapshot path. If repository identity,
visibility, classification, or content checks fail, preflight stops before
preregistration, training, or publication.

## Interruption and retry

Each expensive step has a stable identity derived from the frozen inputs. A
completed step is written to the Storage ledger before the next step starts.
On `--resume`, completed DataForge, Training, Inference-pair, and Evaluator work
is reused. Narrative generation and Git push are separate from scientific
completion, so either can be retried without rerunning the science. When an
immutable snapshot is already present, its receipt continues to name the Git
commit that originally published that snapshot even if later snapshots have
advanced the results repository.

The gateway contract tests exercise the real Storage verifier and Git journal
with injected component clients. They cover a complete 33-step two-treatment,
two-seed, three-role run and resume after failures at the Training and Evaluator
boundaries. Separate pipeline tests cover narrative and Git-push retry.

## Compatibility and limits

The composition extends public commands and services; it does not import their
private APIs. Local Inference readiness uses a long cold-load timeout and the
declared certified hardware profile. Preflight resolves the declared launcher,
runs its no-model probe, and may read GPU inventory, but it never loads a model.
Actual model execution remains a separate, explicitly approved shakedown after
every preflight condition passes.
