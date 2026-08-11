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
                  Storage evidence -> analysis -> private Git journal
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

These are explicit adapters rather than a generic plugin framework because the
set of scientific operations is small and each owner has a different contract.
The separation prevents Experiments from interpreting private model or storage
details.

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
durable evidence store. A real run requires the repository identity
`cognityx/cognityx-experiment-results`, a clean worktree, and private visibility.
If any condition is false, preflight stops before preregistration, training, or
publication. It never changes repository visibility.

Publication writes only a whitelist of sanitized research files. Secret-like
fields, raw answer text, environment values, and local paths are removed or
replaced. A Storage receipt records the exact Git commit and snapshot path.

## Interruption and retry

Each expensive step has a stable identity derived from the frozen inputs. A
completed step is written to the Storage ledger before the next step starts.
On `--resume`, completed DataForge, Training, Inference-pair, and Evaluator work
is reused. Narrative generation and Git push are separate from scientific
completion, so either can be retried without rerunning the science.

The gateway contract tests exercise the real Storage verifier and Git journal
with injected component clients. They cover a complete 33-step two-treatment,
two-seed, three-role run and resume after failures at the Training and Evaluator
boundaries. Separate pipeline tests cover narrative and Git-push retry.

## Compatibility and limits

The composition extends public commands and services; it does not import their
private APIs. Local Inference readiness uses a long cold-load timeout and the
declared certified hardware profile. Preflight may read GPU inventory, but it
never loads a model. Actual model execution remains a separate, explicitly
approved shakedown after every preflight condition passes.
