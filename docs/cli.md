# Command line

The command line turns a written protocol into a checked plan, then calls the
owning Cognityx services. It does not contain a second training, inference, or
evaluation implementation.

## Check a real run without starting it

```bash
cognityx-experiments preflight research.yaml \
  --storage-config storage.toml \
  --results-repo ../cognityx-experiment-results
```

This returns a structured checklist for the research plan, installed software,
Storage inputs, Inference service, GPU requirement, observation backend,
budget, and the policy-gated Git journal. Its output records the observed
repository visibility, frozen publication policy and data classification, and
effective content projection. It may inspect a GPU inventory, but it does not
load a model or start training. Any failed item makes the command return a
non-zero status.

## Run or resume

```bash
cognityx-experiments run research.yaml \
  --storage-config storage.toml \
  --results-repo ../cognityx-experiment-results \
  --push-results

cognityx-experiments run research.yaml \
  --storage-config storage.toml \
  --results-repo ../cognityx-experiment-results \
  --push-results --resume
```

A real run always performs the same preflight before it writes a
preregistration or invokes a component. `--resume` reuses an immutable
completed-step record and retries only work that did not complete. The
publisher accepts only the expected results repository, requires a clean
worktree, stages an exact file list, and uses a fast-forward-only update before
a push. Repository identity, public-summary permission, and data classification
come from the frozen research YAML; there is no command-line option that can
weaken them.

Use `--dry-run` only to check the shape of an experiment. Dry-run records are
marked synthetic and are not scientific evidence.

```text
cognityx-experiments validate RESEARCH.yaml
cognityx-experiments plan RESEARCH.yaml
cognityx-experiments show-plan RESEARCH.yaml
cognityx-experiments preflight RESEARCH.yaml --results-repo PATH
cognityx-experiments run RESEARCH.yaml --dry-run [--resume]
cognityx-experiments run RESEARCH.yaml --results-repo PATH [--push-results] [--resume]
cognityx-experiments status EXECUTION_ID --storage-root PATH
cognityx-experiments research-summary HYPOTHESIS_OR_RQ --results-repo PATH
cognityx-experiments paper-material HYPOTHESIS_OR_RQ --results-repo PATH
```

`validate` checks the hierarchy and controlled design. `plan` prints both
canonical checksums and the logical plan. `show-plan` prints a Mermaid view of
the execution schedule.

`run --dry-run` is deliberately marked synthetic. It proves scheduling,
Storage ledger, observability, and resume behavior without invoking a model or
creating scientific evidence. Production callers construct the executor with
typed DataForge, Training, Inference, and Evaluator gateways.

`research-summary` reads immutable finding objects and groups supportive,
contradictory, inconclusive, and unresolved evidence. `paper-material` assembles
methods-ready experiment IDs, factual Results paragraphs, tables, figure data,
negative results, limitations, and provenance references. Neither command
generates a final paper.
