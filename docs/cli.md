# Command line

```text
cognityx-experiments validate RESEARCH.yaml
cognityx-experiments plan RESEARCH.yaml
cognityx-experiments show-plan RESEARCH.yaml
cognityx-experiments run RESEARCH.yaml --dry-run [--resume]
cognityx-experiments status EXECUTION_ID --storage-root PATH
```

`validate` checks the hierarchy and controlled design. `plan` prints both
canonical checksums and the logical plan. `show-plan` prints a Mermaid view of
the execution schedule.

`run --dry-run` is deliberately marked synthetic. It proves scheduling,
Storage ledger, observability, and resume behavior without invoking a model or
creating scientific evidence. Production callers construct the executor with
typed DataForge, Training, Inference, and Evaluator gateways.
