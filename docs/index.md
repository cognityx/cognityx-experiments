# Cognityx Experiments

Cognityx Experiments is the part of Cognityx that records why a research run
exists and turns that intent into repeatable work. A researcher writes a YAML
file containing a hypothesis, questions, treatments, outcomes, and analysis
rules. The package freezes that file, creates a logical scientific plan, then
creates a separate execution plan for the available hardware.

```text
Research Area → Hypothesis → Research Questions → Experiments
                                              ↓
                                frozen logical plan
                                              ↓
                              local execution schedule
                                              ↓
                 owning Cognityx services → Storage evidence
```

This separation matters. Moving from one local GPU to several remote workers
may change the schedule, but it must not silently change treatments, seeds, or
the outcome being measured.

The repository sits above DataForge, Training, Inference, and Evaluator. It
delegates each task to the service that owns it. It uses Cognityx Resource for
stable governance identity, Cognityx Observability for searchable events and
metrics, and Cognityx Storage for durable evidence.

Start with the [contracts](contracts.md) or the [CLI](cli.md).
