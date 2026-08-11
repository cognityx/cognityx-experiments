# Automatic research materials

Every completed experiment produces a machine-readable finding and a minimum
human-readable note. The finding keeps six categories separate: directly
observed evidence, confirmatory interpretation, exploratory observations,
literature questions, proposed follow-up questions, and later human notes.

The deterministic generator never declares a hypothesis proven, never accepts
a null hypothesis, and never claims novelty. It records the observed effect and
sets the hypothesis relationship to `inconclusive` until human review.

An optional narrative synthesizer may enrich prose through Cognityx Inference.
It cannot call a vendor SDK directly and may only see frozen structured
evidence. Its output cannot modify metrics, treatment identities, hypotheses,
questions, or significance. If synthesis fails, the deterministic finding,
factual note, table, and figure data still exist and the material status becomes
`pending_enrichment`.

The cumulative journal appends immutable finding and evidence rows under the
research area, hypothesis, and research question. Derived Markdown summaries
and CSV tables can be regenerated. This makes paper preparation continuous
rather than a manual step after expensive execution.
