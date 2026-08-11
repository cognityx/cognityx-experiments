# Component idempotency audit

Resume must cover a small but important crash window: an owning service may
finish and publish its manifest just before the experiment ledger records that
result.

- Training already accepts a stable requested run ID and publishes immutable
  terminal manifests at that identity.
- Evaluator already accepts a stable evaluator run ID.
- DataForge had deterministic dataset content identity but generated a new run
  directory on every orchestrated call. Its focused Phase 2 change adds an
  optional requested run ID; old callers are unchanged.
- Inference generated pair and arm IDs internally. Its focused stacked change
  adds an optional caller pair ID, derives stable arm IDs, returns matching
  existing manifests, and rejects changed-input reuse.

The experiment executor derives those caller values from each frozen step
idempotency key. Once a completed component result is recorded, the Storage
ledger skips the call entirely. If a caller crashes before that ledger write,
the component's immutable identity recovers the already-published result.

No experiment identity was added to Cognityx Resource or Core. These are
optional domain execution identities at the service boundary.
