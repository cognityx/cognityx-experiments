# Jobs integration decision

Cognityx Jobs is a good owner for service-local job state, cancellation, and
replayable progress events. Its current create operation expects a new job ID
and does not provide the Storage-backed, caller-idempotency lookup needed to
prove that an expensive cross-service experiment step already completed.

Phase 2 therefore keeps experiment execution synchronous and local. The
experiment ledger in Storage is authoritative for resume. This avoids a second
queue and avoids changing Jobs merely to fit research orchestration.

A future worker may mirror each typed experiment step into Jobs for visibility
and cancellation. That integration should use the existing `JobRepository`
contract while continuing to consult the experiment Storage ledger before
domain work. Jobs would own generic lifecycle; Experiments would still own the
scientific plan and step idempotency.
