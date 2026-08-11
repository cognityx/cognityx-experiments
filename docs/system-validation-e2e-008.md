# SYSTEM-VALIDATION e2e-008 engineering record

> **SYSTEM VALIDATION — NOT TRAIN-H1 SCIENTIFIC EVIDENCE.**

This record preserves the software and machine facts for the successful
end-to-end engineering shakedown. It is a reproducibility checklist, not a
scientific result. The immutable execution and all earlier failed attempts
remain in Cognityx Storage and the public result snapshots; this page does not
replace or rewrite them.

## Frozen execution identity

| Field | Frozen value |
| --- | --- |
| Execution | `system-validation-e2e-008` |
| Research specification checksum | `04f6cd501288695ac9e143ad13c85cbbbf1234b3793737e5825cf6638244993e` |
| Logical plan checksum | `b970bd6dd314d1abb8f3bafe23bc3cd6cda6bb701059de90902db6189ef01224` |
| Execution-plan checksum | `aa11916fcd18c246cec8754f2521966130d275b01eac97e48efc0b59ab7037ef` |
| Terminal snapshot | `73e48df5e55bb16f31af861ea892a057c2f513a726b95725454366c67d905841` |
| Public result commit | `d3249501b2cddbad896d1214f0bc5ba2deee5548` |
| Completed ledger | 11 of 11 steps; 0 unsupported |
| Final exact resume | 11 reused; 0 newly executed; 0 expensive steps repeated |

## Software revisions used by the shakedown

| Repository | Revision |
| --- | --- |
| Core | `571f2291852ad22fe8427172bca71fc6a7a74a3d` |
| Observability | `b02ec3c103d0b3fd22b7c33888155b763d6baa9f` |
| DataForge | `5bb8c1fea5385036a6be13f774f987617fb1ef38` |
| Training | `f62a280a8b427d5e3612011603cc951c961bf410` |
| Inference | `061cb6a9cfd9c999c4a6ac68f650a2b0f6efd3c7` |
| Evaluator | `e951ac66c8ba60261912f71450d7f2cdfd8a5a4a` |
| Experiments | `b7138689c33cd102370932e2269cba8c88a7d692` |
| Resource | `b23220b69fcb182e681cf13276c37474666c9bd2` |
| Storage | `4b47b898b2fb465263d8c44350d4241f52b13c90` |
| Jobs | `e4312fd461df97ffcefc54352b9b76f1dd6e6860` |
| SDK | `c83736fd8eea425c8989f26b42bb9700d87a6dec` |

Later stabilization merges must be recorded separately. They do not alter the
software identity of this frozen historical execution.

## Runtime and model identity

| Field | Frozen or observed value |
| --- | --- |
| Python | `3.12.3` |
| Torch | `2.13.0` |
| Transformers | `5.15.0` |
| PEFT | `0.20.0` |
| BitsAndBytes | `0.50.0` |
| Accelerate | `1.14.0` |
| vLLM | `0.25.1` |
| Model | `Qwen/Qwen3-8B` |
| Model revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Tokenizer revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Tokenizer checksum | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Chat-template checksum | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| Inference runtime | `vllm-0.25.1-profile-f208c6aa7a79f9600326` |
| Certified profile | `f208c6aa7a79f9600326`; INT4; 40,960-token certified context |
| GPU | `NVIDIA GeForce RTX 5090` |
| Driver | `610.62` (readiness re-audit on 2026-08-11) |
| CUDA reported by Training | `13.0` |
| Storage profile | built-in `local-main` filesystem profile |
| Storage configuration checksum | `6d3fbd9ccb2041b07ca47a606f7365ca2509f28ac2024e9ab6662e7bbcb059bc` |

The record deliberately excludes local filesystem paths, credentials, raw
prompts, source passages, answers, model weights, and adapter bytes.

## Historical installation diagnosis

The first e2e-008 Training attempt installed the base
`cognityx-training` distribution directly into the SDK runtime. It did not
request Training's `[training]` execution extra. Other overlapping packages
were present through the wider environment, which made the package importable,
but PEFT was absent. The lock behaved as requested; it did not silently drop an
extra. The same frozen execution was repaired with PEFT and resumed safely.

The permanent boundary is now explicit: Training owns the execution package
set and exposes a no-model capability check; Experiments requires that check
and the existing data/configuration dry run to pass before preregistration.
