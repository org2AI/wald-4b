---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
- en
tags:
- decision-model
- typed-decisions
- calibrated-probabilities
- wald
---

# Wald-Q4B v2 — release candidate

This is a prepared model-card draft. Public v2 weights have not been published.

Wald v2 uses C16B (`04400-c18`), a completed 125-step full-parameter continuation of C16 (`04000-c16`), with 11,974,079 forward tokens. Inference weight SHA256: `2859df99ed481592cdc3731fd768bf29ccd63273b2dab623642ad3b38b0e14c4`. It is a Qwen3.5-4B chat-lineage model; no additional LoRA is attached.

The candidate default is native **Auto 0.7**, with up to **512 thought tokens**. This is an inference policy on the same weights. It differs from the legacy v1.2 plain `Reasoning:` server. Use a validated native v2 package when it is released.

## Available author-run evaluation

| Fixed policy | DI 0.2.1 sample | JevBench public 231 | Multistep 100 |
|---|---:|---:|---:|
| Thinking off | 50.75 | 203/231 (87.88%) | 72% |
| Auto 0.7 | 52.04 | 204/231 (88.31%) | 82% |
| Always 512 | 53.17 | 202/231 (87.45%) | 84% |

DI here is the historical 6,948-request sample, not a full-suite score. JevBench public is a development set, not the hidden official benchmark. These are author-run results, not maintainer-verified leaderboard scores. DI 0.3 public Auto evaluation is preparing; no score is claimed yet. Full score additionally requires organizer-run private tests.

## Provenance and release checks

The parent C16 and C16B use full-parameter weights, distinct from the separate C16 LoRA arm (`040A0-c16`). C16B's admitted continuation contains 39,699 questions in 36,509 rows, combining prior admitted replay with independently verified new TRAIN questions. New DEV questions were excluded. This short curriculum and shared-template data do not establish broad generalization or a universal improvement. A complete training-source/benchmark-overlap disclosure is required before publication; the public development results are not asserted to be held out.

Before publishing: complete public DI evaluation and literal coverage reporting; verify native serving parity; measure serial latency; publish provenance and overlap disclosures; verify every uploaded file at an immutable HF revision. v1.1/v1.2 weights and their results must remain accessible. No old DI full-suite score, latency or robustness result is attributed to v2.
