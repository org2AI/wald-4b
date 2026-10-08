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
- calibration
- calibrated-probabilities
- classification
- tool-selection
- agent-routing
- decision-index
- jevbench
- jev-compatible
- systemone
- wald
- wald-q4b
- qwen3.5
- 4b
- vllm
- reasoning
- gguf
- llama.cpp
model-index:
- name: Wald-Q4B v2 (04701-c22, revision v2.0)
  results:
  - task:
      type: text-classification
      name: Structured decisions (option probabilities)
    dataset:
      name: Decision Index 0.3, complete public suite (140,620 requests)
      type: decision-index
      revision: 62d2f51de34a2de64906345b6bc3e98e27ff55c7
    metrics:
    - type: balanced_skill_index
      name: Public index, Auto 0.7 (author-run with the official kit; not the organizer's Full score)
      value: 54.18
      verified: false
---

# Wald-Q4B v2

An open 4B decision model: send a state and typed options, get a calibrated probability for every option. Served through a Jev-compatible `POST /v1/systemone` API on your own GPU.

## Why Wald-4B v2

- **Probabilities, not text:** a calibrated probability for every option, nothing to parse.
- **Fast:** median **42 ms** one pass, **148 ms** Auto 0.7 (serial, one RTX PRO 6000).
- **Thinks only when unsure:** about **one third** of requests trigger a short native thought.
- **JevBench public: 210/231**, against 200/231 for Jev 1.13.
- **Decision Index 0.3: 54.18** on the complete public suite (our run).
- **Reads images** zero-shot: **+2.8 points** over its base on CV-Bench, BLINK, RealWorldQA.
- **Small and open:** 4B, Apache-2.0; GGUF from 2.7 GB in [org2ai/Wald-4B-GGUF](https://huggingface.co/org2ai/Wald-4B-GGUF).

## Scores

| Benchmark | Wald-Q4B v2 · one pass | **Wald-Q4B v2 · Auto 0.7** | Jev 1.13 (hosted) |
|---|---:|---:|---:|
| Decision Index 0.3, complete public suite | not run | **54.18** (our full run) | 57.96 (board public column) |
| JevBench public set (231) | 204/231 | **210/231** | 200/231 |
| JevBench-XL TEST (8,177) | 65.20 % | **65.78 %** | 67.85 % |

Self-run, not leaderboard results; how each number was produced: [evaluation/v2/](https://huggingface.co/org2ai/Wald-4B/tree/main/evaluation/v2).

## Quick start

```sh
hf download org2ai/Wald-4B --revision v2.0 --local-dir ./Wald-Q4B-v2 && cd Wald-Q4B-v2
./run.sh "$PWD"          # POST /v1/systemone on :8000 (one NVIDIA GPU, uv)

curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "The customer wants to return a damaged kettle.",
  "questions": {"route": {"type": "choice", "instructions": "Choose the support queue.",
    "criteria": {"returns": "Returns and refunds", "delivery": "Delivery tracking", "other": "Other"}}}}'

curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Photo from the returns desk.", "images": ["data:image/jpeg;base64,..."],
  "questions": {"damaged": {"type": "noul", "instructions": "Is the item visibly damaged?"}}}'
```

API fields: [docs/api.md](https://huggingface.co/org2ai/Wald-4B/blob/main/docs/api.md). Server options, Docker and evaluation settings: [RUNBOOK.md](https://huggingface.co/org2ai/Wald-4B/blob/main/RUNBOOK.md).

## How it works

- One forward pass reads a probability for every option from the model's option-letter logits (Qwen chat template), calibrated with a frozen temperature table.
- Auto 0.7 (default): if the top probability is below 0.7, the model thinks once natively (at most 512 tokens) and reads the options again.
- More than 26 options: a one-pass knockout over all options.
- Images go through Qwen3.5-4B's vision tower, unchanged (zero-shot).

## Details

- **Training.** Full-parameter decision training from Qwen3.5-4B (chat); checkpoint `04701-c22`.
- **Data.** About 15% of training tokens come from the train splits of public benchmarks that the Decision Index also draws on, reformatted as decisions. The rest is broad decision data: general decision tasks — rules and policies, classification, entailment and fact checking, tables and documents, web and tool actions, judging (about 40%); math and step-by-step reasoning (about 20%); and hard decisions written and/or labelled by large frontier models, including robustness cases with perturbed or adversarial states (about 25%). Public datasets are used as train splits only. Contamination scan against the full Decision Index 0.3 public suite: 0 strict hits.
- **Limits.** Scores are self-run; confidence is not correctness, so set thresholds on your own data.
  Answers after a thought are sampled and can vary slightly; one pass is deterministic.
- **Licence.** Apache-2.0 for weights and code ([NOTICE](https://huggingface.co/org2ai/Wald-4B/blob/main/NOTICE)). Earlier versions: tags [`v1.2`](https://huggingface.co/org2ai/Wald-4B/tree/v1.2), [`v1.1`](https://huggingface.co/org2ai/Wald-4B/tree/v1.1), [`v1.0`](https://huggingface.co/org2ai/Wald-4B/tree/v1.0). Cite: [CITATION.cff](https://huggingface.co/org2ai/Wald-4B/blob/main/CITATION.cff).
