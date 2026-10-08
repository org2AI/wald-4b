# Wald-Q4B v2 evaluation

Checkpoint `04701-c22`. All numbers are self-run; none is a leaderboard result.

## Scores

| Benchmark | One pass | Auto 0.7 | Jev 1.13 (hosted) |
|---|---:|---:|---:|
| Decision Index 0.3, complete public suite | not run | **54.18** (our full run) | 57.96 (board public column) |
| JevBench public set (231) | 204/231 | **210/231** | 200/231 |
| JevBench-XL TEST (8,177) | 65.20 % | **65.78 %** | 67.85 % |

- **Decision Index 0.3:** ours is a full run of the complete public suite with the official kit (140,620 requests); Jev's number is the board's maintainer-run public column. Neither is the board's Full score, which adds private tests. Per-family scores: [summary.json](summary.json).
- **JevBench public (231):** ours from our evaluation harness, Jev's from the JevBench CLI, same items.
- **JevBench-XL TEST (8,177):** our internal test set; Jev was read through its hosted API on the same items.

## Latency

Serial processing time per request (one in flight, warm), 750 Decision Index 0.3 requests, NVIDIA RTX PRO 6000 Blackwell (96 GB), the packaged server:

| Mode | Median | Mean |
|---|---:|---:|
| One pass | 42 ms | 110 ms |
| Auto 0.7 | 148 ms | 478 ms |

Auto 0.7 thought on 33.1 % of requests. Measured on the release graft without its unused MTP head (same language-model and vision tensors; the MTP head only acts under speculative decoding, which is off). Raw file: [latency.json](latency.json).

## Images (zero-shot)

| Benchmark | Items | Qwen3.5-4B chat | Wald-Q4B v2 |
|---|---:|---:|---:|
| CV-Bench (board subset) | 2,038 | 82.43 % | **85.48 %** |
| BLINK (3 subsets, val) | 387 | 78.55 % | **82.43 %** |
| RealWorldQA | 589 | 73.68 % | **75.04 %** |
| All three | 3,014 | 80.23 % | **83.05 %** (+2.82 pp [+1.69, +3.98]) |

Our harness and our conversions of the public items (one pass, image budget 65,536–1,048,576 px); not Decision Index Vision board numbers. The board runs its own vision suites through this server. Raw file: [vision-sanity.json](vision-sanity.json).

## GGUF (llama.cpp b11312, one pass)

| File | Size | Same answer as BF16, JevBench public | Correct / 231 (BF16 in this harness: 205) | Same answer, DI 0.2.1 subset (5,922 q) | Accuracy Δ |
|---|---:|---:|---:|---:|---:|
| `Wald-4B-v2-Q8_0.gguf` | 4.5 GB | 229/231 | 203 | 5,889/5,922 | +0.07 pp |
| `Wald-4B-v2-Q6_K.gguf` | 3.5 GB | 230/231 | 204 | not run | — |
| `Wald-4B-v2-Q5_K_M.gguf` | 3.1 GB | 223/231 | 203 | not run | — |
| `Wald-4B-v2-Q4_K_M.gguf` | 2.7 GB | 224/231 | 202 | 5,714/5,922 | -0.44 pp |

Raw file: [gguf-parity.json](gguf-parity.json). Serving parity of the packaged server against the evaluation runs: [serving-parity.json](serving-parity.json).
