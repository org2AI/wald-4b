# Runbook: Wald-Q4B v2 (`v2.0`, checkpoint `04701-c22`)

This revision holds **Wald-Q4B v2**: the language model of checkpoint `04701-c22` (its text-only weights file has sha256 `6ae382f5a0ed9f4c53cd0953cb9606b1627a6816350340a59114860e6cf6f29a`; every tensor is byte-identical here) plus Qwen3.5-4B's unchanged vision tower, MTP head and image processor configs, as `Qwen3_5ForConditionalGeneration` in three safetensors shards. MANIFEST.json lists the sha256 of every file. v1.x runbooks are at their tags.

## 1. Serve

```sh
hf download org2ai/Wald-4B --revision v2.0 --local-dir ./Wald-Q4B-v2
cd Wald-Q4B-v2
python - <<'PY'   # optional: check every file against MANIFEST.json
import hashlib, json; m = json.load(open("MANIFEST.json"))
bad = [f for f, v in m.items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != v["sha256"]]; print("bad:", bad)
PY
./run.sh "$PWD"
```

`GET /health` must report `"engine": "native-v2"`, `"vision": true`, `"checkpoint": "04701-c22"`, `"policy": "Auto0.7"`, `"thought_budget": 512` and `"temperature_sha256": "a0f72cd2d0a653e81051e5a0c77fc1a69131552a8102b580a93a6dbe7908b2da"`. Docker: see `Dockerfile` (same runtime). The server pins the reader (`server/src/wald_serve/native_reference/`, every file sha256-checked at start) and the calibration table, and refuses to start if either differs.

**The evaluated runtime:** vLLM 0.30.0, transformers 5.17.0, torch 2.13.0, BF16, one NVIDIA RTX PRO 6000 (96 GB) per model replica, `VLLM_USE_FLASHINFER_SAMPLER=0`, and the vLLM arguments `--max-model-len 131072 --gpu-memory-utilization 0.85 --max-num-seqs 128 --seed 0` plus `--limit-mm-per-prompt {"image": 16, "video": 0}` and the image budget `--mm-processor-kwargs {"size": {"shortest_edge": 65536, "longest_edge": 1048576}}` (what `wald-serve-native-vision --model` launches; `wald-serve-native` serves text only). Smaller GPUs need a lower `--gpu-memory-utilization` or `--max-model-len`, which is a deviation from the evaluated setup. Thought generation samples (temperature 0.6, top-p 0.95, top-k 20) with a seed derived from the request and question id, so repeated requests normally give the same answer; batch composition on the GPU can still change near-tied answers.

`config.json` carries `"use_cache": false` from training. vLLM ignores it; if you load the weights with Transformers `generate`, pass `use_cache=True`.

## 2. Policies and prompt formats

| Request field | Values | Evaluated settings |
|---|---|---|
| `effort` | `none` (one pass), `auto` / `medium` (Auto 0.7, default), `always` / `high` (Always 512) | all three |
| `prompt_format` | `repeat_state_plain` (default), `plain` | Decision Index and JevBench-XL: `repeat_state_plain`; JevBench public 231 and multistep: `plain` |

Auto 0.7 gates on the untempered one-pass distribution: a question with 2–26 options thinks natively (Qwen chat template with `enable_thinking`, at most 512 tokens, stop at `</think>`) when its top probability is below 0.7, then the options are read again. More than 26 options: grouped knockout readout, no thought, every option gets a probability. The returned probabilities are tempered with the frozen table (bucket by question type and option count).

## 3. Images

Requests with images (message content parts `image_url` / `image` in `state`, raw base64 under `state` keys `image`, `image_data`, `image_base64`, `image_b64`, or a top-level `images` / `image_data` list; up to 16 per request, bodies up to 64 MB) go through the same frozen reader, with the images carried to vLLM's chat route at their position in the state. They default to the `plain` prompt format. Text-only requests take the text path unchanged. The Decision Index 0.3 run used the text-only weights file; the language-model tensors here are byte-identical, and the text parity of this repository's weights is in `evaluation/v2/serving-parity.json`.

## 4. GGUF (llama.cpp)

`org2ai/Wald-4B-GGUF` holds the v2 quantisations and the vision projector (`Wald-4B-v2-mmproj-F16.gguf`). `wald-serve-native --gguf FILE --tokenizer-dir DIR --effort none` reads text requests through llama-server with the same chat layout; a letter outside llama-server's top-100 log-probabilities gets the reader's floor instead of its exact value. One pass is the checked mode.

## 5. Decision Index 0.3 public suite

Install the official kit `apolinario/decision-index` at `62d2f51de34a2de64906345b6bc3e98e27ff55c7` (branch `v0.3`), rebuild the public suite and verify its hashes with the kit. The evaluated run used the kit's Engine API with the adapter in `evaluation/v2/code/di03_native_auto_engine.py` (it loads `reference/` and `temperature.json`, Auto 0.7, budget 512, `repeat_state_plain`) against vLLM servers started with the arguments above; `evaluation/v2/code/di03_parallel.py` ran four disjoint replicas (SHA256(run_id) mod 4) at 128 concurrent requests each. The packaged server answers the same requests with the same reader; the kit's `http` engine can be pointed at `POST /v1/systemone` instead. Score with the kit's unmodified scorer after inference. Suite payloads are not redistributed. Concurrent throughput is not the maintainers' serial latency.

## 6. Other reads

JevBench public 231, JevBench-XL partitions and multistep_decisions were read with the same reader and engine (one generation per item; the one-pass, Auto 0.7 and Always 512 answers are fixed views of it) and scored with each benchmark's scorer. JevBench-XL is internal and not distributed.

Training data and evaluation caveats: PROVENANCE.md, CONTAMINATION.md.
