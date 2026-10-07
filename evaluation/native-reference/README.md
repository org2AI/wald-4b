# Native C16B inference reference

Exact source snapshots from the reader used for C16B evaluation. `source-pins.json` records every inference source SHA. No weights, training examples, benchmark payloads or credentials are included.

Use `evaluation/di03_native_auto_engine.py` with the pinned `eval/native_answer_acceptance_v1.py` path and this directory as `reference_root`. The adapter imports the native reader functions; the snapshot's historical standalone preparation/scoring CLI is not the v2 serving entry point. Its old run budget and old sample sizes do not configure the new public evaluation.

Runtime: Python 3.12, pydantic 2, Torch and the previously verified vLLM 0.30 server. Native chat template, readout, wide-option mapping and Auto gate must retain the pinned behavior. No extra LoRA is used.
