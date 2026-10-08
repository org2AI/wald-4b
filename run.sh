#!/usr/bin/env bash
# One command, no Docker: a Python 3.12 environment with the evaluated runtime (vLLM 0.30.0, transformers 5.17.0) and
# wald-serve, then serve the weights on 0.0.0.0:${PORT:-8000} (POST /v1/systemone, GET /health). Linux + CUDA + `uv`.
#
#   ./run.sh /path/to/Wald-4B                 # serving.json: native v2 engine, effort auto (Auto 0.7), repeat_state_plain
#   EFFORT=none ./run.sh /path/to/Wald-4B     # none | auto | always
set -euo pipefail
MODEL=${1:?usage: run.sh <weights dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$HERE/.venv}
[ -x "$VENV/bin/python" ] || { uv venv -p 3.12 "$VENV" && VIRTUAL_ENV="$VENV" uv pip install "vllm==0.30.0" "transformers==5.17.0" "$HERE/server"; }
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
exec "$VENV/bin/wald-serve-native-vision" --model "$MODEL" --port "${PORT:-8000}" ${EFFORT:+--effort "$EFFORT"} ${PROMPT_FORMAT:+--prompt-format "$PROMPT_FORMAT"}
