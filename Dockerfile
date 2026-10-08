# Wald-4B v2 /v1/systemone server: vLLM 0.30.0 (bf16) on the weights as a loopback-only sidecar + wald-serve-native.
#
#   docker build -t wald-serve .
#   docker run --gpus all -v /path/to/Wald-4B:/model:ro -p 8000:8000 wald-serve
#   # ready when GET http://127.0.0.1:8000/health returns {"ok": true, ...}
#
# The weights directory holds config, tokenizer, chat template, model.safetensors, temperature.json and serving.json
# (native v2 engine, effort, prompt format). Another policy: -e WALD_EFFORT=none (none | auto | always).
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.9.5 \
 && uv pip install --system --no-cache "vllm==0.30.0" "transformers==5.17.0" \
 && python -c "import importlib.metadata as m; print('vllm', m.version('vllm'), 'transformers', m.version('transformers'), 'torch', m.version('torch'))"
COPY server /app/server
RUN uv pip install --system --no-cache /app/server
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 WALD_EFFORT=""
EXPOSE 8000
CMD ["sh", "-c", "exec wald-serve-native-vision --model /model --port 8000 ${WALD_EFFORT:+--effort $WALD_EFFORT}"]
