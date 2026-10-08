"""`wald-serve-native-vision`: the Wald-4B v2 vision variant's `POST /v1/systemone` server.

Same as `wald-serve-native` (native.py, unchanged) for every request without images: the frozen native-v2 reader, the
frozen calibration table, the evaluated vLLM arguments. Requests with images go through vision_v2 (same reader, images
carried to vLLM's chat route at their position in the state). Differences from native.py, all for images:
  * vLLM is started on the Qwen3_5ForConditionalGeneration weights with `--limit-mm-per-prompt {"image": 16, "video": 0}`
    and the evaluated image budget `--mm-processor-kwargs {"size": {"shortest_edge": 65536, "longest_edge": 1048576}}`
    (serving.json `vision.vllm_args`);
  * `--trust-request-chat-template`: the image route passes its verbatim template per request (vLLM is a loopback-only
    sidecar of this server, so no outside caller can send a template);
  * request bodies up to 64 MB (images are base64);
  * image requests default to `prompt_format` `plain` (serving.json `vision.image_prompt_format`), text requests keep the
    server default.

    wald-serve-native-vision --model /path/to/Wald-4B-vision --port 8000
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import native, vision_v2

VISION_VLLM_ARGS = ["--limit-mm-per-prompt", '{"image": 16, "video": 0}',
                    "--mm-processor-kwargs", '{"size": {"shortest_edge": 65536, "longest_edge": 1048576}}',
                    "--trust-request-chat-template"]   # the image route sends its verbatim template; vLLM listens on loopback only
MAX_BODY = 64 * 1024 * 1024


class NativeV2Vision(native.NativeV2):
    def __init__(self, *args, image_prompt_format="plain", max_images=vision_v2.MAX_IMAGES, **kw):
        super().__init__(*args, **kw)
        if image_prompt_format not in native.FORMATS:
            raise ValueError(f"image_prompt_format must be one of {sorted(native.FORMATS)}")
        self.image_prompt_format, self.max_images = image_prompt_format, max_images
        self.vidle = {f: [] for f in native.FORMATS}
        self.info.update(vision=True, image_prompt_format=image_prompt_format, max_images=max_images,
                         image_transports=["state data URIs / messages image parts", "state raw base64 keys", "top-level images[]"])

    def _vborrow(self, fmt):
        with self.lock:
            if self.vidle[fmt]:
                return self.vidle[fmt].pop()
        client, sov = self._borrow(fmt)          # a frozen client (shadow checks, table load), re-classed for images
        client.__class__ = vision_v2.image_client(type(client))
        return client, sov

    def __call__(self, request):
        if not (isinstance(request, dict) and vision_v2.has_images(request)):
            return super().__call__(request)
        if not {"state", "questions"} <= request.keys():
            raise ValueError("Expected state and questions")
        effort = request.get("effort") or self.effort
        fmt = request.get("prompt_format") or self.image_prompt_format
        if effort not in native.POLICIES:
            raise ValueError(f"effort must be one of {sorted(native.POLICIES)} for Wald-4B v2")
        if fmt not in native.FORMATS:
            raise ValueError(f"prompt_format must be one of {sorted(native.FORMATS)}")
        req, uris, pat = vision_v2.lift(request, self.max_images)
        client, sov = self._vborrow(fmt)
        client.bind(uris, pat)
        try:
            response = self.native.answer_task(client, sov, {"request": req, "suite": native.FORMATS[fmt]}, self.table,
                                               native.POLICIES[effort][1], native.BUDGET)
            if client.mm_calls == 0:
                raise RuntimeError("image request was answered without the image route")
        finally:
            client.bind([], None); client.mm_calls = 0
            with self.lock:
                self.vidle[fmt].append((client, sov))
        for value in response["answers"].values():
            value.pop("raw", None)
        response["images"] = len(uris)
        return response


def handler(engine):
    capacity = native._capacity_class(engine.root)
    base = native.handler(engine)

    class H(base):
        def do_POST(self):
            if self.path not in ("/", "/v1/systemone"):
                return self.send(404, {"error": "Unknown route"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Invalid request size")
                response = engine(json.loads(self.rfile.read(size)))
            except capacity as e:
                return self.send(422, {"error": str(e)[:400]})
            except (ValueError, KeyError, TypeError) as e:
                return self.send(400, {"error": f"{type(e).__name__}: {e}"[:400]})
            except Exception as e:  # noqa: BLE001
                return self.send(502, {"error": f"Native inference backend failed: {type(e).__name__}"})
            self.send(200, response)
    return H


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="wald-serve-native-vision", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model"); src.add_argument("--vllm")
    ap.add_argument("--served", default=None); ap.add_argument("--temperature", default=None)
    ap.add_argument("--effort", default=None, choices=sorted(native.POLICIES))
    ap.add_argument("--prompt-format", default=None, choices=sorted(native.FORMATS))
    ap.add_argument("--image-prompt-format", default=None, choices=sorted(native.FORMATS))
    ap.add_argument("--max-model-len", type=int, default=131072); ap.add_argument("--vllm-port", type=int, default=8371)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None); ap.add_argument("--vllm-args", default="")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    cfg = native.read_serving(a.model)
    vcfg = cfg.get("vision") or {}
    checkpoint = cfg.get("checkpoint", native.CHECKPOINT); served = a.served or checkpoint
    effort = a.effort or cfg.get("effort", "auto"); fmt = a.prompt_format or cfg.get("prompt_format", "repeat_state_plain")
    ifmt = a.image_prompt_format or vcfg.get("image_prompt_format", "plain")
    temps = a.temperature or (str(Path(a.model) / cfg.get("temperature", "temperature.json")) if a.model else None)
    if not temps or not Path(temps).is_file():
        raise SystemExit(f"temperature table not found: {temps}")
    if a.model:
        from .server import launch
        args = list(native.VLLM_ARGS)
        if a.max_model_len != 131072: args[1] = str(a.max_model_len)
        if a.gpu_memory_utilization is not None: args[3] = str(a.gpu_memory_utilization)
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", a.model, "--served-model-name", served,
               "--host", "127.0.0.1", "--port", str(a.vllm_port), *args, *vcfg.get("vllm_args", VISION_VLLM_ARGS), *shlex.split(a.vllm_args)]
        launch(cmd, a.vllm_port, "vLLM")
        endpoint = f"http://127.0.0.1:{a.vllm_port}"
    else:
        endpoint = a.vllm
    engine = NativeV2Vision(endpoint, temps, served, effort, fmt, a.max_model_len, checkpoint=checkpoint,
                            model_sha256=cfg.get("model_sha256", native.MODEL_SHA),
                            temperature_sha256=cfg.get("temperature_sha256", native.TEMPERATURE_SHA), image_prompt_format=ifmt)
    ThreadingHTTPServer.daemon_threads = True
    httpd = ThreadingHTTPServer((a.host, a.port), handler(engine))
    print(json.dumps({"serving": engine.info, "port": a.port}), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
