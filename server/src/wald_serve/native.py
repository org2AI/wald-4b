"""`wald-serve-native`: the Wald-4B v2 (checkpoint 04400-c18) `POST /v1/systemone` server.

v2 is read with the native Qwen chat template and native `<think>` reasoning. This module wraps the frozen reader
that produced v2's published scores (`native_reference/`, every file SHA-pinned in `source-pins.json`) and the frozen
calibration table (`temperature.json`, SHA-pinned below). It does not reimplement the reader.

Start vLLM on the weights (a loopback-only sidecar with the evaluated engine arguments) and serve:

    wald-serve-native --model /path/to/Wald-4B --port 8000

or attach to a vLLM server that is already running on the same weights:

    wald-serve-native --vllm http://127.0.0.1:8371 --served 04400-c18 --temperature temperature.json --port 8000

Effort (server default from serving.json, per request with "effort"):
    none            one pass: option-letter probabilities from one prefill, no generated tokens
    auto | medium   Auto 0.7 (default): think natively (at most 512 tokens) only when the one-pass top probability < 0.7
    always | high   Always 512: think on every question with 2-26 options

Prompt format (server default from serving.json, per request with "prompt_format"):
    repeat_state_plain   the state is written twice (Decision Index 0.3, JevBench-XL partitions)
    plain                the state once (JevBench public 231, multistep100)

Endpoints: POST /v1/systemone (also POST /), GET /health, GET /v1/models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__

TEMPERATURE_SHA = 'a0f72cd2d0a653e81051e5a0c77fc1a69131552a8102b580a93a6dbe7908b2da'
MODEL_SHA = '2859df99ed481592cdc3731fd768bf29ccd63273b2dab623642ad3b38b0e14c4'
CHECKPOINT = '04400-c18'
BUDGET = 512
# effort -> (policy name, gate threshold on the untempered one-pass distribution)
POLICIES = {'none': ('onepass', 0.0), 'auto': ('Auto0.7', 0.7), 'medium': ('Auto0.7', 0.7),
            'always': ('Always512', 1.01), 'high': ('Always512', 1.01)}
# prompt format -> the frozen reader's suite label (answer_task: 'di' -> repeat_state_plain, anything else -> plain)
FORMATS = {'repeat_state_plain': 'di', 'plain': 'xl'}
# the engine arguments of the evaluated runs (results: Decision Index 0.3 public and the JevBench / XL reads)
VLLM_ARGS = ['--max-model-len', '131072', '--gpu-memory-utilization', '0.85', '--max-num-seqs', '128', '--seed', '0']


class NativeV2:
    def __init__(self, endpoint, temperature, served=CHECKPOINT, effort='auto', prompt_format='repeat_state_plain',
                 max_model_len=131072, reference_root=None, checkpoint=CHECKPOINT, model_sha256=MODEL_SHA,
                 temperature_sha256=TEMPERATURE_SHA, llama_bridge=None):
        if effort not in POLICIES:
            raise ValueError(f'effort must be one of {sorted(POLICIES)}')
        if prompt_format not in FORMATS:
            raise ValueError(f'prompt_format must be one of {sorted(FORMATS)}')
        root = Path(reference_root or Path(__file__).with_name('native_reference'))
        pins = json.loads((root / 'source-pins.json').read_text())['files_sha256']
        for rel, digest in pins.items():
            if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
                raise ValueError('Native source hash mismatch: ' + rel)
        if hashlib.sha256(Path(temperature).read_bytes()).hexdigest() != temperature_sha256:
            raise ValueError('Frozen v2 calibration hash mismatch')
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location('wald_v2_frozen_native', root / 'eval/native_answer_acceptance_v1.py')
        self.native = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.native)
        if llama_bridge is not None:
            from eval import systemone_vllm as sov
            llama_bridge.install(sov)
        self.endpoint, self.served, self.temperature, self.max_len = endpoint, served, temperature, int(max_model_len)
        self.effort, self.prompt_format = effort, prompt_format
        self.root, self.pins = root, pins
        self.lock = threading.Lock(); self.idle = {f: [] for f in FORMATS}; self.table = None
        self.info = {'ok': True, 'model': served, 'checkpoint': checkpoint, 'model_sha256': model_sha256, 'version': __version__,
                     'engine': 'native-v2', 'effort': effort, 'policy': POLICIES[effort][0], 'gate': POLICIES[effort][1],
                     'thought_budget': BUDGET, 'prompt_format': prompt_format, 'max_model_len': self.max_len,
                     'temperature_sha256': temperature_sha256, 'native_reader_sha256': pins['eval/native_answer_acceptance_v1.py'],
                     'efforts': sorted(POLICIES), 'prompt_formats': sorted(FORMATS), 'serial_latency_measured': False,
                     'backend': 'llama.cpp' if llama_bridge is not None else 'vllm'}

    def _borrow(self, fmt):
        with self.lock:
            if self.idle[fmt]:
                return self.idle[fmt].pop()
        client, sov = self.native.make_client(self.endpoint, self.served, self.max_len, None)
        # Fail closed if another installation shadowed the frozen namespace.
        for name in ('eval.systemone_vllm', 'eval.vllm_letter', 'kev.api', 'midtrain.rawprompt'):
            module = sys.modules.get(name)
            if module is not None:
                rel = name.replace('.', '/') + '.py'
                if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != self.pins[rel]:
                    raise RuntimeError('Frozen native import was shadowed: ' + name)
        with self.lock:
            if self.table is None:
                self.table = sov.load_tables(self.temperature).get('A')
        return client, sov

    def __call__(self, request):
        if not isinstance(request, dict) or not {'state', 'questions'} <= request.keys():
            raise ValueError('Expected state and questions')
        effort = request.get('effort') or self.effort
        fmt = request.get('prompt_format') or self.prompt_format
        if effort not in POLICIES:
            raise ValueError(f'effort must be one of {sorted(POLICIES)} for Wald-4B v2')
        if fmt not in FORMATS:
            raise ValueError(f'prompt_format must be one of {sorted(FORMATS)}')
        # Only the inference fields reach the reader (the evaluated adapters sent exactly these two).
        req = {'state': request['state'], 'questions': request['questions']}
        client, sov = self._borrow(fmt)
        try:
            response = self.native.answer_task(client, sov, {'request': req, 'suite': FORMATS[fmt]}, self.table,
                                               POLICIES[effort][1], BUDGET)
        finally:
            with self.lock:
                self.idle[fmt].append((client, sov))
        # Thought text and internal probes are not part of the public serving wire.
        for value in response['answers'].values():
            value.pop('raw', None)
        return response


def _capacity_class(root):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from eval.systemone_vllm import Capacity
    return Capacity


def handler(engine):
    capacity = _capacity_class(engine.root)

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def send(self, code, value):
            raw = json.dumps(value).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_GET(self):
            if self.path.startswith('/v1/models'):
                return self.send(200, {'object': 'list', 'data': [{'id': engine.served, 'object': 'model'}]})
            if self.path not in ('/', '/health'):
                return self.send(404, {'error': 'Unknown route'})
            self.send(200, engine.info)

        def do_POST(self):
            if self.path not in ('/', '/v1/systemone'):
                return self.send(404, {'error': 'Unknown route'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError('Invalid request size')
                response = engine(json.loads(self.rfile.read(size)))
            except capacity as e:
                # A declared capacity limit (prompt longer than the context, too many options): the Decision Index
                # kit's http engine maps 422 to `unsupported`; nothing is truncated.
                return self.send(422, {'error': str(e)[:400]})
            except (ValueError, KeyError, TypeError) as e:
                return self.send(400, {'error': f'{type(e).__name__}: {e}'[:400]})
            except Exception as e:  # noqa: BLE001
                return self.send(502, {'error': f'Native inference backend failed: {type(e).__name__}'})
            self.send(200, response)
    return H


def read_serving(model_dir):
    p = Path(model_dir) / 'serving.json' if model_dir else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='wald-serve-native', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--model', help='weights directory: start vLLM on it (serving.json / temperature.json read from it)')
    src.add_argument('--vllm', help='URL of a running vLLM server on the same weights')
    src.add_argument('--gguf', help='GGUF weights file: start llama-server on it (needs --tokenizer-dir)')
    src.add_argument('--llamacpp', help='URL of a running llama-server on a v2 GGUF (needs --tokenizer-dir)')
    ap.add_argument('--tokenizer-dir', default=None, help='folder with tokenizer.json, tokenizer_config.json, chat_template.jinja, serving.json, temperature.json (GGUF mode)')
    ap.add_argument('--llama-server', default='llama-server')
    ap.add_argument('--llama-parallel', type=int, default=4, help='llama-server slots; context = max-model-len x slots')
    ap.add_argument('--llama-args', default='')
    ap.add_argument('--served', default=None, help='vLLM served model name (also the response "model" field); default: the checkpoint id')
    ap.add_argument('--temperature', default=None, help='calibration table (default: <model>/temperature.json)')
    ap.add_argument('--effort', default=None, choices=sorted(POLICIES), help='default: serving.json, else auto')
    ap.add_argument('--prompt-format', default=None, choices=sorted(FORMATS), help='default: serving.json, else repeat_state_plain')
    ap.add_argument('--max-model-len', type=int, default=131072)
    ap.add_argument('--vllm-port', type=int, default=8371)
    ap.add_argument('--gpu-memory-utilization', type=float, default=None,
                    help='override the evaluated 0.85 (a deviation from the evaluated engine arguments)')
    ap.add_argument('--vllm-args', default='', help='extra vLLM arguments (a deviation from the evaluated engine arguments)')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args(argv)
    cfg = read_serving(a.model or a.tokenizer_dir)
    checkpoint = cfg.get('checkpoint', CHECKPOINT)   # serving.json identifies the release checkpoint; defaults = 04400-c18
    served = a.served or checkpoint
    effort = a.effort or cfg.get('effort', 'auto')
    fmt = a.prompt_format or cfg.get('prompt_format', 'repeat_state_plain')
    home = a.model or a.tokenizer_dir
    temps = a.temperature or (str(Path(home) / cfg.get('temperature', 'temperature.json')) if home else None)
    if not temps or not Path(temps).is_file():
        raise SystemExit(f'temperature table not found: {temps}')
    if a.model:
        import shlex
        from .server import launch
        args = list(VLLM_ARGS)
        if a.max_model_len != 131072:
            args[1] = str(a.max_model_len)
        if a.gpu_memory_utilization is not None:
            args[3] = str(a.gpu_memory_utilization)
        os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', a.model, '--served-model-name', served,
               '--host', '127.0.0.1', '--port', str(a.vllm_port), *args, *shlex.split(a.vllm_args)]
        launch(cmd, a.vllm_port, 'vLLM')
        endpoint = f'http://127.0.0.1:{a.vllm_port}'
    elif a.gguf or a.llamacpp:
        if not a.tokenizer_dir:
            raise SystemExit('--tokenizer-dir is required with --gguf / --llamacpp')
        if a.gguf:
            from .server import launch_llama
            launch_llama(a.llama_server, a.gguf, a.vllm_port, a.max_model_len, a.llama_parallel, a.llama_args)
            endpoint = f'http://127.0.0.1:{a.vllm_port}'
        else:
            endpoint = a.llamacpp
    else:
        endpoint = a.vllm
    bridge = None
    if a.gguf or a.llamacpp:
        from .native_llamacpp import LlamaBridge
        bridge = LlamaBridge(endpoint, a.tokenizer_dir)
    engine = NativeV2(endpoint, temps, served, effort, fmt, a.max_model_len, llama_bridge=bridge, checkpoint=checkpoint,
                      model_sha256=cfg.get('model_sha256', MODEL_SHA), temperature_sha256=cfg.get('temperature_sha256', TEMPERATURE_SHA))
    ThreadingHTTPServer.daemon_threads = True
    httpd = ThreadingHTTPServer((a.host, a.port), handler(engine))
    print(json.dumps({'serving': engine.info, 'port': a.port}), flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    sys.exit(main())
