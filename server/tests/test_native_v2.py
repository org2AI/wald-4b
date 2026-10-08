"""Wald-4B v2 server: transport-trace parity with the evaluated Decision Index 0.3 adapter, the three fixed efforts,
both prompt formats and the error mapping. Synthetic inputs, no model, no GPU.

Run from the repository root:  pip install "./server[test]" && pytest -q server/tests/test_native_v2.py
(the adapter test also needs the Decision Index kit's `decision_index.engines.base`; it is skipped without it).
"""
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from wald_serve.native import NativeV2, handler

ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / 'temperature.json'
REF = ROOT / 'reference'
ADAPTER = ROOT / 'evaluation/v2/code/di03_native_auto_engine.py'


def fake(monkeypatch, top_equal=True):
    e = NativeV2('http://unused', TEMP)
    import eval.systemone_vllm as sov
    calls = []

    def post(self, path, body, retries=1):
        calls.append((path, json.loads(json.dumps(body))))
        if path == '/tokenize':
            text = body.get('prompt')
            if text is None:
                text = json.dumps(body['messages'], sort_keys=True)
                if body.get('chat_template_kwargs', {}).get('enable_thinking'):
                    text += '<think>\n'
            if len(text) > 4000:
                raise sov.Capacity('maximum context length')
            return {'tokens': [ord(c) for c in text]}
        if body['max_tokens'] == 1:
            ids = body['logprob_token_ids']
            top = {f'token_id:{tid}': (-1.0 if top_equal else -float(i)) for i, tid in enumerate(ids)}
            return {'choices': [{'logprobs': {'top_logprobs': [top]}}]}
        return {'choices': [{'text': 'synthetic reasoning', 'finish_reason': 'stop', 'stop_reason': '</think>'}],
                'usage': {'completion_tokens': 3}}
    monkeypatch.setattr(sov.Client, 'post', post)
    return e, calls


def generations(calls):
    return [b for p, b in calls if p == '/v1/completions' and b['max_tokens'] > 1]


QUESTIONS = [
    {'q': {'type': 'choice', 'instructions': 'choose', 'criteria': {'z': 'first', 'a': 'second'}}},
    {'q': {'type': 'noul', 'instructions': 'Is the statement true?'}},
    {'wide': {'type': 'choice', 'instructions': 'choose', 'criteria': {f'k{i}': str(i) for i in range(30)}}},
]


@pytest.mark.parametrize('questions', QUESTIONS)
def test_auto_equals_the_evaluated_di03_adapter(monkeypatch, questions):
    pytest.importorskip('decision_index.engines.base')
    if 'wide' in questions:
        pytest.importorskip('torch')   # the frozen wide-option path imports midtrain.letter -> torch (present with vLLM)
    e, calls = fake(monkeypatch)
    spec = importlib.util.spec_from_file_location('evaluated_wald_adapter', ADAPTER)
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    reference = adapter.WaldNativeAuto(REF / 'eval/native_answer_acceptance_v1.py', TEMP, e.info['temperature_sha256'], None,
                                       reference_root=REF, endpoint='http://unused')
    a, _ = reference('synthetic state', questions)
    for v in a['answers'].values():
        v.pop('raw', None)
    trace = list(calls); calls.clear()
    b = e({'state': 'synthetic state', 'questions': questions, 'effort': 'auto', 'model': 'ignored-wire-name'})
    assert a == b and trace == calls


@pytest.mark.parametrize('effort,thinks', [('none', False), ('auto', True), ('medium', True), ('always', True), ('high', True)])
def test_efforts(monkeypatch, effort, thinks):
    e, calls = fake(monkeypatch)
    r = e({'state': 's', 'questions': QUESTIONS[0], 'effort': effort})
    g = generations(calls)
    assert bool(g) == thinks and all(b['max_tokens'] == 512 for b in g)
    assert r['answers']['q']['mode'] == ('B' if thinks else 'A')


def test_confident_question_does_not_think_under_auto_but_does_under_always(monkeypatch):
    e, calls = fake(monkeypatch, top_equal=False)   # top probability ~0.73 > 0.7
    e({'state': 's', 'questions': QUESTIONS[0], 'effort': 'auto'})
    assert not generations(calls)
    calls.clear()
    e({'state': 's', 'questions': QUESTIONS[0], 'effort': 'always'})
    assert generations(calls)


def test_prompt_formats(monkeypatch):
    e, calls = fake(monkeypatch)
    e({'state': 'the state', 'questions': QUESTIONS[0], 'effort': 'none', 'prompt_format': 'plain'})
    plain = [b for p, b in calls if p == '/tokenize' and 'messages' in b][0]['messages'][-1]['content']
    calls.clear()
    e({'state': 'the state', 'questions': QUESTIONS[0], 'effort': 'none', 'prompt_format': 'repeat_state_plain'})
    rep = [b for p, b in calls if p == '/tokenize' and 'messages' in b][0]['messages'][-1]['content']
    assert plain.count('the state') == 1 and rep.count('the state') == 2


def test_wide_question_never_thinks(monkeypatch):
    pytest.importorskip('torch')
    e, calls = fake(monkeypatch)
    r = e({'state': 's', 'questions': QUESTIONS[2], 'effort': 'always'})
    assert not generations(calls) and len(r['answers']['wide']['probabilities']) == 30


def serve(e):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(e)); t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    return server, t


def post(server, body):
    req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/systemone', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        r = urllib.request.urlopen(req); return r.status, json.load(r)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def test_endpoint(monkeypatch):
    e, _ = fake(monkeypatch); server, t = serve(e)
    try:
        code, r = post(server, {'state': 's', 'questions': {'q': {'type': 'choice', 'criteria': {'a': 'a', 'b': 'b'}}}})
        assert code == 200 and set(r['answers']['q']['probabilities']) == {'a', 'b'} and 'raw' not in r['answers']['q']
        assert r['answers']['q']['mode'] == 'B'
        code, r = post(server, {'state': 's', 'questions': {}, 'effort': 'low'})
        assert code == 400 and 'effort' in r['error']
        code, r = post(server, {'state': 'x' * 5000, 'questions': {'q': {'type': 'choice', 'criteria': {'a': 'a', 'b': 'b'}}}})
        assert code == 422
        h = json.load(urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/health'))
        assert h['checkpoint'] == '04400-c18' and h['policy'] == 'Auto0.7' and h['thought_budget'] == 512
    finally:
        server.shutdown(); server.server_close(); t.join()


def test_calibration_mismatch_fails_closed(tmp_path):
    p = tmp_path / 'temperature.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='calibration'):
        NativeV2('http://unused', p)


def test_package_reference_equals_repository_reference():
    pkg = Path(__import__('wald_serve').__file__).with_name('native_reference')
    pins = json.loads((pkg / 'source-pins.json').read_text())['files_sha256']
    assert pins == json.loads((REF / 'source-pins.json').read_text())['files_sha256']
    for rel in pins:
        assert (pkg / rel).read_bytes() == (REF / rel).read_bytes()
