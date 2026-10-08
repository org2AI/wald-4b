"""llama.cpp backend for the Wald-4B v2 native reader (GGUF weights).

The frozen reader talks to vLLM's /tokenize and /v1/completions. This bridge answers the same calls with
llama.cpp's llama-server: chat-template rendering and tokenization happen locally with the release's own tokenizer
files (tokenizer.json + chat_template.jinja, byte-identical to the BF16 release), the option-letter read uses
llama-server's top-100 next-token log-probabilities (a letter outside them gets the reader's floor, as in v1.x's
llama.cpp path), and thoughts use /completion with the same sampling settings and seed.
"""
from __future__ import annotations

import json
import urllib.request

N_PROBS = 100


class LlamaBridge:
    def __init__(self, url, tokenizer_dir):
        from transformers import AutoTokenizer
        self.url = url.rstrip('/')
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    def _post(self, path, body, timeout=900):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def tokenize(self, body):
        if 'messages' in body:
            text = self.tok.apply_chat_template(body['messages'], tokenize=False, add_generation_prompt=body.get('add_generation_prompt', True),
                                                **(body.get('chat_template_kwargs') or {}))
            return {'tokens': self.tok.encode(text, add_special_tokens=False)}
        return {'tokens': self.tok.encode(body['prompt'], add_special_tokens=bool(body.get('add_special_tokens')))}

    def completions(self, body):
        if body.get('max_tokens') == 1:
            r = self._post('/completion', {'prompt': body['prompt'], 'n_predict': 1, 'temperature': 0.0, 'n_probs': N_PROBS,
                                           'post_sampling_probs': False, 'cache_prompt': True, 'samplers': ['top_k'], 'top_k': 1})
            top = {int(t['id']): float(t['logprob']) for t in r['completion_probabilities'][0]['top_logprobs']}
            # llama-server reports only the top N_PROBS tokens; a wanted letter outside them gets the floor
            # min - 2 (the v1.x llama.cpp rule) instead of vLLM's exact log-probability.
            floor = min(top.values()) - 2.0
            for tid in body.get('logprob_token_ids') or []:
                top.setdefault(int(tid), floor)
            return {'choices': [{'logprobs': {'top_logprobs': [{f'token_id:{tid}': lp for tid, lp in top.items()}]}}]}
        r = self._post('/completion', {'prompt': body['prompt'], 'n_predict': body['max_tokens'], 'temperature': body.get('temperature', 0.6),
                                       'top_p': body.get('top_p', 0.95), 'top_k': body.get('top_k', 20), 'min_p': 0.0,
                                       'seed': body.get('seed', 0), 'stop': body.get('stop') or [], 'cache_prompt': True})
        stopped = r.get('stop_type') == 'word'
        return {'choices': [{'text': r.get('content') or '', 'finish_reason': 'stop' if r.get('stop_type') in ('word', 'eos') else 'length',
                             'stop_reason': r.get('stopping_word') if stopped else None}],
                'usage': {'completion_tokens': int(r.get('tokens_predicted') or 0)}}

    def install(self, sov):
        """Route the frozen reader's HTTP calls through this bridge (one backend per server process)."""
        bridge = self

        def post(self, path, body, retries=1):
            if path == '/tokenize':
                return bridge.tokenize(body)
            if path == '/v1/completions':
                return bridge.completions(body)
            raise ValueError('unsupported path for the llama.cpp bridge: ' + path)
        sov.Client.post = post
