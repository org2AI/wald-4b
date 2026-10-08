"""Native Qwen thought experiment; no paid launch or power operations in this file.

Run against an already registered vLLM server. Preparation and audit are CPU-only.
All intermediate distributions are untempered; gate 0.7 precedes calibration.
"""
import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import threading
import time

THOUGHT_SYSTEM = ('You are a calibration engine inside a decision system. You are given a state, a question and lettered options. '
                  'Think it through briefly first (check every condition, date and number that matters), then reply with one capital letter, no words, no punctuation.')
VERSION = 'native-chat-gate/1'
CONTROL_TOKENS = ('<think>', '</think>', '<|im_start|>', '<|im_end|>', '<|endoftext|>')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def remap(p, order):
    """Displayed position -> canonical option position, with a complete permutation."""
    if len(p) != len(order) or sorted(order) != list(range(len(order))):
        raise ValueError('option map must be a complete permutation')
    out = [0.0] * len(p)
    for value, canonical in zip(p, order):
        out[canonical] = value
    return out


def average_orders(reads):
    if not reads:
        raise ValueError('no order reads')
    mapped = [remap(p, order) for p, order in reads]
    n = len(mapped[0])
    if any(len(p) != n for p in mapped):
        raise ValueError('order reads differ in option count')
    return [sum(p[i] for p in mapped) / len(mapped) for i in range(n)]


def gate(p, threshold=0.7):
    if not p or any(not math.isfinite(x) or x < 0 for x in p) or not math.isclose(sum(p), 1.0, abs_tol=1e-6):
        raise ValueError('invalid option distribution')
    return len(p) > 1 and max(p) < threshold


def prepare(args):
    """Bind exact historical suites; keep private payloads outside the repository."""
    di = rows(args.di_suite)
    xl = [r for r in rows(args.xl_source) if r['set'] == 'xl']
    if len(di) != 6948 or len(xl) != 2691:
        raise ValueError('expected original DI6948 and XL2691 suites')
    ids = [r['_evaluation']['run_id'] for r in di]
    if len(set(ids)) != len(ids) or len({r['id'] for r in xl}) != len(xl):
        raise ValueError('duplicate suite identities')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for r in di:
        # Metadata/expected are retained for CPU kit scoring, never sent to the model.
        tasks.append({'id': r['_evaluation']['run_id'], 'suite': 'di', 'request': {'state': r['state'], 'questions': r['questions']}})
    for r in xl:
        tasks.append({'id': r['id'], 'suite': 'xl', 'request': r['request'], 'gold_key': r['keys'][r['gold']]})
    tasks.sort(key=lambda r: hashlib.sha256(('native-c16-1005:' + r['suite'] + ':' + r['id']).encode()).hexdigest())
    path = out / 'requests.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in tasks))
    receipt = {'status': 'prepared_not_run', 'version': VERSION, 'di_requests': len(di), 'xl_items': len(xl),
               'requests_sha256': sha(path), 'di_suite_sha256': sha(args.di_suite), 'xl_source_sha256': sha(args.xl_source),
               'temperature_sha256': sha(args.temperature), 'native_thinking_measured': False,
               'gate': 0.7, 'budget': 512, 'think_k': 1, 'order': 'original_only', 'wide': 'historical_knockout_onepass',
               'di_prompt_format': 'repeat_state_plain', 'xl_prompt_format': 'plain',
               'calibration': 'fixed historical A table for every policy; native B has no fitted table; additionally report T=1',
               'selection': 'no new selection; original registered historical DI/XL diagnostic replay'}
    dump(out / 'prepared.json', receipt)
    if args.receipt:
        dump(args.receipt, receipt)
    print(json.dumps(receipt))


def native_read(cl, sov, state, q, threshold, budget, seed_text, usage, raw):
    n = len(q['options'])
    if n > 26:
        p, mode, pa = sov.read_question(cl, state, q, 0, budget, seed_text, usage, raw=raw)
        raw['native'] = {'status': 'wide_onepass', 'generated_tokens': 0}
        return p, mode, pa
    ida = sov.prompt_ids(cl, state, q)
    p, mass = cl.readout(ida, n)
    usage['input_tokens'] += len(ida)
    raw.update(A=p, A_mass=mass)
    if not gate(p, threshold):
        raw['native'] = {'status': 'not_gated', 'generated_tokens': 0}
        return p, 'A', None
    from eval.vllm_letter import chat_messages
    pre = cl.post('/tokenize', {'model': cl.served,
        'messages': chat_messages(sov.chat_user(q['chat'], cl.prompt_format), system=THOUGHT_SYSTEM),
        'add_generation_prompt': True, 'add_special_tokens': False,
        'chat_template_kwargs': {'enable_thinking': True}})['tokens']
    # The native Qwen template must actually open thought, not merely accept a flag.
    suffix = cl.ids('<think>\n')
    if not suffix or pre[-len(suffix):] != suffix:
        raise ValueError('native thinking prefix does not end in <think>\\n')
    close = cl.ids('</think>') + cl.ids('\n\n')
    if len(pre) + budget + len(close) + 1 > cl.max_len:
        raw['native'] = {'status': 'capacity_onepass', 'generated_tokens': 0}
        return p, 'A', None
    t0 = time.perf_counter()
    response = cl.post('/v1/completions', {'model': cl.served, 'prompt': pre, 'max_tokens': budget,
        'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'seed': sov.hash_seed(seed_text),
        'stop': ['</think>'], 'include_stop_str_in_output': False})
    c = response['choices'][0]
    text = (c.get('text') or '').strip()
    ntok = response.get('usage', {}).get('completion_tokens')
    if not isinstance(ntok, int) or not 0 <= ntok <= budget:
        raise ValueError('missing or out-of-budget completion usage')
    usage['output_tokens'] += ntok
    # Final typed answers are scored by the benchmark adapter. Generated control
    # text alone is not an official answer-error condition; retain it privately.
    ids = pre + cl.ids(text + '\n') + close
    pb, bmass = cl.readout(ids, n)
    usage['input_tokens'] += len(pre) + len(ids)
    raw.update(B=[pb], B_mass=bmass, native={'status': 'thought_read', 'text': text,
        'generated_tokens': ntok, 'naturally_closed': c.get('finish_reason') == 'stop' and c.get('stop_reason') == '</think>',
        'finish_reason': c.get('finish_reason'), 'stop_reason': c.get('stop_reason'),
        'truncated': c.get('finish_reason') == 'length', 'generation_seconds': time.perf_counter() - t0,
        'prefix_sha256': hashlib.sha256(json.dumps(pre).encode()).hexdigest()})
    return pb, 'B', p


def make_client(endpoint, served, max_len, deadline=None):
    from eval import systemone_vllm as sov
    class StrictClient(sov.Client):
        def __init__(self, *args, **kwargs):
            self.deadline = deadline
            super().__init__(*args, **kwargs)

        def post(self, path, body, retries=1):
            if self.deadline is not None:
                remaining = self.deadline - time.monotonic() - 120
                if remaining <= 0: raise TimeoutError('registered evaluation window ended')
                self.timeout = min(180, remaining)
            return super().post(path, body, retries=1)

        def readout(self, ids, n):
            if len(ids) + 1 > self.max_len:
                raise sov.Capacity('native prompt exceeds context')
            want = [t for letters in self.letter_ids[:n] for t in letters]
            r = self.post('/v1/completions', {'model': self.served, 'prompt': ids, 'max_tokens': 1,
                'temperature': 0.0, 'logprobs': 20, 'logprob_token_ids': want, 'return_tokens_as_token_ids': True})
            from eval.vllm_letter import letter_readout
            p, _, mass = letter_readout(r['choices'][0]['logprobs']['top_logprobs'][0],
                dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', self.letter_ids)), n)
            return p, mass
    cl = StrictClient(endpoint, served, max_len, timeout=180)
    cl.template = 'chat'; cl.wide = 'knockout'; cl.topk = 0; cl.wide_residual = 0.001
    return cl, sov


def answer_task(cl, sov, task, table, threshold, budget):
    from kev.api import SystemOneRequest, to_record
    req = task['request']; rec, meta = to_record(SystemOneRequest.model_validate(req))
    chat = sov.chat_parts(req, rec, meta)
    # Client is per worker/task so DI/XL prompt formatting cannot race.
    cl.prompt_format = 'repeat_state_plain' if task['suite'] == 'di' else 'plain'
    answers = {}; usage = {'input_tokens': 0, 'output_tokens': 0}
    for rq, m in zip(rec['questions'], meta):
        q = {'id': m['id'], 'type': rq['qtype'], 'instructions': rq['instr'], 'options': rq['options'],
             'option_texts': rq['options'], 'chat': chat[m['id']]}
        raw = {}; p, mode, pa = native_read(cl, sov, rec['state'], q, threshold, budget,
            json.dumps(req, sort_keys=True) + m['id'], usage, raw)
        # Fixed historical A table, no pretending a paren B table calibrates native B.
        calibrated = sov.temper(p, sov.temperature(table, m['type'], len(p)))
        keys = m['keys']; j = max(range(len(p)), key=p.__getitem__)
        value = {'type': m['type'], 'mode': mode, 'raw': {**raw, 'keys': keys},
                 'probabilities': dict(zip(keys, calibrated)), 'probabilities_t1': dict(zip(keys, p))}
        if m['type'] == 'noul': value['noul'] = calibrated[keys.index('true')]
        elif m['type'] == 'score': value['score'] = sum(i * v for i, v in enumerate(calibrated))
        else: value['choice'] = keys[j]
        value['confidence'] = max(calibrated)
        answers[m['id']] = value
    return {'model': cl.served, 'answers': answers, 'usage': usage}


def run(args):
    inputs = Path(args.inputs); prepared = json.loads((inputs / 'prepared.json').read_text())
    if sha(inputs / 'requests.jsonl') != prepared['requests_sha256'] or sha(args.temperature) != prepared['temperature_sha256']:
        raise ValueError('frozen input/calibration SHA mismatch')
    if args.wall_seconds > 14400 or args.wall_seconds <= 0:
        raise ValueError('evaluation window must be within 1..14400 seconds under registered USD2 outer guard')
    tasks = rows(inputs / 'requests.jsonl'); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'reads.jsonl').exists():
        raise ValueError('refuse overwrite or ambiguous resume')
    # Read-only inference endpoint has no per-request API billing. Provider box
    # power is registered/guarded by the operator before this process starts.
    table = json.loads(Path(args.temperature).read_text())
    if 'A' in table: table = table['A']
    # The GPU paid window began before checkpoint restoration, not at the first
    # inference. Honour its absolute deadline and reserve four minutes for evidence
    # transfer/shutdown before the independent provider guard.
    budget_path = Path(__file__).with_name('budget.json')
    if budget_path.exists():
        paid = json.loads(budget_path.read_text())
        args.wall_seconds = min(args.wall_seconds, int(paid['gpu_deadline_utc_ts'] - time.time() - 240))
        if args.wall_seconds <= 180: raise TimeoutError('restoration consumed evaluation window')
    tls = threading.local(); start = time.monotonic(); deadline = start + args.wall_seconds
    completed = 0; errors = 0; generated = 0; stopped = threading.Event()
    dump(out / 'run.json', {'status': 'evaluating', 'version': VERSION, 'inputs': prepared,
        'endpoint': args.endpoint, 'served': args.served, 'started_sgt': dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        'wall_seconds': args.wall_seconds, 'cap_usd': 2.0, 'no_per_request_api_charge': True,
        'operator_power_guard_required': True, 'source_sha256': sha(__file__)})
    def one(task):
        if stopped.is_set() or time.monotonic() > deadline - 180:
            stopped.set(); return None
        if not hasattr(tls, 'client'):
            tls.client, tls.sov = make_client(args.endpoint, args.served, args.max_model_len, deadline)
        t0 = time.perf_counter()
        try:
            response = answer_task(tls.client, tls.sov, task, table, 0.7, 512)
            return {'id': task['id'], 'suite': task['suite'], 'status': 'ok', 'response': response,
                    'gold_key': task.get('gold_key'), 'wall_s': time.perf_counter() - t0}
        except Exception as e:
            return {'id': task['id'], 'suite': task['suite'], 'status': 'error', 'error': str(e)[:400]}
    # A bounded queue prevents thousands of queued tasks continuing after a stop.
    with (out / 'reads.jsonl').open('w') as f, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(tasks); pending = {}
        for _ in range(args.workers):
            task = next(iterator, None)
            if task: pending[pool.submit(one, task)] = task
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
            if time.monotonic() > deadline - 180: stopped.set()
            for future in done:
                pending.pop(future); r = future.result()
                if r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush()
                    completed += 1; errors += r['status'] != 'ok'
                    generated += r.get('response', {}).get('usage', {}).get('output_tokens', 0)
                elapsed = time.monotonic() - start
                # Projection starts only after 256 completed tasks; task order is
                # frozen and early rates are diagnostic, never a success guarantee.
                projected = elapsed * len(tasks) / max(completed, 1)
                if completed >= 256 and projected > args.wall_seconds - 180: stopped.set()
                dump(out / 'progress.json', {'completed': completed, 'total': len(tasks), 'errors': errors,
                    'generated_tokens': generated, 'elapsed_s': elapsed, 'projected_s': projected, 'stop_requested': stopped.is_set()})
                if not stopped.is_set():
                    task = next(iterator, None)
                    if task: pending[pool.submit(one, task)] = task
    dump(out / 'DONE.json', {'status': 'completed' if completed == len(tasks) and not errors else 'stopped_or_failed',
        'completed': completed, 'total': len(tasks), 'errors': errors, 'generated_tokens': generated,
        'wall_s': time.monotonic() - start, 'reads_sha256': sha(out / 'reads.jsonl'), 'gpu_shutdown_required_now': True})


def aggregate(args):
    """CPU-only native diagnostics and DI kit envelope preparation, never DI rescoring."""
    read = rows(args.reads); suite = rows(args.di_suite)
    if sha(args.di_suite) != 'be10d45f2cfbc4872aa32a5f1d9b0c5f75a53cceee204463d68198108c689846':
        raise ValueError('historical DI suite SHA mismatch')
    byid = {r['_evaluation']['run_id']: r for r in suite}
    seen = set(); di = []; xl = []; native = []; errors = []
    for r in read:
        key = (r['suite'], r['id'])
        if key in seen: raise ValueError('duplicate read identity')
        seen.add(key)
        if r['status'] != 'ok': errors.append(key); continue
        for answer in r['response']['answers'].values():
            native.append(answer['raw']['native'])
        if r['suite'] == 'di':
            if r['id'] not in byid: raise ValueError('DI read outside frozen suite')
            meta = byid[r['id']]['_evaluation']
            # Preserve the historical frozen kit envelope rather than inventing a
            # custom index; omitted/failed identities require explicit partial label.
            di.append({**meta, 'run_id': r['id'], 'status': 'ok', 'engine': 'http',
                'total_wall_ms': 1000*r['wall_s'], 'response': r['response']})
        elif r['suite'] == 'xl':
            if len(r['response']['answers']) != 1: raise ValueError('XL item must have one question')
            answer = next(iter(r['response']['answers'].values())); raw = answer['raw']; keys = raw['keys']
            pa = raw['A']; pb = list(answer['probabilities_t1'].values()); gold = keys.index(r['gold_key'])
            ca = max(range(len(pa)), key=pa.__getitem__) == gold
            cb = max(range(len(pb)), key=pb.__getitem__) == gold
            xl.append((ca, cb))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'di-results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in di))
    thought = [r for r in native if r['status'] == 'thought_read']
    result = {'status': 'completed' if len(di) == 6948 and len(xl) == 2691 and not errors else 'partial_or_failed',
        'reads_sha256': sha(args.reads), 'di_requests': len(di), 'xl_items': len(xl), 'errors': len(errors),
        'native_questions': len(thought), 'nonempty_thoughts': sum(bool(r['text'].strip()) for r in thought),
        'naturally_closed': sum(r['naturally_closed'] for r in thought), 'truncated': sum(r['truncated'] for r in thought),
        'generated_tokens': sum(r['generated_tokens'] for r in thought),
        'wide_onepass_questions': sum(r['status'] == 'wide_onepass' for r in native),
        'xl_A_correct': sum(a for a,b in xl), 'xl_gate_correct': sum(b for a,b in xl),
        'xl_wrong_to_right': sum(not a and b for a,b in xl), 'xl_correct_to_wrong': sum(a and not b for a,b in xl),
        'order_average': 'not_run; remapping CPU-validated only',
        'interpretation': 'A versus native B also changes system instruction and template branch; useful-thought causal effect needs matched empty-thought control',
        'di_index': 'requires original frozen decision_index kit score plus boot edition0.2.1 B300, after GPU shutdown',
        'native_calibration': 'unfitted; T1 and historical A-table distributions preserved; no TEST fitting'}
    dump(out / 'aggregate.json', result); print(json.dumps(result))


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--di-suite', required=True); p.add_argument('--xl-source', required=True)
    p.add_argument('--temperature', required=True); p.add_argument('--out', required=True); p.add_argument('--receipt')
    r = sub.add_parser('run'); r.add_argument('--inputs', required=True); r.add_argument('--temperature', required=True)
    r.add_argument('--endpoint', required=True); r.add_argument('--served', required=True); r.add_argument('--out', required=True)
    r.add_argument('--wall-seconds', type=int, default=3600); r.add_argument('--max-model-len', type=int, default=131072)
    r.add_argument('--workers', type=int, default=16)
    g = sub.add_parser('aggregate'); g.add_argument('--reads', required=True); g.add_argument('--di-suite', required=True); g.add_argument('--out', required=True)
    args = ap.parse_args(); {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.command](args)


if __name__ == '__main__': main()
