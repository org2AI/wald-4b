"""An untuned model served by vLLM, read as a one-prefill letter classifier (docs/gpt-oss.md): `eval.run --arm
rawlogit:<hf id> --endpoint <vLLM base URL> --prompt-template plain|chat|harmony`.

Same question rendering and readout as eval.predictors.RawLogitPredictor (state, instructions and option strings from
raw_prompts; option letters A-Z; z_L = log(P("L") + P(" L")) over the FULL next-token distribution, p = softmax over the
present options' z, T = 1; `letter_mass` = the share of the distribution on those letters), but the forward pass runs
in a vLLM server (any architecture vLLM serves: gpt-oss MXFP4, Gemma 4, Qwen3.5 hybrids, FP8 MoEs) and the prompt can be
a chat one:

  plain    RawLogitPredictor's text byte for byte (`... Answer:`), special tokens added as its tokenizer adds them.
  chat     the model's own chat template (vLLM /tokenize with messages, add_generation_prompt, `chat_template_kwargs`
           {"enable_thinking": false} by default): a system message (SYSTEM) and a user message (user_message).
  harmony  gpt-oss's harmony format, built here: system (Reasoning: low), developer (# Instructions + SYSTEM), user, and
           the prompt ends exactly at the assistant's final-channel message start
           (`<|start|>assistant<|channel|>final<|message|>`), so the next token is the answer, with no analysis channel.

One /v1/completions call per record (a prompt per question, as token ids, max_tokens 1, temperature 0) with
`logprob_token_ids` = the option letters' token ids (vLLM >= 0.30: the log-probabilities of exactly those ids under the
raw, unmasked distribution; `return_tokens_as_token_ids` keys them `token_id:<id>`). No decoding and no reasoning
tokens. A prompt longer than the server's max_model_len, or more than 26 options, is a kev ContextOverflow (a rejected
record), as RawLogitPredictor's context check is.

`bench` measures serving on the same server: per-decision latency at concurrency 1 and throughput at 8-32 concurrent
requests, over a sample of the questions this run scored, each with a fresh nonce line in front of its state so no
prompt prefix beyond the fixed system part can come from vLLM's prefix cache.
"""
import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from midtrain.rawprompt import LETTERS, plain

PROMPT_VERSION = "vllm-letter/1"
TEMPLATES = ("plain", "chat", "harmony")
SYSTEM = ("You are a calibration engine inside a decision system. You are given a state, a question and lettered options. "
          "Choose exactly one option and reply with its letter only: one capital letter, no words, no punctuation, no explanation.")
HARMONY_DATE = "2026-09-25"      # fixed, so a prompt is the same bytes on every run
HARMONY_REASONING = "low"
DEFAULT_CHAT_KWARGS = {"enable_thinking": False}


def user_message(state, instructions, options):
    """The user turn of the chat / harmony prompts: the state (left out when blank), the question, the lettered options
    (option strings as raw_options renders them: label names shown, `key: text`), and the answer instruction."""
    if len(options) > len(LETTERS):
        from kev.model import ContextOverflow
        raise ContextOverflow(f"{len(options)} options: the letter readout has {len(LETTERS)} letters")
    head = f"State:\n{plain(state)}\n\n" if str(state).strip() else ""
    lines = [f"{LETTERS[j]}. {plain(o)}" for j, o in enumerate(options)]
    return head + f"Question: {plain(instructions)}\nOptions:\n" + "\n".join(lines) + "\n\nAnswer with the letter of one option."


def harmony_prompt(user, system=SYSTEM, reasoning=HARMONY_REASONING, date=HARMONY_DATE):
    """gpt-oss harmony text ending at the final channel's message start (the answer's first token comes next)."""
    sys_msg = ("You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n"
               f"Current date: {date}\n\nReasoning: {reasoning}\n\n"
               "# Valid channels: analysis, commentary, final. Channel must be included for every message.")
    return (f"<|start|>system<|message|>{sys_msg}<|end|>"
            f"<|start|>developer<|message|># Instructions\n\n{system}<|end|>"
            f"<|start|>user<|message|>{user}<|end|>"
            "<|start|>assistant<|channel|>final<|message|>")


def question_parts(record):
    """Kev-format record -> (rendered state, [(question id, instructions, option strings)]), as raw_prompts renders them
    (the predictor's questions; midtrain.chat_lora trains on exactly these)."""
    from .predictors import kev_to_t2m, raw_options
    rec, qids = kev_to_t2m(record)
    return rec["state"], [(qid, q["instructions"], raw_options(record["questions"][qid], q["option_texts"])) for qid, q in zip(qids, rec["questions"])]


def chat_messages(user, system=SYSTEM, system_role=True):
    """[system, user] messages; without a system role the instruction leads the user turn."""
    if system_role:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return [{"role": "user", "content": f"{system}\n\n{user}"}]


def letter_readout(top, letter_ids, n):
    """`top` = one position's {"token_id:<id>": logprob} (vLLM, return_tokens_as_token_ids) -> (p, z, mass) over the
    first n letters: z_L = logsumexp of the letter's token ids' logprobs (a missing or -9999 id adds nothing), p =
    softmax(z), mass = sum exp(z). A letter with no finite logprob is an error (nothing to read)."""
    lp = {}
    for k, v in top.items():
        tid = int(str(k).split(":", 1)[1]) if str(k).startswith("token_id:") else None
        if tid is not None and v is not None and v > -9999.0:
            lp[tid] = float(v)
    z = []
    for j in range(n):
        vals = [lp[t] for t in letter_ids[LETTERS[j]] if t in lp]
        if not vals:
            raise ValueError(f"no logprob for letter {LETTERS[j]} (ids {letter_ids[LETTERS[j]]}) in {sorted(lp)[:12]}")
        m = max(vals)
        z.append(m + math.log(sum(math.exp(v - m) for v in vals)))
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [x / s for x in e], z, sum(math.exp(v) for v in z)


class VllmLetterPredictor:
    """A vLLM OpenAI-compatible server (`endpoint` = its base URL, `served` = its served model name) read as a letter
    classifier (module docstring). `model_id` / `revision` are recorded only (the server loaded the weights)."""

    temperature = 1.0
    prompt_version = PROMPT_VERSION

    def __init__(self, endpoint, served, template="chat", model_id=None, revision=None, chat_kwargs=None, system_role=True, timeout=600, retries=3):
        if template not in TEMPLATES: raise ValueError(f"template {template!r} not in {TEMPLATES}")
        self.endpoint, self.served, self.template = endpoint.rstrip("/"), served, template
        self.model_id, self.revision, self.commit = model_id, revision, revision
        self.chat_kwargs = DEFAULT_CHAT_KWARGS if chat_kwargs is None else dict(chat_kwargs)
        self.system_role, self.timeout, self.retries = system_role, timeout, retries
        self.dtype = "served"
        self.lock = threading.Lock()
        self.mass = {"n": 0, "sum": 0.0, "min": 1.0}
        self.seen = []          # (record, question index) samples for bench: every question scored, reservoir of 4096
        self.n_seen = 0
        self.rng = random.Random(0)
        self.example = None
        self.letters = {L: self._letter(L) for L in LETTERS}
        self.self_test_result = self.self_test()

    # --- HTTP ---------------------------------------------------------------------------------------------------------
    def _post(self, path, body):
        req = urllib.request.Request(self.endpoint + path, data=json.dumps(body).encode(), method="POST", headers={"content-type": "application/json"})
        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode("utf-8", "replace")
                if e.code in (400, 413, 422):
                    from kev.model import ContextOverflow
                    raise ContextOverflow(f"vLLM refused ({e.code}): {detail}") from None
                last = RuntimeError(f"vLLM {path} {e.code}: {detail}")
            except (urllib.error.URLError, OSError) as e:
                last = e
            time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.endpoint}{path} failed after {self.retries} attempts: {last}")

    def _letter(self, L):
        ids = []
        for v in (L, " " + L):
            got = self._post("/tokenize", {"model": self.served, "prompt": v, "add_special_tokens": False})["tokens"]
            if len(got) == 1 and got[0] not in ids:
                ids.append(got[0])
        if not ids:
            raise ValueError(f"no single-token encoding of {L!r} or {' ' + L!r}")
        return ids

    # --- prompts ------------------------------------------------------------------------------------------------------
    def prompt_ids(self, state, instructions, options):
        """(token ids, max_model_len) of one question's prompt under this template."""
        if self.template == "plain":
            from midtrain.rawprompt import raw_prompt
            from kev.model import ContextOverflow
            if len(options) > len(LETTERS): raise ContextOverflow(f"{len(options)} options: the letter readout has {len(LETTERS)} letters")
            body = {"model": self.served, "prompt": raw_prompt(state, instructions, options), "add_special_tokens": True}
        elif self.template == "harmony":
            body = {"model": self.served, "prompt": harmony_prompt(user_message(state, instructions, options)), "add_special_tokens": False}
        else:
            body = {"model": self.served, "messages": chat_messages(user_message(state, instructions, options), system_role=self.system_role),
                    "add_generation_prompt": True, "add_special_tokens": False, "chat_template_kwargs": self.chat_kwargs}
        r = self._post("/tokenize", body)
        return r["tokens"], r.get("max_model_len")

    def _questions(self, record):
        return question_parts(record)

    def self_test(self):
        """One tiny question through /tokenize and /v1/completions: a template or API refusal here is a hard error, not
        a stream of rejected records."""
        from kev.model import ContextOverflow
        try:
            ids, _ = self.prompt_ids("light: red", "Should the car stop?", ["no", "yes"])
            (p, z, mass), = self.score([(ids, 2)])[0]
        except ContextOverflow as e:
            raise RuntimeError(f"self-test refused by the server: {e}") from None
        return {"p": p, "letter_mass": mass, "prompt_tokens": len(ids)}

    def score(self, prompts):
        """[(ids, n options)] -> [(p, z, mass)] in one /v1/completions call; (answers, wall ms, input tokens)."""
        wanted = sorted({t for _, n in prompts for j in range(n) for t in self.letters[LETTERS[j]]})
        body = {"model": self.served, "prompt": [ids for ids, _ in prompts], "max_tokens": 1, "temperature": 0.0, "logprobs": 1,
                "logprob_token_ids": wanted, "return_tokens_as_token_ids": True}
        t0 = time.perf_counter()
        r = self._post("/v1/completions", body)
        ms = 1000 * (time.perf_counter() - t0)
        choices = sorted(r["choices"], key=lambda c: c["index"])
        out = [letter_readout(c["logprobs"]["top_logprobs"][0], self.letters, n) for c, (_, n) in zip(choices, prompts)]
        return out, ms, (r.get("usage") or {}).get("prompt_tokens")

    # --- predictor protocol (eval.run / kev.benchmark / mock_server) -------------------------------------------------
    def __call__(self, record, pause_tokens=None):
        from kev.api import question_keys
        from kev.model import ContextOverflow
        state, qs = self._questions(record)
        built = []
        for qid, instr, opts in qs:
            ids, limit = self.prompt_ids(state, instr, opts)
            if limit and len(ids) >= limit:
                raise ContextOverflow(f"prompt of {len(ids)} tokens: the server's max_model_len is {limit}")
            built.append((ids, len(opts)))
        if self.example is None:
            self.example = self._detok(built[0][0])
        answers, ms, n_in = self.score(built)
        probs, logits, mass = {}, {}, {}
        for (qid, _, _), (p, z, m) in zip(qs, answers):
            keys = question_keys(record["questions"][qid]["type"], record["questions"][qid].get("criteria"))
            probs[qid], logits[qid], mass[qid] = dict(zip(keys, p)), dict(zip(keys, z)), m
        with self.lock:
            for m in mass.values():
                self.mass["n"] += 1; self.mass["sum"] += m; self.mass["min"] = min(self.mass["min"], m)
            for j in range(len(qs)):
                self.n_seen += 1
                if len(self.seen) < 4096: self.seen.append((record, j))
                elif (k := self.rng.randrange(self.n_seen)) < 4096: self.seen[k] = (record, j)
        return {"probabilities": probs, "logits": logits, "inference_temperature": 1.0, "latency_ms": ms, "input_tokens": n_in, "letter_mass": mass}

    def _detok(self, ids):
        try:
            return self._post("/detokenize", {"model": self.served, "tokens": ids})["prompt"]
        except Exception as e:   # a diagnostic only
            return f"(detokenize failed: {e})"

    def mass_summary(self):
        m = self.mass
        return {"questions": m["n"], "mean": m["sum"] / m["n"] if m["n"] else None, "min": m["min"] if m["n"] else None}

    def info(self):
        ex = self.example or ""
        return {"kind": "rawlogit", "backend": "vllm", "endpoint": self.endpoint, "served_model": self.served, "model": self.model_id, "revision": self.revision,
                "commit": self.commit, "dtype": self.dtype, "temperature": 1.0, "prompt_version": PROMPT_VERSION, "template": self.template,
                "chat_template_kwargs": self.chat_kwargs if self.template == "chat" else None, "system_role": self.system_role if self.template == "chat" else None,
                "system": SYSTEM if self.template != "plain" else None, "letter_tokens": {L: self.letters[L] for L in LETTERS[:4]},
                "self_test": self.self_test_result, "prompt_example_head": ex[:700], "prompt_example_tail": ex[-500:]}

    # --- serving measurement ------------------------------------------------------------------------------------------
    def bench(self, n=256, concurrency=(1, 8, 16, 32), n_single=64, seed=0):
        """Latency / throughput on this server over `n` questions sampled from those scored (module docstring). Each
        prompt gets a fresh nonce line in front of its state (plain template: at the very start). Per concurrency c:
        requests (n_single at c = 1, n otherwise), wall, requests per minute, per-request latency p50 / p95 / mean (one
        question per request, tokenization excluded: ids are built first), input tokens mean."""
        rng = random.Random(seed)
        pool = list(self.seen)
        rng.shuffle(pool)
        pool = pool[:n]
        if not pool:
            return None

        def build(rec_j, tag):
            rec, j = rec_j
            state, qs = self._questions(rec)
            _, instr, opts = qs[j]
            nonce = f"[request {tag}-{rng.getrandbits(48):012x}]"
            ids, _ = self.prompt_ids(f"{nonce}\n{state}" if str(state).strip() else nonce, instr, opts)
            return ids, len(opts)

        rows = {}
        q = lambda xs, f: sorted(xs)[min(len(xs) - 1, int(f * len(xs)))]
        for c in concurrency:
            k = n_single if c == 1 else len(pool)
            prompts = [build(pool[i % len(pool)], f"c{c}i{i}") for i in range(k)]
            lat = []

            def one(p):
                _, ms, _ = self.score([p])
                lat.append(ms)
            self.score([prompts[0]])   # warm the path (not timed)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(c) as ex:
                list(ex.map(one, prompts))
            wall = time.perf_counter() - t0
            rows[str(c)] = {"requests": k, "wall_s": wall, "requests_per_min": 60 * k / wall, "latency_ms_p50": q(lat, 0.5), "latency_ms_p95": q(lat, 0.95),
                            "latency_ms_mean": sum(lat) / len(lat), "input_tokens_mean": sum(len(p[0]) for p in prompts) / len(prompts)}
            print(f"bench c={c}: {k} req in {wall:.1f}s = {rows[str(c)]['requests_per_min']:.0f}/min, p50 {rows[str(c)]['latency_ms_p50']:.0f} ms, "
                  f"p95 {rows[str(c)]['latency_ms_p95']:.0f} ms", flush=True)
        return {"sample": len(pool), "from_questions": self.n_seen, "one_question_per_request": True, "nonce": True, "concurrency": rows}
