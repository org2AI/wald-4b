"""A concurrent POST /v1/systemone front-end over a vLLM server, for our Base-lineage letter models (015A0-f3 one-pass,
015D0-f4 gated think) in TypeSafe's wire format (the Decision Index kit's `http` engine and JevBench both read it).

Per question the bytes are eval.think_paren's (kev.api.to_record -> midtrain.letter.question_prompt, paren layout):

    State:\\n{state}\\n\\nQuestion: {instructions}\\n(A) {option 1}\\n(B) {option 2}\\nAnswer: (      <- one-pass read

the letters A..Z read at the next position as z_L = log(P("L") + P(" L")), softmax over the question's options. With
--gate g > 0 a question whose one-pass max p < g is re-read after a short thought (think_paren's think layout:
`Reasoning: {thought}\\nAnswer: (`, ≤ --budget tokens, temperature 0.6 / top-p 0.95 / top-k 20, seed = hash of the
request and question id, as think_paren serve); --gate 0.5 / --budget 256 is eval.effort's `low`, 0.7 / 512 its `medium`. Answers are tempered with the --temperature bucket table (the `A` / `B` tables of a gated
table, or one plain table for both).

Differences from `think_paren serve` (which stays as it is):
  - no global lock: requests are answered concurrently (vLLM batches them); the client is stateless stdlib urllib;
  - answers carry the TypeSafe wire keys (`type`, `choice` + `probabilities` + `confidence`, or `noul`), as
    kev.api.to_answers writes them, plus `mode` (A / B / K / S / T);
  - all 26 letters are read (think_paren's client reads A..P, so 17-26-option questions came back short);
  - more than 26 options ("wide" questions, e.g. BANKING77's 77 intents): --wide knockout reads the options in
    ceil(n / 26) near-equal consecutive chunks with the same prompt, then one final question over the chunk winners;
    P(option) = P_final(its chunk) x P_chunk(option); one-pass only (no thought). --wide refuse answers 422
    "at most 26 options per choice" instead. --wide chunk3 keeps the top three options from each original 26-sized
    chunk (stable ties to the original lower index), then compares all finalists in one read. Its final-read
    probabilities carry `1 - --wide-residual` total mass (default residual 0.001); pruned options share the residual
    in proportion to their normalized within-chunk probabilities (mode `S`). Residual 0 hard-prunes for diagnostics.
    This is a shortlist distribution, distinct from knockout's product distribution. If the finalist read would exceed 26,
    it fails with Capacity rather than silently dropping candidates. --topk K (> 0, <= 26) adds a third round to
    knockout only: the K options with the highest
    knockout probability, in their original order, are re-read as one K-way question and the top K's knockout mass is
    spread by that read (mode `T`; the other options keep their knockout probability). Default 0 = plain knockout;
  - a prompt longer than --max-model-len answers 422 with "maximum context length" (the kit records it as unsupported;
    nothing is truncated); a thought that would not fit falls back to the one-pass read;
  - --think-k k (test-time scaling): a question that thinks samples k thoughts and answers with the mean of the k
    post-thought letter distributions (before tempering). Thought 0 is the k = 1 thought (same seed), thoughts 1..k-1
    come from one n = k-1 request seeded from the same text + "#k"; so k = 1 answers exactly as before;
  - --return-raw adds `raw` to every answer: the untempered one-pass distribution (`A`), each thought's untempered
    post-thought distribution (`B`, in sample order) and the mode, so one always-think read (--gate 1.01) can be
    re-scored offline as one-pass, any gate, k = 1 or the mean of the first j thoughts (trainer/eval/scripts/di_policies.py);
  - --prompt-format (default `plain`: the bytes above, unchanged) applies one general rewrite to every question's prompt,
    never keyed to a benchmark; adapted from featherless-ai/simple-jev @ dae340e (Apache-2.0,
    hf-server/hf_prompt_policies.py; the STRICT / FULL_EXAMPLES / STATE_REPEAT / INPUT_REPEAT strings are copied
    verbatim). Our base-lineage models read a plain letter layout, not a chat template, so the policies' system text
    becomes a preamble before `State:` and there is no `[thinking]` prefill / JSON answer / nine-bin Noul:
      repeat_state          STRICT + FULL_EXAMPLES preamble; the state written twice, joined by STATE_REPEAT
      repeat_state_plain    the state written twice (STATE_REPEAT), no preamble (the repetition alone); also with
                            --template chat, inside the user turn (chat_user)
      strict_mix_repeat2    STRICT preamble; the whole input (state + question + options) twice, joined by INPUT_REPEAT

    python -m eval.systemone_vllm --vllm http://127.0.0.1:8011 --served 015A0-f3 --gate 0 --port 8100 \\
        --temperature results/v2/015A0-f3-cal/temperature.json

Latent readout (off by default; docs/plans/latent-reasoning-1002.md): `--latent-model DIR [--latent-adapter DIR]
--latent-file latent_proj.safetensors --latent-k 6 [--latent-mode normal|shuffle|const|zero]` answers every letter read
from an in-process Hugging Face reader (eval.latent_paren) instead of --vllm: the same prompts, letter ids, knockout,
prompt formats and tempering, but each read is taken after K fed-back hidden states (`... Reasoning:` + K latent
positions + `\\nAnswer: (`) of a midtrain.latent_lora model; --latent-k 0 is the plain one-pass read. Serial (one read
at a time on the card), gate 0 only (nothing is generated), paren layout only. Latent reads are answered as mode `A`,
so --temperature must be a table fitted on latent reads (the one-pass table does not transfer).

Needs trainer/kev and trainer/midtrain importable (kev.api, midtrain.letter).
"""
import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CUE = "Answer: ("
THINK = "Reasoning:"
CLOSE = "\nAnswer: ("
STOP = "\nAnswer"
BUCKETS = ((2, "2"), (4, "3-4"), (8, "5-8"), (math.inf, "9+"))   # t2m_kev.temperature.BUCKETS


class Capacity(ValueError):
    """A declared capacity limit: answered 422 with a marker the kit's http engine maps to `unsupported`."""


# --- temperature (serve_gated's table format) ---------------------------------------------------------------------------
def bucket_key(qtype, n):
    return f"{qtype}|" + next(name for upper, name in BUCKETS if n <= upper)


def temperature(table, qtype, n):
    if not table: return 1.0
    b = (table.get("buckets") or {}).get(bucket_key(qtype, n))
    return float(b["temperature"]) if b else float(table.get("single", 1.0))


def temper(p, t):
    if t == 1.0: return list(p)
    z = [math.log(max(x, 1e-12)) / t for x in p]
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
    return [v / s for v in e]


def load_tables(path, budget=512):
    """-> {"A": table, "B": table, "K": table}; a plain bucket table serves every established mode; a gated table's think entry is
    B<budget> when fitted for this budget, else B."""
    if not path: return {}
    t = json.loads(Path(path).read_text())
    if "buckets" in t or "single" in t: return {"A": t, "B": t, "K": t}
    return {"A": t.get("A"), "B": t.get(f"B{budget}") or t.get("B"), "K": t.get("A")}


# --- simple-jev prompt formats (Apache-2.0, featherless-ai/simple-jev @ dae340e, hf-server/hf_prompt_policies.py) ------
SJ_STRICT = 'Use the question and rubric as the decision rule. For truth or probability, assess the exact proposition, including its conditions: relevance is not truth, lack of mention is not falsity, and a plausible inference is not an explicit fact. For numerical probability, count or derive the favorable outcomes and divide by the total; evaluate the requested event rather than its complement. For ordered levels, choose the most specific level justified by the evidence, without escalating beyond what its definition requires.'
SJ_FULL_EXAMPLES = 'Independent worked examples below are not facts about the actual state. Transfer only the reasoning rules, not their entities, numbers, conclusions or answers.\nA general rule requires P and Q. A more specific applicable exception requires only P for certified repairs. The case is a certified repair; P is satisfied and Q is not. The exception replaces the general prerequisite list, so the action is permitted.\nA 72-hour window starts March 3 at 06:00 and ends March 6 at 06:00. An event exactly at the endpoint satisfies "by the deadline" but not "before the deadline". Adding a full extra calendar day would be incorrect.\nA policy requires a supervisor when total exposure is at most 50 and a director above 50. Existing exposure is 45 and the new commitment is 10, so total exposure is 55 and the director is required. Testing only the new commitment would apply the wrong quantity.\nItem K maps to unit B; unit B maps to zone 4; zone 4 maps to contact P. An override substitutes contact Q only on holidays. Today is not a holiday. Follow all mappings, then test the override condition: the operative contact is P.\nCause X has prior probability 0.2 and triggers a signal with probability 0.8. Cause Y has prior probability 0.8 and triggers it with probability 0.1. Given the signal, the probability of X is (0.2*0.8)/(0.2*0.8+0.8*0.1)=2/3, not 0.8 and not 0.2.\nLevel 0 means no evidence, level 1 requires condition P, and level 2 requires P and Q. If P holds but Q does not, the matching level is 1. Select the defined category rather than averaging the two nearby categories.\nA requested function must return the maximum of any nonempty numeric list, including negative values. A proposed implementation starts best=0 and only replaces best when a value is larger. It works on positive examples but returns 0 for [-5,-2], whose correct maximum is -2. The implementation does not meet the full requirement.\nFor the actual task, use its own evidence, definitions, exceptions and output labels. Return only the required answer; do not reproduce these explanations.'
SJ_STATE_REPEAT = '\n\nRead the same context again before answering. This is a repeated copy, not additional events or independent evidence:\n'
SJ_INPUT_REPEAT = '\n\nRead the same input again before answering:\n'
PROMPT_FORMATS = ("plain", "repeat_state", "repeat_state_plain", "strict_mix_repeat2")


def format_prompt(prompt, fmt):
    """One question's plain letter prompt (`State:\\n{state}\\n\\nQuestion: ... Answer: (`) rewritten by --prompt-format."""
    if fmt == "plain": return prompt
    if not prompt.endswith(CUE): raise ValueError("prompt does not end with the answer cue")
    if fmt in ("repeat_state", "repeat_state_plain"):
        pre = SJ_STRICT + "\n\n" + SJ_FULL_EXAMPLES + "\n\n" if fmt == "repeat_state" else ""
        if prompt.startswith("State:\n") and "\n\nQuestion:" in prompt:
            i = prompt.index("\n\nQuestion:")
            state = prompt[len("State:\n"):i]
            return pre + "State:\n" + state + SJ_STATE_REPEAT + state + prompt[i:]
        return pre + prompt   # blank state: nothing to repeat
    if fmt == "strict_mix_repeat2":
        body = prompt[: -len(CUE)].rstrip("\n")
        return SJ_STRICT + "\n\n" + body + SJ_INPUT_REPEAT + body + "\n" + CUE
    raise ValueError(f"unknown prompt format {fmt!r}")


def hash_seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


# --- vLLM client (stdlib, thread-safe: no shared mutable state after __init__) --------------------------------------------
class Client:
    def __init__(self, endpoint, served, max_len, timeout=900):
        self.url, self.served, self.max_len, self.timeout = endpoint.rstrip("/"), served, max_len, timeout
        self.letter_ids = []
        paren = self.ids("(")
        for L in LETTERS:
            ids = []
            for v in (L, " " + L):
                t = self.ids(v)
                if len(t) == 1 and t[0] not in ids: ids.append(t[0])
            # A SentencePiece tokenizer with a dummy prefix encodes a lone "A" as "▁A", but after "Answer: (" the model
            # emits the bare piece: take the letter as it is encoded after "(" too (the same id for BPE tokenizers).
            t = self.ids("(" + L)
            if len(t) == len(paren) + 1 and t[:-1] == paren and t[-1] not in ids: ids.append(t[-1])
            if not ids: raise RuntimeError(f"no single-token encoding of letter {L!r}")
            self.letter_ids.append(ids)
        self.close = self.ids(CLOSE)

    def post(self, path, body, retries=4):
        last = None
        for a in range(retries):
            try:
                req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST", headers={"content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode("utf-8", "replace")
                if e.code in (400, 413, 422):
                    if "maximum context length" in detail or "max_model_len" in detail or "too long" in detail:
                        raise Capacity(f"maximum context length: {detail[:300]}") from None
                    raise ValueError(f"{e.code}: {detail}") from None
                last = RuntimeError(f"{e.code}: {detail}")
            except (urllib.error.URLError, OSError) as e:
                last = e
            time.sleep(2 ** a)
        raise RuntimeError(f"{path}: {last}")

    def ids(self, text, special=False):
        return self.post("/tokenize", {"model": self.served, "prompt": text, "add_special_tokens": special})["tokens"]

    def readout(self, ids, n):
        if len(ids) + 1 > self.max_len:
            raise Capacity(f"prompt of {len(ids)} tokens is longer than the maximum context length {self.max_len}")
        want = [t for L in self.letter_ids[:n] for t in L]
        body = {"model": self.served, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "logprobs": 20, "logprob_token_ids": want,
                "return_tokens_as_token_ids": True}
        r = self.post("/v1/completions", body)
        top = r["choices"][0]["logprobs"]["top_logprobs"][0]
        lp = {int(k.split(":", 1)[1]): v for k, v in top.items() if k.startswith("token_id:") and v is not None and v > -9999}
        floor = min(lp.values()) - 2.0 if lp else -30.0
        z = [math.log(sum(math.exp(lp.get(t, floor)) for t in L)) for L in self.letter_ids[:n]]
        m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
        return [x / s for x in e], sum(math.exp(v) for v in z)

    def generate(self, ids, max_tokens, seed, n=1):
        """-> (text, completion tokens) for n = 1 (as before); a list of texts and the total tokens for n > 1."""
        body = {"model": self.served, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "seed": seed, "stop": [STOP], "include_stop_str_in_output": False}
        if n > 1: body["n"] = n
        r = self.post("/v1/completions", body)
        ntok = (r.get("usage") or {}).get("completion_tokens", 0)
        if n > 1: return [c.get("text") or "" for c in r["choices"]], ntok
        c = r["choices"][0]
        return c.get("text") or "", ntok


# --- one question ---------------------------------------------------------------------------------------------------------
def chunks(n, size=26):
    k = math.ceil(n / size); base, extra = divmod(n, k)
    out, i = [], 0
    for c in range(k):
        m = base + (1 if c < extra else 0); out.append(list(range(i, i + m))); i += m
    return out


def prompt_ids(cl, state, q):
    """Token ids of one question's one-pass prompt: `paren` = midtrain.letter's plain letter layout (our Base-lineage
    models), `chat` = eval.vllm_letter's chat template (system + user message through the model's own chat template,
    add_generation_prompt, enable_thinking False; the chat-start models, e.g. 013A0-f1)."""
    if cl.template == "chat":
        from eval.vllm_letter import DEFAULT_CHAT_KWARGS, chat_messages
        body = {"model": cl.served, "messages": chat_messages(chat_user(q["chat"], getattr(cl, "prompt_format", "plain"))), "add_generation_prompt": True,
                "add_special_tokens": False, "chat_template_kwargs": DEFAULT_CHAT_KWARGS}
        return cl.post("/tokenize", body)["tokens"]
    from midtrain.letter import question_prompt
    if cl.template == "rawlogit":
        # Match the pretrained-base RawLogitPredictor's A./B. ... Answer: read,
        # including its tokenizer's default special prefix when present.
        return cl.ids(question_prompt(state, q, "rawlogit", "letter"), special=True)
    return cl.ids(format_prompt(question_prompt(state, q, "paren", "letter"), getattr(cl, "prompt_format", "plain")),
                  special=getattr(cl, "paren_special_prefix", False))


CHAT_FORMATS = ("plain", "repeat_state_plain")


def chat_user(parts, fmt="plain"):
    """The chat template's user turn (eval.vllm_letter.user_message) of one question's (state, instructions, options);
    --prompt-format repeat_state_plain writes the state twice, joined by STATE_REPEAT, exactly as format_prompt does in
    the paren prompt (batch 040: the chat-layout arms are read under 03500-c16's DI protocol)."""
    from eval.vllm_letter import user_message
    user = user_message(*parts)
    if fmt == "plain": return user
    if fmt != "repeat_state_plain": raise ValueError(f"--template chat takes --prompt-format {CHAT_FORMATS}, not {fmt!r}")
    if user.startswith("State:\n") and "\n\nQuestion:" in user:
        i = user.index("\n\nQuestion:")
        state = user[len("State:\n"):i]
        return "State:\n" + state + SJ_STATE_REPEAT + state + user[i:]
    return user   # blank state: nothing to repeat


def chat_parts(req, rec, meta):
    """{question id: (rendered state, instructions, option strings)} of a request, as eval.vllm_letter.question_parts
    renders them (kev.api rendering, positional `a: ` prefixes dropped): the chat template's inputs."""
    from midtrain.rawprompt import raw_options   # inlined: its import chain (eval.predictors -> kev.data) needs `datasets`
    out = {}
    for rq, m in zip(rec["questions"], meta):
        src = req["questions"][m["id"]]
        out[m["id"]] = (rec["state"], rq["instr"], raw_options(m["type"], list(src["criteria"]) if m["type"] == "choice" else None, rq["options"]))
    return out


def subset(q, idx):
    sub = {**q, "options": [q["options"][i] for i in idx], "option_texts": [q["option_texts"][i] for i in idx]}
    if "chat" in q: sub["chat"] = (q["chat"][0], q["chat"][1], [q["chat"][2][i] for i in idx])
    return sub


def read_question(cl, state, q, gate, budget, seed_text, usage, k=1, raw=None):
    """-> (probabilities in the question's option order, mode, the one-pass probabilities when it thought else None).
    k > 1: the mean of k post-thought distributions; `raw` (a dict) receives the untempered A and per-thought B reads."""
    from midtrain.letter import question_prompt
    n = len(q["options"])
    if n <= len(LETTERS):
        ida = prompt_ids(cl, state, q); usage["input_tokens"] += len(ida)
        p, _ = cl.readout(ida, n)
        if raw is not None: raw["A"] = p
        if gate > 0 and n > 1 and max(p) < gate and cl.template == "paren":
            prompt = format_prompt(question_prompt(state, q, "paren", "letter"), getattr(cl, "prompt_format", "plain"))
            if not prompt.endswith(CUE): raise ValueError("one-pass prompt does not end with the answer cue")
            pre = cl.ids(prompt[: -len(CUE)] + THINK, special=getattr(cl, "paren_special_prefix", False))
            if len(pre) + budget + 64 <= cl.max_len:   # as think_paren serve: else the one-pass answer stands
                text, ntok = cl.generate(pre, budget, hash_seed(seed_text))
                texts = [text]; usage["output_tokens"] += ntok
                if k > 1:
                    more, ntok = cl.generate(pre, budget, hash_seed(seed_text + "#k"), n=k - 1)
                    texts += more; usage["output_tokens"] += ntok
                reads = []
                for text in texts:
                    body = text.strip()
                    ids = pre + (cl.ids(" " + body) if body else []) + cl.close
                    pb, _ = cl.readout(ids, n); usage["input_tokens"] += len(ids); reads.append(pb)
                if raw is not None: raw["B"] = reads
                pb = [sum(r[i] for r in reads) / len(reads) for i in range(n)]
                return pb, "B", p
        return p, "A", None
    if cl.wide not in ("knockout", "chunk3"):
        raise Capacity(f"{n} options: at most {len(LETTERS)} options per choice")
    if cl.wide == "knockout" and len(chunks(n)) > len(LETTERS):
        raise Capacity(f"{n} options: at most {len(LETTERS) ** 2} options per choice")

    def read(idx):
        ids = prompt_ids(cl, state, subset(q, idx)); usage["input_tokens"] += len(ids)
        return cl.readout(ids, len(idx))[0]
    return wide_read(n, read, int(getattr(cl, "topk", 0) or 0), policy=cl.wide,
                     residual=getattr(cl, "wide_residual", 0.001),
                     diagnostics=raw.setdefault("wide", {}) if raw is not None else None)


def wide_read(n, read, topk=0, *, policy="knockout", residual=0.001, diagnostics=None):
    """More than 26 options, given `read(option indices) -> probabilities over them` (one letter read of that subset):
    knockout (consecutive chunks, then a final over the chunk winners; P(option) = P_final(chunk) x P_chunk(option)),
    then with topk > 0 one K-way re-read of the K options with the highest knockout probability (original order; ties
    to the lower index), whose knockout mass it redistributes. -> (probabilities, mode "K" | "T", knockout
    probabilities when topk ran else None). With policy="chunk3", each chunk promotes its top three (stable ties to
    lower original indices). Finalists get 1-residual mass from the final read; nonfinalists get residual mass
    proportional to normalized within-chunk probabilities. If no nonfinalists exist, the final read gets all mass;
    if their local mass is zero, residual is uniform among them. Mode S is not a knockout product distribution.
    Shared by this reader and the vertical harness's HF
    parity read; the three-argument/default call keeps the legacy knockout behavior. An optional diagnostics dict
    receives original-position routing and untempered read distributions, without adding model calls."""
    if policy not in ("knockout", "chunk3"):
        raise ValueError(f"unknown wide-read policy {policy!r}")
    if policy == "chunk3":
        if type(n) is not int or n < 1:
            raise ValueError("chunk3 needs a positive integer option count")
        if topk != 0:
            raise ValueError("chunk3 and post-knockout topk cannot be combined")
        if not isinstance(residual, (int, float)) or not math.isfinite(residual) or not 0 <= residual < 1:
            raise ValueError("chunk3 residual must satisfy 0 <= residual < 1")
        groups = chunks(n)
        count = sum(min(3, len(g)) for g in groups)
        if count > len(LETTERS):
            raise Capacity(f"{n} options: chunk3 needs {count} finalists, above the {len(LETTERS)}-letter limit")

        def checked_read(indices, stage):
            values = read(indices)
            try:
                values = list(values)
                valid = len(values) == len(indices) and all(math.isfinite(x) and x >= 0 for x in values)
                total = math.fsum(values) if valid else 0.0
                valid = valid and math.isfinite(total) and total > 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"chunk3 {stage} read must return finite nonnegative probabilities for every option")
            return values, total

        finalists = []
        local = [0.0] * n
        within = [] if diagnostics is not None else None
        for g in groups:
            p, total = checked_read(g, "chunk")
            if within is not None:
                within.append([x / total for x in p])
            for j, i in enumerate(g):
                local[i] = p[j] / total
            finalists.extend(sorted(g[j] for j in sorted(range(len(g)), key=lambda j: (-p[j], g[j]))[:3]))
        finalists.sort()  # the final prompt follows source-option order, regardless of within-chunk rank
        final, total = checked_read(finalists, "final")
        if diagnostics is not None:
            diagnostics.update({"policy": "chunk3", "groups": [list(g) for g in groups],
                                "chunk_probabilities": within, "finalist_indices": list(finalists),
                                "final_probabilities": [x / total for x in final], "residual": residual})
        out = [0.0] * n
        selected = set(finalists)
        pruned = [i for i in range(n) if i not in selected]
        keep_mass = 1 - residual if pruned else 1.0
        for j, i in enumerate(finalists):
            out[i] = keep_mass * final[j] / total
        if pruned and residual:
            pruned_mass = sum(local[i] for i in pruned)
            if pruned_mass:
                for i in pruned:
                    out[i] = residual * local[i] / pruned_mass
            else:
                for i in pruned:
                    out[i] = residual / len(pruned)
        return out, "S", None
    groups = chunks(n)
    within, winners = [], []
    for g in groups:
        p = read(g)
        within.append(p); winners.append(g[max(range(len(g)), key=lambda j: p[j])])
    top = read(winners)
    if diagnostics is not None:
        diagnostics.update({"policy": "knockout", "groups": [list(g) for g in groups],
                            "chunk_probabilities": [list(p) for p in within], "finalist_indices": list(winners),
                            "final_probabilities": list(top)})
    out = [0.0] * n
    for c, g in enumerate(groups):
        for j, i in enumerate(g): out[i] = top[c] * within[c][j]
    s = sum(out)
    out = [x / s for x in out]
    k = min(int(topk or 0), len(LETTERS), n)
    if k < 2:
        return out, "K", None
    idx = sorted(sorted(range(n), key=lambda i: (-out[i], i))[:k])
    pk = read(idx)
    if diagnostics is not None:
        diagnostics["post_knockout_topk"] = {"indices": list(idx), "probabilities": list(pk)}
    mass = sum(out[i] for i in idx)
    res = list(out)
    for j, i in enumerate(idx): res[i] = mass * pk[j]
    s = sum(res)
    return [x / s for x in res], "T", out


def answer(cl, req, gate, budget, tables, k=1, return_raw=False):
    from kev.api import SystemOneRequest, to_record
    rec, meta = to_record(SystemOneRequest.model_validate(req))
    answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    seed_base = json.dumps(req, sort_keys=True)   # think_paren serve's seed: hash of the request + question id
    chat = chat_parts(req, rec, meta) if cl.template == "chat" else None
    for rq, m in zip(rec["questions"], meta):
        q = {"id": m["id"], "type": rq["qtype"], "instructions": rq["instr"], "options": rq["options"], "option_texts": rq["options"]}
        if chat is not None: q["chat"] = chat[m["id"]]
        raw = {} if return_raw else None
        p, mode, pa = read_question(cl, rec["state"], q, gate, budget, seed_base + m["id"], usage, k, raw)
        keys = m["keys"]
        # The opt-in shortlist has no fitted calibration table. Tempering all n values would also change its
        # declared residual mass, so preserve the composition until a separate policy-specific calibration exists.
        if mode != "S":
            p = temper(p, temperature(tables.get(mode), m["type"], len(keys)))
        one_pass = None
        if pa is not None and mode == "B":   # the one-pass read this thought replaced, tempered as mode A (so one read yields both policies)
            pa = temper(pa, temperature(tables.get("A"), m["type"], len(keys)))
            one_pass = {"probabilities": dict(zip(keys, pa))}
        j = max(range(len(p)), key=lambda i: p[i])
        if m["type"] == "noul":
            answers[m["id"]] = {"type": "noul", "noul": p[keys.index("true")], "mode": mode}
        elif m["type"] == "choice":
            K = len(p)
            answers[m["id"]] = {"type": "choice", "choice": keys[j], "probabilities": dict(zip(keys, p)),
                                "confidence": 1.0 if K == 1 else (p[j] - 1 / K) / (1 - 1 / K), "mode": mode}
        else:
            answers[m["id"]] = {"type": "score", "score": sum(i * v for i, v in enumerate(p)), "probabilities": {str(i): v for i, v in enumerate(p)},
                                "confidence": p[j], "mode": mode}
        if one_pass is not None:
            answers[m["id"]]["one_pass"] = one_pass
        if raw is not None:
            answers[m["id"]]["raw"] = {"mode": mode, "keys": keys, "A": raw.get("A"), "B": raw.get("B")}
            if "wide" in raw:
                answers[m["id"]]["raw"]["wide"] = raw["wide"]
    return answers, usage


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm", default=None, help="the vLLM server's base URL (required unless --latent-model reads in process)"); ap.add_argument("--served", required=True)
    ap.add_argument("--gate", type=float, default=0.0, help="think when the one-pass max p < gate (0: one-pass only)")
    ap.add_argument("--budget", type=int, default=512); ap.add_argument("--temperature", default=None)
    ap.add_argument("--wide", choices=("knockout", "chunk3", "refuse"), default="knockout")
    ap.add_argument("--wide-residual", type=float, default=0.001, help="chunk3: total mass reserved for nonfinalists (0 <= r < 1; default 0.001)")
    ap.add_argument("--topk", type=int, default=0, help="> 26 options: re-read the knockout's top K (<= 26) as one K-way question (0: off)")
    ap.add_argument("--template", choices=("paren", "chat", "rawlogit"), default="paren", help="paren: midtrain.letter layout; chat: eval.vllm_letter's chat prompt (no thinking); rawlogit: pretrained-base A./B. plain read")
    ap.add_argument("--paren-special-prefix", action="store_true", help="ask the tokenizer to prepend its native special prefix to a paren prompt (Gemma PT training parity); default is the historical no-prefix protocol")
    ap.add_argument("--max-model-len", type=int, default=16384); ap.add_argument("--model-name", default=None)
    ap.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="plain", help="general prompt rewrite adapted from simple-jev (module docstring); plain = unchanged bytes")
    ap.add_argument("--think-k", type=int, default=1, help="thoughts per thinking question; the answer is their mean post-thought distribution")
    ap.add_argument("--return-raw", action="store_true", help="add untempered reads and wide routing diagnostics to answers (offline policy analysis)")
    # Latent readout (docs/plans/latent-reasoning-1002.md; off by default): an in-process Hugging Face reader replaces the vLLM
    # server (vLLM's OpenAI API cannot feed a hidden state back as an input embedding). Everything else is unchanged.
    ap.add_argument("--latent-model", default=None, metavar="DIR", help="read in process from this checkpoint directory instead of --vllm (eval.latent_paren)")
    ap.add_argument("--latent-file", default=None, help="latent_proj.safetensors of a midtrain.latent_lora run (needed for --latent-k > 0)")
    ap.add_argument("--latent-adapter", default=None, help="LoRA adapter directory merged in memory onto --latent-model")
    ap.add_argument("--latent-k", type=int, default=0, help="latent positions before every letter read (0: the plain one-pass read)")
    ap.add_argument("--latent-mode", choices=("normal", "shuffle", "const", "zero"), default="normal", help="latent control: another question's latents / the mean latent / zeros")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args(argv)
    if not math.isfinite(a.wide_residual) or not 0 <= a.wide_residual < 1:
        ap.error("--wide-residual must satisfy 0 <= r < 1")
    if a.wide == "chunk3" and a.topk != 0:
        ap.error("--wide chunk3 cannot be combined with --topk; topk is a post-knockout reread")
    if a.latent_model:
        if a.gate > 0 or a.template != "paren": raise SystemExit("--latent-model serves the paren layout at --gate 0 (it generates nothing)")
        if a.latent_k > 0 and not a.latent_file: raise SystemExit("--latent-k > 0 needs --latent-file")
        from eval.latent_paren import Reader, hf_client
        reader = Reader(a.latent_model, a.latent_file, adapter=a.latent_adapter)
        cl = hf_client(Client, reader, a.latent_k, a.latent_mode, endpoint="hf://in-process", served=a.served, max_len=a.max_model_len)
    else:
        if not a.vllm: ap.error("the following arguments are required: --vllm")
        if a.latent_k or a.latent_file or a.latent_adapter or a.latent_mode != "normal": ap.error("--latent-* flags need --latent-model")
        cl = Client(a.vllm, a.served, a.max_model_len)
    cl.wide = a.wide; cl.template = a.template; cl.prompt_format = a.prompt_format
    if a.paren_special_prefix and a.template != "paren": raise SystemExit("--paren-special-prefix needs --template paren")
    cl.paren_special_prefix = a.paren_special_prefix
    if a.prompt_format != "plain" and a.template != "paren" and not (a.template == "chat" and a.prompt_format in CHAT_FORMATS):
        raise SystemExit(f"--prompt-format needs --template paren (--template chat takes {CHAT_FORMATS})")
    cl.topk = a.topk   # its own statement: on the line above it only ran when the SystemExit was raised (never)
    cl.wide_residual = a.wide_residual
    tables = load_tables(a.temperature, a.budget)
    name = a.model_name or a.served

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args): pass

        def send(self, code, obj):
            b = json.dumps(obj).encode(); self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self.send(200, {"ok": True, "model": name, "gate": a.gate, "budget": a.budget, "wide": a.wide,
                            "wide_residual": a.wide_residual, "topk": a.topk, "max_model_len": a.max_model_len,
                            "think_k": a.think_k, "return_raw": a.return_raw, "prompt_format": a.prompt_format,
                            "paren_special_prefix": a.paren_special_prefix,
                            **({"latent": {**cl.latent, "stats": reader.stats}} if a.latent_model else {})})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
            t0 = time.perf_counter()
            try:
                ans, usage = answer(cl, req, a.gate, a.budget, tables, a.think_k, a.return_raw)
            except Capacity as e:
                return self.send(422, {"error": str(e)[:400]})
            except Exception as e:   # noqa: BLE001
                return self.send(400 if isinstance(e, ValueError) else 500, {"error": f"{type(e).__name__}: {e}"[:400]})
            self.send(200, {"model": name, "answers": ans, "usage": usage, "latency_s": time.perf_counter() - t0})

    ThreadingHTTPServer.daemon_threads = True
    print(json.dumps({"serving": name, "gate": a.gate, "budget": a.budget, "wide": a.wide,
                      "wide_residual": a.wide_residual, "port": a.port}), flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
