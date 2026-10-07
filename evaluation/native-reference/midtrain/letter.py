"""Letter layout: decision records as the plain prompt the base reads (model.readout letter / mix, docs/letter-readout.md).

The pointer layout (encode.encode_decision) wraps state, question and options in delimiter tokens the base never saw and
reads a head trained from scratch. The letter layout keeps the base's own input and output: every question is the plain
prompt an untuned base is scored with, and the answer is read from the LM head's option-letter logits at the prompt's
last token, so at step 0 a letter model scores exactly what the base scores zero-shot on the same prompt.

    layout "paren" (default, decider's style)        layout "rawlogit" (= midtrain.rawprompt, byte for byte)

    State:                                           State:
    {state}                                          {state}

    Question: {instructions}                         Question: {instructions}
    (A) {option 1}                                   A. {option 1}
    (B) {option 2}                                   B. {option 2}
    Answer: (                                        Answer:

    layout "chat" (batch 040: the chat checkpoint's control, = eval.vllm_letter's `chat` prompt byte for byte)

    <|im_start|>system
    {CHAT_SYSTEM}<|im_end|>
    <|im_start|>user
    State:
    {state}

    Question: {instructions}
    Options:
    A. {option 1}
    B. {option 2}

    Answer with the letter of one option.<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

                                                     <- letters read here (z_L as above, A-Z)

    i.e. tokenizer.apply_chat_template([system, user], add_generation_prompt=True, enable_thinking=False) rendered to
    text and tokenized without added special tokens (what vLLM's /tokenize does with chat messages; eval.systemone_vllm
    --template chat and eval.run --prompt-template chat read exactly these bytes). The state part is the template text
    up to the user turn's `Question:` (shared by the record's questions as in paren); a branch ends with the template's
    generation prompt. noul is lettered (letter_noul letter only). The tokenizer must carry a chat template.

The state block is left out when the state is blank; option strings are rawprompt.raw_options' (the rendered option
texts, a positional `a: ` prefix dropped). readout: z_j = log sum_{v in ids(L_j)} exp(W_v . h) at the last token, ids(L) =
the single-token "L" and " L" (rawprompt.letter_token_ids); the softmax over the question's options cancels log Z.
noul "yesno" (model.letter_noul): a noul question whose options are plain no / yes is asked as
`Question: {instructions}\\nAnswer (yes or no):` and read at the no / yes word tokens (yesno_token_ids) instead.

Encoding: the same dict as encode_decision (so every isolation form, packing, the batcher, the decision anchor and the
readout hooks take it unchanged): the state part is `State:\\n{state}\\n\\nQuestion:` (the tokens every question of the
record shares; with the tokenizer's special prefix, e.g. BOS), branch k is the rest of question k's prompt, positions
restart after the state. Each branch is the joint tokenization of the full prompt minus the state part whenever the
state part is a token prefix of it (the usual case; so state + branch == prompt_ids(tok, prompt) exactly), else the
branch's own tokenization (`split_mismatch` counts it). No delimiter tokens, no pauses, no LM labels. decide_idx: the
prompt's last token; opt_idx: the last token of each option line (the mix readout's pointer term reads them; a yes/no
branch has no option lines and points both at the last token, where the pointer term is a constant that cancels).
"""
import re

from .encode import IGNORE, OPT_DECIDE, OPT_NONE, OPT_THINK, ContextOverflow, option_texts, render
from .rawprompt import LETTERS, choice_keys, plain, raw_options

LAYOUTS = ("paren", "rawlogit", "chat")
# chat: eval.vllm_letter's SYSTEM and user turn, copied here (midtrain does not import eval); trainer/eval/tests/
# test_letter_chat_layout.py checks the two stay byte-identical.
CHAT_SYSTEM = ("You are a calibration engine inside a decision system. You are given a state, a question and lettered options. "
               "Choose exactly one option and reply with its letter only: one capital letter, no words, no punctuation, no explanation.")
CHAT_THOUGHT_SYSTEM = ("You are a calibration engine inside a decision system. You are given a state, a question and lettered options. "
                       "Think it through briefly first (check every condition, date and number that matters), "
                       "then reply with one capital letter, no words, no punctuation.")
CHAT_TAIL = "\n\nAnswer with the letter of one option."
CHAT_KWARGS = {"enable_thinking": False}
_CHAT_SENTINEL = "\x00T2M-USER-TURN\x00"
# Not yet a layout here: "semif_chat" (JevK5 / Jobe's prompt; rawprompt.semif_prompt, read by eval's RawLogitPredictor
# with --raw-layout semif_chat, docs/hard-tier-leaders.md technique 1). To train on it: state_text becomes the chat
# prefix up to the evidence (`<|im_start|>system\n{SEMIF_SYSTEM}<|im_end|>\n<|im_start|>user\n{"evidence": <state JSON>,`
# -- evidence comes first in their JSON, so questions still share the state part), branch_text the rest of the JSON
# (`"criterion": ..., "options": [...]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`) with option j's span
# on its "description" value; decide_idx the prompt's last token; the readout the bare letters A-P only (no " L"
# variant: rawprompt.semif_letter_ids), noul asked true-first; tokens with add_special_tokens=False and no `<|name|>`
# rewrite (the template's own specials are real). The state is the record's state as JSON, not encode.render's text.
NOUL_READOUTS = ("letter", "yesno")
_CUE = {"paren": "Answer: (", "rawlogit": "Answer:"}
YESNO_CUE = "Answer (yes or no):"
STATE_TAIL = "Question:"


def option_line(layout, j, text):
    return f"({LETTERS[j]}) {plain(text)}" if layout == "paren" else f"{LETTERS[j]}. {plain(text)}"


def chat_frame(tok, thinking=False):
    """(pre, post): the chat template text around the user turn's content (system turn included in pre, the generation
    prompt with the empty think block in post), cached on the tokenizer. Checked once: pre + content + post equals the
    template's own rendering of two probe contents (so the template does not rewrite the content)."""
    cache_key = "_t2m_thought_chat_frame" if thinking else "_t2m_chat_frame"
    cached = getattr(tok, cache_key, None)
    if cached is not None:
        return cached

    def render(content):
        msgs = [{"role": "system", "content": CHAT_THOUGHT_SYSTEM if thinking else CHAT_SYSTEM},
                {"role": "user", "content": content}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    full = render(_CHAT_SENTINEL)
    if full.count(_CHAT_SENTINEL) != 1:
        raise ValueError("chat layout: the chat template does not place the user content exactly once")
    pre, post = full.split(_CHAT_SENTINEL)
    if thinking:
        # This path supports native ChatML think blocks, not a synthetic Reasoning: cue or an arbitrary template.
        if not post.endswith("<think>\n") or not chat_frame(tok)[1].endswith("<think>\n\n</think>\n\n"):
            raise ValueError("native chat thoughts require a template with an open <think> block and native </think> close")
    for probe in ("State:\nx  y\n\nQuestion: z\nOptions:\nA. a\nB. b" + CHAT_TAIL, "Question: q\nOptions:\nA. 1" + CHAT_TAIL):
        if render(probe) != pre + probe + post:
            raise ValueError("chat layout: the chat template rewrites the user content; the shared state part cannot be cut")
    try:
        setattr(tok, cache_key, (pre, post))
    except AttributeError:
        pass
    return pre, post


def is_yesno(q, noul):
    """Whether question q is read at the no / yes words (model.letter_noul yesno, plain no/yes options only)."""
    return noul == "yesno" and q["type"] == "noul" and option_texts(q) == ["no", "yes"]


def state_text(state, chat_pre=""):
    """The shared state part: `State:\\n{state}\\n\\nQuestion:` (just `Question:` for a blank state); the chat layout
    puts the template's text before the user turn (chat_frame's pre) in front of it."""
    s = render(state)
    return chat_pre + (f"State:\n{plain(s)}\n\n" if str(s).strip() else "") + STATE_TAIL


def branch_text(q, layout="paren", noul="letter"):
    """(text after the state part, [(start, end) char span of option j's text within it]); ValueError past 26 options."""
    if is_yesno(q, noul):
        return f" {plain(q['instructions'])}\n{YESNO_CUE}", []
    texts = option_texts(q)
    if len(texts) > len(LETTERS):
        raise ValueError(f"{len(texts)} options: the letter readout has {len(LETTERS)} letters")
    keys = choice_keys(q["options"]) if q["type"] == "choice" else None
    out, spans = f" {plain(q['instructions'])}" + ("\nOptions:" if layout == "chat" else ""), []
    for j, t in enumerate(raw_options(q["type"], keys, texts)):
        line = option_line(layout, j, t)
        out += "\n"
        spans.append((len(out), len(out) + len(line)))
        out += line
    if layout == "chat":   # the user turn's end; encode_letter / question_prompt append chat_frame's post
        return out + CHAT_TAIL, spans
    return out + "\n" + _CUE[layout], spans


def question_prompt(state, q, layout="paren", noul="letter", tok=None, thinking=False):
    """The full prompt of one question (the text a base is scored on; tests and docs). layout chat needs `tok` (its
    chat template)."""
    if layout == "chat":
        if tok is None:
            raise ValueError("layout chat renders through the tokenizer's chat template: pass tok")
        pre, post = chat_frame(tok, thinking=thinking)
        return state_text(state, pre) + branch_text(q, layout, noul)[0] + post
    if thinking:
        raise ValueError("native thinking prompts require layout chat")
    return state_text(state) + branch_text(q, layout, noul)[0]


def yesno_token_ids(tok):
    """[[no ids], [yes ids]]: the single-token encodings of no / No / " no" / " No" (and yes), deduplicated."""
    out = []
    for w in ("no", "yes"):
        ids = []
        for v in (w, w.capitalize(), " " + w, " " + w.capitalize()):
            enc = tok(v, add_special_tokens=False).input_ids
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        if not ids:
            raise ValueError(f"tokenizer has no single-token encoding of {w!r}")
        out.append(ids)
    return out


def _tokens(tok, text, special):
    enc = tok(text, add_special_tokens=special, return_offsets_mapping=bool(getattr(tok, "is_fast", False)))
    return list(enc.input_ids), (list(enc.offset_mapping) if "offset_mapping" in enc else None)


def _ends(offsets, spans, shift, n, last):
    """Token index (within the branch) of each option span's last token: the last token starting before the span's end."""
    if offsets is None:
        return [last] * len(spans)
    out = []
    for a, b in spans:
        idx = [t for t in range(n) if offsets[t][1] > offsets[t][0] and offsets[t][0] - shift < b]
        out.append(idx[-1] if idx else last)
    return out


def _encode_letter(tok, rec, layout="paren", noul="letter", max_state=1024, max_branch=1024, max_packed=4096, strict=False,
                   drop_many=False, stats=None, thought_mode=False, thought_max_tokens=512, **_):
    """Encode one decision record in the letter layout (module docstring); returns a list of packed encodings like
    encode_decision. A question past 26 options raises ContextOverflow, or is left out with drop_many (training; `stats`
    counts it under "letter_dropped"); a record left without questions raises ContextOverflow. A state over max_state
    tokens raises with strict, else its text is cut (the state part keeps its `\\n\\nQuestion:` tail). Extra keyword
    arguments (n_pause, echo, rationale) are ignored: the layout has none of them."""
    if layout not in LAYOUTS:
        raise ValueError(f"letter layout must be one of {LAYOUTS}, not {layout!r}")
    if noul not in NOUL_READOUTS:
        raise ValueError(f"letter_noul must be one of {NOUL_READOUTS}, not {noul!r}")
    if layout == "chat" and noul != "letter":
        raise ValueError("layout chat reads noul questions as lettered options (letter_noul letter)")
    stats = stats if stats is not None else {}
    special = layout != "chat"     # chat: the template text carries its own specials (no tokenizer prefix, as vLLM)
    pre, post = chat_frame(tok, thinking=thought_mode) if layout == "chat" else ("", "")
    st = state_text(rec["state"], pre)
    S, _ = _tokens(tok, st, special)
    truncated = len(S) > max_state
    if truncated:
        if strict:
            raise ContextOverflow(f"state exceeds {max_state} tokens: {len(S)}")
        tail, _ = _tokens(tok, "\n\n" + STATE_TAIL, False)
        head, _ = _tokens(tok, st[: -len(STATE_TAIL)].rstrip("\n"), special)
        S = head[: max(1, max_state - len(tail))] + tail
    branches = []   # (ids, opt, readout offsets, question index, yes/no)
    for qi, q in enumerate(rec["questions"]):
        try:
            bt, spans = branch_text(q, layout, noul)
            bt += post
        except ValueError as err:
            if not drop_many:
                raise ContextOverflow(str(err)) from None
            stats["letter_dropped"] = stats.get("letter_dropped", 0) + 1
            continue
        ids, offs, shift = None, None, 0
        if not truncated:
            full, foffs = _tokens(tok, st + bt, special)
            if full[: len(S)] == S:
                ids, offs, shift = full[len(S):], (foffs[len(S):] if foffs else None), len(st)
        if ids is None:
            stats["split_mismatch"] = stats.get("split_mismatch", 0) + int(not truncated)
            ids, offs = _tokens(tok, bt, False)
        if not ids:
            raise ContextOverflow("empty question prompt")
        n = len(ids)
        ends = _ends(offs, spans, shift, n, n - 1)
        opt = [OPT_NONE] * n
        labels = [IGNORE] * n
        if offs is not None:
            for t in range(n):
                a = offs[t][0] - shift
                for j, (s0, s1) in enumerate(spans):
                    if s0 <= a < s1:
                        opt[t] = j
                if thought_mode and a >= len(bt) - len(post):
                    opt[t] = OPT_THINK
        if thought_mode:
            if offs is None:
                raise ValueError("native thought supervision requires a fast tokenizer with offsets for anchor masks")
            body, _ = _tokens(tok, q["thought"]["text"].strip() + "\n", False)
            close, _ = _tokens(tok, "</think>", False)
            tail, _ = _tokens(tok, "\n\n", False)
            if len(body) > thought_max_tokens:
                raise ContextOverflow(f"thought exceeds {thought_max_tokens} tokens: {len(body)}")
            if not body or not close or not tail:
                raise ValueError("native chat thought body, close and final tail must tokenize nonempty")
            ids += body + close + tail
            labels += body + close + [IGNORE] * len(tail)
            opt += [OPT_THINK] * (len(body) + len(close) + len(tail))
            n = len(ids)
        opt[-1] = OPT_DECIDE
        if n > max_branch:
            raise ContextOverflow(f"branch too long: {n} tokens (limit {max_branch})")
        if len(S) + n > max_packed:
            raise ContextOverflow(f"state + branch exceeds {max_packed} packed tokens")
        k = len(q["options"]) if not spans else len(spans)
        readout = ends if spans else [n - 1] * k
        branches.append((ids, opt, readout, q, qi, bool(not spans), labels))
    if not branches:
        raise ContextOverflow("no question the letter readout can score")
    out, i = [], 0
    while i < len(branches):
        ids, seg, pos, opt = list(S), [0] * len(S), list(range(len(S))), [OPT_NONE] * len(S)
        decide_idx, opt_idx, targets, qtypes, qids, qidx, yesno = [], [], [], [], [], [], []
        labels = [IGNORE] * len(S)
        k = 0
        while i < len(branches) and len(ids) + len(branches[i][0]) <= max_packed:
            br, bopt, readout, q, qi, yn, blab = branches[i]; k += 1
            base = len(ids)
            ids += br; seg += [k] * len(br); pos += [len(S) + o for o in range(len(br))]; opt += bopt
            labels += blab
            decide_idx.append(base + len(br) - 1); opt_idx.append([base + e for e in readout])
            targets.append(q.get("target", 0)); qtypes.append(q["type"]); qids.append(q.get("id", f"q{qi + 1}")); qidx.append(qi)
            yesno.append(yn)
            i += 1
        L = len(ids)
        out.append({"kind": "decision", "ids": ids, "seg": seg, "pos": pos, "opt": opt, "sib": [0] * L, "decide_idx": decide_idx,
                    "opt_idx": opt_idx, "targets": targets, "qtypes": qtypes, "qids": qids, "qidx": qidx, "tiers": [2 if thought_mode else 0] * k,
                    "rat_corrupt": [False] * k, "lm_labels": labels, "rat_targets": sum(y != IGNORE for y in labels), "n_pause": 0, "echo": False,
                    "tier2_dropped": 0, "state_truncated": truncated, "layout": layout, "yesno": yesno})
        if thought_mode:
            out[-1].update(thought=[True] * k, thought_targets=out[-1]["rat_targets"],
                           thought_acceptance_sha256=[rec["questions"][qi]["thought"]["acceptance_sha256"] for qi in qidx])
    return out


def validate_thought(tok, q):
    """Validate author-supplied metadata. Its marker is not independent data acceptance; source-pass binds that receipt
    to the physical file bytes. Gold targets are never rendered or appended to the assistant input."""
    meta = q.get("thought")
    if (not isinstance(meta, dict) or meta.get("verified") is not True
            or not isinstance(meta.get("text"), str) or not meta["text"].strip()
            or not isinstance(meta.get("acceptance_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", meta["acceptance_sha256"])):
        raise ValueError("thought requires nonempty text, verified true and a lowercase acceptance_sha256")
    controls = ["<think>", "</think>", *getattr(tok, "all_special_tokens", [])]
    if any(c and c in meta["text"] for c in controls) or re.search(r"<\|[^\n]*?\|>", meta["text"]):
        raise ValueError("thought text contains a chat/think control token")
    return meta


def encode_letter(tok, rec, layout="paren", noul="letter", thought=None, **kwargs):
    """Existing letter encoder, plus opt-in native thought LM + final letter CE. Ordinary records are unchanged.
    Thought records require verified metadata; contiguous ordinary/think question groups get separate native system
    prefixes. Each raw question appears exactly once, without a gold letter in ids or LM labels. Max-token overflow
    drops the record through the caller's existing overflow handling; traces are never truncated."""
    settings = thought or {}
    if not settings.get("enabled"):
        return _encode_letter(tok, rec, layout=layout, noul=noul, **kwargs)
    if layout != "chat" or noul != "letter":
        raise ValueError("native thoughts require letter layout chat and noul letter")
    modes = []
    for q in rec["questions"]:
        has_thought = "thought" in q
        if has_thought:
            validate_thought(tok, q)
        modes.append(has_thought)
    if not any(modes):
        return _encode_letter(tok, rec, layout=layout, noul=noul, **kwargs)
    if rec.get("kl_only"):
        raise ValueError("native thought supervision requires labelled decision rows, not kl_only")
    out, start = [], 0
    while start < len(modes):
        end = start + 1
        while end < len(modes) and modes[end] == modes[start]:
            end += 1
        # Preserve IDs derived from the raw question index when a question has no explicit id.
        qs = [{**q, "id": q.get("id", f"q{qi + 1}")} for qi, q in enumerate(rec["questions"][start:end], start)]
        group = {**rec, "questions": qs}
        encs = _encode_letter(tok, group, layout=layout, noul=noul, thought_mode=modes[start],
                              thought_max_tokens=int(settings.get("max_tokens", 512)), **kwargs)
        for e in encs:
            e["qidx"] = [qi + start for qi in e["qidx"]]
        out.extend(encs)
        start = end
    return out


def attach_letter_raw_ids(encs, max_tokens):
    """Reference prompts for the decision anchor / KL-only rows under the letter layout: branch k's prompt is the
    encoding's own state + branch ids, exactly the tokens the model reads, so KL(p_ref || p_model) compares the two
    models on one prompt (rawprompt.attach_raw_ids renders the rawlogit prompt instead). Sets raw_ids / raw_skip as
    attach_raw_ids does: "length" past max_tokens, "yesno" for a no / yes-word branch (its readout is not the letters)."""
    for e in encs:
        Ls = e["seg"].count(0)
        S, start = e["ids"][:Ls], Ls
        ids, skip = [], []
        for d, yn in zip(e["decide_idx"], e.get("yesno") or [False] * len(e["decide_idx"])):
            end = d + 1
            p = S + e["ids"][start:end]
            start = end
            if yn:
                ids.append(None); skip.append("yesno")
            elif len(p) > int(max_tokens):
                ids.append(None); skip.append("length")
            else:
                ids.append(p); skip.append(None)
        e["raw_ids"], e["raw_skip"] = ids, skip
    return encs


def letter_kwargs(cfg):
    """encode_letter keyword arguments for a run config (model.readout letter / mix), or None for the pointer layout."""
    m = cfg.get("model") or {}
    if m.get("readout", "pointer") == "pointer":
        return None
    kwargs = {"layout": m.get("letter_layout", "paren"), "noul": m.get("letter_noul", "letter")}
    if (m.get("thought") or {}).get("enabled"):
        kwargs["thought"] = m["thought"]
    return kwargs
