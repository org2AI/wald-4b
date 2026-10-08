"""Image route of `wald-serve-native` for the Wald-4B v2 vision variant (04701-c22 + the Qwen3.5-4B chat vision tower).

A request without images never reaches this module: it takes the frozen native-v2 path byte for byte.

A request with images is answered by the SAME frozen reader (`native_reference/`, unchanged and SHA-pinned): gate,
budget, sampling seed, thought prompt and calibration table are the text path's. Only the transport differs:

  1. every image (any accepted transport, below) is lifted out of the request and replaced by a per-request sentinel
     `⟦<nonce>:<slot>⟧` at its position (the sentinel is plain text, so the frozen reader renders it like any state text);
     a `{"messages": [...]}` state is flattened first (text parts joined by newlines, images at their position);
  2. the frozen reader builds its token ids exactly as for text (`/tokenize` with the chat template);
  3. `ImageClient` (a subclass of the reader's client) intercepts the two engine calls that take token ids
     (`/v1/completions` for the letter readout and for the native thought): it decodes the ids to text (`/detokenize`),
     replaces each sentinel by the image (`<|vision_start|><|image_pad|><|vision_end|>` placeholder + pixels) and sends
     the same prompt through vLLM's `/v1/chat/completions` with a verbatim chat template (content parts as given, no
     roles), with the reader's own sampling fields; the response is mapped back to the completions shape.

So the prompt bytes around each image are those of the text path, and the vision tokens sit where the image was in the
state. With `prompt_format repeat_state_plain` the state (and so each image) is written twice, as the text path writes
the state twice; `image_prompt_format` (serving.json, default `plain`) is the default for image requests.

Accepted image transports (at most `max_images` per request, PNG / JPEG / WebP / GIF, each <= 20 MB):
  * data URIs anywhere inside `state` (a string, a dict value, a list item, a `messages[].content` part
    `{"type": "image_url", "image_url": {"url": ...}}` or `{"type": "image", "image": ...}`);
  * raw base64 as the value of a `state` key `image` / `image_data` / `image_base64` / `image_b64`;
  * a top-level `images` (or `image_data`) list of data URIs / base64 strings: placed at the start of the state, in order.
"""
from __future__ import annotations

import base64
import binascii
import re
import secrets
from typing import Any

VISION = "<|vision_start|><|image_pad|><|vision_end|>"
# verbatim template: the content parts in order, each image part as Qwen3.5's placeholder; no roles, no special tokens
CHAT_TEMPLATE = ("{%- for message in messages -%}{%- for part in message['content'] -%}"
                 "{%- if part['type'] == 'text' -%}{{ part['text'] }}"
                 "{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}"
                 "{%- endfor -%}{%- endfor -%}")
DATA_URI = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
RAW_KEYS = ("image", "image_data", "image_base64", "image_b64")
_MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8\xff", "jpeg"), (b"GIF87a", "gif"), (b"GIF89a", "gif"))
_B64 = re.compile(r"[A-Za-z0-9+/=\s]+")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGES = 16


class ImageError(ValueError):
    """A malformed image input (HTTP 400)."""


def sniff(raw: bytes):
    for magic, kind in _MAGIC:
        if raw.startswith(magic):
            return kind
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return None


def raw_to_uri(value):
    if not isinstance(value, str) or len(value) < 64 or not _B64.fullmatch(value[:4096]):
        return None
    try:
        head = base64.b64decode(re.sub(r"\s+", "", value[:64]), validate=True)
    except (binascii.Error, ValueError):
        return None
    kind = sniff(head)
    return None if kind is None else f"data:image/{kind};base64," + re.sub(r"\s+", "", value)


def check_uri(uri):
    payload = uri.split(",", 1)[1]
    if len(payload) * 3 // 4 > MAX_IMAGE_BYTES:
        raise ImageError(f"image larger than {MAX_IMAGE_BYTES} bytes")
    try:
        head = base64.b64decode(payload[:64], validate=True)
    except (binascii.Error, ValueError):
        raise ImageError("image data URI is not valid base64") from None
    if sniff(head) is None:
        raise ImageError("image is not PNG, JPEG, WebP or GIF")
    return uri


def _has(v):
    if isinstance(v, str):
        return "data:image/" in v
    if isinstance(v, list):
        return any(_has(x) for x in v)
    if isinstance(v, dict):
        return any(_has(x) or (k in RAW_KEYS and raw_to_uri(x) is not None) for k, x in v.items())
    return False


def has_images(req) -> bool:
    if not isinstance(req, dict):
        return False
    return bool(req.get("images") or req.get("image_data")) or _has(req.get("state"))


def lift(req: dict, max_images: int = MAX_IMAGES):
    """-> (request with images replaced by sentinels, data URIs by slot, sentinel regex)."""
    nonce = secrets.token_hex(6)
    mark = lambda i: f"⟦{nonce}:{i}⟧"
    uris = []

    def slot(uri):
        uris.append(check_uri(uri))
        return mark(len(uris) - 1)

    def text(s):
        return DATA_URI.sub(lambda m: slot(m.group(0)), s) if "data:image/" in s else s

    def walk(v):
        if isinstance(v, str):
            return text(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            out = {}
            for k, x in v.items():
                uri = raw_to_uri(x) if k in RAW_KEYS and isinstance(x, str) and not x.startswith("data:") else None
                out[k] = slot(uri) if uri else walk(x)
            return out
        return v

    top = []
    for key in ("images", "image_data"):
        v = req.get(key)
        if v in (None, []):
            continue
        for x in (v if isinstance(v, list) else [v]):
            if not isinstance(x, str):
                raise ImageError(f"`{key}` items must be data URIs or base64 strings")
            uri = x if DATA_URI.fullmatch(x) else raw_to_uri(x)
            if uri is None:
                raise ImageError(f"`{key}` item is not an image")
            top.append(slot(uri))
    state = flatten_messages(walk(req.get("state")))
    if top:
        state = "\n".join(top + ([state] if isinstance(state, str) and state.strip() else [])) if isinstance(state, str) or state is None \
            else {"images": "\n".join(top), **state} if isinstance(state, dict) else "\n".join(top)
    if len(uris) > max_images:
        raise ImageError(f"{len(uris)} images: at most {max_images} per request")
    out = {"state": state, "questions": req["questions"]}
    return out, uris, re.compile("⟦" + nonce + r":(\d+)⟧")


def flatten_messages(state):
    """{"messages": [...]} -> text: content parts in order, text parts joined by newlines, image parts (already
    sentinels after lift) at their position; any other state is returned unchanged."""
    if not (isinstance(state, dict) and set(state) == {"messages"} and isinstance(state["messages"], list)):
        return state
    parts = []
    for m in state["messages"]:
        c = m.get("content") if isinstance(m, dict) else m
        if isinstance(c, str):
            parts.append(c); continue
        for x in c or []:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict) and x.get("type") in ("text", "input_text"):
                parts.append(str(x.get("text", "")))
            elif isinstance(x, dict) and x.get("type") == "image_url":
                u = x.get("image_url"); parts.append(str(u.get("url", "") if isinstance(u, dict) else u))
            elif isinstance(x, dict) and x.get("type") in ("image", "input_image"):
                parts.append(str(x.get("image") or x.get("image_url") or ""))
            else:
                raise ImageError(f"unsupported content part {x.get('type') if isinstance(x, dict) else type(x)!r}")
    return "\n".join(parts)


def to_parts(text: str, uris, pat):
    """Split a decoded prompt at the sentinels into chat content parts; every slot must appear (repeat formats may
    write a slot more than once)."""
    parts, pos, seen = [], 0, set()
    for m in pat.finditer(text):
        i = int(m.group(1))
        if i >= len(uris):
            raise ImageError("unknown image slot")
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        parts.append({"type": "image_url", "image_url": {"url": uris[i]}}); seen.add(i); pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    if seen != set(range(len(uris))):
        raise ImageError("an image slot is missing from the prompt")
    return parts


def image_client(base):
    """Subclass the frozen reader's client class `base` so its token-id completions carry the request's images."""

    class ImageClient(base):
        uris, pat = [], None
        mm_calls = 0

        def bind(self, uris, pat):
            self.uris, self.pat = uris, pat
            return self

        def post(self, path, body, retries=1):
            if path != "/v1/completions" or not self.uris or not isinstance(body.get("prompt"), list):
                return super().post(path, body, retries)
            text = super().post("/detokenize", {"model": self.served, "tokens": body["prompt"]}, retries)["prompt"]
            parts = to_parts(text, self.uris, self.pat)
            chat = {k: v for k, v in body.items() if k not in ("prompt", "logprobs")}
            chat.update(messages=[{"role": "user", "content": parts}], chat_template=CHAT_TEMPLATE, add_generation_prompt=False,
                        add_special_tokens=False)
            if "logprobs" in body:
                chat.update(logprobs=True, top_logprobs=body["logprobs"])
            r = super().post("/v1/chat/completions", chat, retries)
            self.mm_calls += 1
            out = []
            for c in r["choices"]:
                o = {"index": c.get("index", 0), "text": (c.get("message") or {}).get("content") or "",
                     "finish_reason": c.get("finish_reason"), "stop_reason": c.get("stop_reason")}
                lp = c.get("logprobs")
                if lp and lp.get("content"):
                    o["logprobs"] = {"top_logprobs": [{t["token"]: t["logprob"] for t in lp["content"][0]["top_logprobs"]}]}
                out.append(o)
            return {"choices": out, "usage": r.get("usage") or {}}

    return ImageClient
