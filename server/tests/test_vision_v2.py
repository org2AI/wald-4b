"""CPU tests of the image route (vision_v2): transports, sentinels, prompt splitting and the completions <-> chat mapping."""
import base64, io, json, sys
from pathlib import Path

import pytest

from wald_serve import vision_v2 as v  # noqa: E402


def png_uri(color=(10, 20, 30)):
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (8, 8), color).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


Q = {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "left", "b": "right"}}}


def test_no_images_is_not_routed():
    assert not v.has_images({"state": "light: red", "questions": Q})
    assert not v.has_images({"state": {"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}, "questions": Q})


@pytest.mark.parametrize("make", [
    lambda u: {"state": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Look:"}, {"type": "image_url", "image_url": {"url": u}},
                                                                  {"type": "text", "text": "then decide"}]}]}, "questions": Q},
    lambda u: {"state": {"messages": [{"role": "user", "content": [{"type": "image", "image": u}, {"type": "text", "text": "Look"}]}]}, "questions": Q},
    lambda u: {"state": {"screen": u, "task": "Look"}, "questions": Q},
    lambda u: {"state": {"image": u.split(",", 1)[1], "task": "Look"}, "questions": Q},
    lambda u: {"images": [u], "state": "Look", "questions": Q},
])
def test_transports_lift_to_one_sentinel(make):
    u = png_uri(); req = make(u)
    assert v.has_images(req)
    out, uris, pat = v.lift(req)
    assert uris == [u] and set(out) == {"state", "questions"}
    s = json.dumps(out["state"], ensure_ascii=False)
    assert len(pat.findall(s)) == 1 and "data:image" not in s


def test_messages_flatten_keeps_position():
    u = png_uri()
    out, uris, pat = v.lift({"state": {"messages": [{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "image_url", "image_url": {"url": u}},
                                                                                {"type": "text", "text": "B"}]}]}, "questions": Q})
    assert isinstance(out["state"], str)
    a, rest = out["state"].split("\n", 1)
    assert a == "A" and pat.fullmatch(rest.split("\n")[0]) and rest.endswith("\nB")


def test_to_parts_and_repeat():
    u1, u2 = png_uri((1, 1, 1)), png_uri((2, 2, 2))
    out, uris, pat = v.lift({"images": [u1, u2], "state": "S", "questions": Q})
    m0, m1 = [m.group(0) for m in pat.finditer(out["state"])]
    text = f"<|im_start|>user\nState:\n{m0}\n{m1}\nS\n\nRead again:\n{m0}\n{m1}\nS<|im_end|>"
    parts = v.to_parts(text, uris, pat)
    imgs = [p for p in parts if p["type"] == "image_url"]
    assert [p["image_url"]["url"] for p in imgs] == [u1, u2, u1, u2]
    assert "".join(p["text"] if p["type"] == "text" else v.VISION for p in parts).count(v.VISION) == 4
    with pytest.raises(v.ImageError):
        v.to_parts("no slots here", uris, pat)


def test_limits_and_bad_images():
    u = png_uri()
    with pytest.raises(v.ImageError):
        v.lift({"images": [u] * 17, "state": "s", "questions": Q})
    with pytest.raises(v.ImageError):
        v.lift({"images": ["data:image/png;base64," + base64.b64encode(b"notanimage" * 10).decode()], "state": "s", "questions": Q})


def test_image_client_maps_completions_to_chat():
    calls = []

    class Base:
        served = "m"

        def post(self, path, body, retries=1):
            calls.append((path, body))
            if path == "/detokenize":
                return {"prompt": "<|im_start|>user\n" + body["_text"] if "_text" in body else "<|im_start|>user\nX " + MARK + " Y<|im_end|>"}
            if path == "/v1/chat/completions":
                if body.get("max_tokens") == 1:
                    return {"choices": [{"index": 0, "message": {"content": "A"}, "finish_reason": "length",
                                         "logprobs": {"content": [{"token": "token_id:32", "logprob": -0.1,
                                                                   "top_logprobs": [{"token": "token_id:32", "logprob": -0.1}, {"token": "token_id:33", "logprob": -2.4}]}]}}],
                            "usage": {"prompt_tokens": 50, "completion_tokens": 1}}
                return {"choices": [{"index": 0, "message": {"content": "thinking"}, "finish_reason": "stop", "stop_reason": "</think>"}],
                        "usage": {"completion_tokens": 7}}
            return {"tokens": [1, 2, 3]}

    u = png_uri()
    _, uris, pat = v.lift({"images": [u], "state": "s", "questions": Q})
    global MARK
    MARK = "⟦" + pat.pattern.split("⟦")[1].split(":")[0] + ":0⟧"
    C = v.image_client(Base)
    c = C().bind(uris, pat)
    r = c.post("/v1/completions", {"model": "m", "prompt": [5, 6, 7], "max_tokens": 1, "temperature": 0.0, "logprobs": 20,
                                   "logprob_token_ids": [32, 33], "return_tokens_as_token_ids": True})
    assert r["choices"][0]["logprobs"]["top_logprobs"][0] == {"token_id:32": -0.1, "token_id:33": -2.4}
    chat = calls[-1][1]
    assert calls[-1][0] == "/v1/chat/completions" and chat["chat_template"] == v.CHAT_TEMPLATE and chat["add_generation_prompt"] is False
    assert chat["logprob_token_ids"] == [32, 33] and chat["top_logprobs"] == 20 and chat["logprobs"] is True
    assert [p["type"] for p in chat["messages"][0]["content"]] == ["text", "image_url", "text"]
    g = c.post("/v1/completions", {"model": "m", "prompt": [5], "max_tokens": 512, "stop": ["</think>"], "seed": 3})
    assert g["choices"][0]["text"] == "thinking" and g["usage"]["completion_tokens"] == 7 and g["choices"][0]["stop_reason"] == "</think>"
    # without bound images, every call passes through untouched
    c.bind([], None); n = len(calls)
    c.post("/v1/completions", {"model": "m", "prompt": [5]})
    assert calls[n][0] == "/v1/completions"
