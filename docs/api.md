# Wald-Q4B v2 decision API

`wald-serve-native` (in `server/`, started by `./run.sh`) exposes one decision endpoint, `POST /v1/systemone`. Request shapes follow TypeSafe's `/v1/systemone` format, so clients written for Jev can point at a self-hosted Wald server. Wald is independent and is not affiliated with TypeSafe AI. The server does not check API keys; put it behind your own gateway.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/systemone` (also `POST /`) | Answer one or more typed questions about a state |
| `GET /health` | Checkpoint, weights sha256, default effort / policy, thought budget, prompt format, context limit, calibration sha256 |
| `GET /v1/models` | The served model name (`04701-c22`) |

## Request

| Field | Type | Meaning |
|---|---|---|
| `state` | string, object, array or null | What the decision is about: a message, a conversation, a document, an agent trace. Objects and arrays are flattened to text with their field names kept. |
| `questions` | object, at least one entry | Question id → question, all about the same `state`. |
| `effort` | string, optional | `none` (one pass), `auto` or `medium` (Auto 0.7, the default), `always` or `high` (Always 512). Other values (`low`, `high-k2` …) are v1.x efforts and return HTTP 400. |
| `prompt_format` | string, optional | `repeat_state_plain` (default: the state is written twice) or `plain`. |
| `images` | array of data URIs, optional | Images for the state (also accepted inside `state` as message content parts `{"type": "image_url", "image_url": {"url": "data:..."}}`); up to 16; zero-shot. |
| `model` | string, optional | Ignored; accepted for client compatibility. |

Only `state`, `questions` and the images reach the model. Each question has a `type`, optional `instructions` and `criteria`:

| `type` | `criteria` | Answer fields |
|---|---|---|
| `choice` | Object of options: key → description (description may be null) | `choice` (most probable key), `probabilities` (key → probability), `confidence` |
| `noul` | Optional object with `true` and/or `false` descriptions | `noul` (probability of yes), `probabilities` (`true` / `false`), `confidence` |
| `score` | Array of ordered levels | `score` (expected level index), `probabilities` (level index → probability), `confidence` |

Every answer also carries `type`, `mode` (`A` one pass, `B` after a native thought, `K`/`T` grouped readout for more than 26 options) and `probabilities_t1` (the same distribution before the calibration table). `confidence` is the largest calibrated probability. The response has `model`, `answers` and `usage` (`input_tokens`, `output_tokens`; output tokens are thought tokens). The thought text itself is not returned.

Prompts longer than the context limit (131,072 tokens) are rejected with HTTP 422, never truncated. A malformed request or an unknown effort / prompt format returns HTTP 400; a backend failure returns 502.

## Example: route a tool call

```sh
curl http://localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "User: book me a table for two at 7pm tomorrow near the office",
  "effort": "auto",
  "questions": {
    "tool": {"type": "choice", "instructions": "Which tool should the agent call next?",
             "criteria": {"restaurant_search": "Search restaurants", "calendar_create": "Create a calendar event",
                          "ask_user": "Ask the user a clarifying question"}},
    "enough_info": {"type": "noul", "instructions": "Is the request specific enough to act on without asking?"}
  }
}'
```

Response shape (values illustrative):

```json
{"model": "04701-c22",
 "answers": {
  "tool": {"type": "choice", "mode": "A", "choice": "restaurant_search",
           "probabilities": {"restaurant_search": 0.81, "calendar_create": 0.05, "ask_user": 0.14},
           "probabilities_t1": {"restaurant_search": 0.93, "calendar_create": 0.01, "ask_user": 0.06}, "confidence": 0.81},
  "enough_info": {"type": "noul", "mode": "B", "noul": 0.64, "probabilities": {"true": 0.64, "false": 0.36},
                  "probabilities_t1": {"true": 0.71, "false": 0.29}, "confidence": 0.64}},
 "usage": {"input_tokens": 1450, "output_tokens": 212}}
```

Act on a decision when its probability clears a threshold you set on your own validation data; otherwise escalate or ask. The model picks options; it does not write tool arguments.
