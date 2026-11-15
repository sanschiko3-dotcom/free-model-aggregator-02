#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, live free models.

An entrant for make.fun's free-model-aggregator-02. The server reads the
brief at $MAKEFUN_PROMPT_FILE (the free-model list with the house's ratings
and capabilities, plus the six prompts' manifest), registers itself with the
env at $MAKEFUN_ENV_URL, and answers every prompt with a real completion
from one of the brief's free OpenRouter models, dialled through the gateway
at $OPENAI_BASE_URL on the trial's key.

Design, in the order it matters for the score:

1.  Routing is rating-first, requirements-aware. The requirements of each
    request are read off the request body itself (a response_format means
    JSON mode, a non-empty tools array means tools, an image part means
    vision), and the candidates that meet them are tried in rating order
    (rating_milli descending, id ascending as the deterministic tie-break).
    The brief's manifest is only a fallback when a body carries nothing
    readable, so fresh instances with different requirements still route
    correctly.

2.  Every upstream answer is validated before it is served: it must be a
    2xx JSON (or SSE) answer whose text is non-empty and carries the
    prompt's token -- the token is read out of the request's own messages,
    so the check needs no manifest. A model whose answer fails validation
    is treated exactly like a model that refused: the failure is recorded,
    the model is marked unhealthy for the rest of the trial, and the next
    best eligible model is tried. This is what makes the top-rated model
    safe to try first: if the best model cannot actually serve a prompt
    (it refuses, it rate-limits, or its visible text never carries the
    token), the prompt still lands on the next best model instead of being
    lost. Health is a function of the gateway's answers only, so a replay
    of the same trial takes the same path.

3.  Streams are buffered, then re-emitted canonically. The upstream SSE
    stream is consumed whole, its text, tool calls, finish reason and
    usage collected, and only then is the client's stream written: one
    content chunk, one finish chunk carrying usage, then [DONE] on the
    OpenAI wire; message_start, one text block, message_delta with the
    stop reason and output tokens, message_stop on the Anthropic wire.
    Buffering is what guarantees the event order the wire checks demand
    (exactly one finish, nothing after [DONE] / message_stop) and lets the
    failure trail be reported in the response headers, which are sent
    before any body byte.

4.  Nothing in any answer depends on the wall clock or on unseeded
    randomness. Response ids are fixed strings, `created` comes from
    MAKEFUN_TRIAL_NOW (or a constant), and usage, when the provider
    omits it, is derived from the request and answer text lengths. A
    replay of the same trial therefore leaves the same state export.

5.  The failure trail. Every refusal or invalid answer on the way to a
    served prompt is reported in the x-makefun-upstream-failures header
    (at most 8 rows, in the order the calls failed), which is the only
    way the failover column can see a refusal the server recovered from.

Standard library only. Run as `python3 main.py` in a trial, or
`python3 main.py --list` to print the catalog document and exit.
"""

import json
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
# A trial may make at most 60 model calls; three attempts per prompt keeps
# the worst case at 18 and the usual case at exactly one call per prompt.
MAX_ATTEMPTS_PER_PROMPT = 3
# The gateway refuses an output cap over 8192 tokens, and a call with no
# cap at all, so a request that names none gets this one.
DEFAULT_MAX_TOKENS = 1024
MAX_OUTPUT_TOKENS = 8192
# One gateway call must answer well inside the client's 30-second wait.
GATEWAY_TIMEOUT_SEC = 25
# The failure-trail header holds at most 8 rows and 8 KiB.
MAX_TRAIL_ROWS = 8
FAILURES_HEADER = "x-makefun-upstream-failures"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_URL = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")

SCHEMA_PREFIX = "free-model-aggregator-02/"
OPENROUTER_PUBLIC_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# Fixed response ids and a fixed fallback timestamp: nothing in an answer
# may differ between a trial's live pass and its replay.
COMPLETION_ID = "chatcmpl-fma02"
MESSAGE_ID = "msg_fma02"
FALLBACK_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

# A prompt token looks like SPARROW-7435: 3-12 capitals, a dash, 4 digits.
TOKEN_RE = re.compile(r"[A-Z]{3,12}-[0-9]{4}")

Json = dict[str, object]


def created_epoch() -> int:
    """The trial's opening time as an epoch second, else a fixed constant.

    MAKEFUN_TRIAL_NOW is the only clock a trial may read; the value is the
    same on a live pass and its replay, so `created` is too.
    """
    raw = os.environ.get("MAKEFUN_TRIAL_NOW", "")
    if raw:
        try:
            moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return int(moment.timestamp())
        except ValueError:
            pass
    return FALLBACK_CREATED


CREATED = created_epoch()


# ---------------------------------------------------------------- small checks


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ""


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


# ---------------------------------------------------------------- the brief


def object_end(text: str, start: int) -> int:
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` carrying a string `schema`, in order."""
    out: list[Json] = []
    start = text.find("{")
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def model_row(value: object) -> Json | None:
    """One row of the brief's model list, normalised to the gateway's id."""
    if not is_record(value):
        return None
    raw = string_of(value.get("id")) or string_of(value.get("model"))
    if raw == "":
        return None
    model_id = raw if raw.startswith("openrouter/") else f"openrouter/{raw}"
    rating = number_or_null(value.get("rating_milli"))
    return {
        "id": model_id,
        "provider": string_of(value.get("provider")) or "openrouter",
        "display_name": string_of(value.get("display_name")) or raw,
        "rating_milli": rating if rating is not None else 0,
        "context_length": number_or_null(value.get("context_length")),
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is not False,
    }


def prompt_row(value: object) -> Json | None:
    """One row of the brief's prompt manifest (fallback requirements)."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get("id"))
    if prompt_id == "":
        return None
    requires = value.get("requires") if is_record(value.get("requires")) else {}
    return {
        "id": prompt_id,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": {
            "tools": requires.get("tools") is True,
            "vision": requires.get("vision") is True,
            "json": requires.get("json") is True,
        },
    }


def read_brief(path: str) -> Json:
    """The brief's models and prompts, found by schema under this contest."""
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": f"cannot read {path}: {err}"}
    docs = documents_in(text)
    models: list[Json] = []
    prompts: list[Json] = []
    for doc in docs:
        if SCHEMA_PREFIX not in string_of(doc.get("schema")):
            continue
        rows = doc.get("models") or doc.get("ratings")
        if isinstance(rows, list) and not models:
            models = [m for m in (model_row(v) for v in rows) if m is not None]
        rows = doc.get("prompts") or doc.get("manifest")
        if isinstance(rows, list) and not prompts:
            prompts = [p for p in (prompt_row(v) for v in rows) if p is not None]
    if not models:
        return {"models": [], "prompts": prompts, "error": f"the brief at {path} holds no model list"}
    return {"models": models, "prompts": prompts, "error": None}


BRIEF = read_brief(PROMPT_FILE)


def free_models() -> list[Json]:
    """The brief's free models, in the brief's order, deduplicated by id."""
    seen: set[str] = set()
    out: list[Json] = []
    for model in BRIEF["models"]:
        if not model["free"] or model["id"] in seen:
            continue
        seen.add(model["id"])
        out.append(model)
    return out


def build_catalog() -> Json:
    """The catalog document: one provider, exactly the free models.

    base_url is OpenRouter's public endpoint, never the gateway's trial
    address: the gateway's origin differs between a live pass and its
    replay, and the catalog is part of the export a replay compares.
    """
    models = [
        {"id": m["id"], "free": True, "context_length": m["context_length"]}
        for m in free_models()
    ]
    if not models:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": OPENROUTER_PUBLIC_BASE,
                "key_env": OPENROUTER_KEY_ENV,
                "wire": "openai",
                "models": models,
            }
        ]
    }


CATALOG = build_catalog()
CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


def catalog_models() -> list[Json]:
    """The served models as GET /v1/models names them."""
    return [
        {"id": m["id"], "provider": m["provider"], "display_name": m["display_name"]}
        for m in free_models()
    ]


# ---------------------------------------------------------------- requirements


def meets(requires: Json, model: Json) -> bool:
    """Whether a model's capabilities meet a request's requirements.

    A capability the brief does not claim is treated as absent: a model
    must positively support JSON mode to serve a JSON-mode prompt, and
    likewise for tools and vision.
    """
    if requires.get("tools") and not model["tools"]:
        return False
    if requires.get("vision") and not model["vision"]:
        return False
    if requires.get("json") and not model["json"]:
        return False
    return True


def ranked_candidates(requires: Json) -> list[Json]:
    """Eligible models, best rating first, id ascending to break ties."""
    eligible = [m for m in free_models() if meets(requires, m)]
    eligible.sort(key=lambda m: (-m["rating_milli"], m["id"]))
    return eligible


def message_texts(body: Json) -> list[str]:
    """The text of every message in a request body, on either wire."""
    texts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        texts.append(system)
    elif isinstance(system, list):
        texts.append(text_of(system))
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for message in messages:
        if not is_record(message):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.append(text_of(content))
    return texts


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        b["text"]
        for b in content
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    return "\n".join(parts)


def token_in_request(body: Json) -> str | None:
    """The prompt's unique token, read out of the request's own messages."""
    for text in message_texts(body):
        match = TOKEN_RE.search(text)
        if match:
            return match.group(0)
    return None


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body, on either wire."""
    vision = False
    for message in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    response_format = body.get("response_format")
    response_type = response_format.get("type") if is_record(response_format) else None
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": response_type in ("json_object", "json_schema"),
    }


def requirements_for(body: Json) -> Json:
    """The requirements of the request now arriving.

    The body is the authority: it carries the response_format, the tools
    and the image parts the model must actually support, so fresh
    instances with different requirements still route correctly. (The
    brief's manifest lists the same requirements, but a request's body
    is what the model call must satisfy.)
    """
    return requirements_from_body(body)


# ---------------------------------------------------------------- wire translation


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    source = block.get("source")
    if block.get("type") == "image" and is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source["media_type"] if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object; {} when they are not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as a Chat Completions request.

    The system prompt becomes a system message, image blocks become
    image_url data URLs, tool_use blocks become tool_calls, tool_result
    blocks become tool messages, and tools, tool_choice, the output cap,
    temperature, top_p, stop sequences and streaming pass through.
    """
    messages: list[Json] = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    source = body.get("messages") if isinstance(body.get("messages"), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": block["id"] if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block["name"] if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block["input"] if is_record(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"] if isinstance(block.get("tool_use_id"), str) else "",
                        "content": tool_result_text(block.get("content")),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})

    out: Json = {"model": model, "messages": messages}
    if is_number(body.get("max_tokens")):
        out["max_tokens"] = body["max_tokens"]
    if is_number(body.get("temperature")):
        out["temperature"] = body["temperature"]
    if is_number(body.get("top_p")):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and len(body["stop_sequences"]) > 0:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools: list[Json] = []
        for raw in body["tools"]:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "description": raw["description"] if isinstance(raw.get("description"), str) else "",
                        "parameters": raw["input_schema"] if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                    },
                }
            )
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object."""
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content: list[Json] = []
    text = message["content"] if isinstance(message.get("content"), str) else text_of(message.get("content"))
    if text != "":
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) else "toolu_unknown",
                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                "input": arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return {
        "id": MESSAGE_ID,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": number_or_zero(usage.get("prompt_tokens")),
            "output_tokens": number_or_zero(usage.get("completion_tokens")),
        },
    }


# ---------------------------------------------------------------- the gateway


class Answer:
    """One upstream completion, collected whole: text, stop, tools, usage."""

    def __init__(self) -> None:
        self.text = ""
        self.finish_reason: str | None = None
        self.tool_calls: list[Json] = []
        self.input_tokens: int | float | None = None
        self.output_tokens: int | float | None = None
        self.saw_finish = False

    def usage(self, request_chars: int) -> tuple[int, int]:
        """Whole token counts: the provider's, else a deterministic fall-back.

        The wire requires an input count of at least 0 and an output count
        of at least 1; the fall-back derives both from text lengths, so it
        is the same on a live pass and its replay.
        """
        in_tokens = self.input_tokens
        out_tokens = self.output_tokens
        if not is_number(in_tokens) or in_tokens < 0:
            in_tokens = max(0, request_chars // 4)
        if not is_number(out_tokens) or out_tokens < 1:
            out_tokens = max(1, len(self.text) // 4)
        return int(in_tokens), int(out_tokens)


def error_code_of(raw: bytes) -> str:
    """The gateway error body's code (or type), truncated to 64 chars."""
    try:
        value = json.loads(raw.decode("utf-8", "replace"))
    except (json.JSONDecodeError, ValueError):
        return ""
    if not is_record(value) or not is_record(value.get("error")):
        return ""
    error = value["error"]
    code = error.get("code") if isinstance(error.get("code"), str) else error.get("type")
    return code[:64] if isinstance(code, str) else ""


def chat_completions_url() -> str:
    """The gateway's Chat Completions route from $OPENAI_BASE_URL."""
    if GATEWAY_URL == "":
        return ""
    return f"{GATEWAY_URL}/chat/completions" if GATEWAY_URL.endswith("/v1") else f"{GATEWAY_URL}/v1/chat/completions"


def dial(model: str, body: Json) -> tuple[int, bytes]:
    """POST one request to the gateway; returns (status, raw body bytes)."""
    url = chat_completions_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    payload = dict(body)
    payload["model"] = model
    if not is_number(payload.get("max_tokens")):
        payload["max_tokens"] = DEFAULT_MAX_TOKENS
    elif payload["max_tokens"] > MAX_OUTPUT_TOKENS:
        payload["max_tokens"] = MAX_OUTPUT_TOKENS
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"}
    if body.get("stream") is True:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SEC) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    # URLError / TimeoutError: the gateway could not be reached at all.
    # The caller records it as a 502 refusal and fails over.


def collect_answer(status: int, raw: bytes, stream: bool) -> Answer | None:
    """A gateway answer as one Answer, or None when it is unusable."""
    answer = Answer()
    if not (200 <= status < 300):
        return None
    if not stream:
        try:
            value = json.loads(raw.decode("utf-8", "replace"))
        except (json.JSONDecodeError, ValueError):
            return None
        if not is_record(value):
            return None
        choices = value.get("choices") if isinstance(value.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else None
        if choice is None:
            return None
        message = choice.get("message") if is_record(choice.get("message")) else {}
        answer.text = message["content"] if isinstance(message.get("content"), str) else text_of(message.get("content"))
        finish = choice.get("finish_reason")
        answer.finish_reason = finish if isinstance(finish, str) else None
        answer.saw_finish = answer.finish_reason is not None
        calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
        answer.tool_calls = [c for c in calls if is_record(c)]
        usage = value.get("usage") if is_record(value.get("usage")) else {}
        answer.input_tokens = number_or_null(usage.get("prompt_tokens"))
        answer.output_tokens = number_or_null(usage.get("completion_tokens"))
        return answer
    # Streaming: consume the SSE frames and accumulate them.
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    tool_parts: dict[int, Json] = {}
    for frame in text.split("\n\n"):
        for line in frame.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "" or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            usage = chunk.get("usage") if is_record(chunk.get("usage")) else None
            if usage is not None:
                answer.input_tokens = number_or_null(usage.get("prompt_tokens"))
                answer.output_tokens = number_or_null(usage.get("completion_tokens"))
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            if not choices:
                continue
            choice = choices[0] if is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get("delta") if is_record(choice.get("delta")) else {}
            if isinstance(delta.get("content"), str):
                answer.text += delta["content"]
            for call in delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []:
                if not is_record(call):
                    continue
                index = call.get("index") if is_number(call.get("index")) else 0
                part = tool_parts.setdefault(int(index), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if isinstance(call.get("id"), str) and call["id"] != "":
                    part["id"] = call["id"]
                fn = call.get("function") if is_record(call.get("function")) else {}
                if isinstance(fn.get("name"), str) and fn["name"] != "":
                    part["function"]["name"] = fn["name"]
                if isinstance(fn.get("arguments"), str):
                    part["function"]["arguments"] += fn["arguments"]
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                answer.finish_reason = finish
                answer.saw_finish = True
    answer.tool_calls = [tool_parts[k] for k in sorted(tool_parts)]
    return answer


def answer_is_valid(answer: Answer, token: str | None) -> bool:
    """Whether an upstream answer may serve the prompt that asked for it.

    The text must be non-empty and, when the request carried a token,
    must carry that token back: the scorer credits a prompt only when
    the provider's own answer body carries it. An answer that fails
    here is a failure of the model, not of the prompt, and the next
    model is tried.
    """
    if answer.text == "":
        return False
    if token is not None and token not in answer.text:
        return False
    return True


# Trial-scoped health: a model that refused or answered invalidly once is
# not dialled again in this trial. The set is a function of the gateway's
# answers alone, so a replay takes the same path.
unhealthy: set[str] = set()


def serve(body: Json, api: str) -> tuple[str | None, Answer | None, list[Json]]:
    """Serve one request: (model id, answer, failure trail).

    Tries the eligible models in rating order, at most three attempts.
    A refusal, an unreachable gateway, or an answer that fails validation
    records a failure row, marks the model unhealthy, and moves on.
    """
    token = token_in_request(body)
    requires = requirements_for(body)
    candidates = ranked_candidates(requires)
    failures: list[Json] = []
    request_chars = sum(len(t) for t in message_texts(body))
    stream = body.get("stream") is True
    attempts = 0
    for model in candidates:
        if model["id"] in unhealthy:
            continue
        if attempts >= MAX_ATTEMPTS_PER_PROMPT:
            break
        attempts += 1
        upstream_body = (
            anthropic_to_openai(body, model["id"])
            if api == "anthropic"
            else {**body, "model": model["id"], **({"stream_options": {"include_usage": True}} if stream else {})}
        )
        try:
            status, raw = dial(model["id"], upstream_body)
        except Exception:
            failures.append({"status": 502, "code": "gateway_unreachable", "model": model["id"]})
            unhealthy.add(model["id"])
            continue
        if not (200 <= status < 300):
            failures.append({"status": status, "code": error_code_of(raw), "model": model["id"]})
            unhealthy.add(model["id"])
            continue
        answer = collect_answer(status, raw, stream)
        if answer is None or not answer_is_valid(answer, token):
            failures.append({"status": status, "code": "invalid_answer", "model": model["id"]})
            unhealthy.add(model["id"])
            continue
        return model["id"], answer, failures
    return None, None, failures


# ---------------------------------------------------------------- answers in the client's wire


def openai_completion(model: str, answer: Answer, request_chars: int) -> Json:
    """A collected answer as an OpenAI Chat Completions object."""
    in_tokens, out_tokens = answer.usage(request_chars)
    message: Json = {"role": "assistant", "content": answer.text}
    if answer.tool_calls:
        message["tool_calls"] = answer.tool_calls
    finish = answer.finish_reason
    if finish is None:
        finish = "tool_calls" if answer.tool_calls else "stop"
    return {
        "id": COMPLETION_ID,
        "object": "chat.completion",
        "created": CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens, "total_tokens": in_tokens + out_tokens},
    }


def openai_stream_events(model: str, answer: Answer, request_chars: int) -> list[str]:
    """A collected answer as canonical OpenAI SSE: content, finish, [DONE]."""
    in_tokens, out_tokens = answer.usage(request_chars)
    finish = answer.finish_reason
    if finish is None:
        finish = "tool_calls" if answer.tool_calls else "stop"
    base = {"id": COMPLETION_ID, "object": "chat.completion.chunk", "created": CREATED, "model": model}
    events: list[str] = []
    first: Json = {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer.text}, "finish_reason": None}]}
    if answer.tool_calls:
        first["choices"][0]["delta"]["tool_calls"] = answer.tool_calls
    events.append(f"data: {json.dumps(first, separators=(',', ':'), ensure_ascii=False)}\n\n")
    last: Json = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens, "total_tokens": in_tokens + out_tokens},
    }
    events.append(f"data: {json.dumps(last, separators=(',', ':'), ensure_ascii=False)}\n\n")
    events.append("data: [DONE]\n\n")
    return events


def anthropic_completion(model: str, answer: Answer, request_chars: int) -> Json:
    """A collected answer as an Anthropic Messages object."""
    in_tokens, out_tokens = answer.usage(request_chars)
    content: list[Json] = []
    if answer.text != "":
        content.append({"type": "text", "text": answer.text})
    for call in answer.tool_calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else "toolu_unknown",
                "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                "input": arguments_of(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    finish = answer.finish_reason
    if finish is None:
        finish = "tool_calls" if answer.tool_calls else "stop"
    return {
        "id": MESSAGE_ID,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason_of(finish),
        "stop_sequence": None,
        "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
    }


def sse_event(event: str, data: object) -> str:
    """One named server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def anthropic_stream_events(model: str, answer: Answer, request_chars: int) -> list[str]:
    """A collected answer as the canonical Anthropic event sequence."""
    in_tokens, out_tokens = answer.usage(request_chars)
    finish = answer.finish_reason
    if finish is None:
        finish = "tool_calls" if answer.tool_calls else "stop"
    stop_reason = stop_reason_of(finish)
    events: list[str] = [
        sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": MESSAGE_ID,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": in_tokens, "output_tokens": 0},
                },
            },
        )
    ]
    index = 0
    if answer.text != "":
        events.append(
            sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            )
        )
        events.append(
            sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": answer.text}},
            )
        )
        events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for call in answer.tool_calls:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        events.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call["id"] if isinstance(call.get("id"), str) and call["id"] != "" else f"toolu_{index}",
                        "name": fn["name"] if isinstance(fn.get("name"), str) else "",
                        "input": arguments_of(fn.get("arguments")),
                    },
                },
            )
        )
        events.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    events.append(
        sse_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": out_tokens}},
        )
    )
    events.append(sse_event("message_stop", {"type": "message_stop"}))
    return events


def openai_model_list() -> Json:
    """GET /v1/models on an OpenAI-shaped request."""
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": CREATED, "owned_by": m["provider"]}
            for m in catalog_models()
        ],
    }


def anthropic_model_list() -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for m in catalog_models()
        ],
        "has_more": False,
    }


def openai_error(message: str, error_type: str) -> Json:
    return {"error": {"message": message, "type": error_type, "code": None}}


def anthropic_error(error_type: str, message: str) -> Json:
    return {"type": "error", "error": {"type": error_type, "message": message}}


# ---------------------------------------------------------------- the HTTP server


def log(message: str) -> None:
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires."""

    protocol_version = "HTTP/1.1"
    server_version = "makefun-aggregator-py/1"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error_body(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error_body(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def trail_headers(self, failures: list[Json]) -> list[tuple[str, str]]:
        """The failure-trail header, when the prompt recovered from any."""
        if not failures:
            return []
        trail = failures[:MAX_TRAIL_ROWS]
        return [(FAILURES_HEADER, json.dumps(trail, separators=(",", ":"), ensure_ascii=False))]

    def send_json(self, status: int, body: object, extra_headers: list[tuple[str, str]] | None = None) -> None:
        raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for name, value in extra_headers or []:
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type))

    def send_stream(self, events: list[str], extra_headers: list[tuple[str, str]] | None = None) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        for name, value in extra_headers or []:
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True
        for event in events:
            self.wfile.write(event.encode("utf-8"))
            self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_model_list())
        else:
            self.send_json(200, openai_model_list())

    def refused(self, wire: str, failures: list[Json]) -> None:
        """No model could serve the prompt: a 502 in the client's wire."""
        detail = ""
        if failures:
            last = failures[-1]
            detail = f" (last refusal: HTTP {last['status']} from {last['model']})"
        if wire == "anthropic":
            self.send_json(502, anthropic_error("api_error", f"no free model could serve this prompt{detail}"), self.trail_headers(failures))
        else:
            self.send_json(502, openai_error(f"no free model could serve this prompt{detail}", "upstream_error"), self.trail_headers(failures))

    def chat(self) -> None:
        """POST /v1/chat/completions: the OpenAI wire, answered live."""
        body = self.read_json()
        if body is None:
            self.send_error_body("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        model, answer, failures = serve(body, "openai")
        if model is None or answer is None:
            self.refused("openai", failures)
            return
        request_chars = sum(len(t) for t in message_texts(body))
        if body.get("stream") is True:
            self.send_stream(openai_stream_events(model, answer, request_chars), self.trail_headers(failures))
        else:
            self.send_json(200, openai_completion(model, answer, request_chars), self.trail_headers(failures))

    def messages(self) -> None:
        """POST /v1/messages: the Anthropic wire, translated and answered back."""
        body = self.read_json()
        if body is None:
            self.send_error_body("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        model, answer, failures = serve(body, "anthropic")
        if model is None or answer is None:
            self.refused("anthropic", failures)
            return
        request_chars = sum(len(t) for t in message_texts(body))
        if body.get("stream") is True:
            self.send_stream(anthropic_stream_events(model, answer, request_chars), self.trail_headers(failures))
        else:
            self.send_json(200, anthropic_completion(model, answer, request_chars), self.trail_headers(failures))


# ---------------------------------------------------------------- register and poll


def register(port: int) -> None:
    """POST /register with the port and the one catalog document."""
    body = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    request = urllib.request.Request(
        f"{ENV_URL}/register",
        data=body.encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode('utf-8', 'replace')[:500]}") from err
    if not (200 <= status < 300):
        raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}")
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get("requests")):
        log(f"registered: the trial holds {reply['requests']} prompts")


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f"{ENV_URL}/work", timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_record(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as a clean exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(CATALOG_TEXT + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if chat_completions_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log(f"listening on 0.0.0.0:{port} with {len(catalog_models())} free models")
        register(port)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
