# Free-model aggregator — make.fun `free-model-aggregator-02`

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/sanschiko3-dotcom/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/sanschiko3-dotcom/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Bounty: [The best live free-model aggregator](https://make.fun/bounty/free-model-aggregator-02) — $500 in USDC.

One server, two client wires, live free models behind every answer.

The server reads the brief at `$MAKEFUN_PROMPT_FILE` (the free-model list
with the house's ratings and capabilities, plus the six prompts' manifest),
registers itself with the env at `$MAKEFUN_ENV_URL`, and answers every
prompt with a real completion from one of the brief's free OpenRouter
models, dialled through the grader's gateway at `$OPENAI_BASE_URL` on the
trial's key. It serves:

- `POST /v1/chat/completions` — OpenAI wire, JSON and streaming
- `POST /v1/messages` — Anthropic wire, JSON and streaming
- `GET /v1/models` — the model list in the shape of the auth header it got
  (`Authorization: Bearer` → OpenAI shape, `x-api-key` → Anthropic shape)

`python3 main.py --list` prints the catalog document and exits 0. Standard
library only; Python 3.13, no installs.

## Run it

```sh
python3 main.py --list   # print the catalog and exit
python3 main.py          # serve (needs $MAKEFUN_PROMPT_FILE, $MAKEFUN_ENV_URL, $OPENAI_BASE_URL)
```

## How it routes (the scoring idea)

**Rating-first, requirements-aware.** The requirements of each request are
read off the request body itself — a `response_format` means JSON mode, a
non-empty `tools` array means tools, an image part means vision — and the
models that meet them are tried in rating order (`rating_milli` descending,
id ascending as the deterministic tie-break). The brief's manifest is not
needed to route, so fresh instances with different requirements still route
correctly. In the practice briefs this picks Nemotron 3.5 Super VL (1303,
the best JSON-capable model) for JSON-mode prompts and Inkling 2 Small
(1475, the best-rated free model) for everything else.

**Validated failover.** Every upstream answer is validated before it is
served: it must be a 2xx answer whose text is non-empty and carries the
prompt's token, which is read out of the request's own messages. A model
whose answer fails validation is treated exactly like a model that
refused — the failure is recorded, the model is marked unhealthy for the
rest of the trial, and the next best eligible model is tried. This is what
makes the top-rated model safe to try first: if the best model cannot
actually serve a prompt (it refuses, it rate-limits, or its visible text
never carries the token), the prompt still lands on the next best model
instead of being lost. Health is a function of the gateway's answers only,
so a replay of the same trial takes the same path. At most three attempts
per prompt (worst case 18 calls in a trial, cap 60; the usual case is
exactly one call per prompt, which is also the tie-break).

**Buffered canonical streams.** The upstream SSE stream is consumed whole,
its text, tool calls, finish reason and usage collected, and only then is
the client's stream written: one content chunk, one finish chunk carrying
usage, then `[DONE]` on the OpenAI wire; `message_start`, one text block,
`message_delta` with the stop reason and output tokens, `message_stop` on
the Anthropic wire. Buffering guarantees the event order the wire checks
demand (exactly one finish, nothing after `[DONE]` / `message_stop`) and
lets the failure trail be reported in the response headers.

**Deterministic.** Nothing in any answer depends on the wall clock or on
unseeded randomness. Response ids are fixed strings, `created` comes from
`MAKEFUN_TRIAL_NOW` (or a constant), and usage, when the provider omits
it, is derived from the request and answer text lengths. A replay of the
same trial therefore leaves the same state export.

**Failure trail.** Every refusal or invalid answer on the way to a served
prompt is reported in the `x-makefun-upstream-failures` header (at most 8
rows, in the order the calls failed) — the only way the failover column
can see a refusal the server recovered from.

## The catalog

One provider, `openrouter`, exactly the fourteen free models the brief
lists, each as its dialable id (`openrouter/<upstream id>:free`), with
`base_url` `https://openrouter.ai/api/v1` (the public endpoint, never the
gateway's trial address), `key_env` `OPENROUTER_API_KEY`, `wire` `openai`.
`--list` prints the same document the register body carries, byte for byte.

## Local verification (mock trials)

No OpenRouter key is needed for these: the kit's real env server
(`program/env/server.mjs`) was driven against a stub gateway that answers
with the prompt's own token, and the kit's own scorer
(`program/scorer/score.mjs`) scored the resulting state + a reconstructed
cassette. Five runs, all 6/6 prompts valid, 39/39 wire checks passed,
server exit 0:

| trial | stub mode | score | quality | served | wire | failover | catalog | calls |
|---|---|---|---|---|---|---|---|---|
| 0 | all models answer | **100** | 100 | 100 | 100 | 100 | 100 | 6 |
| 0 | Inkling 429s | **94** | 90 | 100 | 100 | 100 | 100 | 6 |
| 0 | Inkling answers empty | **94** | 90 | 100 | 100 | 100 | 100 | 7 |
| 1 | all models answer | **100** | 100 | 100 | 100 | 100 | 100 | 6 |
| 2 | all models answer | **100** | 100 | 100 | 100 | 100 | 100 | 6 |

Notes:

- Score 100 in the "all answer" runs is the design target: the best-rated
  eligible model serves every prompt, so quality is 100, not the 90 that
  the leaderboard's current top (194 entries at 94) settles for by serving
  Super VL everywhere.
- Score 94 in the two Inkling-failure runs is exactly the leaderboard's
  current ceiling: the probe costs one extra call, failover lands on
  Super VL, and every other column stays at 100. The entry degrades to
  the field's best instead of losing prompts.
- The live score depends on the real gateway and on the entrant's pushed
  OpenRouter key (10+ credits); the stub cannot prove what the real
  Inkling 2 Small returns. The probe-and-failover design is what makes
  that uncertainty safe either way.

## Files

- `main.py` — the whole server (entrypoint)
- `LICENSE` — MIT (`SPDX-License-Identifier: MIT`)
- `.makefun/owner.json` — payout-wallet claim file

---

<sub>Entered on [make.fun](https://make.fun/?ref=s24gd3wm&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
