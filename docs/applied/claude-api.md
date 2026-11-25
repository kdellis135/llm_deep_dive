---
seo_title: "Building with Claude: The Claude API in Practice"
description: "The Claude Messages API from first request to production: model choice, streaming, thinking and effort, tool use, structured outputs, prompt caching, citations, batches and cost."
---

# 28. Building with Claude

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 19](../inference/efficient.md) (KV and prefix caching), [Ch. 22](../frontier/reasoning.md) (thinking), [Ch. 25–27](prompting.md). **You will learn:** how to choose a Claude model; the Messages API from the first request to production: system prompts, multi-turn conversations, streaming, adaptive thinking and effort, tool use (client and server tools, strict schemas, the tool runner), structured outputs, prompt caching (and why it works), documents, images and citations, batches, token counting, long context and compaction, refusals and fallbacks, errors and retries; how to estimate cost; and which agent-building option to pick.
    **Why it matters:** this chapter connects the theory of Parts III–VI to the concrete API you'd use to build products. Each feature maps to something you now understand from first principles: caching is prefix KV reuse (§19.4), effort is test-time compute (§22.7), structured outputs are constrained decoding (§18.7).

!!! warning "APIs change fast"
    Model names, prices and parameters below reflect Anthropic's API as of late 2026. Always check the current [Claude API documentation](https://platform.claude.com/docs) and [pricing](https://platform.claude.com/docs/en/about-claude/pricing) before building. The `GET /v1/models` endpoint reports each model's context window and capabilities live.

---

## 28.1 Choosing a model

| Model | API ID | Context | Input / output ($ per M tokens) | Use for |
|---|---|---|---|---|
| Claude Fable 5.2 | `claude-fable-5-2` | 1M | 8 / 40 | the hardest reasoning and long-horizon agentic work |
| Claude Opus 5.5 | `claude-opus-5-5` | 1M | 4 / 20 | default for complex tasks: coding, agents, analysis |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | 1M | 2 / 10 | strong everyday model for high-volume coding, agent and enterprise work |
| Claude Haiku 5.5 | `claude-haiku-5-5` | 1M | 0.10 / 0.50 (≤100K-token prompts) | fast, cheap: classification, extraction, routing, sub-agents |

Output tokens cost 5× input tokens, reflecting decode's memory-bound cost (§19.1). A sensible process: prototype with the most capable model you can afford to establish what's achievable, build an evaluation set (Ch. 29), then see whether a cheaper model, or the same model at lower effort (§28.5), holds quality on it. Judge cost per *completed task*, not per request.

## 28.2 The first request

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...      # or run `ant auth login`
```

```python
import anthropic

client = anthropic.Anthropic()                 # reads credentials from the environment
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    messages=[{"role": "user", "content": "Explain KV caching in two sentences."}],
)
for block in response.content:                 # a list of typed content blocks
    if block.type == "text":
        print(block.text)
print(response.stop_reason, response.usage.input_tokens, response.usage.output_tokens)
```

Everything goes through one endpoint, `POST /v1/messages`. The response's `content` is a list of **blocks** (`text`, `thinking`, `tool_use`, …), so always check `.type`. `stop_reason` tells you why generation ended: `end_turn` (done), `max_tokens` (truncated: raise the limit), `tool_use` (run the tools and continue), `pause_turn` (a long server-side tool turn paused: send it back to resume), or `refusal` (§28.11). `usage` reports billed tokens.

## 28.3 System prompts and conversations

```python
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    system="You are a concise assistant for Acme Cloud's engineering team. Answer in plain English.",
    messages=[
        {"role": "user", "content": "What's a 502 error?"},
        {"role": "assistant", "content": "A 502 means a gateway got an invalid response from an upstream server."},
        {"role": "user", "content": "How do I debug one on our load balancer?"},
    ],
)
```

The API is **stateless**: send the whole conversation each time. The system prompt carries operator instructions (role, rules, context, Ch. 25). On current models you can also append a `{"role": "system", ...}` message *inside* `messages` to add an operator instruction mid-conversation without changing the cached prefix (§28.7).

## 28.4 Streaming

Long responses take seconds to minutes. Stream them, both for the user experience and to avoid HTTP timeouts on large `max_tokens`:

```python
with client.messages.stream(
    model="claude-opus-5-5",
    max_tokens=64000,
    messages=[{"role": "user", "content": "Write a design doc for a rate limiter."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()          # the complete Message, with usage
```

Time to first token is dominated by prefill and queueing; tokens then arrive at the decode rate (§19.1).

## 28.5 Thinking and effort

Current Claude models use **adaptive thinking**: the model decides when and how much to reason before answering (§22.7). You control depth with **effort**:

```python
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    thinking={"type": "adaptive", "display": "summarized"},   # show a summary of the reasoning
    output_config={"effort": "high"},                          # low | medium | high | xhigh | max
    messages=[{"role": "user", "content": "Find the bug in this function: ..."}],
)
for block in response.content:
    if block.type == "thinking":
        print("[reasoning summary]", block.thinking)
    elif block.type == "text":
        print(block.text)
```

- **Effort** is a test-time compute dial (§22.2). Lower effort means fewer thinking tokens, fewer and more consolidated tool calls, and terser output. Higher effort is worth it for hard coding, math and agentic tasks. Claude Opus 5.5 defaults to `medium`; set it explicitly and tune it per route against your evaluation set.
- Thinking tokens are billed as output tokens. The raw chain of thought is never returned: `display: "summarized"` gives a readable summary; the default `"omitted"` returns empty thinking text (so a streaming UI shows a pause before the answer).
- On current models, thinking can't be configured with a fixed token budget (`budget_tokens`, used by older models, is rejected), and on Claude Opus 5.5 it can't be disabled. Use low effort for simple tasks.
- **Sampling parameters** (`temperature`, `top_p`, `top_k`) aren't accepted by current models. Control output through the prompt, effort and structured outputs instead of the decoding knobs of Ch. 18.
- **No assistant prefill**: you can't start the assistant's turn with your own text on current models. Use structured outputs or instructions to control the format.

## 28.6 Tool use

Tools follow the loop of §27.2. A few API specifics:

- **`strict: true`** on a tool guarantees its arguments match the JSON schema exactly (constrained decoding, §18.7). The schema needs `additionalProperties: false` and `required`.
- **Parallel tool use** is on by default: one response may contain several `tool_use` blocks. Return all results in one user message.
- **Tool choice**: `auto` is the default and the right choice. On Claude Opus 5.5 and Sonnet 5.5, *forcing* a tool call (`tool_choice` of type `any` or `tool`) returns an error. Ask for the tool in the prompt, or use structured outputs if you only wanted JSON back.
- **Tool runner**: `client.beta.messages.tool_runner(...)` with `@beta_tool`-decorated functions runs the loop for you (§27.2).
- **Server tools** run on Anthropic's side, with results in the same response:

```python
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
    messages=[{"role": "user", "content": "What changed in the latest PyTorch release?"}],
)
```

Others include web fetch (`web_fetch_20260209`), code execution (a sandbox for running Python and shell, e.g. for data analysis and generating files), and computer use. The Claude API can also connect to remote **MCP** servers (§27.5).

## 28.7 Structured outputs

When code consumes the output, enforce a schema instead of parsing free text. With Pydantic:

```python
from pydantic import BaseModel

class Triage(BaseModel):
    category: str            # e.g. "billing" | "bug" | "outage" | ...
    urgency: str             # "low" | "medium" | "high"
    reason: str

response = client.messages.parse(
    model="claude-opus-5-5",
    max_tokens=16000,
    messages=[{"role": "user", "content": f"Triage this ticket:\n<ticket>{ticket}</ticket>"}],
    output_format=Triage,
)
triage = response.parsed_output          # a validated Triage instance
```

With `messages.create`, the equivalent is `output_config={"format": {"type": "json_schema", "schema": {...}}}`. Either way the output is guaranteed to parse. (It's still worth validating the *values*, since a schema can't express everything, like whether the category is correct.)

## 28.8 Prompt caching

Every request recomputes the KV cache for its whole prompt (prefill, §19.1) unless the provider can reuse it. **Prompt caching** is exactly the prefix caching of §19.4: if a request begins with the same tokens as a recent one, the stored KV cache for that prefix is reused.

```python
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    system=[{
        "type": "text",
        "text": LONG_INSTRUCTIONS_AND_REFERENCE_DOCS,     # stable, large
        "cache_control": {"type": "ephemeral"},          # cache everything up to here (5-min TTL)
    }],
    messages=[{"role": "user", "content": user_question}],   # varies per request
)
u = response.usage
print(u.cache_creation_input_tokens, u.cache_read_input_tokens, u.input_tokens)
```

You can also put one top-level `cache_control={"type": "ephemeral"}` on the request to cache automatically up to the last cacheable block, or set `"ttl": "1h"` for a longer lifetime.

**Economics.** Cache reads cost about **10%** of the normal input price; writing to the cache costs about 25% more than normal input (5-minute TTL). So a prefix reused even once or twice pays for itself, and long system prompts, tool definitions, documents and conversation histories get dramatically cheaper. Time to first token also drops, because the prefill for the cached part is skipped.

**Design rules** (they follow from how prefix caching works):

1. The prompt is rendered in the order **tools → system → messages**. Any change to an earlier byte invalidates everything after it.
2. Put stable content first, volatile content last. Never put timestamps, request IDs or per-user data in the system prompt.
3. Keep tool definitions and JSON serialization deterministic (same order every time).
4. In multi-turn conversations, the history itself is a growing shared prefix. Append, don't rewrite.
5. Verify: if `cache_read_input_tokens` stays 0 across similar requests, something in the prefix is changing. Prefixes below a model-dependent minimum length aren't cached.

## 28.9 Documents, images and citations

Claude reads images and PDFs directly:

```python
import base64
pdf = base64.standard_b64encode(open("contract.pdf", "rb").read()).decode()
response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    messages=[{"role": "user", "content": [
        {"type": "document",
         "source": {"type": "base64", "media_type": "application/pdf", "data": pdf},
         "citations": {"enabled": True}},
        {"type": "text", "text": "List every termination clause, with page references."},
    ]}],
)
```

With citations enabled, the answer's text blocks carry `citations` pointing to the cited passages (page ranges for PDFs, character ranges for text), which is ideal for RAG (§26.7). Images go in `image` blocks (base64 or URL). The **Files API** (`client.files.upload(...)`) uploads a file once and references it by `file_id` across requests. Images and PDF pages are converted to tokens and count against the context and the bill (§20.7).

## 28.10 Batches, token counting, long context

- **Message Batches**: submit up to 100,000 requests at once for asynchronous processing at **50% of the price**, usually within an hour (at most 24 hours). Ideal for evaluations, data labeling and bulk extraction:

    ```python
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(requests=[
        Request(custom_id=f"doc-{i}", params=MessageCreateParamsNonStreaming(
            model="claude-opus-5-5", max_tokens=2000,
            messages=[{"role": "user", "content": f"Summarize:\n{doc}"}]))
        for i, doc in enumerate(docs)
    ])
    # poll client.messages.batches.retrieve(batch.id) until processing_status == "ended", then:
    for result in client.messages.batches.results(batch.id):   # any order: key by custom_id
        if result.result.type == "succeeded":
            print(result.custom_id, result.result.message.content[0].text)
    ```

- **Token counting**: `client.messages.count_tokens(model=..., messages=..., system=..., tools=...)` returns the exact input token count before you send a request. Use it for cost estimates and to check context limits. Don't estimate Claude token counts with another provider's tokenizer (§10.8).
- **Long context and compaction**: current models accept up to 1M tokens. For conversations or agent loops that would exceed it, server-side **compaction** (beta) summarizes earlier context automatically when a threshold is reached. You must append the full `response.content` (which contains the compaction block) back into the history.

## 28.11 Refusals and fallbacks

If a request is declined by Claude's safety classifiers, the API returns normally with `stop_reason == "refusal"` and a `stop_details` object (with a `category` such as `"cyber"` or `"bio"`). Always check `stop_reason` before reading content. For current models you can opt into **server-side fallbacks**, so that a declined request is automatically re-run on a fallback model within the same call:

```python
response = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=16000,
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",                  # the API picks an appropriate fallback model
    messages=[{"role": "user", "content": question}],
)
if response.stop_reason == "refusal":     # the whole fallback chain declined
    print("Declined:", response.stop_details.category if response.stop_details else None)
```

## 28.12 Errors and retries

The SDK automatically retries connection errors, 408, 409, 429 (rate limit) and 5xx errors with exponential backoff (2 retries by default; configure with `max_retries`). Handle the remaining errors specifically rather than with one broad `except`:

```python
try:
    response = client.messages.create(...)
except anthropic.BadRequestError as e:        # 400: fix the request; don't retry
    log.error("bad request: %s", e.message)
except anthropic.RateLimitError as e:         # 429 after SDK retries: back off, queue, or shed load
    retry_after = e.response.headers.get("retry-after")
except anthropic.APIStatusError as e:         # other HTTP errors (5xx after retries, ...)
    log.error("API error %s (request id %s)", e.status_code, e.request_id)
except anthropic.APIConnectionError:          # network failure
    ...
```

Log the request ID (`response._request_id`) of every call: it's what support needs to investigate an issue.

## 28.13 Estimating cost

A support assistant handles 10,000 conversations per day on Claude Sonnet 5.5. Each request has a 2,500-token cached prefix (instructions, tools, policies), 500 tokens of new input, and 400 output tokens.

| | Without caching | With caching |
|---|---|---|
| Prefix (2,500 tokens) | $2{,}500\times\$2/\text{M} = \$0.0050$ | $2{,}500\times\$0.10/\text{M} = \$0.00025$ |
| New input (500) | $\$0.0010$ | $\$0.0010$ |
| Output (400) | $400\times\$10/\text{M} = \$0.0040$ | $\$0.0040$ |
| **Per request** | **\$0.0100** | **\$0.00525** |
| **Per day (10K)** | **\$100** | **≈\$53** (plus small cache-write costs) |

Output dominates once caching is on, so the next levers are concise outputs (instructions, structured formats), the right effort level, and, for anything that isn't interactive, the Batch API at half price.

## 28.14 Building agents with Claude

| Option | You write | Anthropic provides | Choose when |
|---|---|---|---|
| Messages API, manual loop | the loop and the tools | the model | you want full control (§27.2) |
| Messages API, tool runner | the tool functions | the model + the SDK's loop | you host the agent and don't want to write the loop |
| **Claude Managed Agents** (beta) | agent configuration and custom tools | the loop *and* a hosted sandbox per session (bash, files, code), sessions, scheduling | you want a hosted agent with minimal infrastructure |
| **Claude Agent SDK** | a prompt and options | the Claude Code harness as a library: file and shell tools, context management, sub-agents, hooks | a coding or file-system agent running on your own infrastructure |

**Claude Code** is Anthropic's agentic coding tool (terminal, IDE, desktop and web), built on the same harness as the Agent SDK. It's both a product and a reference design for long-running agents: tools for reading, searching and editing files and running commands, context compaction, memory files (`CLAUDE.md`), sub-agents, hooks and permission controls (§27.8–27.9).

Claude is also available through Amazon Bedrock, Google Cloud Vertex AI and Microsoft Foundry, with platform-specific client classes in the SDK (e.g. `AnthropicBedrockMantle`, `AnthropicVertex`, `AnthropicFoundry`). Feature availability can differ by platform.

## Key takeaways

- Choose the most capable model that meets cost and latency targets, measured on your own evaluation set; tune effort before switching models.
- Messages API: content is a list of typed blocks; check `stop_reason`; the API is stateless; stream long outputs.
- Adaptive thinking with an effort level is the test-time compute dial. Current models don't take sampling parameters or prefills.
- Tools: `strict: true` schemas, parallel results in one message, `auto` tool choice; server tools for web search and code execution.
- Structured outputs guarantee parseable JSON. Prompt caching is prefix KV reuse: stable content first, reads at ~10% of input price.
- Batches halve the price for offline work. Count tokens with the API. Handle refusals, use fallbacks, and handle errors specifically, logging request IDs.

## Exercises

1. Your cache hit rate is 0 even though every request starts with the same 5,000-token system prompt. List four possible causes.
2. Estimate the daily cost of classifying 1 million short support tickets (300 input and 20 output tokens each) with Claude Haiku 5.5, with and without the Batch API.
3. Why are output tokens priced higher than input tokens? Use Ch. 19.
4. A product needs a guaranteed JSON object from every response. Compare (a) instructions only, (b) a strict tool, and (c) structured outputs.
5. Design the request layout (tools, system, messages, cache breakpoints) for a coding assistant that loads a 100K-token repository summary and holds long conversations.
6. **Code:** build the support-triage pipeline from §25.7 with `messages.parse`, run it on 50 labeled tickets via the Batch API, and report accuracy per category.

??? success "Solutions"
    1. A timestamp or request ID in the system prompt; non-deterministic tool or JSON key ordering; a different tool set or model between requests; the prefix is below the minimum cacheable length; requests more than the TTL apart; the `cache_control` breakpoint placed after a varying block.
    2. Without batches: $10^6\times(300\times0.10+20\times0.50)/10^6 = \$40$. With the Batch API: \$20.
    3. Prefill processes input tokens in parallel (compute-bound, efficient), while each output token needs its own sequential decode step that reads all the weights (memory-bound), so output tokens use far more GPU time each.
    4. (a) usually works but can fail; (b) guarantees schema-valid arguments but is a workaround; (c) guarantees schema-valid output directly and is the intended mechanism.
    5. Tools (stable) → system prompt (stable instructions) → the repository summary in a cached block → conversation history appended turn by turn, with a breakpoint after the summary and another near the end of the history.
    6. Expect high accuracy on clear categories; read the errors to find ambiguous category definitions and fix them in the prompt.

## References

- Anthropic, [Claude API documentation](https://platform.claude.com/docs): Messages API, extended/adaptive thinking, tool use, structured outputs, prompt caching, citations, batch processing, token counting.
- Anthropic, [Python SDK](https://github.com/anthropics/anthropic-sdk-python).
- Anthropic, [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk) and [Claude Code](https://code.claude.com/docs).
- Anthropic, [Prompt engineering guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview).
- Anthropic, [Pricing](https://platform.claude.com/docs/en/about-claude/pricing).
