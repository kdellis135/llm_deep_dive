---
seo_title: "LLM Tool Use & Agents: Function Calling, ReAct, MCP"
description: "How function calling works, the agent loop and ReAct, workflows vs agents, tool design, the Model Context Protocol (MCP), multi-step reliability, memory and multi-agent systems."
---

# 27. Tool Use & Agents

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 18](../inference/decoding.md) (constrained decoding), [Ch. 22](../frontier/reasoning.md) (reasoning, RL), [Ch. 25](prompting.md) (context engineering), [Ch. 26](rag.md). **You will learn:** how function calling works mechanically; the agent loop as a policy acting in an environment (ReAct); workflows vs agents and the five standard workflow patterns; how to design tools for models; the Model Context Protocol (MCP); server-side tools (web search, code execution, computer use); the reliability math of multi-step tasks; memory, context management and multi-agent systems; safety and permissions; and how agents are evaluated.
    **Why it matters:** tool use turns an LLM from a text generator into a system that can look things up, run code, operate software and complete multi-step tasks. Coding agents, research agents and customer-service agents are some of the highest-value applications of LLMs today, and they're built from the few ideas in this chapter.

!!! watch "Read"
    - Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (2024). The most-cited practical guide: start simple, prefer workflows, use agents where needed.
    - Anthropic, [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) (2025) and [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (2025).
    - S. Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (2022).
    - [Model Context Protocol](https://modelcontextprotocol.io/) specification and documentation.

---

## 27.1 How function calling works

You describe tools to the model as names, descriptions and JSON-schema parameters. The provider serializes these definitions into the model's context (in a format the model was trained on). When the model decides a tool would help, it emits a **structured tool call** (a tool name and JSON arguments) instead of, or in addition to, text, and generation stops. **Your code** executes the tool and sends the result back as a new message. The model continues with the result in context.

```mermaid
sequenceDiagram
  participant App
  participant LLM
  participant Tool
  App->>LLM: user message + tool definitions
  LLM-->>App: tool_use {name: "get_weather", input: {"city": "Paris"}}
  App->>Tool: get_weather("Paris")
  Tool-->>App: "18°C, light rain"
  App->>LLM: tool_result "18°C, light rain"
  LLM-->>App: "It's 18°C with light rain in Paris; take an umbrella."
```

The model never executes anything itself. It only produces text, some of which your code interprets as calls. Models learn to call tools in post-training: SFT on tool-use transcripts, then RL in environments where tools return real results and the reward depends on the final outcome (Ch. 21–22). Schema-valid arguments can be **guaranteed** with constrained decoding against the tool's JSON schema (§18.7), which the Claude API exposes as `strict: true` on a tool.

## 27.2 The agent loop

An **agent** is an LLM running in a loop, choosing its own actions:

$$
\text{context}_{k+1} = \text{context}_k \oplus a_k\oplus o_k,\qquad a_k\sim\pi_\theta(\cdot\mid\text{context}_k),
$$

where each action $a_k$ is a tool call (or the final answer) and $o_k$ is the observation (the tool's result). It's the reinforcement-learning picture of Ch. 21: the LLM is the policy, the tools and the outside world are the environment, and the context is the state. **ReAct** (Yao et al., 2022) showed that interleaving *reasoning* ("I need the 2023 revenue first; let me search the annual report") with *actions* beats either alone. Modern reasoning models do this natively, thinking between tool calls (§22.7).

Here is the loop with the Claude Python SDK, written out by hand:

```python
import anthropic

client = anthropic.Anthropic()
tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a city. Returns temperature in Celsius and conditions.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name, e.g. 'Paris'"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,                      # arguments are guaranteed to match the schema
}]

def run_tool(name, args):
    if name == "get_weather":
        return f"18°C, light rain in {args['city']}"   # call a real API here
    return f"Error: unknown tool {name}"

messages = [{"role": "user", "content": "Should I bring an umbrella in Paris today?"}]
while True:
    response = client.messages.create(model="claude-opus-5-5", max_tokens=16000,
                                      tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        break                                          # final answer (or another stop reason)
    results = [{"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
               for b in response.content if b.type == "tool_use"]
    messages.append({"role": "user", "content": results})   # ALL results in ONE message

print(next(b.text for b in response.content if b.type == "text"))
```

Details that matter: append the assistant's *full* content (not just its text) so the tool calls stay in the history; return every tool result, each matched to its `tool_use_id`, in a single user message (the model may call several tools in parallel); report failures as results with `"is_error": True` rather than dropping them, so the model can recover. The SDK's **tool runner** writes this loop for you, generating schemas from typed Python functions:

```python
from anthropic import beta_tool

@beta_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. 'Paris'.
    """
    return f"18°C, light rain in {city}"

runner = client.beta.messages.tool_runner(
    model="claude-opus-5-5", max_tokens=16000, tools=[get_weather],
    messages=[{"role": "user", "content": "Should I bring an umbrella in Paris today?"}],
)
for message in runner:          # one message per model turn; stops when no more tool calls
    print(message.content)
```

## 27.3 Workflows vs agents

Anthropic's guide distinguishes **workflows**, where LLM calls and tools are orchestrated along paths **your code** defines, from **agents**, where **the model** decides the path. Start with the simplest thing that works: a single well-prompted call with retrieval often suffices. The five standard workflow patterns:

| Pattern | How it works | Use when |
|---|---|---|
| **Prompt chaining** | fixed sequence of calls, each processing the previous output, with checks between | the task decomposes into known steps (outline → check → draft) |
| **Routing** | classify the input, send it to a specialized prompt or model | distinct input types (refund vs technical question); cheap model for easy cases |
| **Parallelization** | run independent subtasks at once (sectioning), or the same task several times and vote | speed, or confidence via multiple attempts (§18.8) |
| **Orchestrator–workers** | an LLM breaks the task down at run time and delegates to worker calls | subtasks can't be predicted in advance (multi-file code changes) |
| **Evaluator–optimizer** | one call generates, another critiques against criteria, loop | clear evaluation criteria and measurable value from iteration |

Use an **agent** when the number and kind of steps can't be known in advance: debugging, research, operating an unfamiliar system. Agents cost more (many calls, long contexts) and their errors compound, so they need good tools, guardrails and evaluation.

## 27.4 Designing tools for models

Tools are an interface for a new kind of user. Anthropic calls this the **agent–computer interface (ACI)**, and it deserves the same care as a human UI:

- **Few, well-chosen tools.** Build tools that match tasks (`schedule_meeting`), not thin wrappers over every API endpoint (`list_users`, `list_events`, `create_event`). Overlapping tools confuse the choice.
- **Descriptions are prompts.** Explain what the tool does, when to use it (and when not to), what each parameter means and its format, with examples. Write them like documentation for a new teammate.
- **Return high-signal results.** Concise, relevant, human-readable fields (names, not opaque UUIDs) and nothing unnecessary. Every token of output enters the context (§25.4). Support pagination, filtering and a verbosity option for large results.
- **Make errors actionable**: "No customer with email 'jon@acme.co'. Did you mean 'john@acme.co'? Use search_customers to look up by name." beats a stack trace.
- **Make the right thing easy**: absolute paths instead of relative ones, idempotent operations, dry-run options for destructive actions.
- **Evaluate tools** with real tasks, read the transcripts, and revise. Models can help improve tool descriptions from their own failed transcripts.

## 27.5 The Model Context Protocol (MCP)

Every application wiring every tool by hand doesn't scale. **MCP** (introduced by Anthropic in 2024, now an open standard adopted widely across the industry) standardizes how applications expose tools and context to models:

- An **MCP server** offers *tools* (functions the model can call), *resources* (data it can read, such as files or records) and *prompts* (templates), over JSON-RPC, through a local process (stdio) or HTTP.
- An **MCP client** in the host application (Claude apps, IDEs, agent frameworks) connects to servers and presents their capabilities to the model.

Write a server once for your database, ticketing system or internal API, and any MCP-capable client can use it. The Claude API can also connect to remote MCP servers directly.

## 27.6 Server-side tools

Some tools run on the provider's infrastructure, with no execution loop on your side: **web search** and **web fetch** (with citations), a **code-execution** sandbox for data analysis and file generation, and **computer use**, where the model sees screenshots and issues mouse and keyboard actions to operate a desktop or browser. Server tools are declared in the request like other tools; their results arrive in the same response (Ch. 28).

## 27.7 The reliability math of multi-step tasks

If each step of a task succeeds independently with probability $p$, an $n$-step task succeeds with probability $p^n$. With $p = 0.95$: 10 steps give 60%, 50 steps give 8%. Long tasks need either very reliable steps or **error recovery**: noticing a failure and fixing it.

!!! theorem "Proposition 27.1 (Verification and retries)"
    Suppose each attempt at a step succeeds with probability $p$, and a verifier always detects failure, allowing up to $r$ retries. Each step then succeeds with probability $1-(1-p)^{r+1}$, and the expected number of attempts per step is $\frac{1-(1-p)^{r+1}}{p}$.

??? proof "Proof"
    The step fails only if all $r+1$ attempts fail, with probability $(1-p)^{r+1}$. Attempt $j$ (for $j = 1..r+1$) happens iff the first $j-1$ failed, so the expected count is $\sum_{j=0}^{r}(1-p)^j = \frac{1-(1-p)^{r+1}}{p}$. $\blacksquare$

With $p = 0.9$ and one retry, each step succeeds 99% of the time at an average of 1.1 attempts, and a 50-step task goes from 0.5% to 60%. **Cheap, reliable verification is what makes agents work**: tests for code, compilers and linters, schema validation, checking a web page after an action, a second model reviewing a result. That's why coding agents are where agents first became truly useful, and why RL with verifiable rewards trains them so effectively (§22.3).

METR's measurements ([Kwa et al., 2025](https://arxiv.org/abs/2503.14499)) put this in perspective: the length of software tasks (measured by how long they take skilled humans) that frontier agents complete with 50% reliability doubled roughly every 7 months from 2019 to 2025.

## 27.8 Memory, context and multiple agents

Long tasks outgrow any context window, and quality degrades as it fills (§25.4). Techniques:

- **Context editing**: drop old tool results that are no longer needed.
- **Compaction**: when the context nears its limit, summarize the history (decisions, current state, open problems) and continue from the summary. The Claude API offers this server-side.
- **External memory**: the agent writes notes, plans and progress to files or a memory tool, and reads them back in later sessions, like an engineer's notebook. Persistent files such as `TODO.md` or `NOTES.md` let work continue across context resets.
- **Sub-agents**: an orchestrator delegates focused subtasks (search this, read that, test this hypothesis) to sub-agents with fresh contexts, which return condensed results. Anthropic's research system, with a lead agent coordinating parallel sub-agents, outperformed a single agent by 90% on their internal research evaluation, using about 15× the tokens of a chat. Multi-agent setups pay off for broad, parallelizable work, and much less for tightly coupled tasks like most coding.

## 27.9 Safety and permissions

An agent's actions have real effects, and its inputs can be adversarial (§25.5):

- **Least privilege**: give each agent only the tools and access its task needs: read-only where possible, scoped credentials, allow-listed domains and paths.
- **Sandboxing**: run code and shell commands in containers or VMs without access to production systems or secrets.
- **Human in the loop** for irreversible or consequential actions (payments, emails, deletions, deployments), with a clear summary of what will happen.
- **Budgets and limits**: maximum steps, tokens, spend and wall-clock time per task, to stop runaway loops.
- **Audit logs** of every tool call and result, for debugging and accountability.
- **Treat tool outputs as untrusted data**, especially web content, emails and documents.

## 27.10 Evaluating agents

Agents are evaluated by **task completion in realistic environments**, not by single responses:

| Benchmark | Measures |
|---|---|
| [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) ([Jimenez et al., 2024](https://arxiv.org/abs/2310.06770)) | resolving real GitHub issues in Python repositories, checked by the projects' tests |
| [Terminal-Bench](https://www.tbench.ai/) | completing tasks in a command-line environment |
| [τ-bench](https://arxiv.org/abs/2406.12045) | customer-service tasks with simulated users and policies (retail, airline), measuring consistency across repeated trials (pass^k) |
| [OSWorld](https://arxiv.org/abs/2404.07972) | operating real desktop applications (computer use) |
| [GAIA](https://arxiv.org/abs/2311.12983), BrowseComp | research and browsing tasks with verifiable answers |

For your own agents: build a set of realistic tasks with automatic checks of the end state (files changed, tests passing, records created), run each several times (agents are stochastic, and τ-bench's pass^k exposes inconsistency), track success rate, cost and steps per task, and **read the transcripts**. Most improvements come from finding where the agent misunderstood a tool or lacked information (Ch. 29).

## Key takeaways

- Function calling: tools are described in the context, the model emits structured calls, your code executes them and returns results. The model never executes anything itself.
- An agent is an LLM policy in a loop of actions and observations (ReAct). Return all parallel results together; report errors as results.
- Prefer workflows (chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer) when the path is known; use agents when it isn't.
- Design tools as an interface for the model: few, task-shaped, well-described, concise results, actionable errors. MCP standardizes tool servers across applications.
- Errors compound as $p^n$; cheap verification plus retries is what makes long tasks reliable.
- Manage context with editing, compaction, external memory and sub-agents. Apply least privilege, sandboxing, human confirmation and budgets. Evaluate on end-to-end tasks, repeatedly.

## Exercises

1. A 20-step task has $p = 0.97$ per step. What's the success rate with no retries? With verification and up to 2 retries per step?
2. Rewrite this tool description to be useful to a model: `{"name": "q", "description": "query", "input_schema": {"type": "object", "properties": {"s": {"type": "string"}}}}`.
3. Which workflow pattern fits each task: (a) translate a support reply and check its tone; (b) answer billing, technical or sales questions with different instructions; (c) review a large pull request file by file?
4. Design the tool set (names and one-line descriptions) for an agent that helps employees book business travel within a company policy. Which actions need human confirmation?
5. Why does τ-bench report pass^k (all $k$ trials succeed) rather than pass@k for customer-service agents?
6. **Code:** extend the loop in §27.2 with a `search_docs` tool backed by `rag.py` (Ch. 26) and a `max_steps` limit. Log every tool call and result.

??? success "Solutions"
    1. $0.97^{20}\approx54\%$. With 2 retries, per-step success is $1-0.03^3 = 0.999973$, so $\approx99.95\%$ overall.
    2. Name it `search_orders`; describe what it searches, when to use it, the query syntax with an example, what it returns (e.g. up to 10 orders with id, date, status, total) and what to do with no results; name the parameter `query` with its own description; mark it required.
    3. (a) prompt chaining (translate → check tone, maybe evaluator–optimizer); (b) routing; (c) orchestrator–workers or parallelization (sectioning by file).
    4. E.g. `get_travel_policy`, `search_flights`, `search_hotels`, `check_policy_compliance`, `hold_booking`, `confirm_booking`, `request_exception`. Confirmation for `confirm_booking` (spends money) and `request_exception`.
    5. For customer service the same kind of request recurs many times, so the question is whether the agent handles it *reliably every time*, not whether it can succeed once. pass^k drops quickly for inconsistent agents.
    6. Expose `search_docs(query) -> top-5 chunks with ids`, stop after `max_steps` with a message saying the budget was reached, and print each call.

## References

- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) (2024); [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) (2025); [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (2025).
- S. Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (ICLR 2023).
- T. Schick et al., [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (NeurIPS 2023).
- C. Jimenez et al., [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) (ICLR 2024).
- S. Yao et al., [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) (2024).
- T. Kwa et al. (METR), [Measuring AI Ability to Complete Long Tasks](https://arxiv.org/abs/2503.14499) (2025).
- [Model Context Protocol](https://modelcontextprotocol.io/) documentation.
