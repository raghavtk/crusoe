# Crusoe — Learning Guide

This guide explains how agent loops work by walking through Crusoe's code.
No prior experience with LLM agents is assumed.

---

## 1. What is an LLM Agent?

A "raw" LLM call looks like this:

```
You → [question] → LLM → [answer]
```

That's just a function call. An **agent** is different: it wraps the LLM
in a loop so it can *act* — call tools, observe results, and keep thinking.

```
You → [task] → LLM → [call tool A] → execute A → [result of A] → LLM → [answer]
```

The key insight: **the conversation history grows with every tool call**.
The LLM "remembers" what tools it called and what they returned because
we append those messages to the history before each new LLM call.

---

## 2. The Agent Loop — Step by Step

Open `src/core/agent.py`. Here's what happens every iteration:

```
messages = [initial question]

LOOP:
  1. Call LLM(messages, tools)
  2. Did the LLM want to call a tool?
     YES → execute the tool, append result to messages, go to step 1
      NO → return the LLM's final answer
```

The `stop_reason` field in the LLM response tells us which branch to take:
- `"tool_use"` → the model wants to call a function
- `"end_turn"` → the model is done and has a final answer

---

## 3. What is a Tool?

A tool is just a Python function with extra metadata so the LLM knows:
- **name**: how to refer to it ("search_papers")
- **description**: *when* to use it (this is the most important field)
- **parameters**: *what arguments* to pass (JSON Schema)

```python
# From src/core/tool.py
@dataclass
class Tool:
    name: str
    description: str     # ← LLM reads this to decide WHEN to call the tool
    parameters: dict     # ← JSON Schema: what args to provide
    func: Callable       # ← the actual Python function that runs
```

When the LLM says "call search_papers with query='JWT attacks'", the agent
loop finds the matching `Tool`, calls `tool.func(query="JWT attacks")`,
and appends the result back into the messages.

---

## 4. Why Does This Work?

Modern LLMs are trained to output structured JSON for tool calls. When you
provide a list of tools (as JSON Schemas), the model learns to say:

```json
{
  "tool_use": {
    "name": "search_papers",
    "arguments": {"query": "JWT token security", "limit": 20}
  }
}
```

Instead of just writing `"I would search for JWT papers"` as text.

Your code intercepts this, runs the search, and puts the results back:

```json
{
  "role": "tool",
  "name": "search_papers",
  "content": "[{\"title\": \"JWT Attacks ...\", ...}]"
}
```

The LLM reads the search results and can now either call more tools or
produce a final answer.

---

## 5. How Crusoe's Agents Differ

Not all agents in Crusoe use the tool-calling loop. Here's a map:

| Agent | Uses Loop? | Why |
|-------|-----------|-----|
| Topic Decomposition | No | Just needs one LLM call → structured JSON |
| Discovery | No (direct API) | Simpler/cheaper to loop over keywords ourselves |
| Paper Curator | No | One LLM call per batch, plus one repair attempt when invalid |
| Synthesis | No | One or two LLM calls |

The agent loop (`src/core/agent.py`) is most useful when you want the **LLM
to decide** which tools to call and in what order. For Crusoe's discovery
stage, we know exactly which searches to run (one per keyword), so a direct
Python loop is cleaner.

---

## 6. Message Roles Explained

Every LLM conversation is a list of messages. The `role` field tells the
model who said what:

| Role | Who | What it contains |
|------|-----|-----------------|
| `system` | You (developer) | High-level instructions for the agent |
| `user` | The human / pipeline | The task or question |
| `assistant` | The LLM | Its response (text + optional tool calls) |
| `tool` | Your code | The result of executing a tool |

The model sees the **entire list** on every call. This is why agents can
maintain context — the history is explicitly passed back each time.

---

## 7. Provider Normalisation

Different LLM APIs use different formats. Gemini uses `Content` objects and
`FunctionCall`/`FunctionResponse` parts. Cerebras/OpenAI uses a `tool_calls`
array with JSON strings for arguments.

`src/llm/providers.py` handles all translation:

```
Internal format → GeminiProvider → Gemini API → response → Internal format
Internal format → CerebrasProvider → Cerebras API → response → Internal format
```

The agent loop only ever speaks the internal format. You can add a new
provider (e.g. OpenAI, Mistral) by subclassing `LLMProvider` and implementing
`call()`.

---

## 8. State and Checkpointing

`PipelineState` is a dataclass that gets passed through all agents:

```python
state = PipelineState(topic="authentication tokens")

# Each agent mutates and returns state
state = topic_decomposition.run(state, provider)  # adds .keyword_clusters
state = discovery.run(state, provider)             # adds .papers_raw
state = paper_curator.run(state, provider)         # adds .papers_curated
state = synthesis.run(state, provider)             # adds .synthesis
```

After each stage, the orchestrator saves state to JSON:

```python
state.save("data/session_checkpoint.json")
```

On `--resume`, the orchestrator checks which fields are already populated
and skips completed stages. This is a simple but effective form of fault
tolerance — a crash after paper curation doesn't require re-running discovery.

---

## 9. What to Read Next

To go deeper on agent patterns:

1. **ReAct** (Reason + Act): the original paper showing why alternating
   reasoning and tool use outperforms pure chain-of-thought.
   → Yao et al., 2022. "ReAct: Synergizing Reasoning and Acting in Language Models"

2. **Function Calling**: OpenAI's blog post on structured tool use.

3. **Multi-agent systems**: AutoGen, CrewAI papers — but Crusoe intentionally
   avoids these frameworks so you can see exactly what's happening.

4. **Prompt engineering for structured output**: how to reliably get JSON
   back from LLMs (the paper curator and synthesis agents depend on this).

---

## 10. Common Gotchas

| Problem | Cause | Fix |
|---------|-------|-----|
| Agent loop runs forever | LLM keeps calling tools | Check `max_iterations`; improve system prompt to encourage `end_turn` |
| LLM returns malformed JSON | Prompt not explicit enough | Add "Return JSON only — no prose, no markdown fences" to system prompt |
| Gemini tool result error | Tool result must be a dict | Wrap string results in `{"result": "..."}` |
| 429 from Semantic Scholar | Rate limited | `tenacity` handles retry; add `SEMANTIC_SCHOLAR_API_KEY` for higher limits |
| Google auth fails | No `credentials.json` | Download from Google Cloud Console |
