# KernelBench L0/L1 Prompting and Memory Selection Design

## 1. Goal
Upgrade the evolving-agent KernelBench loop to keep the current one-system/one-user prompt pattern while making memory usage more structured and more useful per iteration.

The design keeps code extraction unchanged (extract from coder reply as today), but improves how the model is guided by:
- richer L0 context (action/reasoning/output history),
- structured L1 memory entries with concise descriptions,
- a dedicated extractor model that selects relevant L1 memories each round.

## 2. Scope
This design updates both shared infrastructure and the KernelBench integration.

### Shared (benchmark-agnostic) scope
- `Self-Evolving-Agent/evolving_common/prompt_context.py`
- `Self-Evolving-Agent/evolving_common/memory_manager.py`
- `Self-Evolving-Agent/evolving_common/llm_client.py`
- `Self-Evolving-Agent/evolving_common/governor/promotion.py`

### KernelBench-specific scope
- `Self-Evolving-Agent/kernelbench/config.py`
- `scripts_integration/new_evolving_agent/kb_governor.py`
- tests under:
  - `Self-Evolving-Agent/tests/`
  - `scripts_integration/new_evolving_agent/tests/`

## 3. Requirements (Finalized)
1. Keep one system prompt + one user prompt per coder call (no full chat history send-back).
2. Update prompts to explicitly describe:
   - L0 as raw recent attempts (code, reasoning, terminal output, action),
   - L1 as high-level strategies/insights,
   - fixed iterative budget (`max_iterations`) and allowed actions (`propose_new`, `debug_current`, `refine_current`).
3. L0 should no longer be cleared after promotion.
4. L1 should continue writing to the existing `.txt` journal and additionally write `.jsonl` records.
5. Each L1 memory record must include a concise `description` field.
6. Add a new extractor model stage to select relevant L1 entries before each coder call.
7. Extracted L1 count is configurable with an initial maximum of `10`.
8. Keep existing code extraction flow from coder output (no new LLM pass to summarize code/metrics).

## 4. Architecture

### 4.1 Coder stage (unchanged shape)
Each iteration still sends exactly:
- one `system` message (updated guidance),
- one `user` message (task + selected L1 + structured L0 + latest eval feedback).

### 4.2 L0 structure update
L0 remains append-only during one run. Instead of only exposing flat role/content text, prompt rendering will aggregate entries into iteration-centric blocks where possible:
- action chosen,
- reasoning summary,
- submitted code snippet reference,
- terminal/evaluation highlights.

Raw data remains loggable/auditable, but prompt text becomes structured.

### 4.3 L1 dual persistence
Promotion writes both:
- `shared_l1.txt` (existing human-readable journal format),
- `shared_l1.jsonl` (one JSON object per entry).

JSONL schema:
```json
{
  "entry_id": "uuid",
  "timestamp": "ISO-8601",
  "description": "Concise one-line summary",
  "content": "Detailed strategy or lesson",
  "source": "summarizer|manual|fallback"
}
```

### 4.4 Extractor stage (new)
Before coder call, if L1 JSONL has entries:
1. Build extractor prompt with current task state + latest eval signal + all L1 descriptions.
2. Ask extractor model to return selected `entry_id`s.
3. Cap selection by configurable `extractor_max_memories` (default `10`).
4. Load full `content` for selected entries and inject only those into coder user prompt.

The extractor model is independently configurable (model id, timeout, max tokens).

## 5. Prompt Contracts

### 5.1 Coder system prompt additions
The system prompt will require:
- acknowledge iterative budget (`max_iterations`),
- choose an explicit action each round from:
  - `propose_new`
  - `debug_current`
  - `refine_current`
- provide concise reasoning before code,
- output exactly one fenced Python code block as final submission.

### 5.2 Coder user prompt structure
Updated user prompt sections:
1. Official benchmark task.
2. Iteration context (attempt number, best metric so far).
3. Selected L1 memories (description + content for selected ids).
4. Structured L0 history summary.
5. Latest evaluation feedback.
6. Explicit action guidance.

### 5.3 Summarizer output requirement for L1
Summarizer instruction will require an explicit format containing:
- `Description:` one concise line,
- `Details:` bullet points.

If format is missing, fallback logic derives a safe description from the first non-empty line.

## 6. Data and Compatibility
- Existing `.txt` readers remain supported.
- New JSONL path is additive and non-breaking.
- Promotion no longer clears L0, increasing prompt size over time; mitigation is via structured truncation in prompt formatter if needed later.

## 7. Risks and Mitigations
- **Risk:** L0 grows too large and hurts token budget.
  - **Mitigation:** structured aggregation and optional truncation window in formatter.
- **Risk:** extractor returns invalid IDs.
  - **Mitigation:** robust parser + fallback to top-N most recent L1 entries.
- **Risk:** summarizer does not emit required description.
  - **Mitigation:** fallback description derivation and still persist entry.

## 8. Verification Plan
- Unit tests for memory manager:
  - JSONL append format,
  - promote without L0 clearing,
  - read/select L1 entries.
- Unit tests for prompt context:
  - structured user prompt sections,
  - action guidance rendering.
- Unit tests for llm client/extractor response parsing utilities.
- KernelBench governor tests:
  - extractor stage wiring,
  - selected L1 injection,
  - L0 retention across iterations.

## 9. Out of Scope
- Replacing code extraction with a separate summarization pass.
- Full multi-turn chat history replay.
- Changing benchmark evaluation semantics.