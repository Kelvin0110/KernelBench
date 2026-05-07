# KernelBench L0/L1 Prompt Memory Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured L0/L1 prompting, dual-format L1 persistence (`.txt` + `.jsonl`), and an extractor-model stage that selects relevant L1 memories per iteration while keeping one-system/one-user coder prompting.

**Architecture:** Keep the existing governor loop shape, then add one pre-coder extraction step and memory-formatting upgrades in shared `evolving_common`. KernelBench-specific wiring in `kb_governor.py` uses the new shared utilities and configuration fields. L0 remains append-only through the run.

**Tech Stack:** Python 3.11, Pydantic, pytest, evolving_common helpers, KernelBench integration scripts.

---

### Task 1: Add failing tests for memory/prompt contracts

**Files:**
- Modify: `Self-Evolving-Agent/tests/test_memory_manager.py`
- Create: `Self-Evolving-Agent/tests/test_prompt_context.py`
- Modify: `scripts_integration/new_evolving_agent/tests/test_kb_governor.py`

- [ ] **Step 1: Write failing tests for dual L1 output and no-clear promotion**

```python
def test_promote_writes_txt_and_jsonl_without_clearing_l0(tmp_path):
    ...
    assert len(buf) == 2
    assert jsonl_entry["description"]
```

- [ ] **Step 2: Write failing tests for structured user prompt sections**

```python
def test_build_user_prompt_with_selected_l1_and_actions():
    assert "## Selected L1 memory" in prompt
    assert "Allowed actions" in prompt
```

- [ ] **Step 3: Write failing governor test for extractor-driven L1 selection**

```python
def test_governor_uses_extractor_selected_l1(...):
    assert "selected-memory-1" in coder_user_prompt
```

- [ ] **Step 4: Run tests and verify RED**

Run:
```bash
uv run pytest Self-Evolving-Agent/tests/test_memory_manager.py Self-Evolving-Agent/tests/test_prompt_context.py scripts_integration/new_evolving_agent/tests/test_kb_governor.py -q
```

Expected: failing assertions for not-yet-implemented behavior.

- [ ] **Step 5: Commit test-only RED state**

```bash
git add Self-Evolving-Agent/tests/test_memory_manager.py Self-Evolving-Agent/tests/test_prompt_context.py scripts_integration/new_evolving_agent/tests/test_kb_governor.py
git commit -m "test: add failing coverage for l0/l1 prompt-memory selection"
```

### Task 2: Implement shared evolving_common memory and prompt updates

**Files:**
- Modify: `Self-Evolving-Agent/evolving_common/memory_manager.py`
- Modify: `Self-Evolving-Agent/evolving_common/prompt_context.py`
- Modify: `Self-Evolving-Agent/evolving_common/governor/promotion.py`
- Modify: `Self-Evolving-Agent/evolving_common/__init__.py`

- [ ] **Step 1: Add JSONL L1 schema + append/read helpers and no-clear option**

```python
class L1Entry(TypedDict):
    entry_id: str
    timestamp: str
    description: str
    trigger: str
    content: str
    source: str
```

- [ ] **Step 2: Update promotion flow to write txt + jsonl and keep L0 by config**

```python
def promote_l0_to_l1(..., clear_l0_after_promotion: bool = True):
    append_l1_journal(...)
    append_l1_jsonl(...)
    if clear_l0_after_promotion:
        clear_l0(buffer)
```

- [ ] **Step 3: Add structured prompt builders with selected L1 and action guidance**

```python
def build_user_prompt_with_memory(..., selected_l1_entries: list[dict[str, str]] | None = None, ...):
    ...
```

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run:
```bash
uv run pytest Self-Evolving-Agent/tests/test_memory_manager.py Self-Evolving-Agent/tests/test_prompt_context.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit shared implementation**

```bash
git add Self-Evolving-Agent/evolving_common/memory_manager.py Self-Evolving-Agent/evolving_common/prompt_context.py Self-Evolving-Agent/evolving_common/governor/promotion.py Self-Evolving-Agent/evolving_common/__init__.py Self-Evolving-Agent/tests/test_memory_manager.py Self-Evolving-Agent/tests/test_prompt_context.py
git commit -m "feat: add structured l0/l1 memory persistence and prompting"
```

### Task 3: Add extractor-model client utilities and governor wiring

**Files:**
- Modify: `Self-Evolving-Agent/evolving_common/llm_client.py`
- Modify: `Self-Evolving-Agent/kernelbench/config.py`
- Modify: `scripts_integration/new_evolving_agent/kb_governor.py`
- Modify: `scripts_integration/new_evolving_agent/tests/test_kb_governor.py`

- [ ] **Step 1: Add extractor model resolution and call helper**

```python
def get_tri_llm_model_ids() -> tuple[str, str, str]:
    ...
```

- [ ] **Step 2: Add config fields (`extractor_model`, limits, timeout, max_memories`)**

```python
extractor_max_memories: int = Field(default=10, ge=1)
```

- [ ] **Step 3: Implement extractor stage in governor before coder call**

```python
selected_entries = self._select_relevant_l1_memories(...)
coder_prompt = self._get_coder_prompt(..., selected_l1_entries=selected_entries, ...)
```

- [ ] **Step 4: Capture action/reasoning text into L0 entries while preserving code extraction path**

```python
append_l0(l0, "assistant", raw_text)
append_l0(l0, "system", f"action={action}")
```

- [ ] **Step 5: Run targeted tests and verify GREEN**

Run:
```bash
uv run pytest scripts_integration/new_evolving_agent/tests/test_kb_governor.py -q
```

Expected: all pass with extractor behavior validated.

- [ ] **Step 6: Commit governor + config + llm client changes**

```bash
git add Self-Evolving-Agent/evolving_common/llm_client.py Self-Evolving-Agent/kernelbench/config.py scripts_integration/new_evolving_agent/kb_governor.py scripts_integration/new_evolving_agent/tests/test_kb_governor.py
git commit -m "feat: add l1 extractor model stage for kernelbench governor"
```

### Task 4: Final verification and documentation updates

**Files:**
- Modify: `documents/PROGRESS.md`
- Modify: `documents/PROJECT_STRUCTURE.md`
- Modify: `scripts_integration/new_evolving_agent/README.md` (if prompt behavior docs need update)

- [ ] **Step 1: Run focused integration test suite**

Run:
```bash
uv run pytest Self-Evolving-Agent/tests/test_memory_manager.py Self-Evolving-Agent/tests/test_prompt_context.py scripts_integration/new_evolving_agent/tests/test_kb_governor.py scripts_integration/new_evolving_agent/tests/test_evolve_kb_batch.py -q
```

Expected: all pass.

- [ ] **Step 2: Optional smoke dry-run for batch entrypoint**

Run:
```bash
uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py --dry-run --max-problems 1 --max-iterations 1 --run-name spec_plan_smoke
```

Expected: exits 0 and writes run artifacts.

- [ ] **Step 3: Update progress ledger and project structure docs**

```markdown
- Added extractor model stage and dual-format L1 persistence.
```

- [ ] **Step 4: Commit documentation and verification-aligned updates**

```bash
git add documents/PROGRESS.md documents/PROJECT_STRUCTURE.md scripts_integration/new_evolving_agent/README.md
git commit -m "docs: record l0/l1 prompt and extractor architecture updates"
```