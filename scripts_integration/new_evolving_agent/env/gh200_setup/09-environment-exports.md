# Step 7 — Environment exports

*Part of the [2 × GH200 host setup guide](README.md).*

---

```bash
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:$REPO/.venv/bin:$PATH
```

- **`CUDA_HOME` must be exported, not merely satisfiable.** Putting `nvcc` on `PATH`
  alone satisfies torch, but agent-written kernels literally test
  `os.getenv("CUDA_HOME")`; leaving it unset keeps those guards closed and
  resurrects the dead-code failure from [CUDA toolkit](08-cuda-toolkit.md).
- **`.venv/bin` must be on `PATH`**, or builds die with
  `RuntimeError: Ninja is required to load C++ extensions`. `ninja` exists **only**
  as a pip wheel here; it is not installed system-wide.

`launch_run.sh` sets both itself. Any hand-rolled invocation of `evolve_kb_batch.py`
must set them too. Neither is in the source host's `.bashrc` — `~/.bashrc` only adds
`$HOME/.local/bin` (for uv). Adding them to `.bashrc` on the new host is a
reasonable improvement; just do not rely on it inside scripts.

### `.env`

Copy `.env.example` → `.env` and fill in. Keys actually read by the run path:

| variable | required | purpose |
|---|---|---|
| `NVIDIA_INF_API_KEY` | **yes** — `launch_run.sh:38` greps for it | chat + embeddings, `inference-api.nvidia.com/v1` |
| `NVIDIA_API_KEY` | if using `--nvidia-endpoint integrate` | `integrate.api.nvidia.com/v1` |
| `NVIDIA_ENDPOINT` | optional | endpoint override |
| `NVIDIA_EMBED_ENDPOINT` | optional (default `inference`) | skill-merge embeddings, chosen independently of chat |
| `NVIDIA_SKILL_MERGE_EMBED_MODEL` | optional | default `nvidia/qwen/qwen3-embedding-0.6b` |
| `NVIDIA_{CODER,SUMMARIZER,EXTRACTOR,ACTION_SELECTOR}_MODEL` | optional | per-role model overrides |
| `NVIDIA_API_TIMEOUT_SEC` | optional | client timeout |

`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` /
`TOGETHER_API_KEY` / `FIREWORKS_AI_API_KEY` / `SGLANG_API_KEY` also exist in the
source `.env` but are unused by the evolving-agent path.

Model IDs differ per endpoint (`gpt-oss-120b` → `openai/gpt-oss-120b` on integrate vs
`nvidia/openai/gpt-oss-120b` on inference). Use the aliases in
`Self-Evolving-Agent/evolving_common/llm_client.py`, not raw IDs. Isolate key-vs-model
failures with `probe_integrate_key.py` (referenced in `CLAUDE.md §5` but **not present** in the tree as of 2026-08-22).

**Never copy the source host's `.env` over a network share or paste its contents into
a ticket** — it contains live keys.

---

[← CUDA toolkit](08-cuda-toolkit.md) · [Index](README.md) · [Acceptance test →](10-acceptance-test.md)
