# Step 11 — First run

*Part of the [2 × GH200 host setup guide](README.md).*

---

```bash
bash scripts_integration/new_evolving_agent/env/launch_run.sh \
  0 base_agent_gpt_oss_120b_itr30_GH200b truncation
```

The `b` suffix disambiguates this host from the source host's existing
`base_agent_gpt_oss_120b_itr30_GH200_*` run — see
[Timing baselines](11-timing-baselines.md) on naming across two hosts.

The launcher preflights nvcc, ninja, the baseline dir, the API key, GPU idleness, and
a live `load_inline` compile probe before it launches anything under `nohup`. Every
one of those checks exists because its absence once silently corrupted a long run.

Health checks while running:

```bash
grep -c CUDA_HOME <log>                       # must stay 0
grep -E "^\[batch\]" <log> | tail -2          # problem progress
grep -h "gpu-eval-lock" <log>                 # waits >=5s; "proceeding UNLOCKED" = investigate
uv run --no-sync python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
```

`torch._inductor` "No valid triton configs / OutOfMemoryError: triton_mm" tracebacks
are **benign** autotuner noise.

For `--skill-merging` arms, confirm the merge pass is actually working — it swallows
its own exceptions when `verbose` is off, so a broken embedding path produces zero
merges and zero log output:

```bash
python3 -c "import json;print(len(json.load(open('<run>/l1_skill_embeddings.json'))['skills']))"
wc -l <run>/l1_skill_merges.jsonl
```

Both must be non-zero.

---

[← Multi-arm settings](12-multi-arm-settings.md) · [Index](README.md) · [Failure modes →](14-failure-modes.md)
