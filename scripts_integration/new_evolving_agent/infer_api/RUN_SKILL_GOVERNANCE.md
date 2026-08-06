# Skill-Governance Experiment Matrix (30 iterations, GH200)

Companion to [RUN_WITH_UV_INFER.md](RUN_WITH_UV_INFER.md). That file varies the
**L0 context-management** mode; this one holds context management fixed at the
default (`truncation`) and varies the **L1 skill-governance** add-ons so the
comparison is against the finished base run.

Analysis tooling and the running report live in
[`scripts_integration/new_evolving_agent_analysis/`](../../new_evolving_agent_analysis/).

## Held constant across every cell

| Setting | Value |
|---|---|
| Subset | `subset_selection/selected_problems_50.csv` (50 problems, the default) |
| Iterations | 30 (`--max-iterations 30`) |
| Model | `gpt-oss-120b` (all four roles via `--model`) |
| Endpoint | `inference` (`--nvidia-endpoint inference`) |
| Hardware baseline | `--hardware NVIDIA_GH200x2` |
| Context management | `truncation` (default — flag omitted) |

Only the three skill-governance flags vary. That makes every cell below directly
comparable to the finished baseline run
`base_agent_gpt_oss_120b_itr30_GH200_2026_08_03_04_51`, which is the same
configuration with all three off.

## The matrix

| # | Cell | `--enable-skill-refinement` | `--skill-deletion` | `--skill-merging` | Run name |
|---|------|:--:|:--:|:--:|---|
| — | *baseline (done)* | ✗ | ✗ | ✗ | `base_agent_gpt_oss_120b_itr30_GH200` |
| 1 | refinement only | ✓ | ✗ | ✗ | `base_agent_gpt_oss_120b_refine_itr30_GH200` |
| 2 | merge only | ✗ | ✗ | ✓ | `base_agent_gpt_oss_120b_merge_itr30_GH200` |
| 3 | deletion only | ✗ | ✓ | ✗ | `base_agent_gpt_oss_120b_delete_itr30_GH200` |
| 4 | refine + merge | ✓ | ✗ | ✓ | `base_agent_gpt_oss_120b_refine_merge_itr30_GH200` |
| 5 | refine + deletion | ✓ | ✓ | ✗ | `base_agent_gpt_oss_120b_refine_delete_itr30_GH200` |
| 6 | deletion + merge | ✗ | ✓ | ✓ | `base_agent_gpt_oss_120b_delete_merge_itr30_GH200` |
| 7 | all three | ✓ | ✓ | ✓ | `base_agent_gpt_oss_120b_refine_delete_merge_itr30_GH200` |

`evolve_kb_batch.py` appends a UTC `_YYYY_MM_DD_HH_MM` suffix to `--run-name`, so
the on-disk folder carries a timestamp. Use the full timestamped name when resuming.

> **`--skill-merging` does not actually require `--skill-deletion`.**
> The flag's help text says it does, but the gate in
> `Self-Evolving-Agent/evolving_common/governor/gen3_stages.py` is
> `enable_skill_governance = enable_skill_deletion or enable_skill_merging`, and
> the merge pass itself only checks `enable_skill_merging`. Cells 2 and 4
> (merge without deletion) are therefore real configurations, not silent no-ops.
> Verify this holds if that file changes.

### Why the governance defaults fire at this scale

The defaults are keyed to **global** iterations, which accumulate across the whole
batch rather than resetting per problem. At 50 problems × 30 iterations ≈ 1500
global iterations:

| Default | Value | Effect over the run |
|---|---|---|
| `--skill-merge-interval` | 50 | ~30 merge passes |
| `--l1-skill-consecutive-unused-delete-after` | 50 | unused-streak GC active |
| `--l1-skill-deletion-grace-iterations` | 50 | new skills protected early on |
| `--skill-merge-similarity` | 0.7 | cosine threshold for merge clustering |

No override is needed. Had these been per-problem counters, a 30-iteration run
would never reach the 50-iteration thresholds and every governance cell would have
silently reduced to the baseline — worth re-checking if the counter semantics change.

> **Embeddings still hit the integrate endpoint.** The merge pass clusters skills by
> embedding — `run_skill_merge_pass` → `cluster_active_skills_snapshot` →
> `load_or_compute_skill_embeddings` → `embed_texts_nvidia`, which is pinned to
> `integrate.api.nvidia.com` regardless of `--nvidia-endpoint`. Cells 2, 4, 6 and 7
> therefore need **`NVIDIA_API_KEY`** in `.env` in addition to `NVIDIA_INF_API_KEY`.
>
> This matters more than it looks: `_maybe_run_skill_merge` wraps the whole pass in a
> broad `except Exception` and returns `[]`. A missing integrate key degrades a merge
> cell to the baseline *without failing the run*. The batch script does pass
> `verbose=True`, so the failure prints `[governor][gen3] skill merge skipped: ...` to
> the log — grep for it rather than assuming silence means success.

### Merge diagnostics written next to the L1 journal

Each merge pass appends a snapshot regardless of whether any merge was accepted, so a
zero-merge outcome is still diagnosable:

| File (beside `shared_l1.jsonl`) | Contents |
|---|---|
| `l1_skill_merge_clustering.jsonl` | Per pass: `global_iteration`, `active_skill_count`, pairwise cosine similarities, DBSCAN labels, qualified clusters |
| `l1_skill_merge_state.json` | `last_merge_global_iter` — merge-pass scheduling |
| `l1_skill_embeddings.json` | Cached skill embeddings |

Clustering needs at least `min_cluster_size=2` active skills (`min_samples=2`), so
early passes legitimately produce nothing. If a merge cell ends with zero merges,
read `l1_skill_merge_clustering.jsonl` before concluding merging is ineffective —
`pairs_above_threshold: 0` (skills too dissimilar at the 0.7 cosine threshold) is a
different finding from the pass never having run.

## Scheduling

Both GPUs are occupied until the in-flight context-management runs finish:

| GPU | Current run | Started (UTC) | ETA (UTC) |
|---|---|---|---|
| 0 | `base_agent_gpt_oss_120b_selective_itr30_GH200` | 2026-08-04 17:24 | ~2026-08-06 03:30 |
| 1 | `base_agent_gpt_oss_120b_folding_itr30_GH200` | 2026-08-04 17:26 | ~2026-08-06 03:30 |

ETA uses the measured rate from the two finished runs, not an estimate:

| Finished run | Problems | Mean wall/problem | Total |
|---|---|---|---|
| baseline (truncation) | 50 | 2,468 s (41 min) | 34.3 h |
| markov_report | 50 | 2,296 s (38 min) | 31.9 h |

Budget ~34 h per cell. Confirm a GPU is actually idle before launching:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
pgrep -af "evolve_kb_batch.py --run-name"
```

Seven cells over two GPUs is four waves — roughly six days. Suggested pairing
(one cell per GPU per wave, singles before combinations so the main effects land first):

| Wave | GPU 0 | GPU 1 |
|---|---|---|
| 1 | 1 refinement only | 3 deletion only |
| 2 | 2 merge only | 5 refine + deletion |
| 3 | 4 refine + merge | 6 deletion + merge |
| 4 | 7 all three | *(spare — reruns / seed repeat)* |

The spare slot in wave 4 is worth spending on a **repeat of the baseline** rather
than an eighth configuration. With one run per cell there is no estimate of
run-to-run variance, and without that the between-cell differences cannot be
interpreted. A second baseline gives that estimate at the cost of one slot.

## Commands

Each cell is one command. Substitute the GPU index for whichever device is free.

### 1. Refinement only

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_refine_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --enable-skill-refinement \
  --no-skill-deletion \
  >> base_agent_gpt_oss_120b_refine_itr30_GH200.log 2>&1 &
```

### 2. Merge only

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_merge_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --skill-merging \
  --no-skill-deletion \
  >> base_agent_gpt_oss_120b_merge_itr30_GH200.log 2>&1 &
```

### 3. Deletion only

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_delete_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --skill-deletion \
  >> base_agent_gpt_oss_120b_delete_itr30_GH200.log 2>&1 &
```

### 4. Refinement + merge

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_refine_merge_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --enable-skill-refinement \
  --skill-merging \
  --no-skill-deletion \
  >> base_agent_gpt_oss_120b_refine_merge_itr30_GH200.log 2>&1 &
```

### 5. Refinement + deletion

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_refine_delete_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --enable-skill-refinement \
  --skill-deletion \
  >> base_agent_gpt_oss_120b_refine_delete_itr30_GH200.log 2>&1 &
```

### 6. Deletion + merge

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_delete_merge_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --skill-deletion \
  --skill-merging \
  >> base_agent_gpt_oss_120b_delete_merge_itr30_GH200.log 2>&1 &
```

### 7. All three

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_refine_delete_merge_itr30_GH200 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --enable-skill-refinement \
  --skill-deletion \
  --skill-merging \
  >> base_agent_gpt_oss_120b_refine_delete_merge_itr30_GH200.log 2>&1 &
```

### Optional — baseline repeat (variance estimate)

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --run-name base_agent_gpt_oss_120b_itr30_GH200_rep2 \
  --max-iterations 30 \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --no-skill-deletion \
  >> base_agent_gpt_oss_120b_itr30_GH200_rep2.log 2>&1 &
```

## Verify a cell is actually exercising its feature

A governance flag that silently no-ops produces a clean run indistinguishable from
the baseline. Check within the first hour rather than discovering it 34 hours later:

```bash
RUN=runs_evolving/base_agent_gpt_oss_120b_refine_itr30_GH200_<timestamp>

# Flags recorded as intended
python3 -c "import json;d=json.load(open('$RUN/run_summary.json'));print({k:d[k] for k in ['enable_skill_refinement','skill_deletion','skill_merging','context_management','model','nvidia_endpoint']})"

# Governance phases actually firing in the LLM turns
cat $RUN/workspaces/*/chat_history.jsonl | python3 -c "
import sys,json,collections
c=collections.Counter()
for l in sys.stdin:
    try: c[json.loads(l).get('phase','?')]+=1
    except Exception: pass
print(c.most_common())"

# Merge/deletion events landing in L1
python3 -c "
import json,collections
c=collections.Counter()
for l in open('$RUN/shared_l1.jsonl'):
    l=l.strip()
    if l: c[json.loads(l).get('source','?')]+=1
print(c.most_common())"

# Merge cells: did the pass run at all, and did anything clear the threshold?
python3 -c "
import json
for l in open('$RUN/l1_skill_merge_clustering.jsonl'):
    d=json.loads(l)
    print(d['global_iteration'], 'active=',d.get('active_skill_count'), 'clusters=',len(d.get('clusters',[])))" 2>/dev/null || echo "no clustering log — merge pass never ran"

# Merge cells: did it fail silently on the embedding call?
grep -c "skill merge skipped" base_agent_gpt_oss_120b_merge_itr30_GH200.log
```

Expect `skill_merge` sources in L1 for merge cells, and refinement phases in
`chat_history.jsonl` for refinement cells. Absence means the flag did not take —
stop and investigate before burning 34 hours of GPU on a run that is silently a
duplicate of the baseline.

## Resume

`evolve_kb_batch.py` records every governance flag in `run_summary.json` and aborts
a resume whose flags do not match. Repeat the cell's flags exactly:

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run python scripts_integration/new_evolving_agent/evolve_kb_batch.py \
  --resume \
  --run-name base_agent_gpt_oss_120b_refine_itr30_GH200_YYYY_MM_DD_HH_MM \
  --max-problems 50 \
  --max-iterations 30 \
  --start-problem <N> \
  --nvidia-endpoint inference \
  --model gpt-oss-120b \
  --hardware NVIDIA_GH200x2 \
  --enable-skill-refinement \
  --no-skill-deletion \
  >> base_agent_gpt_oss_120b_refine_itr30_GH200_resume.log 2>&1 &
```

Override with `--allow-resume-config-mismatch` only when the change is deliberate;
a mismatched resume silently mixes two configurations into one run directory and
invalidates the cell.
