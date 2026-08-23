#!/usr/bin/env bash
# Launch N replicate runs of the L1 skill-MERGE arm on one GPU, staggered in time.
#
#   bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/launch_merge_reps.sh [subcommand] [overrides]
#
#   (no args)   preflight, then launch REPS runs staggered by LAG_SEC
#   dry-run     preflight + print the exact commands, launch nothing
#   status      merge-health report for replicates already launched
#
# Overrides are environment variables, e.g.:
#   GPU=0 REPS=2 LAG_SEC=120 SIM=0.85 bash .../launch_merge_reps.sh
#
# ---------------------------------------------------------------------------
# Why this exists instead of calling launch_run.sh three times
# ---------------------------------------------------------------------------
# 1. launch_run.sh launches with a bare `uv run` (line 63) -- NOT `--no-sync`.
#    `uv sync --dry-run` in this repo currently reports:
#        - scikit-learn==1.7.2  -> + scikit-learn==1.5.0
#        - scipy==1.15.3        -> + scipy==1.11.4
#        - pytest, ruff, pluggy, iniconfig  (uninstalled)
#    because pyproject.toml declares scikit-learn but uv.lock predates it.
#    The deletion arm's log opens with "Uninstalled 1 package / Installed 1
#    package" -- proof this already happened on a real run. Three concurrent
#    `uv run` syncs racing on one shared .venv, while another arm is mid-run,
#    is how you corrupt several hundred GPU-hours at once. We use --no-sync.
#
# 2. launch_run.sh refuses any GPU with >1000 MiB used. That check is correct
#    for a solo arm but makes deliberate GPU sharing impossible: replicate 2
#    would be refused because replicate 1 is already resident. We check *free*
#    memory instead, and cap the number of co-resident arms.
#
# 3. Three arms on one GPU need KB_GPU_RESERVE_GB=0. The default is 42.0 GB
#    (kernelbench_integration/governor.py:149) and its own comment says to zero
#    it when sharing, because N reservers fight for headroom and OOM whichever
#    arm is mid-eval. 3 x 42 GB = 126 GB of a 143 GB card leaves nothing for the
#    evals themselves. Reserve 0 is a supported path: GPUMemoryReserver.acquire()
#    returns early when reserve_bytes <= 0.
#
# 4. Merge arms have two silent-failure modes that no existing preflight covers,
#    and both produce a completed run with zero merges and a clean log:
#      - scikit-learn missing  -> every merge iteration dies as coder_call_error
#      - embedding endpoint down -> run_skill_merge_pass swallows its exception
#    We probe both live, before spending ~70 GPU-hours per replicate.
#
# 5. All replicates share ONE --run-name by design (chosen deliberately: the runs
#    are distinguished only by the UTC-minute suffix evolve_kb_batch.py appends).
#    That has two consequences this script handles and a plain loop would not:
#      - Log files would collide too. launch_run.sh derives the log name from the
#        run name and opens it with >>, so three arms would interleave into one
#        unreadable file. We give each replicate its own log.
#      - evolve_kb_batch.py does NOT guard against a run-dir collision on a fresh
#        run: it computes run_dir at line 1244 and mkdir(exist_ok=True) at 1291.
#        Two replicates landing in the same UTC minute would silently share a
#        directory and corrupt each other. We resolve each replicate's actual run
#        dir after launch and abort the series if one repeats.
# ---------------------------------------------------------------------------

set -euo pipefail

MODE="${1:-launch}"

# ---- tunables -------------------------------------------------------------
GPU="${GPU:-1}"                     # all replicates share this GPU
REPS="${REPS:-3}"                   # number of replicate runs
LAG_SEC="${LAG_SEC:-180}"           # stagger between launches; see note below
SIM="${SIM:-0.8}"                   # --skill-merge-similarity
CTX="${CTX:-truncation}"            # context mode held fixed on governance arms
TAG="${TAG:-merge_sim08}"           # encode non-default params here, per CLAUDE.md 3.2
RUN_NAME="${RUN_NAME:-base_agent_gpt_oss_120b_${TAG}_itr30_GH200}"
MAX_ARMS_PER_GPU="${MAX_ARMS_PER_GPU:-3}"
DIR_WAIT_SEC="${DIR_WAIT_SEC:-300}" # how long to wait for a run dir to appear
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"

# LAG_SEC must exceed 60. evolve_kb_batch.py:1205 stamps the run name with
# strftime("%Y_%m_%d_%H_%M") -- MINUTE resolution -- and the stamp is taken by
# the Python process at startup, not by this shell. A lag below one minute does
# not guarantee two replicates land in different minutes.
if [ "$LAG_SEC" -le 60 ]; then
  echo "FATAL: LAG_SEC=$LAG_SEC must be > 60 (run-name timestamp is minute-resolution)"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_HOME="${CUDA_HOME:-$HOME/opt/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$REPO_ROOT/.venv/bin:$PATH"

# Hardware/baseline is a parameter so this script works on another server:
#   HARDWARE=<folder under results/timing> bash <this script> ...
# shellcheck source=./hardware_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/../hardware_env.sh"
kb_resolve_hardware


RESULTS_ROOT="runs_evolving/gpt-oss-120b/"
STAMP="$(date -u +%b_%-d)"
# Launch writes today's manifest; `status` must find the manifest from whatever
# day the launch happened. Deriving it from the CURRENT date broke `status` the
# moment the run crossed a UTC midnight -- which a ~70h run always does.
# Newest matching manifest wins; override with MANIFEST=<path> or `status <path>`.
if [ -n "${2:-}" ] && [ "$MODE" = "status" ]; then
  MANIFEST="$2"
else
  MANIFEST="${MANIFEST:-}"
  if [ -z "$MANIFEST" ]; then
    if [ "$MODE" = "status" ]; then
      MANIFEST="$(ls -1t ${RUN_NAME}_*_reps.manifest.tsv 2>/dev/null | head -1 || true)"
      [ -z "$MANIFEST" ] && MANIFEST="${RUN_NAME}_${STAMP}_reps.manifest.tsv"
    else
      MANIFEST="${RUN_NAME}_${STAMP}_reps.manifest.tsv"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# status subcommand -- merge arms fail silently, so verify they are doing work
# ---------------------------------------------------------------------------
if [ "$MODE" = "status" ]; then
  [ -f "$MANIFEST" ] || {
    echo "no manifest: $MANIFEST"
    echo "candidates:"; ls -1t ${RUN_NAME}_*_reps.manifest.tsv 2>/dev/null | sed 's/^/  /' || echo "  (none)"
    echo "pass one explicitly:  launch_merge_reps.sh status <manifest.tsv>"
    exit 1
  }
  echo "manifest: $MANIFEST"
  printf '%-4s %-14s %-9s %-9s %-8s %s\n' REP PID EMBEDS MERGES PROBLEM RUNDIR
  while IFS=$'\t' read -r rep pid rundir log; do
    [ "$rep" = "rep" ] && continue
    alive="dead"; kill -0 "$pid" 2>/dev/null && alive="$pid"
    emb="-"
    [ -f "$rundir/l1_skill_embeddings.json" ] && emb="$(python3 -c "
import json,sys
try: print(len(json.load(open('$rundir/l1_skill_embeddings.json')).get('skills',[])))
except Exception: print('ERR')" 2>/dev/null)"
    mrg="-"
    [ -f "$rundir/l1_skill_merges.jsonl" ] && mrg="$(wc -l < "$rundir/l1_skill_merges.jsonl" | tr -d ' ')"
    prob="$(grep -E "^\[batch\] \([0-9]+/" "$log" 2>/dev/null | tail -1 | grep -oE '\([0-9]+/[0-9]+\)' || echo '-')"
    printf '%-4s %-14s %-9s %-9s %-8s %s\n' "$rep" "$alive" "$emb" "$mrg" "$prob" "$(basename "$rundir")"
  done < "$MANIFEST"
  echo
  echo "EMBEDS and MERGES must both become non-zero once a merge pass has run"
  echo "(--skill-merge-interval defaults to 50 global iterations, so allow time)."
  echo "Sanity: grep -c CUDA_HOME <log>  must stay 0."
  exit 0
fi

# ---------------------------------------------------------------------------
# preflight -- every check below corresponds to a failure that completes the run
# rather than aborting it, i.e. one you only discover ~70 GPU-hours too late
# ---------------------------------------------------------------------------
echo "=== preflight ==="
command -v nvcc  >/dev/null || { echo "FATAL: nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; exit 1; }
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH -- add .venv/bin"; exit 1; }
kb_require_hardware "$REPO_ROOT"
grep -q "NVIDIA_INF_API_KEY" .env 2>/dev/null || { echo "FATAL: NVIDIA_INF_API_KEY not in .env"; exit 1; }
[ -f "subset_selection/selected_problems_50.csv" ] || { echo "FATAL: missing subset CSV"; exit 1; }
echo "  nvcc: $(nvcc --version | tail -1)"

nvidia-smi -i "$GPU" >/dev/null 2>&1 || { echo "FATAL: no GPU $GPU"; exit 1; }
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")"
echo "  GPU $GPU free: ${free_mib} MiB"
[ "$free_mib" -lt "$MIN_FREE_MIB" ] && { echo "FATAL: GPU $GPU has only ${free_mib} MiB free (< $MIN_FREE_MIB)"; exit 1; }

# Count arms already pinned to this GPU. Unlike launch_run.sh's blanket
# >1000 MiB refusal, this permits deliberate sharing but still bounds it.
existing=0
for p in $(pgrep -f "evolve_kb_batch" 2>/dev/null || true); do
  vis="$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' || true)"
  [ "$vis" = "$GPU" ] && existing=$((existing + 1))
done
existing=$((existing / 2))   # each arm shows as a `uv` wrapper + its python child
echo "  arms already on GPU $GPU: $existing"
if [ $((existing + REPS)) -gt "$MAX_ARMS_PER_GPU" ]; then
  echo "FATAL: $existing existing + $REPS new > MAX_ARMS_PER_GPU=$MAX_ARMS_PER_GPU on GPU $GPU."
  echo "       Raise MAX_ARMS_PER_GPU to override, or place some replicates on another GPU."
  exit 1
fi

# --no-sync is load-bearing; show exactly what a bare `uv run` would have done.
echo "  what a bare 'uv run' would do to the shared .venv:"
uv sync --dry-run 2>&1 | sed -r 's/\x1b\[[0-9;]*m//g' | grep -E '^\s*[-+~] ' | sed 's/^/    /' || echo "    (no venv changes pending)"

echo "  probing CUDA extension build, sklearn, and the embedding endpoint ..."
uv run --no-sync python - <<'PY'
import sys, torch
from torch.utils.cpp_extension import load_inline

# (1) A missing/broken nvcc does not fail the run -- load_inline falls back and
# the agent scores correct=True on plain PyTorch. See env/README.md.
src = r'''
#include <torch/extension.h>
__global__ void k(float* x, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.0f;}
torch::Tensor f(torch::Tensor x){int n=x.numel(); k<<<(n+255)/256,256>>>(x.data_ptr<float>(), n); return x;}
'''
m = load_inline(name="merge_reps_probe", cpp_sources="torch::Tensor f(torch::Tensor x);",
                cuda_sources=src, functions=["f"], verbose=False)
if m.f(torch.zeros(8, device="cuda")).sum().item() != 8.0:
    print("    CUDA extension: WRONG RESULT"); sys.exit(1)
print("    CUDA extension build+run: OK")

# (2) skill_merge_cluster.py imports sklearn for DBSCAN. Without it every
# --skill-merging iteration dies as coder_call_error.
import sklearn
print(f"    sklearn: OK ({sklearn.__version__})")

# (3) run_skill_merge_pass swallows its own exceptions when verbose is off, so a
# dead embedding endpoint yields a completed run with zero merges and a clean
# log. This is the single most important check for a merge arm.
from dotenv import load_dotenv
load_dotenv(".env")   # explicit: find_dotenv() cannot walk frames from a stdin heredoc
# evolving_common is not installed into the venv; evolve_kb_batch.py:22-24
# puts it on sys.path at import time. Mirror that here.
import os
sys.path.insert(0, os.path.join(os.getcwd(), "Self-Evolving-Agent"))
from evolving_common.llm_client import embed_texts_nvidia, get_skill_merge_embed_model_id
model = get_skill_merge_embed_model_id()
vecs = embed_texts_nvidia(["fuse elementwise add into the preceding matmul epilogue",
                           "use warp-shuffle reduction instead of shared memory"])
if len(vecs) != 2 or not vecs[0]:
    print(f"    embeddings: EMPTY RESPONSE from {model}"); sys.exit(1)
import math
a, b = vecs
cos = sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))
print(f"    embeddings: OK ({model}, dim={len(a)}, sample cos={cos:.3f})")
PY

echo "=== preflight passed ==="
echo

# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------
CMD_PREVIEW="CUDA_VISIBLE_DEVICES=$GPU KB_GPU_RESERVE_GB=0 nohup uv run --no-sync python \\
  scripts_integration/new_evolving_agent/evolve_kb_batch.py \\
  --run-name $RUN_NAME --results-root $RESULTS_ROOT \\
  --max-problems 50 --max-iterations 30 --hardware "$HARDWARE" \\
  --nvidia-endpoint inference --model gpt-oss-120b \\
  --context-management $CTX --coder-timeout-sec 600 \\
  --skill-merging --skill-merge-similarity $SIM"

echo "plan: $REPS replicate(s) of the MERGE-ONLY arm on GPU $GPU, staggered ${LAG_SEC}s"
echo "      merge-only = --skill-merging with skill_deletion left false."
echo "      gen3_stages.py:794 sets enable_skill_governance = deletion OR merging,"
echo "      and the merge pass at :933 is gated on merging alone -- so --skill-deletion"
echo "      is NOT required, despite what the --skill-merging help text claims."
echo
echo "$CMD_PREVIEW"
echo

if [ "$MODE" = "dry-run" ]; then
  echo "(dry-run: nothing launched)"
  echo "run names would be ${RUN_NAME}_<UTC minute>, ${LAG_SEC}s apart"
  exit 0
fi

printf 'rep\tpid\trundir\tlog\n' > "$MANIFEST"
declare -a SEEN_DIRS=()   # read under set -u before first append; guarded by ${#SEEN_DIRS[@]}

for i in $(seq 1 "$REPS"); do
  LOG="${RUN_NAME}_${STAMP}_rep${i}.log"
  # Replicates share a run name on purpose, so they would share a log too.
  # Separate logs keep each replicate independently diagnosable.
  : > "$LOG"

  # Belt-and-braces on top of LAG_SEC: never launch inside the same UTC minute
  # as the previously claimed run dir, whose name ends in _YYYY_MM_DD_HH_MM.
  if [ "${#SEEN_DIRS[@]}" -gt 0 ]; then
    prev_min="$(basename "${SEEN_DIRS[-1]}" | grep -oE '[0-9]{4}(_[0-9]{2}){4}$' || true)"
    while [ -n "$prev_min" ] && [ "$prev_min" = "$(date -u +%Y_%m_%d_%H_%M)" ]; do
      echo "   still inside UTC minute $prev_min; waiting for the boundary"
      sleep 10
    done
  fi

  before="$(ls -1d ${RESULTS_ROOT}${RUN_NAME}_2* 2>/dev/null | sort || true)"

  echo ">> [rep $i/$REPS] launching on GPU $GPU  -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" KB_GPU_RESERVE_GB=0 nohup uv run --no-sync python \
    scripts_integration/new_evolving_agent/evolve_kb_batch.py \
    --run-name "$RUN_NAME" \
    --results-root "$RESULTS_ROOT" \
    --max-problems 50 \
    --max-iterations 30 \
    --hardware "$HARDWARE" \
    --nvidia-endpoint inference \
    --model gpt-oss-120b \
    --context-management "$CTX" \
    --coder-timeout-sec 600 \
    --skill-merging \
    --skill-merge-similarity "$SIM" \
    >> "$LOG" 2>&1 &
  pid=$!
  echo "   pid=$pid (uv wrapper)"

  # Resolve which directory this replicate actually claimed. run_dir is mkdir'd
  # at evolve_kb_batch.py:1291, after torch import and the CUDA check, so this
  # can take a while on a cold start.
  rundir=""
  for _ in $(seq 1 "$DIR_WAIT_SEC"); do
    sleep 1
    after="$(ls -1d ${RESULTS_ROOT}${RUN_NAME}_2* 2>/dev/null | sort || true)"
    new="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | head -1)"
    if [ -n "$new" ]; then rundir="$new"; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "   FATAL: rep $i exited before creating a run dir. Tail of $LOG:"
      tail -20 "$LOG" | sed 's/^/     /'
      exit 1
    fi
  done

  if [ -z "$rundir" ]; then
    # This is the branch a run-name COLLISION actually lands in. When two
    # replicates share a UTC minute, the second one resolves to the first's
    # path and mkdir(exist_ok=True) creates nothing -- so "no new directory"
    # means either a slow start or, worse, a live process now writing into the
    # previous replicate's run dir. Kill it before it does damage either way.
    echo "   FATAL: rep $i produced no new run dir within ${DIR_WAIT_SEC}s."
    echo "          Either a slow start, or a run-name collision (it adopted an"
    echo "          earlier replicate's directory). Killing rep $i and aborting."
    pkill -P "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true
    sleep 2; pkill -9 -P "$pid" 2>/dev/null || true; kill -9 "$pid" 2>/dev/null || true
    echo "   Tail of $LOG:"
    tail -20 "$LOG" | sed 's/^/     /'
    exit 1
  fi

  # Second net. Verified by sandbox test: a same-minute collision actually
  # surfaces as *no new directory* (handled above), not as a repeated one, so
  # this loop should never fire. It stays as a cheap invariant assertion in case
  # evolve_kb_batch.py ever starts disambiguating run dirs itself.
  for seen in ${SEEN_DIRS[@]+"${SEEN_DIRS[@]}"}; do
    if [ "$seen" = "$rundir" ]; then
      echo "   FATAL: rep $i landed in an already-claimed run dir: $rundir"
      echo "          Killing rep $i (seconds old) and aborting the series."
      pkill -P "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true
      echo "          Increase LAG_SEC and relaunch the remaining replicates."
      exit 1
    fi
  done
  SEEN_DIRS+=("$rundir")

  printf '%s\t%s\t%s\t%s\n' "$i" "$pid" "$rundir" "$LOG" >> "$MANIFEST"
  echo "   run dir: $rundir"

  if [ "$i" -lt "$REPS" ]; then
    echo "   waiting ${LAG_SEC}s before rep $((i + 1)) (guarantees a distinct UTC minute)"
    sleep "$LAG_SEC"
  fi
done

echo
echo "=== launched $REPS replicate(s) ==="
column -t -s$'\t' "$MANIFEST"
echo
echo "manifest: $MANIFEST   (run names are identical by design; this is the only pid->dir map)"
cat <<EOF

Watch:   tail -f ${RUN_NAME}_${STAMP}_rep1.log
Status:  bash scripts_integration/new_evolving_agent/env/NVIDIA_GH200x2_2nd/launch_merge_reps.sh status
Health:  uv run --no-sync python scripts_integration/new_evolving_agent_analysis/checkpoint_run.py --auto
         grep -c CUDA_HOME ${RUN_NAME}_${STAMP}_rep1.log   # must stay 0

Note: torch._inductor "No valid triton configs" / "OutOfMemoryError: triton_mm"
tracebacks are benign autotuner noise, not failures.
EOF
