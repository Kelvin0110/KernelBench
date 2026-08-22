# Standing up a second KernelBench evolving-agent host (2 × GH200)

Target: a fresh **2 × NVIDIA GH200** box on **`ubuntu-24.04-aarch64-standard-uefi`**,
which must reproduce the toolchain of the current host (`lego-c2g2-smc-034`,
`/localhome/local-tianzheng/KernelBench`) closely enough that its experiment arms are
comparable.

Everything in these files was read off the live host on **2026-08-22**, not from
memory.

Same GPU, same OS, same architecture — so this is a clone rather than a port. No
substitutions, the repo's own CUDA installer runs unmodified, and the shipped
`NVIDIA_GH200x2` timing baseline applies directly. What it is *not* is automatic: the
target image is **stock Ubuntu** while the source host came from a vendor bootstrap
image, and four things still bite.

---

## Read in order

| # | file | what it covers |
|---|---|---|
| — | [00-source-host-inventory.md](00-source-host-inventory.md) | every version on the source host, in one table |
| — | [01-verify-target-matches.md](01-verify-target-matches.md) | the checklist that confirms "same hardware" is actually true |
| — | [02-prerequisites.md](02-prerequisites.md) | sudo scope, network egress, checkout location |
| 1 | [03-os-image-and-boot.md](03-os-image-and-boot.md) | Secure Boot, GH200 memory onlining, NUMA, frozen auto-upgrades |
| 2 | [04-driver.md](04-driver.md) | `nvidia-driver-580-open`, DKMS, persistence daemon |
| 3 | [05-uv.md](05-uv.md) | uv 0.12.0; it fetches CPython 3.10.20 itself |
| 4 | [06-repository.md](06-repository.md) | clone, submodules, **and the uncommitted patches a clone won't give you** |
| 5 | [07-python-environment.md](07-python-environment.md) | `uv sync --extra dev`, and the `--no-sync` rule |
| 6 | [08-cuda-toolkit.md](08-cuda-toolkit.md) | userspace CUDA 12.8 via `dpkg -x`; why skipping it silently corrupts runs |
| 7 | [09-environment-exports.md](09-environment-exports.md) | `CUDA_HOME` + `.venv/bin` on PATH; `.env` keys |
| 8 | [10-acceptance-test.md](10-acceptance-test.md) | the gauntlet — run all of it before launching |
| 9 | [11-timing-baselines.md](11-timing-baselines.md) | reuse the shipped GH200 baseline, and how to validate that decision |
| 10 | [12-multi-arm-settings.md](12-multi-arm-settings.md) | `KB_GPU_*` knobs for sharing a GPU across arms |
| 11 | [13-first-run.md](13-first-run.md) | launch + health checks |

Scripts (in this directory):

| script | what it does |
|---|---|
| [`fix_closed_to_open_driver.sh`](fix_closed_to_open_driver.sh) | repairs a host that has the **proprietary** driver instead of `-open`; creates the memory-onlining unit; no reboot needed |
| [`acceptance_test.sh`](acceptance_test.sh) | the whole of [10-acceptance-test.md](10-acceptance-test.md) as one pass/fail run |

Reference:

- [14-failure-modes.md](14-failure-modes.md) — symptom → cause → fix, ranked by cost
- [appendix-a-venv-inventory.md](appendix-a-venv-inventory.md) — exact package pins
- [appendix-b-alternative-nvcc.md](appendix-b-alternative-nvcc.md) — apt / wheel routes, and why they weren't used

---

## The four things that still bite

1. **A stock image is missing GH200 memory onlining.** The source host runs a
   hand-written systemd unit that ships in no package; without it the GPUs' HBM may
   not come online as NUMA nodes 2 and 10. Secure Boot on a fresh UEFI install will
   also block the driver modules from loading with a misleading error. See
   [03-os-image-and-boot.md](03-os-image-and-boot.md).
2. **A plain clone is missing the narrow GPU-eval lock.** It lives in uncommitted
   working-tree changes on the source host, not in any commit, and it is what makes
   more than ~4 arms per GPU worthwhile. See [06-repository.md](06-repository.md).
3. **No CUDA toolkit means kernels silently fall back to reference PyTorch** and
   still score `correct=True, speedup≈1.0`. This is not hypothetical — it voided four
   ~70 h runs and poisoned up to 73% of their L1 skill catalogs. See
   [08-cuda-toolkit.md](08-cuda-toolkit.md).
4. **"Same GPU" has to be verified, not assumed.** A different power cap or clock
   ceiling makes the shipped baseline wrong in a way that looks like a result. See
   [01-verify-target-matches.md](01-verify-target-matches.md) and
   [11-timing-baselines.md](11-timing-baselines.md).

---

## Quick reference — the whole thing

```bash
# 0. confirm the target really is the same hardware  (01-verify-target-matches.md)
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,power.limit --format=csv
uname -m                                      # expect aarch64

# 1. OS prep  (03-os-image-and-boot.md)
sudo apt install -y mokutil numactl && mokutil --sb-state     # SecureBoot must be disabled
sudo tee /etc/systemd/system/gh200-memory-online.service >/dev/null <<'UNIT'
[Unit]
Description=Configure movable memory onlining for NVIDIA GH200
DefaultDependencies=no
Before=systemd-modules-load.service
Before=nvidia-persistenced.service
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo online_movable > /sys/devices/system/memory/auto_online_blocks'
RemainAfterExit=yes
[Install]
WantedBy=sysinit.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now gh200-memory-online.service
sudo systemctl mask apt-daily.service apt-daily-upgrade.service unattended-upgrades.service

# 2. driver (sudo, once)  -- MUST be the -open module; the proprietary one cannot
#    drive a GH200 no matter how cleanly DKMS builds it. See 04-driver.md.
sudo apt update && sudo apt install -y build-essential linux-headers-$(uname -r) nvidia-driver-580-open
sudo systemctl enable --now nvidia-persistenced && sudo reboot
# already-provisioned host with the WRONG driver? repair it without a reboot:
#   sudo bash scripts_integration/new_evolving_agent/env/gh200_setup/fix_closed_to_open_driver.sh
# after: nvidia-smi ; dkms status ; numactl -H | grep -E "^node (2|10) size"

# 3. uv
curl -LsSf https://astral.sh/uv/install.sh | sh && exec "$SHELL" -l

# 4. repo   (Self-Evolving-Agent is PRIVATE -> its HTTPS submodule URL fails)
export REPO=$HOME/KernelBench
git clone git@github.com:Kelvin0110/KernelBench.git "$REPO" && cd "$REPO"
git checkout features/evolving-agent-final
git config --local submodule."Self-Evolving-Agent".url git@github.com:Kelvin0110/Self-Evolving-Agent.git
git submodule update --init --recursive
# the narrow GPU-eval lock is COMMITTED now -- no patch transfer. Assert the pin:
[ "$(git -C Self-Evolving-Agent rev-parse HEAD)" = "$(git rev-parse HEAD:Self-Evolving-Agent)" ] || echo "PIN MISMATCH"

# 5. venv  (uv fetches CPython 3.10.20 itself)
uv sync --extra dev

# 6. userspace CUDA 12.8  (after step 5 — it symlinks into .venv)
PREFIX=$HOME/opt/cuda-12.8 VENV="$REPO/.venv" \
  bash scripts_integration/new_evolving_agent/env/install_cuda128_local.sh

# 7. environment
export CUDA_HOME=$HOME/opt/cuda-12.8
export PATH=$CUDA_HOME/bin:$REPO/.venv/bin:$PATH
cp .env.example .env && $EDITOR .env          # NVIDIA_INF_API_KEY at minimum

# 8. acceptance test — must print OK
bash scripts_integration/new_evolving_agent/env/gh200_setup/acceptance_test.sh

# 9. baseline: nothing to generate — results/timing/NVIDIA_GH200x2/ already applies.
#    Optionally validate it  (11-timing-baselines.md).
#    The launch scripts already pass --hardware NVIDIA_GH200x2; no edits needed.

# 10. launch
bash scripts_integration/new_evolving_agent/env/launch_run.sh 0 base_agent_gpt_oss_120b_itr30_GH200b truncation
```

---

[Source host inventory →](00-source-host-inventory.md)
