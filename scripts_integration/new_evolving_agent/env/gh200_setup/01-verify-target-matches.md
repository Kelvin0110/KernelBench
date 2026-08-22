# Verify the target matches

*Part of the [2 × GH200 host setup guide](README.md).*

---

The target host is **2 × GH200 on `ubuntu-24.04-aarch64-standard-uefi`** — same GPU
and same OS/architecture as the source host. That makes this a clone, not a port: same
`aarch64`/`sbsa` architecture, same compute capability 9.0, same driver line, same
CUDA debs, same torch wheel, and — importantly — the **same shipped timing baseline**
(`results/timing/NVIDIA_GH200x2/`).

The one real gap is that the target image is **stock Ubuntu** while the source host
came from a vendor bootstrap image. What that costs you is handled in
[OS image & boot](03-os-image-and-boot.md).

Nothing in this guide needs architecture substitution. But "same" is an assumption
the speedup metric depends on, so **verify it rather than assume it**.

## Run this on the target first

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,power.limit --format=csv
nvidia-smi topo -m
uname -m ; cat /etc/os-release | head -2 ; ldd --version | head -1
gcc --version | head -1
nvidia-smi -q -i 0 | grep -E "Product Architecture|Compute Mode|MIG Mode|Persistence Mode" 
nvidia-smi -q -i 0 | grep -A2 "Max Clocks"
systemctl is-active nvidia-persistenced gh200-memory-online 2>/dev/null
```

## What must match, and what it costs you if it doesn't

| property | source host value | if the target differs |
|---|---|---|
| GPU name | `NVIDIA GH200 144G HBM3e` | **stop** — a different SKU invalidates the shipped baseline; see [Timing baselines](11-timing-baselines.md) |
| compute capability | `9.0` | different SASS target; every kernel and skill in L1 is retuned for Hopper |
| GPU count | 2 | fewer GPUs just means fewer parallel arms |
| memory per GPU | 146831 MiB | affects `KB_GPU_RESERVE_GB` headroom only |
| power limit | 900 W (cap 700 W observed) | **a lower cap changes clocks and therefore every baseline number** |
| max SM / mem clock | 1980 MHz / 3201 MHz | same as above — re-validate the baseline |
| `uname -m` | `aarch64` | if x86_64, this is a port, not a clone — see the note at the bottom |
| OS | Ubuntu 24.04.4 (noble) | a different release changes the CUDA deb repo path and the gcc version |
| boot mode | UEFI, Secure Boot **disabled** | SB enabled blocks DKMS module load — [OS image & boot](03-os-image-and-boot.md) |
| `auto_online_blocks` | `online_movable` | GPU HBM may not come online as NUMA nodes 2/10 — [OS image & boot](03-os-image-and-boot.md) |
| glibc | 2.39 | torch manylinux_2_28 wheels need ≥ 2.28 |
| gcc | 13.3.0 | nvcc 12.8 supports it; older/newer needs a check |
| driver | 580.173.02 | ≥ 570 is the practical floor for the 12.8 toolkit |
| Compute Mode | `Default` | `EXCLUSIVE_PROCESS` breaks multi-arm sharing outright |
| MIG | disabled | MIG partitions would change device enumeration and timings |
| ECC | enabled | toggling ECC changes usable memory and bandwidth |

Two clock/power rows are the ones people skip and then cannot explain their numbers.
A GH200 running at a lower power cap produces genuinely slower reference times, so
speedups computed against the source host's baseline come out systematically **high**.
[Timing baselines](11-timing-baselines.md) gives the validation procedure.

## GH200-specific host bits worth checking

- **`gh200-memory-online.service`** — present and `active (exited)` on the source
  host. It is a **hand-written** unit in `/etc/systemd/system/`, shipped by no package
  at all, so a stock image will not have it.
  [OS image & boot](03-os-image-and-boot.md) has the exact file to recreate.
- **NUMA topology.** GH200 exposes GPU memory as CPU-less NUMA nodes — the source
  host reports 18 nodes, with node 2 and node 10 each carrying 146176 MB of GPU HBM,
  GPU0 affine to CPUs 0-71 (node 0) and GPU1 to CPUs 72-143 (node 1). Nothing in the
  run path pins CPUs, but nodes 2/10 reporting `0 MB` is the symptom of missing
  memory onlining.
- **GPU↔GPU link** is `NV18` (18 bonded NVLinks). The evolving-agent runs are
  single-GPU per arm and never use it; it matters only if you later add multi-GPU
  evals.
- **No fabricmanager.** Grace-Hopper superchips are directly C2C/NVLink-coupled, so a
  2-GPU GH200 node needs no `nvidia-fabricmanager`. The source host does not run it.
  (Only an HGX/NVSwitch baseboard would, and then it must match the driver version
  exactly or `nvidia-smi` lists the GPUs while CUDA init fails.)
- **apt mirror.** The source host's `/etc/apt/sources.list.d/nvidia-bootstrap.list`
  points at NVIDIA's internal artifactory
  (`urm.nvidia.com/artifactory/ubuntu-ports-remote`) rather than `ports.ubuntu.com`.
  Driver packages come from there. If the target uses stock Ubuntu ports, package
  versions may differ slightly — check the driver version, not the mirror.
- **All driver packages are `:arm64`.** `dpkg -l | grep nvidia` on the target should
  show `:arm64` suffixes, never `:amd64`.

## If the target turns out to be x86_64 after all

Then it is a port, and exactly three substitutions are needed in
[install_cuda128_local.sh](../install_cuda128_local.sh): the repo path
`ubuntu2404/sbsa` → `ubuntu2404/x86_64`, the deb suffix `_arm64.deb` → `_amd64.deb`,
and the target dir `targets/sbsa-linux` → `targets/x86_64-linux`. All ten component
debs were confirmed to exist for x86_64 at identical version numbers on 2026-08-22,
and `uv.lock` already carries the x86_64 torch wheel. Everything else in this guide
is architecture-neutral — except the baseline, which would then have to be
regenerated from scratch.

---

[← Source host inventory](00-source-host-inventory.md) · [Index](README.md) · [Prerequisites →](02-prerequisites.md)
