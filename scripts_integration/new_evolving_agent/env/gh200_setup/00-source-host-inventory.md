# Source host inventory

*Part of the [2 × GH200 host setup guide](README.md).*

---

| layer | value on the GH200 host | how it was installed |
|---|---|---|
| OS | Ubuntu 24.04.4 LTS (noble), kernel `6.8.0-136-generic` | vendor bootstrap image |
| boot | UEFI; Secure Boot **disabled** (platform in Setup Mode) | firmware |
| kernel cmdline | `console=tty0 console=ttyS0,115200` — no GPU-specific params | — |
| memory onlining | `auto_online_blocks = online_movable`, 9955 blocks | hand-written systemd unit |
| NUMA | 18 nodes; node 2 and node 10 = 146176 MB GPU HBM each | — |
| DKMS | `nvidia/580.173.02` built for 6.8.0-136 and 6.8.0-124 | apt |
| auto-upgrades | `apt-daily`, `apt-daily-upgrade`, `unattended-upgrades` **masked** | deliberate |
| CPU arch | `aarch64` / `arm64`, 2 × 72-core Neoverse-V2, 18 NUMA nodes | — |
| glibc | 2.39-0ubuntu8.8 | image |
| GPU | 2 × **NVIDIA GH200 144G HBM3e**, compute capability **9.0**, 146831 MiB each | — |
| GPU interconnect | `NV18` (18 bonded NVLinks) GPU0↔GPU1 | — |
| Driver | **580.173.02** (`nvidia-driver-580-open`, open kernel modules) | apt |
| Driver-advertised CUDA | 13.0 (backward compatible with the 12.8 toolkit) | — |
| CUDA toolkit | **12.8, V12.8.93** at `$HOME/opt/cuda-12.8` | **userspace `dpkg -x`, no sudo** |
| gcc / g++ | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) | apt (default noble) |
| ninja | 1.13.0 (pip wheel, `.venv/bin/ninja`) — **not** on system PATH | uv |
| uv | **0.12.0**, `~/.local/bin/uv` | standalone installer |
| Python | **CPython 3.10.20**, uv-managed (`~/.local/share/uv/python/cpython-3.10-linux-<arch>-gnu`) | uv |
| venv | `<repo>/.venv`, 139 packages, 6.5 GB | `uv sync` |
| torch | **2.11.0+cu128**, cuDNN 9.19.0, NCCL 2.28.9, arch list `sm_80;sm_90;sm_100;sm_120` | `download.pytorch.org/whl/cu128` |
| triton | 3.6.0 | uv |
| container stack | `nvidia-container-toolkit` 1.19.1, docker present | apt (unused by these runs) |
| services | `nvidia-persistenced` (apt); `gh200-memory-online.service` active | hand-written unit — **not in any package** |

Disk: `/` is 880 G (`/dev/nvme1n1p2`), 715 G free. Budget **~8 GB** for the venv +
toolkit, plus room for `runs_evolving/` (each 50×30 arm writes a few hundred MB).

---

[Index](README.md) · [Verify the target matches →](01-verify-target-matches.md)
