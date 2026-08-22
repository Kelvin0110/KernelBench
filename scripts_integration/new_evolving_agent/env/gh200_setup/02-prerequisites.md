# Prerequisites

*Part of the [2 × GH200 host setup guide](README.md).*

---

- sudo on the target host **for the OS prep and the driver only** — Secure Boot, the
  memory-onlining unit, masking auto-upgrades, `nvidia-driver-580-open`, and a couple
  of small tools (`mokutil`, `numactl`). Everything else (CUDA toolkit, Python, venv)
  is installed strictly under `$HOME` and the repo.
- Console or IPMI access at boot **if Secure Boot is enabled** and you plan to enrol a
  MOK rather than disable it.
- Outbound HTTPS to: `developer.download.nvidia.com`, `download.pytorch.org`,
  `pypi.org`, `github.com`, `astral.sh`, and `inference-api.nvidia.com`.
- A checkout location. This guide uses `$REPO=$HOME/KernelBench`; adjust freely —
  nothing outside `launch_run.sh`'s own `REPO_ROOT` derivation hardcodes a path.

---

[← Verify the target matches](01-verify-target-matches.md) · [Index](README.md) · [OS image & boot →](03-os-image-and-boot.md)
