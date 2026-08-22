# Failure modes

*Part of the [2 × GH200 host setup guide](README.md).*

---

| symptom | cause | fix |
|---|---|---|
| `nvidia-smi: couldn't communicate with the NVIDIA driver` after a clean install | Secure Boot enabled — modules built but unsigned, so unloadable | disable SB or enrol a MOK ([OS image & boot](03-os-image-and-boot.md)) |
| same, but **Secure Boot is already disabled** and `dkms status` says `installed` | the **proprietary** `nvidia-driver-580` is installed; GH100/Hopper requires the open module. Tell: `modinfo nvidia \| grep '^license:'` prints `NVIDIA` instead of `Dual MIT/GPL`. (The firmware list is *not* a tell — both modules look identical there) | `sudo bash gh200_setup/fix_closed_to_open_driver.sh` ([Driver](04-driver.md)) |
| `git submodule update --init` → `could not read Username for 'https://github.com'` | `Self-Evolving-Agent` is a **private** repo and `.gitmodules` uses an HTTPS URL | `git config --local submodule.Self-Evolving-Agent.url git@github.com:Kelvin0110/Self-Evolving-Agent.git`, then re-run ([Repository](06-repository.md)) |
| `upload-pack: not our ref <sha>` on submodule init | the superproject pins a SEA commit that was never pushed — it only exists on the host that made it | push it from that host; do **not** silently fall back to `main`, which carries unrelated MLE-Bench work ([Repository](06-repository.md)) |
| `load_inline` → `IndexError: list index out of range` | torch cannot enumerate a GPU arch because no driver is usable; it is a driver symptom, not a toolkit one | fix the driver; or set `TORCH_CUDA_ARCH_LIST=9.0` to test the build toolchain alone |
| `numactl -H` shows nodes 2/10 at `0 MB` | `gh200-memory-online.service` missing — a stock image has no such unit | create it and reboot ([OS image & boot](03-os-image-and-boot.md)) |
| a run dies overnight after a reboot | unattended kernel upgrade | mask `unattended-upgrades` ([OS image & boot](03-os-image-and-boot.md)) |
| `CUDA_HOME environment variable is not set` | [CUDA toolkit](08-cuda-toolkit.md) skipped, or [Environment exports](09-environment-exports.md) not exported in this shell | install toolkit; export both lines |
| kernels "pass" at `speedup≈1.0` with `__global__` in the source | dead-code fallback idiom — the [CUDA toolkit](08-cuda-toolkit.md) defect | same; **discard the run**, L1 is contaminated |
| `RuntimeError: Ninja is required to load C++ extensions` | `.venv/bin` missing from `PATH` | [Environment exports](09-environment-exports.md) |
| builds fail after a `uv run` without `--no-sync` | an `nvidia-*-cu12` wheel was reinstalled, dangling toolkit symlinks | re-run [CUDA toolkit](08-cuda-toolkit.md) installer |
| `coder_call_error` on every `--skill-merging` iteration | scikit-learn missing | `uv sync --extra dev`; keep `--no-sync` after |
| speedups uniformly shifted vs the source host | target clocks/power cap differ, so the shipped baseline is wrong for it | validate per [Timing baselines](11-timing-baselines.md); chase the hardware delta in [Verify the target matches](01-verify-target-matches.md) |
| speedups wildly wrong | `--hardware` omitted, so it defaulted to `SONG_CPU6_A6000x4` | always pass `--hardware NVIDIA_GH200x2`; the launch scripts do |
| `FATAL: GPU N busy` from the launcher | `launch_run.sh:41` >1000 MiB guard vs ~550 MiB/idle arm | raise the threshold ([Multi-arm settings](12-multi-arm-settings.md)) |
| OOM mid-eval with several arms | `KB_GPU_RESERVE_GB` left at 42 | `export KB_GPU_RESERVE_GB=0` |
| `nvidia-smi` shows GPUs but CUDA init fails | HGX/NVSwitch node without fabricmanager | install `nvidia-fabricmanager-580` at the exact driver version |
| eval workers die en masse mid-run | code edited while runs live (spawn re-imports from disk) | [Multi-arm settings](12-multi-arm-settings.md) |

---

[← First run](13-first-run.md) · [Index](README.md) · [Appendix A: venv inventory →](appendix-a-venv-inventory.md)
