# Step 2 — NVIDIA driver

*Part of the [2 × GH200 host setup guide](README.md).*

---

> **Do [OS image & boot](03-os-image-and-boot.md) first.** Secure Boot must be
> resolved *before* this step, or the modules build and then refuse to load; and the
> memory-onlining unit must be in place before the modules load for the first time.

Install the **open** kernel-module driver at 580.x. The source host runs
`580.173.02-0ubuntu0.24.04.1`.

> ### ⚠ "open" is not a preference — the proprietary driver cannot drive a GH200
>
> Observed on `lego-c2g2-smc-035` (2026-08-22): the host came with
> **`nvidia-driver-580`**, the *proprietary* metapackage, installed instead of
> `nvidia-driver-580-open`. Everything looked healthy — DKMS reported
> `nvidia/580.173.02, 6.8.0-136-generic, arm64: installed`, Secure Boot was
> disabled, the modules had built cleanly — and yet `nvidia-smi` failed with
> *"couldn't communicate with the NVIDIA driver"*, no `nvidia` module was loaded,
> and there were no `/dev/nvidia*` nodes.
>
> The single reliable tell is the module **licence**:
>
> ```
> $ modinfo nvidia | grep '^license:'
> license:   NVIDIA          # proprietary -> cannot drive a GH200
> license:   Dual MIT/GPL    # open        -> correct
> ```
>
> Confirm with the package list, which is unambiguous:
>
> ```bash
> modinfo nvidia | grep '^license:'             # want: Dual MIT/GPL
> dpkg -l | grep -c 'nvidia-driver-580-open '   # 0 = wrong package installed
> ```
>
> No amount of rebuilding, re-signing, or rebooting fixes the proprietary module.
> Swapping to `-open` on `-035` made both GPUs appear immediately, with no reboot.
>
> **Do not use the firmware list as a diagnostic.** It is tempting — the module
> advertises GSP firmware only for Turing and Ampere — but this was measured on
> `-035` (2026-08-22) and it does **not** discriminate: *both* the proprietary and
> the open 580.173.02 modules report exactly `gsp_tu10x.bin` and `gsp_ga10x.bin`,
> and neither mentions GH100. Hopper GSP firmware is not exposed through
> `MODULE_FIRMWARE`. `modinfo nvidia | grep -c gh100` returns `0` on a perfectly
> working host, so it is a false negative, not a test.

### Repairing a host that has the proprietary driver

Use the scripted path — it is idempotent and verifies each step:

```bash
sudo bash scripts_integration/new_evolving_agent/env/gh200_setup/fix_closed_to_open_driver.sh
```

It creates the memory-onlining unit, flips `auto_online_blocks` to
`online_movable`, previews then performs the `apt` swap to
`nvidia-driver-580-open`, asserts the resulting module reports `Dual MIT/GPL`,
`modprobe`s `nvidia` + `nvidia_uvm`, enables `nvidia-persistenced`, and finishes
with `nvidia-smi` and a NUMA check.

**No reboot is required when the nvidia modules are not currently loaded** — which
is exactly the case on a host where the wrong driver never initialised. There is
nothing to unload, and no HBM has been onlined yet under the wrong policy, so
setting `online_movable` before the first `modprobe` is sufficient. Reboot only if
`modprobe` or `nvidia-smi` still fails afterwards.


```bash
sudo apt update
sudo apt install -y build-essential linux-headers-$(uname -r)

# pin the exact version if your mirror carries it; otherwise any 580.x is fine
sudo apt install -y nvidia-driver-580-open
# (this pulls nvidia-dkms-580-open, nvidia-kernel-source-580-open,
#  nvidia-compute-utils-580, nvidia-utils-580, libnvidia-compute-580, ...)

sudo reboot
```

Package set observed on the source host, for reference when diffing:

```
nvidia-driver-580-open            580.173.02-0ubuntu0.24.04.1
nvidia-dkms-580-open              580.173.02-0ubuntu0.24.04.1
nvidia-kernel-source-580-open     580.173.02-0ubuntu0.24.04.1
nvidia-kernel-common-580          580.173.02-0ubuntu0.24.04.1
nvidia-compute-utils-580          580.173.02-0ubuntu0.24.04.1
nvidia-utils-580                  580.173.02-0ubuntu0.24.04.1
nvidia-firmware-580-580.173.02    580.173.02-0ubuntu0.24.04.1
libnvidia-compute-580             580.173.02-0ubuntu0.24.04.1
libnvidia-cfg1-580                580.173.02-0ubuntu0.24.04.1
```

**Version floor.** The CUDA 12.8 toolkit wants driver ≥ 570; CUDA-12.x minor-version
compatibility means torch `+cu128` will run on ≥ 525. Do not go below 570 — you lose
nothing by matching 580.

Then enable persistence (the source host runs `nvidia-persistenced.service`):

```bash
sudo systemctl enable --now nvidia-persistenced
```

### Verify

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
nvidia-smi topo -m
dkms status
lsmod | grep nvidia
systemctl is-active nvidia-persistenced
```

Expect two rows, `compute_cap` **9.0**, driver 580.x. Reference output from the
source host:

```
NVIDIA-SMI 580.173.02   Driver Version: 580.173.02   CUDA Version: 13.0
0, NVIDIA GH200 144G HBM3e, 580.173.02, 146831 MiB, 9.0
1, NVIDIA GH200 144G HBM3e, 580.173.02, 146831 MiB, 9.0
```

Note `CUDA Version: 13.0` in the header is the **driver's** maximum, not the
toolkit. It is expected to disagree with `nvcc --version` (12.8). That is the
supported backward-compatible configuration, not a defect.

`dkms status` on the source host — one entry per installed kernel, which is how it
survives kernel upgrades:

```
nvidia/580.173.02, 6.8.0-136-generic, aarch64: installed
nvidia/580.173.02, 6.8.0-124-generic, arm64:   installed
```

`lsmod | grep nvidia` should show `nvidia`, `nvidia_uvm`, `nvidia_modeset`,
`nvidia_drm`, and on GH200 also `nvidia_cspmu` (the Arm CoreSight PMU module).
`nvidia_uvm` is the one that matters — CUDA will not initialise without it.

The driver package also writes `/etc/modprobe.d/nvidia-graphics-drivers-kms.conf`
automatically. No action needed; listed here so you can diff it:

```
options nvidia_drm modeset=1
options nvidia NVreg_PreserveVideoMemoryAllocations=1
options nvidia NVreg_TemporaryFilePath=/var
```

If `nvidia-smi` fails with *"couldn't communicate with the NVIDIA driver"* right after
a clean install, check Secure Boot before anything else — see
[OS image & boot](03-os-image-and-boot.md).

---

[← OS image & boot](03-os-image-and-boot.md) · [Index](README.md) · [uv →](05-uv.md)
