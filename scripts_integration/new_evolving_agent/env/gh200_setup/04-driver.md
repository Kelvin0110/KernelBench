# Step 2 — NVIDIA driver

*Part of the [2 × GH200 host setup guide](README.md).*

---

> **Do [OS image & boot](03-os-image-and-boot.md) first.** Secure Boot must be
> resolved *before* this step, or the modules build and then refuse to load; and the
> memory-onlining unit must be in place before the modules load for the first time.

Install the **open** kernel-module driver at 580.x. The source host runs
`580.173.02-0ubuntu0.24.04.1`.

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
