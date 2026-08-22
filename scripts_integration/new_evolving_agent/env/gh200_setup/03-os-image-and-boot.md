# Step 1 — OS image, boot, and host prep

*Part of the [2 × GH200 host setup guide](README.md).*

---

Target image: **`ubuntu-24.04-aarch64-standard-uefi`**. That matches the source host
on the two things that matter most — Ubuntu 24.04 (noble) and `aarch64` — so nothing
in this guide needs architecture substitution.

But it is a **stock** image, and the source host was provisioned from a vendor
bootstrap image. Three classes of thing are therefore missing on a fresh install, and
one of them (movable-memory onlining) is a hand-rolled systemd unit that exists in no
package at all. Do this file before the driver.

## 1. Secure Boot — check this first

UEFI plus Secure Boot is the classic fresh-install failure: DKMS builds the NVIDIA
modules fine, the kernel refuses to load them because they are unsigned, and
`nvidia-smi` reports

```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

with no hint that signing is the cause.

```bash
sudo apt install -y mokutil
mokutil --sb-state
```

The source host reports:

```
SecureBoot disabled
Platform is in Setup Mode
```

If yours says `SecureBoot enabled`, pick one **before** installing the driver:

- **Disable Secure Boot in firmware** — matches the source host exactly, and is the
  simplest option on a dedicated compute node.
- **Enroll a MOK** — `apt install nvidia-driver-580-open` prompts for a one-time
  password, then you confirm enrolment in the blue MOK Manager screen on the next
  reboot. This requires console access at boot; over IPMI/serial that is doable but
  easy to miss and the install silently leaves you with unloadable modules if you do.

Confirm UEFI while you are here:

```bash
[ -d /sys/firmware/efi ] && echo "UEFI boot: yes"       # source host: yes
```

## 2. GH200 movable-memory onlining — not in any package

The source host runs a **hand-written** unit at
`/etc/systemd/system/gh200-memory-online.service`. It is not shipped by the driver,
by `nvidia-*` packages, or by Ubuntu. A stock image will not have it. Recreate it
verbatim:

```bash
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

sudo systemctl daemon-reload
sudo systemctl enable --now gh200-memory-online.service
cat /sys/devices/system/memory/auto_online_blocks     # must print: online_movable
```

What it does: sets the policy for **hot-added memory blocks** to come online in
`ZONE_MOVABLE`. On GH200 the GPU's HBM is surfaced to the kernel as hot-added,
CPU-less NUMA nodes, so this determines how ~146 GB per GPU is admitted. The ordering
matters as much as the value — `Before=systemd-modules-load.service` means the policy
is in place *before* the NVIDIA modules load, and `Before=nvidia-persistenced.service`
before the persistence daemon opens the devices. Enabling it late, or setting
`auto_online_blocks` by hand after boot, does not retroactively change blocks that are
already online.

The source host has **9955** memory blocks and `auto_online_blocks = online_movable`.

## 3. Verify the NUMA topology came up correctly

This is the check that tells you §2 actually worked:

```bash
sudo apt install -y numactl
numactl -H | head -8
numactl -H | grep -E "^node (2|10) size"
nvidia-smi topo -m
```

Source host — 18 NUMA nodes, the two GPUs appearing as CPU-less nodes **2** and **10**:

```
available: 18 nodes (0-17)
node 0 cpus: 0-71     size: 483094 MB      # Grace socket 0
node 1 cpus: 72-143   size: 481158 MB      # Grace socket 1
node 2 cpus: (none)   size: 146176 MB      # GPU0 HBM
node 10 cpus: (none)  size: 146176 MB      # GPU1 HBM
```

and from `nvidia-smi topo -m`: GPU0 ↔ GPU1 over `NV18`, GPU0 affine to CPUs 0-71 /
NUMA node 0 / GPU NUMA ID 2; GPU1 to CPUs 72-143 / node 1 / GPU NUMA ID 10.

**If nodes 2 and 10 report `size: 0 MB`**, the GPU memory did not come online — go
back to §2, then reboot. Nothing in the run path pins CPUs or memory, so you do not
need to tune anything further; you only need the memory present.

## 4. Kernel parameters — there are none

Do not add GH200-specific boot flags speculatively. The source host's cmdline is bare:

```
BOOT_IMAGE=/boot/vmlinuz-6.8.0-136-generic root=UUID=... ro console=tty0 console=ttyS0,115200
```

and `/etc/default/grub` carries only

```
GRUB_CMDLINE_LINUX_DEFAULT="console=tty0 console=ttyS0,115200"
GRUB_CMDLINE_LINUX=""
```

No `iommu=`, no `numa=`, no `memhp_default_state=` — the systemd unit in §2 handles
memory onlining instead.

## 5. Freeze automatic upgrades

The source host has these **masked** (symlinked to `/dev/null`):

```
apt-daily.service   apt-daily-upgrade.service   unattended-upgrades.service
fwupd.service       rsync.service
```

This is deliberate and worth replicating. An unattended kernel upgrade triggers a DKMS
rebuild and, on the next reboot, a different running kernel — in the middle of a ~70 h
arm that is an expensive surprise. Arms are also killed outright by an unexpected
reboot.

```bash
sudo systemctl mask apt-daily.service apt-daily-upgrade.service unattended-upgrades.service
```

Do your `apt upgrade`s deliberately, between runs.

## 6. nouveau must be out of the way

The driver package blacklists it; verify rather than assume:

```bash
lsmod | grep -i nouveau        # must print nothing
cat /etc/modprobe.d/exclude-nouveau.conf
```

Source host contents:

```
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```

## 7. What else the stock image lacks — and whether you care

| present on source host | from | needed for these runs? |
|---|---|---|
| `gh200-memory-online.service` | hand-written | **yes** — §2 |
| `nvidia-persistenced.service` | `nvidia-compute-utils-580` | yes — enabled in [Driver](04-driver.md) |
| `nvidia-cdi-refresh.service` | `nvidia-container-toolkit` | no |
| `nvcexporter.service` | NVIDIA-internal telemetry | no |
| `nvidia-container-toolkit` 1.19.1, `docker` | apt, NVIDIA repo | no — the agent runs bare-metal |
| `chronyd` (time sync) | apt | not required, but recommended on a long-running box |

The apt mirror also differs: the source host points at NVIDIA's internal artifactory
(`urm.nvidia.com/artifactory/ubuntu-ports-remote`) rather than `ports.ubuntu.com`. A
stock image will use ports. That is fine — what matters is the **driver version you
end up with**, not where it came from. Check the version, not the mirror.

---

[← Prerequisites](02-prerequisites.md) · [Index](README.md) · [NVIDIA driver →](04-driver.md)
