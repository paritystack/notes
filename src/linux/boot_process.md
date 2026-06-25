# Boot Process & initramfs

## Overview

This page follows a machine from power-on to a running userland: **firmware (UEFI/BIOS) →
bootloader (GRUB, U-Boot) → kernel + initramfs → PID 1 (systemd)**. It connects the firmware
handoff to the kernel internals in [Kernel Architecture](kernel.md), the final handoff to PID 1 in
[systemd](systemd.md), the embedded U-Boot/FIT path used by [Yocto](yocto.md) and
[OpenWrt](openwrt.md), and the hardware description consumed during early boot in
[Device Tree](device_tree.md). The ELF/loader mechanics that take over once userspace starts live
in [ELF, Linking & Loading](elf_linking.md).

The whole sequence is a chain of progressively more capable loaders, each just smart enough to
find and start the next stage.

```
   power on
      ▼
   firmware  UEFI / BIOS  (POST, find boot device)
      ▼
   bootloader  GRUB (x86) / U-Boot (embedded)
      │  loads kernel image + initramfs (+ DTB) into RAM, passes cmdline
      ▼
   kernel  decompress, init subsystems, mount initramfs as rootfs
      ▼
   initramfs /init  (find + mount the REAL root, then switch_root)
      ▼
   PID 1  /sbin/init → systemd  (bring up the system)
```

## Firmware: UEFI vs BIOS

- **Legacy BIOS** runs the 512-byte **MBR** boot code, which chain-loads a bootloader from disk.
  Limited, 16-bit, MBR partition tables only.
- **UEFI** is a richer firmware: it reads a **GPT** partition table, mounts the **EFI System
  Partition** (FAT, `/boot/efi`), and directly executes `.efi` programs (the bootloader, or even
  the kernel via the EFI stub). **Secure Boot** verifies each stage's signature up the chain.

`efibootmgr` lists/edits the firmware boot entries; `/sys/firmware/efi/` exists only when booted
in UEFI mode.

## Bootloader

The bootloader's job: locate the kernel and initramfs, load them into RAM, set the **kernel
command line**, and jump to the kernel entry point.

- **GRUB** (typical x86 distro): config at `/boot/grub/grub.cfg` (generated — edit
  `/etc/default/grub` + `update-grub`). Picks a menu entry, loads `vmlinuz` + `initrd.img`,
  passes `root=`, `ro`, etc.
- **systemd-boot** / EFI stub: simpler UEFI-only alternatives.
- **U-Boot** (embedded/ARM): a programmable bootloader with a shell and environment. Loads a
  kernel (often a **FIT image** bundling kernel + DTB + initramfs), passes the **device tree
  blob**, and sets `bootargs`. Central to [Yocto](yocto.md)/[OpenWrt](openwrt.md) BSPs.

The **kernel command line** is the key tuning surface: `root=UUID=…`, `console=ttyS0,115200`,
`ro`/`rw`, `quiet`, `init=`, `systemd.unit=`, debugging like `initcall_debug`/`earlyprintk`.

```bash
cat /proc/cmdline          # the cmdline the running kernel got
```

## Kernel early boot

The kernel image (`vmlinuz`) is a self-decompressing blob: a small setup stub decompresses the
real kernel, then it initialises core subsystems (memory management, the scheduler, timers),
runs the `initcall` chain to probe built-in drivers, and on embedded parses the **device tree**
(see [Device Tree](device_tree.md)) to learn the hardware. It then needs a root filesystem — but
the driver to *reach* the real root (NVMe, USB, encrypted, network) may be a module not yet
loaded. That chicken-and-egg is what the initramfs solves.

## initramfs: early userspace

The **initramfs** is a cpio archive the bootloader loads into RAM; the kernel unpacks it into a
tmpfs and runs its `/init` as the first userspace program. Its job is to assemble the *real* root
and pivot to it.

```
   /init (in initramfs, often a shell or systemd):
     1. load modules needed to reach root (nvme, dm-crypt, raid, network, fs)
     2. assemble root: cryptsetup luksOpen / lvchange / mdadm / iSCSI mount
     3. mount the real root read-only
     4. switch_root /newroot /sbin/init   (replace rootfs, exec real init)
```

`switch_root` (or the older `pivot_root`) replaces the in-RAM rootfs with the real one and `exec`s
the real init — the initramfs memory is freed. Distros generate the initramfs with **dracut** (or
Debian's `initramfs-tools`/`mkinitcpio`):

```bash
lsinitramfs /boot/initrd.img-$(uname -r)   # list contents (Debian)
lsinitrd                                    # dracut equivalent
dracut --force                              # regenerate
update-initramfs -u                         # Debian/Ubuntu
```

A simple embedded system may skip the pivot entirely and just *stay* in the initramfs as its
whole rootfs (common in OpenWrt-style images).

## Handoff to PID 1

The real `/sbin/init` is **PID 1** — usually [systemd](systemd.md). It mounts the rest of the
filesystems (`/etc/fstab`), starts services by dependency order, and brings the system to its
default target (`multi-user.target`/`graphical.target`). PID 1 is special: it's the ancestor that
reaps orphaned processes and may never exit.

```bash
systemd-analyze            # total boot time (firmware/loader/kernel/userspace)
systemd-analyze blame      # slowest units
systemd-analyze critical-chain
```

## Debugging boot

```bash
dmesg                          # kernel ring buffer from early boot
journalctl -b                  # this boot's log; -b -1 = previous boot
# kernel cmdline knobs:
#   earlyprintk=... / earlycon  → output before console is up
#   initcall_debug             → trace each driver initcall + timing
#   rd.break (dracut)          → drop to a shell inside initramfs
#   systemd.unit=rescue.target → minimal boot for recovery
```

## Where this connects

- [Kernel Architecture](kernel.md) — what the kernel does once the bootloader jumps to it:
  decompression, initcalls, driver probing.
- [systemd](systemd.md) — PID 1 takes over after `switch_root`; targets, units, and ordering bring
  the system up.
- [Device Tree](device_tree.md) — on embedded the bootloader passes a DTB the kernel parses to
  enumerate non-discoverable hardware.
- [Yocto](yocto.md) / [OpenWrt](openwrt.md) — U-Boot, FIT images, and read-only/initramfs root
  strategies for embedded builds.
- [Block I/O Layer](block_layer.md) — the dm-crypt/LVM/RAID stack the initramfs must assemble
  before it can mount root.

## Pitfalls

- **Forgetting to regenerate the initramfs.** After a crypto/LVM/driver change, an unrebuilt
  initramfs can't find root → boot drops to an emergency shell. Run `update-initramfs -u` /
  `dracut -f`.
- **Wrong `root=` after disk changes.** Cloning or repartitioning changes UUIDs; a stale
  `root=UUID=` strands the kernel with no root. Use UUIDs and update the bootloader config.
- **UEFI vs BIOS mismatch.** Installing a BIOS bootloader on a UEFI-booted system (or vice-versa)
  yields an unbootable disk; the partition scheme (GPT+ESP vs MBR) must match the firmware mode.
- **Secure Boot rejecting unsigned modules/kernels.** Custom or out-of-tree modules fail to load
  unless signed/enrolled (MOK) — symptoms look like missing drivers at boot.
- **Missing module in initramfs.** The driver for the root device must be *in* the initramfs;
  relying on a module on the not-yet-mounted root is the classic early-boot deadlock.
- **Editing generated configs directly.** Hand-editing `grub.cfg` is overwritten on the next
  `update-grub`; change `/etc/default/grub` instead.
