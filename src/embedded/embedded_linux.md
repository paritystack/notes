# Embedded Linux & Device Tree

## Overview

Embedded Linux is what you reach for when a microcontroller is no longer enough — when
you need a filesystem, a TCP/IP stack, multiple processes, a display stack, or off-the-
shelf libraries on a system with megabytes of RAM and an MMU. It is the world of the
[Raspberry Pi](raspberry_pi.md) and of application-class SoCs (i.MX, AM335x, STM32MP1,
Allwinner, Rockchip) running a Cortex-A core, in contrast to the bare-metal /
[RTOS](state_machines.md) firmware on Cortex-M parts elsewhere in this book. The jump
brings a **kernel/userspace split**, a real **boot chain** ([bootloader](bootloaders.md) →
kernel → init), and a hardware-description mechanism — the **Device Tree** — that replaces
the hand-written register pokes of MCU firmware.

```
   MCU FIRMWARE (Cortex-M)            EMBEDDED LINUX (Cortex-A)
  ┌──────────────────────┐          ┌──────────────────────────────┐
  │ your code = the whole │          │ userspace: your apps, libs    │ ── unprivileged
  │ system (super-loop /  │          ├──────────────────────────────┤    (MMU-isolated)
  │ RTOS), direct register│          │ kernel: drivers, scheduler,   │ ── privileged
  │ access, no MMU        │          │ filesystems, network stack    │
  └──────────────────────┘          ├──────────────────────────────┤
                                     │ Device Tree describes the HW  │
                                     └──────────────────────────────┘
```

This page covers the boot chain, the Device Tree, and the two dominant ways to *build* an
embedded Linux system — **Buildroot** and **Yocto**.

## The Boot Chain

Unlike an MCU that runs your image straight from the reset vector
([startup code](startup_code.md)), an embedded Linux SoC walks through several stages:

```
   BootROM ──► SPL / first-stage ──► U-Boot ──► Kernel (zImage/Image)
   (on-chip)   (DRAM init, tiny)     (full      + Device Tree blob (.dtb)
                                      bootloader)   + initramfs (optional)
                                          │              │
                                          ▼              ▼
                                     loads kernel    mounts root filesystem
                                     + dtb from        ──► runs init (PID 1)
                                     SD/eMMC/NAND
```

- **BootROM** — masked ROM in the SoC; sets up just enough to load the next stage from a
  boot medium (SD, eMMC, [QSPI](qspi.md) NOR, [USB](usb.md) recovery).
- **SPL / first-stage** — small enough to fit in on-chip SRAM; its job is to initialize
  external DRAM, then load the full bootloader.
- **U-Boot** — the de-facto bootloader: loads the kernel + device tree, can fetch over
  network, and exposes a shell. The Linux analogue of an MCU [bootloader](bootloaders.md);
  also where A/B and [secure-boot](secure_boot.md) verification typically live.
- **Kernel** — decompresses, parses the Device Tree, probes drivers, mounts the **root
  filesystem**, and execs **init** (systemd, SysV, or BusyBox init) as PID 1.

## Device Tree

The Device Tree solves a specific problem: the same ARM kernel binary must run on
thousands of boards with different peripherals, and the hardware is *not* discoverable the
way [USB](usb.md)/PCIe is. Instead of compiling board details into the
kernel, the board is described in a **Device Tree Source** (`.dts`) file, compiled to a
**Device Tree Blob** (`.dtb`) that the bootloader hands to the kernel. The kernel reads it
to know what hardware exists, at what addresses, on which [interrupt](interrupts.md) lines.

```dts
/ {
    soc {
        i2c0: i2c@40003000 {            // a node = a device
            compatible = "st,stm32-i2c";   // matches a kernel driver
            reg = <0x40003000 0x400>;      // MMIO base + size
            interrupts = <31>;             // IRQ line
            clock-frequency = <400000>;    // a property the driver reads

            bme280@76 {                    // child = device on this bus
                compatible = "bosch,bme280";
                reg = <0x76>;              // I2C address
            };
        };
    };
};
```

Key ideas:
- **`compatible`** strings bind a node to a kernel driver — the driver's match table lists
  the strings it supports. This is how "describe, don't code" works: adding a sensor is a
  DT edit, not a recompile.
- **Device Tree Overlays** (`.dtbo`) patch the tree at runtime to add hardware — this is
  exactly the [Raspberry Pi](raspberry_pi.md) HAT / `dtoverlay=` mechanism for enabling
  [SPI](spi.md), [I2C](i2c.md), one-wire, etc.
- **`reg`, `interrupts`, `clocks`, `gpios`, `pinctrl`** properties replace what an MCU
  driver would set by [register writes](gpio.md) — the kernel's subsystems consume them.

The DT describes *what the hardware is*, not *what to do with it*; the matching driver
supplies behavior.

## Kernel- vs Userspace

Where to put your code is the recurring embedded-Linux decision:

```
   KERNEL SPACE                        USER SPACE
   ─ device drivers (char/block/net)   ─ application logic
   ─ kernel modules (.ko)              ─ libraries (glibc/musl)
   ─ runs privileged, can crash all    ─ MMU-isolated; a crash is contained
   ─ no floating-point luxury, careful ─ normal C/C++/Python/Rust
   ─ access HW directly                ─ access HW via /dev, sysfs, ioctl
```

Modern practice pushes as much as possible into **userspace**: GPIO via `gpiod`
(character device), I2C/SPI via `/dev/i2c-*` and `spidev`, industrial I/O via the
**IIO** subsystem. Write a custom kernel driver only when you need interrupt latency,
DMA, or a subsystem integration that userspace can't reach. This mirrors the MCU rule of
keeping [ISRs](interrupts.md) thin — minimize privileged code.

## Buildroot vs Yocto

You don't download "embedded Linux"; you *build a root filesystem* for your board. Two
ecosystems dominate:

| | **Buildroot** | **Yocto / OpenEmbedded** |
|---|---|---|
| Model | Makefile-based, produces one rootfs image | Recipe/layer-based build *system* (BitBake) |
| Output | kernel + rootfs + bootloader image | same, plus an SDK and package feeds |
| Learning curve | gentle, fast first build | steep, slow first build (hours) |
| Customization | `menuconfig`, simple | layers, recipes, `.bbappend` — very powerful |
| Package mgmt | none by default (rebuild image) | optional `opkg`/`rpm` runtime feeds |
| Best for | smaller/fixed appliances, fast iteration | products, long maintenance, many variants |

Both pull cross-toolchains, build U-Boot + kernel + packages, and emit a flashable image.
**Buildroot** wins on simplicity and speed; **Yocto** wins when you need reproducible,
layered, long-lived product builds with security-update tracking. (For a single
prototype, a stock **Raspberry Pi OS** / Debian image skips all of this — see
[Raspberry Pi](raspberry_pi.md).) This is the heavyweight cousin of MCU
[build systems](build_systems.md) like CMake/PlatformIO/west.

## Board Bring-Up Checklist

```
1. Boot ROM → SPL: get DRAM up (this is where most bring-up time goes)
2. U-Boot console over serial (UART) — your lifeline; verify before anything else
3. Kernel boots, parses your .dtb — watch the boot log for driver probe failures
4. Rootfs mounts, init runs, you get a shell
5. Bring up peripherals one DT node at a time (clocks, pinmux, regulators)
6. Userspace: your application + services
```

## Where this connects

- [Raspberry Pi](raspberry_pi.md) — the most common embedded-Linux board; `config.txt` and `dtoverlay=` are Device Tree in practice.
- [Bootloaders](bootloaders.md) / [Secure Boot](secure_boot.md) — U-Boot is the Linux bootloader; verified boot and A/B updates apply here too.
- [Build Systems](build_systems.md) — Buildroot/Yocto are the application-class analogue of CMake/PlatformIO/west.
- [GPIO](gpio.md) / [I2C](i2c.md) / [SPI](spi.md) — exposed to userspace via gpiod, /dev/i2c, spidev rather than direct register access.
- [Power Management](power_management.md) — Linux runtime PM, cpufreq, and suspend replace MCU sleep-mode register pokes.
- [Startup Code & C Runtime](startup_code.md) — contrast: MCU runs your image from reset; Linux interposes a multi-stage boot chain.

## Pitfalls

1. **Editing the kernel `.config` for hardware that belongs in the Device Tree.** Board
   wiring (addresses, IRQs, pinmux) goes in the `.dts`; the kernel config selects *drivers*.
2. **`.dtb` and kernel out of sync.** A device tree describing hardware a driver doesn't
   support (or vice-versa) yields silent probe failures — read the boot log for `-ENODEV`.
3. **DRAM init wrong in SPL.** The single hardest bring-up step; subtle timing errors give
   random crashes later. Use the vendor's DDR calibration tooling, don't guess.
4. **Building a kernel driver when userspace would do.** gpiod/spidev/IIO cover most needs;
   a custom `.ko` is more privileged code to maintain and can take down the whole system.
5. **Treating Yocto like a quick build.** First builds take hours and tens of GB; for a
   one-off prototype, Buildroot or a stock distro image is far faster.
6. **No serial console.** Without UART access to U-Boot/kernel logs you are debugging blind;
   wire it up *first*.
7. **Forgetting the MMU/process model.** Unlike MCU firmware, userspace can't poke physical
   registers directly — `/dev/mem`, mmap, or a driver is required, and a process crash no
   longer halts the machine (which is usually what you want).

## See Also

- [Raspberry Pi](raspberry_pi.md) — embedded Linux you can start with today
- [Bootloaders](bootloaders.md) — U-Boot and the boot chain
- [Build Systems](build_systems.md) — the MCU-side counterpart to Buildroot/Yocto
