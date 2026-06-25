# kdump, crash & kgdb

## Overview

When a userspace program dies you get a [core dump](core_dump.md) and open it in
[gdb](gdb.md). When the **kernel** itself panics or oopses, the machine that would inspect the
wreckage is the same machine that just crashed — so you need a different mechanism. **kdump**
solves this by booting a *second, pre-reserved* kernel via `kexec` the instant the first one
panics, just to copy the dead kernel's memory (`/proc/vmcore`) to disk. You then analyse that
`vmcore` offline with the **`crash`** utility (a kernel-aware gdb), or, for live work, attach
**kgdb** to a running kernel over a serial line. This page is the kernel counterpart to
`core_dump.md` and the post-mortem follow-on to [linux_kernel.md](linux_kernel.md).

```
   userspace crash            kernel crash
   ----------------           ------------
   SIGSEGV → core dump        panic/oops → kexec into capture kernel
   gdb ./prog core            crash vmlinux /proc/vmcore
   (same DWARF machinery, different memory image)
```

## The kdump mechanism

At boot, the primary kernel reserves a slice of RAM (via the `crashkernel=` boot parameter,
e.g. `crashkernel=256M`) and pre-loads a small **capture kernel** into it with `kexec`. That
reserved region is untouched during normal operation. On panic, the kernel `kexec`s *directly*
into the capture kernel — no firmware reset, no BIOS — so the crashed kernel's memory stays
intact and addressable as `/proc/vmcore`.

```
   boot ─▶ reserve crashkernel RAM ─▶ kexec-load capture kernel (dormant)
                                              │
   ... normal operation ...                   │
                                              ▼
   PANIC ─▶ kexec into capture kernel ─▶ read old RAM as /proc/vmcore
                                       ─▶ makedumpfile → /var/crash/<date>/vmcore
                                       ─▶ reboot normally
```

Enable it with the `kdump`/`kexec-tools` service; verify with
`cat /sys/kernel/kexec_crash_loaded` (expects `1`). The capture kernel runs a tiny initramfs
whose only job is to save the dump and reboot.

## makedumpfile & vmcoreinfo

A raw `vmcore` is the size of physical RAM. **`makedumpfile`** shrinks it: it reads
**`vmcoreinfo`** (kernel symbol/struct metadata embedded for exactly this purpose) to know
which pages are free/zero/cache and can be excluded, then compresses the rest.

```bash
# dump level 31 = strip free, cache, cache-private, user, and zero pages; compress
makedumpfile -d 31 -c /proc/vmcore /var/crash/vmcore-stripped
```

## The crash utility

`crash` loads the uncompressed-or-compressed `vmcore` together with a **matching** `vmlinux`
that has debug info, and gives you a gdb-like prompt that understands kernel data structures:

```bash
crash /usr/lib/debug/.../vmlinux /var/crash/.../vmcore
```

```
   crash> bt              # backtrace of the task that panicked
   crash> bt -a           # backtrace on every CPU
   crash> ps              # process list at time of crash
   crash> log             # the kernel dmesg ring buffer (the oops text)
   crash> dmesg
   crash> struct task_struct <addr>     # decode any kernel struct
   crash> dis -l <symbol>               # disassemble with source lines
   crash> foreach bt                    # backtrace every task
```

`log`/`bt` are the first two commands in almost every triage — they recover the oops message
and the call chain that led to it.

## Reading an oops by hand

You don't always have a dump. A kernel **oops** is printed to the [kernel log](linux_kernel.md)
and often that's all you get. The register/stack dump is decodable:

```
   BUG: kernel NULL pointer dereference, address: 0000000000000000
   RIP: 0010:my_driver_probe+0x42/0x180 [my_driver]
   Call Trace:
     really_probe+0x...
     driver_probe_device+0x...
   Tainted: G           O      ← O = out-of-tree module loaded
```

- **`RIP: symbol+0x42/0x180`** — faulting instruction is at offset `0x42` into a function
  whose total size is `0x180`. Resolve to a line with
  `addr2line -e vmlinux <addr>`, `scripts/faddr2line vmlinux 'my_driver_probe+0x42'`, or pipe
  the whole trace through `scripts/decode_stacktrace.sh`.
- **Tainted flags** tell you the kernel's trust state (proprietary/out-of-tree module, prior
  warning, etc.) — always note them before chasing a bug.

## kgdb & kdb: live kernel debugging

For *interactive* source-level kernel debugging (set breakpoints, single-step the kernel),
use **kgdb** — a gdb stub built into the kernel that talks the gdb remote protocol over a
serial console (`kgdboc=ttyS0,115200`), with **kdb** as its simpler on-console front-end.

```
   target kernel (kgdboc=ttyS0)        host
   ----------------------------        ----
   sysrq-g or breakpoint  ──serial──▶  gdb vmlinux
                                       (gdb) target remote /dev/ttyS0
                                       (gdb) break some_kfunc ; continue
```

In practice this is most useful against a **QEMU** guest: boot the kernel with `-s -S` and
`gdb vmlinux` → `target remote :1234`, giving you full [gdb](gdb.md) control of the kernel
without dedicated serial hardware. This is the standard loop for [driver
development](../linux/driver_development.md) and early-boot bugs.

## Debug info

All of the above needs a `vmlinux` built with `CONFIG_DEBUG_INFO` that **matches the exact
kernel that crashed** (same build ID). Distros ship these as `kernel-debuginfo` packages, or
fetch on demand with `debuginfod`. A `vmlinux` from a different build silently produces wrong
symbols and nonsense backtraces — the single most common cause of confusing `crash` output.

## Where this connects

- **[Core dumps](core_dump.md)** — the userspace analogue; same DWARF, different image.
- **[Linux kernel debugging](linux_kernel.md)** — oops/panic basics, `dmesg`, `dynamic_debug`.
- **[gdb](gdb.md)** — the remote target for kgdb; `crash` reuses gdb's disassembler.
- **[ftrace](ftrace.md)** — `trace_dump_on_oops` writes the trace buffer into the panic log
  for "what ran just before the crash".
- **[Driver development](../linux/driver_development.md)** — QEMU+kgdb is the day-to-day loop.

## Pitfalls

- **`crashkernel=` sizing.** Too small and the capture kernel fails to boot (no dump);
  too large and you waste RAM. Modern kernels accept `crashkernel=auto` or ranged syntax.
- **Mismatched debug symbols.** A `vmlinux` that doesn't match the crashed build gives wrong
  backtraces — verify build IDs.
- **Secure Boot / kexec restrictions.** Lockdown mode can block `kexec_load`; use
  `kexec_file_load` with a signed kernel.
- **Dump never appears.** If the panic happens before kdump is armed (very early boot) or in a
  context that can't `kexec`, you get nothing — fall back to serial console + oops text.
- **Storage target full or unreachable.** kdump saving to NFS/SSH can fail silently; test the
  path with a forced crash (`echo c > /proc/sysrq-trigger`) before you rely on it.
