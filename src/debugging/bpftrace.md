# bpftrace

## Overview

**bpftrace** is a high-level tracing language for Linux — think `awk` for kernel and
userspace events. You write a one-liner like `bpftrace -e 'tracepoint:syscalls:sys_enter_openat
{ @[comm] = count(); }'` and it compiles to an [eBPF](../linux/ebpf.md) program, attaches it to
the right hooks, and aggregates results in-kernel. It sits at the top of the same stack as
[ftrace](ftrace.md) and [kprobes/uprobes](kprobes_uprobes.md): where ftrace is the *stable,
in-tree* control surface you drive by hand through tracefs, and raw eBPF is a full programming
exercise in C + libbpf, bpftrace is the **ergonomic middle** — concise, ad-hoc, safe, and
ideal for production "what is happening right now" questions. (`linux/ebpf.md` is the
kernel-subsystem reference; this page is the practical front-end.)

```
   bpftrace one-liner / .bt script
        │  (compiles)
        ▼
   eBPF bytecode → verifier → JIT          ← shares the engine with BCC/libbpf
        │  (attaches to)
        ▼
   kprobes · tracepoints · uprobes · USDT · perf events · profile timers
        │
        ▼
   in-kernel maps  ──(aggregate: count/hist/sum)──▶  printed on exit
```

Compared to its neighbours: **ftrace** = lowest overhead, always present, manual; **BCC** =
Python+C, more boilerplate, good for shipped tools; **bpftrace** = fastest to *write*, best
for interactive investigation.

## Probe types

A program is one or more `probe { action }` blocks. The probe selects *where* to attach:

```
   kprobe:vfs_read           kretprobe:vfs_read        # kernel fn entry / return
   tracepoint:syscalls:sys_enter_openat                # static kernel tracepoint
   uprobe:/bin/bash:readline  uretprobe:/bin/bash:readline   # userspace fn
   usdt:/usr/lib/libc.so:...                            # static userspace probe
   profile:hz:99             interval:s:1               # timer-driven sampling
   software:faults:1  hardware:cache-misses:1000000     # perf events
   BEGIN { }   END { }                                  # start / cleanup
```

`profile:hz:99` (sample all CPUs 99×/s) is the workhorse for CPU profiling and flame-graph
stacks — the same sampling idea as [perf](perf_profiling.md), scripted.

## Language essentials

- **Maps** — `@name[key] = value` are kernel hash maps that survive across events and print
  automatically on exit. `@` alone is an anonymous scalar.
- **Aggregations** — `count()`, `sum(x)`, `avg(x)`, `min`/`max`, and the histogram builders
  `hist(x)` (power-of-two buckets) and `lhist(x, lo, hi, step)` (linear). Aggregation happens
  *in kernel*, so only compact results cross to userspace.
- **Builtins** — `pid`, `tid`, `comm` (process name), `cpu`, `nsecs`, `kstack`/`ustack`
  (stack traces), `arg0..argN` (function args), `retval`, `func`, `probe`.
- **Actions** — `printf()`, `print(@map)`, `delete()`, `clear()`, conditional `if`/filters
  via `/predicate/`.

```
   // latency histogram of read() in microseconds
   kprobe:vfs_read  { @start[tid] = nsecs; }
   kretprobe:vfs_read /@start[tid]/ {
       @us = hist((nsecs - @start[tid]) / 1000);
       delete(@start[tid]);
   }
```

## Cookbook

```bash
# Count syscalls by process name
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# Which processes open which files
bpftrace -e 'tracepoint:syscalls:sys_enter_openat {
    printf("%s %s\n", comm, str(args->filename)); }'

# Histogram of read() sizes returned
bpftrace -e 'tracepoint:syscalls:sys_exit_read { @bytes = hist(args->ret); }'

# CPU profile: 99 Hz kernel+user stacks, top of on-CPU time
bpftrace -e 'profile:hz:99 { @[kstack] = count(); }'

# malloc sizes from a target binary via uprobe
bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc {
    @sizes = hist(arg0); }'

# New processes (exec) with arguments
bpftrace -e 'tracepoint:syscalls:sys_enter_execve {
    printf("%s -> %s\n", comm, str(args->filename)); }'
```

## Discovery

```bash
bpftrace -l                                  # list ALL probes (huge)
bpftrace -l 'tracepoint:syscalls:*'          # filter by glob
bpftrace -l 'kprobe:vfs_*'
bpftrace -lv tracepoint:syscalls:sys_enter_openat   # show its arg fields
bpftrace -l 'usdt:/usr/lib/.../libpthread.so:*'     # USDT probes in a binary
```

`-lv` is the one to remember — it prints the `args->` field names you reference in the action,
saving you from guessing tracepoint formats (which otherwise live under tracefs
`events/*/format`, the [ftrace](ftrace.md) source of truth).

## Where this connects

- **[ftrace & tracefs](ftrace.md)** — bpftrace attaches to the same tracepoints/kprobe sites;
  ftrace is the lower-level, zero-dependency fallback.
- **[kprobes & uprobes](kprobes_uprobes.md)** — the dynamic attach points bpftrace targets by
  name.
- **[eBPF](../linux/ebpf.md)** — the engine bpftrace compiles down to; reach for raw
  libbpf/BCC when you outgrow one-liners.
- **[perf & flame graphs](perf_profiling.md)** — `profile:hz` stacks feed the same
  flame-graph workflow.

## Pitfalls

- **Kernel version & BTF.** Modern bpftrace leans on CO-RE/BTF (`CONFIG_DEBUG_INFO_BTF`);
  on old kernels without it you need matching kernel headers, and some probe types are
  unavailable.
- **Overhead is real.** A `kprobe` on a hot path (e.g. `vfs_read` on a busy box) adds latency
  to *every* call — measure before leaving it running in production.
- **Dropped events.** High-frequency `printf` per-event can overflow the perf ring buffer
  (`Lost N events`); prefer in-kernel aggregation (`count()`/`hist()`) over printing every hit.
- **kretprobe pairing leaks.** If you stash `@start[tid]` on entry but the return probe misses
  (process exits mid-call), the map entry leaks — guard with predicates and `delete()`.
- **Privileges.** Needs root or `CAP_BPF`+`CAP_PERFMON`; often blocked inside containers.
