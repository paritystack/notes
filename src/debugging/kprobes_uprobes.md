# Kprobes & Uprobes

## Overview

Kprobes and uprobes are Linux's **dynamic instrumentation** mechanism: they let you
insert a probe at *almost any* instruction address at runtime — a kernel function for
a **kprobe**, a userspace function for a **uprobe** — without recompiling, restarting,
or rebooting anything. When execution hits the probed address, the kernel diverts to
your handler, runs it, then resumes. The `*ret*` variants (**kretprobe** / **uretprobe**)
fire on function *return* instead, which is how you capture return values and measure
per-call latency.

This is a different trade-off from the neighbours. Static [tracepoints and
ftrace](linux_kernel.md) are stable and cheap but only exist where a kernel developer
put them; [perf](perf_profiling.md) *samples* the stack periodically (great for "where
is time going", blind to rare events); dynamic probes hit *every* call to a *specific*
function you name. The probes themselves are just the attach point — the programmable
engine that usually *consumes* them is [eBPF](../linux/ebpf.md). The interactive cousin
is a debugger [breakpoint](gdb.md): same trap mechanism, but a probe doesn't stop the
world, it runs a handler and keeps going.

```
                entry                         return
  kprobe   ───►  ●  do_sys_openat2(...)         kretprobe ──► ◐  (sees retval/fd)
  uprobe   ───►  ●  ./demo:compute(int x)       uretprobe ──► ◐  (sees retval)

  mechanism: the kernel replaces the target instruction with a breakpoint trap
             (int3 / brk). on trap → run handler → single-step original insn → resume.
             optimised kprobes use a jmp instead of int3 where safe (lower overhead).
```

## How it works under the hood

**Kprobe.** When you register a probe, the kernel saves the original instruction at the
target address, then overwrites it with a breakpoint (`int3` on x86, `brk` on arm64).
Execution flow:

```
  hit address → CPU traps → kprobe pre_handler runs (your code)
              → kernel single-steps the SAVED copy of the original instruction
              → kprobe post_handler runs (optional)
              → execution resumes at address+len, as if nothing happened

  jump-optimised path (when the surrounding insns are safe to relocate):
  hit address → jmp trampoline → handler → jmp back   (no trap, ~no overhead)
```

Two cheaper variants are chosen automatically when possible:

- **Optimized (jump) kprobes** replace the `int3` with a `jmp` to a trampoline, avoiding
  the expensive trap. Controlled by `/proc/sys/debug/kprobes-optimization` (1 = on).
- **ftrace-based kprobes**: a probe placed on a function's entry reuses the `fentry`
  nop that the compiler already emitted for ftrace, so no instruction needs patching.

A handful of functions can't be probed — the trap/handler code itself, and anything
marked `__kprobes`/`nokprobe_inline`. They're listed in
`/sys/kernel/debug/kprobes/blacklist`.

**Uprobe.** A uprobe is keyed by **(inode, file-offset)**, not a process. The kernel
installs the breakpoint copy-on-write into the page cache page backing that offset, so
*every* process that maps the file hits it; the probe is refcounted and removed when the
last user detaches. Because a uprobe handler may take a **page fault** (the user page can
be swapped out), uprobe handlers are allowed to sleep — which is also why they can safely
read user memory that a kprobe can't.

## The three front-ends

You rarely poke the probe machinery by hand. Three layers sit on top, from lowest to
highest level:

```
raw tracefs   /sys/kernel/debug/tracing/{kprobe,uprobe}_events   no tools, always there
perf probe    perf probe --add ... ; perf record / perf trace    symbol + variable resolution
bpftrace      kprobe:fn { ... }  uprobe:/bin/x:fn { ... }         one-liners, maps, histograms
```

**Default to `bpftrace`** — it resolves symbols, reads arguments by name, and aggregates
in-kernel. Reach for raw tracefs only on a stripped-down box with no tooling, and for
`perf probe` when you want probe events that integrate with `perf record`/`perf trace`.

```bash
# Ubuntu/Debian
sudo apt install bpftrace linux-tools-common linux-tools-generic
```

Probing the kernel needs root (or `CAP_BPF`+`CAP_PERFMON`).

## Step-by-step: kprobe on a kernel function

**Goal:** print every file opened on the system, with PID, command, and filename, by
probing `do_sys_openat2` (the function behind the `openat`/`openat2` syscalls).

### Method A — raw tracefs (no tools)

```bash
# 1. Move to the tracing filesystem
cd /sys/kernel/debug/tracing      # (mount -t tracefs none /sys/kernel/tracing if absent)

# 2. Confirm the symbol exists and is probe-able
grep ' do_sys_openat2$' /proc/kallsyms
grep do_sys_openat2 /sys/kernel/debug/kprobes/blacklist   # empty == allowed

# 3. Register a kprobe named "myopen". do_sys_openat2(dfd, filename, how): the
#    filename is the 2nd arg, so it's in %si on x86-64 — fetch it as a string.
echo 'p:myopen do_sys_openat2 fname=+0(%si):string' > kprobe_events

# 4. Enable it
echo 1 > events/kprobes/myopen/enable

# 5. Watch live output (Ctrl-C to stop)
cat trace_pipe
#   bash-2417  [003] .... 91234.5: myopen: (do_sys_openat2+0x0/0x...) fname="/etc/ld.so.cache"

# 6. Clean up — disable, then remove the probe definition
echo 0 > events/kprobes/myopen/enable
echo '-:myopen' >> kprobe_events     # or: echo > kprobe_events  (clears all)
```

The fetch syntax `+OFFSET(REGISTER):TYPE` reads memory: `+0(%si):string` dereferences
the pointer in `%si` and reads a NUL-terminated string. See *Reading argument values*
below for the register order.

### Method B — bpftrace one-liner (same result, one line)

```bash
sudo bpftrace -e '
  kprobe:do_sys_openat2 {
    printf("%-6d %-16s %s\n", pid, comm, str(arg1));   // arg1 = filename
  }'
```

In practice you'd just use bpftrace's built-in `tracepoint` for opens, but the kprobe
form generalises to *any* function with no tracepoint. The simplest possible probe:

```bash
# Count how many times each process calls vfs_read, print on Ctrl-C
sudo bpftrace -e 'kprobe:vfs_read { @[comm] = count(); }'
```

### kretprobe — return values and latency

A kretprobe fires on return, so it can see `retval`. Pair an entry kprobe (timestamp into
a map keyed by thread id) with a kretprobe (read the timestamp, subtract) to get a
**latency histogram** — entirely in-kernel, no per-event userspace cost:

```bash
sudo bpftrace -e '
  kprobe:vfs_read       { @start[tid] = nsecs; }
  kretprobe:vfs_read /@start[tid]/ {
    @ns = hist(nsecs - @start[tid]);   # log2 histogram of latency
    delete(@start[tid]);
  }'
```

```
@ns:
[1K, 2K)    1042 |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|
[2K, 4K)     318 |@@@@@@@@@@@@@@@@                                  |
[4K, 8K)      57 |@@                                                |
```

## Step-by-step: uprobe on a userspace function

Uprobes attach to a function inside a binary or shared library *file*; the probe then
fires for **every process** that maps and runs that file.

```c
// demo.c — build with: gcc -O0 -g -o demo demo.c
#include <stdio.h>
#include <unistd.h>

int compute(int x) {        // -O0 keeps it a real, non-inlined symbol
    return x * x + 1;
}

int main(void) {
    for (int i = 0; ; i++) {
        printf("%d\n", compute(i));
        sleep(1);
    }
}
```

```bash
gcc -O0 -g -o demo demo.c
./demo &                       # leave it running in another terminal

# Confirm the symbol is present (a stripped binary has none → nothing to probe)
nm -C ./demo | grep ' compute'        # ...... T compute
```

### Trace arguments and return value with bpftrace

```bash
# Entry: arg0 is compute()'s first parameter
sudo bpftrace -e 'uprobe:./demo:compute { printf("compute(%d)\n", arg0); }'

# Return: retval is what compute() returned
sudo bpftrace -e 'uretprobe:./demo:compute { printf("  -> %d\n", retval); }'
```

```
compute(7)
  -> 50
```

bpftrace resolves the symbol to a file offset for you and handles **PIE/ASLR** — for a
position-independent executable the load address differs every run, but the probe is
defined by file offset, so it just works. With the raw interface you'd write the offset
yourself (`echo 'p:c /path/demo:0x1149' > uprobe_events`).

### Library uprobe — across the whole system

Probe a function in a shared library to observe every caller. Here, the distribution of
`malloc` request sizes in libc:

```bash
sudo bpftrace -e '
  uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc { @bytes = hist(arg0); }'
# (find the exact path with: ldd ./demo | grep libc)
```

### USDT — stable userspace probes

Raw uprobes break when a function is renamed, inlined, or its signature changes. **USDT**
(Userland Statically Defined Tracing) probes are markers the *author* compiles into the
binary (via `<sys/sdt.h>`), giving a stable name/argument contract — the userspace
analogue of kernel tracepoints. PostgreSQL, the JVM, libc, and Node.js ship them:

```bash
sudo bpftrace -e 'usdt:/usr/lib/.../libc.so.6:libc:memory_sbrk_more { printf("sbrk\n"); }'
sudo bpftrace -l 'usdt:./mybin:*'      # list a binary's USDT probes
```

## Filtering, context & stack traces

A probe that fires on *every* call is rarely what you want. bpftrace **predicates**
(the `/.../` between the probe and the action block) gate the handler, and a set of
built-ins give you the calling context for free:

```bash
# Only when a specific PID opens a large file
sudo bpftrace -e 'kprobe:vfs_read /pid == 1234 && arg2 > 4096/ { printf("big read\n"); }'

# Only the nginx workers
sudo bpftrace -e 'uprobe:./demo:compute /comm == "demo"/ { printf("%d\n", arg0); }'
```

```
context built-ins:  pid tid comm uid cpu nsecs elapsed     str(ptr)  ntop(addr)
```

The real power move is **capturing the call stack** at the probe — answering *"who
actually calls this function?"* without source diving. Aggregate by stack so identical
paths collapse into a count:

```bash
# Which kernel call paths reach tcp_sendmsg, ranked by frequency
sudo bpftrace -e 'kprobe:tcp_sendmsg { @[kstack] = count(); }'

# Userspace stacks into malloc for one process
sudo bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc /pid == 1234/ { @[ustack] = count(); }'
```

```
@[
    tcp_sendmsg+1
    sock_sendmsg+48
    __sys_sendto+238
    __x64_sys_sendto+37
]: 914
```

Combine a predicate with a stack to grab the stack **only on the interesting path** — e.g.
the error case (see the case study below). `kstack`/`ustack` accept a depth argument
(`kstack(5)`) to keep output short.

## perf probe

`perf probe` registers kprobe/uprobe events that integrate with `perf record`/`perf trace`,
and it can resolve source-level variable names when debug info is present.

```bash
# Kernel: add a probe that captures the filename variable by name
sudo perf probe --add 'do_sys_openat2 filename:string'
sudo perf record -e probe:do_sys_openat2 -a -- sleep 5
sudo perf script                                   # decode the captured events

# Userspace: -x points at the binary/library
sudo perf probe -x ./demo --add 'compute x'
sudo perf trace -e probe_demo:compute ./demo

# List and clean up
sudo perf probe --list
sudo perf probe --del '*'                          # remove all perf-created probes
```

## The kernel C API (register_kprobe)

Everything above is built on a small in-kernel API. You only reach for it directly when
writing kernel code (a module) rather than tracing — but it's worth seeing, because it's
the foundation bpftrace/eBPF, ftrace, and perf all sit on.

```c
#include <linux/kprobes.h>
#include <linux/module.h>

static struct kprobe kp = { .symbol_name = "do_sys_openat2" };

/* runs at function entry; pt_regs holds the args (see register table below) */
static int handler_pre(struct kprobe *p, struct pt_regs *regs) {
    pr_info("openat2 by %s, dfd=%ld\n", current->comm, (long)regs->di);
    return 0;
}

static int __init mod_init(void) {
    kp.pre_handler = handler_pre;       /* .post_handler / .fault_handler also exist */
    return register_kprobe(&kp);
}
static void __exit mod_exit(void) { unregister_kprobe(&kp); }

module_init(mod_init);
module_exit(mod_exit);
MODULE_LICENSE("GPL");
```

For return probes, `struct kretprobe` wraps a kprobe with a `.handler` (runs on return,
sees `regs_return_value(regs)`) and an optional `.entry_handler`. Its `.maxactive` field
caps how many calls can be **in flight** simultaneously — each needs a slot to stash the
real return address. If more threads are inside the function than `maxactive`, those
returns are **missed silently** (`nmissed` counts them); size it to your concurrency, or
leave 0 for a sensible default.

```c
static struct kretprobe krp = {
    .kp.symbol_name = "vfs_read",
    .handler        = ret_handler,
    .maxactive      = 64,            /* up to 64 concurrent vfs_read() calls tracked */
};
register_kretprobe(&krp);
```

In almost all debugging situations you should use bpftrace or an [eBPF](../linux/ebpf.md)
program instead — same probes, no module to compile and load, and the verifier keeps you
from crashing the kernel.

## Reading argument values

How you name function arguments depends on the front-end, but they all ultimately read
the same CPU registers (per the SysV x86-64 calling convention):

```
arg #     1     2     3     4     5     6        return value
x86-64   rdi   rsi   rdx   rcx   r8    r9        rax
arm64     x0    x1    x2    x3    x4    x5        x0

bpftrace:  arg0 arg1 arg2 ...            retval
eBPF C:    PT_REGS_PARM1(ctx) ...        PT_REGS_RC(ctx)
raw tracefs: +0(%di):u64  +0(%si):string  (REGISTER + offset + type)
```

Scalars and pointers are easy. Reading a **struct field** (e.g. a field inside a `struct
sock *`) needs the kernel's type layout: in bpftrace just cast (`((struct sock *)arg0)->...`),
and in raw eBPF use CO-RE/BTF so it stays portable across kernel versions — see
[eBPF → CO-RE](../linux/ebpf.md).

## Ready-made tools (you may not need to write anything)

Before writing a script, check whether a packaged tool already does it. The **BCC** and
**bpftrace** projects ship dozens of tools that are just curated kprobe/uprobe programs
(`apt install bpfcc-tools bpftrace`; bpftrace's versions live in
`/usr/share/bpftrace/tools/`):

```bash
funccount  'vfs_*'                 # count calls to every vfs_* function
funclatency -u vfs_read            # latency histogram (µs) of one function
funcslower  vfs_read 10000         # log calls to vfs_read slower than 10ms
stackcount  -P tcp_sendmsg         # tally the stacks reaching a function (per process)
trace 'do_sys_openat2 "%s", arg2'  # printf-style ad-hoc tracing, one line
argdist -C 'p::vfs_read():u32:arg2'# distribution of the size argument
```

Plus the snoop family — `opensnoop`, `execsnoop`, `biosnoop`, `tcpconnect`, `bashreadline`
— each a small uprobe/kprobe tool you can also read as a worked example. For production,
prefer the compiled **libbpf-tools** versions over the Python BCC ones (no runtime LLVM
dependency — see the [eBPF tooling note](../linux/ebpf.md)).

## Multi-attach: wildcards, fprobe, fentry/fexit

Naming one function is the common case, but you can attach broadly — at a cost:

```
kprobe:tcp_*        bpftrace expands the glob and registers ONE kprobe per match
                    → flexible, but N traps = N× overhead; slow to attach thousands
fprobe (multi-kprobe) one attach point covering thousands of functions, ftrace-backed
                    → far cheaper mass attach; used by `funccount`-style mass tracing
fentry / fexit      BPF trampoline at the function's ftrace site (BTF kernels, 5.5+)
                    → lowest overhead, TYPED args/retval (no pt_regs decode) — preferred
                      where available; the modern replacement for k(ret)probes
```

Rule of thumb: a handful of functions → kprobe is fine; *"every `tcp_*`"* → reach for a
multi-attach mechanism; a single hot function on a BTF kernel → use **fentry/fexit**
(documented in [eBPF](../linux/ebpf.md)).

## Worked case study: an intermittent error

**Symptom:** an application occasionally fails with `EIO`, but the logs don't say which
syscall or kernel path produced it. `errno 5 == EIO`, so kernel functions return `-5`.

```bash
# 1. Which read/write path is returning -EIO? Filter a kretprobe on the error value.
sudo bpftrace -e 'kretprobe:vfs_read,kretprobe:vfs_write /retval == -5/ {
    printf("%s -EIO from %s (pid %d)\n", probe, comm, pid);
}'
#   kretprobe:vfs_read -EIO from myapp (pid 4821)

# 2. WHO called it? Capture the kernel stack, but only on the failing path.
sudo bpftrace -e 'kretprobe:vfs_read /retval == -5/ { @[kstack] = count(); }'
#   @[
#       vfs_read+...
#       ksys_read+...
#       __x64_sys_read+...
#   ]: 3

# 3. Confirm WHAT was being read — grab the entry args for the same pid.
sudo bpftrace -e 'kprobe:vfs_read /pid == 4821/ {
    printf("fd-backed file, count=%d\n", arg2);
}'
```

From three short probes you've gone from "an EIO somewhere" to the exact function, its
caller chain, and the offending call's arguments — no kernel rebuild, no restart. The same
pattern (predicate on `retval` → stack → entry args) localises most "it fails sometimes"
bugs.

## Pitfalls

```
- Inlined / optimised-away functions have no symbol to probe — build the target with -g,
  and remember -O2 may inline static helpers or drop a frame pointer entirely.
- -O2 also reshuffles/elides arguments: arg0 may not be the source's first parameter.
  Verify against the disassembly (objdump -d) or prefer a -O0 build for the demo.
- Some kernel functions are un-probeable: anything marked __kprobes/notrace, or listed in
  /sys/kernel/debug/kprobes/blacklist (probing them would recurse into the probe machinery).
- Probes have real overhead on hot paths — a kprobe on a function called millions of
  times/sec will hurt. Aggregate in-kernel (count()/hist()) instead of printing per event.
- A uprobe fires for EVERY process that maps the file; filter by pid if you mean one.
- Stripped binaries / libraries have no symbols — supply the raw file offset, or install
  the -dbg/-debuginfo package (same symbol problem as perf, see perf_profiling.md).
- Needs privilege: root, or CAP_BPF+CAP_PERFMON; kernel.perf_event_paranoid and
  kernel.kptr_restrict can block unprivileged use.
- Stale probes survive your script: a crashed run can leave entries in kprobe_events /
  uprobe_events. Clear them (echo > .../kprobe_events) so they stop firing.
- kretprobe maxactive is finite: if more threads are inside the function than there are
  slots, those returns are missed SILENTLY (check nmissed) — raise maxactive for hot fns.
- Wildcard probes (kprobe:tcp_*) can attach to thousands of functions: slow to register
  and a real overhead spike. Scope the glob, or use a multi-attach mechanism (fprobe).
- USDT probes guarded by a semaphore won't fire unless something has enabled the probe;
  attach via the proper USDT path (bpftrace/perf handle the semaphore refcount for you).
- Ambiguous symbols: a name with multiple addresses (static funcs, or inlined copies)
  needs disambiguation by address/file — perf probe and bpftrace will warn.
```

## Where this connects

- [Linux Kernel Debugging](linux_kernel.md) — ftrace and static tracepoints: the stable,
  fixed-location counterpart to dynamic probes
- [perf & Flame Graphs](perf_profiling.md) — sampling vs probing; `perf probe` lives here
- [eBPF](../linux/ebpf.md) — the programmable engine that consumes kprobes/uprobes;
  fentry/fexit are the modern, lower-overhead replacement for k(ret)probes on BTF kernels
- [Binary Analysis Tools](tools.md) — strace/ltrace trace at the syscall/library boundary
  without naming internal functions
- [GDB](gdb.md) — breakpoints: the same trap mechanism, but interactive and stop-the-world
