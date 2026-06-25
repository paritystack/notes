# ftrace & tracefs

## Overview

**ftrace** is the kernel's built-in tracer, driven entirely through a virtual filesystem
(**tracefs**) — no compiler, no extra tooling, present on essentially every kernel. You turn
tracing on by writing to files under `/sys/kernel/tracing` and read the result back from a
`trace` file. It is the *stable, in-tree* foundation that the fancier tools sit on: static
[tracepoints](linux_kernel.md) are exposed through it, dynamic [kprobes/uprobes](kprobes_uprobes.md)
register through its `kprobe_events` interface, and [eBPF](../linux/ebpf.md) attaches to the same
fentry/tracepoint sites. Where [perf](perf_profiling.md) *samples* the stack periodically, ftrace
*traces* — it records every event you ask for, which is why its **function** and
**function_graph** tracers and its **latency tracers** are the go-to for "what exactly ran, in
order, and for how long."

```
  perf      : statistical SAMPLES  → "where is time spent" (low overhead, blind to rare)
  ftrace    : exhaustive EVENTS    → "what ran, in what order, latencies" (every call)
  kprobes   : dynamic attach points → consumed via ftrace's kprobe_events / eBPF
  tracepoints: static kernel events → toggled under tracefs events/

  control surface = tracefs files (echo into them, cat the result)
```

## tracefs: the control surface

Everything lives under `/sys/kernel/tracing` (modern) or the older
`/sys/kernel/debug/tracing` (when debugfs is mounted). Mount it if absent:

```bash
mount -t tracefs nodev /sys/kernel/tracing
cd /sys/kernel/tracing
ls
# available_tracers  current_tracer  trace  trace_pipe  tracing_on
# set_ftrace_filter  set_ftrace_notrace  set_graph_function  events/  per_cpu/
# trace_marker  buffer_size_kb  options/  available_events  ...
```

Core control files:

| File | Purpose |
|------|---------|
| `current_tracer` | which tracer is active (`nop`, `function`, `function_graph`, …) |
| `available_tracers` | tracers this kernel was built with |
| `tracing_on` | master switch (`1`/`0`) — pause/resume without losing the buffer |
| `trace` | snapshot of the ring buffer (re-reading restarts at the top) |
| `trace_pipe` | consuming, blocking read — drains events as a live stream |
| `set_ftrace_filter` / `set_ftrace_notrace` | limit (or exclude) which functions are traced |
| `buffer_size_kb` | per-CPU ring-buffer size |
| `trace_marker` | userspace writes here to inject markers into the timeline |

Always start clean: `echo nop > current_tracer; echo > trace` resets state left by a previous run.

## The function tracers

```bash
# Trace one function (and only it) — much cheaper than tracing everything
echo vfs_read > set_ftrace_filter      # supports globs: 'vfs_*', and negation via set_ftrace_notrace
echo function > current_tracer
echo 1 > tracing_on ; sleep 1 ; echo 0 > tracing_on
cat trace
```

**`function`** records each entry to traced functions with a timestamp and the CPU/PID.
**`function_graph`** is usually more useful — it traces entry *and* exit, drawing a call tree with
per-function **duration**, so you can see who called what and where the time went:

```bash
echo function_graph > current_tracer
echo vfs_read > set_graph_function     # root the graph at this function
cat trace
#  CPU  DURATION        FUNCTION
#  2)               |  vfs_read() {
#  2)   0.812 us     |    rw_verify_area();
#  2)               |    ext4_file_read_iter() {
#  2) + 18.339 us    |      ... }
#  2) + 21.105 us    |  }                 (+ marks > 10us, ! marks > 100us)
```

`max_graph_depth` bounds nesting; `funcgraph-proc`/`funcgraph-tail` options tune the output.

## Tracepoints (events)

Static tracepoints are toggled under `events/`. They're stable, named, and carry structured
fields — ideal for subsystem-level tracing without naming internal functions:

```bash
# Trace all scheduler switch events
echo 1 > events/sched/sched_switch/enable
echo 1 > tracing_on ; cat trace_pipe        # live stream

# Filter on a field, then disable
echo 'prev_comm == "nginx"' > events/sched/sched_switch/filter
echo 0 > events/sched/sched_switch/enable

cat available_events | head               # everything available
cat events/sched/sched_switch/format      # the fields you can filter on
```

Tracepoints also feed [perf](perf_profiling.md) and [eBPF](../linux/ebpf.md) — the same event,
three front-ends.

## Latency tracers

A distinguishing ftrace feature: tracers that measure how long the kernel disabled
interrupts/preemption or delayed a wakeup — the root-cause tools for latency spikes (and the
companion to [Interrupts & Deferred Work](../linux/interrupts.md)):

| Tracer | Measures |
|--------|----------|
| `irqsoff` | longest stretch with IRQs disabled |
| `preemptoff` | longest stretch with preemption disabled |
| `preemptirqsoff` | both combined |
| `wakeup` / `wakeup_rt` | scheduler wakeup → run latency (RT tasks) |

```bash
echo irqsoff > current_tracer
echo 0 > tracing_max_latency        # reset the high-water mark
# ... run workload ...
cat tracing_max_latency             # worst latency seen (us)
cat trace                           # the trace that produced it
```

There's also a **stack tracer** (`echo 1 > /proc/sys/kernel/stack_tracer_enabled`, read
`stack_max_size`/`stack_trace`) to catch deep kernel-stack usage, and a **function profiler**
(`function_profile_enabled`) for per-function call counts and total time.

## Front-ends: trace-cmd & kernelshark

Driving tracefs by hand is fine for one-offs; **trace-cmd** wraps it for real work, and
**kernelshark** gives a GUI timeline:

```bash
trace-cmd record -p function_graph -g vfs_read -- ./workload   # capture to trace.dat
trace-cmd report                                               # decode it
trace-cmd record -e sched_switch -e 'sched:*' sleep 5          # record tracepoints
kernelshark trace.dat                                          # visual timeline
```

`trace-cmd` handles buffer management, symbol resolution, and multi-CPU merging that are tedious
via raw files.

## When to use ftrace vs the neighbours

| Question | Reach for |
|----------|-----------|
| Exact call sequence + per-call duration | ftrace `function_graph` |
| "What disabled IRQs for 2ms?" | ftrace `irqsoff`/`preemptoff` |
| Subsystem events with fields, low effort | tracepoints under `events/` |
| Statistical "where is CPU going?" | [perf](perf_profiling.md) (sampling) |
| Probe an arbitrary function / read its args | [kprobes/uprobes](kprobes_uprobes.md) + [bpftrace/eBPF](../linux/ebpf.md) |
| Aggregate/histogram in-kernel, programmable | [eBPF](../linux/ebpf.md) |

Rule of thumb: ftrace answers *"what ran and for how long"* with zero install; reach for
bpftrace/eBPF when you need to read arguments, filter richly, or aggregate in-kernel.

## Where this connects

- [Linux Kernel Debugging](linux_kernel.md) — ftrace is one tool in the broader kernel-debug
  kit (KGDB, kdump, SystemTap); this page is its deep dive.
- [Kprobes & Uprobes](kprobes_uprobes.md) — dynamic probes register through tracefs
  (`kprobe_events`) and show up as ftrace events; the dynamic counterpart to static tracepoints.
- [perf & Flame Graphs](perf_profiling.md) — sampling vs exhaustive tracing; perf can also read
  the same tracepoints.
- [eBPF](../linux/ebpf.md) — the programmable engine on the same attach points (tracepoints,
  fentry/fexit, kprobes); use it when ftrace's fixed output isn't enough.
- [Interrupts & Deferred Work](../linux/interrupts.md) — the latency tracers (`irqsoff`/
  `preemptoff`) directly measure the IRQ/preemption-disabled windows discussed there.

## Pitfalls

- **Leaving a tracer running.** `function` tracing every function is a real, system-wide overhead;
  always scope with `set_ftrace_filter` and reset to `nop` when done.
- **`cat trace` vs `trace_pipe`.** `trace` is a non-destructive snapshot (re-reads start over);
  `trace_pipe` consumes as a live stream. Mixing them up loses or duplicates events.
- **Forgetting to clear state.** A previous run's `current_tracer`, filters, or enabled events
  persist. Start with `echo nop > current_tracer; echo > trace; echo > set_ftrace_filter`.
- **Ring buffer overflow.** The per-CPU buffer wraps; a busy trace drops oldest events. Raise
  `buffer_size_kb`, narrow the filter, or use `trace_pipe`/`trace-cmd record` to stream out.
- **`/sys/kernel/debug/tracing` vs `/sys/kernel/tracing`.** Modern kernels mount tracefs
  standalone; scripts hard-coding the debugfs path break when debugfs isn't mounted.
- **Privilege.** tracefs is root-only (or needs CAP_SYS_ADMIN); `kernel.perf_event_paranoid` and
  related sysctls can further restrict access.
- **Stale dynamic events.** Probes added via `kprobe_events` survive a crashed script; clear them
  (`echo > kprobe_events`) so they stop firing — same gotcha as [kprobes](kprobes_uprobes.md).
- **Symbols missing/inlined.** Heavily inlined or stripped functions may not be traceable by name;
  check `available_filter_functions`.
