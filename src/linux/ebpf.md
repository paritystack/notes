# eBPF (Extended Berkeley Packet Filter)

## Overview

eBPF is a programmable in-kernel virtual machine for packet processing, tracing, and security. It attaches at XDP/TC hooks in [networking](networking.md) (bypassing [netfilter](netfilter.md) for maximum performance), and at kprobes/tracepoints for tracing. Programs and maps are loaded through the `bpf()` syscall; [netlink](netlink.md) (rtnetlink) is used to *attach* XDP/TC programs to interfaces. eBPF maps share data between kernel programs and userspace as an alternative to [sysfs](sysfs.md). [Kernel patterns](kernel_patterns.md) like RCU locking govern safe map access.

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Program Types](#program-types)
4. [eBPF Maps](#ebpf-maps)
5. [Development Tools](#development-tools)
6. [Writing eBPF Programs](#writing-ebpf-programs)
7. [Modern eBPF Features](#modern-ebpf-features)
8. [Common Use Cases](#common-use-cases)
9. [Examples](#examples)
10. [Security and Safety](#security-and-safety)
11. [Debugging](#debugging)
12. [Resources](#resources)

---

## Introduction

### What is eBPF?

eBPF (Extended Berkeley Packet Filter) is a revolutionary Linux kernel technology that allows running sandboxed programs in kernel space without changing kernel source code or loading kernel modules. It enables dynamic extension of kernel capabilities for networking, observability, security, and performance analysis.

### History

- **1992**: Original BPF (Berkeley Packet Filter) created for packet filtering in BSD
- **2014**: eBPF introduced in Linux kernel 3.18, extending BPF beyond networking
- **2016-Present**: Rapid evolution with new program types, maps, and helper functions

### Key Features

- **Safe**: Verifier ensures programs are safe to run in kernel space
- **Efficient**: JIT compilation for native performance
- **Dynamic**: Load/unload programs without rebooting
- **Programmable**: Write custom kernel extensions in C/Rust
- **Event-driven**: Attach to kernel/user events without overhead when not triggered

### Use Cases

- Network packet filtering and manipulation
- Performance monitoring and profiling
- Security enforcement and runtime protection
- Tracing and observability
- Load balancing and service mesh
- Container networking

---

## Architecture

### eBPF Virtual Machine

eBPF programs run in a virtual machine within the kernel with:
- **11 64-bit registers** (R0-R10)
- **512-byte stack**
- **RISC-like instruction set** (similar to x86-64)
- **Bounded loops** (since kernel 5.3)

```
R0:  Return value from functions/exit value
R1-R5: Function arguments
R6-R9: Callee-saved registers
R10: Read-only frame pointer
```

### Core Components

#### 1. Verifier
- Static analysis of eBPF bytecode before loading
- Ensures memory safety (no out-of-bounds access)
- Validates control flow (no infinite loops, reachable code)
- Checks register states and types
- Limits program complexity

#### 2. JIT Compiler
- Compiles eBPF bytecode to native machine code
- Available for x86-64, ARM64, RISC-V, etc.
- Provides near-native performance
- Can be disabled (interpreter fallback)

```bash
# Enable JIT compiler
echo 1 > /proc/sys/net/core/bpf_jit_enable

# Enable JIT debug (dump compiled code)
echo 2 > /proc/sys/net/core/bpf_jit_enable
```

#### 3. Helper Functions
- Kernel functions callable from eBPF programs
- Type-safe interfaces to kernel functionality
- Examples: map operations, packet manipulation, time functions

#### 4. Maps
- Data structures for sharing data between eBPF programs and user space
- Persistent storage across program invocations
- Various types: hash, array, ring buffer, etc.

### Attachment Points (Hooks)

eBPF programs attach to kernel events:

- **Network**: XDP, TC, socket operations, cgroups
- **Tracing**: kprobes, uprobes, tracepoints, USDT
- **Security**: LSM hooks, seccomp
- **Cgroups**: Device access, socket operations, sysctl

---

## Program Types

### XDP (eXpress Data Path)

Processes packets at the earliest point in the network stack (driver level).

**Use Cases**: DDoS mitigation, load balancing, packet filtering

**Return Codes**:
- `XDP_DROP`: Drop packet
- `XDP_PASS`: Pass to network stack
- `XDP_TX`: Bounce packet back out same interface
- `XDP_REDIRECT`: Redirect to another interface
- `XDP_ABORTED`: Error, drop packet

**Example Hook**:
```c
SEC("xdp")
int xdp_prog(struct xdp_md *ctx) {
    // Access packet data
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    // Process packet
    return XDP_PASS;
}
```

**XDP Modes**:
- **Native (driver) XDP**: runs in the driver's RX path, fastest (requires driver support)
- **Generic XDP** (`xdpgeneric`): runs after `skb` allocation, works on any NIC, slower
- **Offloaded XDP**: program runs on the SmartNIC itself (e.g. Netronome)

**Redirection maps & zero-copy**:
- `BPF_MAP_TYPE_DEVMAP` / `DEVMAP_HASH`: redirect to another interface for `XDP_REDIRECT`
- `BPF_MAP_TYPE_CPUMAP`: redirect packets to a remote CPU for load distribution
- `BPF_MAP_TYPE_XSKMAP`: redirect into an **AF_XDP (XSK)** socket for zero-copy
  userspace packet processing (DPDK-style fast path without leaving the kernel's NIC ring)

### TC (Traffic Control)

Attaches to network queueing discipline (ingress/egress).

**Use Cases**: QoS, traffic shaping, packet modification

**Attachment**:
```bash
tc qdisc add dev eth0 clsact
tc filter add dev eth0 ingress bpf da obj prog.o sec classifier
```

### Tracepoints

Static instrumentation points in the kernel.

**Advantages**: Stable ABI, defined arguments
**Locations**: Scheduling, system calls, network events

```c
SEC("tracepoint/syscalls/sys_enter_execve")
int trace_execve(struct trace_event_raw_sys_enter *ctx) {
    // Trace execve system call
    return 0;
}
```

### Kprobes/Kretprobes

Dynamic instrumentation of any kernel function.

**Kprobe**: Execute at function entry
**Kretprobe**: Execute at function return

```c
SEC("kprobe/tcp_connect")
int trace_tcp_connect(struct pt_regs *ctx) {
    // Hook tcp_connect function
    return 0;
}

SEC("kretprobe/tcp_connect")
int trace_tcp_connect_ret(struct pt_regs *ctx) {
    // Get return value
    int ret = PT_REGS_RC(ctx);
    return 0;
}
```

### Uprobes/Uretprobes

Dynamic instrumentation of user-space functions.

**Use Cases**: Application profiling, library tracing

```c
SEC("uprobe/usr/lib/libc.so.6:malloc")
int trace_malloc(struct pt_regs *ctx) {
    size_t size = PT_REGS_PARM1(ctx);
    return 0;
}
```

### Socket Filters

Filter and process socket data.

**Types**:
- `BPF_PROG_TYPE_SOCKET_FILTER`: Classic socket filtering
- `BPF_PROG_TYPE_SOCK_OPS`: Socket operations monitoring
- `BPF_PROG_TYPE_SK_SKB`: Socket buffer redirection
- `BPF_PROG_TYPE_SK_MSG`: Socket message filtering

### LSM (Linux Security Module)

Implement security policies using LSM hooks.

**Requirements**: Kernel 5.7+, BPF LSM enabled

```c
SEC("lsm/file_open")
int BPF_PROG(file_open, struct file *file) {
    // Implement access control
    return 0; // Allow
}
```

### Fentry/Fexit (BPF Trampolines)

BTF-based tracing of kernel functions via BPF trampolines (kernel 5.5+). The modern,
lower-overhead replacement for kprobes/kretprobes on BTF-enabled kernels: arguments and
return values are directly typed instead of decoded from `pt_regs`.

```c
// Entry: typed access to function arguments
SEC("fentry/tcp_connect")
int BPF_PROG(tcp_connect_entry, struct sock *sk) {
    return 0;
}

// Exit: typed args plus the return value
SEC("fexit/tcp_connect")
int BPF_PROG(tcp_connect_exit, struct sock *sk, int ret) {
    return 0;
}
```

### Struct_ops

Programs that implement a kernel-defined struct of callbacks (kernel 5.6+). The BPF program
*becomes* an implementation the kernel calls into.

**Use Cases**: pluggable TCP congestion control algorithms, HID-BPF, and `sched_ext`.

```c
SEC("struct_ops/cong_avoid")
void BPF_PROG(my_cong_avoid, struct sock *sk, __u32 ack, __u32 acked) {
    // Custom congestion-control logic
}

SEC(".struct_ops")
struct tcp_congestion_ops my_ca = {
    .cong_avoid = (void *)my_cong_avoid,
    .name       = "bpf_mycc",
};
```

### sched_ext (SCX)

Extensible scheduler class (kernel 6.12+) built on `struct_ops`, allowing CPU schedulers to
be implemented entirely in BPF and loaded/swapped at runtime — with a safety watchdog that
falls back to the default scheduler if the BPF scheduler misbehaves. Used for rapid
scheduler experimentation (e.g. the `scx_*` schedulers like `scx_rusty`, `scx_lavd`).

### Other Program Types

- **Cgroup programs**: Control resource access per cgroup
- **Perf event**: Attach to performance monitoring events
- **Raw tracepoints**: Low-overhead tracing
- **BTF-enabled programs**: Type information for portability

---

## eBPF Maps

Maps are key-value data structures for storing state and communicating between eBPF programs and user space.

### Map Types

#### BPF_MAP_TYPE_HASH
Hash table for arbitrary key-value pairs.

```c
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, u32);
    __type(value, u64);
} my_hash_map SEC(".maps");
```

#### BPF_MAP_TYPE_ARRAY
Fixed-size array indexed by integer.

```c
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 256);
    __type(key, u32);
    __type(value, u64);
} my_array SEC(".maps");
```

#### BPF_MAP_TYPE_PERCPU_HASH / PERCPU_ARRAY
Per-CPU variants for better performance (no locking).

```c
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 256);
    __type(key, u32);
    __type(value, u64);
} percpu_stats SEC(".maps");
```

#### BPF_MAP_TYPE_RINGBUF
Ring buffer for efficient kernel-to-user data streaming (kernel 5.8+).

```c
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Reserve and submit
struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
if (e) {
    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_ringbuf_submit(e, 0);
}
```

#### BPF_MAP_TYPE_PERF_EVENT_ARRAY
Per-CPU event buffers (older than ringbuf).

```c
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(u32));
    __uint(value_size, sizeof(u32));
} events SEC(".maps");
```

#### BPF_MAP_TYPE_LRU_HASH
Hash table with Least Recently Used eviction.

```c
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 10000);
    __type(key, u32);
    __type(value, u64);
} lru_cache SEC(".maps");
```

#### BPF_MAP_TYPE_STACK_TRACE
Store stack traces.

```c
struct {
    __uint(type, BPF_MAP_TYPE_STACK_TRACE);
    __uint(max_entries, 1000);
    __type(key, u32);
    __type(value, u64[127]);
} stack_traces SEC(".maps");
```

#### BPF_MAP_TYPE_PROG_ARRAY
Array of eBPF programs for tail calls.

```c
struct {
    __uint(type, BPF_MAP_TYPE_PROG_ARRAY);
    __uint(max_entries, 10);
    __type(key, u32);
    __type(value, u32);
} prog_array SEC(".maps");

// Tail call
bpf_tail_call(ctx, &prog_array, index);
```

#### BPF_MAP_TYPE_BLOOM_FILTER
Probabilistic set-membership filter (kernel 5.16+). No keys — only values are pushed; lookups
report "possibly present" or "definitely absent". Useful for cheaply skipping expensive work.

```c
struct {
    __uint(type, BPF_MAP_TYPE_BLOOM_FILTER);
    __uint(max_entries, 10000);
    __type(value, u32);
    __uint(map_extra, 5); // number of hash functions
} seen SEC(".maps");

bpf_map_push_elem(&seen, &val, BPF_ANY);
if (bpf_map_peek_elem(&seen, &val) == 0) { /* possibly seen */ }
```

#### Local Storage Maps
Storage attached to the lifetime of a kernel object, automatically freed when the object dies.
Heavily used in modern tracers/security tools to associate state without manual cleanup.

- `BPF_MAP_TYPE_TASK_STORAGE`: per-task (process/thread) storage
- `BPF_MAP_TYPE_SK_STORAGE`: per-socket storage
- `BPF_MAP_TYPE_INODE_STORAGE`: per-inode storage
- `BPF_MAP_TYPE_CGRP_STORAGE`: per-cgroup storage (kernel 6.2+)

```c
struct {
    __uint(type, BPF_MAP_TYPE_TASK_STORAGE);
    __uint(map_flags, BPF_F_NO_PREALLOC);
    __type(key, int);
    __type(value, u64);
} task_state SEC(".maps");

u64 *st = bpf_task_storage_get(&task_state, task, NULL,
                               BPF_LOCAL_STORAGE_GET_F_CREATE);
```

### Map Operations

```c
// Lookup
value = bpf_map_lookup_elem(&my_map, &key);

// Update
bpf_map_update_elem(&my_map, &key, &value, BPF_ANY);

// Delete
bpf_map_delete_elem(&my_map, &key);
```

**Update Flags**:
- `BPF_ANY`: Create or update
- `BPF_NOEXIST`: Create only if doesn't exist
- `BPF_EXIST`: Update only if exists

---

## Development Tools

### bpftrace

High-level, awk-like tracing language that compiles one-liners and short scripts to eBPF —
the fastest way to attach to a probe and aggregate in-kernel. See
[bpftrace](../debugging/bpftrace.md) for the language and cookbook.

**Pros**: Concise, safe, ideal for interactive "what's happening now" investigation
**Cons**: Not for shipping complex tools — graduate to libbpf/BCC when you outgrow one-liners

```bash
bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }'
```

### BCC (BPF Compiler Collection)

Python/Lua framework for writing eBPF programs.

**Pros**: High-level, rapid development, many examples
**Cons**: Runtime compilation, LLVM dependency on target

```python
from bcc import BPF

prog = """
int hello(void *ctx) {
    bpf_trace_printk("Hello, World!\\n");
    return 0;
}
"""

b = BPF(text=prog)
b.attach_kprobe(event="sys_clone", fn_name="hello")
```

> **Note**: The BCC project has largely shifted its tools to **libbpf-tools** (CO-RE,
> compiled ahead of time with no runtime LLVM/kernel-headers dependency). Prefer the
> `libbpf-tools/` versions for production; the Python BCC interface remains useful for
> prototyping.

### libbpf

C library for loading and managing eBPF programs.

**Pros**: No runtime dependencies, CO-RE support, production-ready
**Cons**: Lower-level, more boilerplate

```c
struct bpf_object *obj;
struct bpf_program *prog;
struct bpf_link *link;

obj = bpf_object__open_file("prog.o", NULL);
bpf_object__load(obj);
prog = bpf_object__find_program_by_name(obj, "xdp_prog");
link = bpf_program__attach(prog);
```

### bpftool

Command-line tool for inspecting and managing eBPF programs/maps.

```bash
# List programs
bpftool prog list

# Show program details
bpftool prog show id 123

# Dump program bytecode
bpftool prog dump xlated id 123

# List maps
bpftool map list

# Dump map contents
bpftool map dump id 456

# Load program
bpftool prog load prog.o /sys/fs/bpf/myprog

# Pin map
bpftool map pin id 456 /sys/fs/bpf/mymap
```

### eBPF for Go

```go
import "github.com/cilium/ebpf"

spec, err := ebpf.LoadCollectionSpec("prog.o")
coll, err := ebpf.NewCollection(spec)
defer coll.Close()

prog := coll.Programs["xdp_prog"]
link, err := link.AttachXDP(link.XDPOptions{
    Program:   prog,
    Interface: iface.Index,
})
defer link.Close()
```

### eBPF for Rust

Two main approaches:

- **Aya**: a pure-Rust framework where both the kernel-side BPF program *and* the user-space
  loader are written in Rust, with **no libbpf and no LLVM/bpf-linker runtime dependency** on
  the target. CO-RE and BTF supported. Good fit for distributing self-contained binaries.
- **libbpf-rs**: idiomatic Rust bindings over libbpf; BPF code is still written in C and
  compiled with Clang, with a `libbpf-cargo` build step generating skeletons.

```rust
// Aya user-space loader (excerpt)
use aya::{Ebpf, programs::Xdp};

let mut bpf = Ebpf::load_file("prog.o")?;
let prog: &mut Xdp = bpf.program_mut("xdp_prog").unwrap().try_into()?;
prog.load()?;
prog.attach("eth0", aya::programs::XdpFlags::default())?;
```

### Other Tools

- **Cilium**: Container networking with eBPF
- **Hubble**: Network observability for Cilium (flows, service map)
- **Tetragon**: eBPF runtime security observability & enforcement (Cilium)
- **Katran**: Layer 4 load balancer (Facebook)
- **Falco**: Runtime security monitoring
- **Tracee**: Runtime security & forensics (Aqua Security)
- **Pixie**: Observability platform (auto-instrumentation)
- **Parca**: Continuous (always-on) profiling using eBPF
- **bpftrace**: High-level tracing language

---

## Writing eBPF Programs

### Development Workflow

1. **Write C code** with eBPF program
2. **Compile to eBPF bytecode** using Clang/LLVM
3. **Load into kernel** using libbpf/BCC
4. **Attach to hook point**
5. **Communicate via maps**
6. **Unload/detach** when done

### Basic C Program Structure

```c
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

// Define map
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, u32);
    __type(value, u64);
} stats SEC(".maps");

// eBPF program
SEC("xdp")
int xdp_main(struct xdp_md *ctx) {
    u32 key = 0;
    u64 *count;

    count = bpf_map_lookup_elem(&stats, &key);
    if (count) {
        __sync_fetch_and_add(count, 1);
    }

    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
```

### Compilation

```bash
# Compile to eBPF bytecode
clang -O2 -g -target bpf -c prog.c -o prog.o

# With BTF (Type Information)
clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
    -I/usr/include/bpf -c prog.c -o prog.o
```

### CO-RE (Compile Once - Run Everywhere)

**Problem**: Kernel data structures change across versions
**Solution**: BTF (BPF Type Format) + CO-RE relocations

```c
#include <vmlinux.h>
#include <bpf/bpf_core_read.h>

SEC("kprobe/tcp_connect")
int trace_connect(struct pt_regs *ctx) {
    struct sock *sk = (struct sock *)PT_REGS_PARM1(ctx);
    u16 family;

    // CO-RE read - portable across kernel versions
    BPF_CORE_READ_INTO(&family, sk, __sk_common.skc_family);

    return 0;
}
```

**Generate vmlinux.h** (kernel type definitions):
```bash
bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
```

**Running CO-RE on kernels without BTF**: kernels older than 5.2, or those built without
`CONFIG_DEBUG_INFO_BTF`, lack `/sys/kernel/btf/vmlinux`. **BTFHub** provides pre-generated
BTF for thousands of distro kernels; libbpf can load an external BTF blob (and tools can ship
a `min_core_btf`-trimmed BTF to keep binaries small) so a single CO-RE binary runs everywhere.

### User-Space Loader (libbpf)

```c
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

int main() {
    struct bpf_object *obj;
    struct bpf_program *prog;
    int prog_fd, map_fd;

    // Open and load
    obj = bpf_object__open_file("prog.o", NULL);
    bpf_object__load(obj);

    // Get program
    prog = bpf_object__find_program_by_name(obj, "xdp_main");
    prog_fd = bpf_program__fd(prog);

    // Get map
    map_fd = bpf_object__find_map_fd_by_name(obj, "stats");

    // Attach (XDP example)
    int ifindex = if_nametoindex("eth0");
    bpf_xdp_attach(ifindex, prog_fd, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL);

    // Read from map
    u32 key = 0;
    u64 value;
    bpf_map_lookup_elem(map_fd, &key, &value);
    printf("Count: %llu\n", value);

    // Cleanup
    bpf_xdp_detach(ifindex, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL);
    bpf_object__close(obj);

    return 0;
}
```

**Compile user-space loader**:
```bash
gcc -o loader loader.c -lbpf -lelf -lz
```

---

## Common Use Cases

### 1. Network Packet Filtering

**XDP-based firewall**:
- Drop malicious packets at driver level
- Block by IP, port, protocol
- DDoS mitigation

### 2. Load Balancing

**Layer 4 load balancing**:
- Distribute connections across backends
- Connection tracking
- Health checks

**Examples**: Katran (Facebook), Cilium

### 3. Observability and Tracing

**System call tracing**:
- Monitor file access
- Track network connections
- Profile CPU usage

**Tools**: BCC tools (execsnoop, opensnoop, tcpconnect)

### 4. Security Monitoring

**Runtime security**:
- Detect malicious behavior
- File integrity monitoring
- Process ancestry tracking

**Tools**: Falco, Tracee

### 5. Performance Analysis

**Profiling**:
- CPU flame graphs
- I/O latency
- Memory allocation tracking

### 6. Container Networking

**CNI plugins**:
- Pod networking
- Network policies
- Service mesh data plane

**Examples**: Cilium, Calico eBPF

### 7. Network Monitoring

**Metrics collection**:
- Packet counters
- Bandwidth monitoring
- Protocol analysis

---

## Examples

### Example 1: Packet Counter (XDP)

**prog.c**:
```c
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <bpf/bpf_helpers.h>

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 256);
    __type(key, u32);
    __type(value, u64);
} proto_count SEC(".maps");

SEC("xdp")
int count_packets(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    u32 key = ip->protocol;
    u64 *count = bpf_map_lookup_elem(&proto_count, &key);
    if (count)
        __sync_fetch_and_add(count, 1);

    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
```

**Compile and load**:
```bash
clang -O2 -g -target bpf -c prog.c -o prog.o
ip link set dev eth0 xdp obj prog.o sec xdp
```

**Read stats**:
```bash
bpftool map dump name proto_count
```

### Example 2: Process Execution Tracer

**execsnoop.c**:
```c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>

struct event {
    u32 pid;
    char comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

SEC("tracepoint/syscalls/sys_enter_execve")
int trace_execve(struct trace_event_raw_sys_enter *ctx) {
    struct event *e;

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

**User-space consumer**:
```c
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

struct event {
    u32 pid;
    char comm[16];
};

int handle_event(void *ctx, void *data, size_t len) {
    struct event *e = data;
    printf("PID: %d, COMM: %s\n", e->pid, e->comm);
    return 0;
}

int main() {
    struct bpf_object *obj;
    struct ring_buffer *rb;
    int map_fd;

    obj = bpf_object__open_file("execsnoop.o", NULL);
    bpf_object__load(obj);

    map_fd = bpf_object__find_map_fd_by_name(obj, "events");
    rb = ring_buffer__new(map_fd, handle_event, NULL, NULL);

    while (1) {
        ring_buffer__poll(rb, 100);
    }

    return 0;
}
```

### Example 3: TCP Connection Tracking

**tcpconnect.c**:
```c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>

struct conn_event {
    u32 pid;
    u32 saddr;
    u32 daddr;
    u16 sport;
    u16 dport;
};

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

SEC("kprobe/tcp_connect")
int trace_connect(struct pt_regs *ctx) {
    struct sock *sk = (struct sock *)PT_REGS_PARM1(ctx);
    struct conn_event *e;
    u16 family;

    BPF_CORE_READ_INTO(&family, sk, __sk_common.skc_family);
    if (family != AF_INET)
        return 0;

    e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        return 0;

    e->pid = bpf_get_current_pid_tgid() >> 32;
    BPF_CORE_READ_INTO(&e->saddr, sk, __sk_common.skc_rcv_saddr);
    BPF_CORE_READ_INTO(&e->daddr, sk, __sk_common.skc_daddr);
    BPF_CORE_READ_INTO(&e->sport, sk, __sk_common.skc_num);
    BPF_CORE_READ_INTO(&e->dport, sk, __sk_common.skc_dport);
    e->dport = __bpf_ntohs(e->dport);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

### Example 4: Simple LSM Hook

**file_access.c** (kernel 5.7+):
```c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>

SEC("lsm/file_open")
int BPF_PROG(restrict_file_open, struct file *file, int ret) {
    const char *filename;
    char comm[16];
    char name[256];

    if (ret != 0)
        return ret;

    bpf_get_current_comm(&comm, sizeof(comm));

    filename = BPF_CORE_READ(file, f_path.dentry, d_name.name);
    bpf_probe_read_kernel_str(name, sizeof(name), filename);

    // Block access to /etc/shadow for specific process
    if (__builtin_memcmp(name, "shadow", 6) == 0) {
        bpf_printk("Blocked access to %s by %s\n", name, comm);
        return -1; // EPERM
    }

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
```

---

## Modern eBPF Features

Capabilities that have matured since the early "maps + helpers" model, enabling far richer
programs (timers, dynamic memory, in-kernel data structures, safe unprivileged delegation).

### kfuncs (BPF Kernel Functions)

Type-safe kernel functions exported for direct calling from BPF programs. Unlike the fixed,
UAPI-stable helper list, kfuncs are declared per-program with `__ksym` and can evolve with the
kernel. They are the modern mechanism behind dynptrs, BPF data structures, and much more.

```c
// Declare the kfunc the kernel exports
extern struct task_struct *bpf_task_acquire(struct task_struct *p) __ksym;
extern void bpf_task_release(struct task_struct *p) __ksym;
```

### BPF Timers (5.15+)

Run a callback after a delay, fully inside the kernel, from a timer stored in a map value.

```c
struct elem { struct bpf_timer t; };
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1);
    __type(key, int);
    __type(value, struct elem);
} timers SEC(".maps");

static int cb(void *map, int *key, struct elem *e) { return 0; }

// init -> set callback -> start (nanoseconds)
bpf_timer_init(&e->t, &timers, CLOCK_MONOTONIC);
bpf_timer_set_callback(&e->t, cb);
bpf_timer_start(&e->t, 1000000000ULL, 0);
```

### bpf_loop Helper (5.17+)

Verifier-friendly bounded iteration with far higher limits than `#pragma unroll`, and without
exploding instruction count. The callback runs up to `nr_loops` times.

```c
static long body(u32 i, void *ctx) {
    // returns 0 to continue, 1 to break early
    return 0;
}
bpf_loop(1000000, body, &my_ctx, 0);
```

### Dynptrs (5.19+)

Dynamically-sized, bounds-checked pointers that let the verifier track buffer length at
runtime — safer and more flexible than fixed-size `bpf_probe_read` patterns.

```c
struct bpf_dynptr ptr;
bpf_ringbuf_reserve_dynptr(&events, len, 0, &ptr);
void *p = bpf_dynptr_data(&ptr, 0, len);  // verifier knows the bounds
bpf_ringbuf_submit_dynptr(&ptr, 0);
```

### BPF Iterators (5.8+)

Walk kernel state (tasks, sockets, map entries…) and emit it to user space via a seq_file,
read like a file from a pinned iterator. Great for snapshotting state without polling.

```c
SEC("iter/task")
int dump_tasks(struct bpf_iter__task *ctx) {
    struct task_struct *task = ctx->task;
    if (!task) return 0;
    BPF_SEQ_PRINTF(ctx->meta->seq, "%d %s\n", task->pid, task->comm);
    return 0;
}
```

### Sleepable Programs (5.10+)

Programs that may sleep/fault, allowing helpers that take page faults (e.g. user-memory reads
in uprobes, certain LSM hooks). Marked with `BPF_F_SLEEPABLE` and the `.s` section suffix.

```c
SEC("lsm.s/bprm_check_security")   // ".s" = sleepable
int BPF_PROG(check, struct linux_binprm *bprm) {
    return 0;
}
```

### In-Kernel Data Structures (6.x)

`kptr`s and allocated objects (`bpf_obj_new` / `bpf_obj_drop`) let programs build dynamic
structures the verifier tracks for ownership and safety — backed by **linked lists** and
**rbtrees** via kfuncs.

```c
struct node { struct bpf_list_node n; int val; };
// bpf_obj_new(typeof(*p)) allocates; pushed under a bpf_spin_lock
bpf_list_push_back(&my_list, &p->n);
bpf_rbtree_add(&my_tree, &p->rb, less_cb);
```

### BPF Arena (6.9+)

A sparse, demand-paged memory region shared between a BPF program and user space, addressable
by normal pointers from both sides. Enables building large dynamic data structures (hash
tables, allocators) in BPF without the constraints of fixed map entries.

### BPF Token (6.9+)

A delegation mechanism that lets an unprivileged or containerized workload perform specific
BPF operations (load programs, create maps) without holding full `CAP_BPF`/`CAP_SYS_ADMIN`.
A privileged manager mints a token (from a bpffs mount with delegation options) and hands the
fd to the workload — see [Security and Safety](#security-and-safety) for the capability model.

---

## Security and Safety

### Verifier Guarantees

The eBPF verifier ensures:

1. **Memory Safety**
   - No out-of-bounds access
   - All memory access through pointers is validated
   - Null pointer checks required

2. **Termination**
   - Bounded loops (kernel 5.3+) or loop unrolling
   - No infinite loops
   - Limited complexity (instruction count)

3. **No Undefined Behavior**
   - All code paths return a value
   - No unreachable code
   - Register initialization checked

### Verifier Checks

```c
// ❌ BAD: Unbounded loop (pre-5.3)
for (int i = 0; i < n; i++) { }

// ✅ GOOD: Bounded loop
#pragma unroll
for (int i = 0; i < 10; i++) { }

// ✅ GOOD: Bounded with verifier check (5.3+)
for (int i = 0; i < n && i < 100; i++) { }
```

```c
// ❌ BAD: Unchecked pointer
void *data = (void *)(long)ctx->data;
struct ethhdr *eth = data;
return eth->h_proto; // Verifier error!

// ✅ GOOD: Bounds check
void *data = (void *)(long)ctx->data;
void *data_end = (void *)(long)ctx->data_end;
struct ethhdr *eth = data;
if ((void *)(eth + 1) > data_end)
    return XDP_DROP;
return eth->h_proto;
```

### Required Capabilities

Loading eBPF programs requires:
- `CAP_BPF` (kernel 5.8+) for eBPF operations
- `CAP_PERFMON` for tracing programs
- `CAP_NET_ADMIN` for networking programs

**Legacy** (pre-5.8): `CAP_SYS_ADMIN` required

```bash
# Grant specific capabilities
setcap cap_bpf,cap_perfmon,cap_net_admin+eip ./my_program
```

**BPF tokens** (kernel 6.9+) provide a finer-grained alternative to handing out `CAP_BPF`:
a privileged process configures delegation on a `bpffs` mount and passes a token fd to an
otherwise-unprivileged workload, scoping exactly which BPF commands, map types, and program
types it may use (see [Modern eBPF Features](#modern-ebpf-features)).

### Unprivileged eBPF

Limited eBPF for unprivileged users (disabled by default):

```bash
# Enable (use with caution)
sysctl kernel.unprivileged_bpf_disabled=0

# Disable (recommended)
sysctl kernel.unprivileged_bpf_disabled=1
```

### Restrictions

- Limited helper functions (no arbitrary kernel memory access)
- No direct kernel pointer access
- Stack size limited to 512 bytes
- Program size limits (1M instructions)
- Map size limits (configurable)

---

## Debugging

### Common Verifier Errors

#### 1. Invalid memory access
```
R0 invalid mem access 'inv'
```
**Solution**: Add bounds checks before pointer dereference

#### 2. Unreachable instructions
```
unreachable insn 123
```
**Solution**: Ensure all code paths are reachable

#### 3. Infinite loop detected
```
back-edge from insn 45 to 12
```
**Solution**: Add loop bounds or use `#pragma unroll`

#### 4. Invalid register state
```
R1 !read_ok
```
**Solution**: Initialize register before use

### Debugging Techniques

#### 1. bpf_printk (Kernel Tracing)

```c
bpf_printk("Debug: value=%d\n", value);
```

**Read output**:
```bash
cat /sys/kernel/debug/tracing/trace_pipe
# or
bpftool prog tracelog
```

**Limitations**:
- Limited format strings
- Performance overhead
- Max 3 arguments

#### 2. bpftool Inspection

```bash
# Dump translated bytecode
bpftool prog dump xlated id 123

# Dump JIT code
bpftool prog dump jited id 123

# Show verifier log
bpftool prog load prog.o /sys/fs/bpf/prog 2>&1 | less
```

#### 3. Verbose Verifier Output

```c
// In user-space loader
LIBBPF_OPTS(bpf_object_open_opts, opts,
    .kernel_log_level = 1 | 2 | 4,  // Verbosity levels
);
obj = bpf_object__open_file("prog.o", &opts);
```

Or with bpftool:
```bash
bpftool -d prog load prog.o /sys/fs/bpf/prog
```

#### 4. Map Debugging

```bash
# Dump all map entries
bpftool map dump id 123

# Update map entry
bpftool map update id 123 key 0 0 0 0 value 1 0 0 0 0 0 0 0

# Delete entry
bpftool map delete id 123 key 0 0 0 0
```

#### 5. Statistics

```bash
# Enable statistics
bpftool feature probe kernel | grep stats
sysctl -w kernel.bpf_stats_enabled=1

# View program stats (run count, runtime)
bpftool prog show id 123
```

### Performance Profiling

#### 1. Measure Program Runtime

```c
u64 start = bpf_ktime_get_ns();
// ... program logic ...
u64 duration = bpf_ktime_get_ns() - start;
```

#### 2. Use perf with eBPF

```bash
# Profile eBPF program
perf record -e bpf:bpf_prog_run -a
perf report
```

### Common Issues

#### Issue: Program rejected by verifier
- **Check**: Verifier log for specific error
- **Solutions**: Add bounds checks, limit loop iterations, reduce complexity

#### Issue: Map update fails
- **Check**: Map is full, wrong flags
- **Solutions**: Use LRU maps, increase size, check update flags

#### Issue: Helper function not found
- **Check**: Kernel version, program type
- **Solutions**: Update kernel, use available helpers for program type

#### Issue: BTF/CO-RE errors
- **Check**: BTF available (`/sys/kernel/btf/vmlinux`)
- **Solutions**: Enable CONFIG_DEBUG_INFO_BTF, use correct libbpf version

---

## Resources

### Documentation

- **Official eBPF Docs**: https://ebpf.io/
- **Kernel Documentation**: https://www.kernel.org/doc/html/latest/bpf/
- **BPF and XDP Reference Guide**: https://docs.cilium.io/en/latest/bpf/
- **libbpf Documentation**: https://libbpf.readthedocs.io/

### Books

- **"Learning eBPF"** by Liz Rice (O'Reilly, 2023)
- **"BPF Performance Tools"** by Brendan Gregg (Addison-Wesley, 2019)
- **"Linux Observability with BPF"** by David Calavera & Lorenzo Fontana (O'Reilly, 2019)

### Key Projects

- **BCC**: https://github.com/iovisor/bcc
- **libbpf**: https://github.com/libbpf/libbpf
- **bpftool**: https://github.com/libbpf/bpftool
- **Cilium**: https://github.com/cilium/cilium
- **Tetragon**: https://github.com/cilium/tetragon
- **Katran**: https://github.com/facebookincubator/katran
- **Falco**: https://github.com/falcosecurity/falco
- **Tracee**: https://github.com/aquasecurity/tracee
- **bpftrace**: https://github.com/iovisor/bpftrace
- **Aya (Rust)**: https://github.com/aya-rs/aya — docs: https://aya-rs.dev/
- **libbpf-rs**: https://github.com/libbpf/libbpf-rs
- **sched_ext schedulers**: https://github.com/sched-ext/scx

### Example Collections

- **BCC Tools**: https://github.com/iovisor/bcc/tree/master/tools
- **libbpf-bootstrap**: https://github.com/libbpf/libbpf-bootstrap
- **Linux kernel samples**: https://github.com/torvalds/linux/tree/master/samples/bpf

### Community

- **eBPF Summit**: Annual conference
- **eBPF Slack**: https://ebpf.io/slack
- **Mailing List**: bpf@vger.kernel.org
- **Reddit**: r/ebpf

### Tutorials

- **Cilium eBPF Tutorial**: https://github.com/cilium/ebpf-tutorial
- **XDP Hands-On Tutorial**: https://github.com/xdp-project/xdp-tutorial
- **libbpf-bootstrap Examples**: Step-by-step guides

### Tools and Utilities

```bash
# Install development tools (Ubuntu/Debian)
apt install -y clang llvm libelf-dev libz-dev libbpf-dev \
    linux-tools-common linux-tools-generic bpftool

# Install BCC
apt install -y bpfcc-tools python3-bpfcc

# Install bpftrace
apt install -y bpftrace
```

---

## Quick Reference

### Common Commands

```bash
# List all eBPF programs
bpftool prog list

# List all maps
bpftool map list

# Show program by ID
bpftool prog show id <ID>

# Dump program bytecode
bpftool prog dump xlated id <ID>

# Pin program to filesystem
bpftool prog pin id <ID> /sys/fs/bpf/<name>

# Load program from object file
bpftool prog load prog.o /sys/fs/bpf/myprog

# Attach XDP program
ip link set dev <iface> xdp obj prog.o sec xdp

# Detach XDP program
ip link set dev <iface> xdp off

# Attach TC program
tc qdisc add dev <iface> clsact
tc filter add dev <iface> ingress bpf da obj prog.o

# View trace output
cat /sys/kernel/debug/tracing/trace_pipe
```

### Helper Function Categories

- **Map operations**: `bpf_map_lookup_elem`, `bpf_map_update_elem`, `bpf_map_delete_elem`
- **Time**: `bpf_ktime_get_ns`, `bpf_ktime_get_boot_ns`
- **Process/Thread**: `bpf_get_current_pid_tgid`, `bpf_get_current_uid_gid`, `bpf_get_current_comm`
- **Tracing**: `bpf_probe_read`, `bpf_probe_read_kernel`, `bpf_probe_read_user`
- **Networking**: `bpf_skb_load_bytes`, `bpf_skb_store_bytes`, `bpf_xdp_adjust_head`
- **Output**: `bpf_printk`, `bpf_perf_event_output`, `bpf_ringbuf_submit`
- **Stack**: `bpf_get_stackid`, `bpf_get_stack`

### Kernel Version Features

- **3.18** (2014): Initial eBPF support
- **4.1** (2015): BPF maps, tail calls
- **4.7** (2016): Direct packet access
- **4.8** (2016): XDP support
- **4.18** (2018): BTF (BPF Type Format)
- **5.3** (2019): Bounded loops support
- **5.5** (2020): fentry/fexit, BPF trampolines
- **5.7** (2020): LSM BPF programs
- **5.8** (2020): Ring buffer, `CAP_BPF`, BPF iterators
- **5.10** (2020): Sleepable BPF programs
- **5.13** (2021): Kernel module function calls (kfuncs)
- **5.15** (2021): BPF timers
- **5.16** (2022): Bloom filter map
- **5.17** (2022): `bpf_loop` helper
- **5.19** (2022): Dynptrs, BPF kfuncs maturing
- **6.0** (2022): Sleepable programs enhancements
- **6.4** (2023): BPF allocated objects, linked lists
- **6.6** (2023): BPF exceptions, cpumask kfuncs
- **6.9** (2024): BPF arena, BPF token
- **6.12** (2024): sched_ext (SCX) extensible scheduler

---

**Last Updated**: 2026-05
**Kernel Version Coverage**: Linux 3.18 - 6.13

## Where this connects

- [Networking](networking.md) — eBPF XDP/TC programs attach to the Linux network stack for packet processing
- [Netfilter](netfilter.md) — eBPF bypasses Netfilter at XDP for maximum performance
- [Kernel patterns](kernel_patterns.md) — RCU locking and spinlocks govern safe eBPF map access
- [Netlink](netlink.md) — programs and maps load via the `bpf()` syscall; netlink (rtnetlink) attaches XDP/TC programs to interfaces
- [sysfs](sysfs.md) — eBPF maps are an alternative to sysfs for kernel-userspace data sharing
- [bpftrace](../debugging/bpftrace.md) — the high-level front-end that compiles to the engine described here
