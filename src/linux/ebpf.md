# eBPF (Extended Berkeley Packet Filter)

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Program Types](#program-types)
4. [eBPF Maps](#ebpf-maps)
5. [Development Tools](#development-tools)
6. [Writing eBPF Programs](#writing-ebpf-programs)
7. [Common Use Cases](#common-use-cases)
8. [Examples](#examples)
9. [Security and Safety](#security-and-safety)
10. [Debugging](#debugging)
11. [Resources](#resources)

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

### Other Tools

- **Cilium**: Container networking with eBPF
- **Katran**: Layer 4 load balancer (Facebook)
- **Falco**: Runtime security monitoring
- **Pixie**: Observability platform
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
- **Katran**: https://github.com/facebookincubator/katran
- **Falco**: https://github.com/falcosecurity/falco
- **bpftrace**: https://github.com/iovisor/bpftrace

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
- **4.4** (2016): XDP support
- **4.8** (2016): Direct packet access
- **4.18** (2018): BTF (BPF Type Format)
- **5.2** (2019): Bounded loops support
- **5.7** (2020): LSM BPF programs
- **5.8** (2020): Ring buffer, `CAP_BPF`
- **5.13** (2021): Kernel module function calls
- **6.0** (2022): Sleepable programs enhancements

---

**Last Updated**: 2024
**Kernel Version Coverage**: Linux 3.18 - 6.x
