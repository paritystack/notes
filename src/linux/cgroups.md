# Linux Control Groups (cgroups)

> **Domain:** Linux Kernel, Systems Programming, Containers, Resource Management
> **Key Concepts:** Control Groups, Hierarchy, Controllers/Subsystems, cgroup v1 vs v2, Unified Hierarchy, cgroupfs, CPU/Memory/IO/PIDs/cpuset Controllers, PSI, Delegation, systemd, Containers

**Control groups (cgroups)** are a Linux kernel feature that organizes processes into hierarchical groups and then **limits, accounts for, and isolates** their resource usage — CPU time, memory, block I/O bandwidth, number of processes, and more. They are one of the two pillars of Linux containers: cgroups bound *how much* a group of processes may consume, while **namespaces** bound *what* those processes can see. Together they make `docker`, `podman`, `systemd`, and Kubernetes possible.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Core Concepts](#2-core-concepts)
3. [cgroup v1 vs v2](#3-cgroup-v1-vs-v2)
4. [The cgroupfs Filesystem Interface](#4-the-cgroupfs-filesystem-interface)
5. [Enabling Controllers and subtree_control](#5-enabling-controllers-and-subtree_control)
6. [The CPU Controller](#6-the-cpu-controller)
7. [The Memory Controller](#7-the-memory-controller)
8. [The IO Controller](#8-the-io-controller)
9. [The PIDs Controller](#9-the-pids-controller)
10. [The cpuset Controller](#10-the-cpuset-controller)
11. [Other Controllers](#11-other-controllers)
12. [PSI — Pressure Stall Information](#12-psi--pressure-stall-information)
13. [Delegation](#13-delegation)
14. [systemd Integration](#14-systemd-integration)
15. [Containers and cgroups](#15-containers-and-cgroups)
16. [Tools and Debugging](#16-tools-and-debugging)
17. [Resources](#17-resources)

---

## 1. Introduction

A control group does two things, and it's worth keeping them distinct:

1.  **Resource control** — impose limits (you may use at most 2 CPUs, 512 MiB of RAM, 50 MB/s of disk write bandwidth) and guarantees (you are protected up to 256 MiB before reclaim touches you).
2.  **Accounting and grouping** — track exactly how much CPU, memory, and I/O a group of processes has consumed, independent of any limit. Even with no limits set, cgroups give you per-group statistics.

cgroups are **not** isolation in the security sense. That is the job of **namespaces** (PID, mount, network, user, UTS, IPC, time, and the cgroup namespace itself). The two are orthogonal and complementary:

| Mechanism      | Question it answers              | Examples                                        |
|----------------|----------------------------------|-------------------------------------------------|
| **cgroups**    | *How much* can these processes use? | CPU quota, memory cap, I/O bandwidth, PID count |
| **namespaces** | *What* can these processes see?  | Their own PIDs, mounts, network stack, hostname |

A container runtime creates a set of namespaces (so the workload sees a private view of the system) **and** a cgroup (so the workload can't starve its neighbors). The kernel implementation lives in `kernel/cgroup/` plus per-controller code (`mm/memcontrol.c`, `kernel/sched/`, `block/blk-cgroup.c`, …).

---

## 2. Core Concepts

*   **Control group (cgroup):** A collection of processes bound by the same set of limits and accounting. Represented as a directory in a special filesystem.
*   **Hierarchy:** cgroups form a tree. Child groups inherit constraints from their ancestors — a child can never exceed limits its parent imposes. The root of the tree is the whole machine.
*   **Controller (a.k.a. subsystem):** The kernel module that actually enforces one resource type: `cpu`, `memory`, `io`, `pids`, `cpuset`, `hugetlb`, etc. A controller is *enabled* on a subtree and then exposes interface files in each group.
*   **Tasks / processes:** Every process belongs to exactly one cgroup (per hierarchy). A new process inherits its parent's cgroup; you move it by writing its PID into a control file.
*   **Interface files:** Control and statistics are exposed as plain files (`cpu.max`, `memory.current`, `io.stat`). You configure cgroups by reading and writing text to these files — no special syscall.

```
              root cgroup (/sys/fs/cgroup)
              /            |              \
        system.slice   user.slice     machine.slice   <- systemd's default layout
          /     \                          \
   sshd.service  nginx.service          docker-<id>.scope
       (procs)      (procs)                 (container procs)
```

---

## 3. cgroup v1 vs v2

cgroups exist in two generations. **v2 (the "unified hierarchy") is the modern default** on virtually all current distros (systemd ≥ v232 era, kernels ≥ 4.5 for the core, with controllers maturing through 5.x).

| Aspect                  | cgroup **v1**                                   | cgroup **v2**                                      |
|-------------------------|-------------------------------------------------|----------------------------------------------------|
| Hierarchies             | **Multiple**, independent — one tree per controller (separate `cpu`, `memory`, `blkio` mounts) | **Single unified** tree; all controllers share it  |
| Granularity             | Threads could be split across controllers       | Processes attach at the cgroup; threads only in `threaded` subtrees |
| Internal processes      | Allowed anywhere                                | Forbidden — only **leaf** cgroups hold processes ("no internal process" rule) |
| Controller enablement   | Implicit per-mount                              | Explicit, top-down via `cgroup.subtree_control`    |
| Consistency             | Each controller had its own ad-hoc interface    | Uniform `*.max` / `*.current` / `*.stat` / `*.events` naming |
| Pressure (PSI)          | No                                              | Yes (`*.pressure`)                                 |
| Mount type             | `cgroup` (one per controller)                   | `cgroup2` (single mount)                           |

**Why v2 exists:** v1's independent hierarchies meant a process could sit in different positions in the `memory` and `io` trees simultaneously, making it impossible for the memory and I/O controllers to cooperate (e.g. for coordinated writeback throttling). v2's single tree gives every process one consistent position so controllers can reason jointly.

**The "no internal processes" rule:** in v2, a cgroup that has child cgroups with controllers enabled may **not** itself contain processes. Processes live only in leaf nodes. (The root is exempt.) This keeps resource distribution unambiguous.

```bash
# Which version is in use? cgroup2 mounted on /sys/fs/cgroup => v2 (unified)
mount | grep cgroup
stat -fc %T /sys/fs/cgroup     # "cgroup2fs" => v2 ; "tmpfs" => v1/hybrid
```

**Selecting a mode** (kernel cmdline / systemd):

*   `systemd.unified_cgroup_hierarchy=1` — force unified v2 (default on modern systemd).
*   `systemd.unified_cgroup_hierarchy=0` — legacy v1.
*   `cgroup_no_v1=all` — disable all v1 controllers (clean v2). A *hybrid* mode also exists where v2 is mounted but some controllers remain on v1.

> The rest of this document focuses on **cgroup v2** unless stated otherwise.

---

## 4. The cgroupfs Filesystem Interface

cgroups are driven entirely through a pseudo-filesystem, normally mounted at `/sys/fs/cgroup`:

```bash
# v2 is a single mount of type cgroup2
mount -t cgroup2 none /sys/fs/cgroup        # (systemd does this at boot)
```

You create a group with `mkdir` and remove it (when empty) with `rmdir`:

```bash
cd /sys/fs/cgroup
mkdir myapp                  # create a child cgroup
echo $$ > myapp/cgroup.procs # move the current shell into it
cat myapp/cgroup.procs       # list PIDs in this group
rmdir myapp                  # destroy it (must contain no procs/children)
```

### Key core interface files (present in every v2 group)

| File                    | Read / Write | Purpose                                                            |
|-------------------------|--------------|--------------------------------------------------------------------|
| `cgroup.procs`          | R/W          | PIDs in this group; write a PID to migrate it here                 |
| `cgroup.threads`        | R/W          | TIDs (only meaningful in `threaded` subtrees)                      |
| `cgroup.controllers`    | R            | Controllers *available* in this group (granted by the parent)      |
| `cgroup.subtree_control`| R/W          | Controllers *enabled for children* (write `+cpu`, `-memory`, …)    |
| `cgroup.type`           | R/W          | `domain` (default) / `threaded` / `domain threaded`                |
| `cgroup.events`         | R            | `populated 0/1`, `frozen 0/1` — notifiable via `poll()`/`inotify`  |
| `cgroup.stat`           | R            | `nr_descendants`, `nr_dying_descendants`                           |
| `cgroup.freeze`         | R/W          | Write `1` to freeze (SIGSTOP-like) the whole subtree, `0` to thaw  |
| `cgroup.kill`           | W            | Write `1` to SIGKILL every process in the subtree (kernel ≥ 5.14)  |

Migrating a process is just a write — the kernel moves the PID atomically:

```bash
echo 1234 > /sys/fs/cgroup/myapp/cgroup.procs
cat /proc/1234/cgroup        # confirm: "0::/myapp"
```

---

## 5. Enabling Controllers and subtree_control

A controller must be **explicitly enabled top-down**. A group can only enable for its children the controllers it was itself granted (visible in `cgroup.controllers`). You enable/disable by writing `+name`/`-name` to `cgroup.subtree_control`:

```bash
cd /sys/fs/cgroup
cat cgroup.controllers                 # e.g. "cpuset cpu io memory hugetlb pids"
echo "+cpu +memory" > cgroup.subtree_control   # grant cpu+memory to children

mkdir workloads
cat workloads/cgroup.controllers       # now shows "cpu memory"
echo "+cpu" > workloads/cgroup.subtree_control  # pass cpu down another level
```

Rules and gotchas:

*   You can only enable a controller for children if it is present in *your* `cgroup.controllers`. Enablement therefore propagates only as far down as each level chooses to grant it.
*   **No internal processes:** once a non-root group has controllers enabled in `subtree_control` *and* has child groups, it must not contain processes directly — move them into a leaf first.
*   Disabling a controller (`-cpu`) requires that no descendant still depends on it.

---

## 6. The CPU Controller

Distributes CPU time, both as a **hard bandwidth cap** and as a **proportional weight**.

| File             | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `cpu.max`        | Bandwidth limit: `"$QUOTA $PERIOD"` in microseconds. `"max 100000"` = no limit |
| `cpu.weight`     | Proportional share, `1`–`10000`, default `100` (relative to siblings)   |
| `cpu.weight.nice`| Same as `cpu.weight` but expressed on the nice scale (`-20`..`19`)       |
| `cpu.stat`       | `usage_usec`, `nr_periods`, `nr_throttled`, `throttled_usec`            |
| `cpu.pressure`   | PSI stall metrics for CPU (see §12)                                     |

**Bandwidth (`cpu.max`):** quota is how much CPU time the group may use per period. `"50000 100000"` means 50 ms of CPU every 100 ms = **0.5 of one CPU**. `"200000 100000"` = up to **2 full CPUs**. When the quota is exhausted within a period the group is *throttled* until the next period — visible in `cpu.stat`'s `nr_throttled`/`throttled_usec`.

**Weight (`cpu.weight`):** only matters under contention. Two sibling groups with weights `100` and `300` split a busy CPU `25% / 75%`. With no contention, either can use the whole CPU.

```bash
cd /sys/fs/cgroup
echo "+cpu" > cgroup.subtree_control
mkdir batch
echo "50000 100000" > batch/cpu.max     # cap at half a CPU
echo 200 > batch/cpu.weight             # 2x the default share under contention
echo $$ > batch/cgroup.procs
cat batch/cpu.stat                       # watch nr_throttled climb under load
```

---

## 7. The Memory Controller

Accounts and limits memory (anonymous, page cache, kernel slab, socket buffers) per group. This is the basis of container memory limits — see also [memory_management.md](./memory_management.md) §10 (cgroup memory control) and §11 (OOM killer).

| File              | Meaning                                                                          |
|-------------------|----------------------------------------------------------------------------------|
| `memory.max`      | **Hard limit.** Exceeding it triggers reclaim, then in-cgroup OOM kill           |
| `memory.high`     | **Throttle limit.** Over it, allocations are heavily throttled + reclaimed (no kill) |
| `memory.low`      | **Soft protection.** Memory under this is protected from reclaim when possible    |
| `memory.min`      | **Hard protection.** Memory under this is never reclaimed (can OOM the system)    |
| `memory.current`  | Current usage in bytes                                                            |
| `memory.swap.max` | Cap on swap usage for the group                                                   |
| `memory.stat`     | Detailed breakdown: `anon`, `file`, `slab`, `sock`, `shmem`, `pgfault`, …         |
| `memory.events`   | Counters: `low`, `high`, `max`, `oom`, `oom_kill`                                 |
| `memory.pressure` | PSI memory stall metrics (see §12)                                                |

The distinction that trips people up: **`memory.high` throttles, `memory.max` kills.** Set `high` a bit below `max` to get aggressive reclaim and back-pressure *before* the OOM killer fires.

```bash
cd /sys/fs/cgroup
echo "+memory" > cgroup.subtree_control
mkdir webapp
echo 512M > webapp/memory.max
echo 450M > webapp/memory.high      # throttle/reclaim before hitting the hard cap
echo 128M > webapp/memory.min       # guarantee at least 128M, never reclaimed
echo $$ > webapp/cgroup.procs
cat webapp/memory.current
cat webapp/memory.events            # oom_kill count, etc.
```

When a group exceeds `memory.max` and reclaim can't recover enough, the kernel runs an **OOM kill scoped to that cgroup** rather than the whole machine.

---

## 8. The IO Controller

Controls block-device bandwidth and IOPS. Limits are **per device**, keyed by `major:minor` (find them with `lsblk` / `ls -l /dev`).

| File          | Meaning                                                                            |
|---------------|------------------------------------------------------------------------------------|
| `io.max`      | Hard limits per device: `rbps`, `wbps` (bytes/s), `riops`, `wiops` (ops/s)         |
| `io.weight`   | Proportional weight `1`–`10000` (default `100`); needs `bfq` or `blk-iocost`       |
| `io.latency`  | Target latency (ms) per device — a latency-based QoS / protection mechanism        |
| `io.cost.*`   | The cost-model (`blk-iocost`) tunables for proportional control on any device       |
| `io.stat`     | Per-device `rbytes`, `wbytes`, `rios`, `wios` accounting                            |
| `io.pressure` | PSI I/O stall metrics (see §12)                                                    |

```bash
cd /sys/fs/cgroup
echo "+io" > cgroup.subtree_control
mkdir backup
# Throttle device 8:0 (e.g. /dev/sda) to 10 MB/s write, 1 MB/s read
echo "8:0 wbps=10485760 rbps=1048576" > backup/io.max
echo $$ > backup/cgroup.procs
cat backup/io.stat
```

Proportional weighting (`io.weight`) requires an I/O scheduler/cost model that supports it — historically `cfq` (v1), now **`bfq`** or the schedulerless **`blk-iocost`** cost model in v2. Pure `io.max` rate limits work regardless.

---

## 9. The PIDs Controller

Limits the **number of processes/threads** a group may create — the standard defense against fork bombs and runaway spawning.

| File           | Meaning                                            |
|----------------|----------------------------------------------------|
| `pids.max`     | Maximum number of PIDs (`max` = unlimited)         |
| `pids.current` | Current count of PIDs in the subtree               |
| `pids.events`  | `max` — number of times a fork was denied by the cap |

```bash
cd /sys/fs/cgroup
echo "+pids" > cgroup.subtree_control
mkdir sandbox
echo 100 > sandbox/pids.max          # cap at 100 processes
echo $$ > sandbox/cgroup.procs
cat sandbox/pids.current
# A fork bomb inside 'sandbox' now hits EAGAIN once it reaches 100.
```

---

## 10. The cpuset Controller

Pins a group to specific **CPUs** and **NUMA memory nodes** — used for cache-locality, NUMA-aware placement, and carving the machine into isolated partitions.

| File                    | Meaning                                                            |
|-------------------------|--------------------------------------------------------------------|
| `cpuset.cpus`           | Allowed logical CPUs, e.g. `0-3` or `0,2,4`                         |
| `cpuset.mems`           | Allowed NUMA memory nodes, e.g. `0`                                 |
| `cpuset.cpus.effective` | CPUs actually granted after intersecting with ancestors            |
| `cpuset.cpus.partition` | `member` / `root` / `isolated` — create an exclusive CPU partition |

```bash
cd /sys/fs/cgroup
echo "+cpuset" > cgroup.subtree_control
mkdir lowlatency
echo "2-3" > lowlatency/cpuset.cpus      # pin to CPUs 2 and 3
echo "0"   > lowlatency/cpuset.mems      # allocate from NUMA node 0
echo $$    > lowlatency/cgroup.procs
cat lowlatency/cpuset.cpus.effective
```

Setting `cpuset.cpus.partition` to `root` (or `isolated`) carves the listed CPUs out for the group's exclusive use, keeping other tasks off them — useful for real-time / latency-sensitive workloads. Cross-reference NUMA concepts in [memory_management.md](./memory_management.md) §4.2.

---

## 11. Other Controllers

| Controller   | Key files / interface                | Purpose                                                        |
|--------------|--------------------------------------|----------------------------------------------------------------|
| `hugetlb`    | `hugetlb.<size>.max`, `.current`     | Limit/account explicit huge-page (HugeTLB) usage per group     |
| `rdma`       | `rdma.max`, `rdma.current`           | Cap RDMA/InfiniBand resources (HCA handles, objects)           |
| `misc`       | `misc.max`, `misc.current`           | Account scalar resources (e.g. AMD SEV ASIDs, Intel SGX EPC)   |
| `freezer`    | `cgroup.freeze` (core file in v2)    | Suspend/resume an entire subtree (checkpoint, migration)       |
| `perf_event` | (enables `perf` to target a cgroup)  | Scope performance monitoring to a cgroup                       |

In v2 the freezer is folded into the core `cgroup.freeze` file rather than a separate controller:

```bash
echo 1 > /sys/fs/cgroup/myapp/cgroup.freeze   # SIGSTOP-like pause of whole subtree
echo 0 > /sys/fs/cgroup/myapp/cgroup.freeze   # resume
cat /sys/fs/cgroup/myapp/cgroup.events        # "frozen 1"
```

---

## 12. PSI — Pressure Stall Information

A cgroup v2 feature: each group exposes **`cpu.pressure`**, **`memory.pressure`**, and **`io.pressure`** files quantifying how much time tasks stalled waiting for that resource. (A system-wide version lives in `/proc/pressure/`.)

```
some avg10=0.00 avg60=0.12 avg300=0.05 total=1234567
full avg10=0.00 avg60=0.08 avg300=0.03 total=987654
```

*   **`some`** — share of time *at least one* task was stalled on the resource.
*   **`full`** — share of time *all* runnable tasks were stalled (pure lost work). Not reported for CPU.
*   `avg10/60/300` — percentages over 10 s / 60 s / 5 min windows; `total` — cumulative microseconds.

PSI is the signal behind modern OOM/eviction daemons (e.g. **`oomd`/`systemd-oomd`**) and autoscalers: rather than waiting for hard limits or the kernel OOM killer, they watch pressure rising and act early.

```bash
cat /sys/fs/cgroup/webapp/memory.pressure
cat /proc/pressure/io                 # system-wide I/O pressure
```

---

## 13. Delegation

**Delegation** hands a subtree of the cgroup hierarchy to a less-privileged user or to a container, letting them manage their own sub-cgroups without root.

How it works: a privileged manager (root / systemd) `chown`s the **delegated directory** and a specific set of its interface files (`cgroup.procs`, `cgroup.subtree_control`, `cgroup.threads`) to the target user. That user can then create children, enable controllers granted to them, and move *their own* processes around — but cannot escape upward or grant themselves controllers the parent didn't enable.

Containment rules that make this safe:

*   A delegatee can only move a process between cgroups if it has write access to the **common ancestor**'s `cgroup.procs` as well as the source and destination — preventing it from pulling in arbitrary host processes.
*   A delegatee can only enable controllers present in its own `cgroup.controllers` (i.e. ones the parent passed down via `subtree_control`).
*   The **`nsdelegate`** mount option makes cgroup-namespace boundaries act as delegation boundaries, so a containerized process treats its namespace root as the top of its world.

```bash
mount -o remount,nsdelegate /sys/fs/cgroup
mkdir /sys/fs/cgroup/user-subtree
chown alice:alice /sys/fs/cgroup/user-subtree \
                  /sys/fs/cgroup/user-subtree/cgroup.procs \
                  /sys/fs/cgroup/user-subtree/cgroup.subtree_control \
                  /sys/fs/cgroup/user-subtree/cgroup.threads
# alice can now manage cgroups under user-subtree without root
```

---

## 14. systemd Integration

On modern Linux, **systemd is the cgroup manager.** It owns `/sys/fs/cgroup` and expects to be the single writer — you should generally configure resources *through* systemd rather than poking cgroupfs directly under `system.slice`. systemd organizes the tree into three unit types:

*   **Slices** (`*.slice`) — partition the tree (`system.slice`, `user.slice`, `machine.slice`). They carry resource policy that children inherit.
*   **Services** (`*.service`) — daemons started by systemd; each gets its own cgroup.
*   **Scopes** (`*.scope`) — groups of externally-started processes (login sessions, VMs, containers) placed under systemd's management.

Resource control via unit directives (in a unit file or with `systemctl set-property`):

```ini
# /etc/systemd/system/nginx.service.d/limits.conf
[Service]
CPUQuota=50%          # -> cpu.max  (50% of one CPU)
CPUWeight=200         # -> cpu.weight
MemoryMax=512M        # -> memory.max
MemoryHigh=450M       # -> memory.high
IOWeight=100          # -> io.weight
TasksMax=100          # -> pids.max
```

```bash
systemctl set-property nginx.service MemoryMax=512M CPUQuota=50%
systemctl daemon-reload

# Run a one-off command in a transient scope/service with limits:
systemd-run --scope -p CPUQuota=20% -p MemoryMax=200M stress -c 2

# Inspect the live tree and resource usage:
systemd-cgls                 # tree of slices/services/scopes and their PIDs
systemd-cgtop                # top-like live CPU/Memory/IO per cgroup
systemctl status nginx       # shows the unit's cgroup path and member PIDs
```

---

## 15. Containers and cgroups

Container runtimes translate user-facing flags into cgroup interface files:

| Runtime flag                         | Resulting cgroup setting                        |
|--------------------------------------|-------------------------------------------------|
| `docker run --cpus=2`                | `cpu.max = "200000 100000"`                     |
| `docker run --cpu-shares=512`        | `cpu.weight` (scaled)                           |
| `docker run --memory=512m`           | `memory.max = 512M`                             |
| `docker run --memory-swap=1g`        | `memory.swap.max`                               |
| `docker run --pids-limit=100`        | `pids.max = 100`                                |
| `docker run --blkio-weight=500`      | `io.weight`                                     |

Under the hood **runc** (the OCI runtime used by Docker/containerd) creates the cgroup, applies limits from the OCI spec's `linux.resources`, and on cgroup-v2 hosts can drive systemd via the `systemd` cgroup driver (so containers appear as scopes under `system.slice`/`machine.slice`).

**Kubernetes** layers its QoS model on top: the kubelet creates a cgroup hierarchy (`kubepods.slice` → per-QoS-class → per-pod → per-container) and maps a container's `requests`/`limits` onto `cpu.weight`/`cpu.max` and `memory.min`/`memory.max`.

**The cgroup namespace** (`CLONE_NEWCGROUP`) virtualizes the cgroup path a process sees. Inside a container, `/proc/self/cgroup` shows a root-relative path (`/`) instead of the real host path (`/system.slice/docker-<id>.scope/...`), preventing information leaks and making the container's cgroup view self-contained — this is what `nsdelegate` (see §13) keys off.

```bash
cat /proc/self/cgroup                 # on host: full path; in container: virtualized
docker run --rm --cpus=1.5 --memory=256m alpine cat /sys/fs/cgroup/cpu.max
```

---

## 16. Tools and Debugging

| Tool / file                       | Shows                                                          |
|-----------------------------------|----------------------------------------------------------------|
| `systemd-cgls`                    | The full cgroup tree with the units and PIDs in each group     |
| `systemd-cgtop`                   | Live `top`-style CPU/memory/IO/PIDs usage per cgroup           |
| `cat /proc/<pid>/cgroup`          | Which cgroup a given process belongs to (v2: a single `0::` line) |
| `cat /proc/cgroups`               | Available controllers and whether each is enabled (v1-style listing) |
| `cat /sys/fs/cgroup/.../*.stat`   | Per-resource accounting (`cpu.stat`, `memory.stat`, `io.stat`) |
| `cat /sys/fs/cgroup/.../*.pressure` | PSI stall metrics for the group                              |
| `cat /sys/fs/cgroup/cgroup.controllers` | Controllers compiled-in and available at the root        |
| `lscgroup` / `cgcreate` / `cgexec`| libcgroup userspace tools (create groups, launch a cmd in one) |
| `stat -fc %T /sys/fs/cgroup`      | `cgroup2fs` (v2) vs `tmpfs` (v1/hybrid)                         |

```bash
# Where is process 1234, and what's its CPU/memory accounting?
cat /proc/1234/cgroup
G=/sys/fs/cgroup$(sed 's/^0:://' /proc/1234/cgroup)
cat "$G/cpu.stat" "$G/memory.current" "$G/memory.pressure"

systemd-cgtop -d 2            # refresh every 2s
```

---

## 17. Resources

### Official Documentation
- [Control Group v2 — kernel admin guide](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html)
- [cgroup v1 controllers — kernel docs](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v1/index.html)
- [PSI — Pressure Stall Information](https://www.kernel.org/doc/html/latest/accounting/psi.html)
- [systemd.resource-control(5)](https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html)
- `man 7 cgroups`, `man 7 cgroup_namespaces`

### Articles
- LWN.net cgroup coverage (the unified hierarchy, the "no internal processes" rule, PSI, io.cost)
- [systemd: Control Group Interfaces](https://systemd.io/CGROUP_DELEGATION/) — the authoritative delegation contract

### Related Notes
- [Memory Management](./memory_management.md) — §10 cgroup memory control, §11 OOM killer, §4 NUMA
- [systemd](./systemd.md) — units, slices, and service management
- [Kernel Architecture](./kernel.md) — where the controllers live in the source tree

### Books
- "Linux Kernel Development" — Robert Love
- "The Linux Programming Interface" — Michael Kerrisk (process groups, namespaces, resource limits)

## Where this connects

- [Namespaces](namespace.md) — the other half of container isolation
- [systemd](systemd.md) — drives cgroup delegation on modern systems
- [Process internals](process_internals.md) — what cgroups account and limit
- [Virtualization](virtualization.md), [Container networking](../networking/container_networking.md) — containers built on cgroups + namespaces
- [Memory management](memory_management.md) — the memory controller's accounting
