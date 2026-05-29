# Linux Memory Management

> **Domain:** Linux Kernel, Systems Programming, OS Theory
> **Key Concepts:** Virtual Memory, MMU, Paging, Page Tables, TLB, Zones, NUMA, Slab Allocator, Page Cache, Swap, Reclaim, Huge Pages, cgroups, OOM

Memory Management is the subsystem responsible for managing the computer's primary memory. It provides processes with the illusion of a large, contiguous private memory space (Virtual Memory) while multiplexing the limited physical RAM and backing storage (swap, files).

## Table of Contents
1. [Introduction](#1-introduction)
2. [Address Space Layout](#2-address-space-layout)
3. [Paging and the MMU](#3-paging-and-the-mmu)
4. [Physical Memory Organization](#4-physical-memory-organization)
5. [Kernel Memory Allocation](#5-kernel-memory-allocation)
6. [Page Faults](#6-page-faults)
7. [Page Cache and Writeback](#7-page-cache-and-writeback)
8. [Swap and Memory Reclaim](#8-swap-and-memory-reclaim)
9. [Huge Pages](#9-huge-pages)
10. [cgroup Memory Control](#10-cgroup-memory-control)
11. [The OOM Killer](#11-the-oom-killer)
12. [Overcommit and mmap](#12-overcommit-and-mmap)
13. [Tools and Debugging](#13-tools-and-debugging)
14. [Resources](#14-resources)

---

## 1. Introduction

Every process on Linux thinks it starts at address `0x000...` and owns the entire address space (e.g., 48-bit canonical addressing on x86_64, giving 128 TiB of user space). This abstraction is **Virtual Memory**, and it underpins almost everything else in the subsystem.

*   **Benefits:**
    1.  **Isolation:** Process A cannot touch Process B's memory.
    2.  **Security:** User space cannot touch Kernel space; pages carry permission bits (R/W/X, NX).
    3.  **Efficiency:** RAM is only allocated when strictly needed (Lazy Allocation / demand paging).
    4.  **Flexibility:** The same physical page can be shared (COW, shared libraries, page cache) or backed by disk (swap, mmap'd files).

The core data structures live in `mm/` in the kernel. Each process has an `mm_struct` (its address space), which holds a tree of `vm_area_struct` (VMA) regions and a pointer to the page tables.

---

## 2. Address Space Layout

A process's virtual address space is split between **user space** (low addresses) and **kernel space** (high addresses, shared across all processes). On x86_64 the split is at the canonical-address boundary, with the kernel occupying the top half.

```
 0xffffffffffffffff  +-----------------------+
                     |     Kernel space      |  (shared, mapped in every process)
 0xffff800000000000  +-----------------------+
                     |    (non-canonical)    |
 0x00007fffffffffff  +-----------------------+
                     |        Stack          |  grows down
                     |          |            |
                     |          v            |
                     |                       |
                     |    mmap region        |  shared libs, mmap(), malloc arenas
                     |          ^            |
                     |          |            |
                     |         Heap          |  grows up (brk/sbrk)
                     +-----------------------+
                     |      BSS / Data       |  globals
                     |        Text           |  code (read-only, executable)
 0x0000000000400000  +-----------------------+
                     |     NULL guard        |
 0x0000000000000000  +-----------------------+
```

*   **VMAs (`vm_area_struct`):** Each contiguous region with uniform permissions/backing is one VMA — e.g. the text segment, a mapped file, an anonymous `mmap`. The kernel keeps them in a maple tree (formerly a red-black tree + linked list) for fast lookup on page fault.
*   **ASLR:** Address Space Layout Randomization randomizes the base of stack, heap, mmap, and (with PIE) text to harden against exploitation.
*   **Inspect it:** `/proc/<pid>/maps` lists every VMA with its address range, permissions, and backing file.

```bash
cat /proc/self/maps        # VMAs of the cat process
cat /proc/<pid>/smaps      # per-VMA RSS, PSS, swap, dirty pages
```

---

## 3. Paging and the MMU

The CPU does not operate on Physical Addresses for normal loads/stores. It issues Virtual Addresses, and the **Memory Management Unit (MMU)** translates them to physical addresses using the page tables.

*   **The Page:** Memory is divided into fixed-size chunks — **Pages** (virtual) and **Page Frames** (physical). The base size is 4 KiB on most architectures.
*   **The Page Table:** A radix-tree data structure in RAM that maps Virtual Page Number (VPN) → Physical Frame Number (PFN), plus permission and status bits (present, writable, accessed, dirty, NX).

### 3.1. Multi-Level Page Tables (x86_64)

A single flat array mapping all 48 bits would be enormous, so Linux uses a 4-level (optionally 5-level, with `la57`/57-bit) tree. A virtual address is sliced into indexes, one per level:

```
 Virtual address (48-bit)
 +--------+--------+--------+--------+------------+
 |  PGD   |  PUD   |  PMD   |  PTE   |   offset   |
 |  9 bit |  9 bit |  9 bit |  9 bit |   12 bit   |
 +--------+--------+--------+--------+------------+
     |        |        |        |          |
     v        v        v        v          v
   PGD ---> PUD ---> PMD ---> PTE ---> Physical Frame + offset
```

1.  **PGD** (Page Global Directory)
2.  **P4D** (only with 5-level paging)
3.  **PUD** (Page Upper Directory)
4.  **PMD** (Page Middle Directory) — a PMD entry can map a 2 MiB huge page directly.
5.  **PTE** (Page Table Entry) → points to the 4 KiB physical frame.

`CR3` holds the physical address of the current process's PGD; a context switch reloads `CR3`.

### 3.2. TLB (Translation Lookaside Buffer)

Walking the multi-level tree for every memory access is slow. The **TLB** is a small, fast CPU cache of recent VPN→PFN translations.

| Event   | Cost                | Notes                                  |
|---------|---------------------|----------------------------------------|
| TLB hit | ~1 CPU cycle        | Translation served from cache          |
| TLB miss| ~100+ CPU cycles    | Hardware page walk (up to 4–5 reads)   |

*   **Flushes:** A context switch can flush the TLB. **PCID** (Process Context IDs) and **ASIDs** let entries from multiple address spaces coexist to avoid full flushes.
*   **Huge pages** reduce TLB pressure (one entry covers 2 MiB or 1 GiB) — see §9.

---

## 4. Physical Memory Organization

Physical RAM is not uniform. Linux models it with **nodes** (NUMA) and **zones**.

### 4.1. Zones

Different physical address ranges have different capabilities (legacy DMA limits, 32-bit device addressing). The kernel groups page frames into zones:

| Zone           | Purpose                                                        |
|----------------|----------------------------------------------------------------|
| `ZONE_DMA`     | Legacy 24-bit DMA devices (first 16 MiB)                       |
| `ZONE_DMA32`   | 32-bit DMA-capable devices (first 4 GiB)                       |
| `ZONE_NORMAL`  | Regular memory directly mapped by the kernel                   |
| `ZONE_HIGHMEM` | Memory above the kernel direct map (32-bit only; gone on 64-bit)|
| `ZONE_MOVABLE` | Reclaimable/migratable pages (hotplug, fragmentation control)  |

Each zone has **watermarks** (min/low/high) that drive reclaim — see §8.

### 4.2. NUMA (Non-Uniform Memory Access)

On multi-socket systems, each CPU has local memory that is faster to access than remote memory. Linux represents each as a **node** (`pglist_data`). Allocation policies (`numactl`, `set_mempolicy`) control whether memory is bound, interleaved, or preferred-local.

```bash
numactl --hardware    # node topology and distances
numastat              # per-node hit/miss/foreign allocation counters
```

### 4.3. `struct page`

Every physical frame is described by a `struct page` (now increasingly a `folio`) in the `mem_map` array — tracking reference count, flags (locked, dirty, LRU), and mapping. This metadata costs RAM proportional to physical memory.

---

## 5. Kernel Memory Allocation

The kernel needs memory for itself (process descriptors, inodes, network buffers) and serves several allocator layers.

### 5.1. Page Allocator (Buddy System)

Manages raw physical pages — the foundation everything else is built on.

*   **Concept:** Free memory is tracked in blocks of order `0..MAX_ORDER` ($2^0$ pages, $2^1$ pages, …).
*   **Allocation:** Need 4 pages but only an 8-page block exists? Split the 8 into two 4s; give one, keep the other.
*   **Free:** When two adjacent "buddies" are both free, merge them back into a larger block. This combats fragmentation.
*   **API:** `alloc_pages()`, `__get_free_pages()`, `free_pages()`.

### 5.2. Slab Allocator (SLUB)

The Buddy System is too coarse (minimum 4 KiB). The kernel constantly needs tiny objects (e.g., a 256-byte `dentry`). SLUB (the default since ~2.6; SLAB/SLOB are legacy) sits on top of the page allocator.

*   **Concept:** Carve a page (or contiguous pages) into fixed-size object slots.
*   **Caches:** Dedicated caches for frequently used objects (`task_struct`, `inode_cache`, `dentry`). Created with `kmem_cache_create()`.
*   **kmalloc:** Uses generic power-of-two size caches (`kmalloc-32`, `kmalloc-64`, …) for general allocations.

### 5.3. kmalloc vs vmalloc

| Allocator  | Backing                       | Use when…                                       |
|------------|-------------------------------|-------------------------------------------------|
| `kmalloc`  | Physically **contiguous**     | Small/medium allocations, DMA buffers           |
| `vmalloc`  | Virtually contiguous only     | Large buffers where physical contiguity is hard |

`vmalloc` stitches non-contiguous pages into a contiguous virtual range, so it's slower (extra page-table setup, more TLB pressure) but avoids fragmentation failures.

### 5.4. GFP Flags

Allocation behavior is controlled by **GFP (Get Free Pages) flags**:

*   `GFP_KERNEL` — may sleep, normal kernel context.
*   `GFP_ATOMIC` — must not sleep (interrupt/atomic context); draws on emergency reserves.
*   `GFP_NOWAIT`, `__GFP_ZERO`, `GFP_DMA`, `__GFP_HIGHMEM`, etc.

---

## 6. Page Faults

What happens when a process accesses a virtual address with no valid physical mapping? A **page fault** traps into the kernel's fault handler, which consults the VMA covering that address.

1.  **Minor Fault (Soft):** No disk I/O.
    *   The page is already in RAM (e.g., shared by another process, in the page cache, or anonymous memory reserved but not yet mapped).
    *   Kernel updates the page table and resumes. Fast.
2.  **Major Fault (Disk I/O):**
    *   The data must be fetched — swapped out, or a not-yet-read region of an mmap'd file.
    *   Kernel sleeps the process, issues I/O, fills a frame, updates the page table, resumes.
3.  **Invalid Fault (Segfault):**
    *   Access to an address in no VMA, or a permission violation (writing read-only memory).
    *   Kernel delivers `SIGSEGV`.

### 6.1. Demand Paging & Copy-on-Write

*   **Demand paging:** `malloc`/`mmap` of anonymous memory installs no physical frame up front. The first *write* triggers a minor fault that maps a fresh zeroed page (often initially the shared zero page on read).
*   **Copy-on-Write (COW):** `fork()` does **not** copy the parent's pages. Parent and child share them read-only. The first write by either triggers a fault, and the kernel duplicates just that page. This makes `fork()` cheap and `fork()+exec()` near-free.

### 6.2. Anonymous vs File-Backed Pages

| Type         | Backed by         | Reclaimed by                |
|--------------|-------------------|-----------------------------|
| Anonymous    | swap (if enabled) | Writing to swap             |
| File-backed  | a file on disk    | Writeback (if dirty) / drop |

---

## 7. Page Cache and Writeback

Linux caches file data in RAM in the **page cache** — this is why a second read of a file is fast and why "free" memory always looks low (the `buff/cache` column in `free`).

*   **Reads:** File reads populate the page cache; subsequent reads are served from RAM.
*   **Dirty pages:** Writes mark cached pages dirty; they are flushed to disk asynchronously by per-device **writeback** kernel threads.
*   **Tuning:** `vm.dirty_ratio` / `vm.dirty_background_ratio` control when writeback starts and when writers are throttled.
*   The page cache is reclaimable on demand — clean pages can simply be dropped.

```bash
sync; echo 3 > /proc/sys/vm/drop_caches   # drop clean caches (debugging only)
```

---

## 8. Swap and Memory Reclaim

When free memory falls below zone watermarks, the kernel **reclaims** pages.

*   **LRU lists:** Pages live on **active** and **inactive** LRU lists (separately for anonymous and file pages). The kernel ages pages and reclaims from the inactive lists first.
*   **`kswapd`:** A per-node background thread that reclaims when memory drops below the *low* watermark, until it rises above *high*.
*   **Direct reclaim:** If allocation can't be satisfied even after `kswapd`, the allocating task reclaims synchronously (a latency hit).
*   **Swap:** Anonymous pages have no file backing, so to evict them they're written to a swap area (partition or file). File-backed clean pages need no swap — they're just dropped and re-read later.
*   **`vm.swappiness` (0–200, default 60):** Bias between reclaiming anonymous (swap) vs file-backed (cache) pages. Lower values favor keeping anonymous memory in RAM.

```bash
swapon --show          # active swap areas
cat /proc/swaps
sysctl vm.swappiness
```

---

## 9. Huge Pages

Using 2 MiB or 1 GiB pages instead of 4 KiB dramatically reduces TLB misses (one entry covers far more memory) and shortens page walks. Essential for databases (Postgres, Oracle) and large-heap JVMs.

| Mechanism                       | How it's used                                  | Tradeoff                                   |
|---------------------------------|------------------------------------------------|--------------------------------------------|
| **HugeTLB** (explicit)          | Reserved pool, `mmap(MAP_HUGETLB)` / hugetlbfs | Predictable, but must be pre-reserved      |
| **Transparent Huge Pages (THP)**| Kernel promotes/collapses pages automatically  | Zero config, but `khugepaged` jitter, bloat|

```bash
cat /proc/meminfo | grep -i huge
cat /sys/kernel/mm/transparent_hugepage/enabled   # always | madvise | never
echo 512 > /proc/sys/vm/nr_hugepages              # reserve 512 explicit huge pages
```

THP is often set to `madvise` (opt-in via `madvise(MADV_HUGEPAGE)`) for latency-sensitive workloads to avoid unexpected stalls.

---

## 10. cgroup Memory Control

Control groups (cgroup v2) let the kernel account and limit memory per group of processes — the basis of container memory limits.

*   **`memory.max`:** Hard limit; exceeding it triggers reclaim, then in-cgroup OOM.
*   **`memory.high`:** Soft limit; over it, the kernel throttles and aggressively reclaims (no kill).
*   **`memory.current`:** Current usage. **`memory.stat`** breaks down anon/file/slab/etc.
*   **PSI (`memory.pressure`):** Pressure Stall Information quantifies time tasks stall waiting on memory — useful for autoscaling/eviction.

```bash
cat /sys/fs/cgroup/<group>/memory.max
cat /sys/fs/cgroup/<group>/memory.current
cat /sys/fs/cgroup/<group>/memory.pressure
```

---

## 11. The OOM Killer

When RAM + swap is exhausted and reclaim can't keep up, the **Out-Of-Memory killer** selects a victim process and kills it to free memory.

*   **Scoring:** Each task has an `oom_score` derived mainly from its memory footprint. Tune the bias via `/proc/<pid>/oom_score_adj` (`-1000` to disable, `+1000` to prefer killing).
*   **cgroup OOM:** With cgroup v2, hitting `memory.max` triggers an OOM scoped to that cgroup, not the whole machine.
*   **Diagnosis:** OOM events are logged to the kernel ring buffer.

```bash
cat /proc/<pid>/oom_score
cat /proc/<pid>/oom_score_adj
dmesg | grep -i "killed process"
```

---

## 12. Overcommit and mmap

*   **Overcommit:** Linux allows allocating more virtual memory than physically exists. `malloc`/`mmap` succeed immediately; physical frames are consumed only on first write (demand paging). Modes via `vm.overcommit_memory`:
    *   `0` — heuristic (default): refuse only obviously absurd allocations.
    *   `1` — always overcommit (never refuse).
    *   `2` — strict: limit to swap + `overcommit_ratio`% of RAM; can return `ENOMEM`.
*   **`mmap` / `brk`:** `brk`/`sbrk` adjust the heap end; `mmap` creates arbitrary anonymous or file-backed VMAs. glibc `malloc` uses both (`brk` for small, `mmap` for large allocations).
*   **KSM (Kernel Samepage Merging):** Scans and deduplicates identical anonymous pages (common across VMs); opt-in via `madvise(MADV_MERGEABLE)`. Trades CPU for RAM savings.

---

## 13. Tools and Debugging

| Tool / file                | Shows                                                    |
|----------------------------|----------------------------------------------------------|
| `free -h`                  | RAM/swap totals, used, and page cache (`buff/cache`)     |
| `vmstat 1`                 | Swap-in/out, paging, run queue over time                 |
| `cat /proc/meminfo`        | Detailed kernel memory stats (incl. huge pages, slab)    |
| `slabtop`                  | Live view of active slab caches                          |
| `cat /proc/<pid>/maps`     | A process's VMAs                                          |
| `cat /proc/<pid>/smaps`    | Per-VMA RSS/PSS/swap/dirty breakdown                     |
| `smem`                     | Per-process PSS/USS (shared-aware memory accounting)     |
| `numastat` / `numactl -H`  | Per-NUMA-node allocation stats and topology              |
| `perf mem` / `perf record` | Memory-access profiling, TLB and cache events            |
| `dmesg`                    | OOM kills, allocation failures, THP messages             |

```bash
free -h
vmstat 1 5
grep -E 'Huge|Slab|Dirty|SwapCached' /proc/meminfo
slabtop -o | head
```

---

## 14. Resources

### Official Documentation
- [Linux Kernel — Memory Management docs](https://www.kernel.org/doc/html/latest/admin-guide/mm/index.html)
- [Documentation/vm in the kernel tree](https://www.kernel.org/doc/html/latest/mm/index.html)
- [cgroup v2 — Memory controller](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory)

### Books
- "Understanding the Linux Kernel" — Bovet & Cesati
- "Understanding the Linux Virtual Memory Manager" — Mel Gorman (free PDF online)
- "Linux Kernel Development" — Robert Love

### Articles
- LWN.net memory-management coverage (folios, THP, reclaim, MGLRU)
