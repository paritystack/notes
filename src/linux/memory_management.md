# Linux Memory Management

> **Domain:** Linux Kernel, Systems Programming, OS Theory
> **Key Concepts:** Virtual Memory, MMU, Paging, Page Tables, TLB, Slab Allocator

Memory Management is the subsystem responsible for managing the computer's primary memory. It provides processes with the illusion of a large, contiguous private memory space (Virtual Memory) while multiplexing the limited physical RAM.

---

## 1. Virtual Memory

Every process on Linux thinks it starts at address `0x000...` and owns the entire address space (e.g., 48-bit on x86_64).
*   **Benefits:**
    1.  **Isolation:** Process A cannot touch Process B's memory.
    2.  **Security:** User space cannot touch Kernel space.
    3.  **Efficiency:** RAM is only allocated when strictly needed (Lazy Allocation).

---

## 2. Paging and the MMU

The CPU does not read Physical Addresses. It reads Virtual Addresses. The **Memory Management Unit (MMU)** translates them.

*   **The Page:** Memory is divided into 4KB chunks called Pages (Virtual) and Page Frames (Physical).
*   **The Page Table:** A data structure in RAM that maps Virtual Page Number (VPN) -> Physical Frame Number (PFN).

### 2.1. Multi-Level Page Tables (x86_64)
A single array mapping all 48 bits would be massive. Linux uses a 4-level (or 5-level) tree:
1.  **PGD (Page Global Directory)**
2.  **PUD (Page Upper Directory)**
3.  **PMD (Page Middle Directory)**
4.  **PT (Page Table)** -> Points to Physical Frame.

### 2.2. TLB (Translation Lookaside Buffer)
Walking the 4-level tree for every memory access is slow. The **TLB** is a specialized CPU cache that stores recent translations.
*   *Hit:* 1 CPU cycle.
*   *Miss:* 100+ CPU cycles (Page Walk).

---

## 3. Kernel Memory Allocation

The kernel needs memory for itself (process descriptors, inodes).

### 3.1. Page Allocator (Buddy System)
Manages raw physical pages.
*   **Concept:** Splits memory into blocks of order $0..MAX\_ORDER$ ($2^0$ pages, $2^1$ pages...).
*   **Allocation:** If you need 4 pages and only an 8-page block exists, split the 8 into two 4s. Give one, keep one.
*   **Free:** When two adjacent "buddies" are free, merge them back into a larger block.

### 3.2. Slab Allocator (SLUB)
The Buddy System is too coarse (min 4KB). The Kernel often needs tiny objects (e.g., a 64-byte file descriptor).
*   **Concept:** Carves up a page into fixed-size chunks ("slabs").
*   **Caches:** Dedicated caches for common objects (`task_struct`, `inode_cache`).
*   **kmalloc:** Uses generic size caches (kmalloc-32, kmalloc-64).

---

## 4. Page Faults

What happens when a process accesses a Virtual Address that has no physical mapping? **Page Fault.**

1.  **Major Fault (Disk I/O):**
    *   The data is swapped out or mapped to a file.
    *   Kernel puts process to sleep, fetches data from disk to RAM, updates Page Table, resumes process.
2.  **Minor Fault (Soft):**
    *   The page is in RAM (maybe shared by another process, or just allocated but not mapped).
    *   Kernel updates Page Table, resumes. (Fast).
3.  **Segfault (Invalid):**
    *   Accessing unallocated memory or writing to read-only memory.
    *   Kernel sends `SIGSEGV` signal.

---

## 5. Advanced Concepts

*   **OOM Killer (Out of Memory):** When RAM + Swap is full, the Kernel selects a victim (based on `oom_score`) and kills it to free memory.
*   **Huge Pages:** Using 2MB or 1GB pages instead of 4KB. Reduces TLB misses. Essential for Database servers (Postgres, Oracle).
*   **Overcommit:** Linux allows allocating more memory than physically exists (`malloc` succeeds, but RAM isn't consumed until write).

---

## 6. Tools

*   `free -h`: Show RAM usage.
*   `vmstat 1`: Monitor swap and paging activity.
*   `cat /proc/meminfo`: Detailed kernel memory stats.
*   `slabtop`: View active slab caches.
