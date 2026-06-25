# Block I/O Layer & Storage Stack

## Overview

This page traces a read or write from a filesystem down to spinning rust or flash: the **block
layer** — `bio`s and requests, the multi-queue **blk-mq** engine, **I/O schedulers**, and the
**device-mapper** virtual-block stack (LVM, dm-crypt, RAID, multipath). It sits *below*
[Filesystems](filesystems.md) (VFS, the page cache, and journaling — read that for the layer
above) and *above* the block drivers in [Driver Development](driver_development.md) (its *Block
Device Drivers* section). Writeback and the page cache connect to
[Memory Management](memory_management.md); the `/sys/block` tunables are part of the broader
[sysfs](sysfs.md) model, and I/O throttling integrates with the io controller in
[Control Groups (cgroups)](cgroups.md).

The block layer's job is to turn filesystem requests for *logical blocks* into ordered,
merged, scheduled I/O to a device, while hiding whether that device is a SATA disk, an NVMe SSD,
or a stack of virtual devices.

```
   read()/write()  ── syscall ──►  VFS  ──►  page cache (clean/dirty pages)
                                                │  miss / writeback
                                                ▼
   ┌────────────────────────── block layer ──────────────────────────┐
   │  bio (in-memory I/O desc)  ─►  request (merged bios)              │
   │  blk-mq: per-CPU software queues ─► hardware dispatch queues      │
   │  I/O scheduler (none / mq-deadline / BFQ / Kyber)                 │
   └───────────────────────────────┬──────────────────────────────────┘
                                    ▼
                    block driver (NVMe, SCSI/SATA, virtio-blk)
                                    ▼
                              physical device
```

## bios and requests

The fundamental unit is **`struct bio`**: a description of one I/O — a target device + starting
sector, and a list of memory segments (page + offset + length) to read into or write from. A bio
can describe a scatter/gather transfer across non-contiguous pages.

The block layer assembles bios into **`struct request`** objects (the unit a driver consumes),
**merging** adjacent bios that touch contiguous sectors so the device sees fewer, larger I/Os.
Each request carries a direction, sector range, and the segments. Sorting and merging are where
sequential workloads get their throughput.

## blk-mq: the multi-queue layer

Old kernels had a single request queue per device protected by one lock — a scaling disaster on
many-core machines driving million-IOPS NVMe. Since ~4.x, **blk-mq** is the only path:

```
 CPU0 CPU1 CPU2 CPU3        software queues (per-CPU, lockless-ish)
   │    │    │    │
   └────┴──┬─┴────┘
          ▼
   hardware queues  ◄── mapped to the device's real submission queues
   (NVMe: one per CPU; SATA: usually one)
```

Per-CPU **software staging queues** collect requests without cross-CPU contention; they map onto
the device's **hardware queues**. NVMe exposes many hardware queues (often one per CPU), so the
whole path can run lock-free per core — this is what unlocks SSD throughput.

## I/O schedulers

The scheduler decides the *order* requests leave the software queues. Under blk-mq:

| Scheduler     | Best for | Idea |
|---------------|----------|------|
| **none**      | fast NVMe/SSD | no reordering — lowest overhead; the device's own queue handles it |
| **mq-deadline** | general / SATA SSD/HDD | deadline per request to bound latency; light reordering for throughput |
| **BFQ**       | desktop / interactive HDD | proportional fairness per process/cgroup; great latency, higher CPU |
| **kyber**     | fast multi-queue devices | tracks latency, throttles to hit target read/write latencies |

```bash
cat /sys/block/nvme0n1/queue/scheduler      # [none] mq-deadline kyber bfq
echo mq-deadline > /sys/block/sda/queue/scheduler
```

Rule of thumb: **`none`** for NVMe (scheduling just adds latency to a device that reorders
internally), **`mq-deadline`** as a safe default for SATA, **`bfq`** when interactivity/fairness
on slower media matters.

## Useful queue tunables (`/sys/block/<dev>/queue/`)

```bash
nr_requests        # queue depth (in-flight requests)
read_ahead_kb      # sequential read prefetch window
rotational         # 1 = HDD heuristics, 0 = SSD
nomerges           # disable request merging (debug)
max_sectors_kb     # largest single I/O
add_random         # whether the device feeds the entropy pool
```

`rotational` and `read_ahead_kb` materially change sequential performance; large `read_ahead_kb`
helps streaming reads but wastes bandwidth on random workloads.

## Device-mapper & the virtual block stack

**Device-mapper (dm)** stacks virtual block devices on top of real ones — each maps I/O to one or
more underlying devices via a *target*:

- **LVM** (`dm-linear`/`dm-stripe`) — logical volumes, online resize, snapshots, thin provisioning.
- **dm-crypt** — transparent full-disk encryption (LUKS).
- **dm-raid / md** — software RAID 0/1/5/6/10.
- **dm-thin** — thin pools (allocate-on-write).
- **dm-multipath** — one logical device over multiple SAN paths for HA/failover.

```
   filesystem (ext4/xfs)
        ▼
   /dev/mapper/vg-root      ← LVM logical volume (dm)
        ▼
   /dev/mapper/cryptroot    ← dm-crypt (LUKS)
        ▼
   /dev/sda2                ← physical partition
```

These layers each present a block device, so they compose — encryption under LVM under a
filesystem is just three stacked dm/block targets.

## Observing I/O

```bash
lsblk                       # the block topology (partitions, dm, lvm)
iostat -xz 1                # per-device util%, await (latency), IOPS, throughput
cat /proc/diskstats         # raw per-device counters
blktrace -d /dev/nvme0n1 | blkparse   # trace every bio through the layer
biolatency-bpfcc            # eBPF: I/O latency histogram (see ebpf.md)
```

`await` (average request latency) and `%util` from `iostat` are the first numbers to check;
high `await` with low throughput points at the device or scheduler, not the application.

## Where this connects

- [Filesystems](filesystems.md) — VFS, the page cache, and journaling sit directly above this
  layer; dirty-page **writeback** is what feeds bios down here.
- [Driver Development](driver_development.md) — the block-driver side (`blk_mq_ops`, request
  completion) that consumes the requests this layer builds.
- [Memory Management](memory_management.md) — the page cache, dirty-ratio writeback throttling,
  and reclaim all drive block I/O timing.
- [Control Groups (cgroups)](cgroups.md) — the **io** controller throttles/​weights block I/O per
  cgroup (`io.max`, `io.weight`), built on the block layer.
- [sysfs](sysfs.md) — `/sys/block/*/queue` is where scheduler and queue tunables live.

## Pitfalls

- **Leaving an I/O scheduler on fast NVMe.** `mq-deadline`/`bfq` add latency a million-IOPS device
  doesn't need; use `none` and let the device reorder.
- **Reading `iostat %util` as saturation on SSDs.** For multi-queue devices `%util` can read 100%
  while the device is far from saturated (it parallelises); trust `await`/throughput instead.
- **Huge `read_ahead_kb` on random workloads.** Prefetch reads data you never use, stealing
  bandwidth; tune per workload.
- **Forgetting the cache layer.** Slow "disk" symptoms are often page-cache writeback storms
  (dirty ratio) or fsync latency, not the block device — check [Memory Management](memory_management.md).
- **Stacking dm layers without alignment.** Mis-aligned LVM/dm-crypt/RAID stripe boundaries cause
  read-modify-write amplification on SSDs and RAID; align to the physical/erase block size.
- **`barrier`/`fsync` assumptions.** Disabling flushes (`nobarrier`, volatile write cache) boosts
  throughput but risks data loss on power failure — only safe with battery-backed cache.
