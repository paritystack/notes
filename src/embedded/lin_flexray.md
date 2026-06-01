# LIN & FlexRay

## Overview

LIN and FlexRay are the two automotive buses that sit on either side of
[CAN](can.md) in cost and capability. **LIN** (Local Interconnect Network) is the cheap,
slow, single-wire sub-bus used for body electronics — door modules, mirrors, seat
motors, rain sensors — where a full CAN node is overkill. **FlexRay** is the fast,
deterministic, fault-tolerant bus designed for x-by-wire and chassis control, where CAN's
non-determinism and 1 Mbit/s ceiling aren't enough. A modern vehicle is a hierarchy: a
few FlexRay or CAN-FD backbones, many CAN segments, and LIN twigs hanging off CAN gateway
ECUs.

```
        ┌──────────── FlexRay backbone (10 Mbit/s, x2 channels) ────────────┐
        │                                                                    │
   ┌────┴────┐                      ┌─────────┐
   │ Chassis │                      │ Gateway │──── CAN ───┬──────┬──────┐
   │  ECU    │                      │   ECU   │            │      │      │
   └─────────┘                      └────┬────┘         ECU    ECU    ECU
                                         │                            │
                                    LIN  │  (1 master, ≤16 slaves)    │ LIN
                                  ┌──────┴──────┐              ┌──────┴──────┐
                               mirror  window  seat        rain    light    sensor
```

LIN is essentially a [UART](uart.md) at 19.2 kbit/s with a defined frame and schedule on
top; FlexRay is a TDMA bus with synchronized global time. This page compares both against
[CAN](can.md) and explains where each fits.

## LIN: A Scheduled UART Bus

LIN runs over a **single wire** (plus ground and 12 V) at up to **20 kbit/s**, using
standard UART framing (1 start, 8 data, 1 stop) so an ordinary MCU UART can drive it with
a transceiver. It is strictly **single-master, multi-slave**: only the master decides who
talks and when, following a fixed **schedule table**. There is no arbitration — the
schedule guarantees no collisions.

```
LIN FRAME = HEADER (master) + RESPONSE (master OR a slave)

  ┌──────────────────── HEADER (always from master) ──────────┐ ┌── RESPONSE ──┐
  │  BREAK    SYNC(0x55)    PROTECTED ID                       │ │ data[1..8] + │
  │ ≥13 bits  for baud      6 ID bits + 2 parity bits          │ │  checksum    │
  └───────────────────────────────────────────────────────────┘ └──────────────┘
```

The master sends a **break** (a dominant pulse longer than a byte, to grab attention),
the **sync** byte `0x55` (alternating bits let slaves auto-measure the baud rate against
the master's clock — slaves can use cheap on-chip RC oscillators), then the **protected
identifier** (6-bit frame ID + 2 parity bits). Whichever node is configured to provide
that ID's data sends the **response** (1–8 data bytes + checksum); everyone else may
listen. The ID names a *message*, not a node — publish/subscribe, like
[CAN](can.md), but orchestrated rather than contended.

Other LIN features:
- **Schedule table** — the master cycles through a fixed list of frame IDs at fixed times,
  giving fully deterministic, collision-free timing.
- **Sleep/wake** — a sleep command or bus-line wake-up pulse lets slaves drop to µA; key
  for always-on body electronics. Ties into [power management](power_management.md).
- **LIN checksum** — "classic" (data only) vs "enhanced" (includes the protected ID).
- **Transport layer / diagnostics** — multi-frame messages for flashing and UDS.
- **Node configuration** — assign frame IDs at production time (LIN 2.x).

LIN trades CAN's robustness and speed for **cost**: one wire, no quartz crystal in the
slaves, tiny silicon. Use it for low-rate actuators/sensors where ~10–20 ms update
latency is fine.

## FlexRay: Deterministic TDMA

FlexRay targets what CAN can't promise: **guaranteed latency and bandwidth**. It runs at
**10 Mbit/s** per channel over **two channels** (A and B) that can be used for
**redundancy** (same data twice, tolerate a wire fault) or **double bandwidth** (different
data). Access is **TDMA** (time-division): the bus runs a repeating **communication
cycle** divided into precisely timed slots, and every node shares a synchronized **global
clock**.

```
FlexRay COMMUNICATION CYCLE (fixed duration, repeats forever)
┌──────────────── STATIC SEGMENT ───────────────┬──── DYNAMIC ────┬─ SYM ─┬─ NIT ─┐
│ slot1 │ slot2 │ slot3 │ ... │ slotN           │ minislots (FTDMA)│ window│ idle  │
│  ECU  │  ECU  │  ECU  │     │  (fixed owner    │ event-triggered, │ wakeup│ clock │
│  A    │  B    │  C    │     │   per slot)      │ priority by ID   │       │ sync  │
└───────┴───────┴───────┴─────┴──────────────────┴─────────────────┴───────┴───────┘
   guaranteed, time-triggered, deterministic       on-demand, like CAN
```

- **Static segment** — fixed-length slots, each pre-assigned to one node. A node may
  transmit *only* in its slot, so latency is bounded and known at design time. This is the
  deterministic, safety-critical part.
- **Dynamic segment** — divided into **minislots**; a node with data to send claims its
  minislot, otherwise the slot counter advances quickly to the next. This gives CAN-like
  event-driven flexibility for bursty/optional traffic within the leftover time.
- **Symbol window** — wakeup/collision-avoidance symbols.
- **NIT (Network Idle Time)** — every node corrects its local clock here to maintain the
  shared global time; clock synchronization is the heart of FlexRay.

Because slots are time-based, there is **no bit-by-bit arbitration** like CAN — nodes
don't contend, they wait for their turn. The cost is complexity and price: FlexRay needs a
dedicated communication controller, two transceivers, careful network-design tooling, and
bus-guardian logic. It appears in chassis/active-suspension/x-by-wire domains; for most
ECUs **CAN-FD** has absorbed the "need more than classic CAN" niche more cheaply.

## How They Compare to CAN

| | **LIN** | **CAN** / CAN-FD | **FlexRay** |
|---|---|---|---|
| Speed | ≤20 kbit/s | 1 Mbit/s (CAN-FD ~5–8 in data) | 10 Mbit/s ×2 |
| Wires | 1 (single-ended) | 2 (differential) | 2 or 4 (1–2 differential pairs) |
| Topology | single master, ≤16 slaves | multi-master | multi-master TDMA |
| Access | master schedule (no contention) | bitwise [arbitration](can.md) | time slots (TDMA) |
| Determinism | high (scheduled) | priority-based, not time-guaranteed | hard real-time |
| Redundancy | none | none (single bus) | dual channel |
| Cost | very low (RC-clocked slaves) | low/medium | high |
| Typical use | mirrors, windows, seats, sensors | powertrain, body, general | chassis, x-by-wire, suspension |

The mental model: **LIN** for the cheapest peripherals, **CAN/CAN-FD** for the workhorse
bulk of the vehicle, **FlexRay** when you must guarantee timing and tolerate a fault.

## Where this connects

- [CAN](can.md) — the reference automotive bus; LIN sits below it (sub-bus off a gateway ECU), FlexRay above it (deterministic backbone).
- [UART](uart.md) — LIN is UART framing (break + sync + PID + data) plus a schedule; many MCUs have a dedicated "LIN mode" in their UART/USART.
- [Modbus & RS-485](modbus.md) — another master-driven, polled multidrop bus, but for industrial rather than automotive use.
- [Power Management](power_management.md) — LIN sleep/wake keeps always-on body electronics in µA standby.
- [Signal Integrity](signal_integrity.md) — single-wire LIN and the FlexRay differential pairs each have their own termination and EMC constraints.

## Pitfalls

1. **LIN slave clock drift.** Slaves auto-baud off the `0x55` sync byte; a slave whose RC
   oscillator drifts too far between syncs mis-samples the data. Honor the per-frame resync.
2. **Wrong LIN checksum model.** Mixing "classic" (data-only) and "enhanced" (includes PID)
   checksums between master and slave silently corrupts every frame. Match the LIN version.
3. **Treating LIN like a multi-master bus.** Only the master may initiate; a slave that
   talks unbidden collides. All timing comes from the master's schedule table.
4. **Forgetting the LIN break is longer than a byte.** A too-short break isn't recognized as
   a frame start; the dominant break must exceed a normal character time.
5. **FlexRay node not clock-synchronized.** A node that fails to integrate into global time
   can't claim its static slot; startup/sync (cold-start nodes, NIT correction) is fragile
   and tooling-driven — don't hand-tune slot timing.
6. **Mis-using FlexRay channels.** Dual-channel can mean redundancy *or* double bandwidth,
   not both; choosing bandwidth forfeits the single-wire-fault tolerance.
7. **Reaching for FlexRay when CAN-FD suffices.** FlexRay's cost/complexity is only worth it
   for hard-real-time, fault-tolerant domains; most "faster CAN" needs are met by CAN-FD.

## See Also

- [CAN](can.md) — the bus both of these are defined relative to
- [UART](uart.md) — the framing layer LIN is built on
- [Signal Integrity](signal_integrity.md) — termination and EMC for automotive wiring
