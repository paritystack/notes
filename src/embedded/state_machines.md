# State Machines & Event-Driven Firmware

## Overview

A finite state machine (FSM) is the workhorse structuring pattern of embedded firmware:
the system is always in exactly one of a small set of **states**, and **events** drive
transitions between them. Almost any non-trivial device — a [BLE](ble.md) connection, a
[bootloader](bootloaders.md) handshake, a charger, a vending machine, a UART command
parser — is most honestly described as "in state X, when event E happens, do action A
and go to state Y." Writing it as an explicit FSM beats a tangle of `if` flags because
the legal states and transitions become enumerable, reviewable, and testable.

The companion idea is **event-driven** structure: instead of a "super-loop" that polls
everything every pass, [interrupts](interrupts.md) and tasks post events into a queue,
and a single dispatcher feeds them to the state machine one at a time. This page covers
both the FSM encodings and the event-loop scaffolding that surrounds them, and pairs
closely with [Ring Buffers & Lock-Free Concurrency](ring_buffers.md) (the queue) and
[MISRA C & Defensive Firmware](coding_standards.md) (keeping it analyzable).

```
        event E1                event E2
  ┌────────────┐   E1     ┌────────────┐   E2     ┌────────────┐
  │   IDLE     │ ───────► │  RUNNING   │ ───────► │   DONE     │
  │            │ ◄─────── │            │          │            │
  └────────────┘  E_RESET └────────────┘          └────────────┘
        ▲                                                │
        └────────────────────────────────────────────────┘
                           E_RESET
```

## Super-Loop vs Event-Driven

```
SUPER-LOOP (polled)              EVENT-DRIVEN (queue + dispatch)
  while (1) {                      while (1) {
    poll_buttons();                  evt = queue_pop();   // blocks/sleeps
    poll_uart();                     fsm_dispatch(&fsm, evt);
    poll_sensors();                }
    update_state();                ISR: queue_push(EVT_RX);  // producers
    update_outputs();              ISR: queue_push(EVT_TICK);
  }
```

The super-loop is simplest and fine for tiny systems, but every subsystem must be polled
fast enough for the *worst* case, and shared flags multiply. The event-driven form lets
the CPU sleep ([power management](power_management.md)) until an [interrupt](interrupts.md)
posts an event, decouples producers from the consumer, and gives the state machine a
single, serialized stream of inputs — which is exactly what makes run-to-completion
(below) safe.

## Encodings

Three common ways to write the same machine, trading readability for table-driven rigor:

### 1. Switch-based

```c
typedef enum { S_IDLE, S_RUNNING, S_DONE } state_t;
static state_t state = S_IDLE;

void dispatch(event_t e) {
    switch (state) {
    case S_IDLE:
        if (e == EVT_START) { motor_on(); state = S_RUNNING; }
        break;
    case S_RUNNING:
        if (e == EVT_STOP)  { motor_off(); state = S_DONE; }
        break;
    case S_DONE:
        if (e == EVT_RESET) { state = S_IDLE; }
        break;
    }
}
```

Readable for a handful of states; degrades into deeply nested `switch`/`if` as it grows.

### 2. Transition table

```c
typedef struct {
    state_t  from;
    event_t  on;
    void   (*action)(void);
    state_t  to;
} transition_t;

static const transition_t table[] = {
    { S_IDLE,    EVT_START, motor_on,  S_RUNNING },
    { S_RUNNING, EVT_STOP,  motor_off, S_DONE    },
    { S_DONE,    EVT_RESET, NULL,      S_IDLE    },
};
```

The machine becomes *data*: a `const` table living in [flash](linker_scripts.md), walked
by one generic dispatcher. Easy to audit, easy to diff, and the whole transition set is
visible at a glance — the form static-analysis and reviewers like best.

### 3. Function-pointer (state-as-handler)

```c
typedef void (*state_fn)(event_t e);   // current state IS a function pointer
static state_fn current;

void s_running(event_t e) {
    if (e == EVT_STOP) { motor_off(); current = s_done; }
}
```

The "state" is a pointer to the handler for that state; transitioning means reassigning
the pointer. Scales well and supports entry/exit hooks, at the cost of indirection that
some [MISRA](coding_standards.md) profiles frown on.

## Run-to-Completion

The golden rule of well-behaved FSMs: **each event is processed to completion before the
next is dequeued.** No transition handler blocks, waits, or spins; it does a bounded
chunk of work and returns. This single discipline removes most concurrency hazards —
because only one event is ever "in flight," the state variable is never seen half-updated,
and you reason about the machine as if it were single-threaded even though
[interrupts](interrupts.md) are posting events underneath.

```
queue: [E1][E2][E3]
          │
          ▼  pop E1 ─► handler runs fully ─► return ─► pop E2 ─► ...
        (never re-enter the FSM while a handler is running)
```

Anything that would block (wait for a conversion, a timeout, a [DMA](dma.md) completion)
is turned into "arm a timer / start the transfer, return now; resume on the completion
event."

## Hierarchical State Machines (Statecharts)

Flat FSMs suffer **state explosion**: if "battery low" or "fault" must be handled
identically in ten states, you copy the same transition ten times. Harel **statecharts**
add nested (hierarchical) states — a child state inherits its parent's transitions, so a
single `Operational → Fault` edge on the parent covers every substate.

```
┌─ Operational ───────────────────────────────┐
│   ┌─ Idle ─┐   ┌─ Running ─┐   ┌─ Paused ─┐  │   any-substate
│   │        │──►│           │──►│          │  │ ─────────────► [ Fault ]
│   └────────┘   └───────────┘   └──────────┘  │   on EVT_FAULT
└──────────────────────────────────────────────┘
```

Statecharts also formalize **entry/exit actions** (run on crossing a boundary, e.g.
"enable motor on entry, disable on exit"), **guards** (a transition fires only if a
condition holds), and **orthogonal regions** (concurrent sub-machines). Frameworks like
**QP/QF** (Quantum Platform) and Zephyr's **State Machine Framework (SMF)** implement this
directly; for many MCUs a hand-rolled hierarchical function-pointer machine is enough.

## Events & the Queue

Events are small tagged values; richer ones carry a payload:

```c
typedef struct {
    uint8_t  sig;        // event signal/type
    uint16_t param;      // optional payload (e.g. byte received, ADC value)
} event_t;
```

Producers ([ISRs](interrupts.md), timer callbacks, other tasks) `push` events into a
[ring buffer](ring_buffers.md); the dispatcher `pop`s and runs them. Keep ISRs short —
they post an event and return; the heavy logic happens in the FSM at task level. A
periodic timer event (`EVT_TICK`) drives timeouts so the machine can leave a state that
no external event would otherwise exit (e.g. "no response within 5 s → go to Fault").

## Where this connects

- [Ring Buffers & Lock-Free Concurrency](ring_buffers.md) — the event queue feeding the dispatcher; the FSM is its single consumer.
- [Interrupts](interrupts.md) — ISRs are event *producers*; keep them short and post events for the FSM to handle at task level.
- [BLE](ble.md) — connection/GATT state is a textbook hierarchical FSM (advertising → connecting → connected → bonded).
- [Bootloaders](bootloaders.md) / [OTA Updates](ota_updates.md) — the update handshake (erase → receive → verify → swap) is a small FSM.
- [MISRA C & Defensive Firmware](coding_standards.md) — table-driven transitions are easy to review and keep the `default:`/illegal-event path explicit.
- [Power Management](power_management.md) — an event loop lets the core sleep between events instead of busy-polling.

## Pitfalls

1. **Blocking inside a handler.** A `delay()` or busy-wait in a transition stalls every
   other event. Convert waits into timer/completion events (run-to-completion).
2. **Implicit state in scattered flags.** Booleans like `isRunning`, `gotData`, `error`
   recreate an FSM badly — illegal combinations become reachable. Make the state explicit.
3. **No handling for unexpected events.** An event arriving in a state that doesn't expect
   it should be explicitly ignored or logged, not silently fall through; add a `default:`.
4. **State variable written from an ISR and a task.** Only the dispatcher should mutate
   state; ISRs post events. Otherwise you reintroduce the races run-to-completion removes.
5. **Forgetting entry/exit symmetry.** If entry enables a peripheral, *every* exit path
   (including to Fault) must disable it — hierarchical exit actions make this automatic.
6. **Unbounded event queue.** A fast producer outrunning the FSM overflows the
   [ring buffer](ring_buffers.md); size it for the worst burst and decide drop-vs-block policy.
7. **Timeouts via counting loop iterations.** Loop counts aren't time. Drive timeouts from
   a real [timer](timers.md) tick event so they're independent of CPU load.

## See Also

- [Ring Buffers & Lock-Free Concurrency](ring_buffers.md) — the queue behind the event loop
- [Interrupts](interrupts.md) — where events are born
- [MISRA C & Defensive Firmware](coding_standards.md) — keeping the machine analyzable
