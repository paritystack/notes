# Asynchronous Programming

## Overview

Asynchronous programming lets a program start an operation and continue doing other work
before that operation finishes, instead of blocking a thread while it waits. Where
[concurrency](concurrency.md) covers threads, locks and parallelism in general, this page is
about the *language-level model* of non-blocking work: callbacks, futures/promises,
`async`/`await`, and coroutines. It's central to [JavaScript](javascript.md) (single-threaded
event loop), [Python](python.md) (`asyncio`), [Rust](rust.md) (`async`/`.await` + executors),
[Kotlin](kotlin.md) (coroutines), [Go](go.md) (goroutines), and [C#](java.md)-style
runtimes.

## Why async exists: I/O-bound vs CPU-bound

```
Blocking model                      Async model (one thread)
--------------                      ------------------------
req A ─ wait disk ─ done            req A ─▶ (await disk)  ┐
req B ─ wait disk ─ done                 req B ─▶ (await)  ├─ one thread, interleaved
req C ─ wait disk ─ done                 req C ─▶ (await)  ┘
(thread idle during each wait)      (thread does other work during waits)
```

Async shines for **I/O-bound** workloads (network, disk, DB) where threads would otherwise
sit idle. It does **not** speed up **CPU-bound** work — for that you need real parallelism
(threads/processes, see [concurrency](concurrency.md)). A common architecture mixes both:
async for I/O, a thread pool for CPU-heavy tasks.

## The evolution of the model

1. **Callbacks** — pass a function to run on completion. Simple, but nesting leads to
   "callback hell" and error handling fragments.
2. **Futures / Promises** — an object representing a value that will exist later, with
   `.then`/combinators. Composable and flattenable.
3. **`async`/`await`** — syntactic sugar over futures that makes async code *read* like
   sequential code while staying non-blocking.

```javascript
// Callback        →        Promise          →        async/await
getUser(id, (u) =>          getUser(id)               const u = await getUser(id);
  getPosts(u, (p) =>          .then(getPosts)          const p = await getPosts(u);
    render(p)));              .then(render);           render(p);
```

## The event loop

Single-threaded async runtimes (JS, Python `asyncio`) are driven by an **event loop**: a
queue of ready tasks. An `await` *suspends* the current task and yields control back to the
loop, which runs other ready tasks until the awaited operation signals completion.

```
        ┌──────────────────────────────┐
        │          Event Loop          │
        │  pick ready task ─▶ run until │
        │  it awaits ─▶ park it ─▶ loop │
        └──────────────────────────────┘
   timers ▲   I/O completions ▲   microtasks ▲
```

Because there is one thread, **a blocking call (or a long CPU loop) freezes everything** —
the cardinal sin of async code.

## Coroutines: stackful vs stackless

`async` functions are **coroutines** — functions that can suspend and resume. Two flavours:

- **Stackless** (Rust, JS, Python, C#) — compiled into a state machine; cheap, but the
  `async` "colour" propagates through signatures (callers must also be async).
- **Stackful** (Go goroutines, Kotlin can do both) — each has its own resizable stack;
  ordinary-looking code suspends transparently, no function colouring, at a slightly higher
  memory cost.

This is the root of the ["function colouring"](functional_programming.md) problem: in
stackless systems, `async` and sync functions don't compose freely.

## Cancellation and structured concurrency

Mature async needs a way to **cancel** in-flight work (timeouts, user aborts) and to ensure
spawned tasks don't outlive their scope — **structured concurrency** (Kotlin
`coroutineScope`, Python `TaskGroup`, Trio nurseries). Without it you get orphaned tasks and
leaked resources. Cancellation typically surfaces as a special error that propagates like
any other (see [error handling](error_handling.md)).

## Where this connects

- [Concurrency](concurrency.md) — threads, parallelism, and the synchronization primitives
  async builds on.
- [Error handling](error_handling.md) — propagating failures/cancellation across `await`
  points.
- [Iterators & generators](iterators_generators.md) — async iterators/streams; generators
  and coroutines are close cousins.
- [JavaScript](javascript.md) / [Python](python.md) / [Rust](rust.md) / [Kotlin](kotlin.md)
  pages for concrete runtimes.
- [Closures & lexical scope](closures.md) — callbacks and coroutines close over the variables
  they resume with.

## Pitfalls

- **Blocking the event loop.** A synchronous DB call, `time.sleep`, or tight CPU loop inside
  async code stalls *all* tasks. Offload to a thread/process pool.
- **Fire-and-forget tasks.** Spawning without awaiting (or tracking) swallows exceptions and
  leaks work; use structured concurrency.
- **`await` in a loop, serially.** Awaiting one-by-one defeats concurrency; gather/`join`
  independent tasks to run them together.
- **Mixing sync and async (colouring).** Calling async from sync requires a runtime/bridge;
  blocking on a future from inside the loop can deadlock.
- **Assuming async = faster.** For CPU-bound work it adds overhead with no parallelism gain.
- **Forgotten cancellation/timeouts.** Network calls without timeouts hang tasks forever.
