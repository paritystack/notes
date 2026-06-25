# Closures & Lexical Scope

## Overview

**Scope** is the set of rules deciding which binding a name refers to; a **closure** is a
function bundled together with the bindings from the scope where it was *defined*, so it can
keep using them after that scope has returned. The two ideas are inseparable — a closure only
makes sense under *lexical* scope. They underpin much of the rest of this section:
[functional programming](functional_programming.md) is built on closures (partial application,
higher-order functions), [iterators & generators](iterators_generators.md) capture state in a
closure between yields, [async programming](async_programming.md) callbacks and coroutines
close over the variables they resume with, and [metaprogramming](metaprogramming.md) uses
closures to build decorators and DSLs.

## Lexical vs dynamic scope

**Lexical** (static) scope resolves a free variable by looking *outward through the source
text* where the function was written. **Dynamic** scope resolves it by looking *up the call
stack* at run time. Almost every modern language is lexically scoped; dynamic scope survives
mainly in shells ([Bash](bash.md) variables), Emacs Lisp, and constructs like exception
handlers.

```
x = "global"
def f():  return x          # which x?
def g():
    x = "local"
    return f()

lexical scope  → "global"   # f sees the x where f was DEFINED
dynamic scope  → "local"    # f sees the x where f was CALLED
```

Lexical scope is what lets you read a function in isolation and know what its names mean — the
answer is on the page, not in whoever happens to call it.

## Closures capture their environment

When a function is defined inside another, the inner function keeps a reference to the outer
function's local bindings — the **environment**. That bundle (code + captured environment) is
the closure, and it outlives the enclosing call.

```
def counter():
    n = 0
    def inc():
        n += 1        # `n` is a free variable, captured from counter()
        return n
    return inc        # counter() has returned, but `n` lives on

c = counter(); c() -> 1; c() -> 2     # the closure keeps its own n
```

Each call to `counter()` makes a *fresh* environment, so two counters don't share `n`. This is
the mechanism behind private state, callbacks that remember context, and currying.

## Capture by reference vs by value

The sharp edge is *what* gets captured: the variable itself (by reference) or a snapshot of its
value. Most languages capture the **binding by reference** — the closure sees later mutations.

```
by reference (Python, JS, Go, Java):  closure sees the variable's CURRENT value
by value     (C++ [=], opt-in):       closure copies the value at creation time
```

C++ makes it explicit (`[&]` reference, `[=]` value); [Rust](rust.md) infers it but lets you
force a copy with `move`. In garbage-collected languages, capturing by reference is why a
closure can keep a large object alive far longer than expected (a memory leak — see
[memory management](memory_management.md)).

## The closure-over-loop pitfall

The most famous bug: making closures inside a loop that all capture the *same* loop variable.

```
funcs = [lambda: i for i in range(3)]
[f() for f in funcs]   ->  [2, 2, 2]      # all share one `i`, final value 2

# fix: bind a fresh variable per iteration
funcs = [lambda i=i: i for i in range(3)] ->  [0, 1, 2]
```

Old JavaScript `var` had exactly this (fixed by block-scoped `let`, which rebinds per
iteration); [Go](go.md) fixed its loop-variable capture in 1.22. The cure is always the same:
give each iteration its own binding.

## Implementation & upvalues

A captured variable is called an **upvalue** (Lua's term). If a closure can outlive the stack
frame it captured, the compiler can't keep that variable on the stack — it must **escape** to
the heap so it survives. Compilers run *escape analysis* to decide: variables that never leak
into an escaping closure stay cheap on the stack; the rest are boxed. This is why closures
aren't free, and why hot loops sometimes avoid them.

## Where this connects

- [Functional programming](functional_programming.md) — closures are the substrate for
  higher-order functions, currying, and partial application.
- [Iterators & generators](iterators_generators.md) — generator state lives in a closure-like
  captured frame resumed across yields.
- [Async programming](async_programming.md) — callbacks/coroutines close over the variables
  they resume with; captured references can outlive their logical lifetime.
- [Metaprogramming](metaprogramming.md) — decorators and DSL builders are closures over the
  wrapped target.
- [Memory management](memory_management.md) — escape analysis decides stack vs heap; captured
  references extend object lifetimes.

## Pitfalls

- **Loop-variable capture.** Closures created in a loop share one mutable binding unless each
  iteration gets a fresh one (`let`, a per-iteration copy, or a default-arg trick).
- **Accidental lifetime extension.** A closure capturing a big object (or `self`) by reference
  keeps it alive — a common leak in event handlers and caches; capture only what you need.
- **Reference vs value confusion.** Assuming a snapshot when the language captured the live
  binding (or vice versa) gives stale or surprising reads.
- **Dynamic-scope surprises.** Relying on shell/`dynamic` variable inheritance makes functions
  behave differently depending on the caller — hard to reason about and test.
- **Mutating captured state from multiple threads.** A closure's captured variables are shared
  mutable state; concurrent calls need the same synchronization as any other shared data.
