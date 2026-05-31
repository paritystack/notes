# Error Handling

## Overview

Error handling is how a program detects, reports, and recovers from things going wrong. The
strategy a language picks deeply shapes its feel and is tightly bound to its
[type system](type_systems.md): [C](c.md) uses return codes and `errno`,
[Go](go.md) returns explicit `error` values, [Rust](rust.md) uses `Result`/`Option` with the
`?` operator, while [Java](java.md), [Python](python.md), [C++](cpp.md),
[JavaScript](javascript.md) and [Kotlin](kotlin.md) lean on exceptions. The
[functional](functional_programming.md) approach of encoding failure in the return type is
increasingly mainstream.

## The three families

```
1. Error codes / sentinels        2. Exceptions                  3. Result/Option types
   ----------------------            -----------------              --------------------
   int fd = open(...);               try { risky(); }              match parse(s) {
   if (fd < 0) {                     catch (IOError e) {             Ok(v)  => use(v),
       handle(errno);                    handle(e);                  Err(e) => handle(e),
   }                                 }                              }

   C, POSIX, old Win32              Java, Python, C++, JS          Rust, Swift, Haskell, FP
```

- **Error codes / sentinels** — the callee returns a special value (`-1`, `NULL`, `false`)
  and sets a side channel like `errno`. Cheap and explicit, but easy to ignore and it
  pollutes the return value.
- **Exceptions** — errors unwind the stack to the nearest handler, separating the happy path
  from error code. Convenient, but control flow becomes invisible at the call site.
- **Result/Option types** — failure is a value in the type (`Result<T, E>`, `Option<T>`).
  The [compiler](compilers.md) forces you to acknowledge it; combinators keep it ergonomic.

## Checked vs unchecked

Within the exception world, errors split into:

- **Unchecked** (runtime) — not part of the signature; can be thrown anywhere (Python, C#,
  most JS). Flexible but invisible.
- **Checked** — declared in the signature and enforced by the compiler (Java's `throws`).
  Forces handling but is widely criticised for boilerplate and leaky abstractions, which is
  why newer languages dropped them in favour of `Result`.

## Propagation idioms

Most failures aren't handled where they occur — they propagate up to a level that can
respond. Languages provide sugar for this:

```rust
fn load(path: &str) -> Result<Config, Error> {
    let text = fs::read_to_string(path)?;  // ? returns early on Err
    let cfg  = parse(&text)?;              // chains cleanly
    Ok(cfg)
}
```

- Rust/Swift: `?`/`try` early-return.
- Go: the explicit `if err != nil { return err }` (often wrapped with `fmt.Errorf("...: %w",
  err)` to add context).
- Exceptions: automatic unwinding to the nearest matching `catch`.

## Errors vs panics vs bugs

A crucial distinction: **recoverable errors** (file missing, bad input — expected, handle
them) vs **unrecoverable bugs** (index out of bounds, broken invariant — `panic`/`abort`,
fail fast). Conflating them leads to either swallowed bugs or over-defensive code that
catches things it can't sensibly recover from.

```
expected & recoverable ──▶ Result / checked error / return code
programmer mistake     ──▶ panic / assert / abort  (fail loud, don't catch)
```

## Cleanup and resource safety

Errors must not leak resources. Mechanisms: `try`/`finally` (Java, Python),
`defer` ([Go](go.md)), RAII/destructors ([C++](cpp.md), [Rust](rust.md) `Drop`), and
context managers (`with` in Python). RAII is the most robust because cleanup is tied to
scope exit regardless of *how* the scope is left.

## Where this connects

- [Type systems](type_systems.md) — `Result`/`Option` are sum types; exhaustiveness via
  [pattern matching](pattern_matching.md).
- [Functional programming](functional_programming.md) — errors-as-values, monadic chaining.
- [Concurrency](concurrency.md) — error propagation across threads/tasks (e.g. joined task
  results, cancellation).
- [Rust](rust.md) / [Go](go.md) / [Java](java.md) pages for language-specific idioms.

## Pitfalls

- **Swallowing errors.** Empty `catch {}`, ignored Go `err`, or discarding a `Result` hides
  failures until they surface far away. At minimum log; usually propagate.
- **Exceptions for control flow.** Using `throw` for ordinary outcomes (e.g. "not found") is
  slow and obscures logic; reserve exceptions for the exceptional.
- **Losing context.** Re-throwing or returning a bare error drops the stack/cause. Wrap with
  context (`%w`, exception chaining, `anyhow`/`thiserror`).
- **Catching too broadly.** `catch (Exception)` / `except:` masks bugs you didn't intend to
  handle, including programmer errors that should crash.
- **Error/panic confusion.** Catching panics to "keep going" can continue past a broken
  invariant into corrupted state.
