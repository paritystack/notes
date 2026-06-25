# Regular Expressions

## Overview

A **regular expression** is a compact pattern language for matching text — a tiny declarative
DSL that compiles to a state machine. Regexes show up everywhere in this book: they are the
front half of a [compiler](compilers.md)'s lexer (tokens are usually defined as regexes),
the workhorse of [Bash](bash.md) text processing (`grep`, `sed`, `awk`), and a standard
library in [Python](python.md), [JavaScript](javascript.md), and most other languages. The
theory is small and clean; the practice has one giant trap (catastrophic backtracking) that is
worth understanding before you rely on regex in production.

## Syntax in one screen

A pattern is literal characters plus metacharacters that mean "a class of characters",
"a position", or "a repetition".

```
literals     abc            matches the text "abc"
classes      [a-z] \d \w .  a set of chars; \d=digit \w=word .=any
anchors      ^ $ \b         start / end of line, word boundary (zero-width)
quantifiers  * + ? {2,5}    0+, 1+, 0 or 1, between 2 and 5
groups       (ab)+          group for repetition / capture
alternation  cat|dog        either side
backref      (\w)\1         matches a repeated capture  (NOT regular!)
```

Anchors and `\b` match *positions*, not characters — a frequent source of confusion. Groups do
double duty: they scope quantifiers and, by default, **capture** the matched substring for
later extraction; `(?:...)` makes a non-capturing group.

## Engines: NFA/DFA vs backtracking

How a regex runs depends on the engine, and the two families have opposite performance
profiles.

```
DFA / NFA-simulation (RE2, grep, awk, lexers)
  builds an automaton; scans input ONCE
  guaranteed O(n) in input length
  no backreferences / lookaround

BACKTRACKING (PCRE, Perl, Python re, JS, Java, .NET)
  tries a path, rewinds on failure, tries the next
  supports backrefs, lookahead/behind, capture groups
  worst case EXPONENTIAL — the ReDoS trap
```

The DFA family comes from theory: a "regular" language is exactly what a finite automaton can
recognize, so it runs in linear time but can't do backreferences (which make the language
non-regular). Most *programming-language* regex libraries are backtracking engines that trade
the linear-time guarantee for those richer features. Google's **RE2** and Rust's `regex` are
notable linear-time engines that deliberately drop backreferences to stay safe.

## Greedy, lazy, and catastrophic backtracking

Quantifiers are **greedy** by default — they grab as much as possible, then give characters
back if the rest of the pattern fails. Appending `?` makes them **lazy** (grab as little as
possible). On a backtracking engine, nested quantifiers over overlapping alternatives create an
explosion of paths to try:

```
pattern  (a+)+$        input  "aaaa...aaaaX"   (no final match)
                       the engine tries every way to split the a's
                       → time blows up exponentially with input length
```

This is **catastrophic backtracking**, and when the pattern touches untrusted input it becomes
**ReDoS** — a denial-of-service where a short string pins a CPU core for seconds or minutes.
The defenses: avoid nested/ambiguous quantifiers, anchor patterns, set a match timeout, or use
a linear-time engine (RE2) for untrusted input.

## Where this connects

- [Compilers](compilers.md) — lexers specify tokens as regexes and compile them to a DFA; this
  is regex theory's original home.
- [Bash](bash.md) — `grep`/`sed`/`awk` and shell text munging are regex-driven (note: shell
  *globbing* `*.txt` is a different, simpler pattern language).
- [Python](python.md) / [JavaScript](javascript.md) — standard `re`/`RegExp` modules are
  backtracking engines, so the ReDoS caveats apply.
- [Serialization](serialization.md) — regex is the wrong tool for parsing nested formats
  (JSON/HTML); reach for a real parser instead.

## Pitfalls

- **ReDoS / catastrophic backtracking.** Nested quantifiers (`(a+)+`, `(.*)*`) on untrusted
  input can hang a thread. Anchor, simplify, time out, or use RE2.
- **Parsing nested/recursive formats.** Regex can't match balanced brackets or HTML in general
  — they aren't regular. Use a parser; the famous "don't parse HTML with regex" rule.
- **Unescaped metacharacters.** A literal `.` or `(` from user input changes the pattern; always
  escape interpolated strings (`re.escape`).
- **Dialect differences.** POSIX BRE/ERE, PCRE, JS, and `.go` differ on lookbehind, `\d`
  meaning under Unicode, and escaping — a pattern isn't portable by default.
- **Greedy-by-default surprises.** `<.*>` over `<a><b>` matches the whole thing, not `<a>`; use
  lazy `<.*?>` or a negated class `<[^>]*>`.
- **`^`/`$` and multiline.** Without a multiline flag these anchor the whole string, not each
  line — a common off-by-one in log processing.
