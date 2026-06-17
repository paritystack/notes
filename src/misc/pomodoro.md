# The Pomodoro Technique

## Overview

The **Pomodoro Technique** is a time-management method created by Francesco Cirillo in the
late 1980s. The idea is disarmingly simple: work in fixed, undistracted intervals of ~25
minutes — each called a *pomodoro* — separated by short breaks. The name comes from the
tomato-shaped (*pomodoro* is Italian for "tomato") kitchen timer Cirillo used as a student.

Where [GTD](gtd.md) answers *what* to work on and keeps your commitments in a trusted
system, Pomodoro answers *how* to actually sit down and do the work: it fights
procrastination, protects focus, and makes effort measurable. The two pair naturally — pull
a concrete next action off your GTD list, then run pomodoros on it.

```
The core loop:
  pick ONE task  →  timer 25 min, no interruptions  →  ✓ mark a pomodoro
                 →  short break 5 min  →  repeat
                 →  every 4 pomodoros, take a long break 15–30 min
```

The deeper principle is that **the pomodoro is indivisible**: you protect the 25 minutes
absolutely. A pomodoro that gets interrupted doesn't count — it's voided. This all-or-nothing
rule is what trains focus and reveals how often you're really being pulled away.

## The Six Steps

Cirillo's original method is a short, ordered process:

```
  1. CHOOSE      pick a single task to work on
  2. SET         start a 25-minute timer
  3. WORK        work only on that task until the timer rings
  4. MARK        put a ✓ on your sheet — one pomodoro done
  5. BREAK       short break (5 min): stand up, look away, rest
  6. LONG BREAK  after every 4 pomodoros, take 15–30 min off
```

The unit of work is the pomodoro, not the task. You don't measure "I worked on the report";
you measure "the report took 5 pomodoros". Over time those marks become real data about how
long things actually take and where your day goes.

## Default Timings

```
┌──────────────────┬──────────┬──────────────────────────────────────────────┐
│ Phase            │ Duration │ Notes                                         │
├──────────────────┼──────────┼──────────────────────────────────────────────┤
│ Pomodoro (work)  │  25 min  │ One task, zero interruptions                 │
│ Short break      │   5 min  │ Rest the mind — not more screen/work          │
│ Long break       │ 15–30 min│ After every 4 pomodoros (one "set")          │
└──────────────────┴──────────┴──────────────────────────────────────────────┘
```

These are defaults, not dogma. Some people prefer 50/10 for deep work that needs longer
ramp-up; others use 15/3 when focus is fragile. The ratio and the protected-interval
discipline matter more than the exact numbers — start at 25/5 and adjust to your work.

## Handling Interruptions

Interruptions are the technique's real subject. Cirillo splits them in two and prescribes a
tactic for each:

```
INTERNAL  (you interrupt yourself: an idea, an urge to check something)
   → don't act on it. Jot it on a capture list and keep working.
   → deal with it later (often it turns out not to matter).

EXTERNAL  (someone/something interrupts you)
   → the "inform – negotiate – schedule – call back" protocol:
       Inform     "I'm in the middle of something"
       Negotiate  agree a time to handle it
       Schedule   note it on your list
       Call back  follow up when your pomodoro ends
```

The "inform–negotiate–schedule–call back" strategy lets you protect the current pomodoro
without dropping the request. The capture list for internal interruptions is, in effect, a
mini GTD inbox — see [GTD](gtd.md) for processing it.

```
The rule:  if a pomodoro is broken, it does NOT count.
           you can't pause a pomodoro — you can only void it and start over.
```

## Planning Your Day

A light planning layer turns pomodoros into a daily rhythm:

```
  To Do Today      ──▶  Activity Inventory  ──▶  Records
  (today's tasks,       (everything you            (✓ marks + how many
   estimated in          might do, the              pomodoros each task
   pomodoros)            backlog)                   really took)
```

- **Estimate** each task in pomodoros before starting. Anything bigger than ~5–7 pomodoros
  should be broken down; anything under 1 should be batched with similar small tasks.
- **Compare** your estimate to the actual marks. The gap is the most useful feedback the
  technique produces — most people chronically underestimate.
- Cap the day at the number of pomodoros you can realistically protect; an honest count
  beats an aspirational to-do list.

## Why It Works

```
  Beats procrastination  "just one pomodoro" lowers the barrier to starting
  Protects focus         a closed 25-min window keeps attention single-threaded
  Reduces burnout        enforced breaks prevent grinding past the point of value
  Makes work visible     ✓ marks turn vague effort into countable units
  Improves estimates     estimate-vs-actual data sharpens future planning
  Tames interruptions    a protocol replaces reflexive context-switching
```

The cognitive basis: frequent breaks aid retention and stave off mental fatigue, while a
short, fixed deadline harnesses a mild urgency that crowds out distraction — without the
exhaustion of an open-ended marathon session.

## Tools & Implementation

Pomodoro is deliberately low-tech — the canonical tool is a literal kitchen timer and a
sheet of paper. Digital options just add convenience:

```
Analog     a kitchen timer + a paper sheet of ✓ marks (the original)
Mobile     Forest, Be Focused, Focus To-Do, Pomofocus
Desktop    Pomotroodle/Pomotroid, Flowtime, system menubar timers
Dev/CLI    a shell `sleep 1500 && notify-send` loop, or org-mode's `org-pomodoro`
```

Whatever you use, keep the timer *visible* and the break *real* (step away from the screen).
The point is the discipline of the closed interval, not the app.

## Where this connects

- [Getting Things Done (GTD)](gtd.md) — GTD decides *what* to do and keeps the trusted
  lists; Pomodoro is the *Engage* mechanism that gets the chosen action done. The Pomodoro
  capture list for internal interruptions mirrors GTD's inbox.
- **Timeboxing** — Pomodoro is a specific, fixed-size form of the broader timeboxing idea
  (assigning a bounded slot to an activity).
- **Eisenhower matrix** — a prioritization lens for *which* task earns your next pomodoro.
- **Deep work** (Cal Newport) — Pomodoro is one practical scaffold for entering and
  sustaining distraction-free concentration, though deep-work sessions often run longer.
- **Flow** — short rigid intervals can occasionally cut a session off mid-flow; see Pitfalls.

## Pitfalls

- **Treating the timings as sacred** — rigidly stopping a productive flow state at exactly 25
  minutes can do more harm than good. The interval is a tool; adapt it.
- **Fake breaks** — spending the 5 minutes on email or social feeds isn't a break for your
  attention. Stand up, look away, move.
- **Counting interrupted pomodoros** — if you let broken intervals "count anyway", you lose
  the focus-training and the honest data. Void them.
- **Over-fragmenting deep work** — tasks needing long ramp-up (writing, debugging, design)
  may suffer under 25-minute chops; lengthen the interval rather than abandoning the method.
- **Estimating in pomodoros but never reviewing actuals** — the estimate-vs-actual loop is
  where the real value lives; skipping it reduces Pomodoro to a glorified timer.
- **Using it for the wrong work** — inherently reactive or collaborative work (live support,
  pairing, meetings) resists fixed solo intervals; reserve Pomodoro for focused solo tasks.
