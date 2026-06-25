# Kanban

## Overview

**Kanban** (Japanese for "signboard" or "visual card") is a method for managing work by
making it visible and limiting how much is in progress at once. It began on Toyota's
factory floor in the 1940s–50s as a *pull system*: a downstream station signalled an
upstream one — with a physical card — that it was ready for more parts, so nothing was
built until there was demand for it. In the late 2000s David J. Anderson adapted these
lean ideas to knowledge work in his book *Kanban* (2010), turning the shop-floor signal
into a board of cards that teams use to manage software, support, and personal tasks.

Where [GTD](gtd.md) decides *what* to work on and keeps your commitments in a trusted
system, and the [Pomodoro Technique](pomodoro.md) governs *how* you focus on a single task,
Kanban governs the *flow*: how work moves from "to do" to "done", and how much you allow
yourself to juggle at once. It is deliberately evolutionary — you start with whatever
process you already have and improve it incrementally, rather than imposing a new one.

```
The core idea:
  Make work VISIBLE  →  LIMIT how much is in progress  →  watch it FLOW
                     →  measure, then IMPROVE the flow
```

The heart of Kanban is a single counter-intuitive rule: **stop starting, start finishing**.
Most teams' instinct is to take on more work; Kanban's discipline is to finish what's
already begun before pulling in anything new.

## The Board

The board is the visible model of your workflow. Each column is a stage; each card is one
piece of work that moves left to right as it progresses. WIP limits (the numbers in
brackets) cap how many cards a column may hold.

```
┌──────────────┬──────────────────┬──────────────────┬──────────────┐
│   BACKLOG    │   TO DO  [3]     │ IN PROGRESS [2]  │    DONE       │
├──────────────┼──────────────────┼──────────────────┼──────────────┤
│ ┌──────────┐ │ ┌──────────┐     │ ┌──────────┐     │ ┌──────────┐  │
│ │ Card F   │ │ │ Card C   │     │ │ Card A   │     │ │ Card X   │  │
│ └──────────┘ │ └──────────┘     │ └──────────┘     │ └──────────┘  │
│ ┌──────────┐ │ ┌──────────┐     │ ┌──────────┐     │ ┌──────────┐  │
│ │ Card G   │ │ │ Card D   │     │ │ Card B   │     │ │ Card Y   │  │
│ └──────────┘ │ └──────────┘     │ └──────────┘     │ └──────────┘  │
│     ...      │ ┌──────────┐     │   ◀ at limit ▶   │     ...       │
│              │ │ Card E   │     │                  │               │
│              │ └──────────┘     │                  │               │
└──────────────┴──────────────────┴──────────────────┴──────────────┘
   pull ───────────────────────────────────────────────▶
```

Work is **pulled**, not pushed: a column takes a new card only when it has free capacity
under its WIP limit. When *In Progress* is full, you can't start *Card C* — you must first
finish *A* or *B* and move it to *Done*. That pull constraint is what forces the team to
swarm on finishing instead of fanning out across half-done tasks.

## The Core Practices

Anderson frames Kanban as a set of practices layered on top of your existing process, not a
rip-and-replace methodology:

```
  1. VISUALIZE          put the real workflow on a board; nothing hidden
  2. LIMIT WIP          cap each stage; pull, don't push
  3. MANAGE FLOW        watch how smoothly cards move; attack blockages
  4. EXPLICIT POLICIES  write the rules: what "Done" means, how to pull
  5. FEEDBACK LOOPS     regular reviews — standups, replenishment, retros
  6. IMPROVE            change the process incrementally, guided by data
```

These rest on three change-management principles that make Kanban easy to adopt: start with
what you do now, agree to pursue *incremental* change, and respect current roles and
responsibilities. There's no "Kanban team structure" to reorganize into — you simply make
the present visible and tune it.

## WIP Limits

Limiting work-in-progress is the practice that distinguishes Kanban from an ordinary to-do
board. Counter-intuitively, doing fewer things at once gets them all done *sooner*.

```
NO WIP LIMIT — start everything, finish nothing for a long time
   A ▓▓░░░░░░▓▓     each task stalls while you switch to the next
   B ░░▓▓░░░░░░▓▓   context-switching tax on every hop
   C ░░░░▓▓░░░░░░▓  all four finish near the very end
   D ░░░░░░▓▓░░░░▓
        ───────────────────────────────▶ time  (everything late)

WIP LIMIT = 1 — finish each before starting the next
   A ▓▓▓▓                A done early and shippable
   B     ▓▓▓▓            B done next
   C         ▓▓▓▓        steady, predictable completion
   D             ▓▓▓▓
        ───────────────────────────────▶ time  (early items ship sooner)
```

The intuition is **Little's Law**: average lead time = average WIP ÷ average throughput.
Hold throughput roughly constant and the only lever for shorter lead times is *less WIP*.
Lower limits also expose bottlenecks fast — when a column fills up and stops pulling, the
constraint becomes impossible to ignore, which is exactly the point.

## Flow Metrics

Because cards have a visible history, Kanban produces honest data about how work actually
moves. The key measures:

```
┌──────────────┬───────────────────────────────────────────────────────┐
│ Metric       │ Meaning                                                 │
├──────────────┼───────────────────────────────────────────────────────┤
│ Lead time    │ from request made → delivered (the customer's view)     │
│ Cycle time   │ from work started → finished (the team's view)          │
│ Throughput   │ cards completed per unit time (e.g. per week)           │
│ WIP          │ cards in progress right now                             │
│ Aging        │ how long each in-progress card has sat without moving   │
└──────────────┴───────────────────────────────────────────────────────┘
```

A **Cumulative Flow Diagram** (CFD) plots how many cards sit in each stage over time. The
vertical gap between bands is current WIP; the horizontal gap is approximate lead time.
Bands that widen reveal a stage where work is piling up faster than it leaves.

```
 cards │            ░░░░░░░░░ Backlog
       │        ▒▒▒▒▒▒▒▒▒▒▒▒  To Do
       │     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓   In Progress   ← a widening band = bottleneck
       │  ████████████████    Done
       └───────────────────────────▶ time
```

## Kanban vs Scrum

Both are popular agile approaches, but they answer different questions. Kanban optimizes
*flow*; [Scrum](scrum.md) organizes work into fixed *sprints*.

```
┌───────────────────┬──────────────────────┬──────────────────────────┐
│                   │ KANBAN               │ SCRUM                     │
├───────────────────┼──────────────────────┼──────────────────────────┤
│ Cadence           │ continuous flow      │ fixed sprints (1–4 weeks) │
│ Commitment        │ pull when capacity    │ commit to a sprint backlog│
│ Limits work via   │ WIP limits per column│ sprint backlog size       │
│ Roles             │ none prescribed      │ PO, Scrum Master, team    │
│ Change mid-stream │ anytime              │ avoided during a sprint   │
│ Key metric        │ lead/cycle time      │ velocity per sprint       │
│ Board reset       │ never (persistent)   │ per sprint                │
└───────────────────┴──────────────────────┴──────────────────────────┘
```

They aren't mutually exclusive — **Scrumban** keeps Scrum's cadence and roles while adding
Kanban's WIP limits and flow focus. Kanban tends to suit continuous, interrupt-driven work
(support, ops, maintenance); Scrum suits work that benefits from a planning rhythm.

## Tools & Implementation

Kanban is low-tech at heart: the canonical board is a whiteboard with sticky notes, and
much is lost when teams jump straight to software without first making the *real* process
visible. Options scale from analog to fully digital:

```
Analog     a whiteboard or wall + sticky notes (one per card)
Personal   "Personal Kanban" — three columns on a fridge or notebook
Digital    Trello, Jira, Azure Boards, Linear, GitHub Projects (board view)
Dev/CLI    org-mode TODO states as columns, or a Taskwarrior board
```

For solo use, **Personal Kanban** (Jim Benson & Tonianne DeMaria Barry) strips it to two
rules — *visualize your work* and *limit your WIP* — and pairs cleanly with a GTD
next-actions list as the backlog and Pomodoro for executing each card.

## Where this connects

- [Getting Things Done (GTD)](gtd.md) — GTD's *Next Actions* are an ideal backlog to pull
  cards from; Kanban then manages how many you allow yourself to have in flight.
- [Pomodoro Technique](pomodoro.md) — once a card is *In Progress*, run pomodoros on it;
  Kanban manages the queue, Pomodoro manages the focus on the current item.
- **Scrum / agile** — a sibling agile method; combine as *Scrumban*. See the comparison
  table above.
- **Lean / Toyota Production System** — Kanban's origin; the pull system and *just-in-time*
  flow come straight from lean manufacturing.
- **Theory of Constraints** (Goldratt) — WIP limits surface the bottleneck, which is exactly
  the constraint TOC says to identify and exploit first.
- **Personal Kanban** (Benson & Barry) — the minimal, individual-scale application.

## Pitfalls

- **A board with no WIP limits** — without limits it's just a prettier to-do list; the flow
  benefits and bottleneck signals never appear. The limits *are* the method.
- **A "Done" column that isn't done** — if "Done" really means "done except for review/
  deploy", lead-time data lies. Define explicit policies for what each transition requires.
- **Too many columns** — over-modelling every micro-step makes the board noisy and the WIP
  limits meaningless. Start coarse and split a column only when flow data demands it.
- **Ignoring the metrics** — a board you only look at, never measure, can't drive
  improvement. Track cycle time and aging, and act on widening CFD bands.
- **Treating it as Scrum-lite** — Kanban isn't "Scrum without estimates"; it's a different
  optimization (flow, not iterations). Adopting the board but keeping a push mindset misses
  the point of pulling work.
- **Skipping the "start with what you do now" principle** — designing an idealized board
  nobody follows defeats the visualization. Map the *real* current workflow first.
