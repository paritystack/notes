# Scrum

## Overview

**Scrum** is the most widely used **agile** framework for developing and sustaining complex
products. It organizes work into short, fixed-length iterations called **sprints** (usually
1–4 weeks), at the end of which the team aims to deliver a usable increment. Rather than
planning a whole project up front, Scrum is *empirical*: build a little, inspect the result
with stakeholders, and adapt the plan — repeatedly. The framework was formalized by Ken
Schwaber and Jeff Sutherland, whose *Scrum Guide* is its canonical, deliberately minimal
definition.

Scrum is the cadence-based sibling of [Kanban](kanban.md): both are agile, both visualize
work, but where Kanban manages *continuous flow* with WIP limits, Scrum batches work into
*time-boxed sprints* with a committed goal. The `kanban.md` page already contrasts the two
side by side — Scrum suits work that benefits from a planning rhythm and regular stakeholder
checkpoints.

```
The empirical loop:  PLAN a sprint → BUILD → INSPECT with stakeholders → ADAPT → repeat

Three pillars:  TRANSPARENCY · INSPECTION · ADAPTATION
```

## Roles (Accountabilities)

A Scrum Team is small (typically ≤10) and cross-functional, with three accountabilities:

```
  PRODUCT OWNER     owns the WHAT and the WHY.
                    orders the Product Backlog, maximizes value, the single
                    voice for stakeholders.

  SCRUM MASTER      owns the HOW-WELL (the process).
                    a servant-leader: coaches the team, removes impediments,
                    protects the team from disruption. NOT a project manager.

  DEVELOPERS        own the HOW.
                    the people who build the increment; self-managing,
                    decide how to turn backlog items into working product.
```

## Artifacts

```
  PRODUCT BACKLOG   the single ordered list of everything that might be
                    needed in the product. Owned by the Product Owner;
                    refined continuously. Goal: the Product Goal.

  SPRINT BACKLOG    the items selected for THIS sprint + a plan to deliver
                    them. Owned by the Developers. Goal: the Sprint Goal.

  INCREMENT         the sum of completed work — a usable, potentially
                    shippable product slice. Goal: the Definition of Done.
```

Each artifact carries a commitment (Product Goal / Sprint Goal / Definition of Done) that
makes progress and quality transparent.

## Events

All events are *time-boxed* and happen within the sprint, which is itself the container:

```
  ┌──────────────────────── SPRINT (1–4 weeks) ─────────────────────────┐
  │                                                                      │
  │  SPRINT PLANNING   ── what can we deliver, and how? → Sprint Backlog │
  │        │                                                             │
  │        ▼                                                             │
  │   ┌─ DAILY SCRUM (15 min) ─┐   ← every day: inspect progress toward  │
  │   │  re-plan the next day  │     the Sprint Goal, adapt the plan     │
  │   └────────────────────────┘                                        │
  │        │   ...repeats daily...                                       │
  │        ▼                                                             │
  │  SPRINT REVIEW     ── demo the increment, get stakeholder feedback   │
  │        │                                                             │
  │        ▼                                                             │
  │  SPRINT RETROSPECTIVE ── how can WE (team + process) improve? ───────┤
  │                                              → start next sprint     │
  └──────────────────────────────────────────────────────────────────────┘
```

- **Sprint Planning** — opens the sprint; sets the Sprint Goal and selects backlog items.
- **Daily Scrum** — a 15-minute daily sync for the Developers to re-plan toward the goal
  (not a status report to a manager).
- **Sprint Review** — near the end; inspect the increment *with stakeholders* and adapt the
  Product Backlog.
- **Sprint Retrospective** — closes the sprint; the team inspects *itself* and commits to
  concrete improvements.

## Estimation & Velocity

Teams often size backlog items in relative **story points** rather than hours, and track
**velocity** — points completed per sprint — to forecast how much fits in future sprints.
Velocity is a *planning aid for the team*, not a productivity scoreboard; comparing it across
teams or using it as a target corrupts it (see Pitfalls).

## Scrumban

**Scrumban** blends the two: keep Scrum's cadence, roles, and events, but add
[Kanban](kanban.md)'s WIP limits and flow metrics to the board. It's a common landing spot
for teams that want Scrum's rhythm without rigidly batching every change into a sprint.

## Where this connects

- [Kanban](kanban.md) — the continuous-flow alternative; see the Kanban-vs-Scrum comparison
  table there, and Scrumban as the hybrid.
- **OKRs** (Objectives & Key Results) — a goal-setting layer *above* sprints; the Product
  Goal can ladder up to organizational OKRs.
- [Getting Things Done (GTD)](gtd.md) — a personal analog of Scrum's empirical loop; GTD's
  Weekly Review mirrors the inspect-and-adapt rhythm of a sprint boundary.
- **Lean / agile** — Scrum implements the Agile Manifesto's values; it shares lean roots
  with Kanban.
- [Pomodoro Technique](pomodoro.md) — an individual focus mechanism a developer can use to
  execute their slice of the Sprint Backlog.

## Pitfalls

- **Zombie / cargo-cult Scrum** — performing the ceremonies without the empirical mindset;
  going through the motions while ignoring inspect-and-adapt. The rituals aren't the point.
- **Daily Scrum as a status meeting** — turning the team re-planning session into a
  manager-facing status report kills its purpose. It's *for the Developers*.
- **Skipping the retrospective (or never acting on it)** — without real improvement actions,
  the same problems recur sprint after sprint. The retro is where the framework self-corrects.
- **Velocity as a target** — pressuring teams to "go faster" inflates estimates (Goodhart's
  law); velocity is for forecasting, not performance review, and isn't comparable across teams.
- **Scope creep mid-sprint** — injecting new work into a committed sprint undermines the
  Sprint Goal and predictability; new items belong in the Product Backlog.
- **Scrum Master as project manager** — reverting the servant-leader role into a
  command-and-control PM defeats the team's self-management.
- **Forcing Scrum on unsuitable work** — highly interrupt-driven or unpredictable work
  (support, ops) often fits [Kanban](kanban.md)'s flow model better than fixed sprints.
