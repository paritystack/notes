# Multi-Agent Systems

## Overview

When a single [agent](agent_frameworks.md) with one prompt and a big toolset starts to struggle —
too many tools, conflicting instructions, a task with distinct phases — the answer is often to
split it into several specialized agents that coordinate. This page covers the orchestration
**patterns**, how agents communicate, and when multi-agent helps versus hurts. It builds on
[agent memory](agent_memory.md) (shared vs private state), [tool use](tool_use.md), and
[MCP](mcp.md) (a common way agents expose capabilities to each other). [Coding
agents](coding_agents.md) are a concrete application.

```
  Single agent: one context, one persona, all tools  ──► overload at scale
  Multi-agent:  specialists with focused contexts + a coordinator
```

## Why split (and why not)

| Multi-agent wins when… | Single agent wins when… |
|------------------------|--------------------------|
| distinct sub-skills (research vs code vs review) | task is cohesive |
| parallelizable sub-tasks | steps are tightly sequential |
| tool count overwhelms one prompt | few tools |
| isolation/safety boundaries needed | low latency / cost matters |

Each agent adds latency, token cost, and a **coordination tax** — more agents means more places
for miscommunication. Reach for it only when one agent demonstrably can't cope.

## Orchestration patterns

```
  SUPERVISOR (orchestrator-worker)        SWARM / HANDOFF
  ┌─────────────┐                         A ──► B ──► C
  │ supervisor  │ routes + aggregates     control passes peer-to-peer
  └──┬───┬───┬──┘                         no central boss
     ▼   ▼   ▼
   wkrA wkrB wkrC

  PIPELINE                                 HIERARCHICAL
  A ─► B ─► C  (fixed stages,              supervisor of supervisors
               each transforms output)     (teams of teams)
```

- **Supervisor / orchestrator-worker** — a lead agent decomposes the task, dispatches to
  workers (often in parallel), and synthesizes results. Most common, most controllable.
- **Swarm / handoff** — agents transfer control to whichever specialist fits the current state
  (OpenAI Swarm / Agents SDK model). Flexible, harder to reason about.
- **Pipeline** — fixed sequence of specialists, each consuming the previous output.
- **Hierarchical** — supervisors managing sub-teams, for large decompositions.
- **Debate / reflection** — multiple agents critique each other to improve quality (related to
  [reasoning models](reasoning_models.md)).

## Communication & shared state

```
  Shared scratchpad (blackboard)   — all agents read/write one state object
  Message passing                  — explicit handoff payloads between agents
  Shared memory store              — common vector DB / KB (see agent_memory)
```

The hardest design question is **context isolation**: give each agent only what it needs (focus,
lower cost) versus a shared view (coherence, no information loss). Anthropic's multi-agent
research found the orchestrator must pass *detailed* task briefs to subagents — vague delegation
causes duplicated or divergent work.

## Frameworks

**LangGraph** (graph of agents/edges, explicit state), **CrewAI** (role-based crews),
**AutoGen** (conversational agents), **OpenAI Agents SDK / Swarm** (handoffs). They differ
mainly in how much structure they impose on control flow and state. See
[agent frameworks](agent_frameworks.md) for the single-agent foundations.

## Where this connects

- [Agent frameworks](agent_frameworks.md) — each node here is a single agent.
- [Agent memory](agent_memory.md) — shared vs per-agent memory and context.
- [MCP](mcp.md) — standard way agents expose tools/resources to one another.
- [Tool use](tool_use.md) — sub-agents are often invoked as tools by a supervisor.
- [Coding agents](coding_agents.md) — planner/coder/reviewer is a classic multi-agent split.

## Pitfalls

- **Multiplying agents prematurely.** Most tasks don't need it; a single well-prompted agent is
  cheaper and more reliable.
- **Vague delegation.** Under-specified subagent briefs cause overlap, gaps, and contradictory
  results — pass explicit objectives and boundaries.
- **Cost/latency explosion.** Parallel agents multiply token spend; one user request can fan out
  into dozens of LLM calls.
- **Error propagation.** A wrong intermediate result flows downstream unchecked; add verification
  between stages.
- **Lost context at handoffs.** Information dropped between agents can't be recovered — be
  deliberate about what crosses the boundary.
