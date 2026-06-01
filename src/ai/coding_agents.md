# Coding & Computer-Use Agents

## Overview

The most mature application of [agents](agent_frameworks.md): systems that read, write, run, and
debug code — and more generally *operate software* by using a computer. This page covers
autonomous coding loops, the SWE-bench-driven evaluation that shaped them, and **computer use**
(an agent driving a screen, mouse, and keyboard). It builds on [tool use](tool_use.md),
[agent memory](agent_memory.md), and [software dev prompts](software_dev_prompts.md), and is the
domain of [Claude Code](cli.md). The verification-heavy loop here connects to
[reasoning models](reasoning_models.md).

```
  The coding-agent loop:
  read context ─► plan ─► edit files ─► run (build/test) ─► read output ─► repeat
        ▲                                                          │
        └────────────────── until tests pass ──────────────────────┘
```

## What makes a coding agent work

The defining feature is a **feedback loop with ground truth**: unlike open-ended chat, code can
be *run*, so the agent gets objective signal (compile errors, test failures, stack traces) and
self-corrects. The quality of an agent is largely the quality of that loop:

- **Tools** — file read/edit, shell, search/grep, run tests, language server. See
  [tool use](tool_use.md).
- **Context gathering** — find the *relevant* files (grep/embeddings) instead of dumping the
  repo; see [agent memory](agent_memory.md) and [RAG](rag.md).
- **Verification** — run tests/linters and feed results back. This is the single biggest driver
  of success.
- **Sandboxing** — execute in an isolated container/VM so the agent can't damage the host (see
  [LLM security](llm_security.md)).

## The landscape

| Tool | Form | Notes |
|------|------|-------|
| [Claude Code](cli.md) | terminal agent | plan mode, subagents, hooks, MCP |
| Cursor / Windsurf | AI-native IDE | inline edits + agent mode |
| GitHub Copilot | IDE assistant → agent | completion + agentic PRs |
| Devin, OpenHands | autonomous SWE agent | end-to-end task completion |
| SWE-agent, Aider | open-source CLI agents | research + practical |

**SWE-bench** — the benchmark that drove the field: real GitHub issues from open-source repos;
the agent must produce a patch that passes the project's hidden tests. SWE-bench Verified is the
human-validated subset. Scores jumped from a few percent to most of the benchmark in ~two years.

## Computer use / browser agents

A generalization beyond code: the model is given **screenshots** and emits low-level actions
(click x,y / type / scroll / keypress), looping on the new screenshot. This lets an agent
operate any GUI — fill forms, navigate apps, scrape, test UIs.

```
  screenshot ─► model ─► action (click/type) ─► new screenshot ─► …
  grounded in pixels, not APIs
```

Browser-specific variants (Playwright/Puppeteer-driven, or DOM-aware agents) are more reliable
than raw pixel control because they act on structured page elements. Trade-offs: slow, brittle
to UI changes, and a real security surface — a malicious page can attempt
[prompt injection](llm_security.md) through on-screen text.

## Where this connects

- [Claude Code CLI](cli.md) — a production coding agent; this is its problem domain.
- [Tool use](tool_use.md) — file/shell/test tools are the agent's hands.
- [Agent memory](agent_memory.md) — gathering and compacting repo context.
- [Multi-agent systems](multi_agent_systems.md) — planner/coder/reviewer decompositions.
- [LLM security](llm_security.md) — sandboxing and injection risk from code/pages.
- [LLM evaluation](llm_evaluation.md) — SWE-bench and friends measure these agents.

## Pitfalls

- **No verification loop.** An agent that edits without running tests is just autocomplete;
  ground every change in executable feedback.
- **Context dumping.** Pasting the whole repo wastes the window and distracts the model; retrieve
  the relevant files.
- **Running untrusted actions on the host.** Always sandbox shell/code execution and gate
  destructive commands.
- **Pixel control where an API exists.** Computer use is a last resort; prefer real APIs/DOM
  access when available.
- **Trusting on-screen/page text.** Browser and computer-use agents can be hijacked by injected
  instructions in the content they read.
