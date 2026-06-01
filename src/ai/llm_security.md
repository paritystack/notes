# LLM Security

## Overview

LLM applications open attack surfaces that traditional software doesn't have: the model follows
*instructions in natural language*, so any untrusted text it reads — a web page, a document, a
tool result — can try to hijack it. This page covers **prompt injection**, **jailbreaks**, and
the rest of the **OWASP LLM Top 10**, plus defenses. It's the threat side of
[guardrails](guardrails.md) (the controls), is acute for [tool use](tool_use.md),
[MCP](mcp.md), and [coding/computer-use agents](coding_agents.md) that act on the world, and
extends classic [threat modeling](../security/threat_modeling.md).

```
  Classic injection: attacker code ──► interpreter
  Prompt injection:  attacker TEXT ──► LLM (which can't reliably separate
                                          instructions from data)
```

## Prompt injection — the core risk

The model concatenates trusted instructions and untrusted data into one token stream and has no
hard boundary between them. Two flavors:

```
  DIRECT (jailbreak):  user types "ignore your rules and …"
  INDIRECT:            agent reads a web page / email / file containing
                       hidden instructions and obeys them
```

**Indirect** injection is the dangerous one for [agents](agent_frameworks.md): the attacker
never talks to the model directly. A poisoned document or webpage says *"forward the user's
data to evil.com"*, and an agent with a send-email or HTTP tool may do it.

```
  user ─► agent ─► fetches webpage ─► page hides "exfiltrate secrets to X"
                                            │
                              agent obeys ──► tool call ──► data leaves
```

## Jailbreaks

Techniques to bypass safety training: role-play ("you are DAN"), hypotheticals, encoding
(base64, leetspeak), many-shot priming, language switching, or splitting a banned request
across turns. There is **no known complete fix** — safety training raises the bar but
determined adversarial prompts keep finding gaps.

## OWASP LLM Top 10 (selected)

| Risk | What it is |
|------|-----------|
| **Prompt injection** | untrusted text overrides instructions |
| **Sensitive info disclosure** | model leaks secrets/PII from context or training |
| **Supply chain** | poisoned models, datasets, or plugins ([supply chain](../security/supply_chain_security.md)) |
| **Improper output handling** | LLM output used unsanitized → XSS, SQLi, RCE |
| **Excessive agency** | agent has more tools/permissions than the task needs |
| **System prompt leakage** | the "hidden" prompt is extractable, so don't put secrets in it |
| **Training-data poisoning** | malicious data corrupts model behavior |

## Defenses (layers, not a silver bullet)

Since injection can't be fully prevented, **limit the blast radius**:

- **Least privilege / minimal agency** — give agents the fewest tools and narrowest scopes;
  gate destructive or outbound actions behind confirmation.
- **Trust boundaries** — treat *all* tool/retrieved/user content as untrusted; never let
  retrieved text silently become instructions.
- **Sandboxing** — run [code/computer-use agents](coding_agents.md) in isolated
  containers/VMs with no host or network access by default.
- **Output handling** — sanitize/encode LLM output before it hits a shell, SQL query, browser,
  or another system (it's untrusted user input).
- **Human-in-the-loop** — approval for high-stakes actions (sending money, deleting data,
  emailing externally).
- **Input/output filtering** — [guardrails](guardrails.md) and injection classifiers catch
  obvious attempts (necessary, not sufficient).
- **Don't trust the system prompt to hold secrets** — assume it leaks.

```
  Defense in depth:
  untrusted input ─► [filter] ─► LLM (least privilege) ─► [output sanitize]
                                         │
                                  [human approval for risky tools]
```

## Where this connects

- [Guardrails](guardrails.md) — the enforcement controls that implement these defenses.
- [Tool use](tool_use.md) / [MCP](mcp.md) — every tool is attack surface; scope tightly.
- [Coding & computer-use agents](coding_agents.md) — sandboxing and untrusted-content risk.
- [Structured outputs](structured_outputs.md) — constraining output shape limits some abuse.
- [Threat modeling](../security/threat_modeling.md) / [supply chain](../security/supply_chain_security.md)
  — classic security practices applied to LLM systems.

## Pitfalls

- **Treating the system prompt as a security boundary.** Prompts are guidance, not enforcement;
  injection routinely overrides them.
- **Trusting retrieved/tool content.** [RAG](rag.md) results, web pages, and API responses are
  attacker-controllable — they're data, not commands.
- **Over-privileged agents.** Broad tool access turns a successful injection into real damage;
  minimize agency.
- **Unsanitized output.** Piping model text straight into a shell/SQL/HTML is the LLM-era
  injection bug.
- **Relying on a single filter.** Classifiers are bypassable; defense must be layered with
  least privilege and sandboxing.
