# Threat Modeling

## Overview

Threat modeling is the *design-time* security activity: systematically finding what can go
wrong with a system **before** it's built, so defenses are deliberate rather than
reactive. Where the [OWASP Top 10](owasp_top_10.md) catalogues known *implementation*
bugs and [security testing](../testing/security_testing.md) hunts them after the fact,
threat modeling reasons about the *architecture* — trust boundaries, data flows, and
attacker goals. It pairs naturally with [zero trust](zero_trust.md) (which assumes the
network is hostile) and feeds the controls in [API security](api_security.md),
[key management](key_management.md), and [secrets management](secrets_management.md).

```
The four framing questions (Shostack):
  1. What are we building?      → diagram the system
  2. What can go wrong?         → enumerate threats
  3. What are we doing about it? → choose mitigations
  4. Did we do a good job?      → validate & iterate
```

## Build a data-flow diagram first

You can't model threats you can't see. Draw the system as elements + flows, and mark
**trust boundaries** — the lines where data crosses from one privilege level to another
(these are where most threats live).

```
   [User] ──HTTPS──► (Web API) ──► (Auth svc) ──► [User DB]
      │                  │                            │
   ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ │─ trust boundary ─ ─ ─ ─ ─ ─│─ ─
   external            DMZ / app tier            data tier

  ◻ external entity   ○ process   ▭ data store   → data flow
  ╌╌ trust boundary (internet→app, app→db, tenant→tenant)
```

## STRIDE — what can go wrong

STRIDE is the most-used threat taxonomy. Walk each element/flow and ask whether each
threat applies. Each STRIDE category is the *violation* of a security property:

```
Threat                    Violates          Example                         Mitigation class
──────────────────────────────────────────────────────────────────────────────────────────
Spoofing                  Authentication    fake login, forged token        strong authn, MFA
Tampering                 Integrity         modify data in transit/at rest  signing, HMAC, TLS
Repudiation               Non-repudiation   "I never sent that"             audit logs, signatures
Information disclosure     Confidentiality   leak PII, read others' data     encryption, access control
Denial of service          Availability      flood, resource exhaustion      rate limiting, quotas
Elevation of privilege     Authorization     user → admin                    least privilege, authz checks
```

Related taxonomies: **LINDDUN** focuses on *privacy* threats (linkability,
identifiability), and **MITRE ATT&CK** catalogues real-world attacker *techniques* —
useful for detection engineering rather than design.

## Attack trees

A complementary, attacker-goal-centric view: put the goal at the root, decompose into the
ways to achieve it (OR = alternatives, AND = all required). Useful for reasoning about a
specific high-value asset.

```
GOAL: read another tenant's data
├── OR  exploit broken object-level authz (IDOR)   ← see OWASP API
├── OR  steal a valid session token
│   ├── AND  XSS to exfiltrate cookie
│   └── AND  cookie lacks HttpOnly/SameSite
└── OR  SQL injection to dump the table
```

## Prioritizing — risk ranking

You can't fix everything; rank by risk so effort lands where it matters.

```
Risk ≈ Likelihood × Impact

DREAD (legacy, subjective): Damage, Reproducibility, Exploitability,
                            Affected users, Discoverability — score & sum.
Better in practice: a simple Likelihood×Impact matrix, or CVSS for known CVEs.

         Impact →
   L  │ low   med   high
   i  ├──────────────────
   k  │  ·     ·     ▲      ▲ = fix now
   e  │  ·     ▲     ▲      · = accept/monitor
   →  │  ·     ·     ▲
```

## Mitigation strategies

For each accepted threat, choose one (explicitly — "accept" is a valid, documented choice):

```
Mitigate   add a control (authz check, rate limit, encryption)
Eliminate  remove the feature / data (can't leak what you don't store)
Transfer   shift risk (managed service, insurance, third-party auth)
Accept     document the residual risk and move on (low likelihood × low impact)
```

## When and how to do it

```
- Early: at design, and whenever the architecture changes (new trust boundary, new data).
- Lightweight & continuous beats a one-time heavyweight document.
- Whole-team activity, not a security-team handoff — developers know the data flows.
- Output: a living list of threats → decisions → tracked mitigation tickets.
```

Tools: Microsoft Threat Modeling Tool, OWASP Threat Dragon (open source), or just a
whiteboard and the STRIDE table. The artifact matters less than the conversation.

## Where this connects

- **[OWASP Top 10](owasp_top_10.md)** / **[API security](api_security.md)** — concrete
  vulnerability classes your model should anticipate.
- **[Zero trust](zero_trust.md)** — an architectural answer to "assume breach".
- **[Supply chain security](supply_chain_security.md)** — threats from dependencies and
  build pipelines, often missed in app-only models.
- **[Security testing](../testing/security_testing.md)** — validates that modeled threats
  are actually mitigated.

## Pitfalls

- **Boiling the ocean** — modeling every theoretical threat; focus on assets and trust
  boundaries that matter.
- **One-and-done** — a model that isn't updated with the architecture is quickly fiction.
- **Security-team-only** — without developers, the data-flow diagram is wrong.
- **No follow-through** — threats identified but never turned into tracked, fixed work.
