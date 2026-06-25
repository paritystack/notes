# Deployment Strategies

## Overview

A deployment strategy is **how a new version of a service replaces the old one** at the
infrastructure layer — how instances are swapped and how traffic is shifted from `v1` to `v2`.
This is the *deployment-level* counterpart to the *application-level* progressive delivery in
[feature flags](feature_flags.md): flags decide what a *running* process exposes to a user;
deployment strategies decide *which process* (which version) handles the request at all. The
two compose — you can canary-deploy `v2` and *also* gate a feature inside it behind a flag.

The strategies live wherever traffic can be steered: a [Kubernetes](kubernetes.md) Deployment
rolling pods, a load balancer or DNS flipping a target, or a [service mesh](service_mesh.md)
weighting routes. Under [GitOps](gitops.md) the strategy is declared in Git and a controller
(Argo Rollouts, Flagger) executes it; promotion/rollback decisions are gated on the golden
signals from [monitoring](monitoring.md) and tied to [SRE](sre.md) error budgets.

```
deploy  = a new version is RUNNING in production        ← deployment strategy's job
release = users actually hit that version / feature     ← traffic shift + feature flag's job
A strategy controls the deploy→release transition: how fast, how reversible, at what cost.
```

## The strategies at a glance

```
Strategy     Downtime  Extra capacity   Rollback speed   Blast radius   Versions live at once
─────────────────────────────────────────────────────────────────────────────────────────────
Recreate     yes       none             redeploy old     all users      1 (hard cutover)
Rolling      none      ~1 batch         drain back       grows w/ batch  2 (overlap window)
Blue-Green   none      2x (full env)    instant flip     all-or-nothing  2 (only 1 serves)
Canary       none      small (+canary)  drop canary      tiny → growing  2 (weighted split)
Shadow       none      +mirror env      n/a (no users)   zero (no users) 2 (v2 sees copies)
```

Pick by what you're optimizing: Rolling is the cheap default; Blue-Green buys instant
rollback at double cost; Canary minimizes blast radius at the cost of orchestration; Shadow
validates with real traffic and zero user risk but can't catch user-facing bugs.

## Recreate & Rolling

**Recreate**: stop all `v1`, then start all `v2`. Simple, no version overlap (safe for
incompatible schemas), but incurs **downtime**. Fine for batch jobs or dev; rarely for
user-facing services.

**Rolling** (the default Kubernetes `Deployment` behavior): replace pods in batches, gated by
**readiness probes** so a new pod only takes traffic once healthy.

```
maxSurge=25%        how many EXTRA pods above desired count may be created during the roll
maxUnavailable=25%  how many below desired may be missing at once
readinessProbe      a pod is added to the Service endpoints only after it passes → no
                    traffic to a not-yet-warm pod

Roll:  [v1 v1 v1 v1] → add v2, wait ready, drain a v1 → … → [v2 v2 v2 v2]
```

Because `v1` and `v2` serve simultaneously during the roll, **both versions must be
backward/forward compatible** (API and data) — see schema compatibility below.

## Blue-Green

Run two complete environments. **Blue** serves production; **Green** holds the new version,
fully deployed and smoke-tested. Cut over by repointing the router (LB target group, DNS,
or mesh route) all at once. Rollback = flip back to Blue — near-instant.

```
        ┌─ Blue  (v1)  ◄── 100% traffic
router ─┤
        └─ Green (v2)  ◄── 0%  (warm, tested)

flip:   router → Green ; Blue kept idle as the instant rollback target
```

Trade-offs: needs **double capacity** during the window, and a hard cutover means a bad
release hits *everyone* at once (no gradual exposure — combine with Canary or flags for that).
The sharp edge is **stateful** cutover: in-flight sessions and especially **database
migrations** can't be flipped instantly (the DB is shared) — handle schema changes separately.

## Canary

Shift a **small slice** of real traffic to `v2`, watch its golden signals, then promote in
steps or auto-rollback on breach.

```
       5% ──watch error rate & latency──► 25% ──watch──► 50% ──► 100%
        │  SLO breach?                                         (promote: v2 becomes baseline)
        └──────────────── drop canary, route 100% → v1 (auto-rollback)
```

Where the split happens:
- **Mesh / ingress weights** — Istio/Linkerd or an ingress controller route N% of requests.
- **Replica ratio** — crude canary via pod counts (9× v1 + 1× v2 ≈ 10%); imprecise.
- **Progressive-delivery controllers** — Argo Rollouts / Flagger automate the step-up,
  query Prometheus for the analysis, and roll back automatically. This is the infra-level
  canary that [GitOps](gitops.md) and [feature flags](feature_flags.md) refer to.

Contrast with the **application-level canary** in [feature flags](feature_flags.md): a flag
exposes a feature to a user cohort *inside one running version*; a deployment canary routes
traffic to a *different deployed version*. Same "limit blast radius" goal, different layer.

## Shadow / traffic mirroring

Send a **copy** of live production traffic to `v2` while users still get `v1`'s responses;
`v2`'s responses are **discarded**. Validates performance and correctness against real traffic
shapes with **zero user impact**.

```
                ┌─► v1 ──► response to user
real request ──►┤
                └─► v2 (mirror) ──► response DISCARDED, but metrics/logs captured
```

The critical caveat: **side effects**. Mirrored requests must not double-charge, double-send,
or mutate shared state — `v2` needs sandboxed/idempotent downstreams (stub the payment gateway,
use a scratch queue) or you corrupt production. Best for read-heavy or replay-safe paths.

## Rollback & health gating

Every strategy needs a clear, fast undo:

```
Readiness/liveness probes  gate traffic onto healthy pods; restart unhealthy ones.
Automated analysis         promote/rollback on the golden signals (error rate, latency,
                           saturation) — see monitoring.md / observability.md.
GitOps rollback            the declared version lives in Git → `git revert` and the agent
                           reconciles back. Recovery is a Git operation (see gitops.md).
Tie to error budgets       a rollout that burns the budget auto-halts — see sre.md.
```

The rule: **never deploy a strategy without a tested rollback path**. Blue-Green keeps the old
env warm; Canary drops the canary; Rolling drains new pods back to old.

## The hard part: schema & backward compatibility

Every zero-downtime strategy runs `v1` and `v2` **at the same time**, so they must tolerate
each other's data and API. The discipline is **expand/contract** (parallel change):

```
Expand    add the new column/field/endpoint — additive, both versions still work.
Migrate   backfill data; deploy v2 which writes new + reads either.
Contract  once v1 is fully gone, remove the old column/field in a LATER release.

Never  rename/drop in the same deploy that ships the code needing the new shape —
       that forces a hard cutover and breaks rollback.
```

This is why a destructive DB migration can't ride a Blue-Green flip: the database is shared,
so schema changes get their own expand/contract sequence decoupled from the app cutover.
See [database design](../databases/database_design.md).

## Where this connects

- **[Feature flags](feature_flags.md)** — application-level progressive delivery; composes with
  deployment-level canary/blue-green.
- **[GitOps](gitops.md)** — declares the strategy in Git; Argo Rollouts/Flagger execute it.
- **[Kubernetes](kubernetes.md)** — the Deployment object provides rolling updates and probes.
- **[Service Mesh](service_mesh.md)** — weighted routing and mirroring for canary/shadow.
- **[Monitoring](monitoring.md)** / **[Observability](observability.md)** — the signals that gate
  promotion and trigger auto-rollback.
- **[SRE](sre.md)** — error budgets decide whether a rollout may continue.
- **[Cloud Deployment](cloud-deployment.md)** — LB/DNS/target-group mechanics behind the cutover.

## Pitfalls

- **Non-backward-compatible rollouts** — shipping a breaking schema/API in a rolling or
  blue-green deploy; `v1` and `v2` overlap and one of them breaks. Use expand/contract.
- **Stateful / DB cutover** — treating the database like the app and "flipping" it; migrations
  need their own decoupled sequence.
- **No automated rollback** — a manual "someone notices and redeploys" path is too slow; gate
  on golden signals and revert automatically.
- **Statistically meaningless canary** — too little traffic (or too short a window) to detect a
  regression; the canary passes by luck.
- **Session stickiness ignored** — users bounced between `v1` and `v2` mid-session hit
  inconsistent behavior; pin sessions or keep versions compatible.
- **Mirrored side effects** — shadow traffic that mutates real downstreams (double charges,
  duplicate emails); sandbox or idempotency-guard `v2`.
- **Double-cost surprise** — blue-green/canary need extra capacity; size and budget for the
  overlap window.
