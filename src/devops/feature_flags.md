# Feature Flags & Progressive Delivery

## Overview

A feature flag (feature toggle) is a runtime switch that turns functionality on or off
**without deploying new code**. This decouples *deploy* (shipping code to production) from
*release* (exposing a feature to users) — the foundation of **progressive delivery**:
rolling features out gradually, to specific cohorts, with instant rollback. It complements
[CI/CD](cicd.md) and [GitOps](gitops.md) (which get code safely *deployed*) by controlling
what users actually *experience*, and it's a close cousin of [chaos engineering](chaos_engineering.md)
in the "release safely, limit blast radius" mindset.

```
Decouple deploy from release:
  Deploy   code is in production (but dark/off)        ← CI/CD's job
  Release  the flag turns it on for some/all users     ← feature flag's job
⇒ deploy at 2pm calmly; release with a config change, not a risky deploy.
```

## Types of flags

```
Release flags     gate unfinished/new features; enable gradually. Short-lived.
Experiment flags  A/B tests — split users, measure metrics, pick a winner. Medium-lived.
Ops flags         kill switches & circuit breakers to disable a feature under load/incident.
                  Long-lived, operator-controlled.
Permission flags  entitlements — features per plan/tier/user (premium, beta program).
                  Long-lived, part of the product.
```

## How a flag is evaluated

```
if flags.enabled("new-checkout", user):   ← evaluated at runtime, per request/user
    new_checkout()
else:
    old_checkout()

The flag service decides based on TARGETING RULES:
  - % rollout            (enable for 5% of users → 25% → 100%)
  - user/segment         (internal staff, beta cohort, specific accounts)
  - attributes           (region, plan, device, app version)
  - kill switch          (force off everywhere, instantly)
Evaluation is cached/streamed to the app so it's fast and works if the service blips.
```

## Progressive delivery patterns

Feature flags enable a spectrum of gradual-rollout strategies that limit blast radius:

```
Dark launch    ship code OFF in prod; turn on for internal users to test with real traffic.
Canary release enable for a small % of users; watch error rate & latency; expand if healthy,
               kill if not. (App-level canary; infra-level canary lives in gitops.md.)
Ring deployment internal → beta → 1% → 10% → 100%, each ring a checkpoint.
Blue-green     two environments; flags/router shift traffic; instant switch back.
A/B test       split cohorts to compare variants on a business metric.
Automatic rollback: wire flag rollout to SLOs — breach the error budget → auto-disable.
                    See sre.md for SLOs/error budgets.
```

```
       5% ──watch SLOs──► 25% ──watch──► 50% ──► 100%
        │ error rate up?                          
        └────────────── kill switch (instant off) ◄── no redeploy needed
```

## Tooling

```
Managed     LaunchDarkly, Split, Flagsmith (hosted), Statsig, Unleash (open source).
Open source Unleash, Flagsmith (self-host), GrowthBook (experimentation).
Build-vs-buy A simple boolean in config is fine to start; you need a platform once you
            want targeting rules, % rollouts, audit logs, and a non-engineer UI.
Progressive-delivery controllers (Argo Rollouts, Flagger) do INFRASTRUCTURE-level canaries
            (traffic shifting) — complementary to APPLICATION-level flags.
```

## Operational discipline

```
- Default to OFF (fail safe): if the flag service is unreachable, fall back to a safe value.
- Make flags fast: evaluate locally from a cached/streamed ruleset, not a network call per check.
- Audit & access-control flag changes — a flag flip IS a production change.
- Test BOTH paths in CI (flag on and off) — a dormant branch is untested code.
```

## Technical debt: flags must die

The biggest long-term risk. Every flag is a branch in your code:

```
N boolean flags → up to 2^N code paths → combinatorial complexity, untested states.
Stale flags rot: forgotten toggles, "we think it's safe to remove?", accidental re-enables.

Discipline:
  - Track flags with an OWNER and an EXPIRY date.
  - Remove release flags promptly once a feature is 100% and stable (clean up both branches).
  - Audit for stale flags regularly; treat lingering flags as tech debt to pay down.
  - Keep long-lived ops/permission flags; retire short-lived release/experiment flags.
```

## Where this connects

- **[CI/CD](cicd.md)** — deploy continuously; flags control release timing.
- **[GitOps](gitops.md)** — infra-level canary/blue-green via Argo Rollouts/Flagger.
- **[SRE](sre.md)** — tie rollouts to SLOs/error budgets; flags as kill switches in incidents.
- **[Chaos engineering](chaos_engineering.md)** — controlled experiments and blast-radius limits.
- **[Observability](observability.md)** — measure each rollout stage to decide expand/rollback.

## Pitfalls

- **Flag debt** — never removing stale flags; the codebase fills with dead branches and
  untested path combinations.
- **No safe default** — if the flag service is down and you fail "on", you ship an unfinished
  feature to everyone.
- **Testing only one path** — the off-branch (or on-branch) silently rots.
- **Per-check network calls** — slow and fragile; evaluate from a cached ruleset.
- **Unaudited flag changes** — flipping a flag is a prod change; log who/what/when.
- **Secrets/logic in flag names** — flags are not access control for security boundaries;
  enforce real authz server-side (see ../security/api_security.md).
