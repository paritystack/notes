# GitOps

## Overview

GitOps is a deployment model where **Git is the single source of truth for the desired state
of your system**, and an automated agent continuously reconciles the live cluster to match
what's in Git. Instead of a [CI/CD pipeline](cicd.md) *pushing* changes into production
(`kubectl apply`, scripts with cluster credentials), a controller running *inside* the
cluster *pulls* the declared state and converges to it. It's the natural operating model for
[Kubernetes](kubernetes.md) and pairs with [Helm](helm.md) for packaging and
[Terraform](terraform.md) for infrastructure.

```
Traditional push:   CI ──(has cluster creds)──► kubectl apply ──► cluster
GitOps pull:         Git (desired state) ◄──watches── Agent (in cluster) ──reconciles──► cluster
                                                          │
                              live ≠ Git → fix it (continuous reconciliation)
```

## The four principles

```
1. Declarative      the whole system is described declaratively (YAML manifests, Helm,
                    Kustomize) — WHAT you want, not HOW to get there.
2. Versioned        the desired state lives in Git → versioned, auditable, reviewable.
3. Pulled auto.     an agent automatically pulls the approved state (no manual apply).
4. Continuously     the agent observes live state and reconciles any drift back to Git.
   reconciled
```

## Why pull beats push

```
Security    no cluster credentials in CI; the agent runs IN the cluster and pulls.
            Your CI never needs prod access → smaller blast radius. See secrets_management.
Auditability every change is a Git commit: who, what, when, reviewed by whom.
Drift control manual `kubectl edit` hotfixes get reverted automatically — the cluster
            cannot silently diverge from Git.
Rollback    `git revert` → the agent rolls the cluster back. Recovery = a Git operation.
Consistency rebuild a cluster from Git alone (disaster recovery, new region/environment).
```

## Reconciliation loop

The heart of GitOps (and of Kubernetes itself):

```
loop forever:
    desired = read manifests from Git
    actual  = read live state from cluster
    if actual ≠ desired:
        apply changes to converge actual → desired
        report status (Synced / OutOfSync / Degraded)
```

This is the same controller pattern as Kubernetes' own control loops — GitOps just extends
it so Git, not etcd-via-kubectl, is the authority.

## Tooling

```
Argo CD     most popular; rich UI showing app health & sync status, sync waves, app-of-apps
            for managing many apps, drift visualization. CNCF graduated.
Flux        lightweight, CRD-driven, GitOps Toolkit; strong multi-tenancy & Helm/Kustomize
            integration. CNCF graduated.
Config tooling: Kustomize (overlays per environment) and Helm (templated charts) describe
            the manifests these agents apply.
```

## Repository structure

```
Separate APP code from DEPLOYMENT config (two repos, or clear separation):
  app repo:     source + Dockerfile → CI builds & pushes an image, bumps the tag
                in the config repo.
  config repo:  manifests/Helm/Kustomize per environment (dev/staging/prod overlays).
                The GitOps agent watches THIS repo.

Promotion = a pull request that moves an image tag from staging → prod overlay.
            Review + merge = deploy. Environments differ only by overlay, not by drift.
```

## CI vs CD split

GitOps cleanly divides responsibilities — a key mental model:

```
CI  (push)  build, test, scan, push image, update the image tag in the config repo.
            See cicd.md.
CD  (pull)  the GitOps agent notices the changed tag and deploys it.
⇒ CI never touches the cluster. CD is just "Git changed → reconcile."
```

## Progressive delivery

GitOps integrates with progressive rollout controllers (Argo Rollouts, Flagger) for canary
and blue-green deploys driven by metrics — see [feature flags](feature_flags.md) for the
application-level complement.

```
Canary: shift 5% → 25% → 50% → 100% of traffic, checking SLOs at each step; auto-rollback
        (revert the Git state) if error rate/latency breaches the threshold.
```

## Where this connects

- **[Kubernetes](kubernetes.md)** — the reconciliation substrate GitOps builds on.
- **[Helm](helm.md)** / Kustomize — how the desired state is packaged and parameterized.
- **[CI/CD](cicd.md)** — CI builds & updates tags; GitOps handles CD.
- **[Terraform](terraform.md)** / **[Infrastructure](infrastructure.md)** — declarative IaC,
  the same philosophy for infra.
- **[Secrets management](../security/secrets_management.md)** — secrets in Git must be
  encrypted (Sealed Secrets / External Secrets / SOPS).

## Pitfalls

- **Plaintext secrets in Git** — never; use Sealed Secrets/External Secrets/SOPS so only
  ciphertext is committed.
- **Manual `kubectl` hotfixes** — the agent reverts them; this surprises teams (it's the point).
- **One giant repo with no environment separation** — overlays per environment keep promotion clean.
- **CI with cluster credentials** — defeats GitOps' security benefit; let the agent pull.
- **Image tag `:latest`** — non-deterministic; pin immutable tags/digests so Git fully
  describes what's running.
