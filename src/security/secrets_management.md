# Secrets Management

## Overview

A **secret** is any credential an application needs at runtime but must never expose:
database passwords, API keys, TLS private keys, signing keys, OAuth client secrets. Secrets
management is the discipline of getting those secrets to the right workload, at the right
time, without ever committing them to source control, baking them into images, or
splattering them across logs. It is the operational sibling of [key management](key_management.md)
(which focuses on cryptographic keys specifically) and a core enabler of
[zero trust](zero_trust.md), [DevOps](../devops/index.html) pipelines, and [cloud](../cloud/index.html)
deployments.

```
The cardinal rule:
  secrets NEVER live in   → git, Dockerfiles, config maps, env files in repos, logs,
                            error messages, client-side code, CI logs
  secrets DO live in      → a dedicated secrets store, injected at runtime
```

## Why not just environment variables / config files

The "12-factor" `.env` habit is better than hardcoding but still weak at scale:

```
Plaintext env / .env files:
  ✗ leak into logs, crash dumps, `ps`/`/proc`, child processes
  ✗ no rotation, no audit, no expiry, no per-request access control
  ✗ committed by accident (the #1 source of leaked AWS keys on GitHub)
A dedicated secrets manager adds: encryption at rest, access policy, audit log,
  rotation, dynamic/short-lived secrets, and revocation.
```

## The tooling landscape

```
HashiCorp Vault       general secrets engine: KV, dynamic secrets, PKI, transit
                      encryption, leasing & revocation. Self-hosted standard.
Cloud-native          AWS Secrets Manager / SSM Parameter Store,
                      GCP Secret Manager, Azure Key Vault — managed, IAM-integrated.
Kubernetes Secrets    base64, NOT encrypted by default → enable encryption-at-rest
                      + RBAC; usually fronted by an external store (below).
Sealed Secrets /      encrypt secrets so the CIPHERTEXT is safe to commit to git
External Secrets      (GitOps-friendly); the controller decrypts in-cluster.
SOPS                  encrypts values in YAML/JSON files with KMS/age/PGP.
```

## Static vs dynamic secrets

The biggest leap a real secrets manager offers is **dynamic** secrets — credentials
generated on demand and automatically expired.

```
Static secret    long-lived, shared, manually rotated (a DB password in a vault).
                 Better than hardcoding, but a leak exposes a valid credential until
                 someone notices and rotates.

Dynamic secret   Vault generates a UNIQUE, SHORT-LIVED credential per app/request and
                 revokes it on lease expiry.
   app ──"give me DB creds"──► Vault ──creates ephemeral DB user (TTL 1h)──► DB
   leak window shrinks from "forever" to "minutes"; every lease is audited & revocable.
```

## How secrets reach a workload

```
1. Sidecar / agent injector   Vault Agent / CSI driver mounts secrets into the pod
                              at runtime (tmpfs, never on disk image).
2. SDK pull at startup        app calls the secrets API on boot using a workload identity.
3. Init container             fetches secrets, writes to a shared in-memory volume.
4. Templated env (last resort) injected env vars — convenient but most leak-prone.

Authentication of the workload itself uses a WORKLOAD IDENTITY, not another secret:
  Kubernetes ServiceAccount token, AWS IAM role (IRSA), GCP Workload Identity,
  SPIFFE/SPIRE. → solves the "secret-zero" bootstrap problem.
```

## The secret-zero problem

To fetch secrets you need *a* credential — so what protects *that*? The answer is
**platform-issued identity** rather than a stored secret:

```
Bad:   app holds a Vault token in an env var (just moved the problem)
Good:  the platform (k8s/cloud) cryptographically attests "this is workload X";
       the secrets store trusts that attestation and issues a short-lived token.
       Nothing long-lived is ever stored by the app.
```

## Rotation and revocation

```
Rotation    change secrets on a schedule and on suspected compromise. Managed stores
            can rotate DB/cloud credentials automatically and update consumers.
Revocation  invalidate a leased/compromised secret immediately (dynamic secrets make
            this trivial — revoke the lease).
Versioning  keep old versions briefly so in-flight workloads don't break mid-rotation.
```

## Detecting leaks (defense in depth)

Assume something will slip; catch it:

```
Pre-commit / CI scanning   gitleaks, trufflehog, detect-secrets — block secrets
                           before they land in history. Wire into the pipeline.
Provider-side detection    GitHub secret scanning auto-revokes some leaked tokens.
Audit logs                 every secret access logged → detect anomalous reads.
If a secret leaks → ROTATE it; deleting the commit is NOT enough (git history,
  forks, and scrapers already have it).
```

## Where this connects

- **[Key management](key_management.md)** — the cryptographic-key subset, HSM/KMS,
  envelope encryption.
- **[DevOps / CI-CD](../devops/cicd.md)** — pipelines must fetch secrets securely, never
  print them; see GitOps and [Kubernetes](../devops/kubernetes.md).
- **[Cloud](../cloud/index.html)** — IAM roles and workload identity as secret-zero.
- **[Supply chain security](supply_chain_security.md)** — leaked CI tokens are a top
  supply-chain attack vector.
- **[Zero trust](zero_trust.md)** — short-lived, per-workload credentials over static
  shared ones.

## Pitfalls

- **Committing secrets** — even once; rotate immediately, don't just delete the commit.
- **base64 ≠ encryption** — raw Kubernetes Secrets are encoded, not encrypted.
- **Logging secrets** — redact at the logging layer; secrets in stack traces and request
  dumps are a classic leak.
- **Long-lived shared credentials** — prefer dynamic, short-lived, per-workload secrets.
- **Secret-zero stored as a secret** — bootstrap with platform identity, not another key.
