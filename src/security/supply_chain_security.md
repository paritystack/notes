# Supply Chain Security

## Overview

Modern software is *assembled*, not written: a typical app is a thin layer of your code on
a mountain of third-party dependencies, build tools, base images, and CI/CD plugins. **Supply
chain security** defends every link from source to running artifact — because attackers
have learned it's easier to compromise a dependency 10,000 projects trust than to attack
those projects directly. This complements app-level [threat modeling](threat_modeling.md)
and [OWASP Top 10](owasp_top_10.md) (which focus on *your* code), and lives in the
[CI/CD pipeline](../devops/cicd.md) and [container](../devops/kubernetes.md) layers.

```
The attack surface — every link is a target:
  [source repo] ─► [dependencies] ─► [build/CI] ─► [artifact/image] ─► [registry] ─► [deploy]
       │                │                │              │                  │             │
   account takeover  malicious pkg   poisoned build  tampered binary   typosquat    drift
```

## How these attacks happen

```
Dependency confusion   publish a public package with the same name as a company's
                       INTERNAL package + higher version → the build pulls the attacker's.
Typosquatting          `python-requests` vs `requests`; a typo installs malware.
Malicious maintainer / hijacked account  legit popular package ships a backdoor
                       (e.g. event-stream, ua-parser-js, the xz/liblzma backdoor).
Compromised build      attacker subverts the CI server or a build plugin and injects
                       code AFTER the (clean) source — SolarWinds-style.
Tampered artifact      binary/image swapped between build and deploy.
Transitive risk        you vet your direct deps; the danger is 5 levels deep.
```

## SBOM — know what you ship

A **Software Bill of Materials** is a complete, machine-readable inventory of every
component and version in your software. You can't defend (or patch a Log4Shell in) what you
don't know you have.

```
Formats:  SPDX, CycloneDX
Generate: syft, cdxgen, language tooling (npm/pip/cargo) → emit per build artifact.
Use it:   when CVE-2024-XXXX drops, query SBOMs to find every affected build in minutes,
          not days.
```

## Dependency hygiene

```
Pin & lock      commit lockfiles (package-lock.json, poetry.lock, Cargo.lock,
                go.sum) → reproducible, tamper-evident dependency sets.
Verify hashes   pip --require-hashes, go.sum, subresource integrity — detect swapped pkgs.
Scan (SCA)      Dependabot, Renovate, Snyk, OSV-Scanner, Trivy → flag known-vuln deps,
                open auto-PRs to bump them.
Minimize        fewer/smaller deps = smaller attack surface (distroless base images).
Vendor/mirror   proxy registries (Artifactory, internal PyPI) → control & cache,
                blocks dependency-confusion by namespace ownership.
Review updates  don't auto-merge major bumps of unvetted packages blindly.
```

## SLSA — securing the build itself

**SLSA** (Supply-chain Levels for Software Artifacts, "salsa") is a framework of
progressively stronger guarantees that an artifact is what it claims to be and was built as
claimed.

```
SLSA levels (intuition):
  L1  build is scripted & generates provenance         (basic, automatable)
  L2  hosted build service + signed provenance
  L3  hardened, isolated, non-falsifiable build         (tamper-resistant)
Key ideas: builds are reproducible, isolated, and emit signed PROVENANCE
           (who built what, from which source, with which inputs).
```

## Signing & provenance — prove authenticity

```
Sign artifacts/images so consumers can verify origin & integrity:
  Sigstore / cosign   keyless signing using OIDC identity + transparency log (Rekor).
  in-toto             attest each step of the supply chain.
  SLSA provenance     signed statement linking artifact → source commit → builder.

Verify at deploy:
  admission controller (Kubernetes) rejects unsigned/untrusted images.
  → "only run images signed by our CI, built from our repo."
```

This closes the loop with [secrets management](secrets_management.md): the build's signing
identity is a workload identity, not a long-lived stored key.

## Securing the pipeline

The build system is privileged code with access to source and secrets — treat it as
production:

```
- Least-privilege CI tokens, short-lived (OIDC), scoped per repo/job.
- Pin CI actions/plugins to a commit SHA, not a mutable tag (@v3 → @<sha>).
- Isolated, ephemeral build runners; no shared mutable state between builds.
- Protected branches, mandatory review, signed commits for the source itself.
- Scan IaC and Dockerfiles (checkov, trivy) before they ship.
```

## Where this connects

- **[CI/CD](../devops/cicd.md)** / **[GitOps](../devops/gitops.md)** — where provenance,
  signing, and scanning are enforced.
- **[Container security](../devops/container_security.md)** — base-image and registry
  trust, admission control.
- **[Security testing](../testing/security_testing.md)** — SCA/SAST/DAST in the pipeline.
- **[Secrets management](secrets_management.md)** — leaked CI credentials are a top vector.

## Pitfalls

- **Trusting transitive deps blindly** — the risk is usually deep in the tree.
- **Mutable tags** — `actions/checkout@v4` or `:latest` can change under you; pin SHAs/digests.
- **No SBOM** — leaves you unable to answer "are we affected?" when the next Log4Shell hits.
- **Unsigned artifacts** — nothing stops a swapped binary between build and deploy.
- **Over-privileged CI** — a compromised pipeline with prod credentials is game over.
