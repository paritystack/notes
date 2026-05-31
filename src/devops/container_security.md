# Container Security

## Overview

Containers package an app with its dependencies, which is also their security challenge: a
container image bundles an entire userland (base OS, libraries, your code) that can carry
vulnerabilities, secrets, and excessive privileges into production. Container security spans
the whole lifecycle — **build** (the image), **ship** (the registry), and **run** (the
[Kubernetes](kubernetes.md) workload). It's the operational front line of
[supply chain security](../security/supply_chain_security.md) and a key control in any
[threat model](../security/threat_modeling.md) of a containerized system.

```
The lifecycle — secure each phase:
  BUILD  → minimal, scanned, non-root image, no secrets baked in
  SHIP   → signed images in a trusted registry; verify on pull
  RUN    → least privilege, read-only FS, network policy, runtime detection
```

## Build-time: the image

```
Minimal base images   distroless / Alpine / scratch → fewer packages = smaller attack
                      surface and fewer CVEs. A full Ubuntu base ships hundreds of
                      packages you don't use.
Multi-stage builds    compile in a fat builder stage, copy ONLY the artifact into a tiny
                      runtime image → no compilers/shells/tools in production.
Don't run as root     USER appuser in the Dockerfile; a container root is often host-adjacent.
Pin versions          pin base image DIGESTS (not :latest) and dependency versions →
                      reproducible & tamper-evident. See ../security/supply_chain_security.md.
No secrets in layers  ARG/COPY secrets persist in image history even if later deleted →
                      use build secrets / runtime injection. See ../security/secrets_management.md.
.dockerignore         keep .git, .env, keys out of the build context.
```

## Image scanning

Scan images for known-vulnerable packages before they ship and continuously after:

```
Tools: Trivy, Grype, Clair, Snyk, Docker Scout.
Scan:  in CI (fail the build on critical CVEs) AND in the registry (new CVEs appear in
       already-published images → re-scan continuously).
Also scan: misconfigurations, exposed secrets, and generate an SBOM (see supply chain).
```

## Ship-time: registry & trust

```
Private registry      control access; don't pull arbitrary public images into prod.
Image signing         cosign/Sigstore sign images; verify signatures on deploy.
Admission control     Kubernetes admission controllers (OPA Gatekeeper, Kyverno) REJECT
                      images that are unsigned, from untrusted registries, run as root,
                      or fail policy. → "only signed images from our registry may run."
```

## Run-time: hardening the workload

The container runtime shares the host kernel, so isolation is weaker than a VM — reduce what
a compromised container can do:

```
Least privilege
  runAsNonRoot, drop ALL Linux capabilities (add back only what's needed),
  no privileged: true, no host namespaces (hostPID/hostNetwork), no docker.sock mount.
Read-only root filesystem
  readOnlyRootFilesystem: true → attacker can't write tools/persist. Mount tmpfs for /tmp.
seccomp / AppArmor / SELinux
  restrict the syscalls a container may make (RuntimeDefault seccomp profile at minimum).
  See ../linux/selinux.md.
No privilege escalation
  allowPrivilegeEscalation: false.
Resource limits
  CPU/memory limits prevent a compromised/buggy container from starving the node (a DoS).
```

## Kubernetes-specific controls

```
Pod Security Admission   enforce baseline/restricted pod security standards per namespace.
Network policies         default-deny, then allow only required service-to-service traffic
                         (micro-segmentation) — see ../networking/overlay_networks.md and
                         service_mesh.md for mTLS.
RBAC                     least-privilege service accounts; don't grant cluster-admin to apps.
Secrets                  not in env/images → external secret stores. See secrets_management.
Runtime detection        Falco / eBPF tools alert on anomalous syscalls (a shell spawned in
                         a container, unexpected network connections, writes to /etc).
```

## Defense in depth

```
Build:   minimal base, non-root, scan, no secrets, SBOM
Ship:    signed images, trusted registry, admission policy
Run:     drop caps, read-only FS, seccomp, network policy, resource limits
Detect:  runtime monitoring (Falco), continuous re-scanning, audit logs
No single layer is enough — assume one fails and the next contains the blast.
```

## Where this connects

- **[Kubernetes](kubernetes.md)** — Pod Security, RBAC, network policy, admission control.
- **[Supply chain security](../security/supply_chain_security.md)** — image provenance,
  signing, SBOM, scanning.
- **[Secrets management](../security/secrets_management.md)** — keeping secrets out of images.
- **[Service mesh](service_mesh.md)** — mTLS and authz between containers.
- **[SELinux](../linux/selinux.md)** — kernel-level mandatory access control.

## Pitfalls

- **Running as root** — the most common and most dangerous default; set a non-root USER.
- **Secrets baked into image layers** — persist in history; inject at runtime instead.
- **`:latest` / unpinned bases** — non-reproducible and silently pull in new CVEs.
- **`privileged: true` / mounting docker.sock** — effectively host root; almost never needed.
- **No network policy** — flat cluster networking lets a compromised pod reach everything.
- **Scan once, never again** — new CVEs land in shipped images; re-scan continuously.
