# Service Mesh

## Overview

A service mesh is a dedicated infrastructure layer that handles **service-to-service
communication** — traffic routing, security (mTLS), retries, and observability — *without
changing application code*. As a system grows from a monolith into many
[microservices](../system_design/microservices.md) on [Kubernetes](kubernetes.md), every
service needs the same networking concerns (encryption, retries, timeouts, metrics); a mesh
factors them out of every codebase into the platform. It's the runtime enforcement point for
[zero-trust](../security/zero_trust.md) internal networking and [API security](../security/api_security.md)
(mTLS, authz) between services.

```
Without a mesh: every service re-implements TLS, retries, timeouts, circuit breaking,
                tracing — in every language, inconsistently.
With a mesh:    a SIDECAR proxy next to each service does it uniformly, config-driven.
```

## Architecture: data plane + control plane

```
        ┌─────────── Control Plane (Istiod / Linkerd) ───────────┐
        │   policy, certificates, service discovery, config       │
        └───────┬───────────────────────────────┬────────────────┘
                │ configures                     │ configures
   ┌────────────▼─────────┐         ┌────────────▼─────────┐
   │  Pod A               │         │  Pod B               │
   │ [app] ⇄ [sidecar] ───┼──mTLS──►├─ [sidecar] ⇄ [app]   │   ← Data Plane
   └──────────────────────┘         └──────────────────────┘
   All traffic flows app → local sidecar → remote sidecar → app.
```

```
Data plane    the sidecar proxies (Envoy, or Linkerd's micro-proxy) that intercept and
              handle every request in/out of each service.
Control plane the brain: distributes config, certs, and policy to all sidecars.
```

The **sidecar** pattern: a proxy container injected into each pod, intercepting all traffic
transparently (via iptables/eBPF) so the app just talks to `localhost`.

## What a mesh gives you

```
SECURITY
  mTLS everywhere     automatic mutual TLS between services → encrypted + authenticated.
  Identity            each service gets a cryptographic identity (SPIFFE) → zero-trust.
  Authz policy        "service A may call service B's /read, not /admin" — declarative.

TRAFFIC MANAGEMENT
  Routing             canary, blue-green, A/B by weight/header (90/10 splits).
  Resilience          retries, timeouts, circuit breaking, outlier ejection — uniform.
  Load balancing      L7-aware (least-request, consistent hashing). See ../system_design/load_balancing.md.

OBSERVABILITY
  Golden signals      latency/traffic/errors for EVERY hop, no app code. See observability.md.
  Distributed tracing automatic span propagation across services.
```

The big win: these are consistent across every service and language, configured centrally,
not coded into each app.

## Sidecar vs sidecar-less

The architecture is evolving away from a proxy-per-pod due to its resource and latency cost:

```
Sidecar (classic)   one Envoy per pod. Maximum flexibility; costs CPU/RAM per pod + a
                    small latency hop. (Istio default, Linkerd.)
Ambient / sidecar-less  Istio Ambient mode and eBPF-based meshes (Cilium) move L4 mTLS
                    into a per-NODE component and use an L7 proxy only when needed.
                    → lower overhead, no per-pod injection. The current direction of travel.
```

## Tooling

```
Istio       most feature-rich; Envoy data plane. Powerful but complex; Ambient mode reduces
            the sidecar tax.
Linkerd     simplicity & low overhead; purpose-built Rust micro-proxy. CNCF graduated.
Cilium Service Mesh   eBPF-based, often sidecar-less; ties into kernel networking.
Consul Connect        HashiCorp's mesh, works beyond Kubernetes (VMs too).
```

## Do you even need one?

A mesh adds real operational complexity — adopt it when the problem justifies it:

```
You probably need a mesh when:
  ✓ many services, polyglot, needing uniform mTLS + retries + tracing
  ✓ zero-trust requirement for internal traffic
  ✓ progressive delivery (traffic-shifted canaries) across many services
You probably DON'T when:
  ✗ a handful of services → a library (or just the language's HTTP client) is simpler
  ✗ small team without platform capacity to operate the mesh
Alternative: API gateway (north-south traffic) + client libraries may suffice.
```

## Gateway vs mesh (north-south vs east-west)

```
API gateway / ingress   NORTH-SOUTH: traffic entering the cluster from outside. AuthN,
                        rate limiting, routing at the edge. See ../security/api_security.md.
Service mesh            EAST-WEST: traffic BETWEEN services inside the cluster.
Modern gateways (Gateway API) and meshes increasingly converge.
```

## Where this connects

- **[Kubernetes](kubernetes.md)** — where meshes run; sidecar injection.
- **[Microservices](../system_design/microservices.md)** — the architecture that motivates a mesh.
- **[Zero trust](../security/zero_trust.md)** / **[API security](../security/api_security.md)**
  — mTLS and per-service authz.
- **[Observability](observability.md)** — golden signals & tracing the mesh emits.
- **[Load balancing](../system_design/load_balancing.md)** — L7 balancing inside the mesh.

## Pitfalls

- **Adopting a mesh too early** — its complexity outweighs the benefit for a few services.
- **Ignoring the sidecar tax** — per-pod CPU/RAM/latency adds up; consider ambient/eBPF modes.
- **mTLS misconfiguration** — permissive vs strict mode mismatches cause silent plaintext or
  broken traffic during rollout.
- **Treating the mesh as a gateway** — it handles east-west; you still need an edge gateway
  for north-south.
- **Under-resourcing the control plane** — an unhealthy control plane degrades the whole mesh.
