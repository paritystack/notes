# API Security

## Overview

APIs are the front door to most modern systems — and the most attacked surface, because
they expose business logic and data directly, often to untrusted clients. API security is
about controlling *who* can call *what*, validating everything that comes in, and failing
safely. It builds on [authentication/authorization](auth.md), [OAuth 2.0](oauth2.md), and
[JWT](jwt.md) for identity; on [TLS](ssl_tls.md) for transport; and complements the
browser-centric [web security mechanisms](../web_development/web_security.md). The
canonical reference is the **OWASP API Security Top 10**.

```
AuthN  "who are you?"      → OAuth2 / JWT / mTLS / API keys
AuthZ  "what may you do?"  → per-request checks on EVERY object and action
        ↑ broken authorization is the #1 API vulnerability class.
```

## OWASP API Security Top 10 (2023) — the core risks

```
API1  Broken Object Level Authorization (BOLA/IDOR)  — most common & most damaging
        GET /orders/123  returns someone else's order because the server never checks
        that order 123 belongs to the caller. → authorize the OBJECT, not just the route.
API2  Broken Authentication           weak/absent token validation, no expiry, JWT flaws
API3  Broken Object Property Level     mass assignment / excessive data exposure
        client sends {"role":"admin"} and the server blindly binds it; or the response
        leaks fields the client filters in the UI but the API returned anyway.
API4  Unrestricted Resource Consumption  no rate/size limits → DoS, cost blowups
API5  Broken Function Level Authz      regular user reaches admin endpoints
API6  Unrestricted Access to Business Flows  bots abuse legit flows (scalping, scraping)
API7  Server-Side Request Forgery (SSRF)  API fetches an attacker-supplied URL → internal
API8  Security Misconfiguration        verbose errors, CORS *, missing headers, debug on
API9  Improper Inventory Management    forgotten /v1, staging, undocumented "shadow" APIs
API10 Unsafe Consumption of 3rd-party APIs  trusting upstream responses without validation
```

## Authentication for APIs

```
API keys        simple identifier; OK for service-to-service + rate limiting, NOT for
                user auth (no expiry, often leaked). Treat as a secret.
OAuth 2.0       delegated authz; access tokens (short-lived) + refresh tokens. The
                standard for user-facing & third-party access. See oauth2.md.
JWT             self-contained bearer token. VALIDATE signature, exp, iss, aud; reject
                alg=none; keep them short-lived (can't be revoked easily). See jwt.md.
mTLS            both sides present certs — strong service-to-service auth (service mesh,
                zero-trust internal APIs). See ssl_tls.md and ../devops/service_mesh.md.
```

## Authorization — check every request

The hard part, and where most breaches happen. Identity ≠ permission.

```
- Enforce authz at the OBJECT level on every call: "does THIS caller own/may-access
  THIS resource?" — never trust an ID from the client as proof of ownership.
- Function-level: gate admin/privileged endpoints by role/scope, server-side.
- Models: RBAC (roles), ABAC (attributes), ReBAC (relationships, e.g. Google Zanzibar).
- Default deny: unknown route/action → 403, not 200.
- Never rely on the client (hidden fields, disabled buttons) for authorization.
```

## Input validation & output shaping

```
Validate    schema-validate every input (type, length, range, format) at the boundary —
            allowlist, not denylist. Reject unexpected fields.
Mass assignment  bind only explicitly-allowed fields to your models (DTOs/serializers),
            never the raw request body → blocks API3 privilege escalation.
Excessive exposure  return only the fields the client needs; filter on the SERVER,
            not the UI. No leaking internal IDs, hashes, PII.
Injection   parameterize queries (SQLi), avoid shelling out, sanitize for the sink.
            See ../web_development/web_security.md.
SSRF        validate/allowlist outbound URLs; block link-local/metadata IPs (169.254.169.254).
```

## Rate limiting & abuse protection

```
Rate limit per identity (user/key/IP) — token bucket / sliding window; return 429 +
   Retry-After. Protects against DoS, brute force, scraping, and cost blowups.
Quotas & payload size limits — cap request body, page size, query depth (GraphQL!).
Throttling tiers — different limits per plan/endpoint sensitivity.
Bot defenses — CAPTCHA / device signals on sensitive flows (signup, login, checkout).
```

See [rate limiting](../system_design/rate_limiting.md) for the algorithms (token bucket,
leaky bucket, sliding window) and where to enforce them (gateway vs service).

## Transport & gateway controls

```
- TLS everywhere (1.2+; prefer 1.3); HSTS; no plaintext fallbacks. See ssl_tls.md.
- API gateway centralizes authn, rate limiting, logging, schema validation, and
  request/response transformation — one enforcement point.
- CORS: set explicit allowed origins; NEVER reflect arbitrary Origin or use "*" with
  credentials. See ../web_development/web_security.md.
- Security headers, and DON'T leak stack traces / internal errors to clients.
```

## GraphQL & gRPC specifics

```
GraphQL   a single endpoint hides per-field authz needs; attackers abuse deeply nested
          queries (query-depth/complexity limits), introspection in prod, and batching
          for brute force. Disable introspection externally; limit depth & cost.
gRPC      use mTLS + per-method authz interceptors; validate proto messages; set message
          size limits. See ../web_development/grpc.md.
```

## Inventory & lifecycle

```
- Maintain an API inventory: every version, environment, and endpoint (API9).
- Deprecate & remove old versions; "shadow"/staging APIs are unmonitored back doors.
- Document with OpenAPI; generate validation from the spec (spec = contract = test).
```

## Where this connects

- **[OAuth 2.0](oauth2.md)** / **[JWT](jwt.md)** / **[Auth](auth.md)** — identity & tokens.
- **[Rate limiting](../system_design/rate_limiting.md)** — abuse-protection algorithms.
- **[Web security](../web_development/web_security.md)** — CORS, headers, injection, CSRF.
- **[Threat modeling](threat_modeling.md)** — find these risks at design time.
- **[Service mesh](../devops/service_mesh.md)** — mTLS & authz for internal APIs.

## Pitfalls

- **Authentication without authorization** — a valid token is not permission; check the object.
- **Trusting client-supplied IDs/roles** — the root of BOLA and mass-assignment bugs.
- **`alg:none` / unverified JWTs** — always verify signature, `exp`, `iss`, `aud`.
- **CORS `*` with credentials** — opens your authenticated API to any origin.
- **No rate limits** — invites brute force, scraping, and runaway cost.
- **Forgotten endpoints** — old/undocumented APIs bypass your newer controls.
