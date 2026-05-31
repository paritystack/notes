# Contract Testing

## Overview

When services talk to each other (microservices, a frontend and its API), the
risk is a **broken integration**: the provider changes a field and every
consumer breaks in production. Full end-to-end tests across all services catch
this but are slow, flaky, and need everything deployed together.

**Contract testing** verifies the *boundary* between two services in isolation.
Each side is tested against a shared **contract** describing the requests a
consumer makes and the responses it expects — without running both services at
once.

```
Consumer  ──(contract: "GET /users/1 → {id, name}")──►  Provider
  test against a mock        verify it can fulfill the contract
```

## Consumer-Driven Contracts

The **consumer** defines what it actually needs (not everything the provider
offers). The provider then proves it can satisfy every consumer's expectations.
This keeps providers from breaking real usage and stops them over-committing to
fields nobody uses.

## Pact Flow

[Pact](https://pact.io/) is the most common tool. The flow:

1. **Consumer test** — run the consumer against a Pact mock server; the
   interactions are recorded into a **pact file** (JSON).

```javascript
// Consumer side (Pact JS)
provider
  .given('user 1 exists')
  .uponReceiving('a request for user 1')
  .withRequest({ method: 'GET', path: '/users/1' })
  .willRespondWith({
    status: 200,
    body: { id: 1, name: like('Alice') },  // matchers, not exact values
  });
```

2. **Publish** the pact file to a **Pact Broker** (a central registry).

3. **Provider verification** — the provider replays the pact against its real
   implementation to confirm it responds as promised.

```python
# Provider side (pact-python), pseudo-CLI
pact-verifier --provider-base-url=http://localhost:8000 \
              --pact-broker-url=https://broker.example.com
```

4. **`can-i-deploy`** — before deploying, ask the broker whether this version is
   compatible with the versions of its partners already in the target
   environment.

```bash
pact-broker can-i-deploy --pacticipant Consumer --version $SHA --to-environment prod
```

## Matchers

Contracts assert *shape*, not exact data — use matchers (`like`, `eachLike`,
`term`/regex) so the contract doesn't break on a different-but-valid value. This
is the key difference from snapshot/golden testing.

## Bi-Directional & Schema Approaches

- **Bi-directional contracts**: provider supplies an OpenAPI spec, consumer
  supplies its expectations; the broker checks compatibility without running
  provider verification.
- **Schema validation** (OpenAPI/JSON Schema, Protobuf) catches structural
  breaks but not consumer-specific expectations — weaker than CDC but cheaper.

## Contract vs Integration vs E2E

| Test type | Scope | Speed | Needs both services running |
|-----------|-------|-------|------------------------------|
| Unit | one function | fastest | no |
| Contract | one boundary | fast | no (uses mock + replay) |
| [Integration](integration.md) | a few real components | medium | sometimes |
| E2E | whole system | slow | yes |

Contract testing gives integration-level confidence at near-unit speed for the
specific concern of "do these two services agree?"

## Best Practices

1. **Keep contracts consumer-driven and minimal** — only what the consumer uses.
2. **Use matchers**, not literal values, so contracts survive valid data changes.
3. **Run provider verification in CI** and gate deploys with `can-i-deploy`.
4. **One contract per consumer-provider pair**; version them in the broker.
5. **Don't replace all integration tests** — contracts verify the interface
   agreement, not end-to-end business flows.

## ELI10

Two people agree to meet: one says "I'll ask for a red ball," the other says
"I'll hand over a red ball." Instead of forcing them to meet every time to
check, each practices their half against a written note. As long as both keep
their promise on paper, the real meeting will work.

## Further Resources

- [Pact docs](https://docs.pact.io/)
- [Pact Broker](https://docs.pact.io/pact_broker)
- [Martin Fowler: Contract Tests](https://martinfowler.com/bliki/ContractTest.html)
