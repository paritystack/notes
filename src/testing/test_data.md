# Test Data Management

## Overview

Tests need data: a user, an order, a populated database row. Hand-building that
data inline makes tests verbose, fragile, and full of irrelevant detail (the
"mystery guest" smell — see [Test Smells](test_smells.md)). Good test-data
strategy keeps tests **readable, isolated, and deterministic**.

```python
# Noisy: every field spelled out, only `age` matters to this test
user = User(id=1, name="x", email="x@y.z", age=17, country="US",
            created_at=datetime(2020, 1, 1), is_active=True, ...)

# Intent-revealing: a factory fills the rest with sensible defaults
user = UserFactory(age=17)
```

## Factories

A **factory** produces ready-to-use objects with sane defaults; you override
only the fields relevant to the test.

```python
# Python — factory_boy
import factory
from myapp.models import User

class UserFactory(factory.Factory):
    class Meta:
        model = User
    name = factory.Faker("name")
    email = factory.Faker("email")
    age = 30
    is_active = True

UserFactory()                 # all defaults
UserFactory(age=17)           # override what matters
UserFactory.build_batch(5)    # a list of five
```

```javascript
// JavaScript — Fishery + Faker
import { Factory } from 'fishery';
import { faker } from '@faker-js/faker';

const userFactory = Factory.define(() => ({
  name: faker.person.fullName(),
  email: faker.internet.email(),
  age: 30,
}));

userFactory.build({ age: 17 });
```

**Faker** generates realistic fake values (names, emails, addresses) so you
don't invent them by hand.

## The Builder Pattern

For complex objects, a fluent builder reads well and centralizes construction:

```python
order = (OrderBuilder()
         .with_customer(UserFactory())
         .with_item("widget", qty=3)
         .paid()
         .build())
```

Builders and factories solve the same problem — defaults + selective override;
builders shine when construction has steps or conditional state.

## Fixtures vs Factories

- **Fixtures** (e.g. pytest fixtures, static JSON/SQL files): fixed, shared
  setup. Great for genuinely shared context; brittle when many tests depend on
  the same large blob and one test needs it slightly different.
- **Factories**: generated per test, customized inline. Scale better as the
  suite grows.

Common pattern: a pytest fixture that *returns a factory* or a freshly built
object, combining the lifecycle management of fixtures with the flexibility of
factories. See [pytest](pytest.md) for fixture mechanics and [Mocking](mocking.md)
for faking collaborators rather than data.

## Determinism

Random data finds edge cases but can cause flaky tests if a failure can't be
reproduced.

```python
import factory
factory.Faker._get_faker().seed_instance(12345)   # fixed seed
```

- **Seed** the generator (or capture the seed on failure) so runs are
  reproducible.
- Avoid real `datetime.now()`/`random()` in assertions — freeze time
  (`freezegun`, `jest.useFakeTimers`) and inject clocks.

## Database Seeding & Isolation

Each test must start from a known state and not leak into the next:

- **Transaction rollback**: wrap each test in a transaction and roll back after
  (fast; the default in many frameworks).
- **Truncate/recreate** between tests when transactions aren't viable (e.g.
  cross-connection or committed data).
- **Per-test schema/containers**: spin up a disposable DB (e.g. Testcontainers)
  for [integration tests](integration.md).
- Keep seed data **minimal** — only what the test needs.

## Production Data

Copying prod data into tests is risky: PII exposure and compliance violations.
If you must, **anonymize/mask** it (and prefer synthetic data via factories,
which is safer and more controllable).

## Best Practices

1. **Factories over inline construction** — show only the fields that matter.
2. **Sensible defaults**; override per test.
3. **Deterministic by default** — seed randomness, freeze time.
4. **Isolate** — each test sets up and tears down its own data.
5. **Never use real PII**; anonymize or synthesize.
6. **Keep data minimal** so tests stay fast and intent stays clear.

## ELI10

Instead of drawing a whole person from scratch every time you need one for a
play, you keep a costume rack: grab a ready-made character and just swap the one
detail you care about — "this one's a kid" — and the rest is filled in for you.

## Further Resources

- [factory_boy docs](https://factoryboy.readthedocs.io/)
- [Faker (Python)](https://faker.readthedocs.io/) · [@faker-js/faker](https://fakerjs.dev/)
- [Fishery](https://github.com/thoughtbot/fishery)

## Where this connects

- [Mocking](mocking.md) — stubs versus real data
- [Integration testing](integration.md) — seeding databases for integration tests
- [Property-based testing](property_based_testing.md) — generated versus hand-built data
- [E2E testing](e2e_testing.md) — fixtures for full-stack runs
- [CI/CD test automation](ci_testing.md) — provisioning data in pipelines
