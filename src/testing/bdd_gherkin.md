# BDD with Gherkin

## Overview

**Behavior-Driven Development (BDD)** extends [TDD](tdd.md) by writing tests in
language the whole team — including non-developers — can read. Instead of
asserting on units, you describe *behavior* from the user's point of view using
a structured, plain-English format called **Gherkin**.

```gherkin
Feature: Account withdrawals

  Scenario: Withdraw within balance
    Given an account with balance 1000
    When I withdraw 200
    Then the balance should be 800
```

The goal is a **ubiquitous language**: the same words appear in conversations,
the spec, and the executable tests — reducing the gap between what was asked for
and what was built.

## TDD vs BDD

| | TDD | BDD |
|---|-----|-----|
| Focus | unit behavior | business/user behavior |
| Audience | developers | devs + product + QA |
| Wording | `test_add_negative()` | `Scenario: refund a cancelled order` |
| Granularity | fine, many small tests | coarser, feature-level |

They're complementary: BDD for outward behavior/acceptance, TDD for the units
underneath.

## Gherkin Syntax

```gherkin
Feature: Short description of a capability

  Background:                     # runs before each scenario
    Given a logged-in user

  Scenario: A single concrete example
    Given some initial context
    When an action occurs
    And another action
    Then an outcome is expected
    But something else is not

  Scenario Outline: Run the same steps with data
    When I deposit <amount>
    Then the balance should be <result>

    Examples:
      | amount | result |
      | 100    | 1100   |
      | 0      | 1000   |

  @slow @billing                  # tags select/skip subsets
  Scenario: A tagged scenario
    ...
```

Keywords: **Given** (context/arrange), **When** (action/act), **Then**
(outcome/assert), **And**/**But** (continuations), **Background** (shared setup),
**Scenario Outline** + **Examples** (data-driven), **@tags** (filtering).

## Step Definitions

Each Gherkin line maps to code via a **step definition**.

```python
# Python — behave (features/steps/account_steps.py)
from behave import given, when, then

@given('an account with balance {start:d}')
def step_account(context, start):
    context.account = Account(start)

@when('I withdraw {amount:d}')
def step_withdraw(context, amount):
    context.account.withdraw(amount)

@then('the balance should be {expected:d}')
def step_check(context, expected):
    assert context.account.balance == expected
```

```python
# Python — pytest-bdd (integrates with pytest fixtures)
from pytest_bdd import scenario, given, when, then

@scenario('account.feature', 'Withdraw within balance')
def test_withdraw():
    pass

@given('an account with balance 1000', target_fixture='account')
def account():
    return Account(1000)
```

```javascript
// JavaScript — Cucumber.js (features/step_definitions/steps.js)
const { Given, When, Then } = require('@cucumber/cucumber');
const assert = require('assert');

Given('an account with balance {int}', function (start) {
  this.account = new Account(start);
});
When('I withdraw {int}', function (amount) {
  this.account.withdraw(amount);
});
Then('the balance should be {int}', function (expected) {
  assert.strictEqual(this.account.balance, expected);
});
```

## ATDD & the Three Amigos

BDD scenarios are often written *before* development in a **Three Amigos**
conversation (business + dev + tester) so everyone agrees on "done." This is
**Acceptance-Test-Driven Development (ATDD)**: the agreed scenarios become the
acceptance criteria and the automated tests at once — they double as **living
documentation** that can't drift from the code because they execute.

## Pitfalls

❌ **UI-coupled steps** — "When I click the #submit button" makes scenarios
brittle; describe intent ("When I submit the order"), not mechanics.
❌ **Imperative over declarative** — long step-by-step scripts instead of
business-level statements.
❌ **Gherkin for everything** — the overhead isn't worth it for pure unit logic;
keep that in [TDD](tdd.md)-style tests.
❌ **Steps that leak implementation details** the business doesn't care about.

## Best Practices

1. **Write scenarios with stakeholders**, in their language, before coding.
2. **Declarative, not imperative** — say *what*, not *how*.
3. **Reuse step definitions** and keep them thin (delegate to helpers/page objects).
4. **One behavior per scenario**; use Scenario Outline for data variations.
5. **Tag** to organize runs (`@smoke`, `@wip`, `@slow`).

## ELI10

It's like writing the recipe steps everyone agrees on first — "Given flour and
eggs, When you bake, Then you get a cake" — and then the kitchen robot follows
those exact written steps to prove the cake comes out right.

## Further Resources

- [behave docs](https://behave.readthedocs.io/)
- [pytest-bdd](https://pytest-bdd.readthedocs.io/)
- [Cucumber docs](https://cucumber.io/docs/cucumber/)
- [Gherkin reference](https://cucumber.io/docs/gherkin/reference/)
