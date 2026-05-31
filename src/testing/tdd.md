# Test-Driven Development (TDD)

## Overview

TDD: Write tests BEFORE writing code. Red → Green → Refactor cycle.

## Red-Green-Refactor Cycle

### 1. Red: Write Failing Test
```python
def test_add_positive_numbers():
    assert add(2, 3) == 5
```

### 2. Green: Write Minimal Code
```python
def add(a, b):
    return 5  # Hardcoded to pass test
```

### 3. Refactor: Improve Code
```python
def add(a, b):
    return a + b  # Proper implementation
```

## Red-Green-Refactor Deep Dive

### The Three Laws of TDD

Robert C. Martin's three rules govern the micro-cycle:

1. **You may not write production code until you have written a failing test.**
2. **You may not write more of a test than is sufficient to fail** — and a
   compilation/import error counts as a failure.
3. **You may not write more production code than is sufficient to pass the
   currently failing test.**

These laws force a very short loop — seconds to a couple of minutes per turn.
You are never more than one failing test away from working code.

### What Each Phase Really Does

**🔴 Red — prove the test can fail.**
Run the test and *watch it fail* before writing any code. This is the step
beginners skip, and skipping it is dangerous: a test that never fails might be
asserting nothing, testing the wrong thing, or already passing for the wrong
reason. A red you actually observed is the only proof the test has teeth.

**🟢 Green — get to passing as fast as possible.**
Cleanliness does not matter here. Hardcode, copy-paste, take shortcuts —
whatever turns the bar green quickest. Speed here keeps the cycle tight; the
mess gets cleaned in the next phase.

**🔵 Refactor — clean up under a safety net.**
With tests green, improve the design: remove duplication, rename, extract.
**No behavior changes and no new tests** in this phase — if the tests still
pass, the refactor was safe. New behavior means a new red test first.

### Strategies for Getting to Green

Kent Beck describes three ways to make a failing test pass:

**Fake it** — return a constant, then generalize. The hardcoded `return 5`
above is a fake; it gets you green so you can then drive out the real logic.

**Triangulation** — add a second example to force generalization. One case can
be faked with a constant; two cases that disagree force the real implementation:

```python
def test_add():
    assert add(2, 3) == 5   # return 5 passes this...
    assert add(4, 1) == 5   # ...and this, so triangulate:
    assert add(2, 2) == 4   # now a constant can't pass — forces a + b
```

**Obvious implementation** — when the real code is trivial and you're
confident, just write it directly instead of faking. Use this when the path is
clear; fall back to fake-it/triangulation when you're unsure.

### Cycle Mechanics & Tips

- **Keep cycles short** — one behavior at a time. If a step feels big, take a
  smaller one.
- **Commit on each green** (and again after each refactor). You get a clean,
  revertible history and can always roll back to the last working state.
- **Keep a test list** — jot down cases you want to cover as you think of them,
  but write them one at a time. Don't write all the tests up front.
- **Keep the phases distinct** — never refactor while red. If a refactor breaks
  a test, revert it; don't try to refactor and fix behavior at once.

### Anti-patterns & Pitfalls

❌ **Skipping red** — never seeing the test fail; you don't know it can catch
the bug it's meant to.
❌ **Over-engineering green** — adding abstraction or unrequested features
before a test demands them.
❌ **Testing implementation, not behavior** — asserting on private internals;
the tests then break on every refactor.
❌ **Writing all tests at once** — loses the tight feedback loop and the
test-list discipline.
❌ **Refactoring while red** — mixing cleanup with behavior changes, so a
failure could come from either.
❌ **Asserting nothing** — a test that runs code but checks no outcome is green
theater; it can never fail.

## TDD Schools

Two styles of TDD differ in how they treat collaborators (the other objects the
unit talks to). See [Mocking](mocking.md) for the test-double mechanics behind
each.

### London (mockist) — outside-in

Start at the outer boundary (e.g. a controller) and work inward. **Mock the
collaborators** of the unit under test and verify the *interactions* between
them. Design emerges from the conversations between objects.

```python
def test_checkout_charges_the_card():
    gateway = Mock()                 # collaborator is mocked
    Checkout(gateway).pay(order)
    gateway.charge.assert_called_once_with(order.total)  # assert interaction
```

✅ Drives interfaces top-down; tests are isolated and fast.
❌ Tests couple to *how* objects collaborate → more brittle on refactor; risk of
over-mocking.

### Detroit / Chicago (classicist) — inside-out

Start with the core domain and build outward using **real objects**, mocking
only at true boundaries (network, DB, time). Assert on the resulting *state*.

```python
def test_checkout_records_payment():
    account = Account(balance=1000)  # real collaborator
    Checkout(account).pay(order)
    assert account.balance == 800    # assert state
```

✅ Tests survive internal refactors; fewer mocks.
❌ Failures can span several real objects; design feedback comes later.

**Which to use?** Mock at architectural seams and external dependencies
(classicist by default); reach for outside-in/mockist when designing the
collaboration protocol of a new feature. Most teams blend both.

## TDD on Legacy Code

Classic TDD assumes a clean slate, but most work is on existing code with no
tests. The dilemma: *to change code safely you want tests, but to test it you
often must change it.*

**Characterization tests** break the cycle: pin the code's **current** behavior
(bugs and all) before you touch it, so any change that alters behavior shows up.

```python
def test_characterize_legacy_pricing():
    # Not "correct" — just what it does today, captured as a safety net
    assert legacy_price(qty=3, code="X") == 27.0
```

To get untestable code under test, find **seams** — places to substitute
behavior without rewriting everything (inject a dependency, subclass-and-override,
extract a method/parameter). Then write characterization tests, refactor under
their cover, and only then change behavior with normal red-green-refactor.

This is the core of Michael Feathers' *Working Effectively with Legacy Code*
("legacy code" = code without tests). Watch for [test smells](test_smells.md)
like over-mocking while wrapping crufty code.

## Benefits

✅ Better design (code written to be testable)
✅ Fewer bugs (test before shipping)
✅ Confidence (safe to refactor)
✅ Documentation (tests show usage)
✅ Less debugging (catch issues early)

## Example: Calculator

### Step 1: Red
```python
class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5
```

### Step 2: Green
```python
class Calculator:
    def add(self, a, b):
        return a + b
```

### Step 3: Refactor
```python
class Calculator:
    def add(self, a, b):
        """Add two numbers"""
        if not isinstance(a, (int, float)):
            raise TypeError("a must be number")
        return a + b
```

## TDD Best Practices

1. **Start simple**: Test one behavior
2. **One assertion** per test (usually)
3. **Clear names**: `test_add_positive_numbers`
4. **Arrange-Act-Assert**
```python
def test_withdraw():
    # Arrange
    account = Account(1000)
    # Act
    account.withdraw(200)
    # Assert
    assert account.balance == 800
```

5. **Don't skip red phase**: Ensures test can fail

## Working Test Example

```python
# calculator.py - EMPTY (start)

# test_calculator.py
def test_multiply():
    # Test fails: function doesn't exist (RED)
    assert multiply(3, 4) == 12
```

```python
# calculator.py - implement
def multiply(a, b):
    return a * b
```

```python
# test passes (GREEN)

# Refactor if needed
```

## Coverage with TDD

TDD naturally leads to high coverage:

```python
# Typical TDD: 90%+ coverage
# Non-TDD: 20-40% coverage
```

## TDD vs BDD

**TDD**: Tests focus on unit behavior
```python
test_add_positive_numbers()
test_add_negative_numbers()
```

**BDD**: Tests focus on business behavior
```python
test_user_can_withdraw_money()
test_system_prevents_overdraft()
```

## Tools

- **pytest**: Python testing
- **Jest**: JavaScript testing
- **JUnit**: Java testing
- **RSpec**: Ruby testing

## ELI10

TDD is like building with blueprints:
1. Draw blueprint (write test)
2. Build to match (write code)
3. Improve design (refactor)

Never start building without a plan!

## Further Resources

- [TDD by Example - Kent Beck](https://www.amazon.com/Test-Driven-Development-Kent-Beck/dp/0321146530)
- [What is TDD](https://testdriven.io/blog/what-is-tdd/)
