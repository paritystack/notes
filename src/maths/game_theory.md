# Game Theory

## Overview

Game theory is the maths of **strategic interaction** — how rational agents should act
when their payoff depends on what *other* agents do. It turns "what should I do?" into
"what should I do given that everyone else is also optimizing?" The central solution
concept is the **Nash equilibrium**: a set of strategies where no one can do better by
changing theirs alone. It builds on [probability](probability.md) (mixed strategies are
distributions over actions) and [optimization](optimization.md) (each player maximizes
expected payoff), and shows up in adversarial ML
([GANs](../machine_learning/deep_learning.md) and multi-agent
[reinforcement learning](../machine_learning/reinforcement_learning.md)), the minimax
search behind [game-playing algorithms](../algorithms/dynamic_programming.md), auctions,
and [markets](../finance/README.md).

```
A game needs three things:
  Players      who is deciding             (you, an opponent, a market)
  Strategies   the actions each can take   (cooperate / defect)
  Payoffs      the reward for each outcome  (a number per player)
```

## Normal form and payoff matrices

A **normal-form** (simultaneous) game lists payoffs in a matrix; entries are
`(row payoff, column payoff)`. The **Prisoner's Dilemma**:

```
                    Player B
                Cooperate   Defect
            ┌───────────┬───────────┐
 Cooperate  │  (3, 3)   │  (0, 5)   │
Player A     ├───────────┼───────────┤
 Defect     │  (5, 0)   │  (1, 1)   │
            └───────────┴───────────┘
```

Whatever B does, A scores more by defecting (5>3 if B cooperates, 1>0 if B defects), so
**Defect dominates** for both. They land on `(1,1)` — worse for both than the `(3,3)`
they *could* have reached. That gap between individual rationality and collective good is
the whole point of the field.

## Dominance and Nash equilibrium

```
Dominant strategy   best regardless of others' choices  (Defect above)
Best response       best given a fixed choice by others
Nash equilibrium    every player is playing a best response to the others
                    → nobody can improve by unilaterally deviating
```

In the Prisoner's Dilemma `(Defect, Defect)` is the unique Nash equilibrium. A game can
have zero, one, or many pure-strategy equilibria.

## Mixed strategies

Some games have no equilibrium in pure (deterministic) strategies — **Matching Pennies**,
Rock-Paper-Scissors. The resolution is to **randomize**: a mixed strategy is a
probability distribution over actions, chosen so the opponent is left indifferent.

```
Rock-Paper-Scissors equilibrium:  play each with probability 1/3.
Any predictable bias is exploitable, so the only stable play is uniform.
```

**Nash's theorem**: every finite game has at least one equilibrium in mixed strategies —
the existence result that made the field general.

## Zero-sum games and minimax

In a **zero-sum** game one player's gain is exactly the other's loss (chess, poker
heads-up). The optimal play is **minimax** — maximize your worst-case payoff:

```
  maximize the minimum you can guarantee
  v = max_a  min_b  payoff(a, b)
```

For zero-sum games the minimax value equals the Nash value (von Neumann's minimax
theorem). This is precisely the [minimax search with alpha-beta
pruning](../algorithms/dynamic_programming.md) used in board-game engines.

## Efficiency: Pareto and the price of anarchy

```
Pareto optimal   no one can be made better off without making someone worse off
Price of anarchy  ratio of the best possible outcome to the equilibrium outcome
                  — how much selfish play costs vs. central coordination
```

`(Defect, Defect)` is a Nash equilibrium but **not** Pareto optimal — `(3,3)` Pareto-
dominates it. Equilibria are stable, not necessarily good.

## Repeated games and mechanism design

- **Repeated games** — playing repeatedly changes incentives. Strategies like
  *tit-for-tat* sustain cooperation because defection invites future retaliation; the
  *folk theorem* says many cooperative outcomes become equilibria under repetition.
- **Mechanism design** ("reverse game theory") — design the *rules* so that
  self-interested play yields a desired outcome. A **second-price (Vickrey) auction**
  makes bidding your true value the dominant strategy — the basis of ad auctions.

## Where this shows up

- **ML** — [GANs](../machine_learning/deep_learning.md) are a two-player minimax game
  between generator and discriminator; multi-agent
  [reinforcement learning](../machine_learning/reinforcement_learning.md) seeks
  equilibria; self-play (AlphaZero) is repeated zero-sum.
- **Algorithms** — [minimax / alpha-beta](../algorithms/dynamic_programming.md) game-tree
  search.
- **Finance & markets** — bidding, [market](../finance/README.md) microstructure, and
  strategic trading are equilibrium problems.
- **Probability** — mixed strategies and expected payoffs are [probability](probability.md)
  in action.
- **Systems** — congestion control, resource allocation, and blockchain incentive design
  are mechanism-design problems.

## Pitfalls

- **Assuming a unique or pure equilibrium** — many games have several, or only mixed ones.
- **Equilibrium ≠ optimal** — Nash is about stability, not welfare (Pareto gap).
- **The rationality assumption** — real agents are boundedly rational; predictions break
  when opponents are irrational or make errors.
- **Ignoring repetition** — one-shot analysis misses cooperation that repeated
  interaction sustains.
